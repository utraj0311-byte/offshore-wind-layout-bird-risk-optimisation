"""Model 10: local Band blade-strike contribution at passive Model 9 crossings.

This separate module does not change Model 8 trajectories or Model 9's
``E_RT_CE``. ``BandCrossingAccumulator`` weights the local conditional strike
probability by the *same* discrete flight-height masses and rotor-disc
crossings that Model 9 reports, then takes the maximum for each bird-turbine
pair (the existing inclusive-endpoint deduplication rule). Its output is an
expected potential strike contribution per entering bird, not mortality.

Source geometry: IEA 15 MW v1.1.18 AeroDyn/ElastoDyn in the supplied source
archive. OpenFAST ``BlSpn`` runs from the root along the preconed pitch axis;
the selected projected station coordinate is
``(HubRad + BlSpn) * cos(PreCone)``. This simple mapping deliberately omits
blade curvature/sweep. It never extrapolates chord or twist beyond the first
and last supplied station; zero there means zero *modelled* contribution only.

Band (2012), equation (3):
 p(r,phi) = b*omega/(2*pi*v) * (
     abs(sign*c*sin(gamma) + alpha*c*cos(gamma))
     + max(L, W*alpha*F(phi)))
 alpha = v/(r*omega); gamma = OpenFAST twist + collective pitch.

The gliding projection is abs(cos(phi)): Band says cos(phi), while also
specifying maxima both above and below the hub. Absolute value implements
the nonnegative projected wingspan at both positions. No extra avoidance,
bird-density scaling, or stopped-rotor model is supplied here.

For an oblique Model 9 crossing, the provisional fixed-rotor adaptation uses
``v_normal = bird_speed_m_s * abs(cos(local_heading - fixed_plane_normal))``
throughout the local Band equation. This is a modelling approximation, not a
validated three-dimensional oblique-strike formula. The rotor plane, crossing
point, and Model 9 exposure remain fixed and unchanged.
"""

from __future__ import annotations

import hashlib
import math
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from model9_rotor_transit import RotorCrossingBatch


_BLADE = "IEA-15-240-RWT_AeroDyn15_blade.dat"
_ELASTODYN = "IEA-15-240-RWT-Monopile_ElastoDyn.dat"


def _openfast_value(source: str, label: str) -> float:
    for line in source.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == label:
            return float(parts[0])
    raise ValueError(f"OpenFAST parameter {label!r} was not found")


@dataclass(frozen=True)
class ProjectedBladeStations:
    """Supplied station geometry at physical projected rotor-plane radii."""

    radius_m: np.ndarray
    chord_m: np.ndarray
    twist_deg: np.ndarray
    hub_radius_m: float
    tip_radius_m: float
    precone_deg: float
    number_of_blades: int
    blade_sha256: str
    elastodyn_sha256: str

    @classmethod
    def from_source_zip(cls, archive: str | Path) -> "ProjectedBladeStations":
        """Read the two unmodified IEA files from the supplied ZIP archive."""
        with zipfile.ZipFile(archive) as files:
            blade_bytes = files.read(_BLADE)
            elastodyn_bytes = files.read(_ELASTODYN)
        blade = blade_bytes.decode("utf-8")
        elastodyn = elastodyn_bytes.decode("utf-8")
        nodes = int(_openfast_value(blade, "NumBlNds"))
        header = next(
            (i for i, line in enumerate(blade.splitlines())
             if re.search(r"\bBlSpn\b", line) and "BlChord" in line),
            None,
        )
        if header is None:
            raise ValueError("AeroDyn blade station header was not found")
        columns = blade.splitlines()[header].split()
        rows = [line.split() for line in blade.splitlines()[header + 2:header + 2 + nodes]]
        if len(rows) != nodes or any(len(row) != len(columns) for row in rows):
            raise ValueError("AeroDyn station count/column count is inconsistent")
        span = np.array([float(row[columns.index("BlSpn")]) for row in rows])
        chord = np.array([float(row[columns.index("BlChord")]) for row in rows])
        twist = np.array([float(row[columns.index("BlTwist")]) for row in rows])
        hub = _openfast_value(elastodyn, "HubRad")
        tip = _openfast_value(elastodyn, "TipRad")
        blades = int(_openfast_value(elastodyn, "NumBl"))
        cones = [_openfast_value(elastodyn, f"PreCone({i})") for i in range(1, blades + 1)]
        if (
            nodes < 2 or blades < 1 or len(set(cones)) != 1
            or not np.isclose(span[0], 0.0, atol=1e-10)
            or not np.isclose(hub + span[-1], tip, atol=1e-3)
            or not (np.isfinite(hub) and 0 < hub < tip)
            or not np.all(np.isfinite(span)) or not np.all(np.diff(span) > 0)
            or not np.all(np.isfinite(chord)) or np.any(chord <= 0)
            or not np.all(np.isfinite(twist))
        ):
            raise ValueError("The supplied OpenFAST blade geometry is inconsistent")
        radius = (hub + span) * math.cos(math.radians(cones[0]))
        if not np.all(np.diff(radius) > 0):
            raise ValueError("Projected station radii must increase")
        for array in (radius, chord, twist):
            array.setflags(write=False)
        return cls(
            radius_m=radius, chord_m=chord, twist_deg=twist,
            hub_radius_m=hub, tip_radius_m=tip, precone_deg=cones[0],
            number_of_blades=blades,
            blade_sha256=hashlib.sha256(blade_bytes).hexdigest(),
            elastodyn_sha256=hashlib.sha256(elastodyn_bytes).hexdigest(),
        )

    def at_radius(self, radius_m: float) -> tuple[float, float] | None:
        """Return chord and twist in the closed station interval; never extrapolate."""
        if not math.isfinite(radius_m) or radius_m < 0.0:
            raise ValueError("radius_m must be finite and nonnegative")
        if radius_m < self.radius_m[0] or radius_m > self.radius_m[-1]:
            return None
        return (
            float(np.interp(radius_m, self.radius_m, self.chord_m)),
            float(np.interp(radius_m, self.radius_m, self.twist_deg)),
        )


@dataclass(frozen=True)
class BandScenario:
    """Explicit controlled inputs; none of these is a selected final baseline.

    ``bird_speed_m_s`` is the explicitly supplied nominal horizontal bird
    speed; the fixed-plane crossing adapter projects it onto the plane normal.
    Whether the nominal value represents a ground or air speed remains an
    explicit baseline assumption, not a measured property of these paths.
    ``wind_heading_deg_toward`` is the direction the wind moves in the same
    Cartesian frame as Model 9's headings. It only determines whether each
    crossing is upwind or downwind; this module never rotates Model 9's plane.
    """

    rotor_rpm: float
    collective_pitch_deg: float
    bird_speed_m_s: float
    bird_length_m: float
    bird_wingspan_m: float
    flight_style: Literal["gliding", "flapping"]
    wind_heading_deg_toward: float

    def __post_init__(self) -> None:
        for name in ("rotor_rpm", "bird_speed_m_s",
                     "bird_length_m", "bird_wingspan_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("collective_pitch_deg", "wind_heading_deg_toward"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.flight_style not in ("gliding", "flapping"):
            raise ValueError("flight_style must be 'gliding' or 'flapping'")


def local_strike_probability(
    *, radius_m: float, vertical_from_hub_m: float,
    blade: ProjectedBladeStations, scenario: BandScenario,
    approach: Literal["upwind", "downwind"],
    incidence_cosine: float = 1.0,
) -> float:
    """Conditional Band probability at one valid rotor-plane crossing point.

    Return zero outside the represented projected station interval; the
    nominal rotor disc is checked separately by the Model 9 crossing adapter.
    Values exceeding one in the geometric approximation are capped at one.
    ``incidence_cosine`` is |cos(delta)|, with delta relative to the *fixed*
    rotor-plane normal. The default retains the original normal-incidence API.
    """
    if approach not in ("upwind", "downwind"):
        raise ValueError("approach must be 'upwind' or 'downwind'")
    if not math.isfinite(vertical_from_hub_m):
        raise ValueError("vertical_from_hub_m must be finite")
    if not math.isfinite(incidence_cosine) or not 0.0 < incidence_cosine <= 1.0:
        raise ValueError("incidence_cosine must be finite and in (0, 1]")
    section = blade.at_radius(radius_m)
    if section is None:
        return 0.0
    chord, twist = section
    omega = scenario.rotor_rpm * (2.0 * math.pi / 60.0)
    normal_speed = scenario.bird_speed_m_s * incidence_cosine
    alpha = normal_speed / (radius_m * omega)
    gamma = math.radians(twist + scenario.collective_pitch_deg)
    projection = (1.0 if scenario.flight_style == "flapping"
                  else abs(vertical_from_hub_m) / radius_m)
    sign = 1.0 if approach == "upwind" else -1.0
    blade_width = abs(sign * chord * math.sin(gamma) + alpha * chord * math.cos(gamma))
    bird_extent = max(scenario.bird_length_m,
                      scenario.bird_wingspan_m * alpha * projection)
    p = blade.number_of_blades * omega / (2.0 * math.pi * normal_speed)
    return min(1.0, p * (blade_width + bird_extent))


@dataclass
class BandCrossingAccumulator:
    """Passive Model 9 callback, returning a collision-weighted contribution.

    Supply one instance to ``RotorTransitModel.simulate(crossing_callback=...)``.
    Read ``result.conditional_rotor_transit_intensity`` from Model 9 for the
    separate nominal-disc exposure. Duplicate crossing events for the same
    bird and turbine use the maximum, matching Model 9's event accounting.

    The fixed Model 9 plane supplies the incidence normal. The supplied wind
    direction only determines upwind/downwind from the segment heading; it
    never rotates the plane. Choosing a wind heading aligned with that normal
    is part of the controlled fixed-rotor scenario. No yaw model is included.
    """

    blade: ProjectedBladeStations
    scenario: BandScenario
    number_of_trajectories: int
    number_of_turbines: int
    contribution_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.number_of_trajectories <= 0 or self.number_of_turbines <= 0:
            raise ValueError("Both trajectory and turbine counts must be positive")
        self.contribution_matrix = np.zeros(
            (self.number_of_trajectories, self.number_of_turbines), dtype=float
        )

    @property
    def expected_potential_strikes_per_entrant(self) -> float:
        return float(np.sum(self.contribution_matrix) / self.number_of_trajectories)

    def __call__(self, batch: "RotorCrossingBatch") -> None:
        if batch.height_bins_m is None or batch.height_probability_mass is None:
            raise ValueError("Model 9 must supply a discrete FHD for conditional Band weighting")
        wind = math.radians(self.scenario.wind_heading_deg_toward)
        wind_unit = np.array([math.cos(wind), math.sin(wind)])
        for i, j, offset, heading in zip(
            batch.trajectory_indices, batch.turbine_indices,
            batch.lateral_offset_m, batch.segment_heading_deg_toward,
        ):
            i, j = int(i), int(j)
            if not (0 <= i < self.number_of_trajectories and 0 <= j < self.number_of_turbines):
                raise ValueError("Crossing identity exceeds the declared population/layout")
            angle = math.radians(float(heading))
            incidence_cosine = abs(math.cos(
                angle - math.radians(batch.rotor_plane_heading_deg_toward)
            ))
            if incidence_cosine <= 1e-12:
                raise ValueError("Band normal speed is undefined at a tangent crossing")
            along_wind = float(np.dot([math.cos(angle), math.sin(angle)], wind_unit))
            if abs(along_wind) < 1e-12:
                raise ValueError("Upwind/downwind is undefined for a crosswind trajectory")
            approach = "downwind" if along_wind > 0 else "upwind"
            contribution = 0.0
            for height, mass in zip(batch.height_bins_m, batch.height_probability_mass):
                z = float(height) - batch.hub_height_m
                radius = math.hypot(float(offset), z)
                # Model 9 uses 1e-9 m tolerance for the circular rotor disc.
                if radius <= batch.rotor_radius_m + 1e-9:
                    contribution += float(mass) * local_strike_probability(
                        radius_m=radius, vertical_from_hub_m=z,
                        blade=self.blade, scenario=self.scenario,
                        approach=approach, incidence_cosine=incidence_cosine,
                    )
            self.contribution_matrix[i, j] = max(
                self.contribution_matrix[i, j], contribution
            )

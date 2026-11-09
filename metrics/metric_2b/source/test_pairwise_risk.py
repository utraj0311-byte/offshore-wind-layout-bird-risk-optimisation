import numpy as np

from config import Scenario
from pairwise_risk import (
    PairwiseGaussianRiskModel,
)


def main():
    scenario = Scenario()

    model = PairwiseGaussianRiskModel(
        scenario=scenario,
        interaction_scale_D=4.0,
    )

    print("")
    print("Pairwise Gaussian risk validation")
    print("---------------------------------")
    print(
        f"Sigma: {model.sigma_m:.2f} m"
    )
    print(
        "12-turbine theoretical risk scale: "
        f"{model.risk_scale():.2f}"
    )

    # -------------------------------------------------
    # Test 1: exact two-turbine distance ordering
    # -------------------------------------------------

    two_near = np.array(
        [
            [0.0, 0.0],
            [960.0, 0.0],
        ],
        dtype=float,
    )

    two_far_edge = np.array(
        [
            [0.0, 0.0],
            [6000.0, 0.0],
        ],
        dtype=float,
    )

    two_opposite_corners = np.array(
        [
            [0.0, 0.0],
            [6000.0, 6000.0],
        ],
        dtype=float,
    )

    risk_near = model.collisions(
        two_near
    )

    risk_far_edge = model.collisions(
        two_far_edge
    )

    risk_opposite_corners = (
        model.collisions(
            two_opposite_corners
        )
    )

    print("")
    print("Two-turbine distance test")
    print("-------------------------")
    print(
        f"Risk at 960 m: "
        f"{risk_near:.12f}"
    )
    print(
        f"Risk at 6000 m: "
        f"{risk_far_edge:.12f}"
    )
    print(
        "Risk at opposite corners: "
        f"{risk_opposite_corners:.12f}"
    )

    assert (
        risk_near
        > risk_far_edge
        > risk_opposite_corners
    ), (
        "Risk must decrease as pair distance increases."
    )

    # For two turbines in a square site, opposite
    # corners provide the maximum possible separation.
    expected_diagonal = np.sqrt(
        6000.0**2
        + 6000.0**2
    )

    observed_diagonal = np.linalg.norm(
        two_opposite_corners[0]
        - two_opposite_corners[1]
    )

    assert np.isclose(
        observed_diagonal,
        expected_diagonal,
    )

    # -------------------------------------------------
    # Test 2: permutation invariance
    # -------------------------------------------------

    layout = np.array(
        [
            [0.0, 0.0],
            [1200.0, 0.0],
            [2400.0, 0.0],
            [3600.0, 0.0],
            [4800.0, 0.0],
            [6000.0, 0.0],
            [0.0, 6000.0],
            [1200.0, 6000.0],
            [2400.0, 6000.0],
            [3600.0, 6000.0],
            [4800.0, 6000.0],
            [6000.0, 6000.0],
        ],
        dtype=float,
    )

    permutation = np.array(
        [
            7,
            2,
            11,
            0,
            5,
            9,
            1,
            10,
            4,
            8,
            6,
            3,
        ],
        dtype=int,
    )

    permuted_layout = layout[
        permutation
    ]

    original_risk = model.collisions(
        layout
    )

    permuted_risk = model.collisions(
        permuted_layout
    )

    print("")
    print("Permutation-invariance test")
    print("---------------------------")
    print(
        f"Original risk: "
        f"{original_risk:.12f}"
    )
    print(
        f"Permuted risk: "
        f"{permuted_risk:.12f}"
    )

    assert np.isclose(
        original_risk,
        permuted_risk,
        rtol=1e-12,
        atol=1e-12,
    ), (
        "Pairwise risk changed after turbine relabelling."
    )

    # -------------------------------------------------
    # Test 3: spread three-turbine configuration
    # -------------------------------------------------

    three_compact = np.array(
        [
            [0.0, 0.0],
            [960.0, 0.0],
            [0.0, 960.0],
        ],
        dtype=float,
    )

    three_spread = np.array(
        [
            [0.0, 0.0],
            [6000.0, 0.0],
            [0.0, 6000.0],
        ],
        dtype=float,
    )

    compact_risk = model.collisions(
        three_compact
    )

    spread_risk = model.collisions(
        three_spread
    )

    print("")
    print("Three-turbine spread test")
    print("-------------------------")
    print(
        f"Compact risk: "
        f"{compact_risk:.12f}"
    )
    print(
        f"Boundary-spread risk: "
        f"{spread_risk:.12f}"
    )

    assert spread_risk < compact_risk, (
        "The boundary-spread configuration should have "
        "lower pairwise risk than the compact layout."
    )

    # -------------------------------------------------
    # Test 4: per-turbine accounting
    # -------------------------------------------------

    total_risk = model.collisions(
        layout
    )

    per_turbine_risk = (
        model.collisions_per_turbine(
            layout
        )
    )

    allocated_total = float(
        np.sum(per_turbine_risk)
    )

    print("")
    print("Per-turbine accounting test")
    print("---------------------------")
    print(
        f"Direct total risk: "
        f"{total_risk:.12f}"
    )
    print(
        f"Allocated total risk: "
        f"{allocated_total:.12f}"
    )

    assert np.isclose(
        total_risk,
        allocated_total,
        rtol=1e-12,
        atol=1e-12,
    ), (
        "Per-turbine pair allocation does not sum "
        "to total risk."
    )

    # -------------------------------------------------
    # Test 5: expected pair count
    # -------------------------------------------------

    diagnostics = model.diagnostics(
        layout
    )

    expected_pairs = (
        len(layout)
        * (len(layout) - 1)
        // 2
    )

    assert (
        diagnostics["number_of_pairs"]
        == expected_pairs
    )

    assert expected_pairs == 66

    print("")
    print(
        "Expected 12-turbine pair count: "
        f"{expected_pairs}"
    )
    print("")
    print(
        "All pairwise Gaussian risk tests passed."
    )
    print("")


if __name__ == "__main__":
    main()
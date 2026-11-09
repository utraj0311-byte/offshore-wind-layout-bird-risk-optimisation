# Model 9 v4 change record

## Added

- Complete FLORIS + Model 9 + pymoo NSGA-II runner with smoke, pilot, and full
  profiles.
- Programmatic 12-direction × 7-speed-bin wind resource inherited from Model 8.
- Strict Scenario/FLORIS/bundled-reference geometry and turbine checks.
- Exact reuse of the supplied Model 8 energy endpoint, validated 80-layout
  initial population, FLORIS configuration, turbine reference, and run settings.
- Independent fresh validation of every training Pareto candidate.
- Fresh non-dominated and paired-batch robust-screening flags.
- Energy endpoint, rotor-transit endpoint, and explicitly descriptive balanced
  reference outputs; no claimed knee marker.
- Per-generation checkpoint population and state files.
- Full evaluation logging with every coordinate, raw metrics, objective scales,
  minimum spacing, and constraint violations.
- Software versions plus FLORIS config, installed turbine, reference turbine,
  and FHD provenance in `run_settings.json`.
- Seven coupling tests and a mocked end-to-end runner/output validation.
- `README_OPTIMISATION.md` and `MODEL8_INTEGRATION_AUDIT.md`.

## Corrected

- Replaced the obsolete 240 m prototype rotor diameter with the 242.24 m value
  used by the official IEA 15 MW FLORIS turbine and the completed Model 8 run.
- Consequently fixed the shared 4D spacing to 968.96 m and consistently derived
  all D-scaled bird-response and rotor-transit distances from 242.24 m.

## Preserved

- Frozen Model 8 horizontal path and meso-response logic.
- Model 9's circular fixed-plane rotor-transit definition.
- Conditional `E_RT_CE` as the primary low-variance objective, with sampled
  `E_RT` and Model 8 `E_enc` retained for audit.
- Rotor transit as an exposure endpoint, not collision probability or mortality.
- Exact northern-gannet FHD bytes and checksum.

## Verification

- 50 of 50 unit/regression tests pass under Python 3.12.13.
- A mocked one-generation end-to-end optimisation completed, wrote all expected
  checkpoint, Pareto, fresh-validation, layout, plot, and settings outputs, and
  ended with `run_status = completed`.
- A real FLORIS/pymoo smoke optimisation is intentionally left for the supplied
  `floris-env`; its result archive must be reviewed before running the pilot.

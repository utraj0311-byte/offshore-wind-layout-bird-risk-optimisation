# Model 9 v3 change record

## Added

- Exact conditional integration of the frozen discrete flight-height
  distribution over each circular rotor crossing.
- `N_RT_CE_farm`, `E_RT_CE`, `P_RT_CE`, and per-turbine conditional expected
  transits alongside the original sampled metrics.
- Sampled-versus-conditional agreement intervals and empirical variance
  reduction diagnostics.
- Conditional fixed-layout ranks and paired Student-t intervals.
- Default `E_RT_CE` bird objective in `Model9AEPRotorTransitProblem`, with an
  explicit sampled `E_RT` option.
- Focused CDF, circular-window, Monte Carlo equivalence, aggregation,
  diagnostic, and optimiser-plumbing tests.
- A 5 x 10,000-trajectory precision validation for each controlled layout.

## Preserved

- Frozen Model 8 path and meso-response logic.
- Original sampled constant height per bird and sampled `E_RT` audit output.
- Fixed circular rotor geometry and rotor-plane convention.
- One transit at most per bird-turbine pair.
- Rotor transit as the endpoint; no collision probability or mortality.
- Exact northern-gannet input bytes and SHA-256 provenance record.

## Verification

- 43 of 43 tests pass under Python 3.12.13.
- All 150,000 fixed-layout trajectory simulations complete with zero failures.
- Conditional variance is 61.90x–305.49x lower than sampled variance across
  the three validation layouts.

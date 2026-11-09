# Pre-flight and provenance — revised 1-degree Jensen runs

- 2-degree Metric 1a's freshly optimized endpoint failed the 2-vs-1 degree check (0.2461%). The user explicitly approved 1-degree reoptimisation, including a fresh 1a. Keep the old run as a diagnostic only.
- Historical original endpoint AEP at 12 sectors was reproducible to <0.001 GWh/yr using supplied code. 12 and 72 sectors gave the same Jensen wake-free AEP (~760.663033 GWh/yr). This does **not** imply 1-degree convergence for newly optimized layouts; hence the mandatory independent 0.5-degree endpoint test.
- AEP_ref=754.4007727706155 GWh/yr was inferred exactly from original archives, but the specific origin/layout cannot be determined. This is deliberately preserved.
- `pymoo==0.6.1.5` is confirmed by screenshot in `floris-env`; historical original version is not archived.
- Source modules risk, wake, constraints and original initialization & operators copied unchanged; only `config.py` wind-sector count updated to 360. New driver changes output, bookkeeping, verification and uses recovered archived AEP_ref.
- Jensen D=240 m, FLORIS D=242.24 m, never confuse the two.
- Final Jensen AEP is fresh 1-degree re-evaluation; difference from 1-degree optimiser AEP is (ideally) numerical roundoff. 0.5-degree is used for strict <0.2% convergence check.
- STOP after failure of ANY new endpoint's 1-vs-0.5 degree test; do not launch next metric. Diagnostic outputs retained.

# Model 8 input and integration audit

Date: 7 September 2026

## Supplied material

The integration used the supplied Model 8 Python configuration, FLORIS AEP
wrapper, FLORIS farm YAML, packaged IEA 15 MW turbine YAML, energy endpoint
layout, and complete Model 8 result archive.

The result archive checksum is:

```text
31ce46b53a4089a223a9a8f59168fb4d8fa87d18799c873854c0287ee0937a4c
```

Its settings record a population of 80, 100 generations, seed 42,
`5 × 2,000` objective trajectories, and `3 × 10,000` fresh-validation
trajectories. The completed run used 12 turbines on a 6 km × 6 km site,
`D = 242.24 m`, `H = 150 m`, 4D spacing, bird heading 225°, and a FLORIS
wake-free ceiling of 854.4956585617186 GWh/year.

## Wind input finding

No wind CSV is missing. Model 8 constructs the resource in Python:

- 12 equal direction sectors: 0°, 30°, …, 330°;
- dominant wind-from direction: 225°;
- directional weight proportional to `exp(2 cos(delta))`;
- Weibull shape `k = 2` and scale `A = 10 m/s`;
- seven speed bins: 4, 6, 8, 10, 12, 14, and 16 m/s; and
- outer-product normalisation to 84 condition probabilities summing to one.

Model 9 reproduces that exact programmatic structure in `config.py` and
`floris_aep.py`.

## Corrected geometry mismatch

The supplied older Python `config.py` declared a 240 m rotor diameter. The
official supplied FLORIS `iea_15MW.yaml` declares 242.24 m, and the completed
Model 8 archive confirms that the actual run used 242.24 m and 968.96 m
minimum spacing. Model 9 therefore uses 242.24 m consistently for:

- FLORIS turbine geometry;
- 4D layout spacing;
- response and encounter radii;
- return distance and numerical step; and
- the circular rotor-transit calculation.

The new FLORIS wrapper checks the Scenario, installed FLORIS turbine, and
bundled reference at startup and rejects the obsolete 240 m value.

## Reused layouts and population

`data/energy_endpoint_layout.csv` is exactly the `exact_energy` first row of
the supplied 80-row Model 8 initial population. All 80 layouts were checked:

- 24 decision variables per row (`x,y` for 12 turbines);
- all coordinates finite and within the site;
- smallest observed pairwise spacing approximately 970.112 m; and
- zero violations of the 968.96 m constraint.

Population sources comprise the four exact anchor layouts plus random and
jittered variants. Smoke and pilot profiles preserve all four exact anchors
and use a deterministic subset of the other rows. The full profile preserves
all 80 rows and their order.

## Frozen evidence files

| File | SHA-256 |
|---|---|
| `floris_config/gch_iea15mw.yaml` | `f6b788df9e38f006dfb6c68ddfb80ab090dad9cd555982a8d5c5e4b0c2f2d364` |
| `floris_config/iea_15MW_REFERENCE.yaml` | `93d2bcfcb619b1819efe8149ce9e8231656e51a29f2ae5c1079a46e339f02feb` |
| `data/energy_endpoint_layout.csv` | `53f37f2004c873104f55a840c7e433a2152020f63ecf0b7650c9da2c109b1744` |
| `data/model8_initial_population.csv` | `7505a9499a7bf230d8e126ba1922b76a0e7c673498fff30b094bd4402877abf3` |
| `data/model8_reference_run_settings.json` | `db353e2ea07baa3fa4a36ccdca6401fc6c25dc4aea8e4c9cfb3b6ee0c9af08b2` |
| `data/northern_gannet_Johnston2014_FHD.csv` | `adc9ea903db9c8e34bfe5b88c382381b737c2e676d79af437f66d198208a0f2c` |

These hashes describe the bundled v4 evidence inputs. Generated optimisation
outputs also record the active FLORIS config, installed turbine, bundled
turbine reference, FHD, seeds, and software versions.

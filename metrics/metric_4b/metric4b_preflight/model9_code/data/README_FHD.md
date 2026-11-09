# Flight-height data

`flight_height_distribution_TEMPLATE.csv` shows the canonical schema.
`northern_gannet_Johnston2014_FHD.csv` is the exact user-supplied table used in
the real validation. It retains its native `height,prop` header and original
bytes; the loader explicitly supports that stochLAB/Johnston schema.

The table is documented as the northern gannet maximum-likelihood estimate
(`Gannet.est`) from stochLAB's `Johnston_Flight_heights_SOSS` object, based on:

Johnston, A., Cook, A. S. C. P., Wright, L. J., Humphreys, E. M. & Burton,
N. H. K. (2014). Modelling flight heights of marine birds to more accurately
assess collision risk with offshore wind turbines. *Journal of Applied
Ecology*, 51, 31–41. <https://doi.org/10.1111/1365-2664.12191>

stochLAB documentation describes `prop` as the proportion of birds in each
one-metre flight-height interval:
<https://rdrr.io/cran/stochLAB/man/Johnston_Flight_heights_SOSS.html>

The exact upstream stochLAB release or commit was not embedded in the upload.
If known, add it to this record before the thesis input freeze. Reproducibility
of the actual input used is protected by its SHA-256 checksum in
`northern_gannet_Johnston2014_FHD_metadata.json` and a regression test.

`SYNTHETIC_TEST_ONLY_fhd.csv` exists solely to exercise the software. Its
values are invented and must never be reported as a bird distribution or used
for thesis conclusions, optimisation, or ecological interpretation.

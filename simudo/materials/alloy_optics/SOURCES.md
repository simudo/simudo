# Provenance of the alloy optical-constant data

One entry per dataset: the original measurement, the intermediary the CSVs were
taken from (if any), the licence that applies, and how the files were produced.

## AlGaAs — 9 CSVs, `AlGaAs/Al_*.csv`

**Original publication**

> D. E. Aspnes, S. M. Kelso, R. A. Logan and R. Bhat,
> "Optical properties of Al_x Ga_(1-x) As",
> *J. Appl. Phys.* **60**, 754–767 (1986). https://doi.org/10.1063/1.337426

**Collated via** refractiveindex.info, which is the source the CSVs were taken
from: https://refractiveindex.info/?shelf=other&book=AlAs-GaAs&page=Aspnes-0

> M. N. Polyanskiy, "Refractiveindex.info database of optical constants",
> *Sci. Data* **11**, 94 (2024). https://doi.org/10.1038/s41597-023-02898-2

**Licence.** The refractiveindex.info database is released under the
[CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/):
free to use, modify and redistribute, including commercially, with no permission
required. Attribution is requested but not required — the Polyanskiy citation
above is that attribution. CC0 is compatible with Simudo's LGPLv3.

One file per measured composition, x = 0.000 to 0.804; `OpticsInterpolator` in
`simudo/materials/alloy_optics.py` interpolates between them in composition and
photon energy.

## GaNAs — 5 CSVs, `GaNAs/Ga N_*.csv`

**Original publication**

> G. Leibiger, V. Gottschalch, B. Rheinländer, J. Šik and M. Schubert,
> "Model dielectric function spectra of GaAsN for far-infrared and near-infrared
> to ultraviolet wavelengths",
> *J. Appl. Phys.* **89**, 4927–4938 (2001). https://doi.org/10.1063/1.1359422

**Intermediary: not recorded.** Unlike the AlGaAs set, no database or other
intermediary is named for these files, so it is not established whether they were
digitised from the paper, computed from its model dielectric function, or taken
from a third party. **The licence position therefore differs from AlGaAs's and is
not settled.**

GaNAs is **not part of the released material set** for this reason among others;
`galliumarsenidenitride.py` is internal-only. Establish the route these files came
by before shipping them or relying on them outside the group.

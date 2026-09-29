# Third-party notices

The files in `engine/` descend from the transport implementation used by
ARTEMIS and include project-specific changes for frozen latent coordinates,
configurable diffusion, fold-aware context labels, and GRN drift alignment.
ARTEMIS is licensed under GPL-3.0; the complete license text is in
`LICENSE-GPL-3.0`. Original GRASB contributions are offered under MIT as
described in `LICENSING.md`, but that does not remove the GPL obligations of a
combined or derivative engine file.

Several engine files retain comments identifying code adapted from
[`unbalanced_sb`](https://github.com/matteopariset/unbalanced_sb), Copyright
2023 Matteo Pariset, licensed under the MIT License. Its license is reproduced
in `THIRD_PARTY_UDSB_LICENSE`.

The GRN construction in `scripts/build_grn_drift.py` reimplements small,
documented coefficient-fitting and TF-mask recipes from CellOracle. No
CellOracle source code or promoter map is distributed here. Users must obtain
the relevant promoter-supported TF–target resource under its own terms.

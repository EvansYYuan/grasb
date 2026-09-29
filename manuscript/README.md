# Preprint

`grasb.tex` is the current manuscript source with the updated title:

> GRASB: Gene-Regulatory Alignment of Schrödinger Bridges for Single-Cell
> Trajectories in Latent Space

The three publication figures referenced by the source are included under
`figures/` in PDF, PNG, and SVG form, together with the BibLaTeX database
`grasb.bib`. Figure-generation code, derived plot tables, cached UMAP
coordinates, and bundled fonts are documented in `figures/README.md`.

`grasb_preprint_preview.pdf` is the latest compiled manuscript preview from the
private writing workspace. It predates the title-only rename in `grasb.tex`; the
scientific content is the same. It should be rebuilt before a public release so
the PDF title page and metadata match the source.

The source requires a LaTeX installation with `biblatex`/Biber. A typical build
is:

```bash
latexmk -pdf grasb.tex
```

Author names, affiliations, and the final repository URL remain explicit
placeholders in the draft and should be filled before release.

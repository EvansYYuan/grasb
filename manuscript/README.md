# Preprint

`grasb.tex` is the current manuscript source with the updated title:

> GRASB: Gene-Regulatory Alignment of Schrödinger Bridges for Single-Cell
> Trajectories in Latent Space

The three publication figures referenced by the source are included under
`figures/` in PDF, PNG, and SVG form, together with the BibLaTeX database
`grasb.bib`. Figure-generation code, derived plot tables, cached UMAP
coordinates, and bundled fonts are documented in `figures/README.md`.

`grasb_preprint_preview.pdf` is the compiled preview corresponding to
`grasb.tex`. The source and PDF include the current GRASB title, author list,
affiliations, and PDF metadata.

The source requires a LaTeX installation with `biblatex`/Biber. A typical build
is:

```bash
latexmk -pdf grasb.tex
```

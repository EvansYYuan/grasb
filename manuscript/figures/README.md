# Manuscript figures

The three figures used by `../grasb.tex` are stored in vector PDF, editable SVG,
and high-resolution PNG form:

| Manuscript figure | Files | Generation route |
|---|---|---|
| GRASB architecture | `grasb_main_figure.*` | Component drawings from the `draw_*` and `render_*` scripts, assembled into the final schematic |
| Gene-rank and expression comparison | `figure2_combined_v2.*` | `scripts/plot_figure2_combined_v2.py` |
| Expression-aligned pancreas ranks | `figure3_pancreas_ranked_violins_v1.*` | `scripts/plot_figure3_pancreas_ranked_violins_v1.py` |

Create the plotting environment from the repository root:

```bash
conda env create -f environment-figures.yml
conda run -n grasb-figures python manuscript/figures/scripts/plot_figure2_combined_v2.py
conda run -n grasb-figures python manuscript/figures/scripts/plot_figure3_pancreas_ranked_violins_v1.py
```

The scripts write PDF, SVG, and PNG outputs directly to this directory.

`data/` contains the versioned derived tables, manifests, and cached UMAP
coordinates used during figure assembly. Raw expression, metadata, and
decoded-drift tables are not duplicated here; scripts resolve them from the
repository-level `data/` and `results/` directories described in
`../../docs/data.md`.

The architecture figure was composed from separately generated vector modules,
so there is no single script that reproduces its final manual layout. The
component-generation scripts are included to preserve the programmatic source
of its encoder, decoder, bridge, GRN, matrix, and ranking elements. They write
to `components/`. The 14 rendered components actually used during composition
are already included there in PDF, PNG, and SVG form; see
`components/README.md`. The final SVG is the editable source for the assembled
layout.

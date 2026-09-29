# Results directory

Training artifacts, predictions, and evaluation tables are written here.

The committed release contains:

- `manuscript_summary.csv`: the exact values reported in the two main results
  tables, in tidy form;
- `pancreas/`: per-unit EDT2 GSEA and the calibrated genuine/no-GRN/shuffle
  permutation summaries;
- `zebrafish/`: final corrected-audit model/condition summaries, paired-seed
  contrasts, and held-out Wasserstein summaries.

Large binary artifacts such as `.pkl`, `.npz`, `.npy`, checkpoints, and logs are
ignored by Git. Lightweight reproducibility outputs—including CSV tables, JSON
manifests, and YAML summaries—are intentionally trackable and should be committed
when they support a reported result or figure.

Do not commit a table merely because it is small: it should have a documented
generating command and correspond to a stable analysis rather than a scratch run.
The multi-megabyte per-gene drift/expression matrices and model pickles are not
duplicated here; the compact committed tables are their publication-facing
derivatives. Figure-specific plotted snapshots remain under
`manuscript/figures/data/`.

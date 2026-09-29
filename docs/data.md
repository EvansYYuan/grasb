# Data contract

GRASB operates on a preprocessed expression matrix plus cell annotations and a
candidate TF–target topology. Raw study data and promoter resources are not
redistributed here.

## Expression matrix

`pancreatic_preprocessed.csv` is a cells-by-genes CSV with:

- a unique cell barcode in the first/index column;
- log1p expression values in the modeled HVG space;
- gene-symbol column names; and
- a `time` column containing ordered developmental snapshots.

The reported pancreas split uses timepoint indices `0,1,2,4,5,7` for training
and holds out `3,6`. The VAE, GRN fitting, and bridge fitting must share this
split. `build_fold_aware_grn.py` rejects overlap with the held-out indices.

## Pancreas metadata

`pancreatic_metadata.tsv` is tab-separated, indexed by `library.barcode`, and
contains `Assigned_cluster` and `CellWeek`. Barcodes are reindexed to the
expression matrix; missing labels are rejected during GRN construction.

The study is Veres et al. (2019), *Nature* 569:368–373, GEO GSE114412. Download
and preprocessing details will be added with the manuscript release.

## Candidate GRN

`human_promoter_base_grn.parquet` follows CellOracle's promoter base-GRN table:
`peak_id`, `gene_short_name`, and one indicator column per candidate TF. The
builder removes self-edges and restricts both endpoints to modeled genes.

The repository does not redistribute CellOracle's promoter resource. Obtain it
from CellOracle and pass its path with `--base-grn`.

## Zebrafish

The secondary analysis uses Farrell et al. (2018) zebrafish embryogenesis data
(3.3–12 hpf), with 5.3, 7, and 9 hpf held out. The default filenames are
`zebrafish_expression.csv`, `zebrafish_metadata.csv`,
`zebrafish_groups.csv`, and `zebrafish_promoter_base_grn.parquet`.
`config/zebrafish/segment_map.csv` defines the evaluated lineage segments.

The expression table contains raw counts (cells by genes); the builder applies
`log1p`. Metadata are indexed by cell ID and contain `HPF`. The groups table is
indexed by the same IDs and contains `segment`. The training entry point accepts
fractional-time artifacts and segment labels through
`--label-mode segment --segment-map config/zebrafish/segment_map.csv`.

The Figure 2 renderer expects the corresponding log1p table as
`data/zebrafish_preprocessed.csv`. Figure-generation scripts also consume the
decoded-drift table `results/per_type_drift.csv`; optional trajectory-comparison
figures use the clearly named pancreas trajectory CSVs defined at the top of
`manuscript/figures/scripts/plot_drift_expression_context.py`.

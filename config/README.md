# Experiment configuration

Configuration is organized by study system. Each dataset directory contains an
experiment summary and the versioned biological mappings used for evaluation.

- `pancreas/`: train/validation split, model settings, the frozen Veres EDT2
  marker list, legacy sensitivity panel, lineage map, summary, and references.
- `zebrafish/`: train/validation split, model settings, segment-to-lineage map,
  Farrell module mapping, published modules, and classical-marker sensitivity
  panel.

For pancreas, `veres_edt2_markers.csv` is the human-readable authoritative
primary list used by the manuscript. `marker_panel.csv` is the loader-oriented
union of that frozen list and the superseded `v1` list, retained so historical
sensitivity runs remain reproducible. The two should not be interpreted as two
independent biological references.

The experiment YAML files are human-readable records of the manuscript
configuration. Command-line arguments remain authoritative at runtime; commands
printed in logs or attached to released results should agree with these files.

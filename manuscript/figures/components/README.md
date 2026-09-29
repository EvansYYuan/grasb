# Figure 1 components

These are the rendered components used to construct the editable GRASB
architecture figure. Every component is retained as PDF, PNG, and SVG.

| Component stem | Role in the architecture figure |
|---|---|
| `figure1_pancreatic_reference_snapshots_hollow_heldout` | Training-snapshot timeline with a visibly held-out intermediate state |
| `gene_expression_matrix_icon` | Timepoint-specific expression input |
| `tf_gene_relationship_icon` | Promoter-supported TF–target topology |
| `time_batched_grn_icon` | Context-specific regulatory network |
| `gene_regulatory_network_icon` | Standalone network glyph used in GRN compositions |
| `frozen_encoder_module_no_lock` | Frozen VAE encoder |
| `latent_unbalanced_sb_core` | Latent unbalanced Schrödinger bridge paths |
| `latent_unbalanced_sb_module_v3` | Latest complete bridge-module rendering |
| `usb_sb_drift_vector_field` | Learned latent drift field |
| `usb_grn_guidance_vector_field` | Pushed-forward GRN guidance field |
| `usb_grn_guided_result_vector_field` | Aligned latent drift field |
| `encoder_grn_pushforward_explicit_tex` | Encoder-JVP annotation |
| `frozen_decoder_module_no_lock` | Frozen VAE decoder |
| `drift_gene_ranking_icon` | Gene-resolved decoded-drift ranking |

The corresponding generators live in `../scripts/`. The final arrangement was
assembled manually from these vector components; `../grasb_main_figure.svg` is
the editable master for that composition.


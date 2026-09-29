# Reproducibility notes

## Leakage controls

- The VAE is fit only on training timepoints and then frozen.
- Context-specific regulatory coefficients and pooled shrinkage priors exclude
  held-out timepoints.
- When a validation timepoint lies between two training snapshots, the GRN
  builder creates one merged interval spanning the validation snapshot.
- Latent standardization statistics are estimated from training cells only.
- Genuine and shuffled conditions reuse the same pre-calibrated context scales.

## Regulatory artifact

The serialized artifact stores gene order, interval tiling, context-specific
sparse edges and coefficients, intercepts, scale factors, and construction
metadata. `GRNDrift.from_pkl` converts this representation to dense-per-bin
coefficient rows over a shared sparse edge list.

The manuscript configuration deploys the regulon component
`v_GRN(x,c) = gamma_c A_c x`. The builder therefore calibrates `gamma_c` against
that same component (`--component regulon`); calibrating one field and deploying
another changes the effective velocity scale.

## Randomness

Bridge runs, VAE training, topology shuffles, and evaluation permutations expose
explicit seeds. Reported zebrafish summary values use three independently
trained seeds. The pancreas topology null uses six fixed shuffled topologies.

## Evaluation boundary

Decoded-drift enrichment assesses local biological organization. Held-out W2
assesses integrated population reconstruction. They are different outputs and
must not be collapsed into one performance claim.


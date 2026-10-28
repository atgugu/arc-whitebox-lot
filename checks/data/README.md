# Raw data for `checks/recompute.py`

## `lot_runs/inchain*.jsonl`

Per-network logs of the lean LOT experiments (1-2 October 2026). The lean implementation is a plain-numpy research version of V29's chain in float32, run outside the FLOP meter. One JSON record per network and variant:

- `i`: network index in the public WhestBench Phase 2 dataset, networks 0-11;
- `variant`: `v29+f32` is the V29 chain; `|lot:<level>:<terms>...` adds LOT, where `shape` keeps V29's per-layer level and takes only LOT's neuron-to-neuron shape, and `rep` replaces V29's κ₄ with LOT's own (fit-free level). Term sets: `chainstar` = exact chain + star; `he2` = exact chain + star + induced; `proj32:1:1:0:chain` = chain projected on 32 covariance modes, plus same-hub and star;
- `final_mse`: final-layer MSE against the dataset's Monte Carlo means (10⁹ samples per network); `all_mse` over all layers; `layer_rms` per layer;
- `sec`: run time; `tag`: the experiment batch.

The files are the logs as written, including variants that Table 1 does not use.

## `graded_public_mlps.csv`

The official grader's per-MLP public scores for five submissions of this work: s43 #336780 (base of the controls), its three single-variable controls #336911-#336913, and #337608. Columns: final-layer MSE and adjusted score, as reported by the grader (4 significant digits), and the billed FLOPs of each MLP. Only the 50 public MLPs are included. Because the per-MLP values are rounded, sums over them can differ from the grader's own aggregates in the fifth digit: LOT-off raw is ×1.35747 from per-MLP values and ×1.35752 from the aggregates (the paper's ×1.358).

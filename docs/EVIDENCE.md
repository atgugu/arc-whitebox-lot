# Evidence index for the write-up

Every grey tag in `paper/writeup.pdf` (for example `[E18-GC03]`) and every "Evidence:" line under a table refers to a row of the project's evidence ledger. This file reproduces those rows.

- **Class:** M = measured in a logged run; D = derived from logged data; P = protocol; C = claim from a document or another team.
- **Source** names where the number was logged: a path when the file is in this repository, otherwise a description of a file in the author's project tree (project journal, knowledge base, theory notes, experiment report, grader JSON), which is not published. The data behind every figure and table *is* shipped, under `figures/data/`, `controls/` and `docs/`.
- **Value cells** copied from the ledger keep its field names: `cb` = billed-FLOP ratio; `box` = the machine that ran the arm; `fv` = local paired validation; `dplans`/`dops` = extra plans / calls per network; arm, candidate and tag names (for example `ablLOT`, `hom2`, `s40a`, `k32y`) are internal variant IDs; `lean` = the project's plain-numpy research version of the chain.
- **Graded numbers** come from the author's own grader telemetry; the per-MLP comparisons of the three controls are in `figures/data/graded_controls_paired.csv`.

## Ledger rows cited in the paper

| id | quantity | value | class | source |
|:--|:--|:--|:--|:--|
| E01-06 | LOT port in flopscope (PORTLOT_LMIN=1): raw / extra cost | raw x0.842 [0.823, 0.862], 24/24; +9.2 u; +362 calls | M | `project knowledge base` |
| E01-04 | exact LOT shape (chain + star + induced) at V29's level, lean, networks 0-11 | -25.0% [-29.4, -20.5], 12/12 | M | theory run logs; `figures/data/shape_level_grid.csv` |
| E01-05 | projected K = 32 + same-hub + star at V29's level, lean, networks 0-11 | -17.8% [-20.5, -15.1], 12/12 | M | theory run logs; `figures/data/shape_level_grid.csv` |
| E01-07 | exact chain + star at its own, fit-free level, lean, networks 0-11 | -21.0% [-24.8, -17.1], 12/12 | M | theory run logs; `figures/data/shape_level_grid.csv` |
| E01-08 | projected K = 32 + same-hub + star at its own level, lean, networks 0-11 | +38.1% [+27.5, +48.8], 0/12 | M | theory run logs; `figures/data/shape_level_grid.csv` |
| E01-09 | LOT explains the neuron-to-neuron variance of the true kappa4 (MC N=2e6) vs V29 memoryless core | 86-91% vs 58-69% (ledger: 88-91%) | M | `project knowledge base` |
| E01-16 | share of the final MSE born in layers 12-15 / 8-11 / 1-7 (V29 channel decomposition) | 63% / 30% / 7% | M | `project knowledge base` |
| E01-27 | first-order model c2 dvar + c3 dD3 + c4 dk4 explains the local mean error (clean N=1e8 truth, lean chain) | 93-97% at layers >= 8; shares var 0.52-0.62, kappa3 0.27-0.39, kappa4 0.18-0.26 | M | `project journal` |
| E01-28 | effect on the final MSE of removing ONE layer's whole local error (first-order transport) | L15 20.8%, L14 18.6%, L13 15.0%, L12 11.3%, L11 11.1%, L10 8.2%, L9 6.8%, L8 2.5% | M | `project journal` |
| E02-14 | influence-ranked exact rows vs \|alpha\| ranking: residual energy ratio (best influence score / oracle per-row order) | abs 1.000; coh 1.124; inc_s2 0.984; or_loc 0.947 (pooled, 12 nets) | M | `experiment report "adj"` |
| E04-02 | graded effect of in-chain HZ: raw s04 -> s08 | 1.9426e-8 -> 1.7168e-8 (x0.8838) | M | `grader JSON #335749` |
| E04-04 | theta learning curve (held-out final MSE, nets used for the fit m = 6/12/24/36/47; in-sample) | 1.6557/1.6413/1.6348/1.6348/1.6335e-8; in-sample 1.6283e-8; limit ~1.632e-8 | M | `project journal` |
| E04-08 | young 3+2 kappa4 tree (L10 lean gate) vs production k32y | lean x0.970 [0.961, 0.980] 6/6 -> production raw x1.0004, FLOPs +1.4% | M | `experiment report "k32"` |
| E04-09 | k32y production result | raw x1.0004, FLOPs +1.4%; lean -3% did not transfer | M | `project journal` |
| E04-20 | HZ fits the LOCAL MEAN ERROR, not kappa4 truth (fitting kappa4 truth does not help) | qualitative | M | `project journal` |
| E05-04 | same file (s37, sha 6d825afcf6cd) on two machines: per-net raw-MSE ratio sd | 0.898% (max 2.45%), FLOPs sd 0.0033%, pred_sha identical 0/200 | D | `project table figure_stats.json` |
| E08-01 | ZF7 in-place 7-slot Strassen buffers: cost on nets 0,1 (bit-identical predictions) | cb x0.9834 (-1.66%) | M | `project journal` |
| E08-02 | ZF7 core adds were this share of the s25 bill (before the alternative basis) | 32.3 GF = 9.66% | M | `project journal` |
| E08-03 | alternative-basis (Karstadt-Schwartz) Winograd for the ZF7 core: 8-net table | cb x0.98463 (-1.54%), raw x1.00011 | M | `project journal` |
| E08-04 | dense-product agent: exact cuts (DN set) 8-net table | cb x0.99241 (-0.76%), +1353 ops | M | `project journal` |
| E08-06 | residual cost per metered call, client side | ~11 us/op; ~22.6k ops ~0.25 s | M | `project journal` |
| E08-15 | FLOP profile (s34 era, net 1): _dslices / young transport / tier-1 lift / HZ finish; Strassen engine share | 35.2% / 28.3% / 8.7% / 5.8%; 72.5% | D | `project table flop_profile.csv` |
| E17-21 | marg2 premise check: the dropped correction is MORE accurate vs truth in every band | 5-20x error cut in every band | D | `experiment report "w5marg2"` |
| E17-25 | K22-injection study: true N=1e8 K22 injected into the chain | x1.246 (all programs) | M | `experiment report "g2"` |
| E17-26 | K22-injection study: K22 uniform level 0.97/1.03 and 0.8/1.2 | x1.169/x1.745; x7.88/x71.2 | M | `experiment report "g2"` |
| E17-27 | K22-injection study mechanism: K22 compounding loop tuned by LAM/REF_R tables | loop | M | `experiment report "g2"` |
| E17-29 | Ward-identity study: a K<=4 chain cannot be scale-covariant at first order | V_mean = dropped kappa5/6 faces (corr 0.9993-0.9999) | M | `experiment report "g8"` |
| E17-30 | Ward-identity study: dominant violator K31TWO constant b1 | ~95% of C violation | M | `experiment report "g8"` |
| E17-31 | Ward-identity study: covariant fixes in-chain | full x22.0; births only x351; mean faces x1.113; LOT g4_mean x2.36 | M | `experiment report "g8"` |
| E17-35 | kappa4 truth/chain by alpha band, L10-14 bulk | k4 t/c 1.106 (k3 0.997) | D | `experiment report "w5marg2"` |
| E17-36 | kappa4 truth/chain by alpha band, L1-5 bulk | k4 t/c 0.996 | D | `experiment report "w5marg2"` |
| E17-41 | L10 scale-mode extraction in the lean chain | x1.47-42 worse | M | `experiment report "k32"` |
| E17-70 | cold call: the 5 highest residuals of every run are the first MLP of each worker | s40 0.312-0.329 s vs warm median 0.257 | M | `project journal` |
| E17-A01 | final-stack ablation arm ablLOT vs s40a (nets 0-59) | arm=ablLOT; box=B3; same_box_as_ref=False; n=60; raw_ratio_off_over_on=1.36518; raw_se=0.01355; z=22.98; nets_worse_when_off=60; flops_ratio_approx=1.11147; score_ratio_approx=1.51786; dplans_per_net=25.6; dops_per_net=-5706 | D | `project table final_stack_ablation.csv` |
| E17-C04 | local vs grader pair, A = s40 | A=s40; B=s42; fv_tag_A=s40a; fv_tag_B=w5m2a; n_fv=300; fv_ratio=0.99178; fv_se=0.00075; n_graded_public=50; graded_public_ratio=0.99226; graded_se=0.00186; z_graded_minus_fv=0.24; board_score_ratio=0.99226; note=marg2 | D | `project table calibration_17oct_table.csv` |
| E17-H02 | reused-vs-holdout pair hom2 | candidate=hom2; reference=hos40; net_set=holdout 500-799; nets=500-799; n=300; score_ratio=0.99066; se=0.00075; z=-12.54; raw_ratio=0.99067; flops_ratio=0.99999 | D | `project table holdout_pairs.csv` |
| E17-H04 | reused-vs-holdout pair hod2h | candidate=hod2h; reference=hom2; net_set=holdout 500-799; nets=500-799; n=300; score_ratio=0.99874; se=0.00056; z=-2.26; raw_ratio=0.99879; flops_ratio=0.99995 | D | `project table holdout_pairs.csv` |
| E17-H05 | reused-vs-holdout pair hog1d2 | candidate=hog1d2; reference=hom2; net_set=holdout 500-799; nets=500-799; n=300; score_ratio=0.99879; se=0.00056; z=-2.17; raw_ratio=0.99878; flops_ratio=1.00002 | D | `project table holdout_pairs.csv` |
| E17-H06 | reused-vs-holdout pair w5m2c15 | candidate=w5m2c15; reference=w5m2a; net_set=reused 0-299; nets=0-299; n=300; score_ratio=0.99887; se=0.0005; z=-2.24; raw_ratio=0.99888; flops_ratio=1.0 | D | `project table holdout_pairs.csv` |
| E17-H07 | reused-vs-holdout pair hoc15 | candidate=hoc15; reference=hom2; net_set=holdout 500-799; nets=500-799; n=300; score_ratio=0.99994; se=0.00051; z=-0.11; raw_ratio=0.99994; flops_ratio=1.00001 | D | `project table holdout_pairs.csv` |
| E18-201 | graded control s43cH (HZ_ON=0) #336911 | 2.47489e-9; raw x1.228; F/B 0.1254 | M | `project journal` |
| E18-202 | graded control s43cL (PORTLOT=0) #336912 | 3.18033e-9; raw x1.358; F/B 0.1458 | M | `project journal` |
| E18-80 | LOT's own mean-program value | ~x1.08 | M | `experiment report "txprod"` |
| E18-GC02 | graded single-variable control s43cH vs s43 #336780 | build=s43cH; submission=336911; knob_vs_s43=HZ_ON=0; graded_score=2.47489e-09; graded_raw=1.97377e-08; eff_F_over_B=0.1254; raw_ratio=1.2283; flops_ratio=0.9399; score_ratio=1.1545; local_raw_ratio_s40a=1.26236; local_raw_se=0.00979; local_flops_ratio_approx=0.93983; local_lazy_plans=0.0; … | D | `project table graded_controls.csv` |
| E18-GC03 | graded single-variable control s43cL vs s43 #336780 | build=s43cL; submission=336912; knob_vs_s43=PORTLOT=0; graded_score=3.18033e-09; graded_raw=2.18140e-08; eff_F_over_B=0.1458; raw_ratio=1.3575; flops_ratio=1.0928; score_ratio=1.4835; local_raw_ratio_s40a=1.36518; local_raw_se=0.01355; local_flops_ratio_approx=1.11147; local_lazy_plans=25.6; … | D | `project table graded_controls.csv` |
| E18-GC04 | graded single-variable control s43cD vs s43 #336780 | build=s43cD; submission=336913; knob_vs_s43=D21_CUT=99; graded_score=2.31574e-09; graded_raw=1.60205e-08; eff_F_over_B=0.1445; raw_ratio=0.9970; flops_ratio=1.0835; score_ratio=1.0802; local_raw_ratio_s40a=0.99229; local_raw_se=0.00337; local_flops_ratio_approx=1.09306; local_lazy_plans=30.0; … | D | `project table graded_controls.csv` |
| E18-W09 | s44 graded score / raw (scoreSecondary) | 2.18120e-09 / 1.66103e-08 | D | `project table wall_lesson_s44.csv` |
| E18-W10 | s44 score ratio vs s43 | 1.0175 | D | `project table wall_lesson_s44.csv` |
| E18-W13 | s44 MLPs with HZ skipped (FLOPs < 0.95 x s43) | 16 | D | `project table wall_lesson_s44.csv` |

## Other tags

| tag | meaning |
|:--|:--|
| `09/band scales` | the band-scale audit of the pair closure: 50 paired arms on one base; the ×1.22 row is the uniform ×1.15 scaling of the mean program's κ₄ (n = 34). Project file `project table band_scale_map.csv`. |
| `08/README` | the engineering notes of the project (`project table README.md`); the 72% engine share is from a FLOP profile of build s34. |
| `05/calibration` | 14 pairs of consecutive graded builds against their local paired predictions: χ² = 15.0 on 14 degrees of freedom. Project file `project table calibration_17oct_stats.json`. |
| `theory notes §2`, `§2.5` | the project's theory notes, which hold the lean LOT experiments. Their per-network logs are shipped in `checks/data/lot_runs/`, and `checks/recompute.py` rebuilds Table 1 from them (`figures/data/shape_level_grid.csv`). |
| `theory log, one network` | `theory run log lot1024_mlp0.log`: attenuation-corrected correlations of LOT variants with the Monte Carlo κ₄ on one full-size development network ($N=2\times10^6$); summarised in `docs/LOT_DERIVATION.md` §4. |

Known slips in the ledger itself, corrected in the paper: the projected form at its own level is +38.1% (the ledger's +37.1% is an arithmetic slip; its own means give 3.1827/2.3043 = 1.381), and the D21 all-exact FLOP ratio is ×1.083 (the ledger rounds it to 1.0835). E01-09's range is 86-91% when read per layer from `figures/data/lot_variance_explained.csv` (the ledger says 88-91%). Rows E01-04 to E01-08 above give the values recomputed for Table 1.

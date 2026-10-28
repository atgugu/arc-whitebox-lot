# ARC White-Box Estimation Challenge 2026 (Phase 2) estimator, team antoniu_gugu9.
# Attribution: derives from 504aldo's V29 cumulant-propagation estimator (MIT) and ARC's cumulant-propagation
# reference (MIT). Several components (FIJA2 frozen joins and its FIJA2X follow-ups, OLD_HOT1) are ports of
# LordProcciuto's whestbench-lordprocciuto (MIT). The original copyright and permission notices are reproduced
# below. All other changes (source-based LOT, demand-driven D21/CDP/TDP projections, Edgeworth tails for dropped
# neurons, the learned HZ correction, the fast matmul engine, call-count work, etc.) are this team's own work,
# developed with substantial assistance from Claude (Anthropic) and Codex coding agents; see the technical write-up.
#
# MIT License
#
# Copyright (c) 2026 504aldo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# MIT License
#
# Copyright (c) 2026 504aldo
# Copyright (c) 2026 LordProcciuto
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# MIT License
#
# Copyright (c) 2026 Alignment Research Center
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""MERGE (agent `merge`, 2026-10-02): V29 + the stacked round-2 components, each behind an env kill-switch (CHANGES.md / research/exp_merge.md).
  PORTLOT  (portlot)    LOT kappa4 shape in the mean program, all layers          PORTLOT=0, PORTLOT_LMIN
  TRIMDEMAND (trimdemand) no feedback legs for births 0-3, R_RES 8, frozen rebuilds  TD_OFF=1 (TD_FB_B0, TD_R_RES, TD_FROZEN)
  TRIMALG  (trimalg)    Gram cores, nonlin group-sums, Winograd, C_pre in the hub family, old-tier min block 12   TA_GRAM/TA_CPRE/TA_WICK/TA_NL/TA_WINO/TA_MMFB=0, TA_OLD_MIN
  BASISQ   (basisq)     P-leg-only joiner sketch after the first join              BQ_SRCP=0
  LC_K22   (latecov)    k22swap level reference (measured: no gain with PORTLOT)   LC_K22=1 (default off)
  MERGE_OFF=1 -> every switch takes its V29 value (bit-identical V29 predictions).  All pieces are gated on the suite shape (1024, 16); every other shape runs V29's code path bit for bit.


K3-simple factored cumulant propagation + MEMORYLESS KAPPA4 REGENERATION (V17)
+ D21 FEEDBACK THIN LEGS (V18, 2026-09-02, F69)
+ FINAL-LAYER TRIM (V19, 2026-09-03)
+ RESIDUAL-TIME FUSION (V20, 2026-09-03)
+ AGE-GATED SHARED BASIS FOR OLD SOURCES (V21, 2026-09-03, F66 option C / F72)
+ PRE-ACTIVATION LAMBDA TABLE (V22, 2026-09-03, F68 prefit)
+ NESTED SECOND TIER FOR THE OLDEST SOURCES (V24, 2026-09-03, F72 tiers)
+ ADAPTIVE PER-MLP LAMBDA (V25, 2026-09-03)
+ POOLED BATCHED STRASSEN-WINOGRAD PRODUCTS (V26-V28, 2026-09-05, F79/F80)
+ NEWBORN + COVARIANCE RIDING THE TRANSPORT FAMILY, BLOCK-SYMMETRIC C_PRE (V29, 2026-09-06)
+ LOT KAPPA4 SHAPE FOR THE MEAN PROGRAM (portlot, 2026-10-02).

portlot: the per-neuron fourth cumulant seen by the (1,) mean program keeps V29's per-layer LEVEL
(mean of g4row) but takes its neuron-to-neuron SHAPE from the leading-order closure of the exp_theory
agent: cand = proj + same + star built from V29's own dense legs (A, P of every live source, old
sources included: they are formed densely each layer), the per-birth-layer column scalings
(w1 var, var, w3/3) and a rank-32 basis of the current pre-activation covariance (one-pass range
finder; at the trimmed last layer through W C, never a dense product).  Formulas: see the PORTLOT
block below _lot_basis / _lot_g4.  PORTLOT=0 reproduces V29 bit-identically (same op stream).

V29: the dense young legs are pre-scaled by the wick w1 at the wick stage, so the
transport family multiplies by the raw W and the newborn's A leg (born as w1 * C_off) and
the post-ReLU covariance C ride along as two extra slots; C_pre = (W C) W^T is assembled
from the three block products of its 2x2 partition (one Strassen family), layer 0 uses the
aliased Gram w32^T w32, and the trimmed last layer reads var from (W C) * W.

V25: the regenerated kappa4 off-diagonal coefficient is no longer a frozen per-layer
table but lam_l = LAM[l] * ((mean(dG)/mean(var)) / REF_R[l])^BETA, evaluated online at the
layer from the transported kappa4 diagonal dG and the pre-activation variance (both already
computed there). Fitted offline on public dumps 0-7: the per-MLP LS lambdas track this
ratio with log-log slope ~1 at every layer and the unexplained spread drops from 3-10% to
1-2%. Kill-switch: V25_BETA=0 -> V24 values.

V24: sources older than AGE_OLD2 transports are re-confined to a rank-R_OLD2 sub-basis U
(r1 x r2) INSIDE the shared tier-1 basis Qc:  A_s = Qc U FA2_s.  Legs are formed from
QU = Qc U (r2/n units per leg), the dslice contraction of tier-2 sources is lifted by U^T
into the tier-1 inner product (one trailing Qc^T per layer, unchanged), U rotates with the
tier-1 factors at each rebuild, and the oldest tier-1 member moves into tier 2 in factor
space right after each rebuild (range finder on the r1 x r1 core, no n-space work). Lean
reference (nested tiers 4:384 + 7:256, dumps 0/1): 2.155e-8 vs 2.164e-8 = free.
Kill-switch: V24_AGE_OLD2=0 -> V22 op stream.

V21: sources older than AGE_OLD transports are confined to one shared rank-R_OLD basis
Q (n x r): A_s = Q FA_s, P_s = Q FP_s with static factors. Per old source-layer the
dense legs are formed from the factors (2 n^2 r) and the two dslice contractions go
through the factors (2 n^2 r) instead of 4 dense n^3 units. One source joins the old
group per layer; at a join the basis is rebuilt by a one-pass randomized range finder on
the weighted leg Gram of old + joiner (sketch = a slice of the layer weight), the old
factors are rotated into the new basis and the Gram core S (r x r) is updated. The lean
reference (g_regen_probe.py "...+agerank:3:384") measures +8.4% raw on dumps 0/1 for a
~25% cost cut. Kill-switch: V21_NO_CONFINE=1 -> exact V20 op stream.

V20: exact, bit-identical values; only the client-side op stream changes. The nonlin
term products are written with out= into one persistent (T, n, n) buffer instead of
~30 fresh (n,n) multiplies + a ~50-matrix fnp.stack per layer (measured 4 ms + 3 ms of
residual per layer), and the range-finder sketches are one contiguous copy of the
weight slice per layer instead of strided views (strided qr/matmul input costs ~0.8 ms
each). Kill-switch: none needed (values identical); V19 stays the reference.

V19: exact cost cut, zero accuracy change. The final layer emits only the mean, whose
(1,) nonlin terms need var = diag(C_pre), D3 and the K4 diagonal -- never C_off, the
(2,1) dslice D21, nor the (1,1)/(2,1)/(2,2) term program. V18 computed all of them at
the last layer and discarded them: the two dense dslice contractions (2k n^3 units,
k = 15 live sources) plus the full C_pre sandwich (2 units, only its diagonal is used).
V19 transports only diag(C_pre) = colsum(w32 * (C @ w32)) (1 unit) and runs the source
machinery in D3-only mode at the last layer (transports + n^2 contractions, no D21).
Expected C: 0.509xB -> ~0.48xB. Kill-switch: V19_FULL_LAST=1 -> exact V18 op stream.
Below this line the V18 docstring follows unchanged.


V18: the V1.6 "K3-dslice feedback into the new-block factors" (dropped since V1.6 as
DS-shadowed) is worth -10% raw on the regen base (F69) and is rank <= 16 in D21.
Birth factors of the B1 hub pair become
  X1 = 3 a_b + Xt,   Xt = 1.5 d(w2) D21            ~ F1 R1,  F1 = d(w2) Q,   R1 = 1.5 Bm
  Y1 = a_b d(w2) + Yt, Yt = 0.5 d(w1) D21^T d(w3)  ~ F2 R2,  F2 = d(w1) Bm^T, R2 = 0.5 Q^T d(w3)
with D21 ~ Q Bm from a rank-R_FB randomized range finder (same recipe as Rres). The
identity-born thin legs F1/F2 travel in their own transported stack Zf (like Z, but never
entering the M-leg einsums); in the dslice contraction the new terms either add to the fused left factors
(LA += Xt*P*w2/3 + P*Yt; LP += A*Yt + Xt*A*w2/3 + Xt*Yt/3; D3 += rowsum(P*3*LPadd)) or
contract against the thin right factors ((AP + Xt*P/3) R2^T F2^T and
(AP*w2 + P*Yt)/3 R1^T F1^T). Extra n^3: ZERO; extra ~16 n^2 r per source-layer.
Lean reference: g_regen_probe.py "regen:c+no31+feedc+usec+fitfull+rres:16+sb2+nowk22
+d21r:16": 2.195e-8 / 2.091e-8 on P2 dumps 0/1 (V17 port: 2.51 / 2.25e-8).
Kill-switch: V18_NO_FB=1 -> exact V17 behaviour (zero thin legs).
Below this line the V17 docstring follows unchanged.


V17 (2026-09-02, F68): the augmented-K3 kappa4 channel (F51/F64) without any
per-source kappa4 memory. The kappa4 matrix core G of the aug chain is, to
R^2 ~0.9-0.97, diag(g) + lambda_l * C_off with ONE frozen scalar per layer
(LAM table, fitted offline on public MLPs). Everything the channel needs then
rides on objects V16b already carries:
  - transported diagonal dG_pre = (W*W) g + lambda (diag(C_pre) - (W*W) var)  [n^2]
  - use side: wk4_row = m dG, wk4_mat = (m/6)(dG_i + dG_j) (== V16b's vec+ww form),
    plus the (3,1) slice wk431 = (m/2) lambda C_pre_off feeding 4 extra n^2 terms
  - K4->K3 feed: hub pair X3 = d(w1) G_pre d(w1) = lambda A d(w1) + P d(w1^2 dG)
    (column scalings of the A/P legs), Y3 = (m/4) w2 1^T (rank-1, folds into the
    P-group + one rank-1 outer product), M_t1 = (m/4)(w2*dG)(w1^2)^T (rank-1 M-type
    -> one extra thin Z/L column).  Extra n^3 cost: ZERO.
Lean reference: scratch/lean_k3_aug.py regen chain (g_regen_probe.py
"regen:c+no31+feedc+usec"): 2.196e-8 / 2.161e-8 on P2 dumps 0/1 vs 4.1e-8 for V16b.
Below this line the V16b docstring follows unchanged.


V16b (2026-09-01, F63): exact cost restructuring of the source machinery.
  - The M leg (B3 = Sym(M x P x P)) is never transported: M_b = diag(S3c)
    + 3*S21^T and S21 = S_sep + Rres with S_sep = diag(e) C_off diag(w1) the
    exact leading (2,1) Wick term (a column-scaled copy of a_b) and Rres a
    D21-driven residual that is numerically low-rank (rank-64 keeps final MSE
    at parity, F63). Hence M = P diag(s) + 3 A diag(e) + 3 Z L^T with Z = P Rr
    a thin (n,r) transported leg and L static.
  - V27 (F80): every fresh (n,n)/(k,n,r) result of the layer body goes out= into a
    persistent named buffer (_Pool); thin stacks and shared-basis factors live in
    ping-pong slot buffers (no concatenates); slow einsum forms -> matmul(out=).
  - All dslice contractions collapse into TWO fused einsums per layer (right
    factors A and P) plus O(n^2 r) thin terms: per source-layer 2 dense
    transports + 2 dense contractions (was 3 + 4).
  - Rres range finder: randomized (Omega = a slice of the layer weight), one
    power iteration, flopscope-billed QR; ~4 n^2 r FLOPs per birth.

Port of the reference kprop k_max=3 SIMPLE factored algorithm (companion paper
arXiv:2605.05179) with cost engineering measured in F42-F45, plus two
accuracy riders measured in F47/F49 (combined 1.44x lower final MSE):
  - vec+ww (augmented-lite v0): the radial K4 channel keeps a per-neuron
    diagonal-content vector instead of a single scalar; born from K(4,) and
    the K(2,2) column sums, transported one layer by the true (W*W) row
    action in place of one average-metric cup. ~n^2/layer, cost-neutral.
  - online mean correction: per-layer ridge-fitted delta = feats @ beta
    (13 per-neuron features already computed in the loop; beta fitted
    offline on the public teacher-forced trajectory, embedded below).
    Applied inside the loop so it propagates.
Base cost engineering:
  - shared-P hub blocks per source layer: B1 = Sym(X1 x P x Y1) with X1 = 3*abar,
    B3 = Sym(M x P x P); identity-born legs all evolve by the same map
    leg <- W @ (w1 * leg), so one P per source serves every sub-block, and the
    wick scaling is folded into the next layer's weight (WD).
  - V1.6 ablations (each measured to cost <~1% final MSE): the B2 hub sub-block
    and the K3-dslice feedback into new-block factors are DS-shadowed at birth
    (their diagonal-slice content migrates into the exact M chain automatically);
    the WK22 struct term is dropped from Y1; nonlin terms pruned at relative
    contribution >= 1e-5 (62 of 90 kept).
  - residual-time engineering: per-source matrices held as batched (k,n,n)
    stacks; all wick vectors built at once in the unified form
    w(k,p) = const*sigma^e*(P1(a)phi + P2(a)Phi); nonlin terms evaluated as one
    stacked einsum per arity; dslice hub contractions fused into 3/4-operand
    einsums with no (k,n,n) intermediates; final layer computes only the mean.

Cost: ~1.83e12 flopscope FLOPs per MLP = 0.833x the 2^41 budget (deterministic,
data-independent). Local mini-dump accuracy: mse_final ~3.2e-8 test-set mean
(V1.6 base alone: ~6.2e-8; exact K3-simple reference: 4.8e-8 at an over-budget
2.24x; cov-prop baseline: 4.4e-6).
Prediction: per-layer post-ReLU means, shape (depth, width). No randomness.
"""
from __future__ import annotations
import math
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP
HUB_LOG = []
WICK_PAIRS = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2), (6, 1), (6, 2), (7, 1)]
WICK_UNIFIED = [(1.0, 1, [1.0], [0.0, 1.0]), (1.0, 2, [0.0, 1.0], [1.0, 0.0, 1.0]), (1.0, 3, [2.0, 0.0, 1.0], [0.0, 3.0, 0.0, 1.0]), (1.0, 4, [0.0, 5.0, 0.0, 1.0], [3.0, 0.0, 6.0, 0.0, 1.0]), (1.0, 0, [], [1.0]), (1.0, 1, [2.0], [0.0, 2.0]), (1.0, -1, [1.0], []), (2.0, 0, [], [1.0]), (1.0, -2, [-0.0, -1.0], []), (2.0, -1, [1.0], []), (6.0, 0, [], [1.0]), (1.0, 1, [24.0], [0.0, 24.0]), (1.0, -3, [-1.0, 0.0, 1.0], []), (2.0, -2, [-0.0, -1.0], []), (6.0, -1, [1.0], []), (24.0, 0, [], [1.0]), (1.0, -4, [-0.0, 3.0, -0.0, -1.0], []), (2.0, -3, [-1.0, 0.0, 1.0], []), (1.0, -5, [3.0, 0.0, -6.0, 0.0, 1.0], []), (2.0, -4, [-0.0, 3.0, -0.0, -1.0], []), (1.0, -6, [-0.0, -15.0, -0.0, 10.0, -0.0, -1.0], [])]
TERM_SPECS = {(1,): [('ones1', 'ones1', 0, None, 1.0), ('d3', 'ones1', 8, None, 0.16666666666666666), ('g4', 'ones1', 12, None, 0.041666666666666664), ('d3', 'd3', 18, None, 0.013888888888888888), ('d3', 'g4', 20, None, 0.006944444444444444)], (1, 1): [('c_off', 'ones2', 4, 4, 1.0), ('d21T', 'ones2', 4, 6, 1.0), ('c_off', 'c_off', 6, 6, 0.5), ('wk4m', 'ones2', 6, 6, 0.25), ('d3row', 'c_off', 4, 12, 0.3333333333333333), ('c_off', 'd21T', 6, 8, 1.0), ('d3row', 'd21T', 4, 16, 0.16666666666666666), ('g4row', 'c_off', 4, 16, 0.08333333333333333), ('d3row', 'd21', 6, 12, 0.16666666666666666), ('d21T', 'd21T', 6, 12, 0.25), ('c_off', 'wk4m', 8, 8, 0.25), ('d21T', 'd21', 8, 8, 0.25), ('g4row', 'd21T', 4, 18, 0.041666666666666664), ('d3row', 'wk4m', 6, 16, 0.08333333333333333), ('g4row', 'd21', 6, 16, 0.041666666666666664), ('d21T', 'wk4m', 8, 12, 0.25), ('g4row', 'wk4m', 6, 18, 0.020833333333333332)], (2,): [('ones1', 'ones1', 1, None, 1.0), ('d3', 'ones1', 9, None, 0.16666666666666666), ('g4', 'ones1', 13, None, 0.041666666666666664)], (2, 1): [('c_off', 'ones2', 5, 4, 1.0), ('d21T', 'ones2', 5, 6, 0.5), ('d21', 'ones2', 7, 4, 0.5), ('c_off', 'c_off', 7, 6, 0.5), ('wk4m', 'ones2', 7, 6, 0.25), ('d3row', 'c_off', 5, 12, 0.16666666666666666), ('c_off', 'd21T', 7, 8, 0.5), ('c_off', 'd21', 9, 6, 0.5), ('c_off', 'd3col', 13, 4, 0.16666666666666666), ('d3row', 'd21T', 5, 16, 0.08333333333333333), ('g4row', 'c_off', 5, 16, 0.041666666666666664), ('d3row', 'd21', 7, 12, 0.08333333333333333), ('d21T', 'd21T', 7, 12, 0.125), ('c_off', 'wk4m', 9, 8, 0.25), ('d21T', 'd21', 9, 8, 0.25), ('c_off', 'g4col', 17, 4, 0.041666666666666664), ('d21', 'd3col', 17, 4, 0.08333333333333333), ('g4row', 'd21T', 5, 18, 0.020833333333333332), ('g4row', 'd21', 7, 16, 0.020833333333333332), ('d21', 'g4col', 19, 4, 0.020833333333333332)], (2, 2): [('c_off', 'ones2', 5, 5, 1.0), ('d21T', 'ones2', 5, 7, 1.0), ('c_off', 'c_off', 7, 7, 0.5), ('wk4m', 'ones2', 7, 7, 0.25), ('d3row', 'c_off', 5, 13, 0.3333333333333333), ('c_off', 'd21T', 7, 9, 1.0), ('d3row', 'd21T', 5, 17, 0.16666666666666666), ('g4row', 'c_off', 5, 17, 0.08333333333333333), ('d21T', 'd21T', 7, 13, 0.25), ('c_off', 'wk4m', 9, 9, 0.25), ('g4row', 'd21T', 5, 19, 0.041666666666666664)], (3,): [('ones1', 'ones1', 2, None, 1.0), ('d3', 'ones1', 10, None, 0.16666666666666666), ('g4', 'ones1', 14, None, 0.041666666666666664)], (4,): [('ones1', 'ones1', 3, None, 1.0), ('d3', 'ones1', 11, None, 0.16666666666666666), ('g4', 'ones1', 15, None, 0.041666666666666664)]}
PK2K_TABLE = {(1,): [(((1,),), 1.0)], (1, 1): [(((1, 1),), 1.0)], (2,): [(((1,), (1,)), -1.0), (((2,),), 1.0)], (2, 1): [(((1, 0), (1, 1)), -2.0), (((2, 1),), 1.0)], (2, 2): [(((0, 1), (1, 0), (1, 1)), 4.0), (((0, 1), (2, 1)), -2.0), (((1, 0), (1, 2)), -2.0), (((1, 1), (1, 1)), -2.0), (((2, 2),), 1.0)], (3,): [(((1,), (1,), (1,)), 2.0), (((1,), (2,)), -3.0), (((3,),), 1.0)], (4,): [(((1,), (1,), (1,), (1,)), -6.0), (((1,), (1,), (2,)), 12.0), (((1,), (3,)), -4.0), (((2,), (2,)), -3.0), (((4,),), 1.0)]}
CORR_BETA = [[0.0006411968414813632, 0.0, 0.00032431282848653057, -0.0003042144758593203, 0.0, 0.0, 0.0, 0.0, -0.0007625526069403323, 0.0, 0.0, 0.0, -0.0007625526055534857], [-0.0004483531031709554, -8.08970612775365e-08, 0.00024032689914639557, 0.0003234947612489975, 7.734051179371133e-06, -8.401877300643312e-06, 0.0026976072084027533, -2.9522256040202606e-06, -1.1604426266687203e-05, 9.457618761612473e-06, -3.3186036376400295e-05, -0.008465131129601192, -0.002371947086869632], [-0.00030645743691215543, 2.660655267636766e-06, 0.0006192175275623578, -0.0002468827461681828, -2.9900896046038434e-06, 6.841942903175705e-08, 0.0033128783967715193, 5.123025223409135e-06, -1.831086058750464e-06, -8.948552609048747e-06, -3.292520087183994e-05, 0.024826431355466128, -0.003290775627340336], [-0.00010464301263958924, 1.0735707741330386e-06, 0.0009154900192795534, -0.0006467314093895341, 3.6870994916648376e-06, 6.046342808060834e-06, 0.00341846204288833, -8.975061063301133e-06, -8.178979563111877e-06, 9.51453794000349e-05, -3.005588379050443e-05, 0.031163100958057806, -0.003816642326501922], [-0.0006681684602995246, -7.2789662985629386e-06, 2.591925967081249e-05, 0.0007797247634619618, 8.477664175402185e-06, 1.7025270836169033e-06, 0.003419282618548069, 7.535308544950828e-07, -5.733309150056555e-06, -2.9160257628201262e-05, -1.722583323732972e-05, 0.04076990884979579, -0.004248021529544134], [-0.0006734457317912116, 3.1290157938005224e-05, 0.00022889868689835653, 0.0007465462593369972, 6.851356914590185e-07, 2.1755554980020755e-05, 0.004058574189699476, -4.277653393515956e-06, -6.039384411257449e-05, -9.02015920342041e-05, -1.9819140691513135e-05, 0.041867576221848744, -0.005466361831539052], [-0.0006750844151648486, 6.180105939061905e-05, 0.0004389649313793868, 0.0006339652749835816, -6.694983507237225e-07, 4.690187390705702e-05, 0.003977857680309406, 4.383031188174939e-06, -0.00012887677523401884, 0.0002505563048388552, -2.5566781991513985e-05, 0.07694913516123503, -0.0057040529266279925], [-0.000972879718186106, 5.1156558236482596e-05, -0.0004913239875599442, 0.0017893987001835238, 1.90937415308986e-06, 3.802173926700858e-05, 0.0038578329372384434, 2.8522169318710464e-06, -0.00010738838919747425, -0.00015415811216305824, -2.5539604454967108e-05, 0.08311668913952446, -0.005918296132102073], [-0.0007850991486069435, 7.899688664131035e-05, -0.00022144314048615243, 0.001495955739849875, 1.8359978888302925e-07, 5.023393155879106e-05, 0.0034762600760059295, 3.215904175585605e-06, -0.00015314900768603073, -0.0004475649473906088, -3.028530737478287e-05, 0.06152086703424944, -0.00569825101925022], [-0.0005795122172445841, 0.00011465424961793733, 0.00039693472854898914, 0.0007970833957439235, 6.825630979361196e-06, 7.966232390898656e-05, 0.0032776762940374586, -1.387178929061937e-06, -0.0002567385150822406, 0.00028972790478315425, -2.7005665153789784e-05, 0.07590370226079919, -0.005546910408445549], [-0.0004949253254431177, 9.892837012135289e-05, 0.0005826341044064549, 0.0006128630824666212, 1.8258883411132417e-06, 6.024891049057027e-05, 0.003354817421260185, -6.690225809034046e-06, -0.00020554230085297872, 0.0002695214755196366, -2.5128600253275433e-05, 0.08226262500093764, -0.006092394095370619], [-0.0007584153570441029, 0.0001792621247156503, 0.0001931413288044634, 0.0014013628928244146, 5.054405581529621e-06, 0.00010157511263920278, 0.004187883074782575, -5.420295053799967e-06, -0.00037802160373858374, 0.00014440604770373433, -2.2764810929109396e-05, 0.07738752039037704, -0.008027695439553941], [-0.0008291520607940194, 0.0002115021197150293, -0.00012740149902796318, 0.0017326132691935436, -2.1624432383474912e-06, 0.0001087192480144056, 0.004533661385130805, 4.837349485145022e-07, -0.00041674041200178005, 0.00018467937018199515, -1.9126957306380207e-05, 0.07896437764069765, -0.008986219483970914], [-0.000658293372173689, 0.00018380103458278174, 0.00046374648579651084, 0.001216035726510531, 4.012828172272205e-06, 9.570320459811334e-05, 0.004325867834172417, -1.2894882645758776e-05, -0.00038451298928645256, 0.0005113367132840055, -2.241500653445156e-05, 0.07104629639605271, -0.00897436256704376], [-0.0009527557374376014, 0.00019049799075311402, -0.0009213478791293237, 0.0025553941675787975, 3.508060734949607e-07, 9.248186905581894e-05, 0.004641633979578725, -3.1682752945685096e-06, -0.00038201626600816054, 3.0155769164602614e-05, -2.3334690641924517e-05, 0.07182226701958777, -0.01001369215216409], [-0.0006908532933348147, 0.00018408588405074964, 0.0004249876754929348, 0.0013835702671030146, 5.170891351788878e-06, 9.144035619901057e-05, 0.0044972055980902715, -1.0115381018603137e-05, -0.00038927641644205714, 0.0002930281376816756, -2.8026123982155874e-05, 0.157561818094021, -0.010070547059124272]]
LAM = [0.0049895, 0.0080876, 0.0098291, 0.010549, 0.010851, 0.010828, 0.010589, 0.010048, 0.0096483, 0.009172, 0.008773, 0.0083938, 0.0080555, 0.007677, 0.0072588, 0.0072588]
METRIC_C = float(__import__('os').environ.get('METRIC_C', '2.0'))   # h18f_patch: env knob, default = parent constant
# K31TWO: two-term (3,1) closure wk431_ij = K(j^3,i) = b1_l v_j C_ij + kappa3_j D21[j,i] / v_j (kai_leal_gonzalez, github.com/al07-alt/whest-p2-k31-two-term,
# MIT, on 504aldo's V29): 0 = off, 1 = at the layers where the response-31 path is not active, 2 = everywhere (replaces response-31 too)
K31TWO = int(__import__('os').environ.get('K31TWO', '2'))
K31TWO_B1 = [0.00974, 0.01424, 0.01752, 0.02066, 0.02259, 0.0244, 0.02539, 0.02668, 0.02694, 0.02645, 0.02603, 0.02589, 0.02442, 0.02452]
K31TWO_B2 = float(__import__('os').environ.get('K31TWO_B2', '1.0'))
K31TWO_B1S = float(__import__('os').environ.get('K31TWO_B1S', '1.0'))
H18 = __import__('os').environ.get('H18', '1') == '1'          # per-row metric sum_i W_ai^2 instead of the constant METRIC_C in the kappa4 row estimate (idea from Yinita/whest-l31-side, team Niko, MIT)
H18_K22 = __import__('os').environ.get('H18_K22', '0') == '1'  # same per-row metric in the K22 pair model
H18_K31 = __import__('os').environ.get('H18_K31', '0') == '1'  # sqrt(m_a m_b) scaling of the (3,1) pair model
# h18f_patch (agents_ideas): metric consistency under the active set; every knob '0' = the parent op stream
H18F = __import__('os').environ.get('H18F', '0') == '1'            # g4row metric = live-row sum + dropped-input sum (full row metric)
H18F_FEED = __import__('os').environ.get('H18F_FEED', '0') == '1'  # per-row metric in the kappa4 -> kappa3 feed (y_b)
H18F_K22X = __import__('os').environ.get('H18F_K22X', '0') == '1'  # K22 pair model (dG_a m_b + dG_b m_a) / 6 (Sym(G x M) cross pairing)
import os as _os
import gc as _gc
import contextlib as _contextlib
NO_FEED = _os.environ.get('V17_NO_FEED', '0') == '1'
NO_WK431 = _os.environ.get('V17_NO_WK431', '0') == '1'
NO_REGEN = _os.environ.get('V17_NO_REGEN', '0') == '1'
NO_FB = _os.environ.get('V18_NO_FB', '0') == '1'
MERGE_OFF = _os.environ.get('MERGE_OFF', '0') == '1'
_ON = '0' if MERGE_OFF else '1'
TD_OFF = _os.environ.get('TD_OFF', '1' if MERGE_OFF else '0') == '1'
TD_FB_EXACT0 = _os.environ.get('TD_FB_EXACT0', '0' if TD_OFF else '1') == '1'
TD_FB_B0 = int(_os.environ.get('TD_FB_B0', '-1' if TD_OFF else '3'))
TD_KZ = max(1 if TD_FB_EXACT0 else 0, TD_FB_B0 + 1)
TD_GATE_B = int(_os.environ.get('TD_GATE_B', '-1'))
TD_GATE2_B = int(_os.environ.get('TD_GATE2_B', '-1'))

def _nest_cnt(li, age_old2):
    """number of sources whose tier-2 move is due at ledger layer li (V29: li - age_old2; shifted by one layer for births <= TD_GATE2_B)."""
    if TD_GATE2_B < 0:
        return li - age_old2
    return sum((1 for b_ in range(16) if b_ + age_old2 + 1 - (1 if b_ <= TD_GATE2_B else 0) <= li))
TD_FROZEN = frozenset((int(x) for x in _os.environ.get('TD_FROZEN', '' if TD_OFF else '6,8,10,12,14').split(',') if x.strip()))
# FIJA2 (port of LordProcciuto's X_FIJA, MIT, (c) 504aldo + LordProcciuto): every join after the first is frozen (fixed basis, only
# transported), the first union orders the basis once (top-r2 subspace of the first joiner's Gram first) and tier 2 is the
# least-squares projection (metric of the join's basis) onto its first r2 columns; no Gram cores.  0 = the parent op stream.
FIJA2 = _os.environ.get('FIJA2', '1') == '1'
FIJA2_PASS = int(_os.environ.get('FIJA2_PASS', '1'))
# FIJA2X (exact follow-up savings of the fixed basis; ideas from LordProcciuto e28/e29, MIT): every knob 0 = the parent op stream.
FX_UEYE = _os.environ.get('FX_UEYE', '1') == '1'      # QU / inner2 U^T / U2^T bqc / LOT QU^T Q as copies or slices (needs FIJA2)
FX_TRI = int(_os.environ.get('FX_TRI', '3'))           # frozen-join Rq^-1 Qn^T by FX_TRI levels of block bisection
FX_FJW = _os.environ.get('FX_FJW', '1') == '1'        # frozen-join projection on the joiner's hub width
FX_NESTW = _os.environ.get('FX_NESTW', '1') == '1'    # FIJA2 tier-2 step on the mover's hub width
FX_NESTS = _os.environ.get('FX_NESTS', '1') == '1'    # (with FX_NESTW) Kp F[r2:] through the Strassen core
FX_SGDEAD = _os.environ.get('FX_SGDEAD', '1') == '1'  # no Sg rotation after the first-union ordering (needs FIJA2)
FULL_LAST = _os.environ.get('V19_FULL_LAST', '0') == '1'
NO_CONFINE = _os.environ.get('V21_NO_CONFINE', '0') == '1'
QPASS = int(_os.environ.get('V21_QPASS', '1'))
NO_SRC_LAST = _os.environ.get('V19_NO_SRC_LAST', '0') == '1'
NO_CORR = _os.environ.get('V17_NO_CORR', '1') == '1'
LAM = [c * float(_os.environ.get('V17_LAM_SCALE', '0.95')) for c in LAM]
REF_R = [0.00658815, 0.00818414, 0.00853136, 0.00838859, 0.00810153, 0.00774287, 0.00735951, 0.00691093, 0.00655752, 0.00617892, 0.00583953, 0.00556824, 0.00532459, 0.00505165, 0.00477181]
BETA = float(_os.environ.get('V25_BETA', '1.0'))
DEBUG = []
PL_ON = _os.environ.get('PORTLOT', _ON) == '1'
PL_K = int(_os.environ.get('PORTLOT_K', '12'))
PL_Q = int(_os.environ.get('PORTLOT_Q', '1'))
PL_LMIN = int(_os.environ.get('PORTLOT_LMIN', '1'))
PL_STAR = _os.environ.get('PORTLOT_STAR', '1') == '1'
PL_SAME = _os.environ.get('PORTLOT_SAME', '1') == '1'
PL_RMIN = float(_os.environ.get('PORTLOT_RMIN', '0.35'))
PL_RMAX = float(_os.environ.get('PORTLOT_RMAX', '1.5'))
PL_DEBUG = _os.environ.get('PORTLOT_DEBUG', '0') == '1'
PL_NS = _os.environ.get('PORTLOT_NS', '0') == '1'
HZ_ON = _os.environ.get('HZ_ON', '1') == '1'
SD_C = float(_os.environ.get('SD_C', '0'))        # emulation: neurons with alpha < -SD_C carry no post-activation K3 content and no hub content
SD_COL = float(_os.environ.get('SD_COL', '0'))    # emulation: D21 columns of neurons with alpha < -SD_COL are zero
HUB_CN = float(_os.environ.get('HUB_CN', '0'))    # emulation: hubs with alpha < -HUB_CN removed at birth
HUB_CP = float(_os.environ.get('HUB_CP', '0'))    # emulation: hubs with alpha > HUB_CP removed at birth
HUB_N = int(_os.environ.get('HUB_N', '0'))        # emulation: keep only the HUB_N hubs nearest alpha = HUB_A0
HUB_A0 = float(_os.environ.get('HUB_A0', '0'))
HUB_L0 = int(_os.environ.get('HUB_L0', '0'))
HUB_ACN = float(_os.environ.get('HUB_ACN', '0'))  # emulation: A-leg hub columns with alpha < -HUB_ACN removed
HUB_ACP = float(_os.environ.get('HUB_ACP', '0'))  # emulation: A-leg hub columns with alpha > HUB_ACP removed
HUB_AN = int(_os.environ.get('HUB_AN', '0'))      # emulation: keep only the HUB_AN A-leg hub columns with the largest w2^2 var^2 (or nearest alpha=HUB_A0)
HUB_PART = int(_os.environ.get('HUB_PART', '255'))  # bitmask of masked components: 1 A cols, 2 fb factors, 4 feed vectors, 8 e/s weights, 16 thin residual rows, 32 P cols, 64 w2 weight
HUB_KEY = _os.environ.get('HUB_KEY', 'alpha')
SD_ROW = float(_os.environ.get('SD_ROW', '0'))    # emulation: pre-activation D3 and D21 rows of neurons with alpha < -SD_ROW are zero (not at the last layer)
SD_ON_C = float(_os.environ.get('SD_ON_C', '0'))  # emulation: hub content of always-on neurons (alpha > SD_ON_C) dropped
RESP_SCALE = float(_os.environ.get('RESP_SCALE', '1.0'))
# YR (sdt2): per-source ("ragged") hub windows.  The young transport and the young D21 readout are each ONE batched Strassen family
# over the young sources' slots, padded to the widest source of the window (SDT_HUBW widths fall with depth, the oldest young source
# is the widest).  YR splits the window (oldest first, contiguous slots) into <= 2 families, each on its own max width, when that saves
# at least YR_MINSAVE slot-columns (the newborn's covariance slot: columns beyond its group width come from the exact SDT_CSPLIT
# dense product, costed at YR_DW slot-columns per column).  Exact (the legs are zero beyond each source's width).
YR_T = _os.environ.get('YR_T', '1') == '1'
YR_D = _os.environ.get('YR_D', '1') == '1'
YR_DW = float(_os.environ.get('YR_DW', '2.2'))
YR_MINSAVE = int(_os.environ.get('YR_MINSAVE', '128'))
YR_DMINSAVE = int(_os.environ.get('YR_DMINSAVE', '256'))   # same threshold for the D21 readout split
YR_LOG = []
# LOTW (sdt4): the LOT right query and the KU contraction run on the hub window of each tier (young slots: max hub width of the young
# sources; tier 1: max hub width of its sources = SDT_OLDW's window; tier 2 keeps the full width).  Exact (legs / factors are zero beyond).
LOTW = _os.environ.get('LOTW', '1') == '1'


def _yr_split(ws, spw, ni=0, dw=0.0, minsave=0):
    """YR: contiguous split of the per-source widths ws (oldest first, spw slots per source) into <= 2 groups [(i0, i1, width)];
    ni > 0: the last source is the newborn, whose covariance slot needs ni columns (dense beyond its group width, dw per column)."""
    S = len(ws)

    def cost(groups):
        return sum((spw * (i1 - i0) * w for i0, i1, w in groups)) + (dw * max(0, ni - groups[-1][2]) if ni > 0 else 0.0)
    one = [(0, S, max(ws))]
    best, bc = one, cost(one)
    for p_ in range(1, S):
        g_ = [(0, p_, max(ws[:p_])), (p_, S, max(ws[p_:]))]
        c_ = cost(g_)
        if c_ < bc:
            best, bc = g_, c_
    return best if cost(one) - bc >= minsave else one


# OLD_ROWW (sdt2): the tier-2 (oldest) sources' legs are rebuilt AFTER the exact variance of the layer is known, on rows [p:nr] only;
# p = the leading run of always-on rows (exact alpha > OLD_ROW_CP; the live order is by descending mean, so they lead) floored to
# OLD_ROW_GRAN.  Rows [0:p] of those legs are zero (their tier-2 D3 / D21 / LOT parts dropped, as in respk's OLD_ROW_CP emulation).
OLD_ROWW = _os.environ.get('OLD_ROWW', '1') == '1'
OLD_ROW_CP = float(_os.environ.get('OLD_ROW_CP', '3.0'))
OLD_ROW_GRAN = int(_os.environ.get('OLD_ROW_GRAN', '64'))
# OLD_HOT1 (port of LordProcciuto's X_VB_HOT, MIT): the tier-1 old legs too are formed after the exact variance, on rows [p:nr] only
# (same p as OLD_ROWW), and the old-tier D21 lift runs on columns [p:nr] only.
OLD_HOT1 = int(_os.environ.get('OLD_HOT1', '1'))   # bit 1: deferred tier-1 legs, bit 2: D21 old lift columns [p:nr]
OLD_ROW_LOG = []
# agents_audit AU_* (2026-10-16): exact cuts in the D3/D21 slices and the old tiers; every knob '0' = the parent op stream.
AU_APKZ = _os.environ.get('AU_APKZ', '1') == '1'     # DSF feedback block: A*P only on the feedback slots [kz:k] (bit-identical)
AU_LIFTW = _os.environ.get('AU_LIFTW', '1') == '1'   # D21 old-tier lift on the read columns [:ncw] only
AU_THINW = _os.environ.get('AU_THINW', '1') == '1'   # gated thin D21 term on the read columns [:ncw] only
AU_T1PK = _os.environ.get('AU_T1PK', '1') == '1'     # tier-1 rebuild + D21 tier-1 hub on a packed common width below the widest source
AU_T1PK_EXT = _os.environ.get('AU_T1PK_EXT', '1') == '1'   # (with AU_T1PK) allow the one-extra-slot layout
AU_ROWP = _os.environ.get('AU_ROWP', '1') == '1'     # leading always-on run computed once per layer (bit-identical)


AU_SYM4 = _os.environ.get('AU_SYM4', '1') == '1'   # audit4: C_pre on the 4 x 4 block partition (10 blocks a <= b, one family)
AU_SYM4_MN = int(_os.environ.get('AU_SYM4_MN', '7'))   # min block of that family's base level (LX lifts only from L >= LX_FROM)
AU_SYM4_K3 = _os.environ.get('AU_SYM4_K3', '1') == '1'   # (with AU_SYM4) the 3 x 3 partition when nr % 192 == 0, nr >= 768 (q = 320 / 256)
_AU_SYM_PAIRS = {k_: tuple((a_, b_) for a_ in range(k_) for b_ in range(a_, k_)) for k_ in (3, 4)}


def _au_t1plan(ws, ext_ok):
    """AU_T1PK: packing plan of the tier-1 slot pairs (ws = their hub widths in slot-pair order) or None (no saving).
    ('pair', wb, moves): S slots at wb = the 64-grid ceiling of the mean width; move (js, a, b, jt, c, d) = columns [a:b] of pair js to the
        free columns [c:d] of pair jt (w_jt <= c < d <= wb); the overflow fits because wb >= the mean.
    ('ext', wb, groups): S + 1 slots, all overflow in one extra slot in front; group (s0, s1, w, off) = slots [s0:s1] (all of width w > wb)
        put their columns [wb:w] at extra-slot columns [off : off + (s1 - s0) (w - wb)] (slot order)."""
    np_ = len(ws)
    if np_ < 2:
        return None
    S = 2 * np_
    wmax = int(max(ws))
    best = (S * wmax, None)
    wb0 = -(-sum(ws) // (np_ * 64)) * 64
    if wb0 < wmax and S * wb0 < best[0]:
        free = [[t, int(w), wb0] for t, w in enumerate(ws) if w < wb0]
        moves = []
        fi = 0
        for j, w in enumerate(ws):
            a = wb0
            while a < w:
                t, c, d = free[fi]
                e = min(int(w) - a, d - c)
                moves.append((j, a, a + e, t, c, c + e))
                a += e
                free[fi][1] = c + e
                if c + e >= d:
                    fi += 1
        best = (S * wb0, ('pair', wb0, tuple(moves)))
    if ext_ok:
        sw = [int(w) for w in ws for _ in (0, 1)]
        for wb1 in range(max(384, 64 * ((min(sw) // 2) // 64)), wmax, 64):
            if sum(max(0, w - wb1) for w in sw) <= wb1 and (S + 1) * wb1 < best[0]:
                groups = []
                off = 0
                s = 0
                while s < S:
                    if sw[s] <= wb1:
                        s += 1
                        continue
                    s1 = s
                    while s1 < S and sw[s1] == sw[s]:
                        s1 += 1
                    groups.append((s, s1, sw[s], off))
                    off += (s1 - s) * (sw[s] - wb1)
                    s = s1
                best = ((S + 1) * wb1, ('ext', wb1, tuple(groups)))
                break
    return best[1]


def _au_pair(arr4, j, cols):
    """AU_T1PK: view of slot pair j (slots 2j, 2j+1) of a (slots, 1, rows, n) stack, columns `cols`"""
    return _pview(arr4, (slice(2 * j, 2 * j + 2, None), slice(None, None, None), slice(None, None, None), cols))


def _au_ext(v2, off, c, e):
    """AU_T1PK: (c, rows, e) view of the columns [off : off + c e] of a 2-D (rows, n) view (block b = columns off + b e ..)"""
    return fnp.swapaxes(fnp.reshape(_pview(v2, (slice(None, None, None), slice(off, off + c * e, None))), (v2.shape[0], c, e)), 0, 1)


AU_REUSE = _os.environ.get('AU_REUSE', '1') == '1'   # audit3: per-layer sigma / alpha (and the AS varc sqrt) formed once (bit-identical)


def _au_extc(owner, v2, off, c, e, ax=False):
    """audit3: _au_ext (ax: with a singleton axis 1, the take out= form) created once per (parent view, off, c, e) and cached on `owner`;
    the cached entry keeps its parent and is only reused for the identical parent object"""
    vc = owner.__dict__.get('_au_vcache')
    if vc is None:
        vc = owner._au_vcache = {}
    key = (id(v2), int(off), int(c), int(e), bool(ax))
    hit = vc.get(key)
    if hit is not None and hit[0] is v2:
        return hit[1]
    v = _au_ext(v2, off, c, e)
    if ax:
        v = _pview(v, (slice(None, None, None), None))
    if len(vc) >= 4096:
        vc.clear()
    vc[key] = (v2, v)
    return v


def _au_slots(arr4, s0, s1, rows, cols):
    """AU_T1PK: (s1 - s0, rows, cols) view of slots [s0:s1] of a (slots, 1, R, n) stack"""
    return _pview(arr4, (slice(s0, s1, None), 0, rows, cols))
# SDT ("soft-dead tail", 2026-10-13): the live order of an active-set layer is descending in pre-activation mean with the exact-alpha
# candidates last (alpha descending), so the nearly-dead neurons (alpha < -SDT_C) are the LAST rows.  The rows [n3:] (n3 = the smallest
# SDT_GRAN-multiple window that leaves at most count(alpha < -SDT_C) rows out) carry no post-activation K3 content and no hub content
# (K2 untouched, their own pre-activation D3 row is still read): the response coefficients and every hub-indexed quantity of the source
# born at the layer are zero there; D21 columns [nc:] are zero (nc: the same rule with -SDT_COL).  Windows sit on SDT_GRAN = 64 because
# the Strassen engine runs one level deeper on 64-multiples (an odd 32-multiple window costs more than it saves).  The zero regions then
# become smaller products: SDT_KWIN (the young transport contracts over the first K = n3_prev rows; the covariance slot is repaired by
# one dense tail product), SDT_HUBW (hub width of a source = n3; SDT_CSPLIT: the covariance slot's columns beyond the batch width come
# from one dense product instead of widening the whole batch) and SDT_COLW (D21 demand Gram / dense lift / projections on nc columns).
# SDT_ON=0 -> the op stream of costm3.
SDT_ON = _os.environ.get('SDT_ON', '1') == '1'
SDT_C = float(_os.environ.get('SDT_C', '2.25'))
SDT_COL = float(_os.environ.get('SDT_COL', '2.0'))
SDT_GRAN = int(_os.environ.get('SDT_GRAN', '64'))
SDT_L0 = int(_os.environ.get('SDT_L0', '4'))        # first layer with the SDT order (no hard drop needed: layers below AS_L0 are only permuted)
SDT_ORDER = SDT_ON and _os.environ.get('SDT_ORDER', '1') == '1'
SDT_MASK = SDT_ORDER and _os.environ.get('SDT_MASK', '1') == '1'
SDT_KWIN = SDT_MASK and _os.environ.get('SDT_KWIN', '1') == '1'
SDT_HUBW = SDT_MASK and _os.environ.get('SDT_HUBW', '1') == '1'
SDT_CSPLIT = SDT_HUBW and _os.environ.get('SDT_CSPLIT', '1') == '1'
SDT_COLW = SDT_MASK and _os.environ.get('SDT_COLW', '1') == '1'
SDT_OLDW = SDT_HUBW and _os.environ.get('SDT_OLDW', '1') == '1'   # tier-1 old sources: rebuild / D21 core on their max hub width
SDT_JOINW = SDT_MASK and _os.environ.get('SDT_JOINW', '1') == '1'  # join / shared-basis products contract over the first K rows only
SDT_DEBUG = _os.environ.get('SDT_DEBUG', '0') == '1'
SDT_LOG = []
SDT_PROBE = frozenset((int(x) for x in _os.environ.get('SDT_PROBE', '').split(',') if x.strip()))   # debug: keep W C, C_pre, var and the orders of these layers
SDT_PROBES = {}

def _sdt_win(nr, cnt):
    """prefix window (<= nr) on the SDT_GRAN grid that leaves at most cnt trailing rows out (never more than a quarter of the rows)"""
    return min(nr, -(-(nr - min(cnt, nr // 4)) // SDT_GRAN) * SDT_GRAN)
HZ_K = tuple(int(x) for x in _os.environ.get('HZ_K', '0,0,0,0,0,192,96,64,256,96,128,192,160,128,256,160').split(','))
HZ_LMIN = 2
HZGATE = _os.environ.get('HZGATE', '1') == '1'     # engine agent: deterministic pullback prewarm; HZW windows only on prebuilt (j, c, w); no L6
LXPGATE = _os.environ.get('LXPGATE', '1') == '1'   # engine agent: LX_PAD only onto plans of the static setup plan list
HZG_LOG = _os.environ.get('HZG_LOG', '1') == '1'   # engine agent: Estimator.lazy_builds counter (read after a predict)
HZG_EXT = _os.environ.get('HZG_EXT', '0') == '1'   # engine agent: HZGATE list + one-step-smaller windows (~+750 setup round trips)
HZG_MAX_T = float(_os.environ.get('HZG_MAX_T', '1.5'))   # engine agent: safety cap (s) of the HZGATE setup list (normal: ~0.7 s remote)
HZW = _os.environ.get('HZW', '1') == '1'   # agents_joins: kappa4 pullback products contract over the SDT window (exact zeros beyond it)
HZ_LEV = int(_os.environ.get('HZ_LEV', '5'))
HZ_RCH = int(_os.environ.get('HZ_RCH', '1024'))
HZ_RIDERS = _os.environ.get('HZ_RIDERS', '1') == '1'
LOTD_ON = _os.environ.get('LOTD', '1') == '1'   # lotdiet_patch: fewer LOT / HZ flopscope calls, bit-identical results
_LOTD_SL8 = ('same', 'proj', 'star', 'D3', 'var', 'g4', 'mu', 'port')   # lotdiet A: row order of H['sl8'] = the kappa4 feature order
# HX6: extra all-row riders (theta entries 19..) from row sums / energies of the layer's pair matrices, O(n^2) per layer: q4_d21_w1, q4_d21_e, q4_d21_s, q4_d21_d3, q3_k31b_s
H6_SLABS = ['h6_d21_d3', 'h6_d21_e', 'h6_d21_s', 'h6_d21_w1', 'h6_k31b_s']
# HX: five extra all-row riders (theta entries 14-18): c2 same/var, c2 proj/var, c2 star/var, c4 D3^2/var, c4 same, from the LOT terms
# (response_same / proj_only / response_star) that _lot_g4 already computes for every live row; O(n) per layer.
HZ_SETUP_T = float(_os.environ.get('HZ_SETUP_T', '0.6'))
HZ_STRICT = _os.environ.get('HZ_STRICT', '0') == '1'
HZ_DEBUG = None
# per-layer weights: 0-6 kappa4 replacement on the selected rows (same, proj, star, kappa3^2/var, V29 kappa4, chain, induced), 7-13 riders
HZ_TH = [None, None, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.00905202, 0.00487477, -0.00069957, 0.00073425, -0.25726243, 0.00037372, 0.23026116, -0.05884123, 0.27981941, -0.16698114, 0.5533739, -7.67210137, -0.00386864, 0.4328553, 0.0018302, -0.10034998, 0.00112289], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01443391, 0.00823169, -0.00118099, 0.00112913, -0.4711891, 0.01270926, 0.21909459, -0.10481632, 0.14372522, -0.36590749, 0.97281148, -5.71688088, -0.00380471, 0.25231827, 0.00185309, -0.04309665, 0.00023147], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01392777, 0.00751257, -0.00152218, 0.00134599, -0.59377814, 0.00872483, 0.27361823, -0.13497637, 0.08200912, -0.4436455, 0.77420679, -4.80952852, -0.00379833, 0.21523726, 0.00169393, -0.03818433, 0.00049924], [0.41569528, 0.15063722, 1.47931299, -0.67029876, -0.01851705, 0.80530933, 0.785919, -0.01689179, 0.00738398, -0.00117239, 0.00086547, -0.47083258, 0.00405817, 0.35182877, -0.05024134, -0.01998981, -0.2838747, 0.98980373, -3.094343, -0.00207414, 0.09711906, 0.00085525, -0.01809469, 0.00051813], [0.57165656, 0.34837447, 1.60714129, -0.45874113, -0.06309901, 0.77012552, 0.89133378, -0.01698643, 0.00861084, -0.00173051, 0.00129259, -0.57878351, 0.00477169, 0.40951931, -0.06579839, 0.03527756, -0.31572977, 0.96042425, -3.45565981, -0.00310815, 0.10894156, 0.00115789, -0.01814838, 4.526e-05], [0.42250953, 0.40979353, 1.4392359, -0.67893876, -0.00700855, 0.76559236, 0.88745518, -0.01658709, 0.00793477, -0.00286755, 0.00236121, -0.58071801, 0.00555239, 0.22752816, -0.07752741, 0.01105282, -0.32962286, 0.88075078, -2.82475557, -0.00521164, 0.09315993, 0.00204725, -0.00688479, 0.00084588], [0.28682824, 0.23555851, 1.39591381, -0.59342225, 0.03165184, 0.87272312, 0.94799389, -0.01790269, 0.00723535, -0.00191367, 0.0012531, -0.40377187, -0.0024177, 0.18203695, -0.044844, -0.01271155, -0.30614567, 0.96334425, -0.99103011, -0.00206943, 0.03864672, 0.00050335, -0.00299232, 0.00052166], [0.37080648, 0.21820511, 1.49709239, -0.90504922, 0.04247778, 0.88282021, 0.99565376, -0.02063567, 0.01021866, -0.00277248, 0.00213476, -0.53159094, 0.0031165, 0.21167021, -0.05800173, -0.02980999, -0.30025148, 1.03357175, -1.81693927, -0.00315973, 0.05393, 0.00081403, -0.00491047, 0.00159346], [0.49304865, 0.22015067, 1.45582301, -0.6448468, -0.11707366, 0.84065849, 1.07023782, -0.02363803, 0.01240558, -0.00320616, 0.00252066, -0.48089456, 0.00704908, 0.32404185, -0.05470069, -0.03002672, -0.28645434, 1.12169829, -2.84121846, -0.00449581, 0.05101736, 0.00204699, -0.00526197, 0.00276831], [0.35789613, 0.20500675, 1.34189727, -0.5793427, -0.03465868, 0.8636712, 1.01182126, -0.02036751, 0.00947043, -0.00275769, 0.00196445, -0.40575995, -0.0010602, 0.2731161, -0.04247577, -0.02880291, -0.27100417, 1.02528468, -1.90629789, -0.00277329, 0.03075971, 0.00104021, -0.00354516, 0.00187004], [0.53769715, 0.27536013, 1.43608733, -0.56376751, -0.11170739, 0.79882725, 1.05303675, -0.02304202, 0.01174529, -0.00338962, 0.00240423, -0.46425458, 0.00543434, 0.32190647, 0.02634563, -0.05830974, -0.21715157, 1.07236407, -2.04304373, -0.00337514, 0.03514651, 0.00096317, -0.00315911, 0.00174786], [0.26255745, 0.28647117, 1.30542975, -0.51058316, 0.01213766, 0.84961709, 1.14087676, -0.01962442, 0.00855262, -0.00378117, 0.00301933, -0.38081596, 0.00044669, 0.26420349, -0.03255647, -0.02540046, -0.21234296, 0.84307704, -1.71540074, -0.00435982, 0.02931168, 0.00145142, -0.00127355, 0.00216362], [0.21452726, 0.03945284, 1.13045118, -0.73477604, 0.06963961, 0.86204976, 1.04503117, -0.02610852, 0.01463325, -0.00345033, 0.00244885, -0.28821995, 0.00486939, 0.23002764, 0.01138355, -0.08573504, -0.23607183, 1.13348181, -1.6837698, -0.00355142, 0.02157834, 0.00162539, -0.00211144, 0.00401429], [0.12222986, 0.16045812, 1.26116905, -0.63208041, -0.00682604, 0.90705852, 1.17552569, -0.01199335, 0.00986939, -0.00194475, 0.00122426, -0.3170032, -0.01541981, 0.21955871, -0.02384771, -0.02122266, -0.17813937, 1.21332253, -0.08857295, 0.0, 0.0, 0.0, 0.0, 0.0]]
K22_ON = _os.environ.get('LC_K22', '0') == '1'
K22_L0 = int(_os.environ.get('LC_K22_L0', '2'))
K22_DEBUG = _os.environ.get('LC_DEBUG', '0') == '1'
K22_LOG = []
D21_CUT = float(_os.environ.get('D21_CUT', '2.25'))
D21_RANK = int(_os.environ.get('D21_RANK', '2'))
D21_RANK_L = _os.environ.get('D21_RANK_L', '')   # d21rl_patch (experiment): per-layer D21 projection rank "a-b/r,c/r"
_D21_RANK_LM = {}
for _p in D21_RANK_L.replace(';', ',').split(','):
    if '/' in _p:
        _r, _t = _p.split('/'); _a, _, _b = _r.partition('-')
        for _l in range(int(_a), int(_b or _a) + 1):
            _D21_RANK_LM[_l] = int(_t)
D21_GRAN = int(_os.environ.get('D21_GRAN', '64'))
CDP_C = float(_os.environ.get('CDP_C', '1.625'))   # cdp_patch: young-hub D21 columns with alpha <= -CDP_C projected (0 = off)
CDP_ON = CDP_C > 0
CDP_R = int(_os.environ.get('CDP_R', '1'))
CDP_LMIN = int(_os.environ.get('CDP_LMIN', '4'))
CDP_LMAX = int(_os.environ.get('CDP_LMAX', '14'))
CDP_DEBUG = _os.environ.get('CDP_DEBUG', '0') == '1'
CDP_LOG = []
TDP_C = float(_os.environ.get('TDP_C', '1.25'))   # tdp_patch: previous-layer band alpha in (-SDT_C, -TDP_C] of the w1-scaled transports (0 = off)
TDP_ON = TDP_C > 0
TDP_R = int(_os.environ.get('TDP_R', '2'))       # rank of the young-transport band projection (0 = hard cut)
TDP_MASK = _os.environ.get('TDP_MASK', '')      # pre-screen emulation (FLOPs / plans unchanged): y young input, j join fit, q old basis
TDP_Y = _os.environ.get('TDP_Y', '0') == '1'    # real young transport: batch on kc rows + rank-TDP_R band term + widened C repair
TDP_JOIN = _os.environ.get('TDP_JOIN', '1') == '1'   # real frozen join: Qp / QR / pseudo-inverse / sketch on kc rows
TDP_QC = _os.environ.get('TDP_QC', '0') == '1'  # real old-basis transport on kc rows (hard cut of the band)
TDP_LMIN = int(_os.environ.get('TDP_LMIN', '5'))
TDP_LMAX = int(_os.environ.get('TDP_LMAX', '15'))
TDP_DEBUG = _os.environ.get('TDP_DEBUG', '0') == '1'
TDP_LOG = []


def _tdp_basis(xb, r, sk):
    """tdp_patch: per-slot orthonormal rank-r left basis of xb (S, b, w): one-pass range finder with the sketch sk (w, r) + one
    power step (S batched QRs of (b, r))"""
    q_, _ = fnp.linalg.qr(fnp.matmul(xb, sk))
    q_, _ = fnp.linalg.qr(fnp.matmul(xb, fnp.matmul(fnp.swapaxes(xb, -1, -2), q_)))
    return q_


def _tdp_mask_band(xb, r, sk):
    """tdp_patch TDP_MASK y: in place, xb (S, b, w) <- 0 (r <= 0) or its per-slot rank-r projection U (U^T xb)"""
    if r <= 0:
        fnp.copyto(xb, 0.0)
        return
    q_ = _tdp_basis(xb, r, sk)
    g_ = fnp.matmul(fnp.swapaxes(q_, -1, -2), xb)
    fnp.matmul(q_, g_, out=xb)


D21_BASIS = _os.environ.get('D21_BASIS', 'active_rf')
D21_LIFT_FMM_M = int(_os.environ.get('D21_LIFT_FMM_M', '0'))
JOINT_CP_ON = _os.environ.get('JOINT_CP_ON', '0') == '1'
D21_LOG = []
PL_LOG = []
RESP_ON = not MERGE_OFF and _os.environ.get('RESP_OFF', '0') != '1'
_I = WICK_PAIRS.index
TERM_SPECS[1, 1].append(('wk431', 'ones2', _I((1, 1)), _I((3, 1)), 1.0 / 3.0))
TERM_SPECS[2, 1].append(('wk431', 'ones2', _I((1, 2)), _I((3, 1)), 1.0 / 6.0))
EF_K31T21 = _os.environ.get('EF_K31T21', '1') == '1'   # agents_ideas/k31t21_patch.py: t/18246 sec. 5, needs K(i^3,j) = wk431[j,i]
TERM_SPECS[2, 1].append(('wk431T' if EF_K31T21 else 'wk431', 'ones2', _I((3, 2)), _I((1, 1)), 1.0 / 6.0))
TERM_SPECS[2, 2].append(('wk431', 'ones2', _I((1, 2)), _I((3, 2)), 1.0 / 3.0))
TD_DROP = _os.environ.get('TD_DROP', '')
for _tok in [t for t in TD_DROP.split(',') if t.strip()]:
    _pg, _a, _b = _tok.split(':')
    _ip = tuple((int(_c) for _c in _pg))
    TERM_SPECS[_ip] = [_t for _t in TERM_SPECS[_ip] if not (_t[0] == _a and _t[1] == _b)]
ALPHA_DEG = 5
SIG_EMIN, SIG_EMAX = (-6, 4)
MARG2_ON = _os.environ.get('MARG2', '1') == '1'   # marg2_patch: k = 2 marginal Edgeworth pair corrections only where alpha <= -MARG2_C
MARG2_C = float(_os.environ.get('MARG2_C', '2.0'))
MARG2_DEBUG = _os.environ.get('MARG2_DEBUG', '0') == '1'
_MARG2_W = {1: (WICK_PAIRS.index((5, 1)), WICK_PAIRS.index((5, 2))), 2: (WICK_PAIRS.index((6, 1)), WICK_PAIRS.index((6, 2)))}
_MARG2_SL = {}


def _marg2_slots(rows):
    """marg2_patch: row ranges of the merged-program table Xt (its blocks (f, lo, hi) stacked in order) that hold the k = 2 marginal
    Edgeworth corrections: f = 1 (x D3 = kappa3_ii) rows w(5, p) = w(2 + 3, p) and f = 2 (x g4 = kappa4_ii) rows w(6, p) = w(2 + 4, p),
    p = 1, 2.  The other f = 1 / f = 2 rows are k = 1 corrections (w(4, p) D3, w(5, p) g4); f = 0, 3, 4 rows carry none."""
    key = tuple(rows)
    out = _MARG2_SL.get(key)
    if out is None:
        sl = []
        o = 0
        for f, lo, hi in rows:
            for w in _MARG2_W.get(f, ()):
                if lo <= w < hi:
                    s = o + w - lo
                    if sl and sl[-1][1] == s:
                        sl[-1][1] = s + 1
                    else:
                        sl.append([s, s + 1])
            o += hi - lo
        out = _MARG2_SL[key] = tuple((a, b) for a, b in sl)
    return out


MARG1_ON = _os.environ.get('MARG1', '0') == '1'   # marg1_patch: k = 1 marginal Edgeworth pair corrections only where alpha <= -MARG1_C
MARG1_C = float(_os.environ.get('MARG1_C', '2.0'))
MARG1_DEBUG = _os.environ.get('MARG1_DEBUG', '0') == '1'
D2SQ_S = float(_os.environ.get('D2SQ_S', '0.5'))   # marg1_patch: scale of the D^2 pair diagrams (d21T o d21T, d21 o d21T), merged program
CK22_S = float(_os.environ.get('CK22_S', '1.0'))   # marg1_patch: scale of the C0 o K22 pair diagram (c_off o wk4m), merged program
_MARG1_W = {1: (WICK_PAIRS.index((4, 1)), WICK_PAIRS.index((4, 2))), 2: (WICK_PAIRS.index((5, 1)), WICK_PAIRS.index((5, 2)))}
_MARG1_SL = {}


def _marg1_slots(rows):
    """marg1_patch: row ranges of the merged-program table Xt (blocks (f, lo, hi) stacked in order) holding the k = 1 marginal Edgeworth
    corrections: f = 1 (x D3) rows w(4, p) = w(1 + 3, p), f = 2 (x g4) rows w(5, p) = w(1 + 4, p), p = 1, 2."""
    k1_ = tuple(rows)
    r1_ = _MARG1_SL.get(k1_)
    if r1_ is None:
        q1_ = []
        o1_ = 0
        for f1_, lo1_, hi1_ in rows:
            for w1_ in _MARG1_W.get(f1_, ()):
                if lo1_ <= w1_ < hi1_:
                    s1_ = o1_ + w1_ - lo1_
                    if q1_ and q1_[-1][1] == s1_:
                        q1_[-1][1] = s1_ + 1
                    else:
                        q1_.append([s1_, s1_ + 1])
            o1_ += hi1_ - lo1_
        r1_ = _MARG1_SL[k1_] = tuple((a1_, b1_) for a1_, b1_ in q1_)
    return r1_


def _dgs_coef(mats, coef):
    """marg1_patch: diagram scale of a merged-program term (D2SQ_S for D^2, CK22_S for C0 o K22); exact coef when the scale is 1.0"""
    s = D2SQ_S if mats in (('d21T', 'd21T'), ('d21', 'd21T')) else CK22_S if mats == ('c_off', 'wk4m') else 1.0
    return float(coef) if s == 1.0 else float(coef) * s


D2_IPS = [(1, 1), (2, 1), (2, 2)]
D1_IPS = [(1,), (2,), (3,), (4,)]
D2_OUT = [(1, 1), (2, 1), (2, 2)]
D1_OUT = [(2,), (3,), (4,)]
MODE0_MISSING = {'d21', 'd21T', 'd3col', 'd3row', 'd3', 'wk4m', 'g4col', 'g4row', 'g4', 'wk431', 'wk431T'}

def _statics(n: int) -> dict:
    c = 4 - 2 + n / 2.0 - 1.0
    P2 = (c - 2.0) / (16.0 * 2.0 * c * (1.0 - c) * (2.0 - c))
    return dict(P2=P2, k4_c4=24.0, k4_c22=24.0, wk4_c4=1.0, wk4_c22=1.0 / 3.0, cA=6.0 / (n + 4.0), cI=-3.0 / ((n + 2.0) * (n + 4.0)))

def _zero_diag(A):
    fnp.fill_diagonal(A, 0.0)
    return A

def _ns(name):
    """ledger tag (PORTLOT_NS=1 only); a no-op context otherwise"""
    return flops.namespace(name) if PL_NS else _contextlib.nullcontext()

def _build_wick_consts():
    np_ = len(WICK_PAIRS)
    C1 = [[0.0] * np_ for _ in range(ALPHA_DEG + 1)]
    C2 = [[0.0] * np_ for _ in range(ALPHA_DEG + 1)]
    SELC = [[0.0] * np_ for _ in range(SIG_EMAX - SIG_EMIN + 1)]
    for j, (const, e, p1, p2) in enumerate(WICK_UNIFIED):
        SELC[e - SIG_EMIN][j] = const
        for i, c in enumerate(p1):
            C1[i][j] = c
        for i, c in enumerate(p2):
            C2[i][j] = c
    return (C1, C2, SELC)

def _term_prog(mode: int):
    """Fused-einsum program for the nonlin terms of one availability mode."""
    missing = MODE0_MISSING if mode == 0 else {'wk431', 'wk431T'} if mode == 2 else set()
    np_ = len(WICK_PAIRS)
    d2 = []
    d1 = []
    for g, ip in enumerate(D2_IPS):
        for a, b, wl, wr, coef in TERM_SPECS.get(ip, []):
            if a in missing or b in missing:
                continue
            d2.append((g, a, b, wl, wr, coef))
    for g, ip in enumerate(D1_IPS):
        for a, b, wl, wr, coef in TERM_SPECS.get(ip, []):
            if a in missing or b in missing:
                continue
            d1.append((g, a, b, wl, coef))
    SL2 = [[0.0] * np_ for _ in d2]
    SR2 = [[0.0] * np_ for _ in d2]
    IND2 = [[0.0] * len(D2_IPS) for _ in d2]
    AB2 = []
    for t, (g, a, b, wl, wr, coef) in enumerate(d2):
        SL2[t][wl] = coef
        SR2[t][wr] = 1.0
        IND2[t][g] = 1.0
        AB2.append((a, b) if a <= b else (b, a))
    SL1 = [[0.0] * np_ for _ in d1]
    IND1 = [[0.0] * len(D1_IPS) for _ in d1]
    B1 = []
    for t, (g, a, b, wl, coef) in enumerate(d1):
        SL1[t][wl] = coef
        IND1[t][g] = 1.0
        B1.append((a, b) if a <= b else (b, a))
    return dict(SL2=SL2, SR2=SR2, IND2=IND2, AB2=AB2, SL1=SL1, IND1=IND1, B1=B1, N10=sum((1 for t_ in d1 if t_[0] == 0)))

# agents_joins NLK (2026-10-16): exact / rounding-level cuts of the nonlinear-term block; every knob '0' = the parent op stream.
NLK_PK = _os.environ.get('NLK_PK', '1') == '1'      # PK2 by the merged program (_nlk_prog): weights folded, shared pair products
NLK_K22 = _os.environ.get('NLK_K22', '1') == '1'    # K22 / pk22 only through their row sums; C = K11 copy; K21 scale folded


def _nlk_prog(mode: int):
    """NLK_PK: merged program of the D2 nonlinear terms of one availability mode (the same terms as _term_prog, same groups).
    A term is  base o (u (x) v)  with base = one b2 matrix or a pair product A o B, u = coef WT[wl] o rowfac, v = WT[wr] o colfac:
    the rank-1 b2 entries are factors (d3row = 1 d3^T, g4row = 1 g4^T scale columns; d3col, g4col scale rows; ones2 = 1).  Per group
    and base the terms are covered by stars of the (u-key, v-key) bipartite graph (greedy): a star sums the free weights of its
    terms first (D_u X D_v1 + D_u X D_v2 = D_u X D_(v1 + v2)).  The weights are the rows of SL / SR applied to the extended table
    Xt = [WT[lo0:hi0]; WT[lo1:hi1] o d3; WT[lo2:hi2] o g4] (one contiguous WT row range per factor)."""
    missing = MODE0_MISSING if mode == 0 else {'wk431', 'wk431T'} if mode == 2 else set()
    fac = {'ones2': (0, 0), 'd3row': (2, 1), 'g4row': (2, 2), 'd3col': (1, 1), 'g4col': (1, 2)}
    ents = []
    for g, ip in enumerate(D2_IPS):
        terms = []
        for a, b, wl, wr, coef in TERM_SPECS.get(ip, []):
            if a in missing or b in missing:
                continue
            rm = cm = 0
            mats = []
            for x in (a, b):
                if x in fac:
                    side, f = fac[x]
                    if side == 1:
                        assert rm == 0, (a, b)
                        rm = f
                    elif side == 2:
                        assert cm == 0, (a, b)
                        cm = f
                else:
                    mats.append(x)
            assert 1 <= len(mats) <= 2, (a, b)
            terms.append((tuple(sorted(mats)), (wl, rm), (wr, cm), _dgs_coef(tuple(sorted(mats)), coef)))
        bases = []
        for t in terms:
            if t[0] not in bases:
                bases.append(t[0])
        for base in bases:
            left = [t for t in terms if t[0] == base]
            while left:
                best = None
                for side in (1, 2):
                    cnt = {}
                    for t in left:
                        cnt[t[side]] = cnt.get(t[side], 0) + 1
                    for key, c in cnt.items():
                        if best is None or c > best[2]:
                            best = (side, key, c)
                side, key, _ = best
                star = [t for t in left if t[side] == key]
                left = [t for t in left if t[side] != key]
                free = {}
                for t in star:
                    free[t[3 - side]] = free.get(t[3 - side], 0.0) + t[3]
                u, v = ({key: 1.0}, free) if side == 1 else (free, {key: 1.0})
                ents.append((g, base, u, v))
    rng = {}
    for _, _, u, v in ents:
        for w, f in list(u) + list(v):
            lo, hi = rng.get(f, (w, w + 1))
            rng[f] = (min(lo, w), max(hi, w + 1))
    rows = []
    off = {}
    o = 0
    for f in sorted(rng):
        lo, hi = rng[f]
        rows.append((f, lo, hi))
        off[f] = o - lo
        o += hi - lo
    E = len(ents)
    SL = [[0.0] * o for _ in range(E)]
    SR = [[0.0] * o for _ in range(E)]
    pairs = []
    srcs = []
    groups = []
    for e, (g, base, u, v) in enumerate(ents):
        for (w, f), c in u.items():
            SL[e][off[f] + w] += c
        for (w, f), c in v.items():
            SR[e][off[f] + w] += c
        if len(base) == 1:
            srcs.append(base[0])
        else:
            if base not in pairs:
                pairs.append(base)
            srcs.append(pairs.index(base))
        if groups and groups[-1][0] == g:
            groups[-1][2] = e + 1
        else:
            groups.append([g, e, e + 1])
    return dict(E=E, rows=rows, SL=SL, SR=SR, pairs=pairs, src=srcs, groups=[tuple(x) for x in groups])


# agents_joins round 2 (2026-10-16): small exact cuts; every knob '0' = the parent op stream.
J2_W431 = _os.environ.get('J2_W431', '1') == '1'   # (NLK_PK) K31TWO wk431 expanded into the merged program (no n x n wk431)
J2_LOT = _os.environ.get('J2_LOT', '1') == '1'     # _lot_g4 without T2 = T1 o w2b (w2b folded into gu, the same-weights, V rows)
J2_RRW = _os.environ.get('J2_RRW', '1') == '1'     # thin-residual range finder on Rres[:n3] (SDT zero rows)
J2_D21F = _os.environ.get('J2_D21F', '1') == '1'   # D21_new: the 1/3 scalings folded into small factors / vectors
J2_KP = _os.environ.get('J2_KP', '1') == '1'       # FIJA2 tier-2 Kp = R11^-1 R12 by a depth-2 block bisection (R11 upper triangular)


def _j2_nlk_w431():
    """J2_W431: _nlk_prog(1) with each K31TWO wk431 term (wk431 = C_off D_a + D21^T D_b, diagonal unread) expanded into a c_off term with
    column factor a (f = 3) and a d21T term with column factor b (f = 4), then the same star cover / weight tables as _nlk_prog"""
    fac = {'ones2': (0, 0), 'd3row': (2, 1), 'g4row': (2, 2), 'd3col': (1, 1), 'g4col': (1, 2)}
    ents = []
    for g, ip in enumerate(D2_IPS):
        terms = []
        for a, b, wl, wr, coef in TERM_SPECS.get(ip, []):
            rm = cm = 0
            mats = []
            for x in (a, b):
                if x in fac:
                    side, f = fac[x]
                    if side == 1:
                        assert rm == 0, (a, b)
                        rm = f
                    elif side == 2:
                        assert cm == 0, (a, b)
                        cm = f
                else:
                    mats.append(x)
            assert 1 <= len(mats) <= 2, (a, b)
            if mats == ['wk431T']:
                # EF_K31T21: wk431^T = D_a C_off + D_b D21 -> c_off with ROW factor a (f = 3), d21 with ROW factor b (f = 4)
                assert rm == 0 and cm == 0, (a, b)
                terms.append((('c_off',), (wl, 3), (wr, 0), float(coef)))
                terms.append((('d21',), (wl, 4), (wr, 0), float(coef)))
                continue
            if mats == ['wk431']:
                assert cm == 0, (a, b)
                terms.append((('c_off',), (wl, rm), (wr, 3), float(coef)))
                terms.append((('d21T',), (wl, rm), (wr, 4), float(coef)))
                continue
            assert 'wk431' not in mats, (a, b)
            terms.append((tuple(sorted(mats)), (wl, rm), (wr, cm), _dgs_coef(tuple(sorted(mats)), coef)))
        bases = []
        for t in terms:
            if t[0] not in bases:
                bases.append(t[0])
        for base in bases:
            left = [t for t in terms if t[0] == base]
            while left:
                best = None
                for side in (1, 2):
                    cnt = {}
                    for t in left:
                        cnt[t[side]] = cnt.get(t[side], 0) + 1
                    for key, c in cnt.items():
                        if best is None or c > best[2]:
                            best = (side, key, c)
                side, key, _ = best
                star = [t for t in left if t[side] == key]
                left = [t for t in left if t[side] != key]
                free = {}
                for t in star:
                    free[t[3 - side]] = free.get(t[3 - side], 0.0) + t[3]
                u, v = ({key: 1.0}, free) if side == 1 else (free, {key: 1.0})
                ents.append((g, base, u, v))
    rng = {}
    for _, _, u, v in ents:
        for w, f in list(u) + list(v):
            lo, hi = rng.get(f, (w, w + 1))
            rng[f] = (min(lo, w), max(hi, w + 1))
    rows = []
    off = {}
    o = 0
    for f in sorted(rng):
        lo, hi = rng[f]
        rows.append((f, lo, hi))
        off[f] = o - lo
        o += hi - lo
    E = len(ents)
    SL = [[0.0] * o for _ in range(E)]
    SR = [[0.0] * o for _ in range(E)]
    pairs = []
    srcs = []
    groups = []
    for e, (g, base, u, v) in enumerate(ents):
        for (w, f), c in u.items():
            SL[e][off[f] + w] += c
        for (w, f), c in v.items():
            SR[e][off[f] + w] += c
        if len(base) == 1:
            srcs.append(base[0])
        else:
            if base not in pairs:
                pairs.append(base)
            srcs.append(pairs.index(base))
        if groups and groups[-1][0] == g:
            groups[-1][2] = e + 1
        else:
            groups.append([g, e, e + 1])
    return dict(E=E, rows=rows, SL=SL, SR=SR, pairs=pairs, src=srcs, groups=[tuple(x) for x in groups])
STRASSEN_LEVELS = int(_os.environ.get('V26_STRASSEN', '5'))
STRASSEN_HUB = int(_os.environ.get('V26_STRASSEN_HUB', str(STRASSEN_LEVELS)))
STRASSEN_NEW = int(_os.environ.get('V26_STRASSEN_NEW', '0'))
CPRE_LEV = int(_os.environ.get('V29_CPRE_LEV', _os.environ.get('V26_STRASSEN', '5')))
STRASSEN_SB = int(_os.environ.get('V28_STRASSEN_SB', '6'))
STRASSEN_MIN = int(_os.environ.get('V26_STRASSEN_MIN', '32'))
STRASSEN_FIRST = int(_os.environ.get('V28_STRASSEN_FIRST', '4'))
STRASSEN_FUSE_P = int(_os.environ.get('V28_STRASSEN_FUSE_P', '343'))
WARM = _os.environ.get('V26_WARM', '1') == '1'
TA_GRAM = _os.environ.get('TA_GRAM', _ON) == '1'
TA_CPRE = _os.environ.get('TA_CPRE', _ON) == '1'
TA_YH_LEV = int(_os.environ.get('TA_YH_LEV', '6'))
TA_YH_MIN = int(_os.environ.get('TA_YH_MIN', '16'))
TA_WICK = _os.environ.get('TA_WICK', _ON) == '1'
TA_YH_S6 = int(_os.environ.get('TA_YH_S6', '0'))
TA_YH_SMAX = int(_os.environ.get('TA_YH_SMAX', '9'))
TA_NL = _os.environ.get('TA_NL', _ON) == '1'
TA_WINO = _os.environ.get('TA_WINO', _ON) == '1'
TA_MMFB = _os.environ.get('TA_MMFB', _ON) == '1'
TA_JOIN = _os.environ.get('TA_JOIN', '1') == '1'
BQ_SRCP = _os.environ.get('BQ_SRCP', _ON) == '1'
TA_JOIN_LEV = int(_os.environ.get('TA_JOIN_LEV', '6'))
TA_JOIN_MIN = int(_os.environ.get('TA_JOIN_MIN', '6'))
TA_OLD_MIN = int(_os.environ.get('TA_OLD_MIN', '32' if MERGE_OFF else '6'))
STRASSEN_HUB = int(_os.environ.get('V26_STRASSEN_HUB', str(TA_YH_LEV)))
_TAG = {'wino': False, 'mmfb': False, 'old_min': 32, 'join': False, 'yh_min': 32}
AS_OFF = _os.environ.get('AS_OFF', '0') == '1'
AS_K = _os.environ.get('AS_K', '')
AS_FORCE_PERM = _os.environ.get('AS_FORCE_PERM', '0') == '1'
AS_YMIN = int(_os.environ.get('AS_YMIN', '12'))
AS_DEBUG = _os.environ.get('AS_DEBUG', '0') == '1'
AS_ADAPT = _os.environ.get('AS_ADAPT', '1') == '1'
AS_CDEAD = float(_os.environ.get('AS_CDEAD', '3.0'))
AS_L0 = int(_os.environ.get('AS_L0', '6'))
AS_GRAN = int(_os.environ.get('AS_GRAN', '32'))


def _perlayer(spec, default):
    """per-layer threshold table from a spec "a-b/t;c/t"; layers not listed use default"""
    m = {}
    for part in spec.split(';'):
        if '/' in part:
            r, t = part.split('/')
            a, _, b = r.partition('-')
            for l_ in range(int(a), int(b or a) + 1):
                m[l_] = float(t)
    return lambda li: m.get(li, default)
AS_CDEAD_L = _os.environ.get('AS_CDEAD_L', '6-9/2.5')
SDT_C_L = _os.environ.get('SDT_C_L', '')
SDT_COL_L = _os.environ.get('SDT_COL_L', '')
D21_CUT_L = _os.environ.get('D21_CUT_L', '')
_AS_CDEAD_L = _perlayer(AS_CDEAD_L, AS_CDEAD)
_SDT_C_L = _perlayer(SDT_C_L, SDT_C)
_SDT_COL_L = _perlayer(SDT_COL_L, SDT_COL)
_D21_CUT_L = _perlayer(D21_CUT_L, D21_CUT)
AS_MARGIN = int(_os.environ.get('AS_MARGIN', '4'))
AS_MCORR = _os.environ.get('AS_MCORR', '1') == '1'
AS_LAST = _os.environ.get('AS_LAST', '1') == '1'
AS_GCONST = _os.environ.get('AS_GCONST', '1') == '1'
AS_FULLMEAN = _os.environ.get('AS_FULLMEAN', '0') == '1'
AS_TEST_N = int(_os.environ.get('AS_TEST_N', '0'))
AS_LOG = []
YCOL_TRIM = _os.environ.get('YCOL_TRIM', '1') == '1'
YCOL_READOUT = _os.environ.get('YCOL_READOUT', '1') == '1'
AS_FINAL_TAILS = _os.environ.get('AS_FINAL_TAILS', '1') == '1'
AS_FINAL_C = float(_os.environ.get('AS_FINAL_C', '3.0'))
AS_FINAL_MAX = int(_os.environ.get('AS_FINAL_MAX', '448'))
AS_FINAL_KC = int(_os.environ.get('AS_FINAL_KC', '224'))   # fin_patch: final drop pool per side (0 = parent min(192, 32 (1 + li // 3)))
FIN_KC_ON = AS_FINAL_KC > 0
FIN_GRAN = int(_os.environ.get('FIN_GRAN', '0'))   # fin_patch: final drop-count granularity (0 = AS_GRAN)
FIN_GRAN_ON = FIN_GRAN > 0
FIN_EDGE_ON = _os.environ.get('FIN_EDGE', '1') == '1'   # fin_patch: radial 2-term Edgeworth mean for the dropped final rows
FIN_EDGE_AMIN = float(_os.environ.get('FIN_EDGE_AMIN', '1.5'))
FIN_EDGE_K4S = float(_os.environ.get('FIN_EDGE_K4S', '1.3'))
FIN_EDGE_S3S = float(_os.environ.get('FIN_EDGE_S3S', '1.0'))
FIN_DEBUG = _os.environ.get('FIN_DEBUG', '0') == '1'
MID_EDGE_ON = _os.environ.get('MID_EDGE', '1') == '1'   # mid_patch: radial 2-term Edgeworth mean for the dropped INTERMEDIATE rows
MID_EDGE_AMIN = float(_os.environ.get('MID_EDGE_AMIN', '1.5'))
MID_EDGE_K4S = float(_os.environ.get('MID_EDGE_K4S', '1.3'))
MID_EDGE_S3S = float(_os.environ.get('MID_EDGE_S3S', '1.2'))
MID_DEBUG = _os.environ.get('MID_DEBUG', '0') == '1'

def _win(buf, r, c):
    """Active-set window [..., :r, :c] of a pooled full-size buffer (the buffer itself when it already has that size)."""
    if buf.shape[-2] == r and buf.shape[-1] == c:
        return buf
    return _pview(buf, (..., slice(None, r, None), slice(None, c, None)))

class _Strassen:
    """Recursive Strassen on batched operands. Layouts (last two axes = the matrix):
      plain: X (bx, P, m, kd) @ Y (by, P, kd, w) -> out (by, P, m, w), bx in {1, by}
      hub:   X (k, P, m, kd), Y (k, P, w, kd)    -> out (P, m, w) = sum_k X_k @ Y_k^T
    Every combo / product is ONE flopscope op over the whole batch, written with out=
    into pooled buffers shared by role (F53: fresh result buffers dominate residual).
    The seven products: M1=(X11+X22)(Y11+Y22) M2=(X21+X22)Y11 M3=X11(Y12-Y22)
    M4=X22(Y21-Y11) M5=(X11+X12)Y22 M6=(X21-X11)(Y11+Y12) M7=(X12-X22)(Y21+Y22);
    C11=M1+M4-M5+M7 C12=M3+M5 C21=M2+M4 C22=M1-M2+M3+M6.  For the hub kernel the
    right operand is Y^T, whose blocks are (Y^T)11=Y[..,:h,:q], (Y^T)12=Y[..,h:,:q],
    (Y^T)21=Y[..,:h,q:], (Y^T)22=Y[..,h:,q:] (no transposes: the kernel contracts j)."""

    def __init__(self, dtype, bmax):
        self.dtype = dtype
        self.bmax = int(bmax)
        self.pool = {}
        self.mn = STRASSEN_MIN

    def _buf(self, key, shape):
        """V27: pools grow to the largest batch actually seen (the suite's young-source
        batch is <= 2 (AGE_OLD + 1), a third of the 2 (L-1) capacity), so deeper levels
        stay within memory; a larger batch later simply reallocates once."""
        b = self.pool.get(key)
        if b is None or b.shape[0] < shape[0]:
            b = fnp.empty(shape, dtype=self.dtype)
            fnp.copyto(b, 0.0)
            self.pool[key] = b
        return b

    def _buf2(self, key, shape5):
        """V27: a pooled (b, 7, P, h, q) buffer together with its (b, 7P, h, q) view,
        both made once at allocation (fnp.reshape is a logged op: 0.06-0.3 ms per call,
        462 calls per MLP at four levels)."""
        e = self.pool.get(key)
        b = int(shape5[0])
        if e is None or e[0].shape[0] < b:
            b5 = fnp.empty(shape5, dtype=self.dtype)
            fnp.copyto(b5, 0.0)
            b4 = fnp.reshape(b5, (b, shape5[1] * shape5[2], shape5[3], shape5[4]))
            e = (b5, b4)
            self.pool[key] = e
        return (_pview(e[0], slice(None, b, None)), _pview(e[1], slice(None, b, None)))

    def level(self, m, kd, w, lev):
        """V28: largest level <= lev at which every block side divides and stays
        >= STRASSEN_MIN (a family whose sides do not allow `lev` runs shallower
        instead of falling back to dense)."""
        while lev > 0 and (not self._ok(m, kd, w, lev)):
            lev -= 1
        return lev

    def _ok(self, m, kd, w, lev):
        d = 2 ** lev
        return lev > 0 and m % d == 0 and (kd % d == 0) and (w % d == 0) and (min(m, kd, w) // d >= self.mn)

    def _combos(self, X, Q, kind, key):
        """Seven Strassen combos of the quadrant views Q=(X11,X12,X21,X22) of X
        (b, P, m, kd) into a pooled (bmax, 7, P, h, q) buffer; returns (b, 7P, h, q)."""
        b, P, m, kd = X.shape
        h, q = (m // 2, kd // 2)
        buf, buf4 = self._buf2(key, (b, 7, P, h, q))
        X11, X12, X21, X22 = Q
        if _TAG['wino']:
            if kind == 'L':
                fnp.add(X21, X22, out=_pview(buf, (slice(None, None, None), 4)))
                fnp.subtract(_pview(buf, (slice(None, None, None), 4)), X11, out=_pview(buf, (slice(None, None, None), 5)))
                fnp.subtract(X11, X21, out=_pview(buf, (slice(None, None, None), 6)))
                fnp.subtract(X12, _pview(buf, (slice(None, None, None), 5)), out=_pview(buf, (slice(None, None, None), 2)))
                fnp.copyto(_pview(buf, (slice(None, None, None), 0)), X11)
                fnp.copyto(_pview(buf, (slice(None, None, None), 1)), X12)
                fnp.copyto(_pview(buf, (slice(None, None, None), 3)), X22)
            else:
                fnp.subtract(X12, X11, out=_pview(buf, (slice(None, None, None), 4)))
                fnp.subtract(X22, _pview(buf, (slice(None, None, None), 4)), out=_pview(buf, (slice(None, None, None), 5)))
                fnp.subtract(X22, X12, out=_pview(buf, (slice(None, None, None), 6)))
                fnp.subtract(_pview(buf, (slice(None, None, None), 5)), X21, out=_pview(buf, (slice(None, None, None), 3)))
                fnp.copyto(_pview(buf, (slice(None, None, None), 0)), X11)
                fnp.copyto(_pview(buf, (slice(None, None, None), 1)), X21)
                fnp.copyto(_pview(buf, (slice(None, None, None), 2)), X22)
            return buf4
        if kind == 'L':
            fnp.add(X11, X22, out=_pview(buf, (slice(None, None, None), 0)))
            fnp.add(X21, X22, out=_pview(buf, (slice(None, None, None), 1)))
            fnp.copyto(_pview(buf, (slice(None, None, None), 2)), X11)
            fnp.copyto(_pview(buf, (slice(None, None, None), 3)), X22)
            fnp.add(X11, X12, out=_pview(buf, (slice(None, None, None), 4)))
            fnp.subtract(X21, X11, out=_pview(buf, (slice(None, None, None), 5)))
            fnp.subtract(X12, X22, out=_pview(buf, (slice(None, None, None), 6)))
        else:
            fnp.add(X11, X22, out=_pview(buf, (slice(None, None, None), 0)))
            fnp.copyto(_pview(buf, (slice(None, None, None), 1)), X11)
            fnp.subtract(X12, X22, out=_pview(buf, (slice(None, None, None), 2)))
            fnp.subtract(X21, X11, out=_pview(buf, (slice(None, None, None), 3)))
            fnp.copyto(_pview(buf, (slice(None, None, None), 4)), X22)
            fnp.add(X11, X12, out=_pview(buf, (slice(None, None, None), 5)))
            fnp.add(X21, X22, out=_pview(buf, (slice(None, None, None), 6)))
        return buf4

    @staticmethod
    def _assemble(M, out):
        """M (b, 7, P, h, w) products -> out (b, P, m, w) quadrants (8 ops)."""
        h, w = (M.shape[3], M.shape[4])
        C11, C12 = (_pview(out, (..., slice(None, h, None), slice(None, w, None))), _pview(out, (..., slice(None, h, None), slice(w, None, None))))
        C21, C22 = (_pview(out, (..., slice(h, None, None), slice(None, w, None))), _pview(out, (..., slice(h, None, None), slice(w, None, None))))
        if _TAG['wino']:
            fnp.add(_pview(M, (slice(None, None, None), 0)), _pview(M, (slice(None, None, None), 5)), out=C22)
            fnp.add(_pview(M, (slice(None, None, None), 0)), _pview(M, (slice(None, None, None), 1)), out=C11)
            fnp.add(C22, _pview(M, (slice(None, None, None), 4)), out=C12)
            fnp.add(C12, _pview(M, (slice(None, None, None), 2)), out=C12)
            fnp.add(C22, _pview(M, (slice(None, None, None), 6)), out=C21)
            fnp.add(C21, _pview(M, (slice(None, None, None), 4)), out=C22)
            fnp.subtract(C21, _pview(M, (slice(None, None, None), 3)), out=C21)
            return
        fnp.add(_pview(M, (slice(None, None, None), 0)), _pview(M, (slice(None, None, None), 3)), out=C11)
        fnp.subtract(C11, _pview(M, (slice(None, None, None), 4)), out=C11)
        fnp.add(C11, _pview(M, (slice(None, None, None), 6)), out=C11)
        fnp.add(_pview(M, (slice(None, None, None), 2)), _pview(M, (slice(None, None, None), 4)), out=C12)
        fnp.add(_pview(M, (slice(None, None, None), 1)), _pview(M, (slice(None, None, None), 3)), out=C21)
        fnp.subtract(_pview(M, (slice(None, None, None), 0)), _pview(M, (slice(None, None, None), 1)), out=C22)
        fnp.add(C22, _pview(M, (slice(None, None, None), 2)), out=C22)
        fnp.add(C22, _pview(M, (slice(None, None, None), 5)), out=C22)

    def mm(self, X, Y, out, lev):
        """plain: X (bx, P, m, kd), Y (by, P, kd, w) -> out (by, P, m, w)."""
        bx, P, m, kd = X.shape
        by, w = (Y.shape[0], Y.shape[3])
        if not self._ok(m, kd, w, lev):
            fnp.matmul(X, Y, out=out)
            return
        h, q, v = (m // 2, kd // 2, w // 2)
        XQ = (_pview(X, (..., slice(None, h, None), slice(None, q, None))), _pview(X, (..., slice(None, h, None), slice(q, None, None))), _pview(X, (..., slice(h, None, None), slice(None, q, None))), _pview(X, (..., slice(h, None, None), slice(q, None, None))))
        YQ = (_pview(Y, (..., slice(None, q, None), slice(None, v, None))), _pview(Y, (..., slice(None, q, None), slice(v, None, None))), _pview(Y, (..., slice(q, None, None), slice(None, v, None))), _pview(Y, (..., slice(q, None, None), slice(v, None, None))))
        if P >= STRASSEN_FUSE_P and (not self._ok(h, q, v, lev - 1)):
            (self._leaf_mm_w if _TAG['wino'] else self._leaf_mm)(XQ, YQ, (_pview(out, (..., slice(None, h, None), slice(None, v, None))), _pview(out, (..., slice(None, h, None), slice(v, None, None))), _pview(out, (..., slice(h, None, None), slice(None, v, None))), _pview(out, (..., slice(h, None, None), slice(v, None, None)))), bx, by, P, h, q, v)
            return
        Xc = self._combos(X, XQ, 'L', ('L', P, h, q))
        Yc = self._combos(Y, YQ, 'R', ('R', P, q, v))
        callee_recurses = self._ok(h // 2, q // 2, v // 2, lev - 2)
        mkey = ('L', P, h, q) if callee_recurses and q == v else ('M', P, h, v)
        Mb, Mb4 = self._buf2(mkey, (by, 7, P, h, v))
        self.mm(Xc, Yc, Mb4, lev - 1)
        self._assemble(Mb, out)

    def _leaf_mm(self, XQ, YQ, OQ, bx, by, P, h, q, v):
        """Fused leaf of mm: M6->C22, M4->C11, M1 (+C11, +C22), M2->C21 (C22-=), M3->C12
        (C22+=), M5 (C11-=, C12+=), M7 (C11+=): 7 matmul writes + 8 adds, one combo
        pair alive at a time (copies for the raw-quadrant operands keep every matmul
        input contiguous)."""
        X11, X12, X21, X22 = XQ
        Y11, Y12, Y21, Y22 = YQ
        C11, C12, C21, C22 = OQ
        Lb = _pview(self._buf(('L1', P, h, q), (bx, P, h, q)), slice(None, bx, None))
        Rb = _pview(self._buf(('R1', P, q, v), (by, P, q, v)), slice(None, by, None))
        Mb = _pview(self._buf(('M1', P, h, v), (by, P, h, v)), slice(None, by, None))
        fnp.subtract(X21, X11, out=Lb)
        fnp.add(Y11, Y12, out=Rb)
        fnp.matmul(Lb, Rb, out=C22)
        fnp.add(X21, X22, out=Lb)
        fnp.matmul(Lb, Y11, out=C21)
        fnp.subtract(C22, C21, out=C22)
        fnp.subtract(Y21, Y11, out=Rb)
        fnp.matmul(X22, Rb, out=C11)
        fnp.add(C21, C11, out=C21)
        fnp.add(X11, X22, out=Lb)
        fnp.add(Y11, Y22, out=Rb)
        fnp.matmul(Lb, Rb, out=Mb)
        fnp.add(C11, Mb, out=C11)
        fnp.add(C22, Mb, out=C22)
        fnp.subtract(Y12, Y22, out=Rb)
        fnp.matmul(X11, Rb, out=C12)
        fnp.add(C22, C12, out=C22)
        fnp.add(X11, X12, out=Lb)
        fnp.matmul(Lb, Y22, out=Mb)
        fnp.subtract(C11, Mb, out=C11)
        fnp.add(C12, Mb, out=C12)
        fnp.subtract(X12, X22, out=Lb)
        fnp.add(Y21, Y22, out=Rb)
        fnp.matmul(Lb, Rb, out=Mb)
        fnp.add(C11, Mb, out=C11)

    def hub(self, X, Y, out, lev):
        """hub: X (k, P, m, kd), Y (k, P, w, kd) -> out (P, m, w) = sum_k X_k Y_k^T."""
        k, P, m, kd = X.shape
        w = Y.shape[2]
        if not self._ok(m, kd, w, lev):
            T = _pview(self._buf(('K', P, m, w), (k, P, m, w)), slice(None, k, None))
            fnp.matmul(X, fnp.swapaxes(Y, -1, -2), out=T)
            fnp.sum(T, axis=0, out=out)
            return
        h, q, v = (m // 2, kd // 2, w // 2)
        XQ = (_pview(X, (..., slice(None, h, None), slice(None, q, None))), _pview(X, (..., slice(None, h, None), slice(q, None, None))), _pview(X, (..., slice(h, None, None), slice(None, q, None))), _pview(X, (..., slice(h, None, None), slice(q, None, None))))
        YQ = (_pview(Y, (..., slice(None, v, None), slice(None, q, None))), _pview(Y, (..., slice(v, None, None), slice(None, q, None))), _pview(Y, (..., slice(None, v, None), slice(q, None, None))), _pview(Y, (..., slice(v, None, None), slice(q, None, None))))
        if P >= STRASSEN_FUSE_P and (not self._ok(h, q, v, lev - 1)):
            (self._leaf_hub_w if _TAG['wino'] else self._leaf_hub)(XQ, YQ, (_pview(out, (..., slice(None, h, None), slice(None, v, None))), _pview(out, (..., slice(None, h, None), slice(v, None, None))), _pview(out, (..., slice(h, None, None), slice(None, v, None))), _pview(out, (..., slice(h, None, None), slice(v, None, None)))), k, P, h, q, v)
            return
        Xc = self._combos(X, XQ, 'L', ('L', P, h, q))
        Yc = self._combos(Y, YQ, 'R', ('R', P, v, q))
        Mb, Mb4 = self._buf2(('HM', P, h, v), (1, 7, P, h, v))
        self.hub(Xc, Yc, Mb4[0], lev - 1)
        self._assemble(Mb, _pview(out, None))

    def _leaf_hub(self, XQ, YQ, OQ, k, P, h, q, v):
        """Fused leaf of hub (same product order as _leaf_mm); each product is a batched
        GEMM over (k, P) into T then a k-sum into its quadrant / the (P, h, v) scratch."""
        X11, X12, X21, X22 = XQ
        Y11, Y12, Y21, Y22 = YQ
        C11, C12, C21, C22 = OQ
        Lb = _pview(self._buf(('L1', P, h, q), (k, P, h, q)), slice(None, k, None))
        Rb = _pview(self._buf(('R1', P, v, q), (k, P, v, q)), slice(None, k, None))
        T = _pview(self._buf(('K', P, h, v), (k, P, h, v)), slice(None, k, None))
        Mb = self._buf(('KS', P, h, v), (1, P, h, v))[0]

        def prod(L_, R_, dst):
            fnp.matmul(L_, fnp.swapaxes(R_, -1, -2), out=T)
            fnp.sum(T, axis=0, out=dst)
        fnp.subtract(X21, X11, out=Lb)
        fnp.add(Y11, Y12, out=Rb)
        prod(Lb, Rb, C22)
        fnp.add(X21, X22, out=Lb)
        prod(Lb, Y11, C21)
        fnp.subtract(C22, C21, out=C22)
        fnp.subtract(Y21, Y11, out=Rb)
        prod(X22, Rb, C11)
        fnp.add(C21, C11, out=C21)
        fnp.add(X11, X22, out=Lb)
        fnp.add(Y11, Y22, out=Rb)
        prod(Lb, Rb, Mb)
        fnp.add(C11, Mb, out=C11)
        fnp.add(C22, Mb, out=C22)
        fnp.subtract(Y12, Y22, out=Rb)
        prod(X11, Rb, C12)
        fnp.add(C22, C12, out=C22)
        fnp.add(X11, X12, out=Lb)
        prod(Lb, Y22, Mb)
        fnp.subtract(C11, Mb, out=C11)
        fnp.add(C12, Mb, out=C12)
        fnp.subtract(X12, X22, out=Lb)
        fnp.add(Y21, Y22, out=Rb)
        prod(Lb, Rb, Mb)
        fnp.add(C11, Mb, out=C11)

    @staticmethod
    def _wino_leaf(XQ, YQ, OQ, Lb, Rb, Rb2, Mb, Mb2, prod):
        """Winograd fused leaf: 7 products one at a time, 15 adds, no 7x pools.
        prod(L, R, dst) writes the (summed) product into dst."""
        X11, X12, X21, X22 = XQ
        Y11, Y12, Y21, Y22 = YQ
        C11, C12, C21, C22 = OQ
        add, sub = (fnp.add, fnp.subtract)
        add(X21, X22, out=Lb)
        sub(Y12, Y11, out=Rb)
        prod(Lb, Rb, C21)
        sub(Lb, X11, out=Lb)
        sub(Y22, Rb, out=Rb)
        prod(Lb, Rb, Mb)
        prod(X11, Y11, C11)
        add(C11, Mb, out=C22)
        add(C22, C21, out=C12)
        sub(X12, Lb, out=Lb)
        prod(Lb, Y22, Mb)
        add(C12, Mb, out=C12)
        sub(Rb, Y21, out=Rb)
        prod(X22, Rb, Mb)
        sub(X11, X21, out=Lb)
        sub(Y22, Y12, out=Rb2)
        prod(Lb, Rb2, Mb2)
        add(C22, Mb2, out=Mb2)
        add(Mb2, C21, out=C22)
        sub(Mb2, Mb, out=C21)
        prod(X12, Y21, Mb)
        add(C11, Mb, out=C11)

    def _leaf_mm_w(self, XQ, YQ, OQ, bx, by, P, h, q, v):
        Lb = _pview(self._buf(('L1', P, h, q), (bx, P, h, q)), slice(None, bx, None))
        Rb = _pview(self._buf(('R1', P, q, v), (by, P, q, v)), slice(None, by, None))
        Rb2 = _pview(self._buf(('R2', P, q, v), (by, P, q, v)), slice(None, by, None))
        Mb = _pview(self._buf(('M1', P, h, v), (by, P, h, v)), slice(None, by, None))
        Mb2 = _pview(self._buf(('M2', P, h, v), (by, P, h, v)), slice(None, by, None))
        self._wino_leaf(XQ, YQ, OQ, Lb, Rb, Rb2, Mb, Mb2, lambda L_, R_, dst: _public_out(fnp.matmul, L_, R_, out=dst))

    def _leaf_hub_w(self, XQ, YQ, OQ, k, P, h, q, v):
        Lb = _pview(self._buf(('L1', P, h, q), (k, P, h, q)), slice(None, k, None))
        Rb = _pview(self._buf(('R1', P, v, q), (k, P, v, q)), slice(None, k, None))
        Rb2 = _pview(self._buf(('R2', P, v, q), (k, P, v, q)), slice(None, k, None))
        T = _pview(self._buf(('K', P, h, v), (k, P, h, v)), slice(None, k, None))
        Mb = self._buf(('KS', P, h, v), (1, P, h, v))[0]
        Mb2 = self._buf(('KS3', P, h, v), (1, P, h, v))[0]

        def prod(L_, R_, dst):
            fnp.matmul(L_, fnp.swapaxes(R_, -1, -2), out=T)
            fnp.sum(T, axis=0, out=dst)
        self._wino_leaf(XQ, YQ, OQ, Lb, Rb, Rb2, Mb, Mb2, prod)

    def _leaf_hubg_w(self, XQ, YQ, OQ, k, P, h, q, v, k1):
        assert k - k1 == 1
        Lb = _pview(self._buf(('L1', P, h, q), (k, P, h, q)), slice(None, k, None))
        Rb = _pview(self._buf(('R1', P, v, q), (k, P, v, q)), slice(None, k, None))
        Rb2 = _pview(self._buf(('R2', P, v, q), (k, P, v, q)), slice(None, k, None))
        T = _pview(self._buf(('K', P, h, v), (k, P, h, v)), slice(None, k1, None))
        Mb = self._buf(('KS2', P, h, v), (2, P, h, v))
        Mb2 = self._buf(('KS4', P, h, v), (2, P, h, v))

        def prod(L_, R_, dst):
            fnp.matmul(_pview(L_, slice(None, k1, None)), fnp.swapaxes(_pview(R_, slice(None, k1, None)), -1, -2), out=T)
            fnp.sum(T, axis=0, out=dst[0])
            fnp.matmul(L_[k1], fnp.swapaxes(R_[k1], -1, -2), out=dst[1])
        self._wino_leaf(XQ, YQ, OQ, Lb, Rb, Rb2, Mb, Mb2, prod)

    def hubg(self, X, Y, out, lev, k1):
        """trimalg: grouped hub. X (k, P, m, kd), Y (k, P, w, kd) -> out (2, P, m, w):
        out[0] = sum_{k < k1} X_k Y_k^T, out[1] = sum_{k >= k1} X_k Y_k^T (the same Strassen
        recursion as hub(); the group axis rides the assembly, the leaf sums per group)."""
        k, P, m, kd = X.shape
        w = Y.shape[2]
        if not self._ok(m, kd, w, lev):
            T = _pview(self._buf(('K', P, m, w), (k, P, m, w)), slice(None, k, None))
            fnp.matmul(X, fnp.swapaxes(Y, -1, -2), out=T)
            fnp.sum(_pview(T, slice(None, k1, None)), axis=0, out=out[0])
            fnp.sum(_pview(T, slice(k1, None, None)), axis=0, out=out[1])
            return
        h, q, v = (m // 2, kd // 2, w // 2)
        XQ = (_pview(X, (..., slice(None, h, None), slice(None, q, None))), _pview(X, (..., slice(None, h, None), slice(q, None, None))), _pview(X, (..., slice(h, None, None), slice(None, q, None))), _pview(X, (..., slice(h, None, None), slice(q, None, None))))
        YQ = (_pview(Y, (..., slice(None, v, None), slice(None, q, None))), _pview(Y, (..., slice(v, None, None), slice(None, q, None))), _pview(Y, (..., slice(None, v, None), slice(q, None, None))), _pview(Y, (..., slice(v, None, None), slice(q, None, None))))
        if P >= STRASSEN_FUSE_P and (not self._ok(h, q, v, lev - 1)):
            (self._leaf_hubg_w if _TAG['wino'] else self._leaf_hubg)(XQ, YQ, (_pview(out, (..., slice(None, h, None), slice(None, v, None))), _pview(out, (..., slice(None, h, None), slice(v, None, None))), _pview(out, (..., slice(h, None, None), slice(None, v, None))), _pview(out, (..., slice(h, None, None), slice(v, None, None)))), k, P, h, q, v, k1)
            return
        Xc = self._combos(X, XQ, 'L', ('L', P, h, q))
        Yc = self._combos(Y, YQ, 'R', ('R', P, v, q))
        Mb, Mb4 = self._buf2(('HM2', P, h, v), (2, 7, P, h, v))
        self.hubg(Xc, Yc, Mb4, lev - 1, k1)
        self._assemble(Mb, out)

    def _leaf_hubg(self, XQ, YQ, OQ, k, P, h, q, v, k1):
        """Fused leaf of hubg (same product order as _leaf_hub); group 0 = slots [:k1] summed,
        group 1 = slots [k1:] (a single slot: the matmul writes straight into its destination)."""
        assert k - k1 == 1
        X11, X12, X21, X22 = XQ
        Y11, Y12, Y21, Y22 = YQ
        C11, C12, C21, C22 = OQ
        Lb = _pview(self._buf(('L1', P, h, q), (k, P, h, q)), slice(None, k, None))
        Rb = _pview(self._buf(('R1', P, v, q), (k, P, v, q)), slice(None, k, None))
        T = _pview(self._buf(('K', P, h, v), (k, P, h, v)), slice(None, k1, None))
        Mb = self._buf(('KS2', P, h, v), (2, P, h, v))

        def prod(L_, R_, dst):
            fnp.matmul(_pview(L_, slice(None, k1, None)), fnp.swapaxes(_pview(R_, slice(None, k1, None)), -1, -2), out=T)
            fnp.sum(T, axis=0, out=dst[0])
            fnp.matmul(L_[k1], fnp.swapaxes(R_[k1], -1, -2), out=dst[1])
        fnp.subtract(X21, X11, out=Lb)
        fnp.add(Y11, Y12, out=Rb)
        prod(Lb, Rb, C22)
        fnp.add(X21, X22, out=Lb)
        prod(Lb, Y11, C21)
        fnp.subtract(C22, C21, out=C22)
        fnp.subtract(Y21, Y11, out=Rb)
        prod(X22, Rb, C11)
        fnp.add(C21, C11, out=C21)
        fnp.add(X11, X22, out=Lb)
        fnp.add(Y11, Y22, out=Rb)
        prod(Lb, Rb, Mb)
        fnp.add(C11, Mb, out=C11)
        fnp.add(C22, Mb, out=C22)
        fnp.subtract(Y12, Y22, out=Rb)
        prod(X11, Rb, C12)
        fnp.add(C22, C12, out=C22)
        fnp.add(X11, X12, out=Lb)
        prod(Lb, Y22, Mb)
        fnp.subtract(C11, Mb, out=C11)
        fnp.add(C12, Mb, out=C12)
        fnp.subtract(X12, X22, out=Lb)
        fnp.add(Y21, Y22, out=Rb)
        prod(Lb, Rb, Mb)
        fnp.add(C11, Mb, out=C11)
FMM_OFF = _os.environ.get('FMM_OFF', '0') == '1'
FMM_STRICT = _os.environ.get('FMM_STRICT', '0') == '1'
M2_FAIL_AFTER = int(_os.environ.get('M2_FAIL_AFTER', '-1'))
FMM_HINT_MM = int(_os.environ.get('FMM_HINT_MM', '8'))
FMM_HINT_HUB = int(_os.environ.get('FMM_HINT_HUB', '8'))
FMM_CAP_MM = int(_os.environ.get('FMM_CAP_MM', '8'))
M2_QC = _os.environ.get('M2_QC', '1') == '1'
LEAF_TY_ALIAS = _os.environ.get('LEAF_TY_ALIAS', '1') != '0'
LEAF_PACK = _os.environ.get('LEAF_PACK', '1') != '0'
# cst1 ZF7: the Z core keeps ONE 7-ary buffer per operand side (B0, 7^(L-1), 4, leaf) instead of one buffer per depth: a depth-d node's
# content lives at the {0,1,2,3}-positions of its sub-tree, its 7 Winograd children are formed IN PLACE (the three untouched quadrants
# X11/X12/X22 and B11/B21/B22 already sit at their child slot, X21/B12 are overwritten by the last combo that needs them), the leaf
# products are assembled in place the same way. Per level and side 4 add/sub instead of 4 add/sub + 3 copies; identical arithmetic
# (same operands, same order) -> bit-identical results, ~1.2% fewer billed FLOPs; the per-depth roles ('x', d<L-1) disappear.
ZF7 = _os.environ.get('ZF7', '1') == '1'
ZF7_COST = _os.environ.get('ZF7_COST', '0') == '1'   # 1: level-choice cost model without the copies (changes padded-level decisions -> rounding)
# engine agent ZF7_AB: Winograd in an alternative basis (Karstadt-Schwartz): 3 + 3 + 6 adds per level instead of 4 + 4 + 7, one in-place
# basis-change op per level and side on the root content (X: A21-slot := A21 + A22, Y: B12-slot := B22 - B12, O: C21-slot holds
# C22 - C21; leaf level: X5 / Y5 bases, two ops). Same plans, roles and level decisions; rounding-level change.
ZF7_AB = _os.environ.get('ZF7_AB', '1') == '1'
ZF7_AB_MIN = float(_os.environ.get('ZF7_AB_MIN', '2e6'))   # plans saving less per run keep the standard basis (+2 ops / run, ~35 views / plan)
# engine agent 2: HZ_AB = in-place classic Strassen in an alternative basis in the deferred kappa4 pullback engine (_HZE); ZSRC_CACHE = cached Z-order
# source views also for small registered operands (same key as the big-array cache); ZF7_AB_MERGE = merged inner-level ops of ZF7_AB
HZ_AB = _os.environ.get('HZ_AB', '1') == '1'
ZSRC_CACHE = _os.environ.get('ZSRC_CACHE', '1') == '1'
ZSRC_CACHE_CAP = int(_os.environ.get('ZSRC_CACHE_CAP', '2048'))
ZF7_AB_MERGE = _os.environ.get('ZF7_AB_MERGE', '1') == '1'
# opd_patch (agents_ideas2 L1, 2026-10-17): call / round-trip diet with identical values; OPD=0 -> the parent op stream
OPD = _os.environ.get('OPD', '1') == '1'
OPD_SWAP = OPD and _os.environ.get('OPD_SWAP', '1') == '1'   # _mm_prep (bx == 1): the unread swapaxes views Ys / Os are not formed
OPD_ZT = OPD and _os.environ.get('OPD_ZT', '1') == '1'       # Z entry / exit: split-only operand reshape, permutation on cached plan views
RRES = _os.environ.get('RRES', '1') == '1'         # rres_patch: batched plan views (fnp.unstack) + NLK pair views via _pview
RRES_ON = RRES                                      # (guard3 bool: off-suite predicts run the parent path)
RRES_J3 = _os.environ.get('RRES_J3', '1') == '1'   # rres_patch: J3_TRI pinv forms each view once (with RRES)
_RRES_OK = [True]                                   # cleared after an unstack failure (permanent parent fallback)
RRES_PREWARM_T = float(_os.environ.get('RRES_PREWARM_T', '0.4'))   # rres_patch: extra plan-view prewarm seconds after OPD_ZT (0 = off)
RRES_PREWARM_LIMIT = int(_os.environ.get('RRES_PREWARM_LIMIT', '16384'))   # (its max new views)
RRES_HOLD = _os.environ.get('RRES_HOLD', '1') == '1'   # rres_patch: Strassen run / leaf / HZE run keep their result proxies per call
OPD_NLK = OPD and _os.environ.get('OPD_NLK', '1') == '1'     # NLK_PK entry writes merged into arithmetic-progression runs
OPD_PK = OPD and _os.environ.get('OPD_PK', '1') == '1'       # PK2K vectors: shared pk1 powers, coefficient-1 terms unscaled
OPD_CSE = OPD and _os.environ.get('OPD_CSE', '1') == '1'     # sum(K4v) / sum(k22row) / their cI term shared (g_prev, gI_prev, K4_sigma)
OPD_ZT_SETUP_T = float(_os.environ.get('OPD_ZT_SETUP_T', '0.5'))   # (OPD_ZT) setup seconds for the plan views, after the prewarm
_OPD_PK_OK = {k_: v_ for k_, v_ in PK2K_TABLE.items() if len(k_) == 1} == {(1,): [(((1,),), 1.0)], (2,): [(((1,), (1,)), -1.0), (((2,),), 1.0)], (3,): [(((1,), (1,), (1,)), 2.0), (((1,), (2,)), -3.0), (((3,),), 1.0)], (4,): [(((1,), (1,), (1,), (1,)), -6.0), (((1,), (1,), (2,)), 12.0), (((1,), (3,)), -4.0), (((2,), (2,)), -3.0), (((4,),), 1.0)]}
_OPD_RUNS = {}


def _opd_runs(srcs):
    """OPD_NLK: partition of the merged-program entries e (sources srcs[e]: a b2 name or a pair index) into runs, each one multiply over
    strided views: (0, name, e0, e1, es, 0, 0, 0) = the same b2 matrix at e = e0, e0 + es, ... < e1; (1, None, e0, e1, es, p0, p1, ps) = pair
    products abbuf[E + p] with p and e advancing together (ps >= 1). Greedy longest arithmetic progression first; cached per program."""
    key = tuple(srcs)
    got = _OPD_RUNS.get(key)
    if got is not None:
        return got
    runs = []
    named = {}
    pairs = []
    for e, s in enumerate(key):
        if isinstance(s, str):
            named.setdefault(s, []).append(e)
        else:
            pairs.append((e, int(s)))
    for nm in sorted(named, key=lambda x: named[x][0]):
        left = list(named[nm])
        while left:
            best = [left[0]]
            for i in range(len(left)):
                for j in range(i + 1, len(left)):
                    st = left[j] - left[i]
                    seq = [left[i]]
                    while seq[-1] + st in left:
                        seq.append(seq[-1] + st)
                    if len(seq) > len(best):
                        best = seq
            es = best[1] - best[0] if len(best) > 1 else 1
            runs.append((0, nm, best[0], best[-1] + 1, es, 0, 0, 0))
            left = [x for x in left if x not in best]
    left = list(pairs)
    while left:
        best = [left[0]]
        for i in range(len(left)):
            for j in range(i + 1, len(left)):
                de, dp = left[j][0] - left[i][0], left[j][1] - left[i][1]
                if de <= 0 or dp <= 0:
                    continue
                seq = [left[i]]
                while (seq[-1][0] + de, seq[-1][1] + dp) in left:
                    seq.append((seq[-1][0] + de, seq[-1][1] + dp))
                if len(seq) > len(best):
                    best = seq
        es = best[1][0] - best[0][0] if len(best) > 1 else 1
        ps = best[1][1] - best[0][1] if len(best) > 1 else 1
        runs.append((1, None, best[0][0], best[-1][0] + 1, es, best[0][1], best[-1][1] + 1, ps))
        left = [x for x in left if x not in best]
    cover = sorted(e for r_ in runs for e in range(r_[2], r_[3], r_[4]))
    got = tuple(runs) if cover == list(range(len(key))) else ()
    _OPD_RUNS[key] = got
    return got
# engine agent 3: LX = one deeper Strassen level where the exact ZF7_AB cost says it pays and no padding is needed (L6 -> L7 for the
# 1024-wide families); ROLE_MERGE = one buffer per ZF7 role type instead of one per depth (a run only touches the roles of its depth)
LX = _os.environ.get('LX', '1') == '1'
LX_FROM = int(_os.environ.get('LX_FROM', '4'))
LX_LMAX = int(_os.environ.get('LX_LMAX', '7'))
LX_MINLEAF = int(_os.environ.get('LX_MINLEAF', '4'))
LX_GAIN_MIN = float(_os.environ.get('LX_GAIN_MIN', '2e7'))   # billed FLOPs per run; one more level costs ~11-15 ops
ROLE_MERGE = LX or _os.environ.get('ROLE_MERGE', '1') == '1'


def _rmerge(role):
    """ROLE_MERGE: the depth-indexed ZF7 roles of one type share one flat buffer (sized to the largest need)"""
    if ROLE_MERGE and ZF7 and isinstance(role, tuple) and len(role) == 2 and role[0] in ('x', 'y', 'hx', 'o'):
        return (role[0], 99)
    return role


def _lx_cost(kind, B0, sm, sk, sn, S, L, G, k1):
    """LX: billed FLOPs of one ZF7 run (leaf matmuls 2mnk - mn, ZF7_AB or Winograd adds, Z-order gathers); exact (ab_bench.py)"""
    t, q, p7 = ((7 ** L - 4 ** L) // 3, 4 ** (L - 1), 7 ** L)
    if kind == 'hub':
        ux, uy, uo = (sm * S * sk, sn * S * sk, sm * G * sn)
        mult = p7 * sm * sn * ((2 * k1 * sk - 1) + (2 * (S - k1) * sk - 1) if k1 else 2 * S * sk - 1)
    else:
        ux, uy, uo = (B0 * sm * sk, B0 * sk * S * sn, B0 * sm * S * sn)
        mult = p7 * B0 * sm * S * sn * (2 * sk - 1)
    if ZF7_AB and _ab_saving(kind, B0, sm, sk, sn, S, L, G) >= ZF7_AB_MIN:
        adds = (3 * t + (L + 1) * q) * (ux + uy) + (6 * t + L * q) * uo
    else:
        adds = t * (4 * ux + 4 * uy + 7 * uo)
    return mult + adds + 4 ** L * (ux + uy + uo)


def _lx_plans(n):
    """LX: data-independent L7 plan keys of the 1024-wide young transports (mm) and D21 hubs (2..8 slots), built in setup"""
    s = n >> 7
    return [('mm', 1, s, s, s, S, 7, 0) for S in (8, 6, 4, 2)] + [('hub', 1, s, s, s, S, 7, 0) for S in (8, 6, 4, 2)]


# engine agent round 3: HZ_L6 = one level deeper in the deferred kappa4 pullback (_HZE, needs HZ_AB) for the products whose rows and widths
# are multiples of 2^(HZ_LEV + 1), chosen per layer by the exact billed cost (second W tree counted)
HZ_L6 = _os.environ.get('HZ_L6', '1') == '1'
HZ6_GAIN_MIN = float(_os.environ.get('HZ6_GAIN_MIN', '1e7'))
HZ6_SETUP_T = float(_os.environ.get('HZ6_SETUP_T', '0.25'))
# LX_PAD = the LX deeper level also when it needs padding (same legality as the PAD candidates, exact cost incl. padded leaf + staging)
LX_PAD = _os.environ.get('LX_PAD', '1') == '1'
# LX_MULTI = further free LX steps while each pays (with LX_LMAX=8: L8 for the 1024-wide families; roles ~4.4 GB, see engine agent notes)
LX_MULTI = _os.environ.get('LX_MULTI', '0') == '1'


def _lx_plans8(n):
    """LX_MULTI: data-independent L8 plan keys of the 1024-wide young transports (8/6/4 slots) and D21 hubs (6/4 slots), built in setup"""
    s = n >> 8
    return [('mm', 1, s, s, s, S, 8, 0) for S in (8, 6, 4)] + [('hub', 1, s, s, s, S, 8, 0) for S in (6, 4)]


def _hz_cost(L, r, c, w):
    """HZ_AB: billed FLOPs of one in-place product of _HZE at level L, leaf r x c @ c x w (leaf matmuls, adds, gather / scatter copies)"""
    t, q = ((7 ** L - 4 ** L) // 3, 4 ** (L - 1))
    return 7 ** L * r * w * (2 * c - 1) + (3 * t + 2 * L * q) * r * c + (6 * t + 2 * L * q) * r * w + 4 ** L * (r * c + r * w)


def _hz_tree(L, c, w):
    """HZ_AB: billed FLOPs of set_right at level L (W digit reshape + gather copy + basis change and children of the W tree)"""
    t, q = ((7 ** L - 4 ** L) // 3, 4 ** (L - 1))
    return (3 * t + 2 * L * q) * c * w + 2 * 4 ** L * c * w


def _hz6_flags(lev, rows, hw1, hw0):
    """HZ_L6: per chunk of one layer True = run at level lev + 1 (rows and both widths divisible by 2^(lev + 1)), when the layer's billed
    cost (products + W trees; both trees when the layer mixes levels) drops by >= HZ6_GAIN_MIN; all False otherwise"""
    m6 = (1 << (lev + 1)) - 1
    if (hw1 & m6) or (hw0 & m6):
        return [False] * len(rows)
    c5, w5, c6, w6 = (hw1 >> lev, hw0 >> lev, hw1 >> (lev + 1), hw0 >> (lev + 1))
    elig = [(r & m6) == 0 for r in rows]
    if not any(elig):
        return [False] * len(rows)
    a = sum((_hz_cost(lev, r >> lev, c5, w5) for r in rows)) + _hz_tree(lev, c5, w5)
    b = sum((_hz_cost(lev + 1, r >> (lev + 1), c6, w6) if e else _hz_cost(lev, r >> lev, c5, w5) for r, e in zip(rows, elig)))
    b += _hz_tree(lev + 1, c6, w6) + (0 if all(elig) else _hz_tree(lev, c5, w5))
    return elig if b <= a - HZ6_GAIN_MIN else [False] * len(rows)


# HZGATE: (j, nr_j, ni_j, n3_j) of the pullback layers seen on 34 nets, in build order (value first): the 25 of nets 0..7 with the
# current SDT, then the 11 only seen in the SDT_C=2.5 logs of nets 8..33 (same live widths, larger windows / no window)
_HZG_PROF = ((0, 1024, 1024, 1024), (1, 1024, 1024, 1024), (2, 1024, 1024, 1024), (3, 1024, 1024, 1024), (4, 1024, 1024, 1024),
             (4, 1024, 1024, 960), (5, 1024, 1024, 960), (6, 960, 1024, 960), (6, 960, 1024, 896), (7, 960, 960, 896), (8, 960, 960, 896),
             (9, 928, 960, 896), (9, 928, 960, 832), (10, 960, 928, 896), (10, 928, 928, 832), (10, 928, 928, 896), (11, 928, 960, 832),
             (11, 928, 928, 832), (12, 896, 928, 832), (13, 896, 896, 832), (13, 896, 896, 768), (13, 928, 896, 832), (14, 896, 896, 832),
             (14, 896, 896, 768), (14, 896, 928, 832), (5, 1024, 1024, 1024), (6, 992, 1024, 960), (6, 992, 1024, 992), (7, 960, 960, 960),
             (7, 960, 992, 960), (8, 960, 960, 960), (9, 928, 960, 928), (11, 928, 928, 896), (12, 896, 928, 896), (13, 896, 896, 896),
             (14, 896, 896, 896))
_HZG_NCUR = 25
HZG_L6 = _os.environ.get('HZG_L6', '1') == '1'     # engine agent (hzgate2): HZ_L6 on the prebuilt layer products only (lazy-free)
# HZG_EXT: one window step (64) below the smallest window seen with the current SDT for each (j, nr_j, ni_j) of nets 0..7
_HZG_EXT = ((4, 1024, 1024, 896), (5, 1024, 1024, 896), (6, 960, 1024, 832), (7, 960, 960, 832), (8, 960, 960, 832), (9, 928, 960, 768),
            (10, 928, 928, 768), (10, 960, 928, 832), (11, 928, 928, 768), (11, 928, 960, 768), (12, 896, 928, 768), (13, 896, 896, 704),
            (13, 928, 896, 768), (14, 896, 896, 704), (14, 896, 928, 768))


def _ab_saving(kind, B0, sm, sk, sn, S, L, G):
    """ZF7_AB: modeled billed-FLOP saving of one run of a plan: T(L) - (L+1) 4^(L-1) additions per X / Y leaf-block element and
    T(L) - L 4^(L-1) per output element, T(L) = (7^L - 4^L) / 3 (matches the meter exactly, see agents_engine/ab_bench.py)"""
    t, q = ((7 ** L - 4 ** L) // 3, 4 ** (L - 1))
    if kind == 'hub':
        ux, uy, uo = (sm * S * sk, sn * S * sk, sm * G * sn)
    else:
        ux, uy, uo = (sm * sk, sk * S * sn, sm * S * sn)
    return B0 * ((t - (L + 1) * q) * (ux + uy) + (t - L * q) * uo)
# cst3 QRS: frozen-join factors FA_j = Qp^+ A_j with the pseudo-inverse Qp^+ = Rq^-1 Qn^T formed ONCE (one solve with kj right-hand
# sides) and used as the left operand of the Qn^T [A_j | P_j] family, instead of two solves Rq^-1 (Qn^T A_j), Rq^-1 (Qn^T P_j)
# (saves one general solve 2k^3/3 + 2k^2 n per frozen join; same algebra up to float associativity).
QRS = _os.environ.get('QRS', '1') == '1'
# cst2 DSF: algebraic folds in the (k, n, n) elementwise work of _dslices (rounding-level changes only): the 2/3 of the M-leg term is
# folded into the thin leg L and the (k,1,n) vectors (MP' = 2/3 M*P), D3's A*A*w2 term reuses the AA*w2 left factor (2-operand einsum
# instead of P*w2 + 3-operand einsum), R / PPL are formed while A*P / P*P are still alive (no recompute after the feedback block), the
# 1/3 of the feedback terms is folded into R1T (Xt/3) so the three scalar multiplies vanish. Same algebra up to float associativity.
# DSF=1: all folds (rounding-level changes: predictions move ~3e-6 relative, raw MSE +0.08% / -0.35% on nets 1 / 0 vs sdt5z, cb x0.9980);
# DSF=2 (default): only the exact part (R / PPL formed while A*P and P*P are alive, so the recompute after the feedback block disappears:
# same products in the same order -> bit-identical); DSF=0: sdt5z code.
DSF_MODE = int(_os.environ.get('DSF', '1'))
DSF = DSF_MODE == 1
PUBLIC_Z_VIEWS = _os.environ.get('PUBLIC_Z_VIEWS', '1') == '1'
AS_TAG_OFF = _os.environ.get('AS_TAG_OFF', '1') == '1'   # agents_robust: no symmetric tag on the covariance C (exact; -7 nr^2 FLOPs per tag)
PUBLIC_Z_VIEW_LIMIT = int(_os.environ.get('PUBLIC_Z_VIEW_LIMIT', '49152' if SDT_ON else '16384'))   # SDT: more distinct plans per worker (15.5k views after 12 MLPs); a wipe rebuilds every plan's views inside one predict
PUBLIC_Z_PREWARM = _os.environ.get('PUBLIC_Z_PREWARM', '1') == '1'
HZ_TMAX = float(_os.environ.get('HZ_TMAX', '105'))       # agents_robust: skip _hz_finish once the predict has run > HZ_TMAX s (0 = off)
HZ_TMAX_L = float(_os.environ.get('HZ_TMAX_L', '100'))   # agents_robust: per-layer guard: no HZ work (nor _hz_finish) once > HZ_TMAX_L s (0 = off)
HZ_TMAX_P = float(_os.environ.get('HZ_TMAX_P', '112'))   # agents_robust: abort the _hz_finish pullback (no-HZ result) once > HZ_TMAX_P s (0 = off)
import time as _rb_time
PUBLIC_Z_PREWARM_T = float(_os.environ.get('PUBLIC_Z_PREWARM_T', '1.0'))
PUBLIC_Z_PREWARM_LIMIT = int(_os.environ.get('PUBLIC_Z_PREWARM_LIMIT', '6144'))
PUBLIC_Z_SETUP_MAX_T = float(_os.environ.get('PUBLIC_Z_SETUP_MAX_T', '2.6'))
FACTOR_POOL_CAP = _os.environ.get('FACTOR_POOL_CAP', '1') != '0'
M2_PREALLOC = _os.environ.get('M2_PREALLOC', '1') == '1'
M2_PREALLOC_T = float(_os.environ.get('M2_PREALLOC_T', '2.5'))
PAD_DEEP = _os.environ.get('PAD_DEEP', '1') == '1'
PAD_LMAX = int(_os.environ.get('PAD_LMAX', '6'))
PAD_GAIN_MIN = float(_os.environ.get('PAD_GAIN_MIN', '1e8'))
PAD_DEBUG = _os.environ.get('PAD_DEBUG', '0') == '1'
PAD_KINDS = frozenset(_os.environ.get('PAD_KINDS', 'mm,free,hub').split(','))
PAD_SMAX = int(_os.environ.get('PAD_SMAX', '9'))
PAD_MM_SMAX = int(_os.environ.get('PAD_MM_SMAX', '99'))
PAD_RESERVE = _os.environ.get('PAD_RESERVE', '1') == '1'
PAD_PLANS = _os.environ.get('PAD_PLANS', '0') == '1'
_PAD_NATIVE = [None]
PAD_SETUP_LOG = _os.environ.get('PAD_SETUP_LOG', '')
_SETUP_T = {}

def _setup_mark(name, t0):
    """debug only (PAD_SETUP_LOG=<file>): elapsed seconds of the setup phases"""
    if PAD_SETUP_LOG:
        import time as _time
        _SETUP_T[name] = round(_time.perf_counter() - t0, 4)
PAD_LOG = []

def _deep_roles(n, L):
    """Z-core role sizes (elements) of the largest level-L plans the padded deeper recursion can request at width n (hub batch
    <= PAD_SMAX slots incl. the grouped slot, plain batch <= FMM_CAP_MM): reserved in setup so that no steady-state call grows
    a role (a growth clears every plan and its public views, ~1 s of residual to rebuild them)."""
    need = {}
    s = n >> L
    for kind, B0, S, G, sm, sk, sn, LL in (('hub', 1, PAD_SMAX, 2, s, s, s, L), ('mm', 1, FMM_CAP_MM, 1, s, s, s, L), ('mm', 1, 14, 1, n >> 5, 256 >> 5, n >> 5, 5), ('mm', 3, 1, 1, n >> 6, n >> 5, n >> 6, 5)):
        lay = ZFMM._layout(kind, B0, sm, sk, sn, S, LL, G)
        for v in lay.values():
            for role, shape in v if isinstance(v, list) else [v]:
                if isinstance(role, tuple) and role[0] == 'leaf_out':
                    role, extra = (('hx', role[1]), role[2])
                else:
                    extra = 0
                role = 'ty' if role == 'tm' and LEAF_TY_ALIAS else role
                role = 'tx' if role == 'tp' else role
                role = _rmerge(role)
                need[role] = max(need.get(role, 0), math.prod(shape) + extra)
    return need

def _zf7_roles():
    """ZF7 role sizes reserved in setup: the 7-ary layout of every spec plan (with the batch hint, as _plan reserves it) and the old
    envelope of the leaf-level roles (every plan that fitted the old reservation still fits; the roles of the inner depths are gone)."""
    need = {}

    def add(lay):
        for v in lay.values():
            for role, shape in v if isinstance(v, list) else [v]:
                extra = 0
                if isinstance(role, tuple) and role[0] == 'leaf_out':
                    role, extra = (('hx', role[1]), role[2])
                role = 'ty' if role == 'tm' and LEAF_TY_ALIAS else role
                role = 'tx' if role == 'tp' else role
                role = _rmerge(role)
                need[role] = max(need.get(role, 0), math.prod(shape) + extra)
    for key in _POOL_SPEC['plans'] + (_lx_plans(1024) if LX else []) + (_lx_plans8(1024) if LX and LX_MULTI and LX_LMAX >= 8 else []):
        kind, B0, sm, sk, sn, S, L, k1 = key
        Sr = max(S, FMM_HINT_HUB if kind == 'hub' else FMM_HINT_MM) if B0 == 1 else S
        for SS in sorted({S, Sr}):
            add(ZFMM._layout(kind, B0, sm, sk, sn, SS, L, 2 if k1 else 1))
    for role, n in _POOL_SPEC['roles']:
        role = _tup(role)
        if role in ('tx', 'ty') or (isinstance(role, tuple) and role[1] >= 3):
            need[_rmerge(role)] = max(need.get(_rmerge(role), 0), int(n))
    return sorted(need.items(), key=str)

def _deep_plans(n):
    """Data-independent plan keys of the padded deeper recursion at width n, created in setup after the spec plans (a plan created
    lazily inside predict costs ~25 billed reshapes and ~200 view creations of residual in the remote client): the padded young
    transports (L6, leaves 14..16, 8/6/4 slots) and the tier-2 forming families (L5, 7-wide contraction)."""
    keys = []
    s = n >> 6
    for S in (8, 6, 4):
        for sm in (s - 2, s - 1, s):
            for sk in (s - 2, s - 1, s):
                for sn in (s - 2, s - 1, s):
                    keys.append(('mm', 1, sm, sk, sn, S, 6, 0))
    for S in (14, 12, 10, 8, 6):
        for sm in (28, 29, 30, 31, 32, 24, 27, 26, 25):
            keys.append(('mm', 1, sm, 7, n >> 5, S, 5, 0))
    return keys

def _pad_report():
    """PAD_DEBUG: one line per distinct level decision of the last predict (stderr), then the log is cleared"""
    import sys as _sys
    agg = {}
    for rec in PAD_LOG:
        kind, bx, by, m, kd, w, lev, mn, L0, L, m2, kd2, w2, staged, c0, c = rec
        key = (kind, bx, by, m, kd, w, L0, L, m2, kd2, w2, staged)
        e = agg.setdefault(key, [0, 0.0, 0.0])
        e[0] += 1
        e[1] += c0
        e[2] += c
    tot0 = sum(e[1] for e in agg.values())
    tot = sum(e[2] for e in agg.values())
    print('PAD decisions: %d calls, modeled %.4e -> %.4e (%.3f%%)' % (len(PAD_LOG), tot0, tot, 100.0 * (tot - tot0) / max(tot0, 1.0)), file=_sys.stderr)
    for key, e in sorted(agg.items(), key=lambda kv: kv[1][1] - kv[1][2], reverse=True):
        kind, bx, by, m, kd, w, L0, L, m2, kd2, w2, staged = key
        print('  %-4s bx=%d S=%2d (%4d,%4d,%4d) L%d -> L%d (%4d,%4d,%4d)%s x%d: %.3e -> %.3e (%+.2f%%)' % (kind, bx, by, m, kd, w, L0, L, m2, kd2, w2, ' staged' if staged else '', e[0], e[1], e[2], 100.0 * (e[2] - e[1]) / max(e[1], 1.0)), file=_sys.stderr)
    PAD_LOG.clear()
_POOL_SPEC = {"pool": [["get", "wd", [1024, 1024]], ["get", "l", [15, 1024, 10]], ["get", "t1", [1024, 1024]], ["get", "hub", [1024, 1024]], ["get", "d21inner", [1, 1024, 1024]], ["get", "d21x", [31, 1, 1024, 1024]], ["get", "pk2", [3, 1024, 1024]], ["get", "pk11", [1024, 1024]], ["get", "d21new", [1024, 1024]], ["get", "k21", [1024, 1024]], ["get", "c", [1024, 1024]], ["get", "s21", [1024, 1024]], ["get", "ssep", [1024, 1024]], ["get", "om_r", [1024, 8]], ["get", "y", [1024, 8]], ["get", "y2", [1024, 8]], ["get", "bm", [8, 1024]], ["get", "rr", [1024, 10]], ["get", ["z", 0], [15, 1024, 10]], ["get", "ww", [1024, 1024]], ["get", "wk4m", [1024, 1024]], ["get", "lot_y", [1024, 12]], ["get", "lot_y2", [1024, 12]], ["get", "lot_gu", [15, 1024, 12]], ["get", "qu", [1024, 224]], ["get", "d21w", [1024, 1024]], ["get", "rep21", [1024, 1024]], ["get", ["z", 1], [15, 1024, 10]], ["get", "cpre", [1024, 1024]], ["get", ["qc", 1], [1024, 448]], ["get", ["zf", 0], [15, 1024, 8]], ["get", "xtb", [1024, 1024]], ["get", "ytb", [1024, 1024]], ["get", "x1b", [1024, 1024]], ["get", "y1b", [1024, 1024]], ["get", "omf", [1024, 4]], ["get", "r1t", [15, 1024, 4]], ["get", "r2t", [15, 1024, 4]], ["get", "sym4_x", [10, 1, 256, 1024]], ["get", "sym4_y", [10, 1, 1024, 256]], ["get", "sym4_o", [10, 1, 256, 256]], ["get", ["zf", 1], [15, 1024, 8]], ["get", "c7_wc", [1024, 1024]], ["get", "inner", [1, 1024, 448]], ["get", ["qc", 0], [1024, 448]], ["get", "qp", [1024, 448]], ["get", "thin_o", [1024, 1024]], ["get", "dn_lc", [1, 1, 1024, 288]], ["get", "dn_rc", [1, 1, 1024, 288]], ["get", "dn_oc", [1, 1024, 1024]], ["get", "dn_zi", [1, 1, 1024, 288]], ["get", "dn_zo", [1, 1, 1024, 288]], ["get", "inner2", [1, 1024, 224]], ["pair", ["fap2", 0], [15, 2, 224, 1024]], ["get", "ap", [15, 1024, 1024]], ["get", "pp", [15, 1024, 1024]], ["get", "t", [15, 1024, 1024]], ["get", "mp", [15, 1024, 1024]], ["get", "xt", [15, 1024, 1024]], ["get", "yt", [15, 1024, 1024]], ["get", "u", [15, 1024, 1024]], ["get", "ppl", [15, 1024, 10]], ["get", "d21", [1024, 1024]], ["get", "d21fb", [1024, 1024]], ["get", "gyr", [15, 1024, 4]], ["get", "gxr", [15, 1024, 4]], ["get", ["stage_x", 1024], [1, 1, 1024, 1024]], ["get", "au_t1x", [9, 1, 1024, 1024]], ["get", "sym3_x", [6, 1, 320, 1024]], ["get", "sym3_y", [6, 1, 1024, 320]], ["get", "sym3_o", [6, 1, 320, 320]], ["get", "dap", [2, 1, 1, 1024]], ["get", "abbuf", [52, 1024, 1024]], ["get", "lot_slab", [15, 3, 1024]], ["pair", "AP0", [15, 2, 1024, 1024]], ["pair", "AP1", [15, 2, 1024, 1024]], ["get", "om", [1024, 448]], ["get", "yq", [1024, 448]], ["get", "yq1", [1024, 448]], ["get", "yq2", [1024, 448]], ["get", "y1t", [2, 1, 448, 1024]], ["get", "yqh", [1, 1024, 448]], ["pair", ["fap", 1], [15, 2, 448, 1024]], ["get", "fg", [448, 2048]], ["get", "lot_t", [1024, 12]], ["get", "dn_g0", [1024, 1024]], ["get", "dn_fsub", [448, 224]], ["get", "dn_frot", [2, 1, 448, 1024]], ["get", "dn_qnv", [1024, 448]]], "roles": [["tx", 60236288], ["ty", 60236288], [["hx", 99], 271063296], [["x", 99], 30118144], [["y", 99], 240945152]], "bmax": 30, "plans": [["mm", 1, 8, 8, 8, 8, 7, 0], ["hub", 1, 7, 8, 7, 8, 7, 0], ["mm", 1, 8, 8, 8, 6, 7, 0], ["hub", 1, 8, 8, 8, 6, 7, 0], ["mm", 1, 15, 15, 16, 8, 6, 0], ["mm", 1, 8, 8, 8, 4, 7, 0], ["hub", 1, 8, 8, 8, 4, 7, 0], ["mm", 1, 14, 7, 16, 8, 6, 0], ["mm", 1, 14, 13, 13, 6, 6, 0], ["mm", 1, 8, 8, 8, 2, 7, 0], ["hub", 1, 8, 8, 8, 2, 7, 0], ["hub", 1, 10, 14, 7, 9, 6, 0], ["mm", 1, 15, 7, 16, 4, 6, 0], ["mm", 1, 14, 13, 14, 2, 6, 0], ["mm", 1, 16, 7, 16, 2, 6, 0], ["mm", 10, 8, 32, 8, 1, 5, 0], ["mm", 1, 7, 13, 16, 2, 6, 0], ["mm", 1, 7, 12, 16, 2, 6, 0], ["mm", 10, 7, 29, 7, 1, 5, 0], ["mm", 6, 10, 32, 10, 1, 5, 0], ["mm", 1, 7, 11, 14, 2, 6, 0], ["mm", 6, 10, 30, 10, 1, 5, 0], ["mm", 1, 15, 15, 7, 1, 6, 0], ["mm", 1, 15, 14, 7, 1, 6, 0], ["mm", 1, 13, 7, 14, 1, 6, 0], ["mm", 1, 14, 13, 7, 1, 6, 0], ["mm", 1, 27, 7, 32, 2, 5, 0], ["mm", 1, 12, 7, 14, 1, 6, 0], ["mm", 1, 10, 7, 13, 1, 6, 0], ["mm", 1, 14, 14, 32, 2, 5, 0], ["mm", 1, 29, 28, 14, 1, 5, 0], ["mm", 1, 14, 28, 28, 1, 5, 0], ["mm", 10, 29, 120, 29, 1, 3, 0], ["mm", 1, 16, 32, 16, 1, 5, 0], ["mm", 1, 60, 4, 64, 1, 4, 0], ["mm", 1, 14, 14, 64, 2, 4, 0], ["mm", 1, 14, 14, 60, 2, 4, 0], ["mm", 1, 12, 128, 128, 1, 3, 0], ["mm", 1, 120, 8, 120, 1, 3, 0], ["mm", 1, 16, 120, 120, 1, 3, 0], ["mm", 1, 20, 116, 116, 1, 3, 0], ["mm", 1, 12, 120, 120, 1, 3, 0], ["mm", 1, 16, 116, 116, 1, 3, 0], ["mm", 1, 20, 112, 112, 1, 3, 0], ["mm", 1, 112, 104, 21, 1, 3, 0], ["mm", 1, 116, 112, 14, 1, 3, 0], ["mm", 1, 28, 28, 14, 1, 4, 0], ["mm", 1, 15, 14, 15, 8, 6, 0], ["mm", 1, 24, 7, 32, 12, 5, 0], ["hub", 1, 12, 16, 7, 6, 6, 0], ["hub", 1, 13, 16, 7, 4, 6, 0], ["hub", 1, 20, 32, 7, 6, 5, 0], ["mm", 1, 7, 12, 15, 2, 6, 0], ["mm", 10, 7, 28, 7, 1, 5, 0], ["hub", 1, 22, 32, 7, 2, 5, 0], ["hub", 1, 56, 14, 48, 1, 4, 0], ["mm", 1, 56, 52, 13, 1, 4, 0], ["mm", 1, 56, 52, 12, 1, 4, 0], ["hub", 1, 116, 21, 104, 1, 3, 0], ["mm", 1, 112, 8, 104, 1, 3, 0], ["mm", 1, 112, 112, 8, 1, 3, 0], ["hub", 1, 5, 7, 6, 8, 7, 0], ["mm", 1, 15, 14, 14, 8, 6, 0], ["mm", 1, 14, 13, 14, 8, 6, 0], ["mm", 1, 15, 14, 15, 6, 6, 0], ["mm", 1, 15, 14, 14, 6, 6, 0], ["mm", 1, 12, 7, 13, 9, 6, 0], ["hub", 1, 11, 16, 7, 8, 6, 0], ["hub", 1, 9, 13, 7, 9, 6, 0], ["hub", 1, 9, 14, 12, 4, 6, 0], ["mm", 1, 15, 14, 16, 2, 6, 0], ["hub", 1, 9, 13, 12, 4, 6, 0], ["mm", 1, 15, 14, 15, 2, 6, 0], ["hub", 1, 18, 32, 7, 10, 5, 0], ["mm", 1, 27, 7, 32, 6, 5, 0], ["hub", 1, 14, 16, 7, 2, 6, 0], ["mm", 1, 11, 7, 13, 1, 6, 0], ["mm", 1, 9, 7, 12, 1, 6, 0], ["mm", 10, 29, 116, 29, 1, 3, 0], ["hub", 1, 56, 13, 48, 1, 4, 0], ["mm", 1, 56, 6, 56, 1, 4, 0], ["mm", 1, 116, 8, 120, 1, 3, 0], ["hub", 1, 116, 17, 104, 1, 3, 0], ["mm", 1, 15, 13, 14, 8, 6, 0], ["mm", 1, 12, 7, 15, 8, 6, 0], ["hub", 1, 12, 16, 7, 8, 6, 0], ["mm", 1, 15, 7, 16, 6, 6, 0], ["hub", 1, 12, 15, 13, 6, 6, 0], ["hub", 1, 11, 14, 7, 9, 6, 0], ["hub", 1, 12, 16, 13, 4, 6, 0], ["hub", 1, 12, 15, 13, 4, 6, 0], ["mm", 1, 24, 7, 32, 10, 5, 0], ["hub", 1, 18, 32, 7, 12, 5, 0], ["hub", 1, 12, 16, 13, 2, 6, 0], ["mm", 1, 7, 16, 16, 2, 6, 0], ["hub", 1, 16, 16, 7, 2, 6, 0], ["mm", 1, 27, 7, 32, 4, 5, 0], ["mm", 1, 16, 16, 7, 1, 6, 0], ["mm", 1, 29, 26, 14, 1, 5, 0], ["mm", 1, 32, 14, 14, 1, 5, 0], ["hub", 1, 56, 12, 52, 1, 4, 0], ["hub", 1, 120, 14, 112, 1, 3, 0], ["hub", 1, 116, 19, 104, 1, 3, 0], ["mm", 1, 116, 8, 112, 1, 3, 0], ["hub", 1, 13, 16, 13, 8, 6, 0], ["mm", 1, 15, 7, 16, 8, 6, 0], ["mm", 1, 14, 13, 13, 8, 6, 0], ["mm", 1, 13, 7, 13, 9, 6, 0], ["mm", 1, 12, 7, 14, 8, 6, 0], ["mm", 1, 15, 15, 16, 4, 6, 0], ["mm", 1, 15, 15, 15, 4, 6, 0], ["hub", 1, 10, 15, 7, 8, 6, 0], ["hub", 1, 9, 13, 12, 6, 6, 0], ["mm", 1, 20, 7, 32, 12, 5, 0], ["mm", 1, 26, 7, 32, 8, 5, 0], ["mm", 1, 24, 7, 32, 8, 5, 0], ["hub", 1, 18, 32, 7, 8, 5, 0], ["hub", 1, 9, 14, 12, 2, 6, 0], ["hub", 1, 22, 32, 7, 4, 5, 0], ["hub", 1, 20, 32, 7, 4, 5, 0], ["mm", 1, 11, 7, 14, 1, 6, 0], ["mm", 1, 9, 7, 13, 1, 6, 0], ["mm", 1, 58, 6, 56, 1, 4, 0], ["mm", 1, 116, 112, 19, 1, 3, 0], ["mm", 1, 116, 112, 17, 1, 3, 0], ["mm", 1, 116, 104, 19, 1, 3, 0], ["mm", 1, 116, 120, 8, 1, 3, 0], ["hub", 1, 15, 16, 15, 8, 6, 0], ["hub", 1, 13, 16, 14, 8, 6, 0], ["mm", 1, 13, 7, 15, 8, 6, 0], ["mm", 1, 14, 7, 16, 6, 6, 0], ["mm", 1, 10, 7, 14, 8, 6, 0], ["hub", 1, 11, 14, 13, 6, 6, 0], ["hub", 1, 11, 14, 12, 6, 6, 0], ["hub", 1, 9, 15, 7, 8, 6, 0], ["mm", 1, 21, 7, 32, 12, 5, 0], ["hub", 1, 11, 15, 13, 4, 6, 0], ["hub", 1, 11, 15, 12, 4, 6, 0], ["hub", 1, 11, 14, 13, 4, 6, 0], ["hub", 1, 11, 14, 12, 4, 6, 0], ["hub", 1, 10, 14, 12, 4, 6, 0], ["hub", 1, 20, 32, 7, 8, 5, 0], ["hub", 1, 11, 15, 13, 2, 6, 0], ["mm", 1, 7, 15, 16, 2, 6, 0], ["hub", 1, 11, 15, 12, 2, 6, 0], ["hub", 1, 15, 16, 7, 2, 6, 0], ["mm", 1, 7, 11, 15, 2, 6, 0], ["mm", 1, 16, 15, 7, 1, 6, 0], ["mm", 1, 14, 7, 15, 1, 6, 0], ["mm", 1, 14, 7, 14, 1, 6, 0], ["mm", 1, 30, 14, 14, 1, 5, 0], ["mm", 1, 64, 4, 64, 1, 4, 0], ["hub", 1, 56, 12, 48, 1, 4, 0], ["mm", 1, 56, 6, 52, 1, 4, 0], ["mm", 1, 56, 58, 6, 1, 4, 0], ["hub", 1, 120, 14, 104, 1, 3, 0], ["mm", 1, 112, 8, 112, 1, 3, 0], ["mm", 1, 84, 112, 8, 1, 3, 0], ["hub", 1, 8, 8, 7, 8, 7, 0], ["hub", 1, 5, 7, 6, 6, 7, 0], ["mm", 1, 10, 12, 13, 10, 6, 0], ["mm", 1, 15, 15, 15, 6, 6, 0], ["hub", 1, 12, 15, 13, 8, 6, 0], ["mm", 1, 14, 7, 14, 9, 6, 0], ["mm", 1, 13, 7, 16, 8, 6, 0], ["mm", 1, 14, 7, 15, 8, 6, 0], ["mm", 1, 11, 13, 13, 8, 6, 0], ["mm", 1, 15, 13, 14, 6, 6, 0], ["mm", 1, 11, 7, 15, 8, 6, 0], ["hub", 1, 11, 15, 13, 6, 6, 0], ["hub", 1, 9, 13, 11, 8, 6, 0], ["hub", 1, 8, 13, 11, 8, 6, 0], ["hub", 1, 9, 14, 7, 8, 6, 0], ["hub", 1, 13, 16, 13, 4, 6, 0], ["mm", 1, 26, 7, 32, 10, 5, 0], ["hub", 1, 13, 15, 13, 4, 6, 0], ["hub", 1, 8, 14, 7, 8, 6, 0], ["hub", 1, 9, 13, 11, 6, 6, 0], ["hub", 1, 10, 13, 12, 4, 6, 0], ["mm", 1, 15, 15, 16, 2, 6, 0], ["hub", 1, 16, 32, 7, 12, 5, 0], ["mm", 1, 15, 13, 15, 2, 6, 0], ["mm", 1, 25, 7, 32, 6, 5, 0], ["hub", 1, 11, 16, 13, 2, 6, 0], ["mm", 1, 11, 13, 14, 2, 6, 0], ["hub", 1, 10, 15, 12, 2, 6, 0], ["mm", 1, 28, 7, 32, 4, 5, 0], ["hub", 1, 13, 16, 7, 2, 6, 0], ["hub", 1, 9, 14, 11, 2, 6, 0], ["mm", 1, 7, 12, 14, 2, 6, 0], ["mm", 6, 10, 29, 10, 1, 5, 0], ["mm", 1, 7, 11, 13, 2, 6, 0], ["mm", 1, 12, 7, 13, 1, 6, 0], ["mm", 1, 10, 13, 7, 1, 6, 0], ["mm", 1, 10, 7, 12, 1, 6, 0], ["mm", 1, 10, 12, 7, 1, 6, 0], ["mm", 1, 8, 7, 12, 1, 6, 0], ["mm", 1, 21, 26, 14, 1, 5, 0], ["mm", 1, 42, 52, 14, 1, 4, 0], ["mm", 1, 40, 52, 14, 1, 4, 0], ["mm", 1, 40, 48, 14, 1, 4, 0], ["mm", 1, 40, 8, 52, 1, 4, 0], ["hub", 1, 116, 17, 112, 1, 3, 0], ["mm", 1, 120, 112, 17, 1, 3, 0], ["mm", 1, 116, 104, 17, 1, 3, 0], ["mm", 1, 11, 12, 13, 10, 6, 0], ["mm", 1, 10, 13, 13, 10, 6, 0], ["mm", 1, 14, 12, 13, 8, 6, 0], ["mm", 1, 12, 13, 13, 8, 6, 0], ["mm", 1, 10, 13, 13, 8, 6, 0], ["hub", 1, 9, 14, 12, 8, 6, 0], ["mm", 1, 11, 7, 14, 8, 6, 0], ["hub", 1, 9, 13, 12, 8, 6, 0], ["mm", 1, 10, 7, 13, 9, 6, 0], ["hub", 1, 10, 16, 7, 8, 6, 0], ["mm", 1, 15, 14, 16, 4, 6, 0], ["mm", 1, 15, 14, 15, 4, 6, 0], ["mm", 1, 26, 7, 32, 12, 5, 0], ["hub", 1, 9, 14, 7, 9, 6, 0], ["hub", 1, 10, 13, 7, 9, 6, 0], ["hub", 1, 13, 16, 7, 6, 6, 0], ["hub", 1, 13, 16, 14, 4, 6, 0], ["mm", 1, 23, 7, 32, 12, 5, 0], ["hub", 1, 8, 15, 7, 8, 6, 0], ["hub", 1, 10, 13, 12, 6, 6, 0], ["hub", 1, 13, 15, 14, 4, 6, 0], ["mm", 1, 27, 7, 32, 10, 5, 0], ["hub", 1, 8, 13, 7, 9, 6, 0], ["hub", 1, 8, 13, 12, 6, 6, 0], ["hub", 1, 12, 14, 13, 4, 6, 0], ["hub", 1, 20, 32, 7, 12, 5, 0], ["hub", 1, 10, 15, 12, 4, 6, 0], ["hub", 1, 12, 16, 7, 4, 6, 0], ["hub", 1, 20, 32, 7, 10, 5, 0], ["hub", 1, 16, 32, 7, 10, 5, 0], ["mm", 1, 12, 13, 14, 2, 6, 0], ["hub", 1, 16, 32, 7, 8, 5, 0], ["mm", 1, 10, 13, 14, 2, 6, 0], ["hub", 1, 10, 14, 12, 2, 6, 0], ["hub", 1, 18, 32, 7, 6, 5, 0], ["mm", 1, 25, 7, 32, 4, 5, 0], ["hub", 1, 8, 14, 12, 2, 6, 0], ["mm", 1, 13, 7, 15, 1, 6, 0], ["mm", 1, 14, 12, 7, 1, 6, 0], ["hub", 1, 20, 32, 7, 2, 5, 0], ["mm", 10, 29, 112, 29, 1, 3, 0], ["mm", 1, 23, 26, 14, 1, 5, 0], ["mm", 1, 21, 24, 14, 1, 5, 0], ["hub", 1, 58, 13, 52, 1, 4, 0], ["hub", 1, 56, 14, 52, 1, 4, 0], ["hub", 1, 56, 13, 52, 1, 4, 0], ["mm", 1, 58, 52, 12, 1, 4, 0], ["mm", 1, 56, 8, 52, 1, 4, 0], ["mm", 1, 56, 48, 13, 1, 4, 0], ["mm", 1, 46, 52, 14, 1, 4, 0], ["mm", 1, 42, 48, 14, 1, 4, 0], ["mm", 1, 42, 8, 52, 1, 4, 0], ["hub", 1, 120, 19, 112, 1, 3, 0], ["hub", 1, 120, 19, 104, 1, 3, 0], ["hub", 1, 116, 21, 96, 1, 3, 0], ["hub", 1, 116, 19, 96, 1, 3, 0], ["mm", 1, 92, 8, 104, 1, 3, 0], ["mm", 1, 92, 112, 8, 1, 3, 0]]}

def _tup(x):
    return tuple((_tup(v) for v in x)) if isinstance(x, list) else x
FMM_OLD_ZCORE = _os.environ.get('FMM_OLD_ZCORE', '1') == '1'
_S = slice(None)
import os as _os_dbg
DEBUG = _os_dbg.environ.get('ZFMM_DEBUG', '0') == '1'

class _Plan:
    pass

class ZFMM:

    def __init__(self, dtype=fnp.float32, minblk=32, hint_mm=8, hint_hub=8, cap_mm=None):
        self.dtype = dtype
        self.minblk = int(minblk)
        self.mn = int(minblk)
        self.hint_mm = int(hint_mm)
        self.hint_hub = int(hint_hub)
        self.cap_mm = int(hint_mm if cap_mm is None else cap_mm)
        self.flat = {}
        self.public_setup_slices = None
        self.plans = {}
        self.min_cache_bytes = 8 << 20
        self.mark = False
        self.steady_keys = set()
        self.zb = {}
        self.vc = {}
        self.public_cache_inputs = {}
        self.public_zview_roots = {}
        self.public_zview_count = 0
        self.public_z_prewarm_s = 0.0
        self.public_z_prewarm_eligible = 0
        self.public_z_prewarm_touched = 0
        self.public_z_prewarm_new_views = 0
        self.public_z_prewarm_ready_plans = 0
        self.deep = 0
        self.spool = None
        self.stage_dirty = {}
        self.cache_roots = set()
        self.choices = {}
        self.role_epoch = 0

    @staticmethod
    def _cost(kind, bx, by, m, kd, w, L, k1=0):
        """Billed FLOPs of one run at level L on (m, kd, w) (mirrors _run / _leaf: matmul 2mnk - mn per leaf GEMM, one unit per
        element for the Winograd adds / copies of the transforms, the leaf and the assembly, and the three Z-order gathers).
        Checked against the meter on eight family shapes (error < 0.05 %)."""
        hub = kind == 'hub'
        S = by
        Bx = S if hub else bx
        Bo = (2 if k1 else 1) if hub else by
        if L <= 0:
            return S * m * w * (2 * kd - 1) + ((S - 1) * m * w if hub else 0)
        sm, sk, sn = (m >> L, kd >> L, w >> L)
        P = 7 ** L
        if not hub:
            mult = P * by * sm * sn * (2 * sk - 1)
        elif k1:
            mult = P * sm * sn * (2 * k1 * sk - 1 + (2 * (S - k1) * sk - 1))
        else:
            mult = P * sm * sn * (2 * S * sk - 1)
        top = 1.75 ** (L - 1)
        geo = (top - 1.0) / 0.75
        pre = 4.0 if (ZF7 and ZF7_COST) else 7.0   # ZF7: no quadrant copies at the inner levels (4 add/sub per side and level)
        ax = (pre * geo + 4.0 * top) * (m * kd * Bx) / 4.0
        ay = (pre * geo + 4.0 * top) * (kd * w * S) / 4.0
        ac = (7.0 * geo + 7.0 * top) * (m * w * Bo) / 4.0
        return mult + ax + ay + ac + m * kd * Bx + kd * w * S + m * w * Bo

    def _choose(self, kind, bx, by, m, kd, w, lev, k1, capm, capk, capw, xcapm, xcapk, stage):
        """(L, m2, kd2, w2, staged) of one call. L0 = today's rule (largest level <= lev at which every side divides and the smallest
        block >= the min block) is the floor; deeper candidates up to min(lev, PAD_LMAX) pad every side to the next multiple of 2^L
        within the capacities of the parent buffers (padded rows / columns carry stale finite data and only produce ignored output
        rows / columns; a padded contraction needs the zero columns of the staging copy of an unbatched left operand) and are taken
        when the modeled saving per extra level is >= PAD_GAIN_MIN (each level costs 19 extra ops of residual)."""
        key = (kind, bx, by, m, kd, w, lev, k1, capm, capk, capw, xcapm, xcapk, stage, self.mn, self.deep, self.role_epoch)
        best = self.choices.get(key)
        if best is not None:
            if PAD_DEBUG:
                PAD_LOG.append(best[1])
            return best[0]
        best = self._choose_uncached(kind, bx, by, m, kd, w, lev, k1, capm, capk, capw, xcapm, xcapk, stage)
        if len(self.choices) >= 4096:
            self.choices.clear()
        self.choices[key] = best
        if PAD_DEBUG:
            PAD_LOG.append(best[1])
        return best[0]

    def _fits(self, kind, B0, sm, sk, sn, S, L, G):
        """True when the plan needs no growth of an EXISTING role: a growth clears every plan and its public views (rebuilt lazily, a
        large residual cost in the remote client), so a deeper candidate never triggers one (new roles do not clear anything)"""
        Sr = max(S, self.hint_hub if kind == 'hub' else self.hint_mm) if B0 == 1 else S
        flat = self.flat
        for SS in (Sr, S) if Sr != S else (S,):
            for v in self._layout(kind, B0, sm, sk, sn, SS, L, G).values():
                for role, shape in v if isinstance(v, list) else [v]:
                    need = math.prod(shape)
                    if isinstance(role, tuple) and role[0] == 'leaf_out':
                        need += role[2]
                        role = ('hx', role[1])
                    role = 'ty' if role == 'tm' and LEAF_TY_ALIAS else role
                    role = 'tx' if role == 'tp' else role
                    role = _rmerge(role)
                    e = flat.get(role)
                    if e is not None and e[0].shape[0] < need:
                        return False
        return True

    def _choose_uncached(self, kind, bx, by, m, kd, w, lev, k1, capm, capk, capw, xcapm, xcapk, stage):
        res = self._choose_uncached0(kind, bx, by, m, kd, w, lev, k1, capm, capk, capw, xcapm, xcapk, stage)
        if not LX or self.deep <= 0:
            return res
        best, rec = res
        L = best[0]
        L1 = L + 1
        if not (LX_FROM <= L < LX_LMAX and tuple(best[1:]) == (m, kd, w, False)):
            return res
        d = 1 << L1
        if m % d or kd % d or w % d or min(m, kd, w) // d < LX_MINLEAF:
            if LX_PAD:
                return self._lx_pad(res, kind, bx, by, m, kd, w, k1, capm, capk, capw, xcapm, xcapk, stage, L, L1)
            return res
        hub = kind == 'hub'
        B0, S_ = ((1, by) if hub or bx == 1 else (by, 1))
        G = 2 if k1 else 1
        if not self._fits(kind, B0, m >> L1, kd >> L1, w >> L1, S_, L1, G):
            return res
        c0 = _lx_cost(kind, B0, m >> L, kd >> L, w >> L, S_, L, G, k1)
        c1 = _lx_cost(kind, B0, m >> L1, kd >> L1, w >> L1, S_, L1, G, k1)
        if c1 > c0 - LX_GAIN_MIN:
            return res
        out = ((L1, m, kd, w, False), tuple(rec[:9]) + (L1, m, kd, w, False, rec[14], float(rec[15]) - (c0 - c1)))
        if LX_MULTI and L1 < LX_LMAX:
            return self._lx_more(out, kind, bx, by, m, kd, w, k1, c1)
        return out

    def _lx_more(self, out, kind, bx, by, m, kd, w, k1, c_prev):
        """LX_MULTI: further free steps after a free LX step, each one saving >= LX_GAIN_MIN and fitting the reserved roles, up to
        LX_LMAX"""
        hub = kind == 'hub'
        B0, S_ = ((1, by) if hub or bx == 1 else (by, 1))
        G = 2 if k1 else 1
        L = out[0][0]
        while L < LX_LMAX:
            L1 = L + 1
            d = 1 << L1
            if m % d or kd % d or w % d or min(m, kd, w) // d < LX_MINLEAF:
                break
            if not self._fits(kind, B0, m >> L1, kd >> L1, w >> L1, S_, L1, G):
                break
            c1 = _lx_cost(kind, B0, m >> L1, kd >> L1, w >> L1, S_, L1, G, k1)
            if c1 > c_prev - LX_GAIN_MIN:
                break
            rec = out[1]
            out = ((L1, m, kd, w, False), tuple(rec[:9]) + (L1, m, kd, w, False, rec[14], float(rec[15]) - (c_prev - c1)))
            L, c_prev = (L1, c1)
        return out

    def _lx_pad(self, res, kind, bx, by, m, kd, w, k1, capm, capk, capw, xcapm, xcapk, stage, L, L1):
        """LX_PAD: the deeper level L1 with every side padded to the 2^L1 grid, under the legality of the PAD candidates of
        _choose_uncached0 (parent capacities, staging copy of an unbatched left operand for a padded contraction, a hub never pads
        its contraction, PAD_KINDS, slot caps, no role growth); taken when the exact billed cost incl. the padded leaf and the staging
        copy saves >= LX_GAIN_MIN"""
        best, rec = res
        hub = kind == 'hub'
        d = 1 << L1
        m2, kd2, w2 = (-(-m // d) * d, -(-kd // d) * d, -(-w // d) * d)
        if m2 > capm or kd2 > capk or w2 > capw or (hub and kd2 != kd) or min(m2, kd2, w2) // d < LX_MINLEAF:
            return res
        staged = False
        if (m2, kd2) != (m, kd):
            if kd2 == kd and m2 <= xcapm and kd2 <= xcapk:
                staged = False
            elif stage:
                staged = True
            else:
                return res
        if kind not in PAD_KINDS or (hub and by > PAD_SMAX) or ((not hub) and by > PAD_MM_SMAX):
            return res
        B0, S_ = ((1, by) if hub or bx == 1 else (by, 1))
        G = 2 if k1 else 1
        if not self._fits(kind, B0, m2 >> L1, kd2 >> L1, w2 >> L1, S_, L1, G):
            return res
        if LXPGATE and getattr(self, 'lx_ready', None) is not None and (kind, B0, m2 >> L1, kd2 >> L1, w2 >> L1, S_, L1, k1) not in self.lx_ready:
            return res
        c0 = _lx_cost(kind, B0, m >> L, kd >> L, w >> L, S_, L, G, k1)
        c1 = _lx_cost(kind, B0, m2 >> L1, kd2 >> L1, w2 >> L1, S_, L1, G, k1) + (m * kd * bx if staged else 0)
        if c1 > c0 - LX_GAIN_MIN:
            return res
        return ((L1, m2, kd2, w2, staged), tuple(rec[:9]) + (L1, m2, kd2, w2, staged, rec[14], float(rec[15]) - (c0 - c1)))

    def _choose_uncached0(self, kind, bx, by, m, kd, w, lev, k1, capm, capk, capw, xcapm, xcapk, stage):
        hub = kind == 'hub'
        L0 = self.level(m, kd, w, lev)
        guard = (not hub) and bx == 1 and (by > self.cap_mm)
        if guard and L0 >= 6:
            L0 -= 1
        cap = min(lev, self.deep)
        best = (L0, m, kd, w, False)
        if L0 <= 0 or cap <= L0:
            return (best, (kind, bx, by, m, kd, w, lev, self.mn, L0, L0, m, kd, w, False, 0.0, 0.0))
        c0 = cbest = self._cost(kind, bx, by, m, kd, w, L0, k1)
        for L in range(L0 + 1, cap + 1):
            if guard and L >= 6:
                break
            d = 1 << L
            m2, kd2, w2 = (-(-m // d) * d, -(-kd // d) * d, -(-w // d) * d)
            if m2 > capm or kd2 > capk or w2 > capw or (hub and kd2 != kd):
                break
            staged = False
            if (m2, kd2) != (m, kd):
                if kd2 == kd and m2 <= xcapm and (kd2 <= xcapk):
                    staged = False
                elif stage:
                    staged = True
                else:
                    break
            if ('free' if (m2, kd2, w2) == (m, kd, w) else kind) not in PAD_KINDS or (hub and by > PAD_SMAX) or ((not hub) and by > PAD_MM_SMAX):
                continue
            B0, S_ = (1, by) if hub or bx == 1 else (by, 1)
            if not self._fits(kind, B0, m2 >> L, kd2 >> L, w2 >> L, S_, L, 2 if k1 else 1):
                continue
            c = self._cost(kind, bx, by, m2, kd2, w2, L, k1) + (m * kd * bx if staged else 0)
            if c <= cbest - PAD_GAIN_MIN * (L - best[0]):
                best = (L, m2, kd2, w2, staged)
                cbest = c
        return (best, (kind, bx, by, m, kd, w, lev, self.mn, L0, best[0], best[1], best[2], best[3], best[4], c0, cbest))

    @staticmethod
    def _padded(A, Ap, r, c):
        """the [..., :r, :c] window of the parent Ap (a public view, so its Z-order views are cached); it must start where A starts
        with A's strides, i.e. A == window[..., :A.rows, :A.cols] (the caller's parent hint is checked, never trusted)"""
        V = _pview(Ap, (..., slice(None, r, None), slice(None, c, None)))
        if V.shape[:-2] != A.shape[:-2] or r < A.shape[-2] or c < A.shape[-1]:
            raise ValueError('pad parent mismatch %s %s' % (A.shape, Ap.shape))
        native = _PAD_NATIVE[0]
        if native is None:
            # decided once per process: native arrays expose pointers and strides, the remote client's arrays do not (there the hint is
            # trusted -- the call sites build it from the operand's own parent)
            ai = getattr(V, '__array_interface__', None)
            native = _PAD_NATIVE[0] = isinstance(ai, dict) and isinstance(getattr(A, '__array_interface__', None), dict) and hasattr(V, 'strides') and hasattr(A, 'strides')
        if native and (V.strides != A.strides or V.__array_interface__['data'][0] != A.__array_interface__['data'][0]):
            raise ValueError('pad parent mismatch %s %s' % (A.shape, Ap.shape))
        return V

    def _stage_x(self, X, m2, kd2, dry):
        """Padded copy of the unbatched left operand X (1, 1, m, kd) in a core-owned zero-initialised staging buffer: rows m..m2 keep
        stale finite data, columns kd..kd2 are zero (invariant: columns [stage_dirty, N) are zero)."""
        m, kd = (X.shape[-2], X.shape[-1])
        if self.spool is None:
            self.spool = _Pool(self.dtype)
        N = max(64, 1 << (max(m2, kd2) - 1).bit_length())
        key = ('stage_x', N)
        stg = self.spool.get(key, (1, 1, N, N))
        self.cache_roots.add(id(stg))
        if not dry:
            dirty = self.stage_dirty.get(key, 0)
            fnp.copyto(_pview(stg, (..., slice(None, m, None), slice(None, kd, None))), X)
            dirty = max(dirty, kd)
            if kd2 > kd and dirty > kd:
                fnp.copyto(_pview(stg, (..., slice(kd, dirty, None))), 0.0)
                dirty = kd
            self.stage_dirty[key] = dirty
        return _pview(stg, (..., slice(None, m2, None), slice(None, kd2, None)))

    @staticmethod
    def ok(m, kd, w, lev, minblk):
        d = 1 << lev
        return lev > 0 and m % d == 0 and (kd % d == 0) and (w % d == 0) and (min(m, kd, w) // d >= minblk)

    def level(self, m, kd, w, lev, minblk=None):
        mb = self.mn if minblk is None else int(minblk)
        while lev > 0 and (not self.ok(m, kd, w, lev, mb)):
            lev -= 1
        return lev

    def _reserve(self, role, n):
        if isinstance(role, tuple) and role[0] == 'leaf_out':
            n += role[2]
            role = ('hx', role[1])
        role = 'ty' if role == 'tm' and LEAF_TY_ALIAS else role
        role = 'tx' if role == 'tp' else role
        role = _rmerge(role)
        e = self.flat.get(role)
        if e is None or e[0].shape[0] < n:
            if DEBUG:
                print(f'ZFMM reserve {role}: {(0 if e is None else e[0].shape[0])} -> {n} elements (plans {len(self.plans)})', flush=True)
            if e is not None:
                self._clear_public_zviews()
                self.plans.clear()
            b = fnp.empty((n,), dtype=self.dtype)
            fnp.copyto(b, 0.0)
            self.flat[role] = [b, {}]
            self.role_epoch += 1
            if self.public_setup_slices is not None:
                self.public_setup_slices.clear()

    def _reserve_setup(self, role, n):
        if isinstance(role, tuple) and role[0] == 'leaf_out':
            n += role[2]
            role = ('hx', role[1])
        role = 'ty' if role == 'tm' and LEAF_TY_ALIAS else role
        role = 'tx' if role == 'tp' else role
        role = _rmerge(role)
        e = self.flat.get(role)
        if e is None or e[0].shape[0] < n:
            if DEBUG:
                print(f'ZFMM reserve {role}: {(0 if e is None else e[0].shape[0])} -> {n} elements (plans {len(self.plans)})', flush=True)
            if e is not None:
                self._clear_public_zviews()
                self.plans.clear()
            b = fnp.zeros((n,), dtype=self.dtype)
            self.flat[role] = [b, {}]
            self.role_epoch += 1
            if self.public_setup_slices is not None:
                self.public_setup_slices.clear()

    def _arr(self, role, shape):
        offset = 0
        if isinstance(role, tuple) and role[0] == 'leaf_out':
            offset = role[2]
            role = ('hx', role[1])
        role = 'ty' if role == 'tm' and LEAF_TY_ALIAS else role
        role = 'tx' if role == 'tp' else role
        role = _rmerge(role)
        e = self.flat[role]
        key = (offset, shape) if offset else shape
        v = e[1].get(key)
        if v is None:
            size = math.prod(shape)
            if self.public_setup_slices is None:
                base = _pview(e[0], slice(offset, offset + size, None))
            else:
                slice_key = (role, offset, size)
                base = self.public_setup_slices.get(slice_key)
                if base is None:
                    base = _pview(e[0], slice(offset, offset + size, None))
                    self.public_setup_slices[slice_key] = base
            v = fnp.reshape(base, shape)
            e[1][key] = v
        return v

    @staticmethod
    def _layout(kind, B0, sm, sk, sn, S, L, G=1):
        """(role, shape) of every array of one plan (px/py/po per depth, leaf temporaries, level-0 conversion views)"""
        hub = kind == 'hub'
        cx = S * sk if hub else sk
        ry, cy = (sn, S * sk) if hub else (sk, S * sn)
        co = G * sn if hub else S * sn
        so = G if hub else S
        bits = (2,) * (2 * L)
        bl = (B0,) + (7,) * (L - 1)
        lay = {}
        xr = 'hx' if hub else 'x'
        if ZF7:
            # one 7-ary buffer per side (= the old leaf-level buffer); the Z-order entry/exit views are strided views of it (built in _plan)
            lay['px'] = [((xr, L - 1), bl + (4, sm, cx))]
            lay['py'] = [(('y', L - 1), bl + (4, ry, cy))]
            if hub and LEAF_PACK:
                orole = ('leaf_out', L - 1, math.prod(bl + (4, sm, cx)))
            else:
                orole = ('o', L - 1) if hub else ('hx', L - 1)
            lay['po'] = [(orole, bl + (4, sm, co))]
            lay['tx'] = ('tx', bl + (sm, cx))
            lay['ty'] = ('ty', bl + (ry, cy))
            lay['tm'] = ('tm', bl + (sm, co))
            lay['tp'] = ('tp', bl + (sm, co))
            return lay
        lay['px'] = [((xr, d), (B0,) + (7,) * d + (4,) * (L - d) + (sm, cx)) for d in range(L)]
        lay['py'] = [(('y', d), (B0,) + (7,) * d + (4,) * (L - d) + (ry, cy)) for d in range(L)]
        po = []
        for d in range(L):
            pre = (B0,) + (7,) * d + (4,) * (L - d)
            alias = d <= L - 2 and math.prod(pre) * sm * co <= math.prod(pre) * ry * cy
            role = ('y', d) if alias else ('hx', d) if not hub else ('o', d)
            if hub and d == L - 1 and LEAF_PACK:
                role = ('leaf_out', d, math.prod(lay['px'][d][1]))
            po.append((role, pre + (sm, co)))
        lay['po'] = po
        lay['tx'] = ('tx', bl + (sm, cx))
        lay['ty'] = ('ty', bl + (ry, cy))
        lay['tm'] = ('tm', bl + (sm, co))
        lay['tp'] = ('tp', bl + (sm, co))
        lay['xz'] = ((xr, 0), (B0,) + bits + (sm, S if hub else 1, sk))
        lay['yz'] = (('y', 0), (B0,) + bits + ((sn, S, sk) if hub else (sk, S, sn)))
        lay['oz'] = (po[0][0], (B0,) + bits + (sm, so, sn))
        return lay

    def _plan(self, kind, B0, sm, sk, sn, S, L, k1=0):
        G = 2 if k1 else 1
        key = (kind, B0, sm, sk, sn, S, L, k1)
        pl = self.plans.get(key)
        if pl is not None:
            return pl
        if DEBUG:
            print(f'ZFMM new plan {key}', flush=True)
        if self.mark:
            self.steady_keys.add(key)
        Sr = max(S, self.hint_hub if kind == 'hub' else self.hint_mm) if B0 == 1 else S
        need = {}
        lay_r = self._layout(kind, B0, sm, sk, sn, Sr, L, G)
        for v in lay_r.values():
            for role, shape in v if isinstance(v, list) else [v]:
                need[role] = max(need.get(role, 0), math.prod(shape))
        lay = self._layout(kind, B0, sm, sk, sn, S, L, G)
        for v in lay.values():
            for role, shape in v if isinstance(v, list) else [v]:
                need[role] = max(need.get(role, 0), math.prod(shape))
        for role, n in need.items():
            self._reserve(role, n)
        pl = _Plan()
        pl.kind, pl.B0, pl.L, pl.S, pl.G, pl.k1, pl.sk, pl.sn = (kind, B0, L, S, G, k1, sk, sn)
        g = lambda t: self._arr(t[0], t[1])
        pl.px = [g(t) for t in lay['px']]
        pl.py = [g(t) for t in lay['py']]
        pl.po = [g(t) for t in lay['po']]
        pl.tx, pl.ty, pl.tm, pl.tp = (g(lay['tx']), g(lay['ty']), g(lay['tm']), g(lay['tp']))
        if ZF7:
            hub = kind == 'hub'
            # inner-level digit maps (slot = k + b i + c j + const of the quadrant digits; the content of a node occupies the slots k..k+3
            # of every deeper 7-axis): X11/X12/X21/X22 at slots 1/3/0/2, B11/B12/B21/B22 at 1/2/3/4 (hub: the array digits are those
            # of B^T), the result quadrants C11/C12/C21/C22 at 1/2/3/4; the leaf 4-axis is in Z order for all three
            pl.ab = ZF7_AB and _ab_saving(kind, B0, sm, sk, sn, S, L, G) >= ZF7_AB_MIN
            if pl.ab:
                # alternative-basis layout: X entries at slots 0..3 (slot = 2 (1 - i) + j), Y entries at 2..5 (slot = 2 + 2 j + i of B),
                # output at 2..5 (slot = 2 + 2 j + 1 - i)
                pl.xz = self._zview7(pl.px[0], L, 0, -2, 1, (S if hub else 1, sk))
                pl.yz = self._zview7(pl.py[0], L, 2, 2 if hub else 1, 1 if hub else 2, (S, sk) if hub else (S, sn))
                pl.oz = self._zview7(pl.po[0], L, 2, -1, 2, (G if hub else S, sn))
            else:
                pl.xz = self._zview7(pl.px[0], L, 0, -1, 2, (S if hub else 1, sk))
                pl.yz = self._zview7(pl.py[0], L, 1, 1 if hub else 2, 2 if hub else 1, (S, sk) if hub else (S, sn))
                pl.oz = self._zview7(pl.po[0], L, 1, 2, 1, (G if hub else S, sn))
        else:
            pl.xz, pl.yz, pl.oz = (g(lay['xz']), g(lay['yz']), g(lay['oz']))
        pl.public_zviews = None
        self.plans[key] = pl
        return pl

    def _zinv(self, pl, i):
        """opd_patch OPD_ZT: plan view i (0 xz, 1 yz, 2 oz) in the axis order (T, S, i1..iL, sr, j1..jL, sc) of the operand's split-only
        reshape (_zsrcn): transpose by the inverse of _zsrc's permutation (one free view op per plan view, cached on the plan)"""
        nv = pl.__dict__.get('opd_n')
        if nv is None:
            nv = pl.opd_n = [None, None, None]
        v = nv[i]
        if v is None:
            L = pl.L
            P = [0]
            for l in range(L):
                P += [2 + l, L + 3 + l]
            P += [L + 2, 1, 2 * L + 3]
            inv = [0] * len(P)
            for k_, p_ in enumerate(P):
                inv[p_] = k_
            v = nv[i] = fnp.transpose((pl.xz, pl.yz, pl.oz)[i], tuple(inv))
        return v

    @staticmethod
    def _zview7(buf, L, k, b, c, tail):
        """ZF7: the Z-order entry/exit view (B0, i1, j1, ..., iL, jL, rows, tail...) of a 7-ary buffer (B0, 7^(L-1), 4, rows, cols):
        at every inner level the quadrant (i, j) sits at slot k + 2u + v of the slice k..k+3 with (u, v) = (i, j) up to the flips /
        transposition encoded by (b, c) (|b| = 2: u ~ i, v ~ j; |b| = 1: u ~ j, v ~ i; a negative coefficient flips the digit), the leaf
        4-axis is in Z order; cols is split into `tail`. One billed reshape per plan (a pure view: only axes are split)."""
        B0, r = (buf.shape[0], buf.shape[-2])
        v = fnp.reshape(buf[(slice(None),) + (slice(k, k + 4),) * (L - 1)], (B0,) + (2, 2) * (L - 1) + (2, 2, r) + tuple(tail))
        idx = [slice(None)]
        perm = [0]
        flip = slice(None, None, -1)
        for l in range(L - 1):
            u, w = (1 + 2 * l, 2 + 2 * l)
            if abs(b) == 2:
                idx += [flip if b < 0 else slice(None), flip if c < 0 else slice(None)]
                perm += [u, w]
            else:
                idx += [flip if c < 0 else slice(None), flip if b < 0 else slice(None)]
                perm += [w, u]
        perm += list(range(2 * (L - 1) + 1, v.ndim))
        v = v[tuple(idx)]
        if perm != list(range(v.ndim)):
            v = fnp.transpose(v, tuple(perm))
        return v

    def _rres_unstack(self, base, axis):
        """rres_patch: every slice of `base` along `axis` in one free op (views); None after a failure (parent path from then on)"""
        if not _RRES_OK[0]:
            return None
        try:
            return fnp.unstack(base, axis=axis)
        except Exception:
            _RRES_OK[0] = False
            return None

    def _rres_slots(self, arr, prefix, n4, k, slots):
        """rres_patch: the missing _znode7(arr, prefix, s, n4, k) views of `slots` from one unstack of
        W = arr[(:,) * prefix + (:,) + (k:k+4,) * n4] (W = arr when n4 == 0); only when that saves round trips"""
        views = self._zview_dict(arr)
        miss = [s for s in slots if ('z7', prefix, s, n4, k) not in views]
        if len(miss) < (2 if n4 == 0 else 3):
            return
        if getattr(self, 'rres_setup', False) and (arr.shape[prefix] if n4 == 0 else arr.shape[prefix] + 1) - len(miss) > 1:
            return   # setup prewarm (wall-bound): only groups that create at most one unused view
        base = arr if n4 == 0 else arr[(_S,) * prefix + (_S,) + (slice(k, k + 4),) * n4]
        parts = self._rres_unstack(base, prefix)
        if parts is None:
            return
        for s in miss:
            views['z7', prefix, s, n4, k] = parts[s]
            self.public_zview_count += 1

    def _rres_digits(self, arr, L, k, want):
        """rres_patch: the missing _zdig7(arr, L, lev, s, k) views (want: lev -> slots) from V = arr[(:,) + (k:k+4,) * (L - 1)]:
        digit lev < L - 1 is unstack(V, 1 + lev)[s - k], the leaf digit L - 1 is unstack(V, L)[s]; digits with >= 2 missing views,
        and only when >= 2 such digits share V"""
        if getattr(self, 'rres_setup', False):
            return   # setup prewarm (wall-bound): digit groups create unused views
        views = self._zview_dict(arr)
        levs = []
        for lev in sorted(want):
            m = [s for s in want[lev] if ('zd7', lev, s, k) not in views]
            if len(m) >= 2 and (lev == L - 1 or all(0 <= s - k < 4 for s in m)):
                levs.append((lev, m))
        if len(levs) < 2:
            return
        V = arr[(_S,) + (slice(k, k + 4),) * (L - 1)]
        for lev, m in levs:
            parts = self._rres_unstack(V, 1 + lev)
            if parts is None:
                return
            for s in m:
                views['zd7', lev, s, k] = parts[s] if lev == L - 1 else parts[s - k]
                self.public_zview_count += 1

    def _rres_batch7(self, pl):
        """rres_patch: pre-create (batched) the single-slot / leaf / digit views that _zplanviews7(_ab) requests next for plan pl
        (same keys, same views, same public_zview_count as the per-view getitems they replace)"""
        L = pl.L
        if L < 2:
            return
        ab = getattr(pl, 'ab', False)
        X7, Y7, O7 = (pl.px[0], pl.py[0], pl.po[0])
        if ab and ZF7_AB_MERGE:
            sx, sy, so, kk = ((0, 2, 6), (4, 2, 6), (6, 0, 1, 2), 2)
        else:
            sx = sy = so = tuple(range(7))
            kk = 2 if ab else 1
        for d in range(L - 1):
            self._rres_slots(X7, 1 + d, L - 2 - d, 0, sx)
            self._rres_slots(Y7, 1 + d, L - 2 - d, kk, sy)
            self._rres_slots(O7, 1 + d, L - 2 - d, kk, so)
        self._rres_slots(X7, L, 0, 0, (0, 1, 2, 3))
        self._rres_slots(Y7, L, 0, 1, (0, 1, 2, 3))
        self._rres_slots(O7, L, 0, 1, (0, 1, 2, 3))
        if ab:
            qb = 2 if pl.kind == 'hub' else 1
            self._rres_digits(X7, L, 0, dict([(l, (0, 1)) for l in range(L - 1)] + [(L - 1, (0, 2, 3))]))
            self._rres_digits(Y7, L, 2, dict([(l, (4, 5)) for l in range(L - 1)] + [(L - 1, (0, qb, 3))]))
            self._rres_digits(O7, L, 2, dict([(l, (2, 4)) for l in range(L - 1)] + [(L - 1, (2, 3))]))

    def _znode7(self, arr, prefix, s, n4, k):
        """cached view arr[(:,)*prefix, s, (k:k+4,)*n4] (the children slot / leaf quadrant views of the 7-ary buffers)"""
        views = self._zview_dict(arr)
        key = ('z7', prefix, s, n4, k)
        result = views.get(key)
        if result is None:
            result = arr[(_S,) * prefix + (s,) + (slice(k, k + 4),) * n4]
            views[key] = result
            self.public_zview_count += 1
        return result

    def _zplanviews7(self, pl):
        if getattr(pl, 'ab', False):
            return self._zplanviews7_ab(pl)
        cached = pl.public_zviews
        if cached is not None:
            return cached
        if self.public_zview_count >= PUBLIC_Z_VIEW_LIMIT:
            self._clear_public_zviews()
        if RRES_ON:
            self._rres_batch7(pl)
        L = pl.L
        hub = pl.kind == 'hub'
        X7, Y7, O7 = (pl.px[0], pl.py[0], pl.po[0])
        stages = []
        for d in range(L - 1):
            xs = tuple((self._znode7(X7, 1 + d, s, L - 2 - d, 0) for s in range(7)))
            ys = tuple((self._znode7(Y7, 1 + d, s, L - 2 - d, 1) for s in range(7)))
            os_ = tuple((self._znode7(O7, 1 + d, s, L - 2 - d, 1) for s in range(7)))
            stages.append((xs, ys, os_))
        a = tuple((self._znode7(X7, L, q, 0, 0) for q in range(4)))
        b = tuple((self._znode7(Y7, L, q, 0, 1) for q in ((0, 2, 1, 3) if hub else range(4))))
        c = tuple((self._znode7(O7, L, q, 0, 1) for q in range(4)))
        products = ((pl.tx, pl.ty, c[2]), (pl.tx, pl.ty, c[3]), (pl.tx, pl.ty, c[1]), (pl.tx, b[3], c[0]), (a[3], pl.ty, pl.tp), (a[0], b[0], pl.tm), (a[1], b[2], pl.tp))
        prepared_products = []
        for left, right, dest in products:
            if not hub:
                prepared_products.append(((left, right, dest),))
            elif pl.G == 1:
                prepared_products.append(((left, self._zswap(right), dest),))
            else:
                c1, sn = (pl.k1 * pl.sk, pl.sn)
                prepared_products.append(tuple(((self._ztail(left, lo, hi), self._zswap(self._ztail(right, lo, hi)), self._ztail(dest, oo, oh)) for lo, hi, oo, oh in ((None, c1, None, sn), (c1, None, sn, None)))))
        cached = (tuple(stages), (a, b, c, tuple(prepared_products)))
        pl.public_zviews = cached
        return cached

    def _run7(self, pl):
        """ZF7 in-place recursion (same operands and order as the per-depth code, so the results are bit-identical).
        Children slots (product = X child * Y child): 0 X5*Y5, 1 X11*B11, 2 X22*Y3, 3 X12*B21, 4 X2*B22, 5 X4*Y4, 6 X6*Y6 with
        X4 = X21+X22, X5 = X4-X11, X6 = X11-X21, X2 = X12-X5; Y4 = B12-B11, Y5 = B22-Y4, Y6 = B22-B12, Y3 = Y5-B21.
        Parent quadrants: X11/X12/X21/X22 at slots 1/3/0/2 (X5 overwrites X21 last), B11/B12/B21/B22 at 1/2/3/4 (Y3 overwrites B12 last).
        Assembly with the products M0..M6 at slots 1/3/4/2/5/0/6: C11/C12/C21/C22 land at slots 1/2/3/4."""
        stages, leaf = self._zplanviews7(pl)
        add, sub = (fnp.add, fnp.subtract)
        if RRES_ON and RRES_HOLD:
            # rres_patch RRES_HOLD: keep this call's result proxies, release them together at return (one batched free)
            hold_ = []
            ap_ = hold_.append
            add = lambda a, b, out, f=fnp.add: ap_(f(a, b, out=out))
            sub = lambda a, b, out, f=fnp.subtract: ap_(f(a, b, out=out))
        for xs, ys, _o in stages:
            add(xs[0], xs[2], out=xs[5])
            sub(xs[1], xs[0], out=xs[6])
            sub(xs[5], xs[1], out=xs[0])
            sub(xs[3], xs[0], out=xs[4])
            sub(ys[2], ys[1], out=ys[5])
            sub(ys[4], ys[2], out=ys[6])
            sub(ys[4], ys[5], out=ys[0])
            sub(ys[0], ys[3], out=ys[2])
        self._leaf(pl, leaf)
        for _x, _y, s in reversed(stages):
            add(s[0], s[1], out=s[0])
            add(s[1], s[3], out=s[1])
            add(s[6], s[0], out=s[6])
            add(s[5], s[0], out=s[0])
            sub(s[6], s[2], out=s[3])
            add(s[0], s[4], out=s[2])
            add(s[6], s[5], out=s[4])

    def _zdig7(self, arr, L, lev, s, k):
        """ZF7_AB: cached view of the root content at one digit (the basis-change passes): digit `lev` (0..L-2: that 7-axis, L-1: the
        leaf 4-axis) fixed to slot / index `s`, every other inner 7-axis restricted to the entry slots k..k+3"""
        views = self._zview_dict(arr)
        key = ('zd7', lev, s, k)
        result = views.get(key)
        if result is None:
            cs = slice(k, k + 4)
            if lev < L - 1:
                result = arr[(_S,) + (cs,) * lev + (s,) + (cs,) * (L - 2 - lev)]
            else:
                result = arr[(_S,) + (cs,) * (L - 1) + (s,)]
            views[key] = result
            self.public_zview_count += 1
        return result

    def _znode7s(self, arr, prefix, sl, n4, k):
        """ZF7_AB_MERGE: cached view arr[(:,)*prefix, start:stop:step, (k:k+4,)*n4] (several child slots in one view; sl a tuple)"""
        views = self._zview_dict(arr)
        key = ('z7s', prefix, sl, n4, k)
        result = views.get(key)
        if result is None:
            result = arr[(_S,) * prefix + (slice(*sl),) + (slice(k, k + 4),) * n4]
            views[key] = result
            self.public_zview_count += 1
        return result

    def _zplanviews7_ab(self, pl):
        """ZF7_AB: views of the alternative-basis run (see _run7_ab): inner stages (X entries at slots 0..3, Y and O at 2..5), the leaf
        quadrants with the product list of _leaf_ab, and the basis-change views per digit"""
        cached = pl.public_zviews
        if cached is not None:
            return cached
        if self.public_zview_count >= PUBLIC_Z_VIEW_LIMIT:
            self._clear_public_zviews()
        if RRES_ON:
            self._rres_batch7(pl)
        L = pl.L
        hub = pl.kind == 'hub'
        X7, Y7, O7 = (pl.px[0], pl.py[0], pl.po[0])
        stages = []
        for d in range(L - 1):
            if ZF7_AB_MERGE:
                # merged ops (see _run7_ab): one op forms two children / does two assembly adds through slot slices
                n7 = lambda arr, s, k, d=d: self._znode7(arr, 1 + d, s, L - 2 - d, k)
                s7 = lambda arr, sl, k, d=d: self._znode7s(arr, 1 + d, sl, L - 2 - d, k)
                xs = (n7(X7, 0, 0), n7(X7, 2, 0), n7(X7, 6, 0), s7(X7, (3, 0, -2), 0), s7(X7, (6, 7, 1), 0), s7(X7, (5, 3, -1), 0))
                ys = (n7(Y7, 4, 2), n7(Y7, 2, 2), n7(Y7, 6, 2), s7(Y7, (5, 2, -2), 2), s7(Y7, (6, 7, 1), 2), s7(Y7, (0, 2, 1), 2))
                os_ = (s7(O7, (6, 2, -3), 2), s7(O7, (2, 3, 1), 2), n7(O7, 6, 2), n7(O7, 0, 2), n7(O7, 1, 2), n7(O7, 2, 2), s7(O7, (5, 3, -1), 2), s7(O7, (6, 7, 1), 2))
                stages.append((xs, ys, os_))
                continue
            xs = tuple((self._znode7(X7, 1 + d, s, L - 2 - d, 0) for s in range(7)))
            ys = tuple((self._znode7(Y7, 1 + d, s, L - 2 - d, 2) for s in range(7)))
            os_ = tuple((self._znode7(O7, 1 + d, s, L - 2 - d, 2) for s in range(7)))
            stages.append((xs, ys, os_))
        a = tuple((self._znode7(X7, L, q, 0, 0) for q in range(4)))
        b = tuple((self._znode7(Y7, L, q, 0, 1) for q in ((0, 2, 1, 3) if hub else range(4))))
        c = tuple((self._znode7(O7, L, q, 0, 1) for q in range(4)))
        # Pf, Pb, Pa, Pg, Pd, Pe, Pc (order of _leaf_ab)
        products = ((a[2], b[1], c[1]), (a[0], b[0], c[0]), (pl.tx, pl.ty, c[2]), (pl.tx, pl.ty, c[3]), (a[3], pl.ty, pl.tp), (pl.tx, b[3], pl.tm), (a[1], b[2], pl.tp))
        prepared_products = []
        for left, right, dest in products:
            if not hub:
                prepared_products.append(((left, right, dest),))
            elif pl.G == 1:
                prepared_products.append(((left, self._zswap(right), dest),))
            else:
                c1, sn = (pl.k1 * pl.sk, pl.sn)
                prepared_products.append(tuple(((self._ztail(left, lo, hi), self._zswap(self._ztail(right, lo, hi)), self._ztail(dest, oo, oh)) for lo, hi, oo, oh in ((None, c1, None, sn), (c1, None, sn, None)))))
        g = self._zdig7
        qb = 2 if hub else 1
        xd = tuple(((g(X7, L, l, 0, 0), g(X7, L, l, 1, 0)) for l in range(L - 1))) + ((g(X7, L, L - 1, 0, 0), g(X7, L, L - 1, 2, 0), g(X7, L, L - 1, 3, 0)),)
        yd = tuple(((g(Y7, L, l, 4, 2), g(Y7, L, l, 5, 2)) for l in range(L - 1))) + ((g(Y7, L, L - 1, 0, 2), g(Y7, L, L - 1, qb, 2), g(Y7, L, L - 1, 3, 2)),)
        od = tuple(((g(O7, L, l, 2, 2), g(O7, L, l, 4, 2)) for l in range(L - 1))) + ((g(O7, L, L - 1, 2, 2), g(O7, L, L - 1, 3, 2)),)
        cached = (tuple(stages), (a, b, c, tuple(prepared_products)), (xd, yd, od))
        pl.public_zviews = cached
        return cached

    def _run7_ab(self, pl):
        """ZF7_AB in-place recursion in the alternative basis. Entry: X content (A11, A12, A21, A22) gathered at slots 2/3/0/1 of every
        inner 7-axis, Y content (B11, B12, B21, B22) at 2/4/3/5; the basis changes turn the A21 slot into X4 = A21 + A22 (inner digits) /
        X5 = A21 + A22 - A11 (leaf digit) and the B12 slot into Y6 = B22 - B12 / Y5 = B11 - B12 + B22. Per inner level: children
        X5 / X2 / X6 -> slots 6 / 5 / 4, Y5 / Y4 / Y3 -> 6 / 0 / 1; products Pa Pd Pb Pc Pg Pe Pf at slots 0..6; assembly
        C11 / C12 / D / C22 at slots 3 / 5 / 2 / 4 (D = C22 - C21); exit: C21 = C22 - D on every digit, then the scatter copy."""
        stages, leaf, (xd, yd, od) = self._zplanviews7_ab(pl)
        add, sub = (fnp.add, fnp.subtract)
        if RRES_ON and RRES_HOLD:
            # rres_patch RRES_HOLD: keep this call's result proxies, release them together at return (one batched free)
            hold_ = []
            ap_ = hold_.append
            add = lambda a, b, out, f=fnp.add: ap_(f(a, b, out=out))
            sub = lambda a, b, out, f=fnp.subtract: ap_(f(a, b, out=out))
        for t, s in xd[:-1]:
            add(t, s, out=t)
        q0, q2, q3 = xd[-1]
        add(q2, q3, out=q2)
        sub(q2, q0, out=q2)
        for t, s in yd[:-1]:
            sub(s, t, out=t)
        q0, q1, q3 = yd[-1]
        sub(q3, q1, out=q1)
        add(q1, q0, out=q1)
        if ZF7_AB_MERGE:
            # X5 -> 6; (X2, X6) = (A12, A22) - X5 -> 5, 4; Y5 -> 6; (Y4, -Y3) = (B22, B21) - Y5 -> 0, 1 (slot 1 then carries -Pd);
            # assembly: (Pf, Pc) += Pb -> 6, 3; V = Pf + Pb + Pa -> 6; D = Pa - (-Pd) -> 2; (C12, C22) = (Pe, Pg) + V -> 5, 4
            for xs, ys, _o in stages:
                sub(xs[0], xs[1], out=xs[2])
                sub(xs[3], xs[4], out=xs[5])
                add(ys[0], ys[1], out=ys[2])
                sub(ys[3], ys[4], out=ys[5])
            self._leaf_ab(pl, leaf)
            for _x, _y, s in reversed(stages):
                add(s[0], s[1], out=s[0])
                add(s[2], s[3], out=s[2])
                sub(s[3], s[4], out=s[5])
                add(s[6], s[7], out=s[6])
        else:
            for xs, ys, _o in stages:
                sub(xs[0], xs[2], out=xs[6])
                sub(xs[3], xs[6], out=xs[5])
                sub(xs[1], xs[6], out=xs[4])
                add(ys[4], ys[2], out=ys[6])
                sub(ys[5], ys[6], out=ys[0])
                sub(ys[6], ys[3], out=ys[1])
            self._leaf_ab(pl, leaf)
            for _x, _y, s in reversed(stages):
                add(s[6], s[2], out=s[6])
                add(s[6], s[0], out=s[6])
                add(s[3], s[2], out=s[3])
                add(s[0], s[1], out=s[2])
                add(s[5], s[6], out=s[5])
                add(s[4], s[6], out=s[4])
        for t, s in od:
            sub(s, t, out=t)

    def _leaf_ab(self, pl, leaf):
        """ZF7_AB fused leaf: entries (A11, A12, X5, A22), (B11, Y5, B21, B22) in Z order; 7 products, 3 + 3 + 6 adds; outputs
        (C11, C12, D, C22). tP aliases tX and tM aliases tY (roles tx / ty): every alias is written only after its twin is dead."""
        a, b, c, products = leaf
        A11, A12, X5, A22 = a
        B11, Y5, B21, B22 = b
        C11, C12, D, C22 = c
        tX, tY, tM, tP = (pl.tx, pl.ty, pl.tm, pl.tp)
        add, sub = (fnp.add, fnp.subtract)
        mat_ = fnp.matmul
        if RRES_ON and RRES_HOLD:
            # rres_patch RRES_HOLD: keep this call's result proxies, release them together at return (one batched free)
            hold_ = []
            ap_ = hold_.append
            add = lambda a, b, out, f=fnp.add: ap_(f(a, b, out=out))
            sub = lambda a, b, out, f=fnp.subtract: ap_(f(a, b, out=out))
            mat_ = lambda a, b, out, f=fnp.matmul: ap_(f(a, b, out=out))

        def mm(index):
            for left, right, dest in products[index]:
                mat_(left, right, out=dest)
        mm(0)
        mm(1)
        add(C12, C11, out=C12)
        add(X5, A11, out=tX)
        sub(B22, Y5, out=tY)
        mm(2)
        add(C12, D, out=C12)
        sub(A22, X5, out=tX)
        sub(Y5, B11, out=tY)
        mm(3)
        add(C22, C12, out=C22)
        sub(Y5, B21, out=tY)
        mm(4)
        add(D, tP, out=D)
        sub(A12, X5, out=tX)
        mm(5)
        add(C12, tM, out=C12)
        mm(6)
        add(C11, tP, out=C11)

    def rebuild_steady(self):
        """(re)create every steady-level plan that a role growth cleared (end of the first predict, whose op count has the slack)"""
        for key in sorted(self.steady_keys):
            if key not in self.plans:
                self._plan(*key)

    def _zslots(self, arr, L):
        if not PUBLIC_VIEWS:
            return self._zsrc(fnp.swapaxes(arr, 0, 1), L)
        entry = _public_info(arr)
        if arr.nbytes >= self.min_cache_bytes and entry is not None and (entry[2] is not None):
            root = entry[2]()
            if root is not None and root.ndim == 4 and (root.shape[1:] == arr.shape[1:]):
                start = entry[3]
                if 0 <= start and start + arr.shape[0] <= root.shape[0]:
                    key = (id(root), L)
                    cached = self.zb.get(key)
                    if cached is None or cached[0] is not root:
                        if len(self.zb) >= 64:
                            self.zb.clear()
                        cached = (root, self._zsrc(fnp.swapaxes(root, 0, 1), L))
                        self.zb[key] = cached
                    return cached[1][..., start:start + arr.shape[0], :]
        return self._cached(arr, 'zs', L, lambda: self._zsrc(fnp.swapaxes(arr, 0, 1), L))

    def _cached(self, arr, tag, L, make):
        entry = _public_info(arr)
        if not PUBLIC_VIEWS or entry is None:
            return make()
        root = entry[4]()
        if root is None or (arr.nbytes < self.min_cache_bytes and id(root) not in self.cache_roots and not ZSRC_CACHE):
            return make()
        key = (id(root), entry[5], tuple(arr.shape), tag, L)
        retained = self.public_cache_inputs.get(key)
        view = self.vc.get(key)
        if retained is None or retained[0] is not root or view is None:
            if len(self.vc) >= (ZSRC_CACHE_CAP if ZSRC_CACHE else 512):
                self.vc.clear()
                self.public_cache_inputs.clear()
            view = make()
            self.vc[key] = view
            self.public_cache_inputs[key] = (root, arr)
        return view

    @staticmethod
    def _zsrcn(A4, L):
        """opd_patch OPD_ZT: A4 (T, S, R, C) any strides -> split-only view (T, S, i1..iL, sr, j1..jL, sc) (= _zsrc before its transpose)"""
        T, S, R, C = A4.shape
        return fnp.reshape(A4, (T, S) + (2,) * L + (R >> L,) + (2,) * L + (C >> L,))

    @staticmethod
    def _zsrcn_sw(A4s, L):
        """opd_patch OPD_ZT: _zsrcn(swapaxes(A4s, 0, 1), L) for A4s (S, 1, R, C) without the swapaxes op (the size-1 axis moves in the reshape)"""
        S, T, R, C = A4s.shape
        if T != 1:
            return ZFMM._zsrcn(fnp.swapaxes(A4s, 0, 1), L)
        return fnp.reshape(A4s, (1, S) + (2,) * L + (R >> L,) + (2,) * L + (C >> L,))

    def _zslotsn(self, arr, L):
        """opd_patch OPD_ZT: _zslots with split-only views (root cache key tagged 'n'; the root view is sliced on its S axis = axis 1)"""
        if not PUBLIC_VIEWS:
            return self._zsrcn_sw(arr, L)
        entry = _public_info(arr)
        if arr.nbytes >= self.min_cache_bytes and entry is not None and (entry[2] is not None):
            root = entry[2]()
            if root is not None and root.ndim == 4 and (root.shape[1:] == arr.shape[1:]):
                start = entry[3]
                if 0 <= start and start + arr.shape[0] <= root.shape[0]:
                    key = (id(root), L, 'n')
                    cached = self.zb.get(key)
                    if cached is None or cached[0] is not root:
                        if len(self.zb) >= 64:
                            self.zb.clear()
                        cached = (root, self._zsrcn_sw(root, L))
                        self.zb[key] = cached
                    return cached[1][:, start:start + arr.shape[0]]
        return self._cached(arr, 'zn', L, lambda: self._zsrcn_sw(arr, L))

    @staticmethod
    def _zsrc(A4, L):
        """A4 (T, S, R, C) any strides -> view (T, i1,j1,...,iL,jL, sr, S, sc)"""
        T, S, R, C = A4.shape
        sr, sc = (R >> L, C >> L)
        V = fnp.reshape(A4, (T, S) + (2,) * L + (sr,) + (2,) * L + (sc,))
        perm = [0]
        for l in range(L):
            perm += [2 + l, L + 3 + l]
        perm += [L + 2, 1, 2 * L + 3]
        return fnp.transpose(V, tuple(perm))

    def _run_uncached(self, pl):
        L = pl.L
        px, py, po = (pl.px, pl.py, pl.po)
        hub = pl.kind == 'hub'
        for d in range(L - 1):
            bi = (_S,) * (1 + d)
            X, Xn, Y, Yn = (px[d], px[d + 1], py[d], py[d + 1])
            A11, A12, A21, A22 = (X[bi + (q,)] for q in range(4))
            fnp.copyto(Xn[bi + (slice(0, 2),)], X[bi + (slice(0, 2),)])
            fnp.copyto(Xn[bi + (3,)], A22)
            fnp.add(A21, A22, out=Xn[bi + (4,)])
            fnp.subtract(Xn[bi + (4,)], A11, out=Xn[bi + (5,)])
            fnp.subtract(A11, A21, out=Xn[bi + (6,)])
            fnp.subtract(A12, Xn[bi + (5,)], out=Xn[bi + (2,)])
            if hub:
                B11, B12, B21, B22 = (Y[bi + (q,)] for q in (0, 2, 1, 3))
                fnp.copyto(Yn[bi + (slice(0, 2),)], Y[bi + (slice(0, 2),)])
                fnp.copyto(Yn[bi + (2,)], B22)
            else:
                B11, B12, B21, B22 = (Y[bi + (q,)] for q in range(4))
                fnp.copyto(Yn[bi + (0,)], B11)
                fnp.copyto(Yn[bi + (slice(1, 3),)], Y[bi + (slice(2, 4),)])
            fnp.subtract(B12, B11, out=Yn[bi + (4,)])
            fnp.subtract(B22, Yn[bi + (4,)], out=Yn[bi + (5,)])
            fnp.subtract(B22, B12, out=Yn[bi + (6,)])
            fnp.subtract(Yn[bi + (5,)], B21, out=Yn[bi + (3,)])
        self._leaf(pl)
        for d in range(L - 2, -1, -1):
            bi = (_S,) * (1 + d)
            M, C = (po[d + 1], po[d])
            M0, M1, M2, M3, M4, M5, M6 = (M[bi + (i,)] for i in range(7))
            fnp.add(M0, M1, out=C[bi + (0,)])
            fnp.add(M5, M0, out=M5)
            fnp.add(M6, M5, out=M6)
            fnp.add(M6, M4, out=C[bi + (3,)])
            fnp.subtract(M6, M3, out=C[bi + (2,)])
            fnp.add(M4, M5, out=M4)
            fnp.add(M4, M2, out=C[bi + (1,)])

    def _leaf_uncached(self, pl):
        L = pl.L
        bi = (_S,) * L
        hub = pl.kind == 'hub'
        X, Y, C = (pl.px[L - 1], pl.py[L - 1], pl.po[L - 1])
        A11, A12, A21, A22 = (X[bi + (q,)] for q in range(4))
        if hub:
            B11, B12, B21, B22 = (Y[bi + (q,)] for q in (0, 2, 1, 3))
            C11, C12, C21, C22 = (C[bi + (q,)] for q in range(4))
            if pl.G == 1:

                def mm(a, b, o):
                    fnp.matmul(a, fnp.swapaxes(b, -1, -2), out=o)
            else:
                c1 = pl.k1 * pl.sk
                sn = pl.sn

                def mm(a, b, o):
                    fnp.matmul(_pview(a, (..., slice(None, c1, None))), fnp.swapaxes(_pview(b, (..., slice(None, c1, None))), -1, -2), out=_pview(o, (..., slice(None, sn, None))))
                    fnp.matmul(_pview(a, (..., slice(c1, None, None))), fnp.swapaxes(_pview(b, (..., slice(c1, None, None))), -1, -2), out=_pview(o, (..., slice(sn, None, None))))
        else:
            B11, B12, B21, B22 = (Y[bi + (q,)] for q in range(4))
            C11, C12, C21, C22 = (C[bi + (q,)] for q in range(4))

            def mm(a, b, o):
                fnp.matmul(a, b, out=o)
        tX, tY, tM, tP = (pl.tx, pl.ty, pl.tm, pl.tp)
        fnp.subtract(A11, A21, out=tX)
        fnp.subtract(B22, B12, out=tY)
        mm(tX, tY, C21)
        fnp.add(A21, A22, out=tX)
        fnp.subtract(B12, B11, out=tY)
        mm(tX, tY, C22)
        fnp.subtract(tX, A11, out=tX)
        fnp.subtract(B22, tY, out=tY)
        mm(tX, tY, C12)
        fnp.subtract(A12, tX, out=tX)
        mm(tX, B22, C11)
        if LEAF_TY_ALIAS:
            fnp.subtract(tY, B21, out=tY)
            mm(A22, tY, tP)
        mm(A11, B11, tM)
        fnp.add(C12, tM, out=C12)
        fnp.add(C21, C12, out=C21)
        fnp.add(C12, C22, out=C12)
        fnp.add(C22, C21, out=C22)
        fnp.add(C12, C11, out=C12)
        if not LEAF_TY_ALIAS:
            fnp.subtract(tY, B21, out=tY)
            mm(A22, tY, tP)
        fnp.subtract(C21, tP, out=C21)
        mm(A12, B21, tP)
        fnp.add(tM, tP, out=C11)

    def _mm_prep(self, X, Y, out, lev, pad=None, dry=False):
        """pad = (Xp, Yp, Op): parent views of X / Y / out with the same leading axes and larger last two axes (None: no capacity);
        the padded windows [..., :m2, :kd2] etc. are public views of the parents, so their Z-order views are cached like the originals.
        An unbatched X (bx == 1) without the needed capacity (or with a padded contraction) is staged (one copy of m x kd)."""
        bx, P, m, kd = X.shape
        by, w = (Y.shape[0], Y.shape[3])
        if P != 1:
            return None
        Xp, Yp, Op = (None, None, None) if pad is None else pad
        stage = self.deep > 0 and bx == 1
        xcapm, xcapk = (m, kd) if Xp is None else (Xp.shape[-2], Xp.shape[-1])
        big = 1 << 30
        capm = min(big if stage else xcapm, m if Op is None else Op.shape[-2])
        capk = min(big if stage else xcapk, kd if Yp is None else Yp.shape[-2])
        capw = min(w if Yp is None else Yp.shape[-1], w if Op is None else Op.shape[-1])
        L, m2, kd2, w2, staged = self._choose('mm', bx, by, m, kd, w, lev, 0, capm, capk, capw, xcapm, xcapk, stage)
        if L <= 0:
            return None
        if (kd2, w2) != (kd, w):
            Y = self._padded(Y, Yp, kd2, w2)
        if (m2, w2) != (m, w):
            out = self._padded(out, Op, m2, w2)
        if staged:
            X = self._stage_x(X, m2, kd2, dry)
        elif (m2, kd2) != (m, kd):
            X = self._padded(X, Xp, m2, kd2)
        sm, sk, sn = (m2 >> L, kd2 >> L, w2 >> L)
        if bx == 1:
            B0, S = (1, by)
            Xs = X
            if OPD_SWAP:
                # opd_patch OPD_SWAP: Ys / Os are only read when bx != 1 (bx == 1: Y and out go through _zslots) -> not formed
                Ys = Os = None
            else:
                Ys = fnp.swapaxes(Y, 0, 1)
                Os = fnp.swapaxes(out, 0, 1)
        else:
            B0, S = (by, 1)
            Xs, Ys, Os = (X, Y, out)
        return (L, bx, self._plan('mm', B0, sm, sk, sn, S, L), Xs, Ys, Os, X, Y, out)

    def mm(self, X, Y, out, lev, pad=None):
        """plain: X (bx, 1, m, kd), Y (by, 1, kd, w) -> out (by, 1, m, w); bx in {1, by}"""
        pr = self._mm_prep(X, Y, out, lev, pad)
        if pr is None:
            fnp.matmul(X, Y, out=out)
            return
        L, bx, pl, Xs, Ys, Os, X, Y, out = pr
        if OPD_ZT and ZF7:
            # opd_patch OPD_ZT: split-only operand views against the inverse-permuted plan views (same elements, fewer view ops)
            fnp.copyto(self._zinv(pl, 0), self._cached(X, 'nx%d' % bx, L, lambda: self._zsrcn(Xs, L)))
            fnp.copyto(self._zinv(pl, 1), self._zslotsn(Y, L) if bx == 1 else self._cached(Y, 'ny%d' % bx, L, lambda: self._zsrcn(Ys, L)))
            self._run(pl)
            fnp.copyto(self._zslotsn(out, L) if bx == 1 else self._cached(out, 'no%d' % bx, L, lambda: self._zsrcn(Os, L)), self._zinv(pl, 2))
            return
        fnp.copyto(pl.xz, self._cached(X, 'mx%d' % bx, L, lambda: self._zsrc(Xs, L)))
        fnp.copyto(pl.yz, self._zslots(Y, L) if bx == 1 else self._cached(Y, 'my%d' % bx, L, lambda: self._zsrc(Ys, L)))
        self._run(pl)
        fnp.copyto(self._zslots(out, L) if bx == 1 else self._cached(out, 'mo%d' % bx, L, lambda: self._zsrc(Os, L)), pl.oz)

    def _hub_prep(self, X, Y, out, lev, k1, pad=None):
        """pad = (Xp, Yp, Op) parent views (see _mm_prep); a hub never pads its contraction (the full hub width)."""
        k, P, m, kd = X.shape
        w = Y.shape[2]
        if P != 1:
            return None
        Xp, Yp, Op = (None, None, None) if pad is None else pad
        xcapm, xcapk = (m, kd) if Xp is None else (Xp.shape[-2], Xp.shape[-1])
        capm = min(xcapm, m if Op is None else Op.shape[-2])
        capw = min(w if Yp is None else Yp.shape[-2], w if Op is None else Op.shape[-1])
        L, m2, kd2, w2, _ = self._choose('hub', k, k, m, kd, w, lev, k1, capm, kd, capw, xcapm, xcapk, False)
        if L <= 0:
            return None
        if m2 != m:
            X = self._padded(X, Xp, m2, kd)
        if w2 != w:
            Y = self._padded(Y, Yp, w2, kd)
        if (m2, w2) != (m, w):
            out = self._padded(out, Op, m2, w2)
        return (L, self._plan('hub', 1, m2 >> L, kd >> L, w2 >> L, k, L, k1), X, Y, out)

    def hubg(self, X, Y, out, lev, k1, pad=None):
        """grouped hub (trimalg API): X (k, 1, m, kd), Y (k, 1, w, kd) -> out (2, 1, m, w); out[0] = sum_{k < k1} X_k Y_k^T, out[1] = sum_{k >= k1} X_k Y_k^T"""
        pr = self._hub_prep(X, Y, out, lev, k1, pad)
        if pr is None:
            k, P, m, kd = X.shape
            self._reserve('dk', k * P * m * Y.shape[2])
            T = self._arr('dk', (k, P, m, Y.shape[2]))
            fnp.matmul(X, fnp.swapaxes(Y, -1, -2), out=T)
            fnp.sum(_pview(T, slice(None, k1, None)), axis=0, out=out[0])
            fnp.sum(_pview(T, slice(k1, None, None)), axis=0, out=out[1])
            return
        L, pl, X, Y, out = pr
        if OPD_ZT and ZF7:
            # opd_patch OPD_ZT (see mm)
            fnp.copyto(self._zinv(pl, 0), self._zslotsn(X, L))
            fnp.copyto(self._zinv(pl, 1), self._zslotsn(Y, L))
            self._run(pl)
            fnp.copyto(self._cached(out, 'hgn', L, lambda: self._zsrcn_sw(out, L)), self._zinv(pl, 2))
            return
        fnp.copyto(pl.xz, self._zslots(X, L))
        fnp.copyto(pl.yz, self._zslots(Y, L))
        self._run(pl)
        fnp.copyto(self._cached(out, 'hgo', L, lambda: self._zsrc(fnp.swapaxes(out, 0, 1), L)), pl.oz)

    def hub(self, X, Y, out, lev, pad=None):
        """hub: X (k, 1, m, kd), Y (k, 1, w, kd) -> out (1, m, w) = sum_k X_k Y_k^T"""
        pr = self._hub_prep(X, Y, out, lev, 0, pad)
        if pr is None:
            k, P, m, kd = X.shape
            self._reserve('dk', k * P * m * Y.shape[2])
            T = self._arr('dk', (k, P, m, Y.shape[2]))
            fnp.matmul(X, fnp.swapaxes(Y, -1, -2), out=T)
            fnp.sum(T, axis=0, out=out)
            return
        L, pl, X, Y, out = pr
        if OPD_ZT and ZF7:
            # opd_patch OPD_ZT (see mm)
            fnp.copyto(self._zinv(pl, 0), self._zslotsn(X, L))
            fnp.copyto(self._zinv(pl, 1), self._zslotsn(Y, L))
            self._run(pl)
            fnp.copyto(self._cached(out, 'hn', L, lambda: self._zsrcn(_pview(out, None), L)), self._zinv(pl, 2))
            return
        fnp.copyto(pl.xz, self._zslots(X, L))
        fnp.copyto(pl.yz, self._zslots(Y, L))
        self._run(pl)
        fnp.copyto(self._cached(out, 'ho', L, lambda: self._zsrc(_pview(out, None), L)), pl.oz)

    def prebuild_mm(self, X, Y, out, lev, pad=None):
        pr = self._mm_prep(X, Y, out, lev, pad, dry=True)
        if pr is None:
            return
        L, bx, pl, Xs, Ys, Os, X, Y, out = pr
        if OPD_ZT and ZF7:
            # opd_patch OPD_ZT: prebuild the split-only views the steady call reads
            if X.nbytes >= 8 << 20:
                self._cached(X, 'nx%d' % bx, L, lambda: self._zsrcn(Xs, L))
            if bx == 1:
                self._zslotsn(Y, L)
                self._zslotsn(out, L)
            else:
                self._cached(Y, 'ny%d' % bx, L, lambda: self._zsrcn(Ys, L))
                self._cached(out, 'no%d' % bx, L, lambda: self._zsrcn(Os, L))
            return
        if X.nbytes >= 8 << 20:
            self._cached(X, 'mx%d' % bx, L, lambda: self._zsrc(Xs, L))
        if bx == 1:
            self._zslots(Y, L)
            self._zslots(out, L)
        else:
            self._cached(Y, 'my%d' % bx, L, lambda: self._zsrc(Ys, L))
            self._cached(out, 'mo%d' % bx, L, lambda: self._zsrc(Os, L))

    def prebuild_hub(self, X, Y, out, lev, pad=None):
        pr = self._hub_prep(X, Y, out, lev, 0, pad)
        if pr is None:
            return
        L, pl, X, Y, out = pr
        if OPD_ZT and ZF7:
            # opd_patch OPD_ZT: prebuild the split-only views the steady call reads
            self._zslotsn(X, L)
            self._zslotsn(Y, L)
            self._cached(out, 'hn', L, lambda: self._zsrcn(_pview(out, None), L))
            return
        self._zslots(X, L)
        self._zslots(Y, L)
        self._cached(out, 'ho', L, lambda: self._zsrc(_pview(out, None), L))

    def _clear_public_zviews(self):
        for pl in self.plans.values():
            pl.public_zviews = None
        self.public_zview_roots.clear()
        self.public_zview_count = 0

    def _zview_dict(self, arr):
        key = id(arr)
        entry = self.public_zview_roots.get(key)
        if entry is None or entry[0] is not arr:
            entry = (arr, {})
            self.public_zview_roots[key] = entry
        return entry[1]

    def _znode(self, arr, prefix, selector):
        views = self._zview_dict(arr)
        key = (prefix, selector)
        result = views.get(key)
        if result is None:
            pick = slice(*selector) if isinstance(selector, tuple) else selector
            result = arr[(_S,) * prefix + (pick,)]
            views[key] = result
            self.public_zview_count += 1
        return result

    def _zbatch(self, arr, prefix, selectors):
        views = self._zview_dict(arr)
        missing = [q for q in selectors if (prefix, q) not in views]
        if len(missing) < 2:
            for q in missing:
                self._znode(arr, prefix, q)
            return
        try:
            parts = fnp.unstack(arr, axis=prefix)
        except Exception:
            for q in missing:
                self._znode(arr, prefix, q)
            return
        for q in missing:
            views[prefix, q] = parts[q]
            self.public_zview_count += 1

    def _ztail(self, arr, start, stop):
        views = self._zview_dict(arr)
        key = ('tail', start, stop)
        result = views.get(key)
        if result is None:
            result = arr[..., start:stop]
            views[key] = result
            self.public_zview_count += 1
        return result

    def _zswap(self, arr):
        views = self._zview_dict(arr)
        result = views.get('swap')
        if result is None:
            result = fnp.swapaxes(arr, -1, -2)
            views['swap'] = result
            self.public_zview_count += 1
        return result

    def _zplanviews(self, pl):
        cached = pl.public_zviews
        if cached is not None:
            return cached
        if self.public_zview_count >= PUBLIC_Z_VIEW_LIMIT:
            self._clear_public_zviews()
        L = pl.L
        hub = pl.kind == 'hub'
        stages = []
        for d in range(L - 1):
            prefix = 1 + d
            X, Xn, Y, Yn = (pl.px[d], pl.px[d + 1], pl.py[d], pl.py[d + 1])
            a = tuple((self._znode(X, prefix, q) for q in range(4)))
            xn = tuple((self._znode(Xn, prefix, q) for q in (2, 3, 4, 5, 6)))
            xcopy = (self._znode(Xn, prefix, (0, 2)), self._znode(X, prefix, (0, 2)))
            b = tuple((self._znode(Y, prefix, q) for q in ((0, 2, 1, 3) if hub else range(4))))
            yn = tuple((self._znode(Yn, prefix, q) for q in (3, 4, 5, 6)))
            if hub:
                ycopy = (self._znode(Yn, prefix, (0, 2)), self._znode(Y, prefix, (0, 2)), self._znode(Yn, prefix, 2))
            else:
                ycopy = (self._znode(Yn, prefix, (1, 3)), self._znode(Y, prefix, (2, 4)), self._znode(Yn, prefix, 0))
            stages.append((a, xn, xcopy, b, yn, ycopy))
        a = tuple((self._znode(pl.px[L - 1], L, q) for q in range(4)))
        b = tuple((self._znode(pl.py[L - 1], L, q) for q in ((0, 2, 1, 3) if hub else range(4))))
        c = tuple((self._znode(pl.po[L - 1], L, q) for q in range(4)))
        products = ((pl.tx, pl.ty, c[2]), (pl.tx, pl.ty, c[3]), (pl.tx, pl.ty, c[1]), (pl.tx, b[3], c[0]), (a[3], pl.ty, pl.tp), (a[0], b[0], pl.tm), (a[1], b[2], pl.tp))
        prepared_products = []
        for left, right, dest in products:
            if not hub:
                prepared_products.append(((left, right, dest),))
            elif pl.G == 1:
                prepared_products.append(((left, self._zswap(right), dest),))
            else:
                c1, sn = (pl.k1 * pl.sk, pl.sn)
                prepared_products.append(tuple(((self._ztail(left, lo, hi), self._zswap(self._ztail(right, lo, hi)), self._ztail(dest, oo, oh)) for lo, hi, oo, oh in ((None, c1, None, sn), (c1, None, sn, None)))))
        assembly = []
        for d in range(L - 2, -1, -1):
            prefix = 1 + d
            m = tuple((self._znode(pl.po[d + 1], prefix, q) for q in range(7)))
            co = tuple((self._znode(pl.po[d], prefix, q) for q in range(4)))
            assembly.append((m, co))
        cached = (tuple(stages), (a, b, c, tuple(prepared_products)), tuple(assembly))
        pl.public_zviews = cached
        return cached

    def _run(self, pl):
        if ZF7:
            return self._run7_ab(pl) if getattr(pl, 'ab', False) else self._run7(pl)
        if not PUBLIC_Z_VIEWS:
            return self._run_uncached(pl)
        stages, leaf, assembly = self._zplanviews(pl)
        hub = pl.kind == 'hub'
        for a, xn, xcopy, b, yn, ycopy in stages:
            A11, A12, A21, A22 = a
            X2, X3, X4, X5, X6 = xn
            B11, B12, B21, B22 = b
            Y3, Y4, Y5, Y6 = yn
            fnp.copyto(xcopy[0], xcopy[1])
            fnp.copyto(X3, A22)
            fnp.add(A21, A22, out=X4)
            fnp.subtract(X4, A11, out=X5)
            fnp.subtract(A11, A21, out=X6)
            fnp.subtract(A12, X5, out=X2)
            if hub:
                fnp.copyto(ycopy[0], ycopy[1])
                fnp.copyto(ycopy[2], B22)
            else:
                fnp.copyto(ycopy[2], B11)
                fnp.copyto(ycopy[0], ycopy[1])
            fnp.subtract(B12, B11, out=Y4)
            fnp.subtract(B22, Y4, out=Y5)
            fnp.subtract(B22, B12, out=Y6)
            fnp.subtract(Y5, B21, out=Y3)
        self._leaf(pl, leaf)
        for m, c in assembly:
            M0, M1, M2, M3, M4, M5, M6 = m
            fnp.add(M0, M1, out=c[0])
            fnp.add(M5, M0, out=M5)
            fnp.add(M6, M5, out=M6)
            fnp.add(M6, M4, out=c[3])
            fnp.subtract(M6, M3, out=c[2])
            fnp.add(M4, M5, out=M4)
            fnp.add(M4, M2, out=c[1])

    def _leaf(self, pl, leaf=None):
        if not PUBLIC_Z_VIEWS and (not ZF7):
            return self._leaf_uncached(pl)
        if leaf is None:
            leaf = (self._zplanviews7(pl) if ZF7 else self._zplanviews(pl))[1]
        a, b, c, products = leaf
        A11, A12, A21, A22 = a
        B11, B12, B21, B22 = b
        C11, C12, C21, C22 = c
        tX, tY, tM, tP = (pl.tx, pl.ty, pl.tm, pl.tp)
        add, sub, mat_ = (fnp.add, fnp.subtract, fnp.matmul)
        if RRES_ON and RRES_HOLD:
            # rres_patch RRES_HOLD: keep this call's result proxies, release them together at return (one batched free)
            hold_ = []
            ap_ = hold_.append
            add = lambda a, b, out, f=fnp.add: ap_(f(a, b, out=out))
            sub = lambda a, b, out, f=fnp.subtract: ap_(f(a, b, out=out))
            mat_ = lambda a, b, out, f=fnp.matmul: ap_(f(a, b, out=out))

        def mm(index):
            for left, right, dest in products[index]:
                mat_(left, right, out=dest)
        sub(A11, A21, out=tX)
        sub(B22, B12, out=tY)
        mm(0)
        add(A21, A22, out=tX)
        sub(B12, B11, out=tY)
        mm(1)
        sub(tX, A11, out=tX)
        sub(B22, tY, out=tY)
        mm(2)
        sub(A12, tX, out=tX)
        mm(3)
        if LEAF_TY_ALIAS:
            sub(tY, B21, out=tY)
            mm(4)
        mm(5)
        add(C12, tM, out=C12)
        add(C21, C12, out=C21)
        add(C12, C22, out=C12)
        add(C22, C21, out=C22)
        add(C12, C11, out=C12)
        if not LEAF_TY_ALIAS:
            sub(tY, B21, out=tY)
            mm(4)
        sub(C21, tP, out=C21)
        mm(6)
        add(tM, tP, out=C11)

    def prewarm_public_nodes(self, max_time_s, max_views):
        import time as _time
        t0 = _time.perf_counter()
        if ZF7:
            before = self.public_zview_count
            prepared = 0
            for pl in self.plans.values():
                if _time.perf_counter() - t0 >= max_time_s or self.public_zview_count - before >= max_views:
                    break
                self._zplanviews7(pl)
                prepared += 1
            self.public_z_prewarm_s = _time.perf_counter() - t0
            self.public_z_prewarm_eligible = len(self.plans)
            self.public_z_prewarm_touched = prepared
            self.public_z_prewarm_new_views = self.public_zview_count - before
            self.public_z_prewarm_ready_plans = prepared
            return
        records = {}
        plan_nodes = []
        for pl in self.plans.values():
            needed = set()

            def want(arr, prefix, selector):
                key = (id(arr), prefix, selector)
                if key in needed:
                    return
                needed.add(key)
                item = records.get(key)
                if item is None:
                    records[key] = [arr, prefix, selector, 1]
                else:
                    item[3] += 1
            L = pl.L
            hub = pl.kind == 'hub'
            for d in range(L - 1):
                prefix = 1 + d
                X, Xn, Y, Yn = (pl.px[d], pl.px[d + 1], pl.py[d], pl.py[d + 1])
                for q in range(4):
                    want(X, prefix, q)
                want(Xn, prefix, (0, 2))
                want(X, prefix, (0, 2))
                for q in (3, 4, 5, 6, 2):
                    want(Xn, prefix, q)
                for q in (0, 2, 1, 3) if hub else range(4):
                    want(Y, prefix, q)
                if hub:
                    want(Yn, prefix, (0, 2))
                    want(Y, prefix, (0, 2))
                    want(Yn, prefix, 2)
                else:
                    want(Yn, prefix, 0)
                    want(Yn, prefix, (1, 3))
                    want(Y, prefix, (2, 4))
                for q in (4, 5, 6, 3):
                    want(Yn, prefix, q)
            for q in range(4):
                want(pl.px[L - 1], L, q)
            for q in (0, 2, 1, 3) if hub else range(4):
                want(pl.py[L - 1], L, q)
            for q in range(4):
                want(pl.po[L - 1], L, q)
            for d in range(L - 2, -1, -1):
                prefix = 1 + d
                for q in range(7):
                    want(pl.po[d + 1], prefix, q)
                for q in range(4):
                    want(pl.po[d], prefix, q)
            plan_nodes.append((pl, needed))
        ranked = sorted(records.items(), key=lambda item: -item[1][3])
        integer_groups = {}
        for key, (arr, prefix, selector, frequency) in ranked[:max_views]:
            if isinstance(selector, int):
                integer_groups.setdefault((id(arr), prefix), []).append(selector)
        before = self.public_zview_count
        ready = set()
        touched = 0
        for key, (arr, prefix, selector, frequency) in ranked:
            if touched >= max_views or _time.perf_counter() - t0 >= max_time_s:
                break
            if isinstance(selector, int):
                selectors = integer_groups.pop((id(arr), prefix), None)
                if selectors is not None and len(selectors) > 1:
                    self._zbatch(arr, prefix, selectors)
            self._znode(arr, prefix, selector)
            ready.add(key)
            touched += 1
        prepared = 0
        for pl, needed in plan_nodes:
            if _time.perf_counter() - t0 >= max_time_s:
                break
            if needed.issubset(ready):
                self._zplanviews(pl)
                prepared += 1
        self.public_z_prewarm_s = _time.perf_counter() - t0
        self.public_z_prewarm_eligible = len(records)
        self.public_z_prewarm_touched = touched
        self.public_z_prewarm_new_views = self.public_zview_count - before
        self.public_z_prewarm_ready_plans = prepared
TA_TIER2 = _os.environ.get('TA_TIER2', '1') == '1'
TA_TIER2_LEVEL = int(_os.environ.get('TA_TIER2_LEVEL', '4'))
TA_TIER2_MIN = int(_os.environ.get('TA_TIER2_MIN', '12'))
TA_TIER2_ROT_SLOTS = int(_os.environ.get('TA_TIER2_ROT_SLOTS', '4'))

def _fx_tri_solve(R, B, d):
    """FX_TRI: R X = B for upper-triangular R (r x r) by d levels of block bisection (X2 = R22^-1 B2; X1 = R11^-1 (B1 - R12 X2)), dense
    solves at the leaves.  Same solution as fnp.linalg.solve(R, B) up to rounding; bills ~(1 + 2^-d) r^2 nrhs instead of 2 r^2 nrhs."""
    r = R.shape[0]
    if d <= 0 or r % 2:
        return fnp.linalg.solve(R, B)
    h = r // 2
    X2 = _fx_tri_solve(R[h:, h:], B[h:], d - 1)
    X1 = _fx_tri_solve(R[:h, :h], fnp.subtract(B[:h], fnp.matmul(R[:h, h:], X2)), d - 1)
    return fnp.concatenate([X1, X2], axis=0)


# agents_joins QR knobs (2026-10-16); '0' = the parent op stream
QRB = _os.environ.get('QRB', '1') == '1'   # frozen-join Qp^+ from R only (QR mode 'r') + two block-bisection solves
QRV = _os.environ.get('QRV', '1') == '1'   # FIJA2 first-union complement by QR(Ur, 'complete')


# agents_joins round 3 (2026-10-16); '0' = the parent op stream
J3_TRI = _os.environ.get('J3_TRI', '1') == '1'       # QRB solves by column-block substitution (J3_TRI_NB diagonal blocks)
J3_TRI_NB = int(_os.environ.get('J3_TRI_NB', '16'))


def _j3_qrb_pinv(Rq, Qp, nb):
    """J3_TRI: Qp^+ = Rq^-1 Rq^-T Qp^T by right-looking column-block substitution with nb equal diagonal blocks (dense solves on the
    diagonal blocks, one rank-s update of the remaining right-hand sides per block, in place on a working copy): (1 + 1/nb) r^2 kj per
    solve instead of (1 + 2^-FX_TRI) r^2 kj, 3 nb ops per solve"""
    r = Rq.shape[0]
    s = r // nb
    if RRES_ON and RRES_J3:
        # rres_patch: the same ops on the same operands in the same order; Rq[a:b, c:d].T is the view RqT[c:d, a:b] (same elements,
        # shape, strides, offset) and Bw[b:] / Y[:a] are formed once per step for the read and the out= target
        Bw = fnp.copy(Qp.T)
        RqT = Rq.T
        ys = []
        for i in range(nb):
            a, b = i * s, (i + 1) * s
            Yi = fnp.linalg.solve(RqT[a:b, a:b], Bw[a:b])
            ys.append(Yi)
            if i < nb - 1:
                t_ = Bw[b:]
                fnp.subtract(t_, fnp.matmul(RqT[b:, a:b], Yi), out=t_)
        Y = fnp.concatenate(ys, axis=0)
        xs = []
        for i in range(nb - 1, -1, -1):
            a, b = i * s, (i + 1) * s
            Xi = fnp.linalg.solve(Rq[a:b, a:b], Y[a:b])
            xs.append(Xi)
            if i > 0:
                t_ = Y[:a]
                fnp.subtract(t_, fnp.matmul(Rq[:a, a:b], Xi), out=t_)
        return fnp.concatenate(xs[::-1], axis=0)
    Bw = fnp.copy(Qp.T)
    ys = []
    for i in range(nb):
        a, b = i * s, (i + 1) * s
        Yi = fnp.linalg.solve(Rq[a:b, a:b].T, Bw[a:b])
        ys.append(Yi)
        if i < nb - 1:
            fnp.subtract(Bw[b:], fnp.matmul(Rq[a:b, b:].T, Yi), out=Bw[b:])
    Y = fnp.concatenate(ys, axis=0)
    xs = []
    for i in range(nb - 1, -1, -1):
        a, b = i * s, (i + 1) * s
        Xi = fnp.linalg.solve(Rq[a:b, a:b], Y[a:b])
        xs.append(Xi)
        if i > 0:
            fnp.subtract(Y[:a], fnp.matmul(Rq[:a, a:b], Xi), out=Y[:a])
    return fnp.concatenate(xs[::-1], axis=0)


def _fx_tri_solve_l(Lw, B, d):
    """QRB: Lw X = B for lower-triangular Lw (r x r; QRB passes Rq.T, a free view) by d levels of block bisection (X1 = L11^-1 B1;
    X2 = L22^-1 (B2 - L21 X1)), dense solves at the leaves: the mirror of _fx_tri_solve (same billing, same op count)"""
    r = Lw.shape[0]
    if d <= 0 or r % 2:
        return fnp.linalg.solve(Lw, B)
    h = r // 2
    X1 = _fx_tri_solve_l(Lw[:h, :h], B[:h], d - 1)
    X2 = _fx_tri_solve_l(Lw[h:, h:], fnp.subtract(B[h:], fnp.matmul(Lw[h:, :h], X1)), d - 1)
    return fnp.concatenate([X1, X2], axis=0)


def _tier2_mm(smm, left, right, out):
    smm.mn = TA_TIER2_MIN
    smm.mm(_pview(left, (None, None)), right, out, smm.level(left.shape[0], left.shape[1], right.shape[-1], TA_TIER2_LEVEL))
    smm.mn = STRASSEN_MIN

def _qc_next(smm, WD, Qc, out, on):
    if not on:
        return _public_out(fnp.matmul, WD, Qc, out=out)
    smm.mn = TA_JOIN_MIN
    smm.mm(_pview(WD, (None, None)), _pview(Qc, (None, None)), _pview(out, (None, None)), smm.level(WD.shape[0], WD.shape[1], Qc.shape[1], TA_JOIN_LEV))
    smm.mn = STRASSEN_MIN
    return out

C7_S = frozenset((int(x) for x in _os.environ.get('C7_S', '1,2,3,5').split(',') if x.strip()))   # cost7: small dense products routed through the Z core
C7_LEV = int(_os.environ.get('C7_LEV', '4'))
C7_MIN = int(_os.environ.get('C7_MIN', '12'))


def _c7mm(smm, site, X, Y, out):
    """cost7: out <- X @ Y (2-D) through the pooled Strassen core when site is enabled (dense when no level fits)"""
    if site not in C7_S:
        return fnp.matmul(X, Y, out=out)
    smm.mn = C7_MIN
    smm.mm(_pview(X, (None, None)), _pview(Y, (None, None)), _pview(out, (None, None)), smm.level(X.shape[0], X.shape[1], Y.shape[1], C7_LEV))
    smm.mn = STRASSEN_MIN
    return out

# agents_dense DN_* (2026-10-15): exact / rounding-level cuts of the non-Strassen dense work; every knob '0' = the parent op stream.
DN_YCOL = _os.environ.get('DN_YCOL', '1') == '1'        # thin legs: the zero y column of L dropped from MP / PPL; R^T Yk folded into the thin einsum
DN_THIN2D = DN_YCOL and _os.environ.get('DN_THIN2D', '1') == '1'   # gated thin part + feedback rank-rfb sums as ONE Z-core hub product
DN_THIN2D_LMIN = int(_os.environ.get('DN_THIN2D_LMIN', '8'))
DN_THIN2D_LEV = int(_os.environ.get('DN_THIN2D_LEV', '4'))
DN_Z2D = _os.environ.get('DN_Z2D', '1') == '1'          # thin-leg transport WD [Z | Zf] as ONE Z-core product
DN_Z2D_LMIN = int(_os.environ.get('DN_Z2D_LMIN', '9'))
DN_Z2D_LEV = int(_os.environ.get('DN_Z2D_LEV', '4'))
DN_ZKIN = _os.environ.get('DN_ZKIN', '1') == '1'        # thin-leg transport over the first kin rows (w1_prev rows [kin:] are zero)
DN_FROT = _os.environ.get('DN_FROT', '1') == '1'        # FIJA2 first-union rotation Vr^T [FA | FP] and Qn Vr through the Z core
DN_FROT_LEV = int(_os.environ.get('DN_FROT_LEV', '4'))
DN_G0 = _os.environ.get('DN_G0', '1') == '1'            # layer-0 Gram: two half Grams + one Z-core off-diagonal product
DN_G0_LEV = int(_os.environ.get('DN_G0_LEV', '5'))
DN_EIGH = _os.environ.get('DN_EIGH', '0') == '1'        # APPROXIMATION: FIJA2 ordering by the exact top-r2 eigenspace (eigh)
DN_FSUB = _os.environ.get('DN_FSUB', '1') == '1'        # (below the ops bar) FIJA2 subspace iteration Sg @ Ur through the Z core (DN_EIGH off)
DN_FSUB_LEV = int(_os.environ.get('DN_FSUB_LEV', '4'))
DN_KP = _os.environ.get('DN_KP', '0') == '1'            # (marginal) FIJA2 tier-2 Kp = R11^-1 R12 by the FX_TRI block bisection (R11 upper triangular)
DN_WMAX = 288                                           # 2-D staging width (>= 18 (L-1) = 270 thin columns, room for the 16-grid padding


def _dn_pad(m, kd, w, lev, mn, pad_kd):
    """DN: (L, padded size) -- the deepest L <= lev at which every side divides 2^L and keeps >= mn per block, the padded side
    (the contraction kd when pad_kd, else the width w) rounded up to the 2^L grid"""
    for L_ in range(lev, 0, -1):
        d = 1 << L_
        if pad_kd:
            p = -(-kd // d) * d
            if m % d == 0 and w % d == 0 and min(m, p, w) // d >= mn:
                return (L_, p)
        else:
            p = -(-w // d) * d
            if m % d == 0 and kd % d == 0 and min(m, kd, p) // d >= mn:
                return (L_, p)
    return (0, kd if pad_kd else w)


def _dn_mm(smm, X, Y, out, lev):
    """DN: out <- X @ Y through the pooled Z core (4-D operands as for smm.mm), level capped at lev, min block C7_MIN"""
    smm.mn = C7_MIN
    smm.mm(X, Y, out, smm.level(X.shape[-2], X.shape[-1], Y.shape[-1], lev))
    smm.mn = STRASSEN_MIN

# thin_patch (agents_ideas2 L2 msp, 2026-10-17): thin dense repair products through the pooled Z core; '0' = the parent op stream
THIN_ENG = _os.environ.get('THIN_ENG', '1') == '1'
THIN_SITES = frozenset((s_.strip() for s_ in _os.environ.get('THIN_SITES', 'kwin,csplit').split(',') if s_.strip()))
THIN_LEV = int(_os.environ.get('THIN_LEV', '4'))
THIN_MIN = int(_os.environ.get('THIN_MIN', '2'))
THIN_OPW = float(_os.environ.get('THIN_OPW', '3e5'))


def _thin_level(m, kd, w):
    """THIN_ENG: the Z-core level of a dense (m, kd) @ (kd, w) product (0 = keep the dense matmul): the level L <= THIN_LEV (every side
    divisible by 2^L, leaf blocks >= THIN_MIN) maximising the exact modeled saving (run + uncached operand reshapes) minus THIN_OPW per
    added op (15 L + 20 ops per call, measured on the production core by thin_bench.py)"""
    dense = m * w * (2 * kd - 1) - (m * kd + kd * w + m * w)
    best, bv = (0, 0.0)
    for L in range(1, THIN_LEV + 1):
        d = 1 << L
        if m % d or kd % d or w % d or min(m, kd, w) // d < THIN_MIN:
            break
        v = dense - _lx_cost('mm', 1, m >> L, kd >> L, w >> L, 1, L, 1, 0) - THIN_OPW * (15 * L + 20)
        if v > bv:
            best, bv = (L, v)
    return best


def _thin_mm(smm, X, Y, out, L):
    """THIN_ENG: out <- X @ Y (2-D operands) through the pooled Z core at level L (min block THIN_MIN; the core runs dense if no plan fits)"""
    smm.mn = THIN_MIN
    try:
        smm.mm(_pview(X, (None, None)), _pview(Y, (None, None)), _pview(out, (None, None)), L)
    finally:
        smm.mn = STRASSEN_MIN
    return out


class _ZW(ZFMM):

    def prebuild_hubg(self, X, Y, out, lev, k1, pad=None):
        pr = self._hub_prep(X, Y, out, lev, k1, pad)
        if pr is None:
            return
        L, pl, X, Y, out = pr
        if OPD_ZT and ZF7:
            # opd_patch OPD_ZT: prebuild the split-only views the steady call reads
            self._zslotsn(X, L)
            self._zslotsn(Y, L)
            self._cached(out, 'hgn', L, lambda: self._zsrcn_sw(out, L))
            return
        self._zslots(X, L)
        self._zslots(Y, L)
        self._cached(out, 'hgo', L, lambda: self._zsrc(fnp.swapaxes(out, 0, 1), L))

class _FMMd:
    """Strassen backend of the merge estimator: Z-order core (suite shape) or the merge's own `_Strassen` (FMM_OFF / off-suite / after a failure).
    `steady` (set by the call site before a call, first predict only): the level the SAME call will use from the second predict on; the plans and
    cached Z views of the pooled caller arrays at that level are built right after the call (their one-time op / FLOP cost moves into the first
    predict, which runs the capped levels and has the slack); calls that already run at their steady level register their plans as steady plans."""

    def __init__(self, dtype, bmax, use_z=True):
        self.dtype = dtype
        self.bmax = int(bmax)
        self.old = _Strassen(dtype, bmax)
        self.zmode = bool(use_z) and (not FMM_OFF)
        self.z = _ZW(dtype, STRASSEN_MIN, FMM_HINT_MM, FMM_HINT_HUB, FMM_CAP_MM) if self.zmode else None
        self.first = False
        self.steady = 0
        self.ncalls = 0
        self.dense = False
        self.as_mode = False
        self._mn = STRASSEN_MIN

    @property
    def mn(self):
        return self._mn

    @mn.setter
    def mn(self, v):
        self._mn = v
        self.old.mn = v
        if self.z is not None:
            self.z.mn = v

    def level(self, m, kd, w, lev):
        z = self.z
        if z is None:
            return self.old.level(m, kd, w, lev)
        if PAD_DEEP:
            return lev if lev > 0 else 0
        return z.level(m, kd, w, lev)

    def deep_mode(self, on):
        """cost-model level choice of the Z core (padded deeper recursion): the call sites then pass their level CAP through
        `level()` and the core picks today's level as the floor and a deeper padded level when the model says it pays"""
        if self.z is not None:
            self.z.deep = PAD_LMAX if on and PAD_DEEP else 0

    def _old_args(self, name, args):
        """the merge's own `_Strassen` needs the exact feasible level (it runs dense when `lev` does not divide)"""
        X, Y = (args[0], args[1])
        m, kd = (X.shape[2], X.shape[3])
        w = Y.shape[2] if name != 'mm' else Y.shape[3]
        return (X, Y, args[2], self.old.level(m, kd, w, args[3])) + tuple(args[4:])

    def finish_first(self):
        if self.z is not None:
            try:
                self.z.rebuild_steady()
            except Exception:
                if FMM_STRICT:
                    raise

    def _call(self, name, args, pre, pad=None):
        st, self.steady = (self.steady, 0)
        z = self.z
        if z is None or (not FMM_OLD_ZCORE and self._mn != TA_YH_MIN):
            return getattr(self.old, name)(*self._old_args(name, args))
        if self.dense:
            return getattr(z, name)(*_pview(args, slice(None, 3, None)) + (0,) + _pview(args, slice(4, None, None)))
        try:
            if M2_FAIL_AFTER >= 0:
                self.ncalls += 1
                if self.ncalls == M2_FAIL_AFTER:
                    raise RuntimeError('M2 test failure')
            z.mark = self.first and (not st)
            getattr(z, name)(*args, pad=pad)
            if st and self.first:
                z.mark = True
                deep = z.deep
                z.deep = PAD_LMAX if PAD_DEEP else 0
                try:
                    getattr(z, pre)(*_pview(args, slice(None, 3, None)), st, *_pview(args, slice(4, None, None)), pad=pad)
                finally:
                    z.deep = deep
            z.mark = False
            return None
        except Exception:
            if FMM_STRICT:
                raise
            if self.as_mode:
                self.dense = True
                return getattr(z, name)(*_pview(args, slice(None, 3, None)) + (0,) + _pview(args, slice(4, None, None)))
            self.z = None
            return getattr(self.old, name)(*self._old_args(name, args))

    def mm(self, X, Y, out, lev, pad=None):
        return self._call('mm', (X, Y, out, lev), 'prebuild_mm', pad)

    def hub(self, X, Y, out, lev, pad=None):
        return self._call('hub', (X, Y, out, lev), 'prebuild_hub', pad)

    def hubg(self, X, Y, out, lev, k1, pad=None):
        return self._call('hubg', (X, Y, out, lev, k1), 'prebuild_hubg', pad)

class _Pool:
    """V27 (F80): persistent named scratch buffers. A fresh (n,n) result costs ~0.10 ms
    of residual (allocation + wrapper), the same op written with out= into a pooled
    buffer ~0.013 ms; the pool lives on the estimator across predict() calls (per-MLP
    allocation made the residual creep over a suite, F79). Keys carry the shape, so a
    different width simply adds buffers."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.bufs = {}
        self.alias_work = False
        self.factor_profile = None
        self.public_views = {}
        self.public_view_hits = 0
        self.public_view_misses = 0
        self.public_view_clears = 0

    def get(self, name, shape):
        dims0 = tuple((int(x) for x in shape))
        if self.alias_work and len(dims0) == 2 and (dims0[0] == dims0[1]) and (dims0[0] in (1024, AS_TEST_N)):
            name = {'wk4m': 'pk11', 'wk431': 'pk22', 'rep21': 'd21w', 's21': 'k21', 'ssep': 't1'}.get(name, name)
            if name in ('d3col', 'g4col'):
                colkey = (('shared_pk_view', name), dims0)
                b = self.bufs.get(colkey)
                if b is None:
                    group = 0 if name == 'd3col' else 1
                    b = self.get('pk2', (3, dims0[0], dims0[1]))[group]
                    self.bufs[colkey] = b
                return b
        key = (name, dims0)
        aliases = {'ap': 0, 'pp': 0, 't': 0, 'mp': 1, 'yt': 1, 'xt': 2, 'u': 3}
        dims = key[1]
        suite_work = len(dims) == 3 and dims[1] == dims[2] and (dims[1] in (1024, AS_TEST_N)) and (name in aliases and dims[0] == 15 or (name == 'abbuf' and dims[0] <= 60))
        if self.alias_work and name in ('sym_x', 'sym_y', 'sym_o') and (len(dims) == 4):
            nn = 2 * dims[3] if name == 'sym_o' else max(_pview(dims, slice(2, None, None)))
            syshapes = {'sym_x': (3, 1, nn // 2, nn), 'sym_y': (3, 1, nn, nn // 2), 'sym_o': (3, 1, nn // 2, nn // 2)}
            if nn in (1024, AS_TEST_N) and dims == syshapes[name]:
                backing_key = ('dslices_ab_alias', nn)
                backing = self.bufs.get(backing_key)
                if backing is None:
                    backing = fnp.empty((60 * nn * nn,), dtype=self.dtype)
                    fnp.copyto(backing, 0.0)
                    self.bufs[backing_key] = backing
                key = (('shared_work_sym', name), dims)
                b = self.bufs.get(key)
                if b is None:
                    starts = {'sym_x': 0, 'sym_y': 3 * nn * nn // 2, 'sym_o': 3 * nn * nn}
                    start = 2 * 15 * nn * nn + starts[name]
                    b = fnp.reshape(_pview(backing, slice(start, start + math.prod(dims), None)), dims)
                    self.bufs[key] = b
                return b
        if suite_work and self.alias_work:
            key = (('shared_work_view', name), key[1])
            n = dims[1]
            backing_key = ('dslices_ab_alias', n)
            backing = self.bufs.get(backing_key)
            if backing is None:
                backing = fnp.empty((60 * n * n,), dtype=self.dtype)
                fnp.copyto(backing, 0.0)
                self.bufs[backing_key] = backing
            b = self.bufs.get(key)
            if b is None:
                start = 0 if name == 'abbuf' else aliases[name] * 15 * n * n
                b = fnp.reshape(_pview(backing, slice(start, start + dims[0] * n * n, None)), dims)
                self.bufs[key] = b
            return b
        b = self.bufs.get(key)
        if b is None:
            b = fnp.empty(key[1], dtype=self.dtype)
            fnp.copyto(b, 0.0)
            self.bufs[key] = b
        return b

    def get_pair(self, name, shape):
        """V28: an (S, 2, a, b) slot buffer together with its one-time (2S, 1, a, b) view
        (fnp.reshape is a logged, per-element billed op; slabs of whole slots stay
        contiguous, so [2i:2j] of the view is slots i..j-1 with the pair axis folded
        into the Strassen batch)."""
        shape = tuple((int(x) for x in shape))
        profile = self.factor_profile
        if FACTOR_POOL_CAP and (not FIJA2) and self.alias_work and (len(shape) == 4) and (shape[0] == 15) and (shape[1] == 2) and (shape[3] in (1024, AS_TEST_N)) and isinstance(name, tuple) and (name[0] in ('fap', 'fap2')) and (profile is not None) and (_pview(profile, slice(None, 3, None)) == (16, 4, 7)) and (0 < profile[4] < profile[3] < shape[3]) and (shape[2] == (profile[3] if name[0] == 'fap' else profile[4])) and (TD_GATE_B < 0) and (TD_GATE2_B < 0) and (TD_FROZEN == frozenset((6, 8, 10, 12, 14))) and (not FULL_LAST) and (not NO_CONFINE):
            shape = (5 if name[0] == 'fap' else 8,) + _pview(shape, slice(1, None, None))
        key = ('pair', name, shape)
        e = self.bufs.get(key)
        if e is None:
            b5 = fnp.empty(key[2], dtype=self.dtype)
            fnp.copyto(b5, 0.0)
            b4 = fnp.reshape(b5, (key[2][0] * key[2][1], 1, key[2][2], key[2][3]))
            e = (b5, b4)
            self.bufs[key] = e
        return e
import weakref as _public_weakref
PUBLIC_VIEWS = _os.environ.get('PUBLIC_VIEWS', '1') == '1'
ARR_GUARD = int(_os.environ.get('ARR_GUARD', '85000'))        # agents_robust: > 0 -> drop all view caches at predict start above this many live handles
ARR_GUARD_TEST = _os.environ.get('ARR_GUARD_TEST', '0') == '1'
ARR_GUARD_HZ = _os.environ.get('ARR_GUARD_HZ', '1') == '1'   # (on fire) also drop the HZ view caches + HZE product plans (rebuilt lazily)
import weakref as _rb_weakref


def _rb_live_handles():
    """agents_robust ARR_GUARD: live finalizers in this process = live flopscope RemoteArray handles in RPC mode (-1 if unavailable)"""
    try:
        return len(_rb_weakref.finalize._registry)
    except Exception:
        return -1


def _rb_arr_clear(est):
    """agents_robust ARR_GUARD: drop every re-creatable view cache (no flopscope op; each one is cleared by the parent code when full)"""
    smm = getattr(est, '_smm', None)
    z = getattr(smm, 'z', None) if smm is not None else None
    if z is not None:
        z._clear_public_zviews()
        z.vc.clear()
        z.public_cache_inputs.clear()
        z.zb.clear()
    pool = getattr(est, '_pool', None)
    if pool is not None:
        pool.public_views.clear()
        pool.public_view_clears += 1
    au = est.__dict__.get('_au_vcache')
    if au is not None:
        au.clear()
    H = getattr(est, '_hz', None) if ARR_GUARD_HZ else None
    if H is not None:
        # HZ getters (_hz_sv / _hz_dv / _hz_stage / _hz_slab ...) look H['vc'] up on every call; the HZE product plans hold views of E.vc,
        # so E.vc is only freed together with E.plans / E.bplans (rebuilt lazily by plan() / bplan(); the gate decisions read H['gp']
        # keys, never plan presence, so the billed op stream is unchanged)
        H['vc'].clear()
        for e in (H.get('E'), H.get('E6')):
            if e is not None:
                e.vc.clear()
                e.plans.clear()
                e.bplans.clear()
    est.rb_arr_clears = getattr(est, 'rb_arr_clears', 0) + 1
_PUBLIC_ARRAYS = {}
_PUBLIC_VIEW_LIMIT = int(_os.environ.get('PUBLIC_VIEW_LIMIT', '4096'))

def _public_info(arr):
    entry = _PUBLIC_ARRAYS.get(id(arr))
    if entry is None or entry[0]() is not arr or entry[1]() is None:
        return None
    return entry

def _public_register(arr, owner, root=None, start=0, logical_root=None, path=()):
    if not PUBLIC_VIEWS:
        return
    identity = id(arr)
    try:
        ar = _public_weakref.ref(arr, lambda ignored, key=identity: _PUBLIC_ARRAYS.pop(key, None))
        ow = _public_weakref.ref(owner)
        rr = None if root is None else _public_weakref.ref(root)
        lr = _public_weakref.ref(arr if logical_root is None else logical_root)
    except TypeError:
        return
    _PUBLIC_ARRAYS[identity] = (ar, ow, rr, int(start), lr, path)

def _public_key(index):
    # Most indices are flat tuples. Inline leaf normalization to avoid recursive
    # function calls and three-element generators on every public view lookup.
    if isinstance(index, tuple):
        parts = []
        for value in index:
            if isinstance(value, slice):
                a, b, c = value.start, value.stop, value.step
                if not ((a is None or type(a) is int) and (b is None or type(b) is int) and (c is None or type(c) is int)):
                    return None
                parts.append(('slice', a, b, c))
            elif value is None:
                parts.append(('newaxis',))
            elif value is Ellipsis:
                parts.append(('ellipsis',))
            elif type(value) is int:
                parts.append(('int', value))
            elif isinstance(value, tuple):
                child = _public_key(value)
                if child is None:
                    return None
                parts.append(child)
            else:
                return None
        return ('tuple',) + tuple(parts)
    if index is None:
        return ('newaxis',)
    if index is Ellipsis:
        return ('ellipsis',)
    if type(index) is int:
        return ('int', index)
    if isinstance(index, slice):
        a, b, c = index.start, index.stop, index.step
        if (a is None or type(a) is int) and (b is None or type(b) is int) and (c is None or type(c) is int):
            return ('slice', a, b, c)
    return None

def _public_slots(parent, view, index, entry):
    if parent.ndim != 4 or view.ndim != 4 or entry[2] is None:
        return (None, 0)
    root = entry[2]()
    if root is None or root.shape[1:] != view.shape[1:]:
        return (None, 0)
    indices = index if isinstance(index, tuple) else (index,)
    if any((v is None for v in indices)):
        return (None, 0)
    indices = list(indices)
    ell = [j for j, v in enumerate(indices) if v is Ellipsis]
    if len(ell) > 1:
        return (None, 0)
    if ell:
        j = ell[0]
        indices[j:j + 1] = [slice(None)] * (4 - len(indices) + 1)
    indices += [slice(None)] * (4 - len(indices))
    if len(indices) != 4 or not all((isinstance(v, slice) for v in indices)):
        return (None, 0)
    first = indices[0].indices(parent.shape[0])
    if first[2] != 1:
        return (None, 0)
    for dim, selection in zip(parent.shape[1:], indices[1:]):
        if selection.indices(dim) != (0, dim, 1):
            return (None, 0)
    return (root, entry[3] + first[0])

def _pview(arr, index):
    if not PUBLIC_VIEWS:
        return arr[index]
    entry = _public_info(arr)
    if entry is None:
        return arr[index]
    selection = _public_key(index)
    if selection is None:
        return arr[index]
    owner = entry[1]()
    key = (id(arr), selection)
    found = owner.public_views.get(key)
    if found is not None and found[0] is arr:
        owner.public_view_hits += 1
        return found[1]
    view = arr[index]
    if len(owner.public_views) >= _PUBLIC_VIEW_LIMIT:
        owner.public_views.clear()
        owner.public_view_clears += 1
    owner.public_views[key] = (arr, view)
    owner.public_view_misses += 1
    root, start = _public_slots(arr, view, index, entry)
    _public_register(view, owner, root, start, entry[4](), entry[5] + (selection,))
    return view
_public_original_pool_get = _Pool.get
_public_original_pool_pair = _Pool.get_pair

def _public_pool_get(self, name, shape):
    value = _public_original_pool_get(self, name, shape)
    root = value if value.ndim == 4 and value.shape[1] == 1 else None
    _public_register(value, self, root)
    return value

def _public_pool_pair(self, name, shape):
    pair = _public_original_pool_pair(self, name, shape)
    _public_register(pair[0], self)
    _public_register(pair[1], self, pair[1])
    return pair
_Pool.get = _public_pool_get
_Pool.get_pair = _public_pool_pair
PUBLIC_OUT_IDENTITY = _os.environ.get('PUBLIC_OUT_IDENTITY', '1') == '1'
PUBLIC_BUFFER_READY = _os.environ.get('PUBLIC_BUFFER_READY', '1') == '1'

def _public_out(operation, *args, **kwargs):
    """Dispatch once, then retain the caller-owned public destination object.

    All wrapped sites pass one ndarray destination. Keeping that object lets
    the bounded logical-view registry reuse views after an out operation.
    No pointer, stride, remote handle, storage metadata or cost state is used.
    """
    result = operation(*args, **kwargs)
    destination = kwargs.get('out')
    if not PUBLIC_OUT_IDENTITY or destination is None:
        return result
    return destination

class _HZE:
    """Classic Strassen products X @ W in Morton (Z) order for the deferred kappa4 pullback.

    A product (r x c) @ (c x w) with L levels works on leaf stacks (7^d 4^(L-d), r/2^L, c/2^L): in Z order the four quadrants of every
    block are consecutive along one axis, so each level is four elementwise calls on strided views (A side, C side) and the right
    operand tree is shared by all row chunks of a stage.  All scratch arrays and views are created once and reused across predicts
    (contents are always fully overwritten before they are read)."""

    def __init__(self, L, rmax, cmax):
        self.L = L
        self.rmax = rmax            # largest leaf row count (chunk rows / 2^L)
        self.cmax = cmax            # largest leaf column count (n / 2^L)
        self.A = None               # per depth d: (flat, Q-form, S-form, 3-D form) of the left/result nodes
        self.M = None
        self.MS = None
        self.B = None
        self.vc = {}                # view cache
        self.plans = {}
        self.bplans = {}

    def alloc(self):
        """create the scratch arrays and their maximal-shape forms (the only reshapes of big arrays)"""
        L, rm, cm = (self.L, self.rmax, self.cmax)
        f32 = fnp.float32

        def forms(d, r, c):
            if HZ_L6 and HZ_AB and d < L:
                return None
            cnt = 7 ** d * 4 ** (L - d)
            flat = fnp.empty((cnt * r * c,), dtype=f32)
            full = fnp.reshape(flat, (cnt, r, c))
            q = fnp.reshape(flat, (7 ** d, 4, 4 ** (L - d - 1), r, c)) if d < L else None
            s = fnp.reshape(flat, (7 ** (d - 1), 7, 4 ** (L - d), r, c)) if d > 0 else None
            dg = fnp.reshape(flat, (2, 2) * L + (r, c)) if d == 0 else None
            return (full, q, s, dg)
        self.A = [forms(d, rm, cm) for d in range(L + 1)]
        self.M = fnp.empty((7 ** L, rm, cm), dtype=f32)
        self.MS = fnp.reshape(self.M, (7 ** (L - 1), 7, 1, rm, cm))
        self.B = [forms(d, cm, cm) for d in range(L + 1)]
        if HZ_AB:
            # HZ_AB: the depth-L buffers as in-place 7-ary trees (7,)*L and the root contents in digit form (views only): entries of
            # X, W and the result at slots 0..3 of every axis; the shallower depth buffers stay allocated but unused
            self.A7 = fnp.reshape(self.A[L][0], (7,) * L + (rm, cm))
            self.B7 = fnp.reshape(self.B[L][0], (7,) * L + (cm, cm))
            self.M7 = fnp.reshape(self.M, (7,) * L + (rm, cm))
            self.D7 = {'A': fnp.reshape(self.A7[(slice(0, 4),) * L], (2, 2) * L + (rm, cm)), 'B': fnp.reshape(self.B7[(slice(0, 4),) * L], (2, 2) * L + (cm, cm)), 'O': fnp.reshape(self.M7[(slice(0, 4),) * L], (2, 2) * L + (rm, cm))}

    def _v(self, kind, d, form, r, c, sl):
        """cached view: node `kind` ('A' or 'B') at depth d, form 1 = Q (quadrant axis), 2 = S (slot axis), 0 = 3-D; leaf window r x c; sl = slice on axis 1 or None"""
        key = (kind, d, form, r, c, sl)
        v = self.vc.get(key)
        if v is None:
            base = (self.A if kind == 'A' else self.B)[d][form]
            if form == 0:
                v = base[:, :r, :c]
            else:
                v = base[:, slice(*sl), :, :r, :c]
            self.vc[key] = v
        return v

    def _x7(self, name, sel, r, c):
        """HZ_AB: cached view of an in-place 7-ary tree ('A' = X tree, 'B' = W tree, 'M' = result tree): per 7-axis `sel` holds None (all
        seven slots), an int (that slot) or a (start, stop, step) slot slice; leaf window r x c"""
        key = ('x7', name, sel, r, c)
        v = self.vc.get(key)
        if v is None:
            base = self.A7 if name == 'A' else self.B7 if name == 'B' else self.M7
            idx = tuple((slice(None) if t is None else t if type(t) is int else slice(*t) for t in sel))
            v = base[idx + (slice(None, r, None), slice(None, c, None))]
            self.vc[key] = v
        return v

    def _root_ab(self, kind, r, c):
        """HZ_AB: root content of the X tree ('A': slot = 2 j + 1 - i) or of the W / result trees ('B', 'O': slot = 2 i + j) in the axis
        order of the natural matrix (i_1..i_L, i', j_1..j_L, j'), leaf window r x c"""
        key = (kind, 'root7', r, c)
        v = self.vc.get(key)
        if v is None:
            L = self.L
            f, a = (slice(None, None, -1), slice(None))
            ev, od = (tuple(range(0, 2 * L, 2)), tuple(range(1, 2 * L, 2)))
            if kind == 'A':
                idx, perm = ((a, f) * L, od + (2 * L,) + ev + (2 * L + 1,))
            else:
                idx, perm = ((a, a) * L, ev + (2 * L,) + od + (2 * L + 1,))
            v = fnp.transpose(self.D7[kind][idx + (slice(None, r, None), slice(None, c, None))], perm)
            self.vc[key] = v
        return v

    def _plan_ab(self, r, c, w):
        """HZ_AB op list of one product (kinds as plan()): classic Strassen's seven products in an alternative basis, in place (keeps
        the accuracy of today's classic Strassen; a Winograd-form basis would be ~4x less accurate here). X entries per digit (slots 0..3)
        = (A21 + A22, A11, A22, A11 + A12) after one basis-change op per digit; per depth s = A11 + A22 -> slot 4 and (P2 - s, P5 - s) ->
        slots 5, 6 (3 adds, 2 ops); the leaf matmul on the depth-L stacks gives M2 M3 M4 M5 M1 M6 M7 at slots 0..6; assembly per depth
        z = M1 - M2 - M5 -> 4, (C12, C21) = (M3 + M5, M4 + M2) -> 1, 2, (C11 - C21, C22 - C12) = z + (M7, M6) -> 0, 3 (6 adds, 4 ops); exit
        per digit (C11, C22) = (C11 - C21, C22 - C12) + (C21, C12) (one op). Per level: 3 + 6 adds and 2 + 2 basis-change adds instead of
        5 adds + 2 copies + 8 adds."""
        key = ('ab', r, c, w)
        p = self.plans.get(key)
        if p is None:
            L = self.L
            V = self._x7
            cs = (0, 4, 1)
            p = []
            for l in range(L):
                t = lambda s, l=l: V('A', (cs,) * l + (s,) + (cs,) * (L - 1 - l), r, c)
                p.append((0, t((3, None, -3)), t((1, 3, 1)), t((3, None, -3))))
            for d in range(L):
                x = lambda s, d=d: V('A', (None,) * d + (s,) + (cs,) * (L - 1 - d), r, c)
                p.append((0, x(1), x(2), x(4)))
                p.append((1, x((0, 4, 3)), x((4, 5, 1)), x((5, 7, 1))))
            mv = self.vc.get(('M', r, w))
            if mv is None:
                mv = self.M[:, :r, :w]
                self.vc[('M', r, w)] = mv
            p.append((3, self._v('A', L, 0, r, c, None), self._v('B', L, 0, c, w, None), mv))
            for d in range(L - 1, -1, -1):
                o = lambda s, d=d: V('M', (None,) * d + (s,) + (cs,) * (L - 1 - d), r, w)
                p.append((1, o(4), o(0), o(4)))
                p.append((1, o(4), o(3), o(4)))
                p.append((0, o((1, 3, 1)), o((3, None, -3)), o((1, 3, 1))))
                p.append((0, o((4, 5, 1)), o((6, 4, -1)), o((0, 4, 3))))
            for l in range(L):
                t = lambda s, l=l: V('M', (cs,) * l + (s,) + (cs,) * (L - 1 - l), r, w)
                p.append((0, t((0, 4, 3)), t((2, 0, -1)), t((0, 4, 3))))
            p = tuple(p)
            self.plans[key] = p
        return p

    def _bplan_ab(self, c, w):
        """HZ_AB op list building the W tree in place: W entries per digit (slots 0..3) = (B11, B12 - B22, B21 - B11, B22) after one
        basis-change op per digit; per depth u = B11 + B22 -> slot 4, (Q3 + u, Q4 + u) -> slots 5, 6 (3 adds, 2 ops; was 5 adds + 2 copies)"""
        key = ('ab', c, w)
        p = self.bplans.get(key)
        if p is None:
            L = self.L
            V = self._x7
            cs = (0, 4, 1)
            p = []
            for l in range(L):
                t = lambda s, l=l: V('B', (cs,) * l + (s,) + (cs,) * (L - 1 - l), c, w)
                p.append((1, t((1, 3, 1)), t((3, None, -3)), t((1, 3, 1))))
            for d in range(L):
                y = lambda s, d=d: V('B', (None,) * d + (s,) + (cs,) * (L - 1 - d), c, w)
                p.append((0, y(0), y(3), y(4)))
                p.append((0, y((1, 3, 1)), y((4, 5, 1)), y((5, 7, 1))))
            p = tuple(p)
            self.bplans[key] = p
        return p

    def _root(self, kind, r, c):
        """root node (4^L, r, c) in the axis order of the natural matrix: (i_1..i_L, i', j_1..j_L, j')"""
        if HZ_AB:
            return self._root_ab(kind, r, c)
        key = (kind, 'root', r, c)
        v = self.vc.get(key)
        if v is None:
            L = self.L
            dg = (self.A if kind == 'A' else self.B)[0][3]
            perm = tuple(range(0, 2 * L, 2)) + (2 * L,) + tuple(range(1, 2 * L, 2)) + (2 * L + 1,)
            v = fnp.transpose(dg[(Ellipsis, slice(None, r, None), slice(None, c, None))], perm)
            self.vc[key] = v
        return v

    def digits(self, r, c):
        """shape of a natural (r 2^L x c 2^L) matrix split into its row and column digits"""
        return (2,) * self.L + (r,) + (2,) * self.L + (c,)

    def plan(self, r, c, w):
        """op list of one product with leaf sizes (r x c) @ (c x w): kinds 0 add, 1 subtract, 2 copyto(dst, src), 3 matmul"""
        if HZ_AB:
            return self._plan_ab(r, c, w)
        key = (r, c, w)
        p = self.plans.get(key)
        if p is None:
            L = self.L
            V = self._v
            p = []
            for d in range(L):
                q = lambda sl, d=d: V('A', d, 1, r, c, sl)
                s = lambda sl, d=d: V('A', d + 1, 2, r, c, sl)
                p.append((0, q((0, None, 2)), q((3, 4, 1)), s((0, 2, 1))))
                p.append((2, s((2, 4, 1)), q((0, None, 3)), None))
                p.append((0, q((0, 1, 1)), q((1, 2, 1)), s((4, 5, 1))))
                p.append((1, q((2, 0, -1)), q((0, None, 3)), s((5, 7, 1))))
            mv = self.vc.get(('M', r, w))
            if mv is None:
                mv = self.M[:, :r, :w]
                self.vc[('M', r, w)] = mv
            p.append((3, V('A', L, 0, r, c, None), V('B', L, 0, c, w, None), mv))
            for d in range(L - 1, -1, -1):
                if d == L - 1:
                    s = lambda sl: self._ms(r, w, sl)
                else:
                    s = lambda sl, d=d: V('A', d + 1, 2, r, w, sl)
                q = lambda sl, d=d: V('A', d, 1, r, w, sl)
                p.append((0, s((2, 0, -1)), s((4, 2, -1)), q((1, 3, 1))))
                p.append((0, s((3, 1, -1)), s((0, 1, 1)), q((0, None, 3))))
                p.append((1, q((0, None, 3)), s((4, None, -3)), q((0, None, 3))))
                p.append((0, q((0, None, 3)), s((6, 4, -1)), q((0, None, 3))))
            p = tuple(p)
            self.plans[key] = p
        return p

    def _ms(self, r, w, sl):
        key = ('MSv', r, w, sl)
        v = self.vc.get(key)
        if v is None:
            v = self.MS[:, slice(*sl), :, :r, :w]
            self.vc[key] = v
        return v

    def bplan(self, c, w):
        """op list building the right-operand leaf stack (7^L, c, w) from the root node"""
        if HZ_AB:
            return self._bplan_ab(c, w)
        key = (c, w)
        p = self.bplans.get(key)
        if p is None:
            V = self._v
            p = []
            for d in range(self.L):
                q = lambda sl, d=d: V('B', d, 1, c, w, sl)
                s = lambda sl, d=d: V('B', d + 1, 2, c, w, sl)
                p.append((0, q((0, 1, 1)), q((3, 4, 1)), s((0, 1, 1))))
                p.append((2, s((1, None, 3)), q((0, None, 3)), None))
                p.append((1, q((1, 3, 1)), q((3, None, -3)), s((2, 4, 1))))
                p.append((0, q((0, None, 2)), q((1, None, 2)), s((5, 7, 1))))
            p = tuple(p)
            self.bplans[key] = p
        return p

    @staticmethod
    def run(p):
        add, sub, cp, mm = (fnp.add, fnp.subtract, fnp.copyto, fnp.matmul)
        if RRES_ON and RRES_HOLD:
            # rres_patch RRES_HOLD (HZE product run): one batched free per run
            hold_ = []
            ap_ = hold_.append
            add = lambda a, b, out, f=fnp.add: ap_(f(a, b, out=out))
            sub = lambda a, b, out, f=fnp.subtract: ap_(f(a, b, out=out))
            mm = lambda a, b, out, f=fnp.matmul: ap_(f(a, b, out=out))
        for kind, a, b, o in p:
            if kind == 0:
                add(a, b, out=o)
            elif kind == 1:
                sub(a, b, out=o)
            elif kind == 2:
                cp(a, b)
            else:
                mm(a, b, out=o)

    def set_right(self, W, c, w):
        """load the right operand W (c 2^L x w 2^L) and build its leaf stack"""
        fnp.copyto(self._root('B', c, w), fnp.reshape(W, self.digits(c, w)))
        self.run(self.bplan(c, w))

    def mm(self, Xd, Yd, r, c, w):
        """Yd <- X @ W for the loaded right operand; Xd, Yd are the digit forms (reshape views) of the natural operands"""
        fnp.copyto(self._root('A', r, c), Xd)
        self.run(self.plan(r, c, w))
        fnp.copyto(Yd, self._root('O' if HZ_AB else 'A', r, w))


_POST_S26_KNOBS = ('J2_W431', 'J2_LOT', 'J2_RRW', 'J2_D21F', 'J2_KP', 'AU_REUSE', 'AU_T1PK_EXT', 'J3_TRI', 'AU_SYM4', 'AS_TAG_OFF', 'CDP_ON', 'EF_K31T21', 'FIN_EDGE_ON', 'FIN_KC_ON', 'FIN_GRAN_ON', 'THIN_ENG', 'OPD', 'OPD_SWAP', 'OPD_ZT', 'OPD_NLK', 'OPD_PK', 'OPD_CSE', 'TDP_ON', 'MID_EDGE_ON', 'MARG2_ON', 'MARG1_ON', 'LOTD_ON', 'RRES_ON', 'DN_YCOL', 'DN_ZKIN', 'DN_FROT', 'DN_G0', 'DN_THIN2D', 'DN_Z2D', 'DN_EIGH', 'DN_KP', 'DN_FSUB', 'ZF7_AB_MERGE', 'HZ_AB', 'ZSRC_CACHE', 'ROLE_MERGE', 'LX', 'LX_PAD', 'LX_MULTI', 'HZ_L6', 'NLK_PK', 'NLK_K22', 'HZW', 'QRB', 'QRV', 'AU_T1PK', 'AU_LIFTW', 'AU_THINW', 'AU_APKZ', 'AU_ROWP', 'AU_YXT', 'HZGATE', 'LXPGATE', 'HZG_LOG', 'HZG_EXT', 'HZG_L6')   # guard_patch: switched off for off-suite predicts

class Estimator(BaseEstimator):
    AGE_OLD = int(_os.environ.get('V21_AGE_OLD', '4'))
    R_OLD = int(_os.environ.get('V21_R_OLD', '448'))
    AGE_OLD2 = int(_os.environ.get('V24_AGE_OLD2', '8'))
    R_OLD2 = int(_os.environ.get('V24_R_OLD2', '224'))
    QPASS2 = int(_os.environ.get('V24_QPASS2', '2'))
    R_FB = int(_os.environ.get('TD_R_FB', '4'))
    R_RES = int(_os.environ.get('TD_R_RES', '16' if TD_OFF else '8'))

    def __init__(self) -> None:
        self._setup_rng = None
        self._wick_consts = _build_wick_consts()
        self._progs = {0: _term_prog(0), 1: _term_prog(1), 2: _term_prog(2)}
        self._nlk = {0: _nlk_prog(0), 1: _nlk_prog(1), 2: _nlk_prog(2)} if NLK_PK else None
        self._j2w = _j2_nlk_w431() if (J2_W431 and NLK_PK) else None
        self._gb2 = {}
        for _m, _p in self._progs.items():
            _b, _prev = ([], -1)
            for _t, _row in enumerate(_p['IND2']):
                _g = _row.index(1.0)
                assert _g >= _prev, 'term groups must be contiguous'
                if _g != _prev:
                    _b.append([_g, _t, _t + 1])
                    _prev = _g
                else:
                    _b[-1][2] = _t + 1
            self._gb2[_m] = [tuple(x) for x in _b]
        self._i11 = WICK_PAIRS.index((1, 1))
        self._i21 = WICK_PAIRS.index((2, 1))
        self._i12 = WICK_PAIRS.index((1, 2))
        self._i31 = WICK_PAIRS.index((3, 1))
        self._hz = None
        self._hz_st = {}
        self._hz_rid = {}
        self._hz_act = False

    def setup(self, ctx: SetupContext) -> None:
        import time as _time
        _t0 = _time.perf_counter()
        self._setup_rng = fnp.random.default_rng(ctx.seed)
        if WARM:
            try:
                f32 = fnp.float32
                z = fnp.zeros((64, 16), dtype=f32)
                fnp.linalg.qr(z + 1.0)
                fnp.linalg.solve(fnp.eye(16, dtype=f32) * 2.0, _pview(z, (slice(None, 16, None), slice(None, 4, None))) + 1.0)
                if DN_EIGH:
                    fnp.linalg.eigh(flops.as_symmetric(fnp.eye(16, dtype=f32), symmetry=(0, 1)))
                v = fnp.zeros(64, dtype=f32)
                flops.stats.norm.pdf(v)
                flops.stats.norm.cdf(v)
                e = flops.as_symmetric(fnp.eye(64, dtype=f32), symmetry=(0, 1))
                fnp.einsum('ij,ia,jb->ab', e, _pview(z, (slice(None, None, None), slice(None, 16, None))) + 1.0, _pview(z, (slice(None, None, None), slice(None, 16, None))) + 1.0)
                a = fnp.zeros((4, 21), dtype=f32) + 1.0
                b = fnp.zeros((21, 64), dtype=f32) + 1.0
                m = fnp.zeros((64, 64), dtype=f32) + 1.0
                t3 = fnp.zeros((3, 64, 64), dtype=f32) + 1.0
                g3 = fnp.zeros((3, 3), dtype=f32) + 1.0
                _ = a @ b
                _ = m @ m
                _ = m @ v
                fnp.einsum('tij,ti,tj,tg->gij', t3, _pview(t3, (slice(None, None, None), slice(None, None, None), 0)), _pview(t3, (slice(None, None, None), 0, slice(None, None, None))), g3)
                fnp.einsum('ti,ti,tg->gi', _pview(t3, (slice(None, None, None), slice(None, None, None), 0)), _pview(t3, (slice(None, None, None), slice(None, None, None), 1)), g3)
                fnp.einsum('iq,kqn->kin', _pview(m, (slice(None, None, None), slice(None, 16, None))), _pview(t3, (slice(None, None, None), slice(None, 16, None), slice(None, None, None))))
                fnp.einsum('ki,kc->ic', _pview(t3, (slice(None, None, None), slice(None, None, None), 0)), _pview(t3, (slice(None, None, None), slice(None, None, None), 1)))
                fnp.einsum('kij,kij->i', t3, t3)
                fnp.einsum('kij,kj->ki', t3, _pview(t3, (slice(None, None, None), 0, slice(None, None, None))))
                fnp.einsum('ki,ki->i', _pview(t3, (slice(None, None, None), slice(None, None, None), 0)), _pview(t3, (slice(None, None, None), slice(None, None, None), 1)))
                fnp.einsum('kij,kqj->iq', t3, _pview(t3, (slice(None, None, None), slice(None, 16, None), slice(None, None, None))))
                fnp.einsum('kiq,kjq->kij', t3, t3)
                fnp.einsum('kij,kjq->kiq', t3, t3)
                fnp.einsum('kiq,kcq->ic', t3, t3)
                fnp.einsum('pq,kqn->kpn', _pview(m, (slice(None, 16, None), slice(None, 16, None))), _pview(t3, (slice(None, None, None), slice(None, 16, None), slice(None, None, None))))
                fnp.einsum('cij,cjk->cik', _pview(t3, slice(None, 1, None)), t3)
                fnp.sum(t3, axis=(0, 2))
                fnp.sum(t3, axis=0)
                fnp.diag(m)
                fnp.diagflat(v)
                fnp.fill_diagonal(m, 0.0)
                fnp.matmul(t3, fnp.swapaxes(t3, -1, -2))
                fnp.concatenate([m, m], axis=1)
                fnp.stack([v, v], axis=1)
                fnp.maximum(v, 1e-10)
                fnp.sqrt(v + 1.0)
                fnp.power(v + 1.0, 1.0)
                fnp.clip(v, 0.5, 2.0)
                fnp.mean(v)
                fnp.copy(m)
                fnp.abs(v)
                g2 = fnp.zeros((16, 32), dtype=f32) + 1.0
                fnp.einsum('aj,bj->ab', g2, g2)
                fnp.power(_pview(v, (slice(None, None, None), None)) + 1.0, _pview(fnp.asarray([0.0, 1.0, 2.0], dtype=f32), (None, slice(None, None, None))))
                fnp.sum(_pview(t3, slice(None, 2, None)), axis=0, out=m)
                fnp.concatenate([g2, g2], axis=1)
                if SDT_ON:
                    fnp.copy(fnp.flip(v))
                    fnp.sum(fnp.less(v, 0.5))
            except Exception:
                pass
        if M2_PREALLOC and (not MERGE_OFF) and (not FMM_OFF) and (int(getattr(ctx, 'width', 0)) == 1024) and (int(getattr(ctx, 'depth', 0)) == 16):
            _setup_mark('warm', _t0)
            try:
                self._prealloc(16)
            except Exception:
                self._pool = None
                self._smm = None
                self._ncall = 0
            _setup_mark('prealloc', _t0)
        self._hz = None
        if HZ_ON and (int(getattr(ctx, 'width', 0)) == 1024) and (int(getattr(ctx, 'depth', 0)) == 16):
            try:
                self._hz_setup(1024, 16)
            except Exception:
                if HZ_STRICT:
                    raise
                self._hz = None
        _setup_mark('total', _t0)
        if PAD_SETUP_LOG:
            try:
                with open(PAD_SETUP_LOG, 'a') as _f:
                    _f.write(repr(_SETUP_T) + '\n')
            except Exception:
                pass

    def _prealloc(self, L):
        """merge2 stage C: create every pooled buffer of the steady-state op stream (spec embedded; shapes are data independent) in setup();
        complete named-buffer and Z-role allocation lifts the first-call level cap; remaining plan views are created lazily and billed in predict()."""
        import time as _time
        t0 = _time.perf_counter()
        f32 = fnp.float32
        pool = _Pool(f32)
        pool.alias_work = L == 16
        pool.factor_profile = (L, int(self.AGE_OLD), int(self.AGE_OLD2), int(self.R_OLD), int(self.R_OLD2))
        smm = _FMMd(f32, 2 * (L - 1), True)
        self._pool = pool
        self._smm = smm
        done = smm.z is not None and smm.bmax == int(_POOL_SPEC['bmax'])
        for kind, name, shape in _POOL_SPEC['pool'] if done else []:
            if _time.perf_counter() - t0 > M2_PREALLOC_T:
                done = False
                break
            if kind == 'pair':
                pool.get_pair(_tup(name), _tup(shape))
            else:
                pool.get(_tup(name), _tup(shape))
        for role, n in (_zf7_roles() if ZF7 else _POOL_SPEC['roles']) if done else []:
            if _time.perf_counter() - t0 > M2_PREALLOC_T:
                done = False
                break
            smm.z._reserve_setup(_tup(role), int(n))
        _setup_mark('p_roles', t0)
        for role, n in sorted(_deep_roles(1024, PAD_LMAX).items(), key=str) if done and PAD_DEEP and PAD_RESERVE else []:
            if _time.perf_counter() - t0 > M2_PREALLOC_T:
                done = False
                break
            smm.z._reserve_setup(role, int(n))
        _setup_mark('p_deep_roles', t0)
        buffers_ready = done
        if done:
            smm.z.public_setup_slices = {}
        for key in _POOL_SPEC['plans'] + (_deep_plans(1024) if PAD_DEEP and PAD_RESERVE and PAD_PLANS else []) + (_lx_plans(1024) if LX else []) + (_lx_plans8(1024) if LX and LX_MULTI and LX_LMAX >= 8 else []) if done else []:
            if _time.perf_counter() - t0 > PUBLIC_Z_SETUP_MAX_T:
                done = False
                break
            smm.z._plan(*[int(v) if not isinstance(v, str) else v for v in key])
        if smm.z is not None:
            smm.z.public_setup_slices = None
            if LXPGATE:
                # LXPGATE: the static setup plan list (independent of the setup clock), the only plans LX_PAD may pad onto
                smm.z.lx_ready = frozenset((tuple((int(v) if not isinstance(v, str) else v for v in k_)) + ((0,) if len(k_) == 7 else ()) for k_ in list(_POOL_SPEC['plans']) + _lx_plans(1024) + (_lx_plans8(1024) if '_lx_plans8' in globals() else [])))
        _setup_mark('p_plans', t0)
        self._ncall = 1 if done or (PUBLIC_BUFFER_READY and buffers_ready) else 0
        if self._ncall == 1 and PUBLIC_Z_VIEWS and PUBLIC_Z_PREWARM:
            remaining = max(0.0, min(PUBLIC_Z_PREWARM_T, PUBLIC_Z_SETUP_MAX_T - (_time.perf_counter() - t0)))
            if remaining > 0.0:
                smm.z.rres_setup = True   # rres_patch: lean batching in the wall-bound prewarm
                try:
                    smm.z.prewarm_public_nodes(remaining, PUBLIC_Z_PREWARM_LIMIT)
                finally:
                    smm.z.rres_setup = False
        _setup_mark('p_prewarm', t0)
        if OPD_ZT and ZF7 and smm.z is not None:
            # opd_patch OPD_ZT: inverse-permuted entry / exit views of the setup plans, AFTER the prewarm (its clock is untouched),
            # own budget OPD_ZT_SETUP_T; plans not reached build theirs on first use
            t_o = _time.perf_counter()
            for pl_ in list(smm.z.plans.values()):
                if _time.perf_counter() - t_o > OPD_ZT_SETUP_T:
                    break
                for i_ in range(3):
                    smm.z._zinv(pl_, i_)
        if RRES_ON and RRES_PREWARM_T > 0 and ZF7 and smm.z is not None and self._ncall == 1 and PUBLIC_Z_VIEWS and PUBLIC_Z_PREWARM:
            # rres_patch RRES_PREWARM_T: more plan views (getitem / swapaxes / unstack only: free, no billed op moves out of predict),
            # same plan order, own clock; plans already prepared return their cached views at once
            smm.z.rres_setup = True
            try:
                smm.z.prewarm_public_nodes(RRES_PREWARM_T, RRES_PREWARM_LIMIT)
            finally:
                smm.z.rres_setup = False
        if PAD_SETUP_LOG:
            _SETUP_T['plans'] = len(smm.z.plans) if smm.z is not None else -1
            _SETUP_T['done'] = bool(done)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        was = _gc.isenabled()
        _gc.disable()
        # off-suite guard (smoke-test shapes): every knob added after s26 is switched off for this predict, so any shape
        # other than the scored (1024, 16) runs the s26 code path; the suite shape is untouched
        _sv = None
        if not (int(getattr(mlp, 'width', 0)) == 1024 and int(getattr(mlp, 'depth', 0)) == 16):
            _g = globals()
            _sv = {_k: _g[_k] for _k in _POST_S26_KNOBS if _k in _g}
            for _k in _sv:
                _g[_k] = False
        try:
            return self._predict_core(mlp, budget)
        finally:
            if _sv is not None:
                globals().update(_sv)
            if was:
                _gc.enable()
            if PAD_DEBUG:
                _pad_report()

    def _predict_core(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        self._rb_t0 = _rb_time.perf_counter()   # agents_robust HZ_TMAX / HZ_TMAX_L: predict start
        self._rb_hz_skip = None
        if ARR_GUARD > 0 and (ARR_GUARD_TEST or _rb_live_handles() > ARR_GUARD):
            _rb_arr_clear(self)   # agents_robust ARR_GUARD: stay clear of the server's array-store cap
        if HZG_LOG:
            self._hzg_fb = []
            self._hzg_t = self._hzg_snap()
        self._young_column_shapes = []
        self._young_readout_shapes = []
        SDT_LOG.clear()
        YR_LOG.clear()
        OLD_ROW_LOG.clear()
        t2_pending = 0
        t1_pending = None
        SDT_PROBES.clear()
        n = mlp.width
        f32 = fnp.float32
        st = _statics(n)
        metric2 = 2.0 ** 2
        L = len(mlp.weights)
        riders = (n == 1024 or _os.environ.get('TA_RIDERS_ANY', '0') == '1' or n == AS_TEST_N) and L == len(CORR_BETA)
        ta_wick = TA_WICK and riders
        ta_nl = TA_NL and riders
        ta_gram = TA_GRAM and riders
        ta_join = TA_JOIN and riders
        m2_qc = M2_QC and riders
        bq_srcp = BQ_SRCP and riders
        _TAG['wino'] = TA_WINO and riders
        _TAG['mmfb'] = TA_MMFB and riders
        _TAG['old_min'] = TA_OLD_MIN if riders else STRASSEN_MIN
        _TAG['yh_min'] = TA_YH_MIN if riders else STRASSEN_MIN
        _TAG['join'] = ta_join
        as_k = {}
        if riders and (not NO_REGEN) and (not AS_OFF):
            for tok in AS_K.split(','):
                if tok.strip():
                    a_, b_ = tok.split(':')
                    as_k[int(a_)] = int(b_)
            for l_, k_ in as_k.items():
                assert 1 <= l_ < L and k_ % 32 == 0 and (0 <= k_ <= 192), (l_, k_)
        as_on = bool(any((v > 0 for v in as_k.values()))) or (riders and (not NO_REGEN) and (not AS_OFF) and (AS_FORCE_PERM or AS_ADAPT))
        self._mn_y = AS_YMIN if as_on else _TAG['yh_min']
        self._nfull = n
        self._as_on = as_on
        gI_prev = None
        dead_ids = None
        dead_m = None
        dead_m_cur = None
        nr = n
        ni = n
        idx_cur = None
        inv_cur = None
        kl = 0
        C1l, C2l, SELCl = self._wick_consts
        C1 = fnp.asarray(C1l, dtype=f32)
        C2 = fnp.asarray(C2l, dtype=f32)
        SELC = fnp.asarray(SELCl, dtype=f32)
        FP = {}
        for mode, p in self._progs.items():
            FP[mode] = {k: fnp.asarray(v, dtype=f32) for k, v in p.items() if k in ('SL2', 'SR2', 'IND2', 'SL1', 'IND1')}
            if NLK_PK:
                FP[mode]['NLK_SL'] = fnp.asarray(self._nlk[mode]['SL'], dtype=f32)
                FP[mode]['NLK_SR'] = fnp.asarray(self._nlk[mode]['SR'], dtype=f32)
                if J2_W431 and mode == 1 and getattr(self, '_j2w', None) is not None:
                    FP[mode]['J2W_SL'] = fnp.asarray(self._j2w['SL'], dtype=f32)
                    FP[mode]['J2W_SR'] = fnp.asarray(self._j2w['SR'], dtype=f32)
        C12 = fnp.concatenate([C1, C2], axis=1) if ta_wick else None
        PW6 = fnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=f32) if ta_wick else None
        EX11 = fnp.asarray([-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=f32) if ta_wick else None
        ones_full = fnp.ones(n, dtype=f32)
        zeros_full = fnp.zeros(n, dtype=f32)
        ones_n = ones_full
        zeros_n = zeros_full
        ones_col = _pview(ones_n, (slice(None, None, None), None))
        ones_row = ones_col.T
        ones2_full = ones_col * ones_row
        ones2 = ones2_full
        ones_row_f = ones_row
        beta_rows = [fnp.asarray(b, dtype=f32) for b in CORR_BETA]
        t2max = max((len(p['AB2']) for p in self._progs.values()))
        pool = getattr(self, '_pool', None)
        if pool is None or pool.dtype != f32:
            pool = _Pool(f32)
            self._pool = pool
        pool.alias_work = L == 16

        def NN(name):
            return _win(pool.get(name, (n, n)), nr, nr)

        def NW(name, rr_, cc_):
            return _win(pool.get(name, (n, n)), rr_, cc_)

        def hubv(v):
            """per-source hub vector padded with zeros to the full hub width n (identity when the layer is full width)"""
            pad = n - v.shape[0]
            return v if pad == 0 else fnp.concatenate([v, _pview(zeros_full, slice(None, pad, None))])
        abbuf_full = pool.get('abbuf', (t2max, n, n))
        abbuf = abbuf_full
        T1 = NN('t1')
        mu = fnp.zeros(n, dtype=f32)
        C = fnp.eye(n, dtype=f32) if AS_TAG_OFF else flops.as_symmetric(fnp.eye(n, dtype=f32), symmetry=(0, 1))
        A_st = P_st = Z_st = L_st = None
        newborn = None
        w2b_list = []
        s_list = []
        e_list = []
        c1_list = []
        c2_list = []
        y_list = []
        g_prev = None
        K22 = K4v = k22row = None
        var_prev = None
        lam_prev = 0.0
        regen = riders and (not NO_REGEN)
        lot = PL_ON and regen
        lot_slab = None
        lot_Q = lot_G = None
        bufs_nr = -1
        legs = None
        smm = getattr(self, '_smm', None)
        if smm is None or smm.bmax != 2 * (L - 1) or smm.dtype != f32 or (smm.zmode != (riders and (not MERGE_OFF) and (not FMM_OFF))):
            smm = _FMMd(f32, 2 * (L - 1), riders and (not MERGE_OFF))
            self._smm = smm
        ncall = getattr(self, '_ncall', 0)
        self._ncall = ncall + 1
        smm.first = ncall == 0
        smm.steady = 0
        smm.as_mode = as_on
        smm.deep_mode(ncall > 0 and riders)
        s_lev = STRASSEN_LEVELS if ncall > 0 else min(STRASSEN_LEVELS, STRASSEN_FIRST)
        s_yh = (TA_YH_LEV if riders else 5) if ncall > 0 else min(TA_YH_LEV if riders else 5, STRASSEN_FIRST)
        s_sb = STRASSEN_SB
        self._s_hub = s_yh
        r = min(int(self.R_RES if riders else 16), n)
        rfb = 0 if NO_FB else min(int(self.R_FB), n)
        kz = TD_KZ if rfb > 0 and n == 1024 and (L == 16) else 0
        rsk = max(r, rfb, PL_K if lot else 0)
        r_old = int(self.R_OLD)
        confine = not NO_CONFINE and r_old < n
        age_old = int(self.AGE_OLD)
        ka = 0
        Qc = None
        fa_side = f2_side = qc_side = z_side = 0
        FAP = FAP2 = None
        fap5 = None
        fap4 = fap24 = None
        fa_off = 0
        legs4 = None
        FAo = FPo = None
        Sg = None
        r2 = int(self.R_OLD2)
        age_old2 = int(self.AGE_OLD2)
        pool.factor_profile = (L, age_old, age_old2, r_old, r2)
        nest = confine and age_old2 > 0 and (r2 < r_old)
        kb = 0
        U = None
        FA2 = FP2 = None
        S2 = None
        QU = None
        dA_list = []
        dP_list = []
        R1T_st = R2T_st = None
        Zf_st = None
        K4_sigma = None
        K4_vec = None
        rows = []
        hub_widths = []
        self._hub_birth_widths = hub_widths
        hz_W = []
        hz_c1 = []
        hz_n3 = []
        self._hz_n3 = hz_n3
        hz_c2 = []
        hz_nu = []
        hz_Phi = []
        hz_map = []
        hz_tail = None
        self._hz_st = {}
        self._hz_rid = {}
        self._hz_act = HZ_ON and (self._hz is not None) and n == 1024 and L == 16

        def _out(v):
            """alive-ordered vector -> natural-order n-vector (dropped neurons predicted 0: their post-ReLU mean is < ~1e-4)"""
            if idx_cur is None:
                return v
            return fnp.take(fnp.concatenate([_pview(zeros_full, slice(None, n - nr, None)) if dead_m_cur is None else dead_m_cur, v]), inv_cur)
        w1_prev = None
        hm = None
        nd_cur = 0
        self._sdt_ord = False
        self._sdt_nc = -1
        self._sdt_ow1 = n
        self._hot_p = 0
        self._au_t1pk = None
        self._au_nc = -1
        d21_gate_on = riders and lot and (D21_CUT > 0)
        merge_cp = TA_CPRE and riders and (STRASSEN_HUB > 0) and (not d21_gate_on or JOINT_CP_ON)
        bufs_box = [None, -1]

        def get_bufs(nr_):
            """pooled (k, nr, n) windows of the full-size stacks (rows = alive neurons of this layer, hub width n); rebuilt when nr changes"""
            b = bufs_box[0]
            if b is None or bufs_box[1] != nr_:
                b = {nm: _win(pool.get(nm, (L - 1, n, n)), nr_, n) for nm in ('ap', 'pp', 't', 'mp', 'xt', 'yt', 'u')}
                b['hub'] = _win(pool.get('hub', (n, n)), nr_, nr_)
                if merge_cp:
                    b['hubg'] = _win(pool.get('hubg', (2, 1, n, n)), nr_, nr_)
                b['ppl'] = _win(pool.get('ppl', (L - 1, n, r + 2)), nr_, r + 2)
                b['d21'] = _win(pool.get('d21', (n, n)), nr_, nr_)
                b['t1'] = _win(pool.get('t1', (n, n)), nr_, nr_)
                if rfb > 0:
                    b['gyr'] = _win(pool.get('gyr', (L - 1, n, rfb)), nr_, rfb)
                    b['gxr'] = _win(pool.get('gxr', (L - 1, n, rfb)), nr_, rfb)
                    b['d21fb'] = _win(pool.get('d21fb', (n, n)), nr_, nr_)
                b['cp'] = False
                bufs_box[0] = b
                bufs_box[1] = nr_
            assert legs is not None and legs4 is not None
            b['lap'] = _win(legs['AP1'], nr_, n)
            b['lap4'] = _win(legs4['AP1'], nr_, n)
            b['lap4p'] = legs4['AP1']
            b['apb4p'] = legs4['AP0']
            b['hubp'] = _pview(pool.get('hub', (n, n)), None)
            b['hubgp'] = pool.get('hubg', (2, 1, n, n)) if merge_cp else None
            return b
        for li, w in enumerate(mlp.weights):
            if HZ_TMAX_L > 0 and self._hz_act and _rb_time.perf_counter() - self._rb_t0 > HZ_TMAX_L:
                # agents_robust HZ_TMAX_L: late predict -> no more HZ work this predict (the rows of the chain are untouched)
                self._hz_act = False
                self._rb_hz_skip = ('L', li)
            last = li == L - 1
            cp_now = False
            trim = last and (not FULL_LAST)
            lot_Q = lot_G = None
            lot_g4m = None
            self._resp_data = None
            g4m_k22 = None
            skip_src = trim and NO_SRC_LAST
            w32 = w if fnp.dtype(w.dtype) == fnp.dtype(f32) else w.astype(f32)
            ni = nr
            nd_in = nd_cur
            nd_cur = 0
            kin = ni - nd_in
            tdp_k = -1
            tdp_y = False
            if TDP_ON:
                # tdp_patch: band rows [tdp_k:kin] of this layer's w1-scaled inputs (window set by the previous layer; -1 = no band)
                tdp_k = getattr(self, '_tdp_kn', -1)
                self._tdp_kn = -1
                if not (64 <= tdp_k < kin and TDP_LMIN <= li <= TDP_LMAX):
                    tdp_k = -1
            sdt_ord = False
            kwin_fix = False
            hc_young = n
            self._sdt_ow1 = n
            self._hot_p = 0
            self._sdt_ord = False
            self._sdt_nc = -1
            idx_in = idx_cur
            idx_cur = None
            inv_cur = None
            kl = 0
            dead_w = None
            dead_in = dead_ids
            dead_ids = None
            m_in = dead_m
            dead_m = None
            dead_m_cur = None
            fin_dead = None
            mid_dead = None
            sqd_in = None
            mu_corr = None
            dwt = None
            if as_on and li > 0 and (dead_in is not None) and (AS_MCORR and m_in is not None or (regen and AS_GCONST and (gI_prev is not None))):
                dwt = fnp.take(w32, dead_in, axis=0)
                if AS_MCORR and m_in is not None:
                    mu_corr = fnp.matmul(m_in, dwt)
            w32in = w32
            sk_cur = None
            if as_on and li > 0:
                kl = as_k.get(li, 0)
                if idx_in is not None:
                    w32in = fnp.take(w32, idx_in, axis=0)
                mu_all = None
                ord_ad = None
                # SDT: this layer runs in the SDT live order even when nothing is dropped (kl == 0: a pure permutation)
                sdt_perm = SDT_ORDER and (not AS_FORCE_PERM) and (li >= SDT_L0) and (not last)
                if AS_ADAPT and (not AS_FORCE_PERM) and (li >= AS_L0) and (AS_LAST or li < L - 1):
                    kc = min(192, 32 * (1 + li // 3))
                    mu_all = fnp.matmul(w32in.T, mu)
                    if mu_corr is not None:
                        mu_all = mu_all + mu_corr
                    ord0 = fnp.argsort(mu_all)
                    final_tails = last and AS_FINAL_TAILS and AS_MCORR
                    if final_tails:
                        kside = min(AS_FINAL_KC if (FIN_KC_ON and self._nfull == 1024) else kc, n // 2)   # fin_patch AS_FINAL_KC
                        cand = fnp.concatenate([_pview(ord0, slice(None, kside, None)), _pview(ord0, slice(n - kside, None, None))])
                        kc = 2 * kside
                    else:
                        cand = _pview(ord0, slice(None, kc, None))
                    wcT = fnp.take(w32in, cand, axis=1).T
                    varc = fnp.sum(fnp.multiply(_c7mm(smm, 1, wcT, C, _win(pool.get('c7_wc', (n, n)), wcT.shape[0], C.shape[1])), wcT), axis=1)
                    au_sv_ = fnp.sqrt(fnp.maximum(varc, 1e-10)) if AU_REUSE else None
                    alc = fnp.take(mu_all, cand) / (au_sv_ if au_sv_ is not None else fnp.sqrt(fnp.maximum(varc, 1e-10)))
                    nd = int(float(fnp.sum(fnp.abs(alc) > AS_FINAL_C if final_tails else alc < -_AS_CDEAD_L(li))))
                    drop_cap = min(kc, AS_FINAL_MAX) if final_tails else kc
                    gran_ = FIN_GRAN if (FIN_GRAN_ON and final_tails and self._nfull == 1024) else AS_GRAN   # fin_patch FIN_GRAN
                    kl = max(0, min(drop_cap, nd - AS_MARGIN) // gran_ * gran_)
                    if FIN_KC_ON and final_tails and self._nfull == 1024 and getattr(self, '_hz', None) is not None and li < len(self._hz['KL']):
                        # fin_patch safety: _hz_select switches the whole HZ off when the final nr < KL[li]; keep >= KL + 64 live rows
                        kl = max(0, min(kl, (n - int(self._hz['KL'][li]) - 64) // gran_ * gran_))
                    if kl > 0 or sdt_perm:
                        oa = fnp.argsort(-fnp.abs(alc) if final_tails else alc)
                        remainder = _pview(ord0, slice(kside, n - kside, None)) if final_tails else _pview(ord0, slice(kc, None, None))
                        ord_ad = fnp.concatenate([fnp.take(cand, oa), remainder])
                        if AS_MCORR and kl > 0:
                            sg_ = fnp.take(au_sv_ if au_sv_ is not None else fnp.sqrt(fnp.maximum(varc, 1e-10)), oa)
                            a_c = _pview(fnp.take(alc, oa), slice(None, kl, None))
                            dead_m_cur = _pview(sg_, slice(None, kl, None)) * (flops.stats.norm.pdf(a_c).astype(f32) + a_c * flops.stats.norm.cdf(a_c).astype(f32))
                            if FIN_EDGE_ON and final_tails and self._nfull == 1024:
                                fin_dead = (a_c, sg_, kl)   # fin_patch FIN_EDGE: corrected at the final assembly below
                            if MID_EDGE_ON and (not final_tails) and self._nfull == 1024:
                                mid_dead = (a_c, sg_, kl)   # mid_patch MID_EDGE: corrected after this layer's mean program
                if kl > 0 or AS_FORCE_PERM or sdt_perm:
                    if mu_all is None:
                        mu_all = fnp.matmul(w32in.T, mu)
                        if mu_corr is not None:
                            mu_all = mu_all + mu_corr
                    if AS_FORCE_PERM and kl == 0:
                        key_ = fnp.random.default_rng(7919 + li).standard_normal(n)
                    else:
                        key_ = mu_all
                    ordr = fnp.argsort(key_) if ord_ad is None else ord_ad
                    if SDT_ORDER and (ord_ad is not None or sdt_perm) and (not last):
                        # SDT: the live part reversed = descending mean, then the live exact-alpha candidates by descending alpha
                        # (nearly-dead neurons are the last rows); the dropped prefix ordr[:kl] is unchanged
                        ordr = fnp.concatenate([_pview(ordr, slice(None, kl, None)), fnp.flip(_pview(ordr, slice(kl, None, None)))]) if kl > 0 else fnp.copy(fnp.flip(ordr))
                        sdt_ord = True
                        self._sdt_ord = SDT_MASK
                    idx_cur = _pview(ordr, slice(kl, None, None))
                    dead_ids = _pview(ordr, slice(None, kl, None)) if kl > 0 else None
                    dead_m = dead_m_cur
                    inv_cur = fnp.argsort(ordr)
                    if kl > 0 and regen and (BETA != 0.0) and AS_FULLMEAN:
                        wdT = fnp.take(w32in, _pview(ordr, slice(None, kl, None)), axis=1).T
                        dead_w = (wdT, fnp.sum(fnp.multiply(fnp.matmul(wdT, C), wdT), axis=1))
                    w32g = fnp.take(w32in, idx_cur, axis=1)
                    mu = fnp.take(mu_all, idx_cur)
                    nr = n - kl
                    if AS_DEBUG:
                        AS_LOG.append(dict(li=li, kl=kl, ni=ni, nr=nr, mu_dead=_pview(fnp.asarray(mu_all), _pview(ordr, slice(None, kl, None))) if kl else None, ord=ordr, mu_all=mu_all))
                else:
                    w32g = w32in
                    nr = n
                    mu = w32g.T @ mu
                    if mu_corr is not None:
                        mu = mu + mu_corr
                if idx_cur is not None:
                    sk_cur = fnp.take(_pview(w32, (slice(None, None, None), slice(None, rsk, None))), idx_cur, axis=0)
            else:
                w32g = w32
                nr = n
                mu = w32g.T @ mu
            W = w32g.T
            hz_W.append(W)
            hz_map.append((inv_cur, nr))
            if self._hz_act and last and as_on and (dead_ids is not None) and (mu_all is not None):
                hz_tail = (fnp.take(w32in, dead_ids, axis=1), fnp.take(mu_all, dead_ids))
            if as_on and regen and AS_GCONST and (dead_in is not None) and (gI_prev is not None):
                sqd_in = fnp.sum(fnp.multiply(dwt, dwt), axis=0)
            if nr != n or ni != n:
                ones_n = _pview(ones_full, slice(None, nr, None))
                zeros_n = _pview(zeros_full, slice(None, nr, None))
                ones2 = _pview(ones2_full, (slice(None, nr, None), slice(None, nr, None)))
                ones_row = _pview(ones_row_f, (slice(None, None, None), slice(None, nr, None)))
                abbuf = _pview(abbuf_full, (slice(None, None, None), slice(None, nr, None), slice(None, nr, None)))
            else:
                ones_n, zeros_n, ones2, ones_row, abbuf = (ones_full, zeros_full, ones2_full, ones_row_f, abbuf_full)
            sk_src = w32 if sk_cur is None else sk_cur
            T1 = NN('t1')
            C_pre = None
            if li == 0:
                if trim:
                    var = fnp.maximum(fnp.sum(w32 * w32, axis=0), 1e-10)
                elif DN_G0 and n % 128 == 0 and riders:
                    # DN_G0: W^T W = [[W1^T W1, W1^T W2], [., W2^T W2]]: two aliased half Grams (billed on their unique entries, as the full
                    # aliased einsum) + ONE Z-core product for the off-diagonal block (no symmetric tag needed: layer 0 only reads diag(C_pre)
                    # and zero_diag drops the tag anyway; the downstream billing is unchanged, checked in mt_g0.py)
                    h_ = n // 2
                    g0_ = pool.get('dn_g0', (n, n))
                    W1_ = _pview(w32, (slice(None, None, None), slice(None, h_, None)))
                    W2_ = _pview(w32, (slice(None, None, None), slice(h_, None, None)))
                    fnp.einsum('ia,ib->ab', W1_, W1_, out=_pview(g0_, (slice(None, h_, None), slice(None, h_, None))))
                    fnp.einsum('ia,ib->ab', W2_, W2_, out=_pview(g0_, (slice(h_, None, None), slice(h_, None, None))))
                    c12_ = _pview(g0_, (slice(None, h_, None), slice(h_, None, None)))
                    _dn_mm(smm, _pview(fnp.swapaxes(W1_, 0, 1), (None, None)), _pview(W2_, (None, None)), _pview(c12_, (None, None)), DN_G0_LEV)
                    fnp.copyto(_pview(g0_, (slice(h_, None, None), slice(None, h_, None))), fnp.swapaxes(c12_, 0, 1))
                    C_pre = g0_
                else:
                    C_pre = fnp.einsum('ia,ib->ab', w32, w32)
            elif skip_src:
                if trim:
                    var = fnp.maximum(fnp.sum(w32 * (C @ w32), axis=0), 1e-10)
                else:
                    C_pre = fnp.einsum('ij,ia,jb->ab', C, w32, w32)
            if A_st is not None and (not skip_src):
                k = A_st.shape[0]
                WD = _public_out(fnp.multiply, W, _pview(w1_prev, (None, slice(None, None, None))), out=NW('wd', nr, ni))
                WDb = _pview(WD, None)
                # SDT_JOINW: the w1-scaled rows [kin:] of the joiner's legs and of the old basis are zero, so every join / basis product
                # (sketch, range finder, factors, next basis) runs on the first kj rows (exact; the QR sees the same non-zero rows)
                kj = kin if SDT_JOINW and nd_in > 0 and ta_join and (QPASS == 1) and m2_qc else ni
                Qck = Qc if kj == ni or Qc is None else _pview(Qc, slice(None, kj, None))
                WDk = WD if kj == ni else NW('wd', nr, kj)
                ka_t = ka
                if confine and (not last):
                    if TD_GATE_B >= 0:
                        ka_t = max(ka, min(k, sum((1 for b_ in range(L) if b_ + age_old + 1 - (1 if b_ <= TD_GATE_B else 0) <= li))))
                    else:
                        ka_t = max(ka, min(k, li - age_old))
                if ka_t > ka and ka > 0 and (FIJA2 or li in TD_FROZEN) and (n == 1024) and (L == 16):
                    j = ka
                    w1c = _pview(w1_prev, (slice(None, None, None), None))
                    Aj = A_st[j]
                    Pj = P_st[j]
                    dAj = _pview(dA_list[j], (slice(None, None, None), None))
                    dPj = _pview(dP_list[j], (slice(None, None, None), None))
                    kjq = kj
                    if tdp_k > 0 and tdp_k < kj and TDP_JOIN and ta_join:
                        # tdp_patch (real): the join's least-squares fit (Qp, QR, pseudo-inverse, sketch) runs on the first tdp_k rows; the band
                        # rows of the joiner's w1-scaled legs stay carried by the fitted basis rows (_qc_next keeps kj rows)
                        kjq = tdp_k
                    Qp = _public_out(fnp.multiply, w1c if kjq == ni else _pview(w1_prev, (slice(None, kjq, None), None)), Qck if kjq == kj else _pview(Qck, slice(None, kjq, None)), out=_win(pool.get('qp', (n, r_old)), kjq, r_old))
                    if tdp_k > 0 and tdp_k < kj and kjq == kj and 'j' in TDP_MASK:
                        # tdp_patch TDP_MASK j (pre-screen; FLOPs / plans unchanged): zero band rows of Qp = the real path's fit (R of the kept
                        # rows; the pseudo-inverse has zero band columns, so the joiner's band rows drop out of the sketch)
                        fnp.copyto(_pview(Qp, slice(tdp_k, kj, None)), 0.0)
                    if TDP_DEBUG and tdp_k > 0:
                        TDP_LOG.append(('j', li, ni, kj, tdp_k, kjq))
                    if QRB and QRS and FX_TRI > 0:
                        # QRB: the R factor only (factorisation billed once; the same Rq as the reduced QR) -- Qp^+ = Rq^-1 Rq^-T Qp^T below
                        Rq = fnp.linalg.qr(Qp, mode='r')
                        Qn = None
                    else:
                        Qn, Rq = fnp.linalg.qr(Qp)
                    m1 = ka - kb
                    slot = fap5[fa_off + m1]
                    # QRS: the left operand is the pseudo-inverse Qp^+ = Rq^-1 Qn^T (one solve, kj right-hand sides) -> the slots come
                    # out as the final factors; otherwise Qn^T and two solves afterwards
                    if Qn is None:
                        if J3_TRI and J3_TRI_NB > 1 and Rq.shape[0] % J3_TRI_NB == 0:
                            # J3_TRI: both solves by column-block substitution (J3_TRI_NB diagonal blocks)
                            QnT = _j3_qrb_pinv(Rq, Qp, J3_TRI_NB)
                        else:
                            QnT = _fx_tri_solve(Rq, _fx_tri_solve_l(Rq.T, Qp.T, FX_TRI), FX_TRI)
                    else:
                        QnT = ((_fx_tri_solve(Rq, fnp.swapaxes(Qn, 0, 1), FX_TRI) if FX_TRI > 0 else fnp.linalg.solve(Rq, fnp.swapaxes(Qn, 0, 1))) if QRS else fnp.swapaxes(Qn, 0, 1))
                    if ta_join:
                        # cost1: frozen-join sketch Qn^T [A_j | P_j] as ONE batched Strassen family (the two
                        # (r_old, ni, n) products were plain matmuls); slot (fa_off + m1) of fap5 is fap4[2 (fa_off + m1) : +2]
                        smm.mn = TA_JOIN_MIN
                        wj_ = min(n, int(hub_widths[j])) if FX_FJW else n
                        if wj_ < n:
                            # FX_FJW: the joiner's legs are zero beyond its hub width (SDT_HUBW) -> project only those columns
                            _fs = _pview(fap4, slice(2 * (fa_off + m1), 2 * (fa_off + m1) + 2, None))
                            smm.mm(_pview(QnT, (None, None)), _win(_pview(legs['AP0'][j], (slice(None, None, None), None)), kjq, wj_), _win(_fs, r_old, wj_), smm.level(r_old, kjq, wj_, TA_JOIN_LEV))
                            fnp.copyto(_pview(_fs, (..., slice(wj_, None, None))), 0.0)
                        else:
                            smm.mm(_pview(QnT, (None, None)), _win(_pview(legs['AP0'][j], (slice(None, None, None), None)), kjq, n), _pview(fap4, slice(2 * (fa_off + m1), 2 * (fa_off + m1) + 2, None)), smm.level(r_old, kjq, n, TA_JOIN_LEV))
                        smm.mn = STRASSEN_MIN
                    else:
                        fnp.matmul(QnT, Aj, out=slot[0])
                        fnp.matmul(QnT, Pj, out=slot[1])
                    if not QRS:
                        fnp.copyto(slot[0], fnp.linalg.solve(Rq, slot[0]))
                        fnp.copyto(slot[1], fnp.linalg.solve(Rq, slot[1]))
                    FAP = _pview(fap5, slice(fa_off, fa_off + m1 + 1, None))
                    FAo = _pview(FAP, (slice(None, None, None), 0))
                    FPo = _pview(FAP, (slice(None, None, None), 1))
                    FAj = slot[0]
                    FPj = slot[1]
                    if not FIJA2:
                        # (FIJA2: no Gram cores -- nothing reads Sg / S2 once every join is frozen)
                        if ta_gram:
                            Sj = self._gram2(pool, FAj, dA_list[j], FPj, dP_list[j])
                        else:
                            Sj = FAj @ (dAj * FAj.T) + FPj @ (dPj * FPj.T)
                        Sg = Sj if Sg is None else Sg + Sj
                    kb_t = kb
                    if nest:
                        kb_t = max(kb, min(ka_t - 1, _nest_cnt(li, age_old2)))
                    if kb_t > kb and FIJA2:
                        # FIJA2 tier 2 (LordProcciuto X_FIJA, MIT): the mover (oldest tier-1 member) is projected by least squares, in the
                        # metric of this join's basis Qp = Qn Rq, onto the first r2 columns of the fixed basis: Qp[:, :r2] F2 ~ Qp F with
                        # F2 = F[:r2] + Rq[:r2, :r2]^-1 Rq[:r2, r2:] F[r2:].  U = [I; 0] stays fixed: no range finder, no rotation, no cores.
                        assert kb_t == kb + 1, (li, kb, kb_t)
                        if DN_KP and FX_TRI > 0:
                            # DN_KP: R11 is upper triangular -> FX_TRI block bisection (same solution up to rounding)
                            Kp = _fx_tri_solve(_pview(Rq, (slice(None, r2, None), slice(None, r2, None))), _pview(Rq, (slice(None, r2, None), slice(r2, None, None))), FX_TRI)
                        elif J2_KP:
                            # J2_KP: R11 is upper triangular -> depth-2 block bisection (same solution up to rounding, +12 ops)
                            Kp = _fx_tri_solve(_pview(Rq, (slice(None, r2, None), slice(None, r2, None))), _pview(Rq, (slice(None, r2, None), slice(r2, None, None))), 2)
                        else:
                            Kp = fnp.linalg.solve(_pview(Rq, (slice(None, r2, None), slice(None, r2, None))), _pview(Rq, (slice(None, r2, None), slice(r2, None, None))))
                        fap2_cur, fap24_cur = pool.get_pair(('fap2', f2_side), (L - 1, 2, r2, n))
                        wm_ = min(n, int(hub_widths[kb])) if FX_NESTW else n
                        if FX_NESTW and FX_NESTS:
                            # FX_NESTS: Kp [FA | FP][r2:, :wm] as one batched Strassen product into the tier-2 slot, then + F[:r2]
                            Fm_ = _pview(fap4, slice(2 * fa_off, 2 * fa_off + 2, None))
                            o2_ = _pview(fap24_cur, slice(2 * kb, 2 * kb + 2, None))
                            o2w_ = _pview(o2_, (slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, wm_, None)))
                            _tier2_mm(smm, Kp, _pview(Fm_, (slice(None, None, None), slice(None, None, None), slice(r2, None, None), slice(None, wm_, None))), o2w_)
                            fnp.add(o2w_, _pview(Fm_, (slice(None, None, None), slice(None, None, None), slice(None, r2, None), slice(None, wm_, None))), out=o2w_)
                            if wm_ < n:
                                fnp.copyto(_pview(o2_, (..., slice(wm_, None, None))), 0.0)
                        elif FX_NESTW:
                            # FX_NESTW: the mover's factors are zero beyond its hub width -> only those columns
                            for t_, F_ in ((0, FAo[0]), (1, FPo[0])):
                                fnp.add(_pview(F_, (slice(None, r2, None), slice(None, wm_, None))), fnp.matmul(Kp, _pview(F_, (slice(r2, None, None), slice(None, wm_, None)))), out=_pview(fap2_cur[kb, t_], (slice(None, None, None), slice(None, wm_, None))))
                                if wm_ < n:
                                    fnp.copyto(_pview(fap2_cur[kb, t_], (slice(None, None, None), slice(wm_, None, None))), 0.0)
                        else:
                            for t_, F_ in ((0, FAo[0]), (1, FPo[0])):
                                fnp.add(_pview(F_, slice(None, r2, None)), fnp.matmul(Kp, _pview(F_, slice(r2, None, None))), out=fap2_cur[kb, t_])
                        FAP2 = _pview(fap2_cur, slice(None, kb + 1, None))
                        fap24 = fap24_cur
                        FA2 = _pview(FAP2, (slice(None, None, None), 0))
                        FP2 = _pview(FAP2, (slice(None, None, None), 1))
                        if U is None:
                            U = fnp.zeros((r_old, r2), dtype=f32)
                            fnp.fill_diagonal(U, 1.0)
                        S2 = None
                        FAP = _pview(FAP, slice(1, None, None))
                        fa_off += 1
                        FAo = _pview(FAP, (slice(None, None, None), 0))
                        FPo = _pview(FAP, (slice(None, None, None), 1))
                        kb = kb_t
                    elif kb_t > kb:
                        assert kb_t == kb + 1, (li, kb, kb_t)
                        jb = kb
                        FA1 = FAo[0]
                        FP1 = FPo[0]
                        dAb = _pview(dA_list[jb], (slice(None, None, None), None))
                        dPb = _pview(dP_list[jb], (slice(None, None, None), None))
                        if ta_gram:
                            S_s = self._gram2(pool, FA1, dA_list[jb], FP1, dP_list[jb])
                        else:
                            S_s = FA1 @ (dAb * FA1.T) + FP1 @ (dPb * FP1.T)
                        G2 = S_s if kb == 0 else S_s + U @ (S2 @ U.T)
                        Om2 = fnp.copy(_pview(w32, (slice(None, r_old, None), slice(None, r2, None))))
                        for _pass in range(int(self.QPASS2)):
                            Un, _ = fnp.linalg.qr(G2 @ Om2)
                            Om2 = Un
                        f2_side ^= 1
                        fap2_new, fap24_new = pool.get_pair(('fap2', f2_side), (L - 1, 2, r2, n))
                        if TA_TIER2 and _TAG['join']:
                            _tier2_mm(smm, Un.T, _pview(fap4, slice(2 * fa_off, 2 * (fa_off + 1), None)), _pview(fap24_new, slice(2 * kb, 2 * (kb + 1), None)))
                        else:
                            fnp.matmul(Un.T, FA1, out=fap2_new[kb, 0])
                            fnp.matmul(Un.T, FP1, out=fap2_new[kb, 1])
                        if kb > 0:
                            T2 = Un.T @ U
                            if TA_TIER2 and _TAG['join'] and (2 * kb >= TA_TIER2_ROT_SLOTS):
                                _tier2_mm(smm, T2, _pview(fap24, slice(None, 2 * kb, None)), _pview(fap24_new, slice(None, 2 * kb, None)))
                            else:
                                fnp.matmul(_pview(T2, (None, None)), FAP2, out=_pview(fap2_new, slice(None, kb, None)))
                        FAP2 = _pview(fap2_new, slice(None, kb + 1, None))
                        fap24 = fap24_new
                        FA2 = _pview(FAP2, (slice(None, None, None), 0))
                        FP2 = _pview(FAP2, (slice(None, None, None), 1))
                        S2 = Un.T @ (G2 @ Un)
                        U = Un
                        FAP = _pview(FAP, slice(1, None, None))
                        fa_off += 1
                        FAo = _pview(FAP, (slice(None, None, None), 0))
                        FPo = _pview(FAP, (slice(None, None, None), 1))
                        Sg = Sg - S_s
                        kb = kb_t
                    qc_side ^= 1
                    if tdp_k > 0 and tdp_k < kj and (TDP_QC or 'q' in TDP_MASK):
                        if TDP_QC:
                            # tdp_patch (real) TDP_QC: the old basis is transported over the first tdp_k rows only (hard cut of the band)
                            WDk = NW('wd', nr, tdp_k)
                            Qck = _pview(Qck, slice(None, tdp_k, None))
                        else:
                            # tdp_patch TDP_MASK q (pre-screen): band rows of the old basis zeroed before its transport (their last read)
                            fnp.copyto(_pview(Qck, slice(tdp_k, kj, None)), 0.0)
                    Qc = _qc_next(smm, WDk, Qck, _win(pool.get(('qc', qc_side), (n, r_old)), nr, r_old), m2_qc)
                    ka = ka_t
                elif ka_t > ka:
                    j = ka
                    w1c = _pview(w1_prev, (slice(None, None, None), None))
                    Aj = A_st[j]
                    Pj = P_st[j]
                    dAj = _pview(dA_list[j], (slice(None, None, None), None))
                    dPj = _pview(dP_list[j], (slice(None, None, None), None))
                    Om = _win(pool.get('om', (n, r_old)), kj, r_old)
                    fnp.copyto(Om, _pview(w32in, (slice(None, kj, None), slice(None, r_old, None))))
                    Qp = _public_out(fnp.multiply, w1c if kj == ni else _pview(w1_prev, (slice(None, kj, None), None)), Qck, out=_win(pool.get('qp', (n, r_old)), kj, r_old)) if ka > 0 else None
                    Yq = _win(pool.get('yq', (n, r_old)), kj, r_old)
                    Yq1 = pool.get('yq1', (n, r_old))
                    Yq2 = _win(pool.get('yq2', (n, r_old)), kj, r_old)
                    pair4 = _win(_pview(legs['AP0'][j], (slice(None, None, None), None)), kj, n)
                    for _pass in range(QPASS):
                        if ta_join and QPASS == 1 and (not (bq_srcp and ka > 0)):
                            smm.mn = TA_JOIN_MIN
                            Y1T = pool.get('y1t', (2, 1, r_old, n))
                            smm.mm(_pview(fnp.swapaxes(Om, 0, 1), (None, None)), pair4, Y1T, smm.level(r_old, kj, n, TA_JOIN_LEV))
                            dAP = pool.get('dap', (2, 1, 1, n))
                            fnp.copyto(dAP[0, 0, 0], dA_list[j])
                            fnp.copyto(dAP[1, 0, 0], dP_list[j])
                            fnp.multiply(Y1T, dAP, out=Y1T)
                            outH = _win(pool.get('yqh', (1, n, r_old)), kj, r_old)
                            smm.hub(pair4, Y1T, outH, smm.level(kj, n, r_old, TA_JOIN_LEV))
                            smm.mn = STRASSEN_MIN
                            Yq = outH[0]
                        elif ka == 0 or not bq_srcp:
                            fnp.matmul(Aj.T, Om, out=Yq1)
                            fnp.multiply(dAj, Yq1, out=Yq1)
                            fnp.matmul(Aj, Yq1, out=Yq)
                            fnp.matmul(Pj.T, Om, out=Yq1)
                            fnp.multiply(dPj, Yq1, out=Yq1)
                            fnp.matmul(Pj, Yq1, out=Yq2)
                            fnp.add(Yq, Yq2, out=Yq)
                        elif ta_join and QPASS == 1:
                            # cost1: P-leg-only joiner sketch P_j d(dP) P_j^T Om on the same pooled Strassen
                            # family as the two-leg path above (slot 1 of pair4 / y1t / dap)
                            smm.mn = TA_JOIN_MIN
                            Y1P = _pview(pool.get('y1t', (2, 1, r_old, n)), slice(1, 2, None))
                            pairP = _pview(pair4, slice(1, 2, None))
                            smm.mm(_pview(fnp.swapaxes(Om, 0, 1), (None, None)), pairP, Y1P, smm.level(r_old, kj, n, TA_JOIN_LEV))
                            dAP = pool.get('dap', (2, 1, 1, n))
                            fnp.copyto(_pview(dAP, (1, 0, 0)), dP_list[j])
                            fnp.multiply(Y1P, _pview(dAP, slice(1, 2, None)), out=Y1P)
                            outH = _win(pool.get('yqh', (1, n, r_old)), kj, r_old)
                            smm.hub(pairP, Y1P, outH, smm.level(kj, n, r_old, TA_JOIN_LEV))
                            smm.mn = STRASSEN_MIN
                            Yq = outH[0]
                        else:
                            fnp.matmul(Pj.T, Om, out=Yq1)
                            fnp.multiply(dPj, Yq1, out=Yq1)
                            fnp.matmul(Pj, Yq1, out=Yq)
                        if ka > 0:
                            Sfull = Sg if kb == 0 else Sg + U @ (S2 @ U.T)
                            if 2 in C7_S:
                                _t1 = _c7mm(smm, 2, Qp.T, Om, pool.get('c7_ro1', (r_old, r_old)))
                                _t2 = _c7mm(smm, 2, Sfull, _t1, pool.get('c7_ro2', (r_old, r_old)))
                                _c7mm(smm, 2, Qp, _t2, Yq2)
                            else:
                                fnp.matmul(Qp, Sfull @ (Qp.T @ Om), out=Yq2)
                            fnp.add(Yq, Yq2, out=Yq)
                        Qn, _ = fnp.linalg.qr(Yq)
                        Om = Qn
                    fa_side ^= 1
                    fap_new, fap4_new = pool.get_pair(('fap', fa_side), (L - 1, 2, r_old, n))
                    m1 = ka - kb
                    if ta_join:
                        smm.mn = TA_JOIN_MIN
                        smm.mm(_pview(fnp.swapaxes(Qn, 0, 1), (None, None)), pair4, _pview(fap4_new, slice(2 * m1, 2 * m1 + 2, None)), smm.level(r_old, kj, n, TA_JOIN_LEV))
                        smm.mn = STRASSEN_MIN
                        FAj = fap_new[m1, 0]
                        FPj = fap_new[m1, 1]
                    else:
                        FAj = _public_out(fnp.matmul, Qn.T, Aj, out=fap_new[m1, 0])
                        FPj = _public_out(fnp.matmul, Qn.T, Pj, out=fap_new[m1, 1])
                    if ka > 0:
                        Tq = _c7mm(smm, 3, Qn.T, Qp, pool.get('c7_tq', (r_old, r_old))) if 3 in C7_S else Qn.T @ Qp
                        if ka > kb:
                            if ta_join:
                                smm.mn = TA_JOIN_MIN
                                smm.mm(_pview(Tq, (None, None)), _pview(fap4, slice(2 * fa_off, 2 * (fa_off + m1), None)), _pview(fap4_new, slice(None, 2 * m1, None)), smm.level(r_old, r_old, n, TA_JOIN_LEV))
                                smm.mn = STRASSEN_MIN
                            else:
                                fnp.matmul(_pview(Tq, (None, None)), FAP, out=_pview(fap_new, slice(None, m1, None)))
                            Sg = _c7mm(smm, 4, Tq, _c7mm(smm, 4, Sg, Tq.T, pool.get('c7_sg1', (r_old, r_old))), pool.get('c7_sg2', (r_old, r_old))) if 4 in C7_S else Tq @ (Sg @ Tq.T)
                        else:
                            Sg = None
                        if kb > 0:
                            U = Tq @ U
                    FAP = _pview(fap_new, slice(None, m1 + 1, None))
                    fap4 = fap4_new
                    fap5 = fap_new
                    fa_off = 0
                    FAo = _pview(FAP, (slice(None, None, None), 0))
                    FPo = _pview(FAP, (slice(None, None, None), 1))
                    if ta_gram:
                        Sj = self._gram2(pool, FAj, dA_list[j], FPj, dP_list[j])
                    else:
                        Sj = FAj @ (dAj * FAj.T) + FPj @ (dPj * FPj.T)
                    Sg = Sj if Sg is None else Sg + Sj
                    if FIJA2 and ka == 0 and kb == 0 and nest:
                        # FIJA2 (LordProcciuto X_FIJA_RC ordering, MIT): order the basis once so that its first r2 columns span the top-r2
                        # subspace of the first joiner's Gram (subspace iteration, FIJA2_PASS passes); tier 2 later lives on those columns
                        if DN_EIGH:
                            # DN_EIGH (APPROXIMATION): the exact top-r2 eigenspace of Sg first (eigh is ascending -> reversed columns);
                            # the complement columns are its remaining eigenvectors (any orthonormal complement serves)
                            _ev, _V = fnp.linalg.eigh(Sg)
                            Vr = fnp.flip(_V, axis=1)
                        else:
                            Ur = fnp.copy(_pview(Sg, (slice(None, None, None), slice(None, r2, None))))
                            for _pass in range(FIJA2_PASS):
                                if DN_FSUB:
                                    # DN_FSUB: Sg Ur through the Z core (the QR re-orthonormalises; Strassen-level rounding only)
                                    _su = pool.get('dn_fsub', (r_old, r2))
                                    _dn_mm(smm, _pview(Sg, (None, None)), _pview(Ur, (None, None)), _pview(_su, (None, None)), DN_FSUB_LEV)
                                    Ur, _ = fnp.linalg.qr(_su)
                                else:
                                    Ur, _ = fnp.linalg.qr(Sg @ Ur)
                            if QRV:
                                # QRV: complement of span(Ur) from the complete QR of Ur (first r2 columns unchanged; consumers covariant)
                                Vr = fnp.linalg.qr(Ur, mode='complete')[0]
                            else:
                                Ie = fnp.zeros((r_old, r_old - r2), dtype=f32)
                                fnp.fill_diagonal(Ie, 1.0)
                                Vr, _ = fnp.linalg.qr(fnp.concatenate([Ur, Ie], axis=1))
                        VrT = fnp.swapaxes(Vr, 0, 1)
                        if DN_FROT:
                            # DN_FROT: Vr^T [FA_j | FP_j] as ONE batched Z-core family (slot pair of fap4_new) and Qn Vr through the Z core
                            _fs = _pview(fap4_new, slice(2 * m1, 2 * m1 + 2, None))
                            _fr = pool.get('dn_frot', (2, 1, r_old, n))
                            _dn_mm(smm, _pview(VrT, (None, None)), _fs, _fr, DN_FROT_LEV)
                            fnp.copyto(_fs, _fr)
                            _qv = _win(pool.get('dn_qnv', (n, r_old)), Qn.shape[0], r_old)
                            _dn_mm(smm, _pview(Qn, (None, None)), _pview(Vr, (None, None)), _pview(_qv, (None, None)), DN_FROT_LEV)
                            Qn = _qv
                        else:
                            fnp.copyto(fap_new[m1, 0], fnp.matmul(VrT, fap_new[m1, 0]))
                            fnp.copyto(fap_new[m1, 1], fnp.matmul(VrT, fap_new[m1, 1]))
                            Qn = fnp.matmul(Qn, Vr)
                        if not FX_SGDEAD:
                            Sg = fnp.matmul(VrT, fnp.matmul(Sg, Vr))
                    kb_t = kb
                    if nest:
                        kb_t = max(kb, min(ka_t - 1, _nest_cnt(li, age_old2)))
                    if kb_t > kb:
                        assert kb_t == kb + 1, (li, kb, kb_t)
                        jb = kb
                        FA1 = FAo[0]
                        FP1 = FPo[0]
                        dAb = _pview(dA_list[jb], (slice(None, None, None), None))
                        dPb = _pview(dP_list[jb], (slice(None, None, None), None))
                        if ta_gram:
                            S_s = self._gram2(pool, FA1, dA_list[jb], FP1, dP_list[jb])
                        else:
                            S_s = FA1 @ (dAb * FA1.T) + FP1 @ (dPb * FP1.T)
                        G2 = S_s if kb == 0 else S_s + U @ (S2 @ U.T)
                        Om2 = fnp.copy(_pview(w32, (slice(None, r_old, None), slice(None, r2, None))))
                        for _pass in range(int(self.QPASS2)):
                            Un, _ = fnp.linalg.qr(G2 @ Om2)
                            Om2 = Un
                        f2_side ^= 1
                        fap2_new, fap24_new = pool.get_pair(('fap2', f2_side), (L - 1, 2, r2, n))
                        if TA_TIER2 and _TAG['join']:
                            _tier2_mm(smm, Un.T, _pview(fap4, slice(2 * fa_off, 2 * (fa_off + 1), None)), _pview(fap24_new, slice(2 * kb, 2 * (kb + 1), None)))
                        else:
                            fnp.matmul(Un.T, FA1, out=fap2_new[kb, 0])
                            fnp.matmul(Un.T, FP1, out=fap2_new[kb, 1])
                        if kb > 0:
                            T2 = Un.T @ U
                            if TA_TIER2 and _TAG['join'] and (2 * kb >= TA_TIER2_ROT_SLOTS):
                                _tier2_mm(smm, T2, _pview(fap24, slice(None, 2 * kb, None)), _pview(fap24_new, slice(None, 2 * kb, None)))
                            else:
                                fnp.matmul(_pview(T2, (None, None)), FAP2, out=_pview(fap2_new, slice(None, kb, None)))
                        FAP2 = _pview(fap2_new, slice(None, kb + 1, None))
                        fap24 = fap24_new
                        FA2 = _pview(FAP2, (slice(None, None, None), 0))
                        FP2 = _pview(FAP2, (slice(None, None, None), 1))
                        S2 = Un.T @ (G2 @ Un)
                        U = Un
                        FAP = _pview(FAP, slice(1, None, None))
                        fa_off += 1
                        FAo = _pview(FAP, (slice(None, None, None), 0))
                        FPo = _pview(FAP, (slice(None, None, None), 1))
                        Sg = Sg - S_s
                        kb = kb_t
                    qc_side ^= 1
                    if ta_join:
                        Qc = _win(pool.get(('qc', qc_side), (n, r_old)), nr, r_old)
                        smm.mn = TA_JOIN_MIN
                        smm.mm(_pview(W, (None, None)) if kj == ni else _pview(W, (None, None, slice(None, None, None), slice(None, kj, None))), _pview(Qn, (None, None)), _pview(Qc, (None, None)), smm.level(nr, kj, r_old, TA_JOIN_LEV))
                        smm.mn = STRASSEN_MIN
                    else:
                        Qc = _public_out(fnp.matmul, W, Qn, out=_win(pool.get(('qc', qc_side), (n, r_old)), nr, r_old))
                    ka = ka_t
                elif ka > 0:
                    if nest and (TD_GATE_B >= 0 or TD_GATE2_B >= 0) and (not last):
                        kb_t = max(kb, min(ka - 1, _nest_cnt(li, age_old2)))
                        if kb_t > kb:
                            assert kb_t == kb + 1, (li, kb, kb_t)
                            jb = kb
                            FA1 = FAo[0]
                            FP1 = FPo[0]
                            dAb = _pview(dA_list[jb], (slice(None, None, None), None))
                            dPb = _pview(dP_list[jb], (slice(None, None, None), None))
                            S_s = FA1 @ (dAb * FA1.T) + FP1 @ (dPb * FP1.T)
                            G2 = S_s if kb == 0 else S_s + U @ (S2 @ U.T)
                            Om2 = fnp.copy(_pview(w32, (slice(None, r_old, None), slice(None, r2, None))))
                            for _pass in range(int(self.QPASS2)):
                                Un, _ = fnp.linalg.qr(G2 @ Om2)
                                Om2 = Un
                            f2_side ^= 1
                            fap2_new, fap24_new = pool.get_pair(('fap2', f2_side), (L - 1, 2, r2, n))
                            if TA_TIER2 and _TAG['join']:
                                _tier2_mm(smm, Un.T, _pview(fap4, slice(2 * fa_off, 2 * (fa_off + 1), None)), _pview(fap24_new, slice(2 * kb, 2 * (kb + 1), None)))
                            else:
                                fnp.matmul(Un.T, FA1, out=fap2_new[kb, 0])
                                fnp.matmul(Un.T, FP1, out=fap2_new[kb, 1])
                            if kb > 0:
                                T2 = Un.T @ U
                                if TA_TIER2 and _TAG['join'] and (2 * kb >= TA_TIER2_ROT_SLOTS):
                                    _tier2_mm(smm, T2, _pview(fap24, slice(None, 2 * kb, None)), _pview(fap24_new, slice(None, 2 * kb, None)))
                                else:
                                    fnp.matmul(_pview(T2, (None, None)), FAP2, out=_pview(fap2_new, slice(None, kb, None)))
                            FAP2 = _pview(fap2_new, slice(None, kb + 1, None))
                            fap24 = fap24_new
                            FA2 = _pview(FAP2, (slice(None, None, None), 0))
                            FP2 = _pview(FAP2, (slice(None, None, None), 1))
                            S2 = Un.T @ (G2 @ Un)
                            U = Un
                            FAP = _pview(FAP, slice(1, None, None))
                            fa_off += 1
                            FAo = _pview(FAP, (slice(None, None, None), 0))
                            FPo = _pview(FAP, (slice(None, None, None), 1))
                            Sg = Sg - S_s
                            kb = kb_t
                    qc_side ^= 1
                    Qc = _qc_next(smm, WDk, Qck, _win(pool.get(('qc', qc_side), (n, r_old)), nr, r_old), m2_qc)
                if ka > kb:
                    m1 = ka - kb
                    smm.mn = _TAG['old_min']
                    _ap1s = _pview(legs4['AP1'], slice(2 * kb, 2 * ka, None))
                    # SDT_OLDW: the tier-1 legs are zero beyond the widest hub window of their sources (the factors inherit the zero columns
                    # of the joined legs), so only those columns are formed; the rest of the slots is zeroed
                    ow1 = max(_pview(hub_widths, slice(kb, ka, None)), default=n) if SDT_OLDW else n
                    self._sdt_ow1 = ow1
                    if (OLD_HOT1 & 1) and (not last) and (not merge_cp) and riders and as_on:
                        # OLD_HOT1: deferred until the exact variance is known (below, with the tier-2 rebuild)
                        t1_pending = (kb, ka, fa_off, m1, ow1, qc_side)
                    elif AU_T1PK and SDT_OLDW and riders and as_on and _au_t1plan(list(_pview(hub_widths, slice(kb, ka, None))), AU_T1PK_EXT and FIJA2 and kb >= 1 and fa_off >= 1 and not smm.first and nr % 64 == 0) is not None:
                        # AU_T1PK (no D21 demand reads the packed factors afterwards on this path): pack, rebuild on wb, move the rebuilt
                        # overflow columns back; the tier-2 rebuild below writes the extra leg slot (ext) afterwards
                        pl1_ = _au_t1plan(list(_pview(hub_widths, slice(kb, ka, None))), AU_T1PK_EXT and FIJA2 and kb >= 1 and fa_off >= 1 and not smm.first and nr % 64 == 0)
                        wb1_ = pl1_[1]
                        _fs1 = _pview(fap4, slice(2 * fa_off, 2 * (fa_off + m1), None))
                        _a1w = _win(_ap1s, nr, n)
                        if pl1_[0] == 'pair':
                            for js_, a_, b_, jt_, c_, d_ in pl1_[2]:
                                fnp.copyto(_au_pair(_fs1, jt_, slice(c_, d_, None)), _au_pair(_fs1, js_, slice(a_, b_, None)))
                            _fx1, _ax1 = _fs1, _ap1s
                        else:
                            _fx1 = _pview(fap4, slice(2 * fa_off - 1, 2 * (fa_off + m1), None))
                            _ax1 = _pview(legs4['AP1'], slice(2 * kb - 1, 2 * ka, None))
                            _x0 = _pview(_fx1, (0, 0))
                            used_ = 0
                            for s0_, s1_, w_, off_ in pl1_[2]:
                                fnp.copyto(_au_extc(self, _x0, off_, s1_ - s0_, w_ - wb1_), _au_slots(_fs1, s0_, s1_, slice(None, None, None), slice(wb1_, w_, None)))
                                used_ = off_ + (s1_ - s0_) * (w_ - wb1_)
                            if used_ < wb1_:
                                fnp.copyto(_pview(_x0, (slice(None, None, None), slice(used_, wb1_, None))), 0.0)
                        smm.mm(_pview(Qc, (None, None)), _win(_fx1, r_old, wb1_), _win(_ax1, nr, wb1_), smm.level(nr, r_old, wb1_, s_sb), pad=(_pview(pool.get(('qc', qc_side), (n, r_old)), (None, None)), None, _ax1))
                        fnp.copyto(_pview(_a1w, (..., slice(wb1_, None, None))), 0.0)
                        if pl1_[0] == 'pair':
                            for js_, a_, b_, jt_, c_, d_ in pl1_[2]:
                                fnp.copyto(_au_pair(_a1w, js_, slice(a_, b_, None)), _au_pair(_a1w, jt_, slice(c_, d_, None)))
                                fnp.copyto(_au_pair(_a1w, jt_, slice(c_, d_, None)), 0.0)
                                fnp.copyto(_au_pair(_fs1, jt_, slice(c_, d_, None)), 0.0)
                        else:
                            _o0 = _pview(_ax1, (0, 0, slice(None, nr, None)))
                            for s0_, s1_, w_, off_ in pl1_[2]:
                                fnp.copyto(_au_slots(_ap1s, s0_, s1_, slice(None, nr, None), slice(wb1_, w_, None)), _au_extc(self, _o0, off_, s1_ - s0_, w_ - wb1_))
                    elif ow1 < n:
                        smm.mm(_pview(Qc, (None, None)), _win(_pview(fap4, slice(2 * fa_off, 2 * (fa_off + m1), None)), r_old, ow1), _win(_ap1s, nr, ow1), smm.level(nr, r_old, ow1, s_sb), pad=(_pview(pool.get(('qc', qc_side), (n, r_old)), (None, None)), None, _ap1s))
                        fnp.copyto(_pview(_win(_ap1s, nr, n), (..., slice(ow1, None, None))), 0.0)
                    else:
                        smm.mm(_pview(Qc, (None, None)), _pview(fap4, slice(2 * fa_off, 2 * (fa_off + m1), None)), _win(_ap1s, nr, n), smm.level(nr, r_old, n, s_sb), pad=(_pview(pool.get(('qc', qc_side), (n, r_old)), (None, None)), None, _ap1s))
                    smm.mn = STRASSEN_MIN
                if kb > 0:
                    if FX_UEYE and FIJA2:
                        # FX_UEYE: U = [I; 0] -> Qc U = Qc[:, :r2]: a copy into the pooled window (still the operand / pad parent below)
                        QU = _win(pool.get('qu', (n, r2)), nr, r2)
                        fnp.copyto(QU, _pview(Qc, (slice(None, None, None), slice(None, r2, None))))
                    else:
                        QU = _c7mm(smm, 5, Qc, U, _win(pool.get('qu', (n, r2)), nr, r2)) if 5 in C7_S else _public_out(fnp.matmul, Qc, U, out=_win(pool.get('qu', (n, r2)), nr, r2))
                    if OLD_ROWW and (not last) and (not merge_cp) and riders and as_on:
                        # OLD_ROWW: deferred until the exact variance is known (below, before the slices)
                        t2_pending = kb
                    else:
                        smm.mn = _TAG['old_min']
                        _ap1s = _pview(legs4['AP1'], slice(None, 2 * kb, None))
                        smm.mm(_pview(QU, (None, None)), _pview(fap24, slice(None, 2 * kb, None)), _win(_ap1s, nr, n), smm.level(nr, r2, n, s_sb), pad=(_pview(pool.get('qu', (n, r2)), (None, None)), None, _ap1s))
                        smm.mn = STRASSEN_MIN
                extra = 2 if newborn is not None else 0
                if 2 * k + extra > 2 * ka:
                    nsl = 2 * k + extra - 2 * ka
                    smm.mn = self._mn_y
                    # SDT_HUBW: a source's hub width is its n3 (<= birth width); the newborn's P slot holds the covariance C, which needs all
                    # ni columns: either the batch is widened to ni or (SDT_CSPLIT) its columns [hcols:ni] come from one dense product below
                    hcols = max(_pview(hub_widths, slice(ka, k + (1 if extra else 0), None)) + ([ni] if extra and (not SDT_CSPLIT) else []), default=n) if YCOL_TRIM and riders and as_on else n
                    assert (ni <= hcols or SDT_CSPLIT) and hcols <= n if extra else 0 < hcols <= n
                    self._young_column_shapes.append((li, nr, ni, nsl, hcols))
                    hc_young = hcols
                    # SDT_KWIN: rows [kin:] of every w1-scaled young slot are zero (nearly-dead inputs), so the batch contracts over the first
                    # kin rows only; the covariance slot (newborn P) is repaired below with its dense tail product
                    kt = kin if SDT_KWIN and nd_in > 0 else ni
                    if tdp_k > 0:
                        tdp_sl = slice(2 * ka, 2 * k + (1 if extra else 0), None)   # w1-scaled input slots (all but the newborn's covariance)
                        if TDP_Y:
                            # tdp_patch (real): the batch contracts over the first tdp_k rows (=> kwin_fix below when there is a newborn); the band
                            # [tdp_k:kin] of the w1-scaled slots enters through the rank-TDP_R term after the batch (TDP_R = 0: hard cut); the
                            # covariance slot's rows [tdp_k:ni] through the widened SDT_KWIN repair (exact)
                            kt = tdp_k
                            tdp_y = True
                        elif 'y' in TDP_MASK:
                            # tdp_patch TDP_MASK y (pre-screen; FLOPs / plans unchanged): the band rows of the w1-scaled input slots are zeroed
                            # (TDP_R = 0) or replaced in place by their rank-TDP_R projection = the real path's predictions
                            _tdp_mask_band(_pview(legs4['AP0'], (tdp_sl, 0, slice(tdp_k, kin, None), slice(None, hcols, None))), TDP_R, _pview(w32, (slice(None, hcols, None), slice(None, max(1, TDP_R), None))))
                        if TDP_DEBUG:
                            TDP_LOG.append(('y', li, nr, ni, kin, tdp_k, nsl, hcols, int(tdp_y), int(extra > 0)))
                    _lv_y = smm.level(nr, kt, hcols, s_yh if TA_YH_S6 <= nsl <= TA_YH_SMAX else min(s_yh, 5))
                    if smm.first and TA_YH_LEV > s_yh and (not as_on):
                        _st = smm.level(nr, ni, n, TA_YH_LEV if TA_YH_S6 <= nsl <= TA_YH_SMAX else min(TA_YH_LEV, 5))
                        smm.steady = _st if _st > _lv_y else 0
                    ws_y = list(_pview(hub_widths, slice(ka, k + (1 if extra else 0), None)))
                    if YR_T and YCOL_TRIM and riders and as_on and SDT_CSPLIT and len(ws_y) >= 2:
                        yr_groups = _yr_split(ws_y, 2, ni if extra else 0, YR_DW, YR_MINSAVE)
                    else:
                        yr_groups = [(0, len(ws_y), hcols)]
                    if len(yr_groups) > 1:
                        # YR: the newborn (last source) sits in the last group; its covariance slot's batch width is that group's width
                        hc_young = yr_groups[-1][2]
                        YR_LOG.append(('t', li, [(ka + g0_, ka + g1_, wg_) for g0_, g1_, wg_ in yr_groups], hcols))
                    for g0_, g1_, wg_ in yr_groups:
                        s0_ = 2 * (ka + g0_)
                        s1_ = 2 * (ka + g1_)
                        nsl_g = s1_ - s0_
                        _lv_g = _lv_y if len(yr_groups) == 1 else smm.level(nr, kt, wg_, s_yh if TA_YH_S6 <= nsl_g <= TA_YH_SMAX else min(s_yh, 5))
                        _ap0s = _pview(legs4['AP0'], slice(s0_, s1_, None))
                        _ap1s = _pview(legs4['AP1'], slice(s0_, s1_, None))
                        if kt < ni:
                            smm.mm(_pview(W, (None, None, slice(None, None, None), slice(None, kt, None))), _win(_ap0s, kt, wg_), _win(_ap1s, nr, wg_), _lv_g, pad=(None, _ap0s, _ap1s))
                            kwin_fix = extra > 0
                        else:
                            smm.mm(_pview(W, (None, None)), _win(_ap0s, ni, wg_), _win(_ap1s, nr, wg_), _lv_g, pad=(None, _ap0s, _ap1s))
                        if wg_ < n:
                            fnp.copyto(_pview(_win(_ap1s, nr, n), (..., slice(wg_, None, None))), 0.0)
                    if tdp_y and TDP_R > 0:
                        # tdp_patch (real): out_s += (W[:, band] U_s)(U_s^T X_s[band]) on the w1-scaled slots; X_s is exactly zero beyond a slot's
                        # own width, so the term is too (the zeroed output columns stay zero)
                        xb_t = _pview(legs4['AP0'], (tdp_sl, 0, slice(tdp_k, kin, None), slice(None, hcols, None)))
                        q_t = _tdp_basis(xb_t, TDP_R, _pview(w32, (slice(None, hcols, None), slice(None, TDP_R, None))))
                        wq_t = fnp.matmul(_pview(W, (slice(None, None, None), slice(tdp_k, kin, None))), q_t)
                        ob_t = _pview(legs4['AP1'], (tdp_sl, 0, slice(None, nr, None), slice(None, hcols, None)))
                        fnp.add(ob_t, fnp.matmul(wq_t, fnp.matmul(fnp.swapaxes(q_t, -1, -2), xb_t)), out=ob_t)
                    smm.mn = STRASSEN_MIN
                z_side ^= 1
                if (DN_Z2D and li >= DN_Z2D_LMIN) or (DN_ZKIN and kin < ni):
                    Z_st, Zf_st = self._dn_ztransport(smm, Z_st, Zf_st, k, kz, nr, ni, kin if DN_ZKIN else ni, z_side, li, r, rfb, L)
                else:
                    Z_st = _public_out(fnp.matmul, WDb, Z_st, out=_pview(_win(pool.get(('z', z_side), (L - 1, n, r + 2)), nr, r + 2), slice(None, k, None)))
                if Zf_st is not None and k > kz and not ((DN_Z2D and li >= DN_Z2D_LMIN) or (DN_ZKIN and kin < ni)):
                    Zf_st = _public_out(fnp.matmul, WDb, _pview(Zf_st, slice(kz, None, None)), out=_pview(_win(pool.get(('zf', z_side), (L - 1, n, 2 * rfb)), nr, 2 * rfb), slice(kz, k, None)))
                    Zf_st = _pview(_win(pool.get(('zf', z_side), (L - 1, n, 2 * rfb)), nr, 2 * rfb), slice(None, k, None))
                legs['AP0'], legs['AP1'] = (legs['AP1'], legs['AP0'])
                legs4['AP0'], legs4['AP1'] = (legs4['AP1'], legs4['AP0'])
            else:
                k = 0
            if newborn is not None and (not skip_src):
                a_b, Rr, Lr, s_b, e_b, Ff = newborn
                if A_st is None:
                    smm.mn = self._mn_y
                    _lv_y = smm.level(nr, ni, n, s_yh if 2 >= TA_YH_S6 else min(s_yh, 5))
                    if smm.first and TA_YH_LEV > s_yh and (not as_on):
                        _st = smm.level(nr, ni, n, TA_YH_LEV if 2 >= TA_YH_S6 else min(TA_YH_LEV, 5))
                        smm.steady = _st if _st > _lv_y else 0
                    _ap0s = _pview(legs4['AP0'], slice(0, 2, None))
                    _ap1s = _pview(legs4['AP1'], slice(0, 2, None))
                    smm.mm(_pview(W, (None, None)), _win(_ap0s, ni, n), _win(_ap1s, nr, n), _lv_y, pad=(None, _ap0s, _ap1s))
                    smm.mn = STRASSEN_MIN
                    legs['AP0'], legs['AP1'] = (legs['AP1'], legs['AP0'])
                    legs4['AP0'], legs4['AP1'] = (legs4['AP1'], legs4['AP0'])
                WC = _win(legs['AP0'][k, 1], nr, ni)
                if A_st is not None:
                    hc_ = min(hc_young, ni)
                    if kwin_fix:
                        # SDT_KWIN repair (exact): W C = W[:, :kin] C[:kin, :] (batch) + W[:, kin:] C[kin:, :]; C is the pre-swap slot
                        WCw = WC if hc_ == ni else legs['AP0'][k, 1, :nr, :hc_]
                        _tl = _thin_level(nr, ni - kin, hc_) if (THIN_ENG and n == 1024 and smm is not None and 'kwin' in THIN_SITES) else 0
                        if _tl > 0:
                            # thin_patch kwin: the rank-nd repair product through the pooled Z core (level from the exact cost model)
                            _tko = _win(pool.get('thin_o', (n, n)), nr, hc_)
                            _thin_mm(smm, W[:, kin:], _pview(legs['AP1'], (k, 1, slice(kin, ni, None), slice(None, hc_, None))), _tko, _tl)
                            fnp.add(WCw, _tko, out=WCw)
                        else:
                            fnp.add(WCw, fnp.matmul(W[:, kin:], legs['AP1'][k, 1, kin:ni, :hc_]), out=WCw)
                    if hc_ < ni:
                        # SDT_CSPLIT (exact): the covariance columns beyond the batch width (nearly-dead neurons of the previous layer)
                        _tl = _thin_level(nr, ni, ni - hc_) if (THIN_ENG and n == 1024 and smm is not None and 'csplit' in THIN_SITES) else 0
                        if _tl > 0:
                            # thin_patch csplit: the covariance tail columns through the pooled Z core (written into the same slot window)
                            _thin_mm(smm, W, _pview(legs['AP1'], (k, 1, slice(None, ni, None), slice(hc_, ni, None))), _pview(legs['AP0'], (k, 1, slice(None, nr, None), slice(hc_, ni, None))), _tl)
                        else:
                            fnp.matmul(W, legs['AP1'][k, 1, :ni, hc_:ni], out=legs['AP0'][k, 1, :nr, hc_:ni])
                if li in SDT_PROBE:
                    SDT_PROBES[li] = dict(wc=fnp.copy(WC), idx_in=idx_in, idx_cur=idx_cur, nr=nr, ni=ni, kin=kin, hc=hc_young, w=fnp.copy(W), cin=fnp.copy(legs['AP1'][k, 1, :ni, :ni]) if A_st is not None else None)
                cp_now = False
                if trim:
                    var = fnp.maximum(fnp.sum(_public_out(fnp.multiply, WC, W, out=NW('t1', nr, ni)), axis=1), 1e-10)
                    if lot and li >= PL_LMIN:
                        with _ns('lot'):
                            lot_Q, lot_G = self._lot_basis(None, WC, w32g, nr, ni, pool, sk_src)
                elif merge_cp:
                    bb = get_bufs(nr)
                    fnp.copyto(_pview(bb['lap'], (k + 1, 0, slice(None, None, None), slice(None, ni, None))), WC)
                    fnp.copyto(_win(legs['AP0'][k + 1, 0], nr, ni), W)
                    if ni < n:
                        fnp.copyto(_pview(bb['lap'], (k + 1, 0, slice(None, None, None), slice(ni, None, None))), 0.0)
                        fnp.copyto(_pview(legs['AP0'], (k + 1, 0, slice(None, nr, None), slice(ni, None, None))), 0.0)
                    bb['cp'] = True
                    bb['cp_ni'] = ni
                    if d21_gate_on:
                        var = fnp.maximum(fnp.sum(_public_out(fnp.multiply, WC, W, out=NW('t1', nr, ni)), axis=1), 1e-10)
                    cp_now = True
                else:
                    C_pre = self._sym_product(WC, w32g, nr, ni, NN('cpre'), min(CPRE_LEV, s_lev), sym4=AU_SYM4)
                fnp.copyto(_win(legs['AP0'][k, 1], nr, ni), W)
                if nd_in > 0:
                    # SDT: the nearly-dead hubs of the newborn carry no direct-weight (P) columns
                    fnp.copyto(legs['AP0'][k, 1, :nr, kin:ni], 0.0)
                if hm is not None and (HUB_PART & 32):
                    fnp.multiply(_win(legs['AP0'][k, 1], nr, ni), _pview(hm, (None, slice(None, None, None))), out=_win(legs['AP0'][k, 1], nr, ni))
                A_st = _win(_pview(legs['AP0'], (slice(None, k + 1, None), 0)), nr, n)
                P_st = _win(_pview(legs['AP0'], (slice(None, k + 1, None), 1)), nr, n)
                zb = _win(pool.get(('z', z_side), (L - 1, n, r + 2)), nr, r + 2)
                fnp.matmul(W, Rr, out=zb[k])
                Z_st = _pview(zb, slice(None, k + 1, None))
                L_st = _pview(pool.get('l', (L - 1, n, r + 2)), slice(None, k + 1, None))
                if Ff is not None and k >= kz:
                    zfb = _win(pool.get(('zf', z_side), (L - 1, n, 2 * rfb)), nr, 2 * rfb)
                    fnp.matmul(W, Ff, out=zfb[k])
                    Zf_st = _pview(zfb, slice(None, k + 1, None))
                s_list.append(hubv(s_b))
                e_list.append(hubv(e_b))
                newborn = None
            if trim:
                C_off = None
            elif not cp_now:
                if lot and li >= PL_LMIN and (A_st is not None):
                    with _ns('lot'):
                        lot_Q, lot_G = self._lot_basis(C_pre, None, w32g, nr, ni, pool, sk_src)
                var = fnp.maximum(fnp.diag(C_pre), 1e-10)
                C_off = _zero_diag(C_pre)
            self._hot_p = 0
            au_run_ = None
            au_sig_ = au_alpha_ = None
            if AU_REUSE and not cp_now:
                # AU_REUSE: the exact variance of the layer is final here (mu too): one sqrt and one divide for the whole layer
                au_sig_ = fnp.sqrt(var)
                au_alpha_ = mu / au_sig_
                if LOTD_ON:
                    self._lotd_sa = (var, mu, au_sig_, au_alpha_)   # lotdiet C: reused by _hz_select (identity-checked)
            if t1_pending is not None:
                # OLD_HOT1: tier-1 legs on rows [p:nr] (post-swap slots), rows [0:p] zero
                kb1_, ka1_, fo1_, m1_, ow1_, qs1_ = t1_pending
                t1_pending = None
                run_t1 = int(float(fnp.sum(fnp.cumprod(fnp.greater(au_alpha_ if au_alpha_ is not None else mu / fnp.sqrt(var), OLD_ROW_CP).astype(f32)))))
                if AU_ROWP:
                    au_run_ = run_t1
                p_t1 = max(0, min(run_t1 // OLD_ROW_GRAN * OLD_ROW_GRAN, nr - OLD_ROW_GRAN))
                self._hot_p = p_t1
                _ap1h = _pview(legs4['AP0'], slice(2 * kb1_, 2 * ka1_, None))
                qcP = pool.get(('qc', qs1_), (n, r_old))
                _aph = _ap1h
                pl1_ = _au_t1plan(list(_pview(hub_widths, slice(kb1_, ka1_, None))), AU_T1PK_EXT and FIJA2 and kb1_ >= 1 and fo1_ >= 1 and bool(t2_pending) and not smm.first) if AU_T1PK and SDT_OLDW and d21_gate_on and (lot_Q is not None) and (not trim) else None
                if pl1_ is not None:
                    # AU_T1PK: rebuild (and, in _d21_demand, the tier-1 hub) on wb; 'pair' packs the factors in place (re-zeroed in _dslices
                    # after the demand), 'ext' uses the free factor slot 2 fo - 1 and the free tier-2 leg slot 2 kb - 1 (written by the tier-2
                    # rebuild below) as one extra slot in front
                    ow1o_ = ow1_
                    ow1_ = pl1_[1]
                    _fs1 = _pview(fap4, slice(2 * fo1_, 2 * (fo1_ + m1_), None))
                    if pl1_[0] == 'pair':
                        for js_, a_, b_, jt_, c_, d_ in pl1_[2]:
                            fnp.copyto(_au_pair(_fs1, jt_, slice(c_, d_, None)), _au_pair(_fs1, js_, slice(a_, b_, None)))
                        self._au_t1pk = pl1_ + (_fs1,)
                        _fa1 = _win(_fs1, r_old, ow1_)
                    else:
                        _fx1 = _pview(fap4, slice(2 * fo1_ - 1, 2 * (fo1_ + m1_), None))
                        _x0 = _pview(_fx1, (0, 0))
                        used_ = 0
                        for s0_, s1_, w_, off_ in pl1_[2]:
                            fnp.copyto(_au_extc(self, _x0, off_, s1_ - s0_, w_ - ow1_), _au_slots(_fs1, s0_, s1_, slice(None, None, None), slice(ow1_, w_, None)))
                            used_ = off_ + (s1_ - s0_) * (w_ - ow1_)
                        if used_ < ow1_:
                            fnp.copyto(_pview(_x0, (slice(None, None, None), slice(used_, ow1_, None))), 0.0)
                        self._au_t1pk = pl1_ + (_fx1,)
                        if (nr - p_t1) % 64 == 0:
                            _fa1 = _win(_fx1, r_old, ow1_)
                            _aph = _pview(legs4['AP0'], slice(2 * kb1_ - 1, 2 * ka1_, None))
                        else:
                            # audit3: rows not a 64-multiple -> the 9-slot rebuild could not follow the parent's padded level: parent rebuild
                            # (8 slots at ow1); only the D21 tier-1 hub uses the extra factor slot
                            ow1_ = ow1o_
                            pl1_ = None
                            _fa1 = _win(_pview(fap4, slice(2 * fo1_, 2 * (fo1_ + m1_), None)), r_old, ow1_)
                else:
                    _fa1 = _win(_pview(fap4, slice(2 * fo1_, 2 * (fo1_ + m1_), None)), r_old, ow1_)
                smm.mn = _TAG['old_min']
                smm.mm(_pview(_pview(qcP, slice(p_t1, nr, None)), (None, None)), _fa1, _pview(_aph, (slice(None, None, None), slice(None, None, None), slice(p_t1, nr, None), slice(None, ow1_, None))), smm.level(nr - p_t1, r_old, ow1_, s_sb), pad=(_pview(_pview(qcP, slice(p_t1, None, None)), (None, None)), None, _pview(_aph, (slice(None, None, None), slice(None, None, None), slice(p_t1, None, None), slice(None, None, None)))))
                smm.mn = STRASSEN_MIN
                if p_t1 > 0:
                    fnp.copyto(_pview(_ap1h, (slice(None, None, None), slice(None, None, None), slice(None, p_t1, None), slice(None, n, None))), 0.0)
                if ow1_ < n:
                    fnp.copyto(_pview(_win(_ap1h, nr, n), (..., slice(ow1_, None, None))), 0.0)
                if pl1_ is not None and pl1_[0] == 'pair':
                    # AU_T1PK: rebuilt overflow columns back to their own pair (rows [p:nr]; rows [:p] and columns [wb:] are zero), packed columns re-zeroed
                    _a1w = _pview(_ap1h, (slice(None, None, None), slice(None, None, None), slice(p_t1, nr, None), slice(None, None, None)))
                    for js_, a_, b_, jt_, c_, d_ in pl1_[2]:
                        fnp.copyto(_au_pair(_a1w, js_, slice(a_, b_, None)), _au_pair(_a1w, jt_, slice(c_, d_, None)))
                        fnp.copyto(_au_pair(_a1w, jt_, slice(c_, d_, None)), 0.0)
                elif pl1_ is not None:
                    # AU_T1PK ext: overflow columns from the extra leg slot back to their slots (rows [p:nr])
                    _o0 = _pview(_aph, (0, 0, slice(p_t1, nr, None)))
                    for s0_, s1_, w_, off_ in pl1_[2]:
                        fnp.copyto(_au_slots(_ap1h, s0_, s1_, slice(p_t1, nr, None), slice(ow1_, w_, None)), _au_extc(self, _o0, off_, s1_ - s0_, w_ - ow1_))
            if t2_pending:
                # OLD_ROWW: tier-2 rebuild on rows [p:nr] (post-swap slots), rows [0:p] zero
                kb_t2 = t2_pending
                t2_pending = 0
                if AU_ROWP and au_run_ is not None:
                    # AU_ROWP: the same leading run as the tier-1 rebuild of this layer (mu, var unchanged)
                    run_t2 = au_run_
                else:
                    run_t2 = int(float(fnp.sum(fnp.cumprod(fnp.greater(mu / fnp.sqrt(var), OLD_ROW_CP).astype(f32)))))
                p_t2 = max(0, min(run_t2 // OLD_ROW_GRAN * OLD_ROW_GRAN, nr - OLD_ROW_GRAN))
                OLD_ROW_LOG.append((li, nr, run_t2, p_t2))
                _apt = _pview(legs4['AP0'], slice(None, 2 * kb_t2, None))
                quP = pool.get('qu', (n, r2))
                smm.mn = _TAG['old_min']
                if p_t2 > 0:
                    smm.mm(_pview(_pview(quP, slice(p_t2, nr, None)), (None, None)), _pview(fap24, slice(None, 2 * kb_t2, None)), _pview(_apt, (slice(None, None, None), slice(None, None, None), slice(p_t2, nr, None), slice(None, n, None))), smm.level(nr - p_t2, r2, n, s_sb), pad=(_pview(_pview(quP, slice(p_t2, None, None)), (None, None)), None, _pview(_apt, (slice(None, None, None), slice(None, None, None), slice(p_t2, None, None), slice(None, None, None)))))
                    fnp.copyto(_pview(_apt, (slice(None, None, None), slice(None, None, None), slice(None, p_t2, None), slice(None, n, None))), 0.0)
                else:
                    smm.mm(_pview(_win(quP, nr, r2), (None, None)), _pview(fap24, slice(None, 2 * kb_t2, None)), _win(_apt, nr, n), smm.level(nr, r2, n, s_sb), pad=(_pview(quP, (None, None)), None, _apt))
                smm.mn = STRASSEN_MIN
            mode = 0 if A_st is None else 1
            _h18f_m = None
            j2ab_ = None
            if mode == 1 and skip_src:
                D3, D21 = (fnp.zeros(nr, dtype=f32), None)
                if regen:
                    WW = _public_out(fnp.multiply, W, W, out=NW('ww', nr, ni))
                    dG = WW @ (g_prev - var_prev * lam_prev) + var * lam_prev
                    if H18 and (H18F or H18F_FEED or H18F_K22X):
                        mrow_s_ = fnp.sum(WW, axis=1)
                        _h18f_m = fnp.add(mrow_s_, sqd_in if idx_cur is None else fnp.take(sqd_in, idx_cur)) if (H18F and (sqd_in is not None)) else mrow_s_
                        g4row = dG * (_h18f_m if H18F else mrow_s_)
                    else:
                        g4row = dG * (fnp.sum(WW, axis=1) if H18 else METRIC_C)
                    wk4m = wk431 = None
                elif riders:
                    g4row = W * W @ K4_vec * 0.5 * float(st['wk4_c4'] * metric2)
                    wk4m = None
                else:
                    g4row = ones_n * (K4_sigma * float(st['wk4_c4'] * metric2))
                    wk4m = None
            elif mode == 1:
                bufs = get_bufs(nr)
                D3, D21 = self._dslices(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list, c1_list, c2_list, y_list, nr, bufs, r, rfb, Zf_st, R1T_st, R2T_st, need_d21=not trim, kz=kz, ka=ka, Qc=Qc, kb=kb, U2=U, apb=legs['AP0'], apb4=_win(legs4['AP0'], nr, n), sb1=_pview(fap4, slice(2 * fa_off, 2 * (fa_off + ka - kb), None)) if ka > kb else None, sb2=_pview(fap24, slice(None, 2 * kb, None)) if kb > 0 else None, s_sb=s_sb, gate_Q=(sk_src if merge_cp else lot_Q) if d21_gate_on else None, gate_sk=sk_src, gate_alpha=(au_alpha_ if au_alpha_ is not None else mu / fnp.sqrt(var)) if d21_gate_on else None, gate_li=li)
                if SD_ROW > 0 and (not last):
                    rm_ = fnp.greater(mu + fnp.sqrt(var) * SD_ROW, 0.0).astype(f32)
                    D3 = D3 * rm_
                    if D21 is not None:
                        D21 = D21 * _pview(rm_, (slice(None, None, None), None))
                if SD_COL > 0 and D21 is not None:
                    D21 = D21 * _pview(fnp.greater(mu + fnp.sqrt(var) * SD_COL, 0.0).astype(f32), (None, slice(None, None, None)))
                if cp_now:
                    C_pre = bufs['hubg'][1, 0]
                    bufs['cp'] = False
                    if lot and li >= PL_LMIN:
                        with _ns('lot'):
                            lot_Q, lot_G = self._lot_basis(C_pre, None, w32g, nr, ni, pool, sk_src)
                    var = fnp.maximum(fnp.diag(C_pre), 1e-10)
                    C_off = _zero_diag(C_pre)
                if regen:
                    WW = _public_out(fnp.multiply, W, W, out=NW('ww', nr, ni))
                    if BETA != 0.0:
                        t_g = WW @ g_prev
                        if sqd_in is not None:
                            t_g = t_g + (sqd_in if idx_cur is None else fnp.take(sqd_in, idx_cur)) * gI_prev
                        t_v = var - WW @ var_prev
                        dG0 = t_g + t_v * lam_prev
                        if dead_w is None:
                            rr = fnp.mean(dG0) / fnp.mean(var)
                        else:
                            wdT, var_d = dead_w
                            WWd = fnp.multiply(wdT, wdT)
                            dG0d = WWd @ g_prev + (var_d - WWd @ var_prev) * lam_prev
                            if sqd_in is not None:
                                dG0d = dG0d + fnp.take(sqd_in, _pview(ordr, slice(None, kl, None))) * gI_prev
                            rr = (fnp.sum(dG0) + fnp.sum(dG0d)) / n / ((fnp.sum(var) + fnp.sum(var_d)) / n)
                        ref = float(REF_R[min(li - 1, len(REF_R) - 1)])
                        cl = fnp.clip(rr / ref, 0.5, 2.0)
                        lam_prev = lam_prev * (cl if ta_wick and BETA == 1.0 else fnp.power(cl, BETA))
                        dG = t_g + t_v * lam_prev
                    else:
                        dG = WW @ (g_prev - var_prev * lam_prev) + var * lam_prev
                        if sqd_in is not None:
                            dG = dG + (sqd_in if idx_cur is None else fnp.take(sqd_in, idx_cur)) * gI_prev
                    mrow_ = fnp.sum(WW, axis=1) if (H18 or H18_K22 or H18_K31 or H18F_FEED or H18F_K22X) else None
                    if (H18F or H18F_FEED or H18F_K22X) and (mrow_ is not None):
                        # h18f: full row metric sum_all W_ai^2 = live-row sum + dropped inputs' sum (sqd_in, the AS_GCONST vector)
                        _h18f_m = fnp.add(mrow_, sqd_in if idx_cur is None else fnp.take(sqd_in, idx_cur)) if (H18F and (sqd_in is not None)) else mrow_
                    g4row = dG * (((_h18f_m if (H18F and _h18f_m is not None) else mrow_)) if H18 else METRIC_C)
                    if K22_ON and BETA != 0.0 and (li >= K22_L0) and (K22 is not None):
                        u1_ = K22 @ (K22 @ k22row)
                        Ku_ = K22 @ u1_
                        c22_ = 3.0 * float(fnp.dot(u1_, Ku_)) / (float(fnp.dot(u1_, u1_)) ** 2 + 1e-30)
                        Wv_ = WW @ u1_
                        WW2_ = _public_out(fnp.multiply, WW, WW, out=NN('ww2'))
                        g4m_k22 = t_v * lam_prev * METRIC_C + WW2_ @ K4v + c22_ * Wv_ * Wv_
                    if H18F_K22X and (_h18f_m is not None):
                        # Sym(G x M) on the (a, a, b, b) slice: (G_aa M_bb + G_bb M_aa) / 6 with the per-row metric (cross pairing)
                        g22c = _pview(dG * (1.0 / 6.0), (slice(None, None, None), None))
                        x22_ = fnp.multiply(g22c, _pview(_h18f_m, (None, slice(None, None, None))))
                        wk4m = _zero_diag(_public_out(fnp.add, x22_, x22_.T, out=NN('wk4m')))
                    else:
                        g22c = _pview(dG * ((mrow_ / 6.0) if H18_K22 else (METRIC_C / 6.0)), (slice(None, None, None), None))
                        wk4m = _zero_diag(_public_out(fnp.add, g22c, g22c.T, out=NN('wk4m')))
                    _resp_here = RESP_ON and regen and (mode == 1) and (not trim) and (self._resp_data is not None)
                    if J2_W431 and NLK_PK and ta_nl and getattr(self, '_j2w', None) is not None and (mode == 1) and (not trim) and (not NO_WK431) and K31TWO and (D21 is not None) and 1 <= li <= len(K31TWO_B1) and (K31TWO == 2 or not _resp_here) and (not H18_K31):
                        # J2_W431: wk431 = C_off D_a + D21^T D_b enters the merged program as c_off / d21T terms (column factors a, b)
                        wk431 = None
                        j2ab_ = (var * (K31TWO_B1[li - 1] * K31TWO_B1S), (D3 / var) * K31TWO_B2)
                    else:
                        wk431 = None if trim else _public_out(fnp.multiply, C_off, 0.5 * METRIC_C * lam_prev, out=NN('wk431'))
                    if K31TWO and wk431 is not None and D21 is not None and 1 <= li <= len(K31TWO_B1) and (K31TWO == 2 or not _resp_here):
                        fnp.multiply(C_off, _pview(var * (K31TWO_B1[li - 1] * K31TWO_B1S), (None, slice(None, None, None))), out=wk431)
                        fnp.add(wk431, fnp.multiply(D21.T, _pview((D3 / var) * K31TWO_B2, (None, slice(None, None, None)))), out=wk431)
                        wk431 = _zero_diag(wk431)
                    if H18_K31 and wk431 is not None:
                        sq_ = fnp.sqrt(mrow_ / METRIC_C)
                        fnp.multiply(wk431, _pview(sq_, (slice(None, None, None), None)) * _pview(sq_, (None, slice(None, None, None))), out=wk431)
                    if _os.environ.get('V17_DEBUG', '0') == '1':
                        DEBUG.append(dict(layer=li, dG=dG, g_prev=g_prev, var_prev=var_prev, lam=lam_prev, var=var, D3=D3, D21=D21))
                elif riders:
                    gv = W * W @ K4_vec * 0.5
                    g4row = gv * float(st['wk4_c4'] * metric2)
                    g22 = gv * float(0.5 * st['wk4_c22'] * metric2)
                    g22c = _pview(g22, (slice(None, None, None), None))
                    wk4m = _zero_diag(_public_out(fnp.add, g22c, g22c.T, out=NN('wk4m')))
                else:
                    g4v = K4_sigma * float(st['wk4_c4'] * metric2)
                    g22v = K4_sigma * float(st['wk4_c22'] * metric2)
                    g4row = ones_n * g4v
                    wk4m = _zero_diag(ones2 * g22v)
            else:
                D3 = D21 = g4row = wk4m = None
            if lot and mode == 1 and (not skip_src) and (lot_Q is not None):
                with _ns('lot'):
                    lot_g4m = self._lot_g4(bufs, A_st, P_st, A_st.shape[0], n, lot_slab, lot_Q, lot_G, g4row if g4m_k22 is None else g4m_k22, li, D3=D3, var=var, mu=mu, old=(ka, kb, FAo, FPo, FA2, FP2, Qc, QU if kb > 0 else None) if ka > 0 else None)
            if g4m_k22 is not None and K22_DEBUG:
                K22_LOG.append(dict(li=li, ratio=float(fnp.mean(g4m_k22)) / float(fnp.mean(g4row))))
            if lot_g4m is None and g4m_k22 is not None:
                lot_g4m = g4m_k22
            if RESP_ON and regen and (mode == 1) and (not trim) and (self._resp_data is not None):
                with _ns('response31'):
                    if K31TWO != 2:
                        wk431 = self._response31(C_off, var, lot_Q, self._resp_data)
                    g4row = lot_g4m
            sigma = au_sig_ if au_sig_ is not None else fnp.sqrt(var)
            alpha = au_alpha_ if au_alpha_ is not None else mu / sigma
            if li in SDT_PROBE and li in SDT_PROBES:
                SDT_PROBES[li].update(var=fnp.copy(var), mu=fnp.copy(mu), cpre=None if C_pre is None else fnp.copy(C_pre), d3=None if D3 is None else fnp.copy(D3), d21=None if D21 is None else fnp.copy(D21))
            phi = flops.stats.norm.pdf(alpha).astype(f32)
            Phi = flops.stats.norm.cdf(alpha).astype(f32)
            hz_Phi.append(Phi)
            phic = _pview(phi, (slice(None, None, None), None))
            Phic = _pview(Phi, (slice(None, None, None), None))
            if ta_wick:
                APOW = fnp.power(_pview(alpha, (slice(None, None, None), None)), _pview(PW6, (None, slice(None, None, None))))
                SB = fnp.power(_pview(sigma, (slice(None, None, None), None)), _pview(EX11, (None, slice(None, None, None))))
                P12 = APOW @ C12
                W_all = (_pview(P12, (slice(None, None, None), slice(None, len(WICK_PAIRS), None))) * phic + _pview(P12, (slice(None, None, None), slice(len(WICK_PAIRS), None, None))) * Phic) * (SB @ SELC)
            else:
                a2 = alpha * alpha
                a3 = a2 * alpha
                a4 = a3 * alpha
                a5 = a4 * alpha
                APOW = fnp.stack([ones_n, alpha, a2, a3, a4, a5], axis=1)
                inv = 1.0 / sigma
                i2 = inv * inv
                i3 = i2 * inv
                i4 = i3 * inv
                i5 = i4 * inv
                i6 = i5 * inv
                s2 = sigma * sigma
                s3 = s2 * sigma
                s4 = s3 * sigma
                SB = fnp.stack([i6, i5, i4, i3, i2, inv, ones_n, sigma, s2, s3, s4], axis=1)
                W_all = (APOW @ C1 * phic + APOW @ C2 * Phic) * (SB @ SELC)
            WT = W_all.T
            pmode = mode if mode == 0 or (regen and (not NO_WK431)) else 2
            prog = self._progs[pmode]
            fpm = FP[pmode]
            b2 = {'ones2': ones2, 'c_off': C_off}
            b1 = {'ones1': ones_n}
            if mode == 1:
                if not trim:
                    if NLK_PK and ta_nl:
                        # NLK_PK: d3 / g4 enter the merged program as row / column weights (no rank-1 n x n matrices)
                        b2.update({'d21': D21, 'd21T': D21.T, 'wk4m': wk4m})
                    else:
                        d3col = _public_out(fnp.multiply, _pview(D3, (slice(None, None, None), None)), ones_row, out=NN('d3col'))
                        g4col = _public_out(fnp.multiply, _pview(g4row, (slice(None, None, None), None)), ones_row, out=NN('g4col'))
                        b2.update({'d21': D21, 'd21T': D21.T, 'wk4m': wk4m, 'd3col': d3col, 'd3row': d3col.T, 'g4col': g4col, 'g4row': g4col.T})
                    if regen and (not NO_WK431):
                        b2['wk431'] = wk431
                        if EF_K31T21:
                            b2['wk431T'] = None if wk431 is None else wk431.T
                b1.update({'d3': D3, 'g4': g4row})
            if self._hz_act and self._hz['rid_any'] and li in self._hz_rid and int(self._hz_rid[li]) == nr:
                if b2.get('d21') is not None:
                    with _ns('hz6'):
                        _K = b2.get('wk431'); _D = b2.get('d21'); _C = b2.get('c_off'); _M = b2.get('wk4m')
                        _w1 = _pview(W_all, (slice(None, None, None), self._i11)); _w2 = _pview(W_all, (slice(None, None, None), self._i21)); _w3 = _pview(W_all, (slice(None, None, None), self._i31))
                        _dv = D3 / var if D3 is not None else None
                        if LOTD_ON:
                            # lotdiet B: the last elementwise op of each rider writes its slab row (was temporary + copyto)
                            fnp.divide(fnp.matmul(_D, D3), var, out=self._hz_slab('h6_d21_d3', li, nr))
                            fnp.divide(fnp.einsum('ij,ij->i', _D, _D), var, out=self._hz_slab('h6_d21_e', li, nr))
                            fnp.multiply(fnp.sum(_D, axis=1), (au_sig_ if au_sig_ is not None else fnp.sqrt(var)), out=self._hz_slab('h6_d21_s', li, nr))
                            fnp.multiply(fnp.matmul(_D, _w1), (au_sig_ if au_sig_ is not None else fnp.sqrt(var)), out=self._hz_slab('h6_d21_w1', li, nr))
                            fnp.subtract(fnp.matmul(_dv, _D), fnp.diag(_D) * _dv, out=self._hz_slab('h6_k31b_s', li, nr))
                        else:
                            _x = fnp.matmul(_D, D3)
                            fnp.copyto(self._hz_slab('h6_d21_d3', li, nr), _x / var)
                            _x = fnp.einsum('ij,ij->i', _D, _D)
                            fnp.copyto(self._hz_slab('h6_d21_e', li, nr), _x / var)
                            _x = fnp.sum(_D, axis=1)
                            fnp.copyto(self._hz_slab('h6_d21_s', li, nr), _x * (au_sig_ if au_sig_ is not None else fnp.sqrt(var)))
                            _x = fnp.matmul(_D, _w1)
                            fnp.copyto(self._hz_slab('h6_d21_w1', li, nr), _x * (au_sig_ if au_sig_ is not None else fnp.sqrt(var)))
                            _x = fnp.matmul(_dv, _D) - fnp.diag(_D) * _dv
                            fnp.copyto(self._hz_slab('h6_k31b_s', li, nr), _x)
                else:
                    fnp.copyto(self._hz_slab('h6_d21_w1', li, nr), 0.0)
                    fnp.copyto(self._hz_slab('h6_d21_e', li, nr), 0.0)
                    fnp.copyto(self._hz_slab('h6_d21_s', li, nr), 0.0)
                    fnp.copyto(self._hz_slab('h6_d21_d3', li, nr), 0.0)
                    fnp.copyto(self._hz_slab('h6_k31b_s', li, nr), 0.0)
            ab_cache = {}

            def ab(pair, tbl):
                if pair in ab_cache:
                    return ab_cache[pair]
                a, b = pair
                if a in ('ones2', 'ones1'):
                    val = tbl[b]
                elif b in ('ones2', 'ones1'):
                    val = tbl[a]
                else:
                    val = tbl[a] * tbl[b]
                ab_cache[pair] = val
                return val
            if not trim and NLK_PK and ta_nl:
                # NLK_PK: pair products once, one row-weighted write per merged entry, one batched column weighting, group sums
                j2on_ = j2ab_ is not None and pmode == 1
                nk_ = self._j2w if j2on_ else self._nlk[pmode]
                E_ = nk_['E']
                if j2on_:
                    # J2_W431: factors 3, 4 = the K31TWO column weights a, b
                    fv_ = (None, D3, g4row, j2ab_[0], j2ab_[1])
                    xt_ = [_pview(WT, slice(lo_, hi_, None)) if f_ == 0 else fnp.multiply(_pview(WT, slice(lo_, hi_, None)), _pview(fv_[f_], (None, slice(None, None, None)))) for f_, lo_, hi_ in nk_['rows']]
                else:
                    xt_ = [_pview(WT, slice(lo_, hi_, None)) if f_ == 0 else fnp.multiply(_pview(WT, slice(lo_, hi_, None)), _pview(D3 if f_ == 1 else g4row, (None, slice(None, None, None)))) for f_, lo_, hi_ in nk_['rows']]
                Xt_ = xt_[0] if len(xt_) == 1 else fnp.concatenate(xt_, axis=0)
                if MARG2_ON and self._nfull == 1024 and len(xt_) > 1:
                    # marg2_patch MARG2 (t/18249 sec. 3.5, re-derived): the k = 2 Wick coefficient keeps its marginal Edgeworth correction
                    # w(5,p) kappa3_ii / 6 + w(6,p) kappa4_ii / 24 only where alpha_i <= -MARG2_C, plain Gaussian w(2,p) elsewhere; the masked Xt
                    # rows feed U_ (row side) and V_ (column side) of Q11, Q21 and Q22 together
                    m2s_ = _marg2_slots(nk_['rows'])
                    if m2s_:
                        m2m_ = _pview(fnp.less_equal(alpha, -MARG2_C).astype(f32), (None, slice(None, None, None)))
                        if MARG2_DEBUG:
                            import json as _m2j, sys as _m2s
                            _m2a = sum(float(fnp.sum(fnp.multiply(_pview(Xt_, slice(s0_, s1_, None)), _pview(Xt_, slice(s0_, s1_, None))))) for s0_, s1_ in m2s_)
                        for s0_, s1_ in m2s_:
                            fnp.multiply(_pview(Xt_, slice(s0_, s1_, None)), m2m_, out=_pview(Xt_, slice(s0_, s1_, None)))
                        if MARG2_DEBUG:
                            _m2b = sum(float(fnp.sum(fnp.multiply(_pview(Xt_, slice(s0_, s1_, None)), _pview(Xt_, slice(s0_, s1_, None))))) for s0_, s1_ in m2s_)
                            _m2s.stderr.write('MARG2 ' + _m2j.dumps(dict(li=int(li), nr=int(nr), pmode=int(pmode), slots=[list(x) for x in m2s_],
                                                                         tail=int(float(fnp.sum(m2m_))), removed=1.0 - (_m2b / _m2a if _m2a > 0 else 1.0))) + '\n')
                if MARG1_ON and self._nfull == 1024 and len(xt_) > 1:
                    # marg1_patch MARG1: the k = 1 Wick coefficient keeps its marginal Edgeworth correction w(4,p) kappa3_ii / 6 +
                    # w(5,p) kappa4_ii / 24 only where alpha_i <= -MARG1_C (Q11 / Q21 / Q22 together through the shared Xt rows)
                    m1s_ = _marg1_slots(nk_['rows'])
                    if m1s_:
                        m1m_ = _pview(fnp.less_equal(alpha, -MARG1_C).astype(f32), (None, slice(None, None, None)))
                        if MARG1_DEBUG:
                            import json as _m1j, sys as _m1s
                            _m1a = sum(float(fnp.sum(fnp.multiply(_pview(Xt_, slice(s0_, s1_, None)), _pview(Xt_, slice(s0_, s1_, None))))) for s0_, s1_ in m1s_)
                        for s0_, s1_ in m1s_:
                            fnp.multiply(_pview(Xt_, slice(s0_, s1_, None)), m1m_, out=_pview(Xt_, slice(s0_, s1_, None)))
                        if MARG1_DEBUG:
                            _m1b = sum(float(fnp.sum(fnp.multiply(_pview(Xt_, slice(s0_, s1_, None)), _pview(Xt_, slice(s0_, s1_, None))))) for s0_, s1_ in m1s_)
                            _m1s.stderr.write('MARG1 ' + _m1j.dumps(dict(li=int(li), nr=int(nr), pmode=int(pmode), slots=[list(x) for x in m1s_],
                                                                         tail=int(float(fnp.sum(m1m_))), removed=1.0 - (_m1b / _m1a if _m1a > 0 else 1.0))) + '\n')
                U_ = (fpm['J2W_SL'] if j2on_ else fpm['NLK_SL']) @ Xt_
                V_ = (fpm['J2W_SR'] if j2on_ else fpm['NLK_SR']) @ Xt_
                for p_, (a_, b_) in enumerate(nk_['pairs']):
                    fnp.multiply(b2[a_], b2[b_], out=_pview(abbuf, E_ + p_) if RRES_ON else abbuf[E_ + p_])
                opd_runs_ = _opd_runs(nk_['src']) if (OPD_NLK and n == 1024) else ()
                opd_nlk_ = len(opd_runs_) > 0
                if opd_nlk_:
                    # opd_patch OPD_NLK: the same entry writes as runs over strided views (named source broadcast over the run, or pair
                    # products advancing with e); every output element is the same single product
                    for ok_, on_, oe0_, oe1_, oes_, op0_, op1_, ops_ in opd_runs_:
                        uu_ = _pview(U_, (slice(oe0_, oe1_, oes_), slice(None, None, None), None))
                        fnp.multiply(b2[on_] if ok_ == 0 else _pview(abbuf, slice(E_ + op0_, E_ + op1_, ops_)), uu_, out=_pview(abbuf, slice(oe0_, oe1_, oes_)))
                for e_, s_ in (() if opd_nlk_ else enumerate(nk_['src'])):
                    fnp.multiply(b2[s_] if isinstance(s_, str) else abbuf[E_ + s_], _pview(U_, (e_, slice(None, None, None), None)), out=abbuf[e_])
                ABstack = _pview(abbuf, slice(None, E_, None))
                fnp.multiply(ABstack, _pview(V_, (slice(None, None, None), None, slice(None, None, None))), out=ABstack)
                PK2 = _win(pool.get('pk2', (3, n, n)), nr, nr)
                for g_, t0_, t1_ in nk_['groups']:
                    fnp.sum(_pview(ABstack, slice(t0_, t1_, None)), axis=0, out=PK2[g_])
            elif not trim:
                ab2 = prog['AB2']
                for t_, (a_, b_) in enumerate(ab2):
                    if a_ in ('ones2', 'ones1'):
                        fnp.copyto(abbuf[t_], b2[b_])
                    elif b_ in ('ones2', 'ones1'):
                        fnp.copyto(abbuf[t_], b2[a_])
                    else:
                        fnp.multiply(b2[a_], b2[b_], out=abbuf[t_])
                ABstack = _pview(abbuf, slice(None, len(ab2), None))
                WL2 = fpm['SL2'] @ WT
                WR2 = fpm['SR2'] @ WT
                if ta_nl:
                    fnp.multiply(ABstack, _pview(WL2, (slice(None, None, None), slice(None, None, None), None)), out=ABstack)
                    fnp.multiply(ABstack, _pview(WR2, (slice(None, None, None), None, slice(None, None, None))), out=ABstack)
                    PK2 = _win(pool.get('pk2', (3, n, n)), nr, nr)
                    for g_, t0_, t1_ in self._gb2[pmode]:
                        fnp.sum(_pview(ABstack, slice(t0_, t1_, None)), axis=0, out=PK2[g_])
                else:
                    PK2 = fnp.einsum('tij,ti,tj,tg->gij', ABstack, WL2, WR2, fpm['IND2'])
            if lot_g4m is not None:
                b1m = dict(b1)
                b1m['g4'] = lot_g4m
                rows_ = []
                for t_, p_ in enumerate(prog['B1']):
                    if t_ < prog['N10'] and 'g4' in p_:
                        a_, b_ = p_
                        rows_.append(b1m[b_] if a_ in ('ones2', 'ones1') else b1m[a_] if b_ in ('ones2', 'ones1') else b1m[a_] * b1m[b_])
                    else:
                        rows_.append(ab(p_, b1))
                B1stack = fnp.stack(rows_, axis=0)
            else:
                B1stack = fnp.stack([ab(p, b1) for p in prog['B1']], axis=0)
            WL1 = fpm['SL1'] @ WT
            PK1 = fnp.einsum('ti,ti,tg->gi', B1stack, WL1, fpm['IND1'])
            if not trim:
                pk11 = _public_out(fnp.add, PK2[0], PK2[0].T, out=NN('pk11'))
                pk11 = _zero_diag(_public_out(fnp.multiply, pk11, 0.5, out=pk11))
                pk21 = _zero_diag(PK2[1])
                if NLK_K22 and (not K22_ON):
                    # NLK_K22: pk22 = sym(PK2[2]) with a zero diagonal feeds only K22's row sums:
                    # rowsum(pk22) = (rowsum + colsum)(PK2[2]) / 2 - diag(PK2[2])
                    pk22 = None
                    rs22_ = fnp.subtract(fnp.multiply(fnp.add(fnp.sum(PK2[2], axis=1), fnp.sum(PK2[2], axis=0)), 0.5), fnp.diag(PK2[2]))
                else:
                    pk22 = _public_out(fnp.add, PK2[2], PK2[2].T, out=NN('pk22'))
                    pk22 = _zero_diag(_public_out(fnp.multiply, pk22, 0.5, out=pk22))
            pk1v, pk2v, pk3v, pk4v = (PK1[0], PK1[1], PK1[2], PK1[3])
            if riders and (not NO_CORR):
                if mode == 1:
                    D3f = D3
                    D21n = _public_out(fnp.abs, D21, out=T1) @ ones_n if D21 is not None else fnp.zeros(nr, dtype=f32)
                    k4f = ones_n * K4_sigma
                else:
                    D3f = fnp.zeros(nr, dtype=f32)
                    D21n = fnp.zeros(nr, dtype=f32)
                    k4f = fnp.zeros(nr, dtype=f32)
                feats = fnp.stack([ones_n, mu, var, sigma, alpha, fnp.abs(alpha), phi, Phi, pk1v, D3f, D21n, k4f, sigma * phi], axis=1)
                delta = feats @ beta_rows[li]
            else:
                delta = None
            if last:
                if FIN_EDGE_ON and (fin_dead is not None) and (dead_m_cur is not None) and mode == 1 and (D3 is not None) and self._nfull == 1024:
                    # fin_patch FIN_EDGE: dropped final rows -> 2-term Edgeworth mean with the radial proxy fitted on the kept rows
                    dead_m_cur = dead_m_cur + self._fin_edge(fin_dead, alpha, sigma, var, D3, lot_g4m if lot_g4m is not None else g4row)
                rows.append(_out(pk1v if delta is None else pk1v + delta))
                break
            if MID_EDGE_ON and (mid_dead is not None) and (dead_m_cur is not None) and mode == 1 and (D3 is not None) and self._nfull == 1024:
                # mid_patch MID_EDGE: dropped rows of this layer -> 2-term Edgeworth mean (radial proxy from the kept rows); dead_m is what
                # the next layer consumes (m_in -> mu_corr), so BOTH names are rebound to the corrected array
                dead_m_cur = dead_m_cur + self._mid_edge(mid_dead, alpha, sigma, var, D3, lot_g4m if lot_g4m is not None else g4row, li)
                dead_m = dead_m_cur
            w1 = _pview(W_all, (slice(None, None, None), self._i11))
            w2 = _pview(W_all, (slice(None, None, None), self._i21))
            n3 = nr
            if SDT_MASK and sdt_ord:
                # SDT tail of this layer: the last nd rows (SDT_GRAN-granular count of alpha < -SDT_C; the order puts them last) keep
                # their mean / variance / covariance but carry no post-activation K3 response (W_all rows zero => w1, w2, w3, e, the
                # feed vectors and the thin feedback factors vanish there) and, below, no hub content
                n3 = _sdt_win(nr, int(float(fnp.sum(fnp.less(alpha, -_SDT_C_L(li))))))
                nd_cur = nr - n3
                if TDP_ON and TDP_LMIN <= li + 1 <= TDP_LMAX and self._nfull == 1024:
                    # tdp_patch: the next layer's w1-scaled transports keep the 64-grid live-order prefix [:kc] (kc <= n3: the n3 / nc
                    # window rule with -TDP_C); rows [kc:n3] = the band (alpha in (-SDT_C, -TDP_C] while the alpha-sorted candidates cover it)
                    self._tdp_kn = min(n3, _sdt_win(nr, int(float(fnp.sum(fnp.less(alpha, -TDP_C))))))
                    if TDP_DEBUG:
                        al_t = [float(x_) for x_ in alpha.tolist()]
                        TDP_LOG.append(('w', li, nr, n3, self._tdp_kn, max(al_t[self._tdp_kn:n3], default=None), min(al_t[self._tdp_kn:n3], default=None), min(al_t[:self._tdp_kn], default=None), sum((a_ <= -TDP_C for a_ in al_t[:self._tdp_kn]))))
                if nd_cur > 0:
                    fnp.copyto(W_all[n3:], 0.0)
            SDT_LOG.append((li, nr, ni, kin, n3, self._sdt_nc if self._sdt_nc >= 0 else nr))
            if SDT_DEBUG:
                al_ = [float(x) for x in alpha.tolist()]
                SDT_LOG[-1] = dict(li=li, nr=nr, ni=ni, kl=kl, sdt_ord=bool(sdt_ord), nd=nd_cur, ndc=nr - self._sdt_nc if self._sdt_nc >= 0 else 0, cnt_c=sum((a_ < -SDT_C for a_ in al_)), cnt_col=sum((a_ < -SDT_COL for a_ in al_)), tail_bad=sum((a_ >= -SDT_C for a_ in _pview(al_, slice(n3, None, None)))), tail_max=max(_pview(al_, slice(n3, None, None)), default=None), head_bad=sum((a_ < -SDT_C for a_ in _pview(al_, slice(None, n3, None)))), head_min=min(_pview(al_, slice(None, n3, None)), default=None), col_tail_max=max(_pview(al_, slice(self._sdt_nc, None, None)), default=None) if self._sdt_nc >= 0 else None, alpha=al_ if _os.environ.get('SDT_DEBUG_ALPHA', '0') == '1' else None)
            if SD_C > 0 or SD_ON_C > 0:
                sdm_ = fnp.greater(alpha, -SD_C).astype(f32) if SD_C > 0 else ones_n
                if SD_ON_C > 0:
                    hubm_ = sdm_ * fnp.less(alpha, SD_ON_C).astype(f32)
                    W_all = fnp.concatenate([_pview(W_all, (slice(None, None, None), slice(None, self._i11 + 1, None))) * _pview(sdm_, (slice(None, None, None), None)), _pview(W_all, (slice(None, None, None), slice(self._i11 + 1, None, None))) * _pview(hubm_, (slice(None, None, None), None))], axis=1) if False else W_all * _pview(sdm_, (slice(None, None, None), None))
                    w1 = _pview(W_all, (slice(None, None, None), self._i11))
                    w2 = _pview(W_all, (slice(None, None, None), self._i21)) * fnp.less(alpha, SD_ON_C).astype(f32)
                else:
                    W_all = W_all * _pview(sdm_, (slice(None, None, None), None))
                    w1 = _pview(W_all, (slice(None, None, None), self._i11))
                    w2 = _pview(W_all, (slice(None, None, None), self._i21))
            if self._hz_act:
                hz_c1.append(w1)
                if HZW:
                    hz_n3.append(n3)
                hz_c2.append(w2)
                m1_ = phi + alpha * Phi
                hz_nu.append(fnp.maximum(var * (Phi * (1.0 - Phi) + m1_ * (alpha - m1_)), 0.0))
            hm = None
            if (HUB_CN > 0 or HUB_CP > 0 or HUB_N > 0) and li >= HUB_L0:
                hm = ones_n * 1.0
                if HUB_CN > 0:
                    hm = hm * fnp.greater(alpha, -HUB_CN).astype(f32)
                if HUB_CP > 0:
                    hm = hm * fnp.less(alpha, HUB_CP).astype(f32)
                if 0 < HUB_N < nr:
                    key_h = fnp.abs(alpha - HUB_A0) if HUB_KEY == 'alpha' else -(w2 * w2 * var * var)
                    thr_h = fnp.sort(key_h)[HUB_N - 1]
                    hm = hm * fnp.less_equal(key_h, thr_h).astype(f32)
                HUB_LOG.append((li, nr, float(fnp.sum(hm))))
            if lot:
                with _ns('lot'):
                    if lot_slab is None:
                        lot_slab = pool.get('lot_slab', (L - 1, 3, n))
                    if nr < n:
                        fnp.copyto(_pview(lot_slab, (li, slice(None, None, None), slice(nr, None, None))), 0.0)
                    fnp.multiply(w1, var, out=_pview(lot_slab, (li, 0, slice(None, nr, None))))
                    fnp.copyto(_pview(lot_slab, (li, 1, slice(None, nr, None))), var)
                    if PL_STAR:
                        fnp.multiply(_pview(W_all, (slice(None, None, None), self._i31)), 1.0 / 3.0, out=_pview(lot_slab, (li, 2, slice(None, nr, None))))
            w1col = _pview(w1, (slice(None, None, None), None))
            w1_prev = w1
            if mode == 1 and ka < A_st.shape[0]:
                kk = A_st.shape[0]
                fnp.multiply(_win(_pview(legs4['AP0'], slice(2 * ka, 2 * kk, None)), nr, n), _pview(w1col, (None, None)), out=_win(_pview(legs4['AP0'], slice(2 * ka, 2 * kk, None)), nr, n))
            if mode == 1:
                D3_w = D3 * w1 ** 3
                D21_w = _public_out(fnp.multiply, w1col * w1col, D21, out=NN('d21w'))
                fnp.multiply(D21_w, _pview(w1, (None, slice(None, None, None))), out=D21_w)
            else:
                D3_w = None
                D21_w = None
            if legs is None:
                pairs = {nm: pool.get_pair(nm, (L - 1, 2, n, n)) for nm in ('AP0', 'AP1')}
                legs = {nm: pairs[nm][0] for nm in pairs}
                legs4 = {nm: pairs[nm][1] for nm in pairs}
            k_b = 0 if A_st is None else A_st.shape[0]
            assert len(hub_widths) == k_b
            hub_widths.append(n3 if SDT_HUBW else nr)
            a_b = _public_out(fnp.multiply, w1col, C_off, out=_win(legs['AP0'][k_b, 0], nr, nr))
            if nd_cur > 0:
                # SDT: no A-leg columns for the nearly-dead hubs (and the padding to the full hub width, zeroed here instead of below)
                fnp.copyto(legs['AP0'][k_b, 0, :nr, n3:], 0.0)
            if hm is not None and (HUB_PART & 1):
                fnp.multiply(a_b, _pview(hm, (None, slice(None, None, None))), out=a_b)
            if (HUB_ACN > 0 or HUB_ACP > 0 or HUB_AN > 0) and li >= HUB_L0:
                hmA = ones_n * 1.0
                if HUB_ACN > 0:
                    hmA = hmA * fnp.greater(alpha, -HUB_ACN).astype(f32)
                if HUB_ACP > 0:
                    hmA = hmA * fnp.less(alpha, HUB_ACP).astype(f32)
                if 0 < HUB_AN < nr:
                    key_h = fnp.abs(alpha - HUB_A0) if HUB_KEY == 'alpha' else -(w2 * w2 * var * var)
                    thr_h = fnp.sort(key_h)[HUB_AN - 1]
                    hmA = hmA * fnp.less_equal(key_h, thr_h).astype(f32)
                HUB_LOG.append((li, nr, float(fnp.sum(hmA))))
                fnp.multiply(a_b, _pview(hmA, (None, slice(None, None, None))), out=a_b)
            k_fb = 0 if A_st is None else A_st.shape[0]
            fb_here = rfb > 0 and mode == 1 and (k_fb >= kz)
            if fb_here:
                w3 = _pview(W_all, (slice(None, None, None), self._i31))
                Omf = _win(pool.get('omf', (n, rfb)), nr, rfb)
                fnp.copyto(Omf, _pview(sk_src, (slice(None, None, None), slice(None, rfb, None))))
                Yf = D21 @ Omf
                Qf, _ = fnp.linalg.qr(Yf)
                Yf = D21 @ (D21.T @ Qf)
                Qf, _ = fnp.linalg.qr(Yf)
                Bf = Qf.T @ D21
                F1_b = _pview(w2, (slice(None, None, None), None)) * Qf
                R1T_b = Bf.T * 1.5
                F2_b = w1col * Bf.T
                R2T_b = Qf * _pview(w3 * 0.5, (slice(None, None, None), None))
                if nd_cur > 0 and (not 0 <= self._sdt_nc <= n3):
                    # SDT: thin feedback right factors of the nearly-dead hubs (already zero when the D21 column tail covers the hub tail)
                    R1T_b = fnp.concatenate([R1T_b[:n3], fnp.zeros((nd_cur, rfb), dtype=f32)], axis=0)
                if hm is not None and (HUB_PART & 2):
                    R1T_b = R1T_b * _pview(hm, (slice(None, None, None), None))
                    R2T_b = R2T_b * _pview(hm, (slice(None, None, None), None))
                if J2_D21F:
                    # J2_D21F: X1 / 3 = a + F1 (R1T / 3)^T; every D21_new term then carries the 1/3 (xd = diag(Xt) / 3 included)
                    Xt_b = _public_out(fnp.matmul, F1_b, (R1T_b * (1.0 / 3.0)).T, out=NN('xtb'))
                    Yt_b = _public_out(fnp.matmul, F2_b, R2T_b.T, out=NN('ytb'))
                    X1_b = _public_out(fnp.add, a_b, Xt_b, out=NN('x1b'))
                else:
                    Xt_b = _public_out(fnp.matmul, F1_b, R1T_b.T, out=NN('xtb'))
                    Yt_b = _public_out(fnp.matmul, F2_b, R2T_b.T, out=NN('ytb'))
                    X1_b = _public_out(fnp.multiply, a_b, 3.0, out=NN('x1b'))
                    fnp.add(X1_b, Xt_b, out=X1_b)
                Y1_b = _public_out(fnp.multiply, a_b, _pview(w2, (None, slice(None, None, None))), out=NN('y1b'))
                fnp.add(Y1_b, Yt_b, out=Y1_b)
                xd = fnp.diag(Xt_b)
                yd = fnp.diag(Yt_b)
                D21_new = _public_out(fnp.multiply, _pview(xd, (slice(None, None, None), None)), Y1_b.T, out=NN('d21new'))
                fnp.multiply(X1_b, Y1_b, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                fnp.multiply(_pview(yd, (slice(None, None, None), None)), X1_b.T, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                if J2_D21F:
                    D3_new = xd * yd * 3.0
                else:
                    fnp.multiply(D21_new, 1.0 / 3.0, out=D21_new)
                    D3_new = xd * yd
            else:
                D21_new = _public_out(fnp.multiply, a_b, a_b, out=NN('d21new'))
                fnp.multiply(D21_new, _pview(w2, (None, slice(None, None, None))), out=D21_new)
                D3_new = None
                if rfb > 0 and k_fb >= kz:
                    F1_b = fnp.zeros((nr, rfb), dtype=f32)
                    F2_b = F1_b
                    R1T_b = F1_b
                    R2T_b = F1_b
            if regen and mode == 1 and (not NO_FEED):
                w1sq = w1 * w1
                y_b = (w2 * (0.25 * _h18f_m)) if (H18F_FEED and (_h18f_m is not None)) else (w2 * (0.25 * METRIC_C))
                dgw = w1sq * dG
                u_b = y_b * dG
                c1_b = w1 * lam_prev
                w1sq_h, dgw_h, c1_h = (w1sq, dgw, c1_b) if (hm is None or not (HUB_PART & 4)) else (w1sq * hm, dgw * hm, c1_b * hm)
                if J2_D21F:
                    # J2_D21F: (dgw (x) y) / 3 with the 1/3 on the vector
                    fnp.multiply(_pview(dgw_h * (1.0 / 3.0), (slice(None, None, None), None)), _pview(y_b, (None, slice(None, None, None))), out=T1)
                else:
                    fnp.multiply(_pview(dgw_h, (slice(None, None, None), None)), _pview(y_b, (None, slice(None, None, None))), out=T1)
                    fnp.multiply(T1, 1.0 / 3.0, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                fnp.multiply(a_b, _pview(c1_h, (None, slice(None, None, None))), out=T1)
                fnp.multiply(_pview(y_b * (2.0 / 3.0), (slice(None, None, None), None)), T1, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                fnp.multiply(_pview(w1sq_h, (slice(None, None, None), None)), _pview(u_b * (1.0 / 3.0), (None, slice(None, None, None))), out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                D3_new = dgw_h * y_b + u_b * w1sq_h if D3_new is None else D3_new + dgw_h * y_b + u_b * w1sq_h
            else:
                y_b = fnp.zeros(nr, dtype=f32)
                u_b = y_b
                c1_b = y_b
                dgw = y_b
                w1sq = y_b
                w1sq_h = dgw_h = c1_h = y_b
            rep21 = D21_new if D21_w is None else _public_out(fnp.add, D21_w, D21_new, out=NN('rep21'))
            rep21 = _zero_diag(rep21)
            pk = {(1,): pk1v, (2,): pk2v, (3,): pk3v, (4,): pk4v, (1, 1): pk11, (2, 1): pk21, (2, 2): pk22}

            def pk_slice(vec):
                nz = tuple(sorted((x for x in vec if x > 0), reverse=True))
                base = pk[nz]
                if len(vec) == 1:
                    return base
                if len(nz) == 2:
                    return base if vec == tuple(sorted(vec, reverse=True)) else base.T
                return _pview(base, (slice(None, None, None), None)) if vec[0] > 0 else _pview(base, (None, slice(None, None, None)))
            K = {}
            opd_pk_ = OPD_PK and n == 1024 and _OPD_PK_OK
            if opd_pk_:
                # opd_patch OPD_PK: PK2K_TABLE's vector parts with pk1^2 / pk1^3 formed once and the coefficient-1 terms unscaled
                # (x * 1.0 == x); same operand order and left-to-right sums as the loop below -> bit-identical (K1 keeps its * 1.0: fresh)
                p11_ = pk1v * pk1v
                p111_ = p11_ * pk1v
                K[1,] = pk1v * 1.0
                K[2,] = p11_ * -1.0 + pk2v
                K[3,] = p111_ * 2.0 + (pk1v * pk2v) * -3.0 + pk3v
                K[4,] = (p111_ * pk1v) * -6.0 + (p11_ * pk2v) * 12.0 + (pk1v * pk3v) * -4.0 + (pk2v * pk2v) * -3.0 + pk4v
            for out_part, entries in (() if opd_pk_ else PK2K_TABLE.items()):
                if len(out_part) == 2:
                    continue
                acc2 = None
                for vpart, coef in entries:
                    prod = None
                    for blk in vpart:
                        fct = pk_slice(blk)
                        prod = fct if prod is None else prod * fct
                    contrib = prod * coef
                    acc2 = contrib if acc2 is None else acc2 + contrib
                K[out_part] = acc2
            pk1c = _pview(pk1v, (slice(None, None, None), None))
            pk1r = _pview(pk1v, (None, slice(None, None, None)))
            K11 = pk11
            if pk22 is None:
                # NLK_K22: K21 = (-2 pk1) o_rows pk11 + pk21 (scaling by 2 is exact: bit-identical); K22 only through its row sums
                # sum_j K22[i, j] = 4 pk1_i (pk11 pk1)_i - 2 (pk21 pk1)_i - 2 pk1_i (1^T pk21)_i - 2 sum_j pk11_ij^2 + rowsum(pk22)_i
                # (pk11, pk21, pk22 have zero diagonals => K22's diagonal is zero and needs no masking)
                K21 = _public_out(fnp.multiply, pk11, _pview(pk1v * -2.0, (slice(None, None, None), None)), out=NN('k21'))
                fnp.add(K21, pk21, out=K21)
                k22r_ = fnp.multiply(pk1v, fnp.matmul(pk11, pk1v)) * 4.0 - fnp.matmul(pk21, pk1v) * 2.0 - pk1v * fnp.sum(pk21, axis=0) * 2.0 - fnp.einsum('ij,ij->i', pk11, pk11) * 2.0 + rs22_
                K21 = _zero_diag(K21)
                K22 = None
            else:
                K21 = _public_out(fnp.multiply, pk1c, pk11, out=NN('k21'))
                fnp.multiply(K21, -2.0, out=K21)
                fnp.add(K21, pk21, out=K21)
                K22 = _public_out(fnp.multiply, pk1r, pk1c, out=NN('k22'))
                fnp.multiply(K22, pk11, out=K22)
                fnp.multiply(K22, 4.0, out=K22)
                fnp.multiply(pk1r, pk21, out=T1)
                fnp.multiply(T1, -2.0, out=T1)
                fnp.add(K22, T1, out=K22)
                fnp.multiply(pk1c, pk21.T, out=T1)
                fnp.multiply(T1, -2.0, out=T1)
                fnp.add(K22, T1, out=K22)
                fnp.multiply(pk11, pk11, out=T1)
                fnp.multiply(T1, -2.0, out=T1)
                fnp.add(K22, T1, out=K22)
                fnp.add(K22, pk22, out=K22)
                K21 = _zero_diag(K21)
                K22 = _zero_diag(K22)
            K1v = K[1,]
            K2v, K3v, K4v = (K[2,], K[3,], K[4,])
            mu = K1v if delta is None else K1v + delta
            if pk22 is None:
                # NLK_K22: K11 = pk11 is exactly symmetric, so (K11 + K11^T) / 2 equals K11 bit for bit
                C = NN('c')
                fnp.copyto(C, K11)
            else:
                C = _public_out(fnp.add, K11, K11.T, out=NN('c'))
                fnp.multiply(C, 0.5, out=C)
            fnp.fill_diagonal(C, K2v)
            if not AS_TAG_OFF:
                C = flops.as_symmetric(C, symmetry=(0, 1))
            fnp.copyto(_win(legs['AP0'][k_b, 1], nr, nr), C)
            S3c = K3v - D3_w if D3_w is not None else K3v
            if D3_new is not None:
                S3c = S3c - D3_new
            S21 = _zero_diag(_public_out(fnp.subtract, K21, rep21, out=NN('s21')))
            e_b = _pview(W_all, (slice(None, None, None), self._i12)) - 2.0 * pk1v * w1
            if nd_cur > 0:
                # SDT: no M-leg diagonal weight for the nearly-dead hubs (e_b is already zero there: W_all rows)
                S3c = fnp.concatenate([S3c[:n3], zeros_full[:nd_cur]])
            if hm is not None and (HUB_PART & 8):
                e_b = e_b * hm
                S3c = S3c * hm
            S_sep = _public_out(fnp.multiply, _pview(e_b, (slice(None, None, None), None)), C_off, out=NN('ssep'))
            S_sep = _zero_diag(_public_out(fnp.multiply, S_sep, _pview(w1, (None, slice(None, None, None))), out=S_sep))
            Rres = _public_out(fnp.subtract, S21, S_sep, out=S21)
            if nd_cur > 0:
                # SDT: no thin-residual rows for the nearly-dead hubs (=> the L leg rows [n3:] are exactly zero)
                fnp.copyto(Rres[n3:], 0.0)
            if hm is not None and (HUB_PART & 16):
                fnp.multiply(Rres, _pview(hm, (slice(None, None, None), None)), out=Rres)
            Om = _win(pool.get('om_r', (n, r)), nr, r)
            fnp.copyto(Om, _pview(sk_src, (slice(None, None, None), slice(None, r, None))))
            j2r_ = J2_RRW and nd_cur > 0 and 0 < n3 < nr and (hm is None or not (HUB_PART & 16))
            if j2r_:
                # J2_RRW: Rres rows [n3:] are exactly zero -> the range finder on Rres[:n3] (plain slices; Q rows [n3:] are zero)
                Rw_ = Rres[:n3]
                Y = _public_out(fnp.matmul, Rw_, Om, out=pool.get('y', (n, r))[:n3, :r])
                Q, _ = fnp.linalg.qr(Y)
                Y2 = _public_out(fnp.matmul, Rw_.T, Q, out=_win(pool.get('y2', (n, r)), nr, r))
                Y = _public_out(fnp.matmul, Rw_, Y2, out=Y)
                Q, _ = fnp.linalg.qr(Y)
                Bm = _public_out(fnp.matmul, Q.T, Rw_, out=_win(pool.get('bm', (r, n)), r, nr))
            else:
                Y = _public_out(fnp.matmul, Rres, Om, out=_win(pool.get('y', (n, r)), nr, r))
                Q, _ = fnp.linalg.qr(Y)
                Y2 = _public_out(fnp.matmul, Rres.T, Q, out=_win(pool.get('y2', (n, r)), nr, r))
                Y = _public_out(fnp.matmul, Rres, Y2, out=Y)
                Q, _ = fnp.linalg.qr(Y)
                Bm = _public_out(fnp.matmul, Q.T, Rres, out=_win(pool.get('bm', (r, n)), r, nr))
            Rr_full = _win(pool.get('rr', (n, r + 2)), nr, r + 2)
            fnp.multiply(Bm.T, 3.0, out=_pview(Rr_full, (slice(None, None, None), slice(None, r, None))))
            fnp.copyto(_pview(Rr_full, (slice(None, None, None), r)), u_b)
            fnp.copyto(_pview(Rr_full, (slice(None, None, None), r + 1)), y_b)
            k_b = 0 if A_st is None else A_st.shape[0]
            lb = pool.get('l', (L - 1, n, r + 2))
            if j2r_:
                fnp.copyto(_pview(lb, (k_b, slice(None, nr, None), slice(None, r, None)))[:n3], Q)
                fnp.copyto(_pview(lb, (k_b, slice(None, nr, None), slice(None, r, None)))[n3:], 0.0)
            else:
                fnp.copyto(_pview(lb, (k_b, slice(None, nr, None), slice(None, r, None))), Q)
            fnp.copyto(_pview(lb, (k_b, slice(None, nr, None), r)), w1sq_h)
            fnp.copyto(_pview(lb, (k_b, slice(None, nr, None), r + 1)), zeros_n)
            Lr_full = None
            Ff_b = fnp.concatenate([F1_b, F2_b], axis=1) if rfb > 0 and k_b >= kz else None
            newborn = (a_b, Rr_full, Lr_full, S3c, e_b, Ff_b)
            if rfb > 0 and k_b >= kz:
                r1b = pool.get('r1t', (L - 1, n, rfb))
                r2b = pool.get('r2t', (L - 1, n, rfb))
                fnp.copyto(_pview(r1b, (k_b, slice(None, nr, None))), R1T_b)
                fnp.copyto(_pview(r2b, (k_b, slice(None, nr, None))), R2T_b)
                R1T_st = _pview(r1b, slice(None, k_b + 1, None))
                R2T_st = _pview(r2b, slice(None, k_b + 1, None))
            if nr < n:
                if nd_cur == 0:
                    fnp.copyto(_pview(legs['AP0'], (k_b, 0, slice(None, nr, None), slice(nr, None, None))), 0.0)
                fnp.copyto(_pview(legs['AP0'], (k_b, 1, slice(None, nr, None), slice(nr, None, None))), 0.0)
                fnp.copyto(_pview(lb, (k_b, slice(nr, None, None))), 0.0)
                if rfb > 0 and k_b >= kz:
                    fnp.copyto(_pview(r1b, (k_b, slice(nr, None, None))), 0.0)
                    fnp.copyto(_pview(r2b, (k_b, slice(nr, None, None))), 0.0)
            w2b_list.append(hubv(w2 if (hm is None or not (HUB_PART & 64)) else w2 * hm))
            dA_list.append(hubv(9.0 + w2 * w2 + 9.0 * e_b * e_b))
            dP_list.append(hubv(1.0 + S3c * S3c))
            c1_list.append(hubv(c1_h))
            c2_list.append(hubv(dgw_h))
            y_list.append(hubv(y_b))
            opd_s_ = None
            if regen and OPD_CSE and n == 1024 and K22 is None:
                # opd_patch OPD_CSE: sum(K4v), sum(k22row) (k22row is k22r_ here) and their cI term once for g_prev / gI_prev / K4_sigma
                k22row = k22r_
                opd_s_ = (fnp.sum(K4v), fnp.sum(k22row))
                opd_gi_ = (opd_s_[0] + opd_s_[1]) * float(st['cI'])
                g_prev = (K4v + k22row) * float(st['cA']) + opd_gi_
                if as_on and AS_GCONST:
                    gI_prev = opd_gi_
                var_prev = K2v
                lam_prev = float(LAM[min(li, len(LAM) - 1)])
            elif regen:
                k22row = K22 @ ones_n if K22 is not None else k22r_
                g_prev = (K4v + k22row) * float(st['cA']) + (fnp.sum(K4v) + fnp.sum(k22row)) * float(st['cI'])
                if as_on and AS_GCONST:
                    gI_prev = (fnp.sum(K4v) + fnp.sum(k22row)) * float(st['cI'])
                var_prev = K2v
                lam_prev = float(LAM[min(li, len(LAM) - 1)])
            if opd_s_ is not None:
                K4_sigma = (opd_s_[0] * float(st['k4_c4']) + opd_s_[1] * float(st['k4_c22'])) * float(st['P2'])
            else:
                K4_sigma = (fnp.sum(K4v) * float(st['k4_c4']) + fnp.sum(K22 if K22 is not None else k22r_) * float(st['k4_c22'])) * float(st['P2'])
            if riders:
                K4_vec = (K4v * float(st['k4_c4']) + (K22 @ ones_n if K22 is not None else k22r_) * float(st['k4_c22'])) * float(n * st['P2'])
            rows.append(_out(mu))
        if smm.first:
            smm.finish_first()
        if HZ_TMAX > 0 and self._hz_act and _rb_time.perf_counter() - self._rb_t0 > HZ_TMAX:
            # agents_robust HZ_TMAX: late predict -> the no-HZ prediction (_hz_finish would only add its correction to rows[nb + 1])
            self._hz_act = False
            self._rb_hz_skip = ('F',)
        if self._hz_act:
            try:
                self._hz_finish(rows, (hz_W, hz_c1, hz_c2, hz_nu, hz_Phi, hz_map, hz_tail))
            except Exception:
                if HZ_STRICT:
                    raise
        return fnp.stack(rows, axis=0)

    def _lot_basis(self, Cp, WC, w32, nr, ni, pool, sk):
        """portlot: basis Q (nr, kk) of the current pre-activation covariance C = W C_post W^T (one-pass range finder with
        the fixed sketch sk[:, :kk] = layer-weight columns gathered on the alive order, PL_Q QR passes) and its Gram G = Q^T C Q.
        Cp = C_pre (non-trim layers) or None; WC = W C_post (trim layer, where C_pre is never formed: C w = WC (w32 w)).
        ACTIVE SET: w32 = gathered weight (ni, nr), WC (nr, ni)."""
        n = self._nfull
        kk = min(PL_K, nr)
        Om = _pview(sk, (slice(None, None, None), slice(None, kk, None)))
        yb = _win(pool.get('lot_y', (n, kk)), nr, kk)
        tb = None if Cp is not None else _win(pool.get('lot_t', (n, kk)), ni, kk)

        def cmul(X, out):
            if Cp is not None:
                return _public_out(fnp.matmul, Cp, X, out=out)
            fnp.matmul(w32, X, out=tb)
            return _public_out(fnp.matmul, WC, tb, out=out)
        Q, _ = fnp.linalg.qr(cmul(Om, yb))
        for _ in range(PL_Q - 1):
            Q, _ = fnp.linalg.qr(cmul(Q, yb))
        y2 = cmul(Q, _win(pool.get('lot_y2', (n, kk)), nr, kk))
        return (Q, fnp.matmul(Q.T, y2))

    def _hz_setup(self, n, L):
        """setup(): scratch arrays, coefficient tables and the commonly used views of the deferred kappa4 refinement (all data independent)"""
        import time as _time
        t0 = _time.perf_counter()
        f32 = fnp.float32
        lev = HZ_LEV
        KL = list(HZ_K)
        nb = L - 2
        if n != 1024 or L != 16 or len(KL) != L or lev < 1 or n % (1 << lev) or any(k % (1 << lev) or k > HZ_RCH for k in KL) or HZ_RCH % (1 << lev) or KL[L - 1] <= 0 or KL[0] or KL[1]:
            return
        kb = [KL[nb + 1 - b] for b in range(nb)]
        off = [0]
        for k in kb:
            off.append(off[-1] + k)
        R = off[-1]
        E = _HZE(lev, HZ_RCH >> lev, n >> lev)
        E.alloc()
        H = dict(E=E, n=n, nb=nb, kb=kb, off=off, R=R, KL=KL, vc={})
        if HZ_L6 and HZ_AB and HZ_RCH % (1 << (lev + 1)) == 0 and n % (1 << (lev + 1)) == 0:
            H['E6'] = _HZE(lev + 1, HZ_RCH >> (lev + 1), n >> (lev + 1))
            H['E6'].alloc()
        self._hz = H
        for nm in ('V', 'G'):
            H[nm] = fnp.empty((L - 1, R, n), dtype=f32)
        H['S'] = [fnp.empty((R * n,), dtype=f32) for _ in range(2)]
        H['acc_c'] = fnp.zeros((R,), dtype=f32)
        H['acc_i'] = fnp.zeros((R,), dtype=f32)
        H['delta'] = fnp.zeros((R,), dtype=f32)
        H['zeros'] = fnp.zeros((n,), dtype=f32)
        # per-row coefficients of the kappa4 refinement (block order) and per-layer rider coefficients (layer order), constants folded
        rows = lambda fn: [fn(nb + 1 - b) for b in range(nb) for _ in range(kb[b])]
        th = lambda li, f: (HZ_TH[li][f] if HZ_TH[li] is not None and KL[li] > 0 else 0.0)
        H['C3'] = fnp.asarray([rows(lambda li: 12.0 * th(li, f)) for f in range(3)], dtype=f32)
        H['cD'] = fnp.asarray(rows(lambda li: th(li, 3) * 4.0 / 3.0), dtype=f32)
        H['cG'] = fnp.asarray(rows(lambda li: th(li, 4)), dtype=f32)
        H['cC'] = fnp.asarray(rows(lambda li: 12.0 * th(li, 5)), dtype=f32)
        H['cI'] = fnp.asarray(rows(lambda li: 12.0 * th(li, 6)), dtype=f32)
        s2p = 0.3989422804014327
        rc = lambda li, f: (HZ_TH[li][7 + f] if HZ_RIDERS and HZ_TH[li] is not None and li >= 2 else 0.0)
        kf = (-s2p / 6.0, -s2p / 6.0, s2p / 2.0, s2p / 2.0, s2p / 24.0, s2p / 24.0, s2p / 24.0, s2p / 2.0, s2p / 2.0, s2p / 2.0, s2p / 24.0, s2p / 24.0, s2p / 24.0, s2p / 24.0, s2p / 24.0, s2p / 24.0, -s2p / 6.0)
        H['RC'] = [fnp.asarray([[rc(li, f) * kf[f]] for li in range(L)], dtype=f32) for f in range(17)]
        H['rid_any'] = bool(HZ_RIDERS and any(rc(li, f) != 0.0 for li in range(L) for f in range(17)))
        if LOTD_ON and n == 1024:
            # lotdiet A: one (8, L, n) slab for the rider vectors _lot_g4 writes (row views keep the H['sl_*'] names)
            H['sl8'] = fnp.zeros((8, L, n), dtype=f32)
            for i_, nm in enumerate(_LOTD_SL8):
                H['sl_' + nm] = H['sl8'][i_]
            fnp.copyto(H['sl_var'], 1.0)
            H['F8'] = fnp.zeros((8, R), dtype=f32)   # lotdiet F: kappa4 feature blocks, block b at columns off[b]:off[b + 1]
            for nm in ('out',) + tuple(H6_SLABS):
                H['sl_' + nm] = fnp.zeros((L, n), dtype=f32)
        else:
            for nm in ('D3', 'mu', 'port', 'g4', 'out', 'same', 'proj', 'star') + tuple(H6_SLABS):
                H['sl_' + nm] = fnp.zeros((L, n), dtype=f32)
            H['sl_var'] = fnp.zeros((L, n), dtype=f32) + 1.0
        # chunking of the stacked row blocks (the first cnt blocks present) into products of at most HZ_RCH rows
        ch = {}
        for cnt in range(1, nb + 1):
            lst = []
            b0 = 0
            while b0 < cnt:
                b1 = b0 + 1
                while b1 < cnt and off[b1 + 1] - off[b0] <= HZ_RCH:
                    b1 += 1
                if off[b1] > off[b0]:
                    lst.append((b0, b1))
                b0 = b1
            ch[cnt] = tuple(lst)
        H['chunks'] = ch
        for b in range(nb):
            if kb[b]:
                self._hz_fill('V', b, nb + 1 - b, n)
                self._hz_fill('G', b, nb + 1 - b, n)
                self._hz_drow(b)
        for li in range(L):
            for nm in ('D3', 'var', 'mu', 'port', 'g4', 'out', 'same', 'proj', 'star') + tuple(H6_SLABS):
                self._hz_slab(nm, li, n)
        sizes = sorted({(off[b1] - off[b0]) >> lev for lst in ch.values() for b0, b1 in lst})
        widths = [n - 32 * q for q in range(9)]
        # optional: views of the likely operand widths (dead-neuron counts are multiples of 32), the most likely first, for at most
        # HZ_SETUP_T seconds from here; anything else is created on demand in predict
        if HZGATE:
            self._hzg_t0 = _time.perf_counter()
            self._hzg_build()
        if H.get('E6') is not None and not HZGATE:
            # HZ_L6: views of the level-(lev + 1) instance for the widths divisible by 2^(lev + 1), at most HZ6_SETUP_T seconds
            E6 = H['E6']
            l6 = lev + 1
            t6 = _time.perf_counter()
            w6s = [x for x in widths if x % (1 << l6) == 0]
            rs6 = sorted({(off[b1] - off[b0]) >> l6 for lst in ch.values() for b0, b1 in lst if (off[b1] - off[b0]) % (1 << l6) == 0})
            for hw6 in w6s:
                for c6, w6 in [(hw6, x) for x in w6s]:
                    if _time.perf_counter() - t6 > HZ6_SETUP_T:
                        break
                    E6.bplan(c6 >> l6, w6 >> l6)
                    E6._root('B', c6 >> l6, w6 >> l6)
                    for r6 in rs6:
                        E6.plan(r6, c6 >> l6, w6 >> l6)
                        E6._root('A', r6, c6 >> l6)
                        E6._root('O', r6, w6 >> l6)
                for cnt in range(1, nb + 1):
                    for sb in (0, 1):
                        for b0, b1 in ch[cnt]:
                            if (off[b1] - off[b0]) % (1 << l6) == 0 and _time.perf_counter() - t6 <= HZ6_SETUP_T:
                                self._hz_dv6(sb, b0, b1, hw6)
        t1 = _time.perf_counter() if not HZGATE else self._hzg_t0
        for q, hw in enumerate(widths):
            for c, w in [(hw, hw)] + [p for x in widths[:q] for p in ((hw, x), (x, hw))]:
                if _time.perf_counter() - t1 > HZ_SETUP_T:
                    return
                E.bplan(c >> lev, w >> lev)
                E._root('B', c >> lev, w >> lev)
                for r in sizes:
                    E.plan(r, c >> lev, w >> lev)
                    E._root('A', r, c >> lev)
                    E._root('O' if HZ_AB else 'A', r, w >> lev)
            for cnt in range(1, nb + 1):
                if _time.perf_counter() - t1 > HZ_SETUP_T:
                    return
                for sb in (0, 1):
                    self._hz_sv(sb, cnt, hw)
                    for b0, b1 in ch[cnt]:
                        self._hz_dv(sb, b0, b1, hw)
            for a in range(L - 1):
                if _time.perf_counter() - t1 > HZ_SETUP_T:
                    return
                for cnt in (min(nb + 1 - a, nb), min(nb + 2 - a, nb)):
                    self._hz_stage('V', a, cnt, hw)
                    self._hz_stage('G', a, cnt, hw)

    def _hz_sv(self, sb, cnt, hw):
        """natural (rows x hw) form of state buffer sb holding the first cnt row blocks; also the rows of the first cnt - 1 blocks and the last block"""
        H = self._hz
        key = ('s', sb, cnt, hw)
        v = H['vc'].get(key)
        if v is None:
            off = H['off']
            rows = off[cnt]
            full = fnp.reshape(H['S'][sb][:rows * hw], (rows, hw))
            r0 = off[cnt - 1]
            v = (full, full[:r0], full[r0:]) if 0 < r0 < rows else (full, None, None)
            H['vc'][key] = v
        return v

    def _hz_dv(self, sb, b0, b1, hw):
        """row blocks b0..b1-1 of state buffer sb at width hw, split into row and column digits (the product's operand form)"""
        H = self._hz
        key = ('d', sb, b0, b1, hw)
        v = H['vc'].get(key)
        if v is None:
            off = H['off']
            E = H['E']
            v = fnp.reshape(H['S'][sb][off[b0] * hw:off[b1] * hw], E.digits((off[b1] - off[b0]) >> E.L, hw >> E.L))
            H['vc'][key] = v
        return v

    def _hz_dv6(self, sb, b0, b1, hw):
        """HZ_L6: _hz_dv for the level-(HZ_LEV + 1) instance (own cache key: the digit split depends on the level)"""
        H = self._hz
        key = ('d6', sb, b0, b1, hw)
        v = H['vc'].get(key)
        if v is None:
            off = H['off']
            E6 = H['E6']
            v = fnp.reshape(H['S'][sb][off[b0] * hw:off[b1] * hw], E6.digits((off[b1] - off[b0]) >> E6.L, hw >> E6.L))
            H['vc'][key] = v
        return v

    def _hz_stage(self, nm, a, cnt, hw):
        """stored legs of birth a for the first cnt row blocks at width hw: (all their rows, the rows of the last block)"""
        H = self._hz
        key = (nm, a, cnt, hw)
        v = H['vc'].get(key)
        if v is None:
            off = H['off']
            base = H[nm][a]
            v = (base[:off[cnt], :hw], base[off[cnt - 1]:off[cnt], :hw] if off[cnt] > off[cnt - 1] else None)
            H['vc'][key] = v
        return v

    def _hz_acc(self, nm, cnt):
        H = self._hz
        key = (nm, cnt)
        v = H['vc'].get(key)
        if v is None:
            v = H[nm][:H['off'][cnt]]
            H['vc'][key] = v
        return v

    def _hz_fill(self, nm, b, k, nc):
        """destination of the legs (k births) of row block b"""
        H = self._hz
        key = ('f', nm, b, k, nc)
        v = H['vc'].get(key)
        if v is None:
            off = H['off']
            v = H[nm][:k, off[b]:off[b + 1]] if nc == H['n'] else H[nm][:k, off[b]:off[b + 1], :nc]
            H['vc'][key] = v
        return v

    def _hz_drow(self, b):
        H = self._hz
        key = ('dr', b)
        v = H['vc'].get(key)
        if v is None:
            off = H['off']
            v = H['delta'][off[b]:off[b + 1]]
            H['vc'][key] = v
        return v

    def _lotd_sl8(self, li, nr):
        """lotdiet A: rows li (first nr entries) of the 8 rider slabs as one (8, nr) view of H['sl8']"""
        H = self._hz
        key = ('sl8', li, nr)
        v = H['vc'].get(key)
        if v is None:
            v = H['sl8'][:, li, :nr]
            H['vc'][key] = v
        return v

    def _lotd_f8(self, b):
        """lotdiet F: columns of row block b of the kappa4 feature array H['F8']"""
        H = self._hz
        key = ('f8', b)
        v = H['vc'].get(key)
        if v is None:
            off = H['off']
            v = H['F8'][:, off[b]:off[b + 1]]
            H['vc'][key] = v
        return v

    def _hz_slab(self, nm, li, nr):
        """row li (first nr entries) of the rider slab nm"""
        H = self._hz
        key = ('sl', nm, li, nr)
        v = H['vc'].get(key)
        if v is None:
            row = H['vc'].get(('sl', nm, li, H['n']))
            if row is None:
                row = H['sl_' + nm][li]
                H['vc'][('sl', nm, li, H['n'])] = row
            v = row if nr == H['n'] else row[:nr]
            H['vc'][key] = v
        return v

    def _hz_z(self, m):
        H = self._hz
        key = ('z', m)
        v = H['vc'].get(key)
        if v is None:
            v = H['zeros'][:m]
            H['vc'][key] = v
        return v

    def _hz_select(self, T2, T3, k, li, mu, var, response_same, w2b=None):
        """rows with the largest kappa4 sensitivity of the mean; their legs (V = c2 P Gamma in T2, Gamma in T3) are stored for the deferred pullback"""
        H = self._hz
        K = H['KL'][li] if li < len(H['KL']) else 0
        if K <= 0:
            return None
        nr = int(T2.shape[1])
        nc = int(T2.shape[2])
        if nr < K or nc != H['n'] or k != li or li in self._hz_st:
            self._hz_act = False
            return None
        sa_ = self.__dict__.get('_lotd_sa') if LOTD_ON else None
        if LOTD_ON and sa_ is not None and sa_[0] is var and sa_[1] is mu:
            sg, al = sa_[2], sa_[3]   # lotdiet C: = fnp.sqrt(var), mu / sg of this layer (AU_REUSE), not recomputed
        else:
            sg = fnp.sqrt(var)
            al = mu / sg
        a2 = al * al
        order = fnp.argsort(fnp.abs(a2 - 1.0) * fnp.exp(a2 * -0.5) / (var * sg) * response_same)
        R = order[nr - K:]
        b = H['nb'] + 1 - li
        if w2b is not None:
            # J2_LOT: T2 here is T1: the stored V rows are take(T1) o w2b, written in place; G rows gathered into their slots
            fnp.multiply(fnp.take(T2, R, axis=1), w2b, out=self._hz_fill('V', b, k, nc))
            fnp.take(T3, R, axis=1, out=self._hz_fill('G', b, k, nc))
        else:
            fnp.copyto(self._hz_fill('V', b, k, nc), fnp.take(T2, R, axis=1))
            fnp.copyto(self._hz_fill('G', b, k, nc), fnp.take(T3, R, axis=1))
        return (order, R, nr)

    def _hz_pull_w(self, hz_W, hz_c1, hz_c2, hz_nu):
        """HZW pullback (module note in hzw_patch.py): every product X @ W_j contracts over the SDT window n3_j of layer j (the rows are
        exactly zero beyond it); S[0] holds the product input at the contraction width, S[1] the product output at the full width"""
        H = self._hz
        E = H['E']
        E6 = H.get('E6')
        lev = E.L
        msk = (1 << lev) - 1
        nb = H['nb']
        off = H['off']
        chunks = H['chunks']
        n3s = self._hz_n3

        def win(j_):
            hw_ = int(hz_W[j_].shape[0])
            c_ = int(n3s[j_])
            if HZGATE and 0 < c_ < hw_ and (c_ & msk) == 0 and (j_, c_, int(hz_W[j_].shape[1])) not in H.get('gp', ()):
                if HZG_LOG:
                    self._hzg_note(('w', j_, c_, int(hz_W[j_].shape[1])))
                return hw_
            return c_ if (0 < c_ < hw_ and (c_ & msk) == 0) else hw_
        cnt = 1
        cw = win(nb)
        fnp.copyto(self._hz_sv(0, cnt, cw)[0], self._hz_stage('V', nb, cnt, cw)[0])
        for j in range(nb, -1, -1):
            if HZ_TMAX_P > 0 and _rb_time.perf_counter() - self._rb_t0 > HZ_TMAX_P:
                self._rb_hz_skip = ('P', j)   # agents_robust HZ_TMAX_P: late -> stop the pullback; _hz_finish returns before rows
                return False
            W = hz_W[j]
            hw1 = int(W.shape[0])
            hw0 = int(W.shape[1])
            rin = off[cnt]
            cnt_out = min(cnt + 1, nb) if j > 0 else cnt
            rout = off[cnt_out]
            Y = self._hz_sv(1, cnt, hw0)[0]
            Wc = W if cw == hw1 else W[:cw]
            if (cw & msk) == 0 and (hw0 & msk) == 0:
                c = cw >> lev
                w = hw0 >> lev
                # engine3 HZ_L6 (E6 present): level choice by its exact cost rule (_hz6_flags) on the CONTRACTED width cw
                if HZG_LOG and (j, cw, hw0) not in H.get('gp', ()):
                    self._hzg_note(('p', j, cw, hw0))
                f6 = _hz6_flags(lev, [off[b1] - off[b0] for b0, b1 in chunks[cnt]], cw, hw0) if E6 is not None and (not HZGATE or (HZG_L6 and (j, cw, hw0) in H.get('gp', ()))) else None
                if f6 is None or not all(f6):
                    E.set_right(Wc, c, w)
                if f6 is not None and any(f6):
                    E6.set_right(Wc, c >> 1, w >> 1)
                for i6, (b0, b1) in enumerate(chunks[cnt]):
                    if f6 is not None and f6[i6]:
                        E6.mm(self._hz_dv6(0, b0, b1, cw), self._hz_dv6(1, b0, b1, hw0), (off[b1] - off[b0]) >> (lev + 1), c >> 1, w >> 1)
                    else:
                        E.mm(self._hz_dv(0, b0, b1, cw), self._hz_dv(1, b0, b1, hw0), (off[b1] - off[b0]) >> lev, c, w)
            else:
                fnp.matmul(self._hz_sv(0, cnt, cw)[0], Wc, out=Y)
            ac = self._hz_acc('acc_c', cnt)
            if j == 0:
                fnp.add(ac, fnp.einsum('ij,ij->i', Y, Y), out=ac)
                break
            a = j - 1
            fnp.add(ac, fnp.einsum('ij,ij,j->i', Y, Y, hz_nu[a]), out=ac)
            ai = self._hz_acc('acc_i', cnt)
            Gs = self._hz_stage('G', a, cnt, hw0)[0]
            fnp.add(ai, fnp.einsum('ij,ij,ij,j->i', Gs, Gs, Y, hz_c2[a]), out=ai)
            cwn = win(a)
            Xn = self._hz_sv(0, cnt_out, cwn)
            Xi = Xn[0] if rout == rin else Xn[1]
            fnp.multiply(Y if cwn == hw0 else Y[:, :cwn], hz_c1[a] if cwn == hw0 else hz_c1[a][:cwn], out=Xi)
            fnp.add(Xi, self._hz_stage('V', a, cnt, cwn)[0], out=Xi)
            if rout > rin:
                fnp.copyto(Xn[2], self._hz_stage('V', a, cnt_out, cwn)[1])
            cnt = cnt_out
            cw = cwn

    def _hzg_build(self):
        """HZGATE (hzgate_patch.py): deterministic prewarm of the pullback layer products of _HZG_PROF, in this order: (1) the
        current-SDT shapes as predict runs them (c = n3 when HZW and the window is legal, else nr; w = ni), (2) with HZW the full-width
        product (j, nr, ni) of every windowed shape (the exact fallback of a refused window), (3) the old-log shapes, (4) HZG_EXT.  Each
        product (j, c, w) gets its level-5 plans, W tree and roots and every state / stage / digit / accumulator view its iteration
        of the pullback loop touches (HZW loop: input in S[0] at c, output in S[1] at w; parent loop: alternating buffers), plus the
        transport's zero rows of its widths.  No time check, except the HZG_MAX_T safety cap (far above the normal list time; only a
        pathologically slow host stops early: the products not built simply fall back).  H['gp'] = the prebuilt (j, c, w)."""
        import time as _time
        t0 = _time.perf_counter()
        H = self._hz
        E = H['E']
        lev = E.L
        msk = (1 << lev) - 1
        nb = H['nb']
        off = H['off']
        ch = H['chunks']
        KL = H['KL']
        gp = set()
        H['gp'] = gp
        win = lambda nr, n3: n3 if (HZW and 0 < n3 < nr and (n3 & msk) == 0) else nr
        # build order (value first, matters only under the cap): current-SDT shapes, full-width fallbacks of the windowed shapes,
        # the old-log shapes, HZG_EXT
        todo = [(j, win(nr, n3), ni, nr) for j, nr, ni, n3 in _HZG_PROF[:_HZG_NCUR]]
        if HZW:
            todo += [(j, nr, ni, nr) for j, nr, ni, n3 in _HZG_PROF if win(nr, n3) != nr]
        todo += [(j, win(nr, n3), ni, nr) for j, nr, ni, n3 in _HZG_PROF[_HZG_NCUR:]]
        if HZG_EXT:
            todo += [(j, win(nr, n3), ni, nr) for j, nr, ni, n3 in _HZG_EXT]
        for j, cw, ni, nr in todo:
            if j > nb or (j, cw, ni) in gp:
                continue
            if _time.perf_counter() - t0 > HZG_MAX_T:
                break
            # transport (_hz_finish): zero rows of the unselected neurons (layer j; layer nb + 1 proxied by the widths of layer nb)
            for li_, nr_ in ((j, nr),) + (((nb + 1, nr),) if j == nb else ()):
                if 0 < KL[li_] <= nr_:
                    self._hz_z(nr_ - KL[li_])
            if j == nb and nr < H['n']:
                self._hz_z(H['n'] - nr)
            cnt = min(nb - j + 1, nb)
            cnt_o = min(cnt + 1, nb) if j > 0 else cnt
            si = 0 if HZW else (nb - j) % 2
            so = 1 - si
            if (cw & msk) == 0 and (ni & msk) == 0:
                c = cw >> lev
                w = ni >> lev
                E6 = H.get('E6') if HZG_L6 else None
                f6 = _hz6_flags(lev, [off[b1] - off[b0] for b0, b1 in ch[cnt]], cw, ni) if E6 is not None else None
                if f6 is None or not all(f6):
                    E.bplan(c, w)
                    E._root('B', c, w)
                if f6 is not None and any(f6):
                    E6.bplan(c >> 1, w >> 1)
                    E6._root('B', c >> 1, w >> 1)
                for i6, (b0, b1) in enumerate(ch[cnt]):
                    if f6 is not None and f6[i6]:
                        r6 = (off[b1] - off[b0]) >> (lev + 1)
                        E6.plan(r6, c >> 1, w >> 1)
                        E6._root('A', r6, c >> 1)
                        E6._root('O' if HZ_AB else 'A', r6, w >> 1)
                        self._hz_dv6(si, b0, b1, cw)
                        self._hz_dv6(so, b0, b1, ni)
                        continue
                    r = (off[b1] - off[b0]) >> lev
                    E.plan(r, c, w)
                    E._root('A', r, c)
                    E._root('O' if HZ_AB else 'A', r, w)
                    self._hz_dv(si, b0, b1, cw)
                    self._hz_dv(so, b0, b1, ni)
            else:
                self._hz_sv(si, cnt, cw)
            self._hz_acc('acc_c', cnt)
            if HZW:
                # _hz_pull_w: Y = S[1] at w (rows of cnt); the input of layer j at c is formed by the step through layer j + 1
                self._hz_sv(1, cnt, ni)
                self._hz_sv(0, cnt, cw)
                if j == nb:
                    self._hz_stage('V', nb, cnt, cw)
                else:
                    cnt_p = min(nb - j, nb)
                    self._hz_stage('V', j, cnt_p, cw)
                    if off[cnt] > off[cnt_p]:
                        self._hz_stage('V', j, cnt, cw)
                if j > 0:
                    self._hz_acc('acc_i', cnt)
                    self._hz_stage('G', j - 1, cnt, ni)
            else:
                # parent loop: output into the other buffer, at the rows of the next stage
                self._hz_sv(so, cnt_o, ni)
                if j == nb:
                    self._hz_sv(0, cnt, cw)
                    self._hz_stage('V', nb, cnt, cw)
                if j > 0:
                    self._hz_acc('acc_i', cnt)
                    self._hz_stage('G', j - 1, cnt, ni)
                    self._hz_stage('V', j - 1, cnt, ni)
                    if off[cnt_o] > off[cnt]:
                        self._hz_stage('V', j - 1, cnt_o, ni)
            gp.add((j, cw, ni))
        H['gp'] = frozenset(gp)

    def _hzg_note(self, item):
        fb = getattr(self, '_hzg_fb', None)
        if fb is not None:
            fb.append(item)

    def _hzg_snap(self):
        """HZG_LOG: sizes of every cache / registry a predict can grow (no flopscope op)"""
        H = self._hz
        s = [len(H['vc']) if H is not None else 0]
        for e in ((H['E'], H.get('E6')) if H is not None else (None, None)):
            s += [len(e.vc), len(e.plans), len(e.bplans)] if e is not None else [0, 0, 0]
        smm = getattr(self, '_smm', None)
        z = getattr(smm, 'z', None) if smm is not None else None
        s += [len(z.plans), int(z.public_zview_count), len(z.vc), len(z.zb), len(z.flat)] if z is not None else [0] * 5
        pool = getattr(self, '_pool', None)
        s += [len(pool.public_views), len(pool.bufs)] if pool is not None else [0, 0]
        return s

    @property
    def lazy_builds(self):
        """HZG_LOG: creations during the last predict (module note of hzgate_patch.py); None with the knob off / before a predict"""
        s0 = getattr(self, '_hzg_t', None)
        if not HZG_LOG or s0 is None:
            return None
        s1 = self._hzg_snap()
        d = [b - a for a, b in zip(s0, s1)]
        H = self._hz
        smm = getattr(self, '_smm', None)
        z = getattr(smm, 'z', None) if smm is not None else None
        new = lambda dct, n0: [k for k in list(dct)[n0:]] if dct is not None and len(dct) >= n0 else ['<wiped>']
        E = H['E'] if H is not None else None
        E6 = H.get('E6') if H is not None else None
        hzk = new(H['vc'], s0[0]) if H is not None else []
        pull = [k for k in hzk if k[0] in ('s', 'd', 'd6', 'V', 'G', 'acc_c', 'acc_i')]
        return dict(hz_vc=d[0], hz_pull=len(pull), hz_other=d[0] - len(pull), hz_pull_new=pull[:40],
                    e5=tuple(d[1:4]), e6=tuple(d[4:7]), zf=tuple(d[7:12]), pool=tuple(d[12:14]),
                    e5_new=(new(E.plans, s0[2]) + new(E.bplans, s0[3])) if E is not None else [],
                    e6_new=(new(E6.plans, s0[5]) + new(E6.bplans, s0[6])) if E6 is not None else [],
                    zf_new=new(z.plans, s0[7]) if z is not None else [],
                    gate=list(getattr(self, '_hzg_fb', None) or ()))

    def _mid_edge(self, fd, alpha, sigma, var, D3, k4u, li):
        """mid_patch MID_EDGE: Edgeworth correction of an intermediate layer's dropped rows (same algebra as fin_patch._fin_edge, MID_* scales).
        Kept rows give s3 = sum(g1 a m) / sum(a^2 m), g1 = D3 / sigma^3, k4 = mean(kappa4 / var^2 | m), m = |alpha| >= MID_EDGE_AMIN;
        returns sigma_d phi(a_d) [-s3 a_d^2 / 6 + k4 (a_d^2 - 1) / 24 + s3^2 a_d^2 (a_d^4 - 6 a_d^2 + 3) / 72]."""
        f32 = fnp.float32
        a_c, sg_, kl = fd
        sgd = _pview(sg_, slice(None, kl, None))
        msk = fnp.greater_equal(fnp.abs(alpha), MID_EDGE_AMIN).astype(f32)
        am = fnp.multiply(alpha, msk)
        s3 = float(fnp.sum(fnp.multiply(fnp.divide(D3, fnp.multiply(var, sigma)), am)))
        den = float(fnp.sum(fnp.multiply(alpha, am)))
        k4 = float(fnp.sum(fnp.multiply(fnp.divide(k4u, fnp.multiply(var, var)), msk)))
        cnt = float(fnp.sum(msk))
        s3 = s3 / den if den > 1e-6 else 0.0
        k4 = k4 / cnt if cnt > 0.5 else 0.0
        s3 = min(max(s3 * MID_EDGE_S3S, 0.0), 0.2) if s3 == s3 else 0.0
        k4 = min(max(k4 * MID_EDGE_K4S, 0.0), 0.5) if k4 == k4 else 0.0
        a2 = fnp.multiply(a_c, a_c)
        ph = flops.stats.norm.pdf(a_c).astype(f32)
        t3 = fnp.multiply(a2, -s3 / 6.0)
        t4 = fnp.multiply(fnp.subtract(a2, 1.0), k4 / 24.0)
        t33 = fnp.multiply(fnp.multiply(a2, fnp.add(fnp.multiply(a2, fnp.subtract(a2, 6.0)), 3.0)), s3 * s3 / 72.0)
        corr = fnp.multiply(fnp.multiply(sgd, ph), fnp.add(fnp.add(t3, t4), t33))
        if MID_DEBUG:
            import json as _mj, sys as _ms
            _ms.stderr.write('MID_EDGE ' + _mj.dumps(dict(li=int(li), kl=int(kl), s3=s3, k4=k4, rms=float(fnp.sqrt(fnp.mean(fnp.multiply(corr, corr)))),
                                                          amax=float(fnp.max(fnp.abs(corr))))) + '\n')
        return corr

    def _fin_edge(self, fd, alpha, sigma, var, D3, k4u):
        f32 = fnp.float32   # fin_patch_v2: local alias (f32 is not module-level)
        import json as _fj, sys as _fs   # fin_patch_v2: FIN_DEBUG branch names
        """fin_patch FIN_EDGE: Edgeworth correction of the dropped final rows' Gaussian ReLU mean (module docstring of fin_patch.py).
        Kept rows of the final layer give the per-net radial proxy: s3 = sum(g1 a m) / sum(a^2 m), g1 = D3 / sigma^3, k4 = mean(kappa4 / var^2 | m),
        m = |alpha| >= FIN_EDGE_AMIN.  Returns sigma_d phi(a_d) [-s3 a_d^2 / 6 + k4 (a_d^2 - 1) / 24 + s3^2 a_d^2 (a_d^4 - 6 a_d^2 + 3) / 72]."""
        a_c, sg_, kl = fd
        sgd = _pview(sg_, slice(None, kl, None))
        msk = fnp.greater_equal(fnp.abs(alpha), FIN_EDGE_AMIN).astype(f32)
        am = fnp.multiply(alpha, msk)
        s3 = float(fnp.sum(fnp.multiply(fnp.divide(D3, fnp.multiply(var, sigma)), am)))
        den = float(fnp.sum(fnp.multiply(alpha, am)))
        k4 = float(fnp.sum(fnp.multiply(fnp.divide(k4u, fnp.multiply(var, var)), msk)))
        cnt = float(fnp.sum(msk))
        s3 = s3 / den if den > 1e-6 else 0.0
        k4 = k4 / cnt if cnt > 0.5 else 0.0
        s3 = min(max(s3 * FIN_EDGE_S3S, 0.0), 0.2) if s3 == s3 else 0.0
        k4 = min(max(k4 * FIN_EDGE_K4S, 0.0), 0.5) if k4 == k4 else 0.0
        a2 = fnp.multiply(a_c, a_c)
        ph = flops.stats.norm.pdf(a_c).astype(f32)
        t3 = fnp.multiply(a2, -s3 / 6.0)
        t4 = fnp.multiply(fnp.subtract(a2, 1.0), k4 / 24.0)
        t33 = fnp.multiply(fnp.multiply(a2, fnp.add(fnp.multiply(a2, fnp.subtract(a2, 6.0)), 3.0)), s3 * s3 / 72.0)
        corr = fnp.multiply(fnp.multiply(sgd, ph), fnp.add(fnp.add(t3, t4), t33))
        if FIN_DEBUG:
            import json as _fj, sys as _fs
            _fs.stderr.write('FIN_EDGE ' + _fj.dumps(dict(kl=int(kl), s3=s3, k4=k4, rms=float(fnp.sqrt(fnp.mean(fnp.multiply(corr, corr)))),
                                                          amax=float(fnp.max(fnp.abs(corr))))) + '\n')
        return corr

    def _hz_finish(self, rows, hz):
        """deferred refinement of the means: pullback of the stored legs through all earlier layers, batched over the layers per birth stage
        (row blocks in descending layer order: stage a holds the layers > a); the fitted first-order mean changes (selected rows: kappa4
        replacement; all rows: riders) are carried to the final layer with the Gaussian mean response and added to its prediction."""
        hz_W, hz_c1, hz_c2, hz_nu, hz_Phi, hz_map, hz_tail = hz
        H = self._hz
        st = self._hz_st
        nb = H['nb']
        KL = H['KL']
        off = H['off']
        kb = H['kb']
        if any((KL[li] > 0) != (li in st) for li in range(2, nb + 2)) or len(hz_c1) != nb + 1 or len(hz_W) != nb + 2 or len(hz_Phi) != nb + 2:
            return False
        E = H['E']
        lev = E.L
        n = H['n']
        msk = (1 << lev) - 1
        chunks = H['chunks']
        fnp.copyto(H['acc_c'], 0.0)
        fnp.copyto(H['acc_i'], 0.0)
        if HZW and len(getattr(self, '_hz_n3', ())) == nb + 1:
            self._hz_pull_w(hz_W, hz_c1, hz_c2, hz_nu)
        else:
            sb = 0
            cnt = 1
            hw = int(hz_W[nb].shape[0])
            fnp.copyto(self._hz_sv(sb, cnt, hw)[0], self._hz_stage('V', nb, cnt, hw)[0])
            for j in range(nb, -1, -1):
                if HZ_TMAX_P > 0 and _rb_time.perf_counter() - self._rb_t0 > HZ_TMAX_P:
                    self._rb_hz_skip = ('P', j)   # agents_robust HZ_TMAX_P: late -> stop the pullback; _hz_finish returns before rows
                    break
                W = hz_W[j]
                hw1 = int(W.shape[0])
                hw0 = int(W.shape[1])
                rin = off[cnt]
                cnt_out = min(cnt + 1, nb) if j > 0 else cnt
                rout = off[cnt_out]
                So = self._hz_sv(1 - sb, cnt_out, hw0)
                Y = So[0] if rout == rin else So[1]
                if (hw1 & msk) == 0 and (hw0 & msk) == 0:
                    c = hw1 >> lev
                    w = hw0 >> lev
                    E6 = H.get('E6')
                    if HZG_LOG and (j, hw1, hw0) not in H.get('gp', ()):
                        self._hzg_note(('p', j, hw1, hw0))
                    f6 = _hz6_flags(lev, [off[b1] - off[b0] for b0, b1 in chunks[cnt]], hw1, hw0) if E6 is not None and (not HZGATE or (HZG_L6 and (j, hw1, hw0) in H.get('gp', ()))) else None
                    if f6 is None or not all(f6):
                        E.set_right(W, c, w)
                    if f6 is not None and any(f6):
                        E6.set_right(W, c >> 1, w >> 1)
                    for i6, (b0, b1) in enumerate(chunks[cnt]):
                        if f6 is not None and f6[i6]:
                            E6.mm(self._hz_dv6(sb, b0, b1, hw1), self._hz_dv6(1 - sb, b0, b1, hw0), (off[b1] - off[b0]) >> (lev + 1), c >> 1, w >> 1)
                        else:
                            E.mm(self._hz_dv(sb, b0, b1, hw1), self._hz_dv(1 - sb, b0, b1, hw0), (off[b1] - off[b0]) >> lev, c, w)
                else:
                    fnp.matmul(self._hz_sv(sb, cnt, hw1)[0], W, out=Y)
                ac = self._hz_acc('acc_c', cnt)
                if j == 0:
                    fnp.add(ac, fnp.einsum('ij,ij->i', Y, Y), out=ac)
                    break
                a = j - 1
                fnp.add(ac, fnp.einsum('ij,ij,j->i', Y, Y, hz_nu[a]), out=ac)
                ai = self._hz_acc('acc_i', cnt)
                Gs = self._hz_stage('G', a, cnt, hw0)[0]
                fnp.add(ai, fnp.einsum('ij,ij,ij,j->i', Gs, Gs, Y, hz_c2[a]), out=ai)
                fnp.multiply(Y, hz_c1[a], out=Y)
                fnp.add(Y, self._hz_stage('V', a, cnt, hw0)[0], out=Y)
                if rout > rin:
                    fnp.copyto(So[2], self._hz_stage('V', a, cnt_out, hw0)[1])
                sb = 1 - sb
                cnt = cnt_out
        if HZ_TMAX_P > 0 and self.__dict__.get('_rb_hz_skip') is not None:
            return False   # agents_robust HZ_TMAX_P fired in the pullback: rows untouched (no-HZ prediction)
        # the kappa4 replacement on the selected rows (block order): features 0 same, 1 proj, 2 star, 3 D3, 4 var, 5 g4row, 6 mu, 7 kappa4 used
        if LOTD_ON and 'F8' in H and H['rid_any']:
            F = H['F8']   # lotdiet F: filled block by block by _lot_g4 (every KL > 0 layer is present: checked above)
        else:
            F = fnp.concatenate([st[nb + 1 - b][2] for b in range(nb) if kb[b]], axis=1)
        sps = F[0:3]
        d3 = F[3]
        vr = F[4]
        mu = F[6]
        new = fnp.einsum('fr,fr->r', H['C3'], sps) + H['acc_c'] * H['cC'] + H['acc_i'] * H['cI'] + d3 * d3 / vr * H['cD'] + F[5] * H['cG']
        sg = fnp.sqrt(vr)
        al = mu / sg
        a2 = al * al
        c4 = (a2 - 1.0) * fnp.exp(a2 * -0.5) / (vr * sg) * (0.3989422804014327 / 24.0)
        dl = H['delta']
        fnp.subtract(new, F[7], out=dl)
        fnp.multiply(dl, c4, out=dl)
        # riders on all live rows of every layer (batched over the layers; padded entries are finite and never read)
        rid = H['rid_any'] and len(self._hz_rid) > 0
        if rid:
            RC = H['RC']
            vs = H['sl_var']
            sgs = fnp.sqrt(vs)
            als = H['sl_mu'] / sgs
            a2s = als * als
            ps = fnp.exp(a2s * -0.5)
            if LOTD_ON:
                # lotdiet D: vs * sgs, als * ps and (a2s - 1) * ps / (vs * sgs) once (same operands, same order as each inline copy)
                vsg_ = vs * sgs
                aps_ = als * ps
                pq_ = (a2s - 1.0) * ps / vsg_
                t = aps_ / vs * H['sl_D3'] * (RC[0] + RC[1] * a2s) + ps * sgs * (RC[2] + RC[3] * a2s)
                t = t + ps / vsg_ * (H['sl_same'] * RC[7] + H['sl_proj'] * RC[8] + H['sl_star'] * RC[9])
                t = t + pq_ * H['sl_h6_d21_w1'] * RC[12] + pq_ * H['sl_h6_d21_e'] * RC[13] + pq_ * H['sl_h6_d21_s'] * RC[14] + pq_ * H['sl_h6_d21_d3'] * RC[15] + aps_ / vsg_ * H['sl_h6_k31b_s'] * RC[16]
                fnp.add(t, pq_ * (H['sl_port'] * (RC[4] + RC[5] * a2s) + H['sl_g4'] * RC[6] + H['sl_D3'] * H['sl_D3'] / vs * RC[10] + H['sl_same'] * RC[11]), out=H['sl_out'])
            else:
                t = als * ps / vs * H['sl_D3'] * (RC[0] + RC[1] * a2s) + ps * sgs * (RC[2] + RC[3] * a2s)
                t = t + ps / (vs * sgs) * (H['sl_same'] * RC[7] + H['sl_proj'] * RC[8] + H['sl_star'] * RC[9])
                t = t + (a2s - 1.0) * ps / (vs * sgs) * H['sl_h6_d21_w1'] * RC[12] + (a2s - 1.0) * ps / (vs * sgs) * H['sl_h6_d21_e'] * RC[13] + (a2s - 1.0) * ps / (vs * sgs) * H['sl_h6_d21_s'] * RC[14] + (a2s - 1.0) * ps / (vs * sgs) * H['sl_h6_d21_d3'] * RC[15] + als * ps / (vs * sgs) * H['sl_h6_k31b_s'] * RC[16]
                fnp.add(t, (a2s - 1.0) * ps / (vs * sgs) * (H['sl_port'] * (RC[4] + RC[5] * a2s) + H['sl_g4'] * RC[6] + H['sl_D3'] * H['sl_D3'] / vs * RC[10] + H['sl_same'] * RC[11]), out=H['sl_out'])
        if HZ_DEBUG is not None:
            HZ_DEBUG.append(dict(F=F, chd=fnp.copy(H['acc_c']), ind=fnp.copy(H['acc_i']), new=new, c4=c4, delta=fnp.copy(dl), st={li: (s[0], s[1]) for li, s in st.items()}, rid=(fnp.copy(H['sl_out']) if rid else None), rid_nr=dict(self._hz_rid)))
        # transport of the mean changes to the final layer (alive order of every layer; Gaussian mean response Phi(alpha) W)
        e = None
        ep = None
        for li in range(2, nb + 2):
            d = None
            s = st.get(li)
            if s is not None:
                b = nb + 1 - li
                d = fnp.take(fnp.concatenate([self._hz_z(s[1] - kb[b]), self._hz_drow(b)]), fnp.argsort(s[0]))
            if rid and li in self._hz_rid:
                r_ = self._hz_slab('out', li, self._hz_rid[li])
                d = r_ if d is None else d + r_
            if e is not None:
                t = hz_Phi[li] * fnp.matmul(hz_W[li], e)
                d = t if d is None else d + t
            ep = e
            e = d
        if e is None:
            return False
        inv, nrl = hz_map[nb + 1]
        if inv is None:
            v = e
        elif hz_tail is not None and ep is not None:
            # dropped neurons of the last layer include the always-on tail: their mean follows the sign of the pre-activation mean
            dd = fnp.maximum(fnp.sign(hz_tail[1]), 0.0) * fnp.matmul(ep, hz_tail[0])
            v = fnp.take(fnp.concatenate([dd, e]), inv)
        else:
            v = fnp.take(fnp.concatenate([self._hz_z(n - nrl), e]), inv)
        rows[nb + 1] = rows[nb + 1] + v
        return True

    def _lot_g4(self, bufs, A_st, P_st, k, n, slab, Q, Gq, g4row, li, D3=None, var=None, mu=None, old=None):
        """portlot: LOT shape of the kappa4 diagonal for the mean program at the current layer (module docstring).
        Returns the rescaled vector (V29's per-layer level x LOT shape) or None when the scale guard keeps V29's value.
        The op stream does not depend on the data (the guard only chooses which array the caller uses)."""
        T1 = _pview(bufs['t'], slice(None, k, None))
        T2 = _pview(bufs['mp'], slice(None, k, None))
        T3 = _pview(bufs['u'], slice(None, k, None))
        CV = _pview(_pview(slab, (slice(None, k, None), 0)), (slice(None, None, None), None, slice(None, None, None)))
        pool = self._pool
        kk = Q.shape[1]
        fnp.multiply(P_st, CV, out=T3)
        fnp.add(T3, A_st, out=T3)
        fnp.multiply(P_st, T3, out=T1)
        j2l_ = J2_LOT
        if j2l_:
            # J2_LOT: T2 = T1 o w2b is not formed (w2b folded into gu, the response_same weights and the stored V rows below)
            T2 = T1
        else:
            fnp.multiply(T1, bufs['w2b'], out=T2)
        gu = _pview(pool.get('lot_gu', (bufs['t'].shape[0], n, kk)), slice(None, k, None))
        ka_, kb_ = (int(old[0]), int(old[1])) if old is not None else (0, 0)
        if 0 < ka_ <= k and bufs['xt'].shape[1] >= max(old[2].shape[1] if ka_ > kb_ else 0, old[4].shape[1] if kb_ > 0 else 0):
            # cost1 (LOT old-source right query): old slot j < ka is stored as Gamma_j = B (FA_j + FP_j CV_j) with
            # B = Qc (tier 1) or Qc U (tier 2, the lifted QU), so Gamma_j^T Q = (FA_j + FP_j CV_j)^T (B^T Q): the
            # (n, nr, kk) products of the old slots become (n, r, kk) ones plus ONE shared (r, nr, kk) right
            # query per tier (exact regrouping); young slots keep the n-space product.  Scratch for the factor
            # combination: bufs['xt'] (free after _dslices; alias group of its own).
            _, _, FAo_, FPo_, FA2_, FP2_, Qc_, QU_ = old
            hw_y = hw_1 = n
            if LOTW and self._as_on:
                hw_y = max(_pview(self._hub_birth_widths, slice(ka_, k, None)), default=n)
                hw_1 = max(_pview(self._hub_birth_widths, slice(kb_, ka_, None)), default=n) if SDT_OLDW else n
            if ka_ < k:
                if hw_y < n:
                    fnp.matmul(fnp.swapaxes(_pview(T3, (slice(ka_, None, None), slice(None, None, None), slice(None, hw_y, None))), -1, -2), Q, out=_pview(gu, (slice(ka_, None, None), slice(None, hw_y, None))))
                else:
                    fnp.matmul(fnp.swapaxes(_pview(T3, slice(ka_, None, None)), -1, -2), Q, out=_pview(gu, slice(ka_, None, None)))
            if ka_ > kb_:
                bq1 = fnp.matmul(fnp.swapaxes(Qc_, 0, 1), Q)
                F1 = _pview(bufs['xt'], (slice(None, ka_ - kb_, None), slice(None, FAo_.shape[1], None)))
                if hw_1 < n:
                    F1 = _pview(F1, (slice(None, None, None), slice(None, None, None), slice(None, hw_1, None)))
                    fnp.multiply(_pview(FPo_, (slice(None, None, None), slice(None, None, None), slice(None, hw_1, None))), _pview(CV, (slice(kb_, ka_, None), slice(None, None, None), slice(None, hw_1, None))), out=F1)
                    fnp.add(F1, _pview(FAo_, (slice(None, None, None), slice(None, None, None), slice(None, hw_1, None))), out=F1)
                    fnp.matmul(fnp.swapaxes(F1, -1, -2), bq1, out=_pview(gu, (slice(kb_, ka_, None), slice(None, hw_1, None))))
                else:
                    fnp.multiply(FPo_, _pview(CV, slice(kb_, ka_, None)), out=F1)
                    fnp.add(F1, FAo_, out=F1)
                    fnp.matmul(fnp.swapaxes(F1, -1, -2), bq1, out=_pview(gu, slice(kb_, ka_, None)))
            if kb_ > 0:
                bq2 = _pview(bq1, slice(None, QU_.shape[1], None)) if (FX_UEYE and FIJA2 and ka_ > kb_) else fnp.matmul(fnp.swapaxes(QU_, 0, 1), Q)
                F2 = _pview(bufs['xt'], (slice(ka_ - kb_, ka_, None), slice(None, FA2_.shape[1], None)))
                fnp.multiply(FP2_, _pview(CV, slice(None, kb_, None)), out=F2)
                fnp.add(F2, FA2_, out=F2)
                fnp.matmul(fnp.swapaxes(F2, -1, -2), bq2, out=_pview(gu, slice(None, kb_, None)))
        else:
            ka_ = kb_ = 0
            hw_y = hw_1 = n
            fnp.matmul(fnp.swapaxes(T3, -1, -2), Q, out=gu)
        if j2l_:
            # J2_LOT: KU = sum_kj (T1 o w2b) gu = sum_kj T1 (gu o w2b)
            gu = fnp.multiply(gu, fnp.swapaxes(bufs['w2b'], -1, -2))
        if LOTW and (hw_y < n or hw_1 < n) and ka_ > 0:
            # LOTW: KU = sum over tiers of their windowed contractions (tier 2 full width, tier 1 on hw_1, young on hw_y)
            KU = None
            for s0_, s1_, w_ in ((0, kb_, n), (kb_, ka_, hw_1), (ka_, k, hw_y)):
                if s1_ <= s0_:
                    continue
                part_ = fnp.einsum('kij,kjq->iq', _pview(T2, (slice(s0_, s1_, None), slice(None, None, None), slice(None, w_, None))), _pview(gu, (slice(s0_, s1_, None), slice(None, w_, None))))
                KU = part_ if KU is None else KU + part_
        else:
            KU = fnp.einsum('kij,kjq->iq', T2, gu)
        response_X = fnp.linalg.solve(Gq, KU.T)
        cand = fnp.einsum('iq,qi->i', KU, response_X)
        proj_only = cand
        response_same = response_star = None
        if PL_SAME:
            if j2l_:
                # J2_LOT: sum_j (T1 w2b)^2 slab1 = sum_j T1^2 (w2b^2 slab1)
                w2k_ = _pview(bufs['w2b'], (slice(None, None, None), 0))
                response_same = fnp.einsum('kij,kij,kj->i', T2, T2, fnp.multiply(_pview(slab, (slice(None, k, None), 1)), fnp.multiply(w2k_, w2k_)))
            else:
                response_same = fnp.einsum('kij,kij,kj->i', T2, T2, _pview(slab, (slice(None, k, None), 1)))
            cand = cand + response_same
        hz_sel = None
        hz_here = self._hz_act and li >= HZ_LMIN and response_same is not None and PL_STAR and D3 is not None and k >= 2
        if hz_here:
            hz_sel = self._hz_select(T2, T3, k, li, mu, var, response_same, w2b=bufs['w2b'] if j2l_ else None)
        if PL_STAR:
            fnp.multiply(T3, T3, out=T3)
            fnp.multiply(T3, T1, out=T3)
            response_star = fnp.einsum('kij,kj->i', T3, _pview(slab, (slice(None, k, None), 2)))
            cand = cand + response_star
        m1 = float(fnp.mean(cand))
        mg = float(fnp.mean(g4row))
        ok = m1 > 0.0 and mg > 0.0 and (m1 == m1) and (mg == mg) and (m1 < float('inf')) and (mg < float('inf')) and (PL_RMIN <= mg / (12.0 * m1) <= PL_RMAX)
        g4m = fnp.multiply(cand, mg / m1 if ok else 1.0)
        if hz_here and self._hz_act:
            port_used = g4m if ok else g4row
            if LOTD_ON and self._hz['rid_any'] and 'sl8' in self._hz:
                # lotdiet A: 1 stack into the rider slab (was 8 copyto) and the features taken from it (was a second stack)
                nr_ = int(var.shape[0])
                s8_ = self._lotd_sl8(li, nr_)
                fnp.stack([response_same, proj_only, response_star, D3, var, g4row, mu, port_used], axis=0, out=s8_)
                if hz_sel is not None:
                    f8_ = self._lotd_f8(self._hz['nb'] + 1 - li)
                    fnp.take(s8_, hz_sel[1], axis=1, out=f8_)
                    self._hz_st[li] = (hz_sel[0], hz_sel[2], f8_)
                self._hz_rid[li] = nr_
            elif hz_sel is not None:
                self._hz_st[li] = (hz_sel[0], hz_sel[2], fnp.take(fnp.stack([response_same, proj_only, response_star, D3, var, g4row, mu, port_used]), hz_sel[1], axis=1))
            if self._hz['rid_any'] and not (LOTD_ON and 'sl8' in self._hz):
                nr_ = int(var.shape[0])
                for nm_, vec_ in (('D3', D3), ('var', var), ('mu', mu), ('port', port_used), ('g4', g4row), ('same', response_same), ('proj', proj_only), ('star', response_star)):
                    fnp.copyto(self._hz_slab(nm_, li, nr_), vec_)
                self._hz_rid[li] = nr_
        if PL_DEBUG:
            m2 = float(fnp.mean(fnp.abs(cand)))
            PL_LOG.append(dict(li=li, m1=m1, m2=m2, mg=mg, ok=ok, level=mg / m1 if m1 != 0.0 else float('nan'), cand=[float(x) for x in cand.tolist()], g4=[float(x) for x in g4row.tolist()]))
        if RESP_ON and 8 <= li <= 14 and ok:
            self._resp_data = (KU, response_X, response_same, response_star, mg / m1)
        return g4m if ok else None

    def _au_dn_thin(self, PPL, Z_st, fb, nr, k, ncw):
        """AU_THINW: _dn_thin on the read columns [:ncw] (right operand rows [:ncw]); same staging, same contraction"""
        pool = self._pool
        n = self._nfull
        q = PPL.shape[2]
        Lc = pool.get('dn_lc', (1, 1, n, DN_WMAX))
        Rc = pool.get('dn_rc', (1, 1, n, DN_WMAX))
        Oc = pool.get('dn_oc', (1, n, n))
        L2 = _pview(Lc, (0, 0, slice(None, nr, None)))
        R2 = _pview(Rc, (0, 0, slice(None, nr, None)))
        w = q * k
        fnp.multiply(fnp.swapaxes(PPL, 0, 1), 1.0 / 3.0, out=fnp.reshape(_pview(L2, (slice(None, None, None), slice(None, w, None))), (nr, k, q)))
        fnp.copyto(fnp.reshape(_pview(R2, (slice(None, None, None), slice(None, w, None))), (nr, k, q)), fnp.swapaxes(Z_st, 0, 1))
        if fb is not None:
            GYR, GXR, Zf_s = fb
            ks, rf = (GYR.shape[0], GYR.shape[2])
            Lf = fnp.reshape(_pview(L2, (slice(None, None, None), slice(w, w + 2 * rf * ks, None))), (nr, ks, 2 * rf))
            fnp.copyto(_pview(Lf, (Ellipsis, slice(None, rf, None))), fnp.swapaxes(GXR, 0, 1))
            fnp.copyto(_pview(Lf, (Ellipsis, slice(rf, None, None))), fnp.swapaxes(GYR, 0, 1))
            fnp.copyto(fnp.reshape(_pview(R2, (slice(None, None, None), slice(w, w + 2 * rf * ks, None))), (nr, ks, 2 * rf)), fnp.swapaxes(Zf_s, 0, 1))
            w += 2 * rf * ks
        Lv, wp = _dn_pad(nr, w, ncw, DN_THIN2D_LEV, C7_MIN, True)
        if wp > w:
            fnp.copyto(_pview(R2, (slice(None, None, None), slice(w, wp, None))), 0.0)
        smm = self._smm
        smm.mn = C7_MIN
        smm.hub(_pview(Lc, (slice(None, None, None), slice(None, None, None), slice(None, nr, None), slice(None, wp, None))), _pview(Rc, (slice(None, None, None), slice(None, None, None), slice(None, ncw, None), slice(None, wp, None))), _pview(Oc, (slice(None, None, None), slice(None, nr, None), slice(None, ncw, None))), Lv)
        smm.mn = STRASSEN_MIN
        return _pview(Oc, (0, slice(None, nr, None), slice(None, ncw, None)))

    def _dn_thin(self, PPL, Z_st, fb, nr, k):
        """DN_THIN2D: thin = (1/3) sum_k PPL_k Z_k^T (incl. (1/3) R^T Yk: DN_YCOL put R into PPL's last column) + sum_s GYR_s F2_s^T +
        GXR_s F1_s^T as ONE Z-core hub product Lc Rc^T on 2-D staging buffers (rows = the nr live neurons):
        Lc = [PPL_1/3 .. PPL_k/3 | GXR_s GYR_s ..], Rc = [Z_1 .. Z_k | F1_s F2_s .. (= Zf_s)], contraction zero-padded to the level grid"""
        pool = self._pool
        n = self._nfull
        q = PPL.shape[2]
        Lc = pool.get('dn_lc', (1, 1, n, DN_WMAX))
        Rc = pool.get('dn_rc', (1, 1, n, DN_WMAX))
        Oc = pool.get('dn_oc', (1, n, n))
        L2 = _pview(Lc, (0, 0, slice(None, nr, None)))
        R2 = _pview(Rc, (0, 0, slice(None, nr, None)))
        w = q * k
        fnp.multiply(fnp.swapaxes(PPL, 0, 1), 1.0 / 3.0, out=fnp.reshape(_pview(L2, (slice(None, None, None), slice(None, w, None))), (nr, k, q)))
        fnp.copyto(fnp.reshape(_pview(R2, (slice(None, None, None), slice(None, w, None))), (nr, k, q)), fnp.swapaxes(Z_st, 0, 1))
        if fb is not None:
            GYR, GXR, Zf_s = fb
            ks, rf = (GYR.shape[0], GYR.shape[2])
            Lf = fnp.reshape(_pview(L2, (slice(None, None, None), slice(w, w + 2 * rf * ks, None))), (nr, ks, 2 * rf))
            fnp.copyto(_pview(Lf, (Ellipsis, slice(None, rf, None))), fnp.swapaxes(GXR, 0, 1))
            fnp.copyto(_pview(Lf, (Ellipsis, slice(rf, None, None))), fnp.swapaxes(GYR, 0, 1))
            fnp.copyto(fnp.reshape(_pview(R2, (slice(None, None, None), slice(w, w + 2 * rf * ks, None))), (nr, ks, 2 * rf)), fnp.swapaxes(Zf_s, 0, 1))
            w += 2 * rf * ks
        Lv, wp = _dn_pad(nr, w, nr, DN_THIN2D_LEV, C7_MIN, True)
        if wp > w:
            fnp.copyto(_pview(R2, (slice(None, None, None), slice(w, wp, None))), 0.0)
        smm = self._smm
        smm.mn = C7_MIN
        smm.hub(_pview(Lc, (slice(None, None, None), slice(None, None, None), slice(None, nr, None), slice(None, wp, None))), _pview(Rc, (slice(None, None, None), slice(None, None, None), slice(None, nr, None), slice(None, wp, None))), _pview(Oc, (slice(None, None, None), slice(None, nr, None), slice(None, nr, None))), Lv)
        smm.mn = STRASSEN_MIN
        return _pview(Oc, (0, slice(None, nr, None), slice(None, nr, None)))

    def _dn_ztransport(self, smm, Z_st, Zf_st, k, kz, nr, ni, kc, z_side, li, r, rfb, L):
        """DN_Z2D / DN_ZKIN: Z <- WD Z, Zf[kz:] <- WD Zf[kz:] contracting over the first kc rows (DN_ZKIN: kc = kin, the rows [kin:] of
        every thin leg meet zero columns of WD = W diag(w1_prev), w1_prev[n3:] = 0); DN_Z2D (li >= DN_Z2D_LMIN): both as ONE Z-core
        product WD[:, :kc] [Z_0 .. Z_{k-1} | Zf_kz .. Zf_{k-1}] on 2-D staging buffers (output columns padded to the level grid)"""
        pool = self._pool
        n = self._nfull
        q = r + 2
        f = 2 * rfb
        zo = _pview(_win(pool.get(('z', z_side), (L - 1, n, q)), nr, q), slice(None, k, None))
        fb = Zf_st is not None and k > kz
        zfa = _win(pool.get(('zf', z_side), (L - 1, n, f)), nr, f) if fb else None
        WDc = _win(pool.get('wd', (n, n)), nr, kc)
        Zin = _pview(Z_st, (slice(None, None, None), slice(None, kc, None)))
        Zfin = _pview(Zf_st, (slice(kz, None, None), slice(None, kc, None))) if fb else None
        if DN_Z2D and li >= DN_Z2D_LMIN:
            ks = k - kz if fb else 0
            w = q * k + f * ks
            Lz, wp = _dn_pad(nr, kc, w, DN_Z2D_LEV, C7_MIN, False)
            Si = pool.get('dn_zi', (1, 1, n, DN_WMAX))
            So = pool.get('dn_zo', (1, 1, n, DN_WMAX))
            I2 = _pview(Si, (0, 0, slice(None, kc, None)))
            O2 = _pview(So, (0, 0, slice(None, nr, None)))
            fnp.copyto(fnp.reshape(_pview(I2, (slice(None, None, None), slice(None, q * k, None))), (kc, k, q)), fnp.swapaxes(Zin, 0, 1))
            if fb:
                fnp.copyto(fnp.reshape(_pview(I2, (slice(None, None, None), slice(q * k, w, None))), (kc, ks, f)), fnp.swapaxes(Zfin, 0, 1))
            smm.mn = C7_MIN
            smm.mm(_pview(WDc, (None, None)), _pview(Si, (slice(None, None, None), slice(None, None, None), slice(None, kc, None), slice(None, wp, None))), _pview(So, (slice(None, None, None), slice(None, None, None), slice(None, nr, None), slice(None, wp, None))), Lz)
            smm.mn = STRASSEN_MIN
            fnp.copyto(zo, fnp.swapaxes(fnp.reshape(_pview(O2, (slice(None, None, None), slice(None, q * k, None))), (nr, k, q)), 0, 1))
            if fb:
                fnp.copyto(_pview(zfa, slice(kz, k, None)), fnp.swapaxes(fnp.reshape(_pview(O2, (slice(None, None, None), slice(q * k, w, None))), (nr, ks, f)), 0, 1))
        else:
            fnp.matmul(_pview(WDc, None), Zin, out=zo)
            if fb:
                fnp.matmul(_pview(WDc, None), Zfin, out=_pview(zfa, slice(kz, k, None)))
        return (zo, _pview(zfa, slice(None, k, None)) if fb else Zf_st)

    def _response31(self, C_off, var, Q, data):
        """Minimum symmetric-H response, full cumulant units, wk431[i,j]=K31[j,i]."""
        KU, X, same, star, level = data
        C = fnp.copy(C_off)
        fnp.fill_diagonal(C, var)
        scale = level / 12.0
        E = fnp.matmul(KU, X) * scale
        E = 0.5 * (E + E.T)
        if same is not None:
            fnp.fill_diagonal(E, fnp.diag(E) + same * scale)
        D = fnp.matmul(X.T, fnp.matmul(Q.T, C)) * scale ** 0.5
        q, d = (fnp.diag(E), fnp.diag(D))
        aa = C / _pview(var, (slice(None, None, None), None))
        bb = (D - aa * _pview(d, (slice(None, None, None), None))) / _pview(var, (slice(None, None, None), None))
        K31 = 6.0 * (aa * _pview(q, (slice(None, None, None), None)) + bb * _pview(d, (slice(None, None, None), None)) + aa.T * E + bb.T * D)
        if star is not None:
            star = star * level
            K31 = K31 + 0.75 * _pview(star, (slice(None, None, None), None)) * aa + 0.25 * _pview(star, (None, slice(None, None, None))) * aa.T * aa.T * aa.T
        if RESP_SCALE != 1.0:
            K31 = K31 * RESP_SCALE
        return _zero_diag(K31.T)

    @staticmethod
    def _gram2(pool, FA, dA, FP, dP):
        """trimalg: FA d(dA) FA^T + FP d(dP) FP^T (r x r, symmetric) as ONE aliased Gram
        G = F F^T with F = [FA sqrt(dA) | FP sqrt(dP)] (r x 2n): the dA, dP weights are
        positive (9 + w2^2 + 9 e^2, 1 + s^2); flopscope bills the aliased einsum on the
        r (r + 1) / 2 unique entries, i.e. half the two plain products (a syrk does the same)."""
        r, n = FA.shape
        FG = pool.get('fg', (r, 2 * n))
        fnp.multiply(FA, _pview(fnp.sqrt(dA), (None, slice(None, None, None))), out=_pview(FG, (slice(None, None, None), slice(None, n, None))))
        fnp.multiply(FP, _pview(fnp.sqrt(dP), (None, slice(None, None, None))), out=_pview(FG, (slice(None, None, None), slice(n, None, None))))
        return fnp.einsum('aj,bj->ab', FG, FG)

    def _sym_product(self, X, Y, nr, ni, out, lev, sym4=False):
        """V29: out = X Y for a SYMMETRIC (nr x nr) result (X = W C (nr x ni), Y = w32 = W^T (ni x nr)): the three block
        products (11, 12, 22) of the 2x2 partition as ONE Strassen family (0.75 of the
        dense count before the recursion); the 21 block is the copy of 12^T. Odd n (smoke
        shapes) falls back to the plain product.  ACTIVE SET: nr, ni are the alive counts (<= the full width nn)."""
        h = nr // 2
        if nr % 2 or nr < 4:
            return _public_out(fnp.matmul, X, Y, out=out)
        if sym4 and nr % 4 == 0 and nr >= 64 and self._nfull % 4 == 0:
            return self._au_sym4(X, Y, nr, ni, out, lev)
        pool = self._pool
        nn = self._nfull
        hn = nn // 2
        X3 = _win(pool.get('sym_x', (3, 1, hn, nn)), h, ni)
        Y3 = _win(pool.get('sym_y', (3, 1, nn, hn)), ni, h)
        O3 = _win(pool.get('sym_o', (3, 1, hn, hn)), h, h)
        fnp.copyto(X3[0, 0], _pview(X, slice(None, h, None)))
        fnp.copyto(X3[1, 0], _pview(X, slice(None, h, None)))
        fnp.copyto(X3[2, 0], _pview(X, slice(h, None, None)))
        fnp.copyto(Y3[0, 0], _pview(Y, (slice(None, None, None), slice(None, h, None))))
        fnp.copyto(Y3[1, 0], _pview(Y, (slice(None, None, None), slice(h, None, None))))
        fnp.copyto(Y3[2, 0], _pview(Y, (slice(None, None, None), slice(h, None, None))))
        self._smm.mn = self._mn_y
        self._smm.mm(X3, Y3, O3, self._smm.level(h, ni, h, lev))
        self._smm.mn = STRASSEN_MIN
        fnp.copyto(_pview(out, (slice(None, h, None), slice(None, h, None))), O3[0, 0])
        fnp.copyto(_pview(out, (slice(None, h, None), slice(h, None, None))), O3[1, 0])
        fnp.copyto(_pview(out, (slice(h, None, None), slice(h, None, None))), O3[2, 0])
        fnp.copyto(_pview(out, (slice(h, None, None), slice(None, h, None))), O3[1, 0].T)
        return out

    def _au_sym4(self, X, Y, nr, ni, out, lev):
        """AU_SYM4: out = X Y for the SYMMETRIC (nr x nr) C_pre from the k (k + 1) / 2 blocks (a <= b) of the k x k partition as ONE
        Strassen family of (q x ni x q) products, q = nr / k: k = 4 (10 blocks, 0.625 of the dense count), or k = 3 (6 blocks, 0.667)
        when AU_SYM4_K3 and nr % 192 == 0, nr >= 768 (q = 320 / 256 = 5 / 4 x 64: two more recursion levels than q = nr / 4 = 240 / 192).
        Slots row-major (_AU_SYM_PAIRS[k]); left slots = row block a of X (broadcast copies), right slots = column block b of Y (blocks
        0..k-1, then the suffix runs [a:k] copied slot to slot); the blocks below the diagonal are the transposes of the column blocks
        above it."""
        k_ = 3 if (AU_SYM4_K3 and nr % 192 == 0 and nr >= 768 and self._nfull >= 960) else 4
        q = nr // k_
        P_ = _AU_SYM_PAIRS[k_]
        S_ = len(P_)
        pool = self._pool
        nn = self._nfull
        qn = nn // 4 if k_ == 4 else nn // 3 // 64 * 64
        Xs_ = _win(pool.get('sym%d_x' % k_, (S_, 1, qn, nn)), q, ni)
        Ys_ = _win(pool.get('sym%d_y' % k_, (S_, 1, nn, qn)), ni, q)
        Os_ = _win(pool.get('sym%d_o' % k_, (S_, 1, qn, qn)), q, q)
        s0_ = 0
        for a_ in range(k_):
            s1_ = s0_ + k_ - a_
            fnp.copyto(_pview(Xs_, (slice(s0_, s1_, None), 0)), _pview(X, (None, slice(a_ * q, (a_ + 1) * q, None))))
            s0_ = s1_
        for b_ in range(k_):
            fnp.copyto(_pview(Ys_, (b_, 0)), _pview(Y, (slice(None, None, None), slice(b_ * q, (b_ + 1) * q, None))))
        s0_ = k_
        for a_ in range(1, k_):
            s1_ = s0_ + k_ - a_
            fnp.copyto(_pview(Ys_, (slice(s0_, s1_, None), 0) if s1_ - s0_ > 1 else (s0_, 0)), _pview(Ys_, (slice(a_, k_, None), 0) if s1_ - s0_ > 1 else (a_, 0)))
            s0_ = s1_
        # the parent's (h x ni x h) family reaches its leaf of 7-8 through LX (base level >= LX_FROM = 5); the q-block family's base
        # level under the young min block would stay below LX_FROM (q = 256: L4, leaf 16) -> its base level is taken at min block
        # AU_SYM4_MN (q = 256: L5, leaf 8; q = 224: L5, leaf 7), the parent's leaf size
        self._smm.mn = min(self._mn_y, AU_SYM4_MN)
        self._smm.mm(Xs_, Ys_, Os_, self._smm.level(q, ni, q, lev))
        self._smm.mn = STRASSEN_MIN
        for s_, (a_, b_) in enumerate(P_):
            fnp.copyto(_pview(out, (slice(a_ * q, (a_ + 1) * q, None), slice(b_ * q, (b_ + 1) * q, None))), _pview(Os_, (s_, 0)))
        for b_ in range(1, k_):
            fnp.copyto(_pview(out, (slice(b_ * q, (b_ + 1) * q, None), slice(None, b_ * q, None))), _pview(out, (slice(None, b_ * q, None), slice(b_ * q, (b_ + 1) * q, None))).T)
        return out

    def _hub2(self, bufs, apb4, k0, k, n, nr):
        """V26: sum over slots k0..k-1 of LA_k A_k^T + LP_k P_k^T as ONE Strassen family
        (LA/LP live interleaved in bufs["lap"], A/P in the legs, so the pair axis folds
        into the contraction batch; V28: on the one-time 4-D views).  ACTIVE SET: operands are
        (nr, n) windows (hub width n), the result (nr, nr).
        trimalg: with bufs["cp"] set, ONE extra slot (X = W C, Y = W) is appended to the same
        family as a second output group: C_pre = (W C) W^T comes out of the same recursion and
        the separate C_pre family (101 ops per layer) disappears."""
        smm = self._smm
        lv = min(STRASSEN_HUB, self._s_hub)
        if not TA_YH_S6 <= 2 * (k - k0) <= TA_YH_SMAX:
            lv = min(lv, 5)
        if lv > 0:
            smm.mn = self._mn_y
        try:
            if bufs.get('cp'):
                outg = bufs['hubg']
                _lv_h = smm.level(nr, n, nr, lv)
                if smm.first and (not self._as_on):
                    _sl = min(STRASSEN_HUB, TA_YH_LEV)
                    if not TA_YH_S6 <= 2 * (k - k0) <= TA_YH_SMAX:
                        _sl = min(_sl, 5)
                    _st = smm.level(nr, n, nr, _sl)
                    smm.steady = _st if _st > _lv_h else 0
                _sl = slice(2 * k0, 2 * k + 1, None)
                smm.hubg(_pview(bufs['lap4'], _sl), _pview(apb4, _sl), outg, _lv_h, 2 * (k - k0), pad=(_pview(bufs['lap4p'], _sl), _pview(bufs['apb4p'], _sl), bufs['hubgp']))
                return outg[0, 0]
            out = bufs['hub']
            _lv_h = smm.level(nr, n, nr, lv)
            if smm.first and (not self._as_on):
                _sl = min(STRASSEN_HUB, TA_YH_LEV)
                if not TA_YH_S6 <= 2 * (k - k0) <= TA_YH_SMAX:
                    _sl = min(_sl, 5)
                _st = smm.level(nr, n, nr, _sl)
                smm.steady = _st if _st > _lv_h else 0
            _sl = slice(2 * k0, 2 * k, None)
            smm.hub(_pview(bufs['lap4'], _sl), _pview(apb4, _sl), _pview(out, None), _lv_h, pad=(_pview(bufs['lap4p'], _sl), _pview(bufs['apb4p'], _sl), bufs['hubp']))
            return out
        finally:
            smm.mn = STRASSEN_MIN

    def _d21_demand(self, bufs, apb4, ka, kb, k, n, nr, sb1, sb2, Qc, U2, s_sb, Q, alpha, li, sk=None, thin=None):
        """Read exact kink rows of the heavy D21 core, project remaining rows.

        D3, stored K3 and every inexpensive thin contribution remain exact.
        All nonlinear programs and residual bookkeeping receive this same D21.
        Source-factor right projections include their actual matmul bills.
        """
        rk = min(_D21_RANK_LM.get(li, D21_RANK), Q.shape[1])   # d21rl_patch
        Q = _pview(Q, (slice(None, None, None), slice(None, rk, None)))
        nk = int(float(fnp.sum(fnp.abs(alpha) <= _D21_CUT_L(li))))
        m = min(nr, max(D21_GRAN, (nk + D21_GRAN - 1) // D21_GRAN * D21_GRAN))
        permutation = fnp.arange(nr) if m == nr else fnp.argsort(fnp.abs(alpha))
        order = _pview(permutation, slice(None, m, None))
        smm = self._smm
        # SDT: D21 columns [nc:] (the ndc nearly-dead neurons with alpha < -SDT_COL, last in the live order) are zero: the caller zeroes
        # them; with SDT_COLW the whole demand (young Gram, dense old-tier lift, projection basis and tail rows) runs on the first ncw
        # columns only (= rows [:ncw] of the right operands, a prefix view)
        nc = nr
        if self._sdt_ord:
            # (AU_THINW: the same count was taken in _dslices from the same alpha / layer)
            nc = self._au_nc if (AU_THINW and self._au_nc >= 0) else _sdt_win(nr, int(float(fnp.sum(fnp.less(alpha, -_SDT_COL_L(li))))))
        self._au_nc = -1
        self._sdt_nc = nc
        ncw = nc if SDT_COLW and (not bufs.get('cp')) else nr
        # cdp_patch: exact young-hub column window ncwp <= ncw (64-grid live-order prefix leaving out alpha <= -CDP_C);
        # the band [ncwp:ncw] of the young part is filled by a rank-CDP_R left projection after the young parts (below)
        ncwp = ncw
        if CDP_ON and self._sdt_ord and SDT_COLW and (not bufs.get('cp')) and CDP_LMIN <= li <= CDP_LMAX and self._nfull == 1024:
            ncwp = min(ncw, _sdt_win(nr, int(float(fnp.sum(fnp.less(alpha, -CDP_C))))))
            if ncwp < 64:
                ncwp = ncw
        cdp_parts = [] if ncwp < ncw else None
        cdp_acc = [fnp.zeros((m, ncwp), dtype=fnp.float32) if cdp_parts is not None else None]
        dq = fnp.zeros((nr - m, rk), dtype=fnp.float32)
        exact = fnp.zeros((m, ncw), dtype=fnp.float32)
        left = bufs['lap4']
        parts = []
        cp_exact = None

        def part(a, b, right, lift, lev, mb):
            """lift None: young slots, exact rows accumulated here; 'qc' / 'u2': old-tier slots, the (m, r) factor-space
            core is RETURNED (cost1: the lift to n-space is applied once to the summed cores, see below)."""
            nonlocal dq, exact, cp_exact
            if b <= a:
                return None
            x = _pview(left, slice(2 * a, 2 * b, None))
            ys = right
            if YCOL_READOUT and self._as_on and (lift is None) and (b <= k) and (not bufs.get('cp')):
                widths = self._hub_birth_widths
                assert len(widths) >= b
                cols = max(_pview(widths, slice(a, b, None)), default=n)
                assert 0 < cols <= n
                if cols < n:
                    x = _pview(x, (..., slice(None, cols, None)))
                    ys = _pview(ys, (..., slice(None, cols, None)))
                self._young_readout_shapes.append((li, m, nr, b - a, cols))
            if lift is None and ncw < nr:
                ys = _win(right, ncw, ys.shape[-1])
            ys_h = ys
            if lift is None and cdp_parts is not None:
                # cdp_patch: the exact young hub reads right-operand rows (= D21 columns) [:ncwp] only
                ys_h = _win(right, ncwp, ys.shape[-1])
                cdp_parts.append((x, ys))
            x_au = x
            if lift == 'qc' and AU_T1PK and self._au_t1pk is not None and self._au_t1pk[0] == 'pair':
                # AU_T1PK pair: the left legs LA/LP of the narrow pairs are exactly zero beyond their hub width (zero legs and zero hub weights);
                # their free columns take the overflow columns of the wide pairs, exactly as the (packed) factors ys
                wbk_ = self._au_t1pk[1]
                for js_, a_, b_, jt_, c_, d_ in self._au_t1pk[2]:
                    fnp.copyto(_au_pair(x, jt_, slice(c_, d_, None)), _au_pair(x, js_, slice(a_, b_, None)))
                x = _pview(x, (..., slice(None, wbk_, None)))
                ys = _pview(ys, (..., slice(None, wbk_, None)))
            elif lift == 'qc' and self._sdt_ow1 < n:
                # SDT_OLDW: tier-1 factors and left legs are zero beyond the tier's hub window
                x = _pview(x, (..., slice(None, self._sdt_ow1, None)))
                ys = _pview(ys, (..., slice(None, self._sdt_ow1, None)))
            xb = None
            dim = ys.shape[2]
            smm.mn = mb
            core = None
            if lift is None and b == k and bufs.get('cp'):
                _sl = slice(2 * a, 2 * b + 1, None)
                cx = _pview(left, _sl)
                if m == nr:
                    xp = _pview(bufs['lap4p'], _sl)
                else:
                    xp = _pview(self._pool.get('d21x', (left.shape[0] + 1, 1, n, n)), slice(None, 2 * (b - a) + 1, None))
                    cx = fnp.take(cx, order, axis=2, out=_pview(xp, (..., slice(None, m, None), slice(None, n, None))))
                cy = _pview(apb4, _sl)
                inner = _pview(bufs['hubg'], (slice(None, None, None), slice(None, None, None), slice(None, m, None), slice(None, dim, None)))
                smm.hubg(cx, cy, inner, smm.level(m, n, dim, lev), 2 * (b - a), pad=(xp, _pview(bufs['apb4p'], _sl), bufs['hubgp']))
                ex = inner[0, 0]
                cp_exact = inner[1, 0]
            elif lift is None:
                _sl = slice(2 * a, 2 * b, None)
                if m == nr:
                    xb = x
                    xp = _pview(bufs['lap4p'], _sl)
                else:
                    xp = _pview(self._pool.get('d21x', (left.shape[0] + 1, 1, n, n)), slice(None, 2 * (b - a), None))
                    xb = fnp.take(x, order, axis=2, out=_pview(xp, (..., slice(None, m, None), slice(None, x.shape[-1], None))))
                innerp = self._pool.get('d21inner', (1, n, n))
                inner = _win(innerp, m, ys_h.shape[2])
                smm.hub(xb, ys_h, inner, smm.level(m, ys_h.shape[-1], ys_h.shape[2], lev), pad=(xp, _pview(bufs['apb4p'], _sl), innerp))
                ex = inner[0]
            elif lift == 'qc' and AU_T1PK and self._au_t1pk is not None and self._au_t1pk[0] == 'ext':
                # AU_T1PK ext: S + 1 gathered left slots (extra slot 0 = the overflow columns [wb:w] of the wide slots, slot order) against the
                # factor stack with its extra slot in front; the tail below keeps the plain windows (factors / left legs untouched)
                wbk_, grk_, fxk_ = self._au_t1pk[1], self._au_t1pk[2], self._au_t1pk[3]
                S_ = x_au.shape[0]
                xbb_ = self._pool.get('au_t1x', (max(9, S_ + 1), 1, n, n))
                xb = _pview(xbb_, (slice(None, S_ + 1, None), slice(None, None, None), slice(None, m, None), slice(None, wbk_, None)))
                fnp.take(_pview(x_au, (..., slice(None, wbk_, None))), order, axis=2, out=_pview(xbb_, (slice(1, S_ + 1, None), slice(None, None, None), slice(None, m, None), slice(None, wbk_, None))))
                x0_ = _pview(xbb_, (0, 0, slice(None, m, None)))
                used_ = 0
                for s0_, s1_, w_, off_ in grk_:
                    fnp.take(_pview(x_au, (slice(s0_, s1_, None), slice(None, None, None), slice(None, None, None), slice(wbk_, w_, None))), order, axis=2, out=_au_extc(self, x0_, off_, s1_ - s0_, w_ - wbk_, True))
                    used_ = off_ + (s1_ - s0_) * (w_ - wbk_)
                if used_ < wbk_:
                    fnp.copyto(_pview(x0_, (slice(None, None, None), slice(used_, wbk_, None))), 0.0)
                inner = _win(self._pool.get('inner', (1, n, dim)), m, dim)
                smm.hub(xb, _win(fxk_, dim, wbk_), inner, smm.level(m, wbk_, dim, lev))
                core = _pview(inner, 0)
            else:
                xb = x if m == nr else fnp.take(x, order, axis=2)
                inner = _win(self._pool.get('inner' if lift == 'qc' else 'inner2', (1, n, dim)), m, dim)
                smm.hub(xb, ys, inner, smm.level(m, ys.shape[-1], dim, lev))
                core = _pview(inner, 0)
            smm.mn = STRASSEN_MIN
            if core is None:
                if cdp_parts is not None:
                    cdp_acc[0] = cdp_acc[0] + ex   # cdp_patch: exact young block, columns [:ncwp]
                else:
                    exact = exact + ex
            parts.append((x, ys, lift))
            return core
        if ka < k:
            yr_g = [(ka, k)]
            if YR_D and YCOL_READOUT and self._as_on and (not bufs.get('cp')) and k - ka >= 2:
                yr_g = [(ka + g0_, ka + g1_) for g0_, g1_, _w in _yr_split(list(self._hub_birth_widths[ka:k]), 2, 0, 0.0, YR_DMINSAVE)]
                if len(yr_g) > 1:
                    YR_LOG.append(('d', li, yr_g))
            for a_, b_ in yr_g:
                lv = min(STRASSEN_HUB, self._s_hub)
                if not TA_YH_S6 <= 2 * (b_ - a_) <= TA_YH_SMAX:
                    lv = min(lv, 5)
                part(a_, b_, _pview(apb4, slice(2 * a_, 2 * b_, None)), None, lv, self._mn_y)
        if cdp_parts:
            # cdp_patch: young D21 band columns [ncwp:ncw] of the exact rows ~ U (U^T Y_band); U = top-CDP_R left basis of the exact young
            # block (one-pass range finder on it, sketch = the layer weight slice, + one power iteration)
            ey_ = cdp_acc[0]
            rc_ = max(1, min(CDP_R, ncwp, m))
            skc_ = sk if sk is not None else Q
            om_ = _pview(skc_, (slice(None, ncwp, None), slice(None, rc_, None)))
            U_, _ = fnp.linalg.qr(fnp.matmul(ey_, om_))
            U_, _ = fnp.linalg.qr(fnp.matmul(ey_, fnp.matmul(fnp.swapaxes(ey_, 0, 1), U_)))
            UT_ = fnp.swapaxes(U_, 0, 1)
            G_ = None
            for x_, ys_ in cdp_parts:
                x3_ = _pview(x_, (slice(None, None, None), 0))
                xg_ = x3_ if m == nr else fnp.take(x3_, order, axis=1)
                yb_ = _pview(ys_, (slice(None, None, None), 0, slice(ncwp, ncw, None)))
                t2_ = fnp.matmul(fnp.matmul(UT_, xg_), fnp.swapaxes(yb_, -1, -2))
                g_ = fnp.sum(t2_, axis=0)
                G_ = g_ if G_ is None else G_ + g_
            exact = exact + fnp.concatenate([ey_, fnp.matmul(U_, G_)], axis=1)
            if CDP_DEBUG:
                CDP_LOG.append((li, nr, m, ncw, ncwp))
        elif cdp_parts is not None:
            exact = exact + fnp.concatenate([cdp_acc[0], fnp.zeros((m, ncw - ncwp), dtype=fnp.float32)], axis=1)
        c1 = part(kb, ka, sb1, 'qc', s_sb, _TAG['old_min']) if ka > kb else None
        c2 = part(0, kb, sb2, 'u2', s_sb, _TAG['old_min']) if kb > 0 else None
        if c1 is not None or c2 is not None:
            # cost1: exact regrouping (inner1 + inner2 U2^T) Qc^T of inner1 Qc^T + inner2 (U2^T Qc^T): one
            # (m, r_old, nr) lift instead of two lifts and the (r2, r_old, nr) product U2^T Qc^T; the lift runs
            # on the pooled Strassen family exactly like the ungated path of _dslices when the exact-row count
            # m is large enough for the ~110-op family call to pay (D21_LIFT_FMM_M rows), dense otherwise.
            core = c1
            if c2 is not None and FX_UEYE and FIJA2:
                # FX_UEYE: inner2 U^T with U = [I; 0] = inner2 in the first r2 columns (zero-padded)
                r2c_ = c2.shape[1]
                if core is None:
                    core = _win(self._pool.get('lift', (n, U2.shape[0])), m, U2.shape[0])
                    fnp.copyto(_pview(core, (slice(None, None, None), slice(None, r2c_, None))), c2)
                    fnp.copyto(_pview(core, (slice(None, None, None), slice(r2c_, None, None))), 0.0)
                else:
                    cv_ = _pview(core, (slice(None, None, None), slice(None, r2c_, None)))
                    fnp.add(cv_, c2, out=cv_)
            elif c2 is not None:
                lift2 = _public_out(fnp.matmul, c2, U2.T, out=_win(self._pool.get('lift', (n, U2.shape[0])), m, U2.shape[0]))
                core = lift2 if core is None else _public_out(fnp.add, core, lift2, out=core)
            hp_ = self._hot_p if (OLD_HOT1 & 2) else 0
            if hp_ > 0:
                # OLD_HOT1: the old-tier part of D21 lives on columns [p:ncw] (the hot rows of the old legs are zero)
                exb = _win(bufs['t1'], m, nr)
                QcT_h = fnp.swapaxes(_pview(Qc, slice(hp_, nr, None)), 0, 1)
                if _TAG['join'] and m >= D21_LIFT_FMM_M:
                    hl_ = _win(self._pool.get('hotlift', (n, n)), m, nr - hp_)
                    smm.mn = TA_JOIN_MIN
                    smm.mm(_pview(core, (None, None)), _pview(QcT_h, (None, None)), _pview(hl_, (None, None)), smm.level(m, core.shape[1], nr - hp_, TA_JOIN_LEV))
                    smm.mn = STRASSEN_MIN
                else:
                    hl_ = fnp.matmul(core, QcT_h)
                fnp.copyto(_pview(exb, (slice(None, None, None), slice(hp_, nr, None))), hl_)
                fnp.copyto(_pview(exb, (slice(None, None, None), slice(None, hp_, None))), 0.0)
                exact = exact + (exb if ncw == nr else bufs['t1'][:m, :ncw])
            elif _TAG['join'] and m >= D21_LIFT_FMM_M and AU_LIFTW and ncw < nr:
                # AU_LIFTW: only the columns [:ncw] of the lift are read -> the pooled Z-core lift on Qc[:ncw]
                exb = _win(bufs['t1'], m, ncw)
                smm.mn = TA_JOIN_MIN
                smm.mm(_pview(core, (None, None)), _pview(fnp.swapaxes(_pview(Qc, slice(None, ncw, None)), 0, 1), (None, None)), _pview(exb, (None, None)), smm.level(m, core.shape[1], ncw, TA_JOIN_LEV))
                smm.mn = STRASSEN_MIN
                exact = exact + exb
            elif _TAG['join'] and m >= D21_LIFT_FMM_M:
                # (SDT: the pooled Strassen lift keeps its full-width plan -- a column window would only add plan keys)
                exb = _win(bufs['t1'], m, nr)
                smm.mn = TA_JOIN_MIN
                smm.mm(_pview(core, (None, None)), _pview(Qc.T, (None, None)), _pview(exb, (None, None)), smm.level(m, core.shape[1], nr, TA_JOIN_LEV))
                smm.mn = STRASSEN_MIN
                exact = exact + (exb if ncw == nr else bufs['t1'][:m, :ncw])
            elif ncw < nr:
                exact = exact + fnp.matmul(core, Qc[:ncw].T)
            else:
                exact = exact + fnp.matmul(core, Qc.T)
        if cp_exact is not None:
            cp_out = bufs['hubg'][1, 0]
            if m == nr:
                fnp.copyto(cp_out, cp_exact)
            else:
                rest = _pview(permutation, slice(m, None, None))
                ni = bufs['cp_ni']
                cx = fnp.take(_pview(left, (2 * k, 0, slice(None, None, None), slice(None, ni, None))), rest, axis=0)
                cy = fnp.take(_pview(apb4, (2 * k, 0, slice(None, None, None), slice(None, ni, None))), rest, axis=0)
                tail = fnp.empty((nr - m, nr - m), dtype=fnp.float32)
                self._sym_product(cx, cy.T, nr - m, ni, tail, min(CPRE_LEV, self._s_hub))
                cross = fnp.take(cp_exact, rest, axis=1)
                top = fnp.take(cp_exact, permutation, axis=1)
                bottom = fnp.concatenate((cross.T, tail), axis=1)
                arranged_cp = fnp.concatenate((top, bottom), axis=0)
                inverse = fnp.argsort(permutation)
                arranged_cp = fnp.take(arranged_cp, inverse, axis=0)
                fnp.take(arranged_cp, inverse, axis=1, out=cp_out)
        if m == nr:
            fnp.copyto(bufs['d21'] if ncw == nr else bufs['d21'][:, :ncw], exact)
            return bufs['d21']
        if D21_BASIS == 'active_rf':
            rows = exact if thin is None else exact + fnp.take(thin if ncw == nr else thin[:, :ncw], order, axis=0)
            rows = fnp.where(_pview(fnp.arange(ncw), (None, slice(None, None, None))) == _pview(order, (slice(None, None, None), None)), 0.0, rows)
            om = fnp.take(Q if sk is None else _pview(sk, (slice(None, None, None), slice(None, rk, None))), order, axis=0)
            Q, _ = fnp.linalg.qr(fnp.matmul(rows.T, om))
            Q, _ = fnp.linalg.qr(fnp.matmul(rows.T, fnp.matmul(rows, Q)))
        elif D21_BASIS != 'lot':
            raise ValueError(D21_BASIS)
        elif ncw < nr:
            Q = Q[:ncw]
        # Exact rows above overwrite their projection, so form only the tail.
        # Public gathers and the smaller output are ordinary billed operations.
        rest = _pview(permutation, slice(m, None, None))
        bqc = None
        for x, ys, lift in parts:
            if lift is None:
                bq = Q
            else:
                # cost1: Qc^T Q shared by both old tiers; the nested tier reads U2^T (Qc^T Q) (regrouped (U2^T Qc^T) Q)
                if bqc is None:
                    bqc = fnp.matmul(Qc.T if ncw == nr else Qc[:ncw].T, Q)
                bq = bqc if lift == 'qc' else (_pview(bqc, slice(None, U2.shape[1], None)) if (FX_UEYE and FIJA2) else fnp.matmul(U2.T, bqc))
            yr = fnp.matmul(fnp.swapaxes(_pview(ys, (slice(None, None, None), 0)), -1, -2), bq)
            xt = fnp.take(_pview(x, (slice(None, None, None), 0)), rest, axis=1)
            qp = fnp.matmul(xt, yr)
            dq = dq + fnp.sum(qp, axis=0)
        tail = fnp.matmul(dq, Q.T)
        arranged = fnp.concatenate((exact, tail), axis=0)
        dd = bufs['d21']
        fnp.take(arranged, fnp.argsort(permutation), axis=0, out=dd if ncw == nr else dd[:, :ncw])
        if _os.environ.get('D21_DEBUG', '0') == '1':
            D21_LOG.append(dict(li=li, nr=nr, exact_rows=m, kink_rows=nk, rank=rk, basis=D21_BASIS))
        return dd

    def _dslices(self, A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list, c1_list, c2_list, y_list, nr, bufs, rres, rfb, Zf_st, R1T_st, R2T_st, need_d21=True, kz=0, ka=0, FAo=None, FPo=None, Qc=None, kb=0, FA2=None, FP2=None, U2=None, apb=None, apb4=None, sb1=None, sb2=None, s_sb=0, gate_Q=None, gate_alpha=None, gate_li=0, gate_sk=None):
        """(3,) and (2,1) dslices with the M leg expressed through A, P and the thin
        residual leg: M = P*diag(s) + 3*A*diag(e) + Z L^T (the factor 3 of the
        residual is folded into Z at birth). Two dense contractions (right factors
        A and P) + O(n^2 r) thin terms.
          D21 = 2H + T2 + T3/3 + 2T4/3,  H=(A*P*w2)A^T, T2=(A*A*w2)P^T,
                T3=(P*P)M^T, T4=(M*P)P^T
          D3  = 3 sum_j w2_j A_ij^2 P_ij + sum_j M_ij P_ij^2
        Every (k,n,n) intermediate is written with out= into pooled buffers
        (F53: fresh result buffers dominate residual time).
        need_d21=False (V19, final layer): D3 only -- no left factors, no dense
        contractions, no thin right-factor terms; returns D21 = None.
        """
        k = len(w2b_list)
        n = self._nfull
        W2B = _pview(fnp.stack(w2b_list, axis=0), (slice(None, None, None), None, slice(None, None, None)))
        bufs['w2b'] = W2B
        Sb = _pview(fnp.stack(s_list, axis=0), (slice(None, None, None), None, slice(None, None, None)))
        Eb = _pview(fnp.stack(e_list, axis=0), (slice(None, None, None), None, slice(None, None, None)))
        T = _pview(bufs['t'], slice(None, k, None))
        dsf = DSF and need_d21
        ycg = False
        dn_fb = None
        dn_t2 = DN_THIN2D and dsf and gate_Q is not None and gate_alpha is not None and gate_li >= max(1, DN_THIN2D_LMIN)
        if dsf:
            # DSF: MP' = (2/3) M*P with the 2/3 folded into the thin leg (k x n x (r+2)) and the (k,1,n) vectors; PP stays in T and
            # AP in LA (unscaled) long enough to take the thin terms R / PPL from them; D3's A*A*w2 term reads the AA*w2 left factor.
            LA = _pview(bufs['lap'], (slice(None, k, None), 0))
            LP = _pview(bufs['lap'], (slice(None, k, None), 1))
            U = _pview(bufs['u'], slice(None, k, None))
            C1 = fnp.stack(c1_list, axis=0)
            C2 = fnp.stack(c2_list, axis=0)
            if DN_YCOL:
                # DN_YCOL: column rres + 1 of L is zero (birth), so M = Z L^T needs the first rres + 1 columns only (bit-identical)
                MP = _public_out(fnp.matmul, _pview(Z_st, (Ellipsis, slice(None, rres + 1, None))), fnp.swapaxes(_pview(L_st, (Ellipsis, slice(None, rres + 1, None))) * (2.0 / 3.0), -1, -2), out=_pview(bufs['mp'], slice(None, k, None)))
            else:
                MP = _public_out(fnp.matmul, Z_st, fnp.swapaxes(L_st * (2.0 / 3.0), -1, -2), out=_pview(bufs['mp'], slice(None, k, None)))
            fnp.multiply(MP, P_st, out=MP)
            fnp.multiply(P_st, P_st, out=T)
            fnp.multiply(T, Sb * (2.0 / 3.0), out=U)
            fnp.add(MP, U, out=MP)
            fnp.multiply(A_st, P_st, out=LA)
            R = fnp.einsum('kij,kj->ki', LA, C1)
            fnp.multiply(LA, Eb * 2.0, out=LP)
            fnp.add(MP, LP, out=MP)
            fnp.multiply(LA, W2B * 2.0, out=LA)
            fnp.multiply(T, Eb, out=LP)
            fnp.add(LA, LP, out=LA)
            fnp.multiply(A_st, A_st, out=LP)
            fnp.multiply(LP, W2B, out=LP)
            D3 = fnp.einsum('kij,kij->i', LP, P_st) * 3.0 + fnp.einsum('kij,kij->i', MP, P_st) * 1.5
            fnp.multiply(T, Sb * (1.0 / 3.0), out=U)
            fnp.add(LP, U, out=LP)
            fnp.add(LP, MP, out=LP)
            R = R + fnp.einsum('kij,kj->ki', T, C2)
            if DN_YCOL and gate_Q is not None and gate_alpha is not None and (gate_li >= 1):
                # DN_YCOL: PPL on the first rres + 1 columns (its last column would be (P*P) 0); the free column carries R, so the gated
                # thin einsum below yields (1/3) R^T Yk as well (Yk = Z[..., rres + 1])
                PPL = _pview(bufs['ppl'], slice(None, k, None))
                fnp.matmul(T, _pview(L_st, (Ellipsis, slice(None, rres + 1, None))), out=_pview(PPL, (Ellipsis, slice(None, rres + 1, None))))
                fnp.copyto(_pview(PPL, (Ellipsis, rres + 1)), R)
                ycg = True
            else:
                PPL = _public_out(fnp.matmul, T, L_st, out=_pview(bufs['ppl'], slice(None, k, None)))
        else:
            if DN_YCOL:
                # DN_YCOL: (bit-identical) M = Z L^T on the first rres + 1 columns
                MP = _public_out(fnp.matmul, _pview(Z_st, (Ellipsis, slice(None, rres + 1, None))), fnp.swapaxes(_pview(L_st, (Ellipsis, slice(None, rres + 1, None))), -1, -2), out=_pview(bufs['mp'], slice(None, k, None)))
            else:
                MP = _public_out(fnp.matmul, Z_st, fnp.swapaxes(L_st, -1, -2), out=_pview(bufs['mp'], slice(None, k, None)))
            fnp.multiply(MP, P_st, out=MP)
        if dsf:
            pass
        elif need_d21:
            # AP0 contains the source inputs; AP1 contains disjoint LA/LP slots.
            # T/PP/AP alias one backing, so keep P^2 in T only until LP is formed.
            LA = _pview(bufs['lap'], (slice(None, k, None), 0))
            LP = _pview(bufs['lap'], (slice(None, k, None), 1))
            fnp.multiply(P_st, P_st, out=T)
            dsx = DSF_MODE == 2
            if dsx:
                # DSF=2 (exact): the thin terms of P*P (R's second half, PPL) and of A*P (R's first half) are taken now, while the
                # products are alive; the old recompute of A*P / P*P after the feedback block disappears (same products, same order)
                C1 = fnp.stack(c1_list, axis=0)
                C2 = fnp.stack(c2_list, axis=0)
                R_pp = fnp.einsum('kij,kj->ki', T, C2)
                PPL = _public_out(fnp.matmul, T, L_st, out=_pview(bufs['ppl'], slice(None, k, None)))
            fnp.multiply(T, Sb, out=LP)
            fnp.add(MP, LP, out=MP)
            fnp.multiply(A_st, P_st, out=LA)
            if dsx:
                R = fnp.einsum('kij,kj->ki', LA, C1) + R_pp
            fnp.multiply(LA, Eb * 3.0, out=LP)
            fnp.add(MP, LP, out=MP)
            fnp.multiply(LA, W2B * 2.0, out=LA)
            fnp.multiply(T, Eb, out=LP)
            fnp.add(LA, LP, out=LA)
            fnp.multiply(A_st, A_st, out=LP)
            fnp.multiply(LP, W2B, out=LP)
            fnp.multiply(T, Sb * (1.0 / 3.0), out=T)
            fnp.add(LP, T, out=LP)
            fnp.multiply(MP, 2.0 / 3.0, out=T)
            fnp.add(LP, T, out=LP)
        else:
            fnp.multiply(P_st, P_st, out=T)
            fnp.multiply(T, Sb, out=T)
            fnp.add(MP, T, out=MP)
            fnp.multiply(A_st, P_st, out=T)
            fnp.multiply(T, Eb * 3.0, out=T)
            fnp.add(MP, T, out=MP)
        if not dsf:
            fnp.multiply(P_st, W2B, out=T)
            D3 = fnp.einsum('kij,kij,kij->i', A_st, A_st, T) * 3.0 + fnp.einsum('kij,kij->i', MP, P_st)
        if dsf and rfb > 0 and k > kz:
            # DSF feedback block: Xt carries the 1/3 (R1T/3), so the three scalar multiplies of the old block vanish; Ts = (Xt/3)*P is
            # formed once; AP is formed once into the (dead) MP buffer and R / PPL were taken earlier.
            ks = k - kz
            F1 = _pview(Zf_st, (slice(kz, None, None), slice(None, None, None), slice(None, rfb, None)))
            F2 = _pview(Zf_st, (slice(kz, None, None), slice(None, None, None), slice(rfb, None, None)))
            R1T3 = _pview(R1T_st, slice(kz, None, None)) * (1.0 / 3.0)
            R2T = _pview(R2T_st, slice(kz, None, None))
            A_s = _pview(A_st, slice(kz, None, None))
            P_s = _pview(P_st, slice(kz, None, None))
            W2B_s = _pview(W2B, slice(kz, None, None))
            if _TAG['mmfb']:
                Xt = _public_out(fnp.matmul, F1, fnp.swapaxes(R1T3, -1, -2), out=_pview(bufs['xt'], slice(None, ks, None)))
                Yt = _public_out(fnp.matmul, F2, fnp.swapaxes(R2T, -1, -2), out=_pview(bufs['yt'], slice(None, ks, None)))
            else:
                Xt = _public_out(fnp.einsum, 'kiq,kjq->kij', F1, R1T3, out=_pview(bufs['xt'], slice(None, ks, None)))
                Yt = _public_out(fnp.einsum, 'kiq,kjq->kij', F2, R2T, out=_pview(bufs['yt'], slice(None, ks, None)))
            U = _pview(bufs['u'], slice(None, ks, None))
            Ts = _pview(bufs['t'], slice(None, ks, None))
            fnp.multiply(A_s, Yt, out=Ts)
            fnp.multiply(Xt, A_s, out=U)
            fnp.multiply(U, W2B_s, out=U)
            fnp.add(Ts, U, out=Ts)
            fnp.multiply(Xt, Yt, out=U)
            fnp.add(Ts, U, out=Ts)
            D3 = D3 + fnp.einsum('kij,kij->i', P_s, Ts) * 3.0
            LP_s = _pview(LP, slice(kz, None, None))
            LA_s = _pview(LA, slice(kz, None, None))
            fnp.add(LP_s, Ts, out=LP_s)
            fnp.multiply(Xt, P_s, out=Ts)
            fnp.multiply(Ts, W2B_s, out=U)
            fnp.add(LA_s, U, out=LA_s)
            fnp.multiply(P_s, Yt, out=U)
            fnp.add(LA_s, U, out=LA_s)
            if AU_APKZ:
                # AU_APKZ: only the feedback slots [kz:k] of A*P are read (AP_s); same products, same memory
                AP_s = _public_out(fnp.multiply, A_s, P_s, out=_pview(bufs['mp'], slice(kz, k, None)))
            else:
                AP = _public_out(fnp.multiply, A_st, P_st, out=_pview(bufs['mp'], slice(None, k, None)))
                AP_s = _pview(AP, slice(kz, None, None))
            fnp.add(Ts, AP_s, out=Ts)
            Xs = _pview(Xt, (slice(None, None, None), slice(None, None, None), slice(None, nr, None)))
            GYR = _public_out(fnp.matmul, Ts, R2T, out=_pview(bufs['gyr'], slice(None, ks, None)))
            if dn_t2:
                # DN_THIN2D: the two rank-rfb sums move into the gated thin hub product (_dn_thin)
                fnp.multiply(AP_s, W2B_s, out=Ts)
                fnp.add(Ts, U, out=Ts)
                GXR = _public_out(fnp.matmul, Ts, R1T3, out=_pview(bufs['gxr'], slice(None, ks, None)))
                dn_fb = (GYR, GXR, _pview(Zf_st, slice(kz, None, None)))
                D21_fb = None
            else:
                fnp.matmul(GYR, fnp.swapaxes(F2, -1, -2), out=Xs)
                D21_fb = _public_out(fnp.sum, Xs, axis=0, out=bufs['d21fb'])
                fnp.multiply(AP_s, W2B_s, out=Ts)
                fnp.add(Ts, U, out=Ts)
                GXR = _public_out(fnp.matmul, Ts, R1T3, out=_pview(bufs['gxr'], slice(None, ks, None)))
                fnp.matmul(GXR, fnp.swapaxes(F1, -1, -2), out=Xs)
                fnp.sum(Xs, axis=0, out=bufs['t1'])
                fnp.add(D21_fb, bufs['t1'], out=D21_fb)
        elif rfb > 0 and k > kz:
            ks = k - kz
            F1 = _pview(Zf_st, (slice(kz, None, None), slice(None, None, None), slice(None, rfb, None)))
            F2 = _pview(Zf_st, (slice(kz, None, None), slice(None, None, None), slice(rfb, None, None)))
            R1T = _pview(R1T_st, slice(kz, None, None))
            R2T = _pview(R2T_st, slice(kz, None, None))
            A_s = _pview(A_st, slice(kz, None, None))
            P_s = _pview(P_st, slice(kz, None, None))
            W2B_s = _pview(W2B, slice(kz, None, None))
            if _TAG['mmfb']:
                Xt = _public_out(fnp.matmul, F1, fnp.swapaxes(R1T, -1, -2), out=_pview(bufs['xt'], slice(None, ks, None)))
                Yt = _public_out(fnp.matmul, F2, fnp.swapaxes(R2T, -1, -2), out=_pview(bufs['yt'], slice(None, ks, None)))
            else:
                Xt = _public_out(fnp.einsum, 'kiq,kjq->kij', F1, R1T, out=_pview(bufs['xt'], slice(None, ks, None)))
                Yt = _public_out(fnp.einsum, 'kiq,kjq->kij', F2, R2T, out=_pview(bufs['yt'], slice(None, ks, None)))
            U = _pview(bufs['u'], slice(None, ks, None))
            Ts = _pview(bufs['t'], slice(None, ks, None))
            fnp.multiply(A_s, Yt, out=Ts)
            fnp.multiply(Xt, A_s, out=U)
            fnp.multiply(U, W2B_s * (1.0 / 3.0), out=U)
            fnp.add(Ts, U, out=Ts)
            fnp.multiply(Xt, Yt, out=U)
            fnp.multiply(U, 1.0 / 3.0, out=U)
            fnp.add(Ts, U, out=Ts)
            D3 = D3 + fnp.einsum('kij,kij->i', P_s, Ts) * 3.0
            if need_d21:
                LP_s = _pview(LP, slice(kz, None, None))
                LA_s = _pview(LA, slice(kz, None, None))
                fnp.add(LP_s, Ts, out=LP_s)
                fnp.multiply(Xt, P_s, out=Ts)
                fnp.multiply(Ts, W2B_s * (1.0 / 3.0), out=Ts)
                fnp.add(LA_s, Ts, out=LA_s)
                fnp.multiply(P_s, Yt, out=U)
                fnp.add(LA_s, U, out=LA_s)
                AP = _public_out(fnp.multiply, A_st, P_st, out=_pview(bufs['mp'], slice(None, k, None)))
                AP_s = _pview(AP, slice(kz, None, None))
                fnp.multiply(Xt, P_s, out=Ts)
                fnp.multiply(Ts, 1.0 / 3.0, out=Ts)
                fnp.add(Ts, AP_s, out=Ts)
                Xs = _pview(Xt, (slice(None, None, None), slice(None, None, None), slice(None, nr, None)))
                GYR = _public_out(fnp.matmul, Ts, R2T, out=_pview(bufs['gyr'], slice(None, ks, None)))
                fnp.matmul(GYR, fnp.swapaxes(F2, -1, -2), out=Xs)
                D21_fb = _public_out(fnp.sum, Xs, axis=0, out=bufs['d21fb'])
                fnp.multiply(AP_s, W2B_s, out=Ts)
                fnp.add(Ts, U, out=Ts)
                fnp.multiply(Ts, 1.0 / 3.0, out=Ts)
                GXR = _public_out(fnp.matmul, Ts, R1T, out=_pview(bufs['gxr'], slice(None, ks, None)))
                fnp.matmul(GXR, fnp.swapaxes(F1, -1, -2), out=Xs)
                fnp.sum(Xs, axis=0, out=bufs['t1'])
                fnp.add(D21_fb, bufs['t1'], out=D21_fb)
            else:
                D21_fb = None
        else:
            D21_fb = None
        dsx = DSF_MODE == 2 and need_d21
        if not (dsf or dsx):
            AP = _public_out(fnp.multiply, A_st, P_st, out=_pview(bufs['mp'], slice(None, k, None)))
            PP = _public_out(fnp.multiply, P_st, P_st, out=T)
            C1 = fnp.stack(c1_list, axis=0)
            C2 = fnp.stack(c2_list, axis=0)
        Yk = _pview(Z_st, (slice(None, None, None), slice(None, None, None), rres + 1))
        if not (dsf or dsx):
            R = fnp.einsum('kij,kj->ki', AP, C1) + fnp.einsum('kij,kj->ki', PP, C2)
        D3 = D3 + fnp.einsum('ki,ki->i', R, Yk)
        if not need_d21:
            return (D3, None)
        if not (dsf or dsx):
            PPL = _public_out(fnp.matmul, PP, L_st, out=_pview(bufs['ppl'], slice(None, k, None)))
        fnp.multiply(P_st, _pview(C2, (slice(None, None, None), None, slice(None, None, None))), out=MP)
        fnp.multiply(A_st, _pview(C1, (slice(None, None, None), None, slice(None, None, None))), out=T)
        fnp.add(T, MP, out=T)
        fnp.multiply(T, _pview(Yk * (2.0 / 3.0), (slice(None, None, None), slice(None, None, None), None)), out=T)
        fnp.add(LP, T, out=LP)
        if gate_Q is not None and gate_alpha is not None and (gate_li >= 1):
            ncw_t = nr
            self._au_nc = -1
            if AU_THINW and self._sdt_ord and SDT_COLW and (not bufs.get('cp')):
                # AU_THINW: D21 columns [nc:] are zeroed and the demand reads thin[:, :ncw] only -> thin on Z rows [:ncw]
                ncw_t = _sdt_win(nr, int(float(fnp.sum(fnp.less(gate_alpha, -_SDT_COL_L(gate_li))))))
                self._au_nc = ncw_t
            if dn_t2 and ycg:
                thin = self._dn_thin(PPL, Z_st, dn_fb, nr, k) if ncw_t >= nr else self._au_dn_thin(PPL, Z_st, dn_fb, nr, k, ncw_t)
            elif ycg and ncw_t < nr:
                # AU_THINW (DN_YCOL einsum path)
                thin = fnp.einsum('kiq,kcq->ic', PPL, _pview(Z_st, (slice(None, None, None), slice(None, ncw_t, None)))) * (1.0 / 3.0)
                if D21_fb is not None:
                    fnp.add(thin, _pview(D21_fb, (slice(None, None, None), slice(None, ncw_t, None))), out=thin)
            elif ycg:
                # DN_YCOL: R sits in PPL's last column -> the einsum carries (1/3) R^T Yk
                thin = fnp.einsum('kiq,kcq->ic', PPL, Z_st) * (1.0 / 3.0)
                if D21_fb is not None:
                    fnp.add(thin, D21_fb, out=thin)
            else:
                fnp.matmul(R.T, Yk, out=bufs['t1'])
                fnp.multiply(bufs['t1'], 1.0 / 3.0, out=bufs['t1'])
                thin = fnp.einsum('kiq,kcq->ic', PPL, Z_st) * (1.0 / 3.0)
                fnp.add(thin, bufs['t1'], out=thin)
                if D21_fb is not None:
                    fnp.add(thin, D21_fb, out=thin)
            D21 = self._d21_demand(bufs, apb4, ka, kb, k, n, nr, sb1, sb2, Qc, U2, s_sb, gate_Q, gate_alpha, gate_li, sk=gate_sk, thin=thin)
            if self._au_t1pk is not None:
                if self._au_t1pk[0] == 'pair':
                    # AU_T1PK pair: re-zero the packed (free) factor columns of the narrow tier-1 pairs (LOT and later layers read the plain factors)
                    for js_, a_, b_, jt_, c_, d_ in self._au_t1pk[2]:
                        fnp.copyto(_au_pair(self._au_t1pk[3], jt_, slice(c_, d_, None)), 0.0)
                self._au_t1pk = None
            nc_ = self._sdt_nc
            ncw_ = nc_ if SDT_COLW and (not bufs.get('cp')) else nr
            if ncw_ < nr:
                Dw = D21[:, :ncw_]
                fnp.add(Dw, thin[:, :ncw_], out=Dw)
            else:
                fnp.add(D21, thin, out=D21)
            if nc_ < nr:
                # SDT: D21 columns of the nearly-dead neurons (alpha < -SDT_COL) are zero
                fnp.copyto(D21[:, nc_:], 0.0)
            return (D3, _zero_diag(D21))
        elif ka > 0:
            inner = None
            smm = self._smm
            if ka > kb:
                r1 = sb1.shape[2]
                inner = _win(self._pool.get('inner', (1, n, r1)), nr, r1)
                smm.mn = _TAG['old_min']
                _sl = slice(2 * kb, 2 * ka, None)
                smm.hub(_pview(bufs['lap4'], _sl), sb1, inner, smm.level(nr, n, r1, s_sb), pad=(_pview(bufs['lap4p'], _sl), None, self._pool.get('inner', (1, n, r1))))
                smm.mn = STRASSEN_MIN
                inner = inner[0]
            if kb > 0:
                r2_ = sb2.shape[2]
                inner2 = _win(self._pool.get('inner2', (1, n, r2_)), nr, r2_)
                smm.mn = _TAG['old_min']
                _sl = slice(None, 2 * kb, None)
                smm.hub(_pview(bufs['lap4'], _sl), sb2, inner2, smm.level(nr, n, r2_, s_sb), pad=(_pview(bufs['lap4p'], _sl), None, self._pool.get('inner2', (1, n, r2_))))
                smm.mn = STRASSEN_MIN
                if FX_UEYE and FIJA2:
                    # FX_UEYE: inner2 U^T with U = [I; 0] = inner2 in the first r2_ columns (zero-padded)
                    if inner is None:
                        inner = _win(self._pool.get('lift', (n, U2.shape[0])), nr, U2.shape[0])
                        fnp.copyto(_pview(inner, (slice(None, None, None), slice(None, r2_, None))), inner2[0])
                        fnp.copyto(_pview(inner, (slice(None, None, None), slice(r2_, None, None))), 0.0)
                    else:
                        iv_ = _pview(inner, (slice(None, None, None), slice(None, r2_, None)))
                        fnp.add(iv_, inner2[0], out=iv_)
                else:
                    lift = _public_out(fnp.matmul, inner2[0], U2.T, out=_win(self._pool.get('lift', (n, U2.shape[0])), nr, U2.shape[0]))
                    inner = lift if inner is None else _public_out(fnp.add, inner, lift, out=inner)
            if ka < k and STRASSEN_HUB > 0:
                D21 = self._hub2(bufs, apb4, ka, k, n, nr)
                if _TAG['join']:
                    smm.mn = TA_JOIN_MIN
                    smm.mm(_pview(inner, (None, None)), _pview(Qc.T, (None, None)), _pview(bufs['t1'], (None, None)), smm.level(nr, inner.shape[1], nr, TA_JOIN_LEV))
                    smm.mn = STRASSEN_MIN
                    fnp.add(D21, bufs['t1'], out=D21)
                else:
                    fnp.add(D21, _public_out(fnp.matmul, inner, Qc.T, out=bufs['t1']), out=D21)
            else:
                D21 = _public_out(fnp.matmul, inner, Qc.T, out=bufs['d21'])
                if ka < k:
                    D21 = D21 + fnp.einsum('kij,kcj->ic', _pview(LA, slice(ka, None, None)), _pview(A_st, slice(ka, None, None))) + fnp.einsum('kij,kcj->ic', _pview(LP, slice(ka, None, None)), _pview(P_st, slice(ka, None, None)))
            fnp.matmul(R.T, Yk, out=bufs['t1'])
            fnp.multiply(bufs['t1'], 1.0 / 3.0, out=bufs['t1'])
            fnp.add(D21, bufs['t1'], out=D21)
        elif STRASSEN_HUB > 0:
            D21 = self._hub2(bufs, apb4, 0, k, n, nr)
            fnp.matmul(R.T, Yk, out=bufs['t1'])
            fnp.multiply(bufs['t1'], 1.0 / 3.0, out=bufs['t1'])
            fnp.add(D21, bufs['t1'], out=D21)
        else:
            D21 = fnp.einsum('kij,kcj->ic', LA, A_st) + fnp.einsum('kij,kcj->ic', LP, P_st) + fnp.einsum('ki,kc->ic', R, Yk) * (1.0 / 3.0)
        t1 = fnp.einsum('kiq,kcq->ic', PPL, Z_st)
        fnp.multiply(t1, 1.0 / 3.0, out=t1)
        D21 = _public_out(fnp.add, D21, t1, out=bufs['d21'])
        if D21_fb is not None:
            fnp.add(D21, D21_fb, out=D21)
        return (D3, _zero_diag(D21))
if __name__ == '__main__':
    from local_engine import build_mlp, compare_against_monte_carlo
    mlp = build_mlp(width=1024, depth=16, seed=0)
    compare_against_monte_carlo(Estimator(), mlp, seed=0)

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
import os as _os
K31TWO_B2 = float(_os.environ.get('K31TWO_B2', '1.2'))
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
PATHB_ON = _os.environ.get('PATHB', '1') == '1'   # path_patch: K22 path birth (ARC augment, t/18249 sec 3.4) as feed c1 += KB dG w3
PATHB_KB = float(_os.environ.get('PATHB_KB', '1.0'))
PATHB_NC = _os.environ.get('PATHB_NC', '0') == '1'
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


def _tdp_win2(nr, cnt):
    """tdpcap_patch: _sdt_win with the left-out cap at half of the rows (TDP join window only)"""
    return min(nr, -(-(nr - min(cnt, nr // 2)) // SDT_GRAN) * SDT_GRAN)
HZ_K = tuple(int(x) for x in _os.environ.get('HZ_K', '0,0,0,0,0,96,0,32,96,96,128,192,160,128,192,160').split(','))
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
NLHZ = _os.environ.get('NLHZ', '1') == '1'   # nlhz_patch: non-linear HZ rider head (MLP on the rider slabs, layers 2..15)
_NLHZ_P = {'M': [[0.00387779437, 0.00159529771, 0.802928746, 0.33793661, -0.00717844581, -0.000688535685, -0.00723257475, 0.00154444145, 5.1553794e-05, -0.000246414304, -8.68847292e-06, -0.000860710745, -0.560064137, -0.0185984988, 0.0005813241, -0.0298701301, -9.2615719e-06, -0.00347814104, 0.0112363053, -5.13385839e-06, 0.016214963, 0.016130561, 0.00192876987, 7.38074959e-05, -0.000327777612, 0.0679427013, 0.0421702228, -0.00944583584, 1.2762574, 9.28508996e-07, 0, 0, 0, 0, 0, -0.000273480458, 0.000546255556], [0.00477399398, 0.00388853042, 0.626329839, 0.348117113, -0.00415387889, -0.000264652219, -0.00418634852, 0.00148839643, 8.7595603e-05, -0.000429607491, -2.99772387e-06, -0.000460873707, -0.410856754, -0.015213185, 2.6050413e-05, -0.0294199884, 1.41791861e-05, 0.00118179061, -0.238076851, 6.49433059e-05, 0.0216098763, 0.0214890968, 0.00237527629, 0.000162879151, -0.000714677095, 0.136437759, 0.0699489489, -0.00300621544, 1.90054238, 5.57766725e-06, 0.570624352, 0.187369674, -1.1920064e-05, -1.38332052e-06, 1.11066561e-06, -0.000434502261, 0.00177728198], [0.00499859639, 0.00619498547, 0.491488516, 0.319115758, -0.000489360478, -5.04067793e-06, -0.000493931002, 0.00136431307, 0.000134898131, -0.000533444574, -5.11152471e-08, -5.18228444e-05, -0.0580870584, -0.00201056828, -0.0155291287, -0.00405288767, -0, 0.00207507145, -0.451781869, -2.39902311e-05, 0.0262900069, 0.0261304136, 0.00276994868, 0.000315265846, -0.00111833587, 0.172250479, 0.0840628669, 0.925827622, 2.48403716, 0, 0.445851564, 0.124734297, 4.25705093e-06, 1.30075568e-05, -3.24621709e-07, 0.0105252611, 0], [0.0048915362, 0.00785919372, 0.382850438, 0.284755975, 0.000742698554, 0.00249314681, 0.000768725469, 0.00120900723, 0.000181683776, -0.000570478325, 1.54174068e-05, 7.79671755e-05, 0.106406704, 0.00324368686, 0.0410224907, 0.00831893366, -0, 0.00551689835, -0.639784157, 5.17941808e-05, 0.0305126347, 0.0303432103, 0.00313979527, 0.000531650323, -0.00152532104, 0.245269001, 0.112056747, 1.22806346, 3.02017903, 0, 0.3599976, 0.0883208886, 5.95242e-06, -1.7001588e-06, -2.13336334e-05, 0.010679476, 0.00512398593], [0.00509930542, 0.00893084612, 0.334439158, 0.272345483, 0.00136107591, 0.00370139745, 0.00140239333, 0.00117675704, 0.000208726648, -0.000631279137, 2.57944594e-05, 0.000141989483, 0.21594353, 0.00728122564, 0.115732387, 0.018213205, 0.00915929954, 0.144525126, -0.808959126, 0.00353034935, 0.0343010277, 0.0341448002, 0.00350372144, 0.00069603388, -0.00193434278, 0.301908851, 0.134849682, 1.97378814, 3.48354793, 0.00597824575, 0.297080338, 0.0650396943, 2.00639097e-05, 8.56998304e-06, 0.00023378426, 0.0153178778, 0.259659261], [0.00469114818, 0.00962032843, 0.268173754, 0.23800379, 0.00170077826, 0.00529119745, 0.00175265281, 0.00103921175, 0.000242366514, -0.000614793913, 4.67726095e-05, 0.000187236452, 0.316025138, 0.0114144878, 0.181246251, 0.0290690493, 0.00864701718, 0.164595425, -0.961290181, 0.00441939058, 0.0353998318, 0.0355532505, 0.00384918135, 0.000973797461, -0.00233279425, 0.371720374, 0.164608583, 2.29503655, 3.90395212, 0.00735143758, 0.250514597, 0.0498875268, 1.90299761e-05, -5.60941317e-06, 0.000239664951, 0.0143599072, 0.276674062], [0.00426326646, 0.00961619895, 0.212169781, 0.208094388, 0.00174533715, 0.00687709497, 0.00175617542, 0.000896768761, 0.000257579755, -0.000570704869, 6.35236429e-05, 0.000189269413, 0.3450059, 0.0132878479, 0.219546646, 0.0316831991, 0.00396637945, 0.182476982, -1.1001904, 0.00499092182, 0.0382880792, 0.0391190089, 0.00418540463, 0.00127020944, -0.00271599251, 0.406573236, 0.192186356, 2.7270124, 4.26569223, 0.00381479575, 0.214706063, 0.0391642824, 7.12351584e-06, 1.8250732e-05, 0.000198850088, 0.0161987431, 0.273759902], [0.00408470165, 0.00945071597, 0.184539884, 0.192031667, 0.00187917077, 0.00710268458, 0.00186611479, 0.000843018643, 0.00027904904, -0.000565813039, 6.27274931e-05, 0.000203529678, 0.395931274, 0.0162938535, 0.277660489, 0.0364442281, 0.00691243075, 0.294724941, -1.23006499, 0.0084741842, 0.0407867581, 0.0423276164, 0.00451895827, 0.00156488712, -0.00308824913, 0.438946873, 0.218595192, 3.23677826, 4.5807519, 0.0152265839, 0.185821921, 0.0313212723, 1.9667681e-05, -1.37303368e-05, 0.000267682772, 0.018670395, 0.404287964], [0.00363063929, 0.00912200101, 0.146046996, 0.167541996, 0.00166839198, 0.00772473728, 0.00162592425, 0.000715589675, 0.000275542465, -0.000504355179, 7.77047462e-05, 0.000186964753, 0.38502413, 0.0171202179, 0.278815359, 0.0369743146, 0.00475148857, 0.301438212, -1.34966695, 0.00909951236, 0.0415208824, 0.0435977578, 0.00483988412, 0.00191211537, -0.00346070901, 0.486765444, 0.249481201, 3.54157495, 4.89626026, 0.0131865097, 0.162917286, 0.0255562402, 3.97108379e-05, 1.27388012e-05, 0.000301197724, 0.0182259325, 0.39500314], [0.00319716171, 0.00826533046, 0.117912464, 0.14612712, 0.00145563949, 0.00744800176, 0.00140564633, 0.000616078381, 0.00026343955, -0.000452470966, 7.70819533e-05, 0.000162902521, 0.35271731, 0.016835494, 0.260915816, 0.0345195755, 0.00232824567, 0.33165127, -1.46045065, 0.00970339868, 0.0435145162, 0.0464231148, 0.00515259802, 0.0022408776, -0.00381566258, 0.521761477, 0.279782563, 3.81138492, 5.14750385, 0.00661541289, 0.144094959, 0.0211636685, 1.90647934e-05, 2.68468102e-05, 0.00026393935, 0.0129992217, 0.381660372], [0.00299977511, 0.00802424923, 0.102914564, 0.134615377, 0.00150609983, 0.00739471894, 0.00141552021, 0.000571935496, 0.000266656541, -0.000432750734, 6.77003554e-05, 0.000168277824, 0.378462404, 0.0192994345, 0.291230679, 0.0366698019, 0.0033109677, 0.464676887, -1.56680024, 0.0133945327, 0.0450721271, 0.0492037088, 0.00546967704, 0.00255986396, -0.0041497089, 0.53859508, 0.309593976, 4.12684584, 5.3759222, 0.0185788088, 0.128712565, 0.0177925229, 2.07237608e-05, -1.77188558e-05, 0.000291035394, 0.014884227, 0.486968428], [0.00264744228, 0.00732566509, 0.0839202255, 0.119376086, 0.00128885731, 0.00697181048, 0.00118978019, 0.00049251935, 0.000247310294, -0.000384066399, 6.65984553e-05, 0.000149193045, 0.350518644, 0.0188892949, 0.27945444, 0.0333478637, 0.00095778075, 0.497328609, -1.66378427, 0.0136607355, 0.0449637659, 0.0500744246, 0.00576938223, 0.00288569042, -0.00447578728, 0.553064406, 0.338852525, 4.45301294, 5.58329391, 0.00927928369, 0.115242757, 0.0150574176, 2.44842249e-05, 1.17631616e-05, 0.000283350586, 0.0126018561, 0.465547502], [0.00228552078, 0.0065968791, 0.0671855584, 0.103392728, 0.0010760416, 0.00639745034, 0.000981679885, 0.000415510149, 0.000222674862, -0.000332162308, 6.31122821e-05, 0.000126056621, 0.303772777, 0.0173601452, 0.247284353, 0.0288777296, 5.03914052e-05, 0.511525154, -1.75720572, 0.0136421015, 0.0459145717, 0.052041851, 0.00606077258, 0.00322374492, -0.00480125239, 0.562426865, 0.366035491, 4.67951965, 5.73772001, 0.000994303962, 0.103998974, 0.0128529817, -1.69226514e-05, -4.23567399e-05, 0.000258292013, 0.0118941078, 0.439093977], [0.00404543383, 0.0114854351, 0.135949001, 0.156435892, 0.00261179125, 0.0115166623, 0.0028089718, 0.000896063459, 0.000334176671, -0.000711340399, 0.000140886565, 0.000381214486, 0, 0, 0, 0, 0, -0.0120002069, -1.87945592, -0.000257507199, 0.0476825349, 0.0476739854, 0.00656637643, 0.00263767922, -0.00536067132, 0, 0, 0, 0, 0, 0.0947146341, 0.0111093111, 1.24681374e-05, -4.01058642e-05, -2.67937467e-05, 0, 0]], 'IS': [[-95.6123886, -56.0641975, 1.68122613, 1.55421233, 49.8233681, 73.8103943, 49.3297806, 857.407654, 15893.5967, 4526.09277, 5012.05371, 415.827271, 0.637432158, 19.2331944, 4.85666227, 11.9607849, -412.469727, 0.508433163, 7.09118032, 31.6771526, 481.326569, 438.988922, 4984.15186, 16477.7656, 8041.85059, 58.7463188, 133.760376, 2.39227438, 3.76045156, 165.454941, 0, 0, 0, 0, 0, 9.6133852, 0.206775844], [-87.5992355, -40.9029465, 1.53101635, 1.77404439, 41.5536461, 39.993412, 40.8298225, 637.033386, 7935.74609, 2056.52441, 2814.35181, 373.573486, 0.419498682, 11.3225012, 4.45349264, 5.87699938, -265.932648, 0.407915086, 5.82882261, 22.0295372, 345.853088, 274.844116, 3961.24121, 7952.92041, 4545.0249, 25.9720917, 67.7703705, 1.96576321, 2.68560219, 90.0044861, 17.800827, 47.5471802, 1629.28894, 1320.93188, 100.28672, 9.80696201, 0.175845787], [-84.5848312, -36.2785568, 1.52919626, 1.99091399, 38.6857224, 32.7922478, 37.8916702, 545.839966, 4509.5293, 1346.14746, 2107.64966, 361.790161, 0.327099144, 9.53765488, 0.911743283, 4.75574207, -200.975525, 0.345443606, 5.01935387, 16.6024952, 269.681366, 190.610107, 3192.57031, 4553.06885, 3041.25269, 18.4822159, 46.7992821, 1.7329011, 2.04369593, 60.2993927, 21.0944958, 63.158123, 1247.12659, 1128.20898, 85.7997894, 9.78974438, 0.180602312], [-83.7908936, -34.4757347, 1.59957659, 2.18019986, 37.7026558, 30.1154423, 36.8499184, 503.196198, 2955.86328, 1048.61841, 1782.81921, 360.25531, 0.276922554, 7.30411625, 0.709490776, 3.44473267, -173.031723, 0.302160621, 4.4284358, 13.2324343, 223.325974, 143.350876, 2634.15381, 2996.90894, 2196.1062, 11.7479162, 29.9314156, 1.30053413, 1.65518034, 43.4466095, 24.0984974, 78.6870499, 1034.36755, 992.968201, 76.3299255, 10.1873198, 0.175358683], [-82.4960632, -33.004818, 1.75706959, 2.40102434, 36.8255348, 28.4378834, 35.9092026, 495.131439, 2492.15942, 908.676941, 1532.22559, 354.072632, 0.241290957, 6.10440016, 0.443186164, 2.80563688, -44.5583305, 0.292473793, 4.06528091, 11.82796, 194.682083, 111.52623, 2235.14478, 2282.04443, 1690.72119, 9.02303791, 24.1775684, 1.24446154, 1.41568136, 7.30629158, 26.6598511, 93.3267517, 946.654907, 919.205872, 70.6187897, 10.2796249, 0.175459266], [-83.1890335, -32.9351463, 1.86509514, 2.58582783, 39.2721214, 29.4198704, 38.747879, 477.245056, 1928.99866, 803.740479, 1417.57019, 357.984161, 0.221336976, 5.10663366, 0.388053089, 2.34526777, -41.6098785, 0.266254216, 3.72977281, 10.2060957, 178.75531, 95.3778992, 1860.50708, 1778.09656, 1346.57568, 6.93804312, 17.9633274, 1.21351147, 1.18036103, 6.00317812, 29.9302692, 115.108521, 871.023132, 862.028992, 65.7442703, 10.6467609, 0.177174151], [-85.0113297, -33.226078, 1.99543703, 2.75531626, 39.5093231, 28.6359177, 39.809433, 468.105927, 1588.61267, 739.173096, 1354.50452, 365.406158, 0.20884496, 4.48593807, 0.33507061, 2.20699668, -54.9195518, 0.243780315, 3.46138835, 9.11810017, 157.714966, 78.2932205, 1546.29907, 1393.9906, 1092.6731, 5.65497875, 14.1512251, 0.866539598, 0.997838318, 7.69404745, 32.2543488, 131.718048, 856.468018, 809.515381, 61.8122063, 11.0194979, 0.188049868], [-85.6372299, -33.4582901, 2.14687395, 2.92190886, 39.8048286, 28.1690025, 40.7976608, 465.74353, 1380.86353, 696.785583, 1295.40723, 369.342133, 0.198112875, 4.02083683, 0.285789013, 2.08317757, -39.5484581, 0.234942719, 3.21676779, 8.6286087, 140.951126, 66.7414322, 1322.66711, 1128.9436, 905.014404, 4.89009094, 11.6507235, 0.854822516, 0.873851061, 4.58694792, 34.2429314, 149.862366, 818.914978, 767.849731, 58.713295, 11.4203815, 0.196347132], [-88.1581955, -34.1840248, 2.29648972, 3.09365964, 42.416153, 29.2866402, 44.095192, 464.139954, 1219.92566, 667.898621, 1277.46106, 378.626068, 0.192235067, 3.62527561, 0.267963767, 1.93953419, -44.7928238, 0.218628913, 3.02785325, 7.86581182, 132.787415, 61.5005951, 1150.71704, 948.352844, 761.617126, 4.28081846, 9.68320465, 0.800536573, 0.770341337, 4.72677231, 37.5300293, 174.665695, 787.196899, 742.99353, 56.2111435, 11.8709612, 0.204777882], [-91.2176361, -35.1121826, 2.45854902, 3.26191306, 43.6256714, 29.3152237, 46.2873116, 465.130707, 1113.8656, 647.43811, 1274.50391, 390.554352, 0.189663738, 3.33131051, 0.25818038, 1.86713696, -51.2069511, 0.20604676, 2.87293744, 7.37719727, 121.379456, 53.8725281, 991.029236, 804.239624, 650.173218, 3.69175935, 8.00630093, 0.70536232, 0.679371178, 4.94368744, 40.7077446, 201.786179, 802.369019, 721.529968, 54.1926804, 12.400898, 0.214964494], [-93.2901535, -35.8648491, 2.61785865, 3.43523979, 44.2591972, 29.0611172, 48.2536621, 464.958313, 1028.23401, 629.300476, 1256.89673, 396.918854, 0.184627682, 3.06372762, 0.241592586, 1.8306483, -41.9069176, 0.200394347, 2.72347498, 7.28505802, 113.385788, 46.3198891, 845.496948, 671.100952, 551.303101, 3.30444407, 6.59780025, 0.596474767, 0.599779427, 3.71424437, 43.5332565, 225.864105, 788.138062, 694.603271, 52.2652664, 12.5573378, 0.22663784], [-95.8447647, -36.7259369, 2.79179025, 3.60479212, 47.3268738, 30.1977024, 52.9620857, 468.743713, 972.028748, 622.325989, 1263.61951, 410.564117, 0.184685573, 2.89962125, 0.233970791, 1.8515079, -51.420742, 0.190613911, 2.60254145, 6.94209003, 106.181519, 41.5686073, 726.155518, 575.315735, 474.639404, 2.97392559, 5.5808444, 0.50676465, 0.525707066, 4.34821558, 46.0546494, 250.985733, 768.410461, 673.681213, 51.0945244, 12.9629526, 0.242105916], [-100.049232, -38.0968742, 2.97262287, 3.8005414, 49.6207123, 30.8317547, 56.7511406, 473.276581, 924.106934, 616.673462, 1293.16467, 426.734467, 0.187295124, 2.78499985, 0.231475383, 1.88731682, -67.1955795, 0.181349888, 2.48636651, 6.65011311, 103.428757, 37.1968689, 628.505798, 491.784393, 413.463379, 2.73906255, 4.732337, 0.46411708, 0.468270481, 5.32861757, 49.5511665, 283.820343, 793.686768, 649.136597, 49.7203979, 13.4767456, 0.259514272], [-91.2535858, -34.2522583, 3.05846167, 4.4543395, 50.2739029, 33.66119, 48.3525276, 456.245605, 1168.13269, 571.891602, 970.288818, 353.462646, 0, 0, 0, 0, 0, 0.248470411, 2.51442838, 6.88845968, 99.9427872, 43.2512932, 805.225769, 566.10321, 454.307495, 0, 0, 0, 0, 0, 51.1238213, 306.372772, 788.957886, 630.689026, 48.6242905, 0, 0]], 'W0': [[-0.286916405, 0.0814359114, -0.171515182, 0.344385326, 0.111456469, 0.0462083966, 0.133299664, 0.162447855, 0.0869078487, -0.0641538352, 0.103415847, -0.138378024, 0.0229955856, 0.0255667679, -0.02417567, -0.157022536, -0.0298248809, 0.167134255, -0.0862716362, -0.0337990671, -0.0346468277, 0.201914489, 0.0835123807, 0.102936454, 0.14775826, 0.00422901614, -0.116718106, 0.125563413, 0.339504063, -0.0967945233, 0.186863601, 0.0556979366, -0.0343041383, -0.0331220217, 0.0385099202, -0.193940952, 0.0695181862, 0.169887453, 0.121896848, -0.0885837153, -0.158536583, -0.0743617788, 0.0576444492, 0.0429104976, 0.0099173123, 0.0214461125, 0.19617404, -0.0724828765, -0.000563000212, -0.111823902, -0.0582662709, 0.159676954, 0.19481656, -0.123055033, -0.0335187502, -0.18892175, -0.00285245199, -0.0442808643, 0.207851827, 0.196827769, 0.117579378, -0.104293562, -0.257337481, 0.0205201581], [-0.117663547, -0.0068550366, 0.123619862, -0.160225481, 0.0403996371, 0.0199421905, -0.0279437695, 0.0109851249, -0.0323332846, 0.267754376, 0.288702786, 0.222858235, -0.126047656, 0.161569193, -0.0679385066, 0.183386609, -0.0028689953, -0.0368064083, -0.0932076126, 0.00770624494, 0.0157288555, 0.176327661, 0.19552137, 0.121501088, 0.0735615641, 0.132576779, 0.0372693539, -0.232664362, -0.0896869972, 0.162356123, 0.0530800037, -0.0117047466, -0.091848217, -0.102273531, 0.0517592169, -0.0483447909, -0.137173206, -0.215628758, -0.258680552, 0.0585895106, 0.248716533, -0.12541987, 0.220706746, 0.282278568, -0.184742093, -0.0420485809, -0.141202599, -0.0491759777, -0.0237847026, 0.0397328474, -0.0155430948, -0.0917836651, 0.0414903276, -0.148743242, -0.0245254189, -0.0987136215, 0.135905236, 0.00488044275, 0.113457322, 0.0433024094, 0.495245755, 0.158061862, 0.0476827137, -0.0986748934], [-0.102028236, 0.0362714306, 0.163012639, 0.0455113687, -0.00385159906, 0.00309431017, 0.0239916388, -0.18949914, 0.0752977207, 0.0689817294, 0.0459700041, -0.1912622, -0.0576441661, 0.071113795, -0.0198920593, -0.0918804705, 0.253961831, -0.00118238782, -0.100353666, -0.0718600079, 0.103924371, -0.0771211311, 0.00639446499, -0.0860418901, -0.0965069905, -0.00283540692, -0.04816081, -0.211050272, 0.121475637, -0.0992356837, -0.0239421986, 0.0497001112, 0.0173272714, 0.0430744924, 0.0374178961, 0.0674288198, -0.122498952, 0.0396784209, 0.139839634, -0.0829491764, 0.0916188359, 0.112225652, -0.101574942, 0.0678097606, 0.0268344898, 0.0983720571, -0.0799754858, -0.171901107, -0.13263154, -0.2382202, 0.259778172, 0.0111567248, -0.0708825365, 0.0619351305, -0.103999324, 0.0885907933, -0.0389869027, -0.0103869047, 0.0438022539, -0.182107523, -0.308350861, 0.0207046233, 0.0027512887, -0.082814008], [-0.032386709, -0.0334226824, 0.282971084, -0.0826090723, -0.19543457, -0.610093534, 0.0247303285, -0.0375215337, 0.633480906, 0.200681537, -0.143703476, 0.0902508944, -0.885174274, 0.162088141, 0.168372735, 0.103030376, 0.0334953517, 0.251022547, 0.190967843, 0.0271236226, 0.0445049293, 0.0267755762, 0.233275563, 0.113143124, -0.180451065, 0.123100631, -0.111800663, -0.196740881, 0.260060579, 0.147422343, -0.0902862772, -0.0252631493, -0.0323038921, 0.147779971, 0.00418894924, 0.0748314038, -0.224068135, -0.217545331, -0.285902977, 0.0559552014, 0.256201804, 0.151606977, 0.292746276, -0.174787387, 0.107239895, 0.00465249596, -0.010666593, -0.091942206, 0.255462021, 0.0386017673, 0.140638709, 0.172271356, -0.170074195, 0.0382979587, -0.0547491275, -0.162315741, 0.118669897, 0.0982740447, 0.0697442368, -0.333395213, -0.285968721, -0.0953842551, 0.239126354, 0.0764578208], [-0.0750127509, -0.0999605656, 0.253567994, -0.0742810741, -0.206460848, 0.0758967549, 0.157161221, 0.00467772409, -0.121401846, 0.150918782, -0.0812050402, 0.178355262, -0.261982709, 0.073603861, 0.130147308, 0.148488402, 0.00241599116, 0.300832093, 0.125727251, 0.190982506, -0.116121888, 0.0579395331, -0.0644290596, -0.083378762, 0.00915177166, 0.0838520154, -0.157815039, 0.0259621553, -0.0125757717, -0.0195986051, 0.184719354, 0.0910583213, -0.147638336, 0.0835392997, 0.0910407454, -0.00443936931, -0.223472133, -0.155608028, 0.0536998361, 0.103548683, 0.0784065798, -0.0760399401, 0.201249227, 0.177416667, -0.215706363, -0.0539387316, -0.10723459, -0.112914138, 0.192050248, 0.107363202, 0.152229071, 0.213134289, -0.124990739, -0.0401005819, -0.0573071577, 0.106298052, 0.103398837, 0.152653217, 0.0939941108, 0.00825203676, 0.213269055, 0.0398866087, 0.106694251, -0.0860390365], [-0.130869627, -0.0567621887, -0.0192627255, 0.235461488, 0.0274137612, 0.421120405, 0.215008855, 0.0735204071, -0.165282294, -0.0771147683, -0.2389981, -0.151895195, 0.459511548, 0.0455132574, -0.194072768, -0.0313415192, -0.0286380444, 0.381752372, -0.213044122, 0.297873199, 0.165898919, -0.246211901, -0.0241436791, -0.071109131, -0.0339701921, -0.0428783409, -0.125426918, 0.0520478003, 0.412298083, -0.192460984, 0.0771089047, -0.0740233809, 0.186520308, -0.18663758, -0.0146604003, -0.173450515, 0.308550268, 0.214421943, 0.230464369, 0.00247354107, -0.159407258, 0.0532143489, -0.286412627, -0.269650459, -0.150222167, -0.0230171084, 0.0454476811, -0.147954822, -0.0895207375, -0.209162474, -0.158820152, 0.159004197, -0.0666436777, -0.18807824, -0.244127825, -0.23579362, -0.488943756, -0.0411524437, 0.119168997, 0.015156257, -0.254884839, -0.500800371, -0.500159085, 0.177017465], [-0.103614405, -0.0757459477, 0.0307237841, 0.0536627211, 0.0351855978, -0.192572579, 0.179853275, 0.215279534, 0.0777207538, 0.0776425824, 0.0720194131, 0.197747439, -0.0954186544, -0.0270625688, 0.0687715858, -0.0677442476, 0.0658729151, 0.332123607, 0.0367069952, 0.123069726, -0.0379419737, 0.050161209, 0.183170155, 0.0646056905, 0.0882479697, 0.146475717, 0.0438200794, -0.0413571335, 0.139903367, -0.058849141, 0.107088715, 0.0956249312, 0.126887709, 0.121149965, -0.103264138, -0.156192914, 0.0639235824, -0.0655656233, -0.188241497, -0.0721894279, 0.16296497, 0.014748916, 0.350268841, -0.154682443, 0.107621752, -0.0112420479, -0.0647681132, 0.0339128971, 0.227394491, -0.0496447682, -0.0265970789, 0.119155355, -0.111211568, 0.0874027908, 0.150493771, 0.000190455525, 0.0508088693, 0.0483383387, -0.0784854814, 0.0661916435, 0.0807889253, -0.0560292192, 0.15667364, -0.0618722551], [-0.0371618755, -0.065623261, -0.030624982, 0.0752954036, -0.132507414, -0.0148406411, -0.0218416005, -0.272150517, -0.0639538392, 0.0652733669, -0.0373223722, -0.171382606, 0.0567369424, 0.0927349031, 0.0405650623, -0.0641416386, 0.0412163474, -0.0483582951, -0.199179053, -0.0410788096, 0.0773501247, -0.0555732958, 0.111221835, 0.0384065285, -0.194455862, -0.0444193445, 0.026527578, -0.203410208, 0.053093221, 0.0837365687, -0.0885688812, 0.157641366, -0.0414631292, -0.118538998, -0.000200635957, 0.0910231397, -0.0498553328, -0.0935729221, 0.0541547872, 0.066553846, 0.0328868479, 0.0789146349, -0.00557512324, 0.0660799071, 0.00122447475, 0.0438664928, -0.115187928, -0.0606704168, 0.101695247, -0.308731884, 0.111662567, 0.0358340107, -0.0131986476, -0.100181483, -0.0337543376, 0.137866512, 0.048824627, 0.132682189, 0.128782496, -0.207650706, -0.362812877, -0.134209439, -0.0544710681, -0.104323067], [0.0383074135, -0.0707978904, -0.0795932412, 0.00342137786, -0.107119396, 0.0947794169, -0.0611382462, -0.214485437, 0.0439989157, -0.0494525097, -0.283713907, 0.074979946, -0.0265423991, 0.0328703858, -0.060282778, 0.0317670815, 0.0459214188, -0.0983253419, -0.0429399461, 0.0154699516, -0.221041679, -0.0775370225, 0.143228456, -0.0830790326, -0.0683986098, -0.101403743, -0.203673318, -0.267773271, 0.0275626015, -0.108153783, 0.106378622, -0.0920100808, -0.171895236, 0.0286872201, -0.0051354696, -0.0600221194, 0.0181995295, 0.00521857059, 0.254003197, 0.165943593, 0.123410903, -0.0951166749, -0.0842475221, 0.171985552, 0.107414179, -0.290143788, 0.0710087642, -0.0170318484, -0.0519083552, -0.170862108, 0.0234802496, 0.0661935508, -0.204947308, -0.0239538811, 0.0506577604, 0.192946896, -0.047429204, 0.248366237, 0.0205909591, -0.358583987, -0.260838389, -0.0151928794, 0.0308969188, 0.0866072997], [0.0661574155, -0.104488328, -0.158531636, -0.126728937, 0.0219497755, 0.154032767, 0.149151355, 0.129601404, -0.163529113, 0.0456676371, 0.268677771, 0.110769302, -0.127657801, 0.0329213254, 0.121168897, -0.0194127578, 0.0561179966, -0.0462739989, -0.0219480097, -0.0201380681, -0.093640767, 0.10822919, 0.0357405245, -0.0149242189, 0.3444736, 0.0455260575, -0.115320638, 0.253413051, 0.0436320752, 0.054674454, 0.0677308589, -0.0756034702, 0.000364842912, 0.156522155, -0.152666971, -0.0806326196, -0.0335869454, -0.257145077, 0.114541449, -0.000233448955, 0.196226761, 0.0102236401, 0.110653549, -0.0915616155, 0.126717493, 0.0627476797, -0.211695567, 0.121779375, 0.0689034835, 0.155077055, -0.0325625278, -0.0703405663, 0.00265550707, 0.142785162, 0.00412340369, 0.200387806, 0.0256776027, 0.108925924, 0.153660506, 0.260234386, 0.316615552, 0.208445787, 0.161869958, -0.134571433], [0.10281682, -0.0460933559, -0.234992281, 0.140180185, -0.150663674, 0.0268999878, 0.0310908277, -0.0936671272, -0.019485468, 0.0583101362, -0.221086755, -0.170961529, -0.0697304979, -0.0334959626, -0.0295736156, -0.0697431862, 0.0179524086, -0.0745980367, -0.0689823329, -0.0580924414, -0.0368746892, 0.218523681, 0.122667149, -0.00401210785, -0.102103695, 0.00494175451, -0.0812229216, 0.0717576817, 0.127460808, 0.0701656342, -0.204312801, -0.0886587203, -0.0404019989, 0.00020933518, 0.044787012, 0.10638728, 0.148024291, 0.114612028, 0.223426893, -0.0256973188, -0.00715304492, -0.00151897816, 0.0503005795, -0.0935885385, 0.126326978, 0.0474740788, 0.137280717, 0.0821284801, -0.0438504107, -0.0359211527, -0.00426691119, 0.190313086, -0.12138585, 0.0260360427, -0.0491033308, 0.0164364092, -0.184999257, -0.0409013964, 0.116683483, 0.183636859, -0.124437191, -0.140866876, -0.096979849, -0.00110032759], [0.0474472679, 0.0442374684, 0.0783937722, -0.128290907, -0.109697215, -0.17107515, 0.102607653, 0.146481022, 0.177461609, -0.0220808517, -0.052491542, 0.112653673, -0.286586404, 0.0964113474, -0.017540995, -0.0236306041, 0.0511269532, 0.306511849, 0.0507537797, 0.159492552, 0.047326263, 0.103665993, 0.167365924, 0.0122856665, 0.16771172, 0.0679452196, -0.0531254187, 0.0126511585, 0.00823899731, 0.190036729, -0.0678914711, 0.00471867528, 0.0442533568, 0.0250893496, 0.0731335729, -0.0642253309, -0.168070704, 0.0200308803, -0.102102168, 0.061479196, 0.131927282, 0.129035488, 0.22012347, -0.103449255, -0.0204121508, -0.0123094413, 0.00140473119, 0.132556319, 0.183767319, -0.147834003, -0.00273925369, 0.0730325133, 0.0900817811, 0.0465570949, 0.109825343, -0.036785353, 0.199763715, -0.127323613, -0.123835757, 0.122686401, 0.0988439098, 0.179722011, 0.161703065, -0.01385435], [0.107117034, -0.0765623003, 0.251124382, -0.103798114, -0.0895892829, -0.0537022613, -0.00228555663, 0.0755741894, 0.0278620813, 0.0242246166, 0.0376017615, 0.0221847147, -0.212440804, 0.0673689768, -0.0437704325, -0.0827002153, 0.248598695, 0.181191057, 0.127131879, 0.228082106, 0.0291856397, 0.190663576, -0.0821726322, -0.00435485551, 0.00929334294, 0.057327494, -0.169055909, 0.0267466195, 0.0296370443, 0.17752786, 0.117740929, 0.104566209, 0.00361873186, 0.0545219183, -0.0128987618, -0.0309606232, -0.0234247316, -0.11247357, 0.114257053, -0.0682081655, 0.0361062028, -0.011532967, 0.154996574, -0.0212284643, -0.0115379672, -0.142871961, 0.144803345, -0.0348480456, -0.0353444815, 0.0248347651, -0.0537628047, 0.318005204, -0.0369126387, 0.073710002, -0.00333038112, -0.0642097071, 0.0104359677, -0.137676209, -0.0881909728, -0.119558387, -0.0355902798, 0.0815970078, -0.0450337976, -0.0865515843], [0.129809007, -0.0116356704, 0.109696962, -0.0380097032, 0.015970625, -0.121931911, -0.0107301315, 0.378561288, 0.123458512, -0.010162551, 0.0911884457, 0.030781718, -0.0968731791, -0.106262974, -0.119784407, -0.0336863175, -0.0435968302, 0.0410462543, 0.0297836941, 0.000873542042, -0.200768381, 0.103627965, 0.149303287, -0.149180293, -0.0998194143, 0.00832927786, -0.0235325992, -0.0655623004, 0.115644448, 0.0326744057, -0.128204688, 0.215379283, -0.132783949, 0.0595081449, -0.149457335, 0.140875965, -0.0779619217, -0.124318071, 0.0151579296, -0.0933726355, 0.247763634, -0.0783724934, 0.0261432864, -0.124623775, 0.126286969, -0.0762075037, 0.139569491, 0.198052615, 0.256764323, 0.0538714714, -0.159587204, -0.051093854, -0.153892085, 0.0752850994, -0.0649832413, -0.0139162838, 0.00215374143, 0.106494576, -0.0635717437, -0.293924302, -0.0778175518, 0.0085509168, -0.0518189222, -0.0668544397], [0.00144044228, 0.0103598544, 0.147333726, -0.078439191, -0.0333183445, -0.0305051692, 0.0306065213, 0.190302446, 0.101114355, 0.0381378345, -0.00444097817, -0.0378457718, -0.0844679549, -0.131556332, -0.143697068, 0.127555385, 0.0207742006, 0.234902054, 0.00772846211, 0.188369647, -0.00923931133, -0.0227941815, 0.067189835, 0.0309017431, -0.0587793551, -0.0562707186, -0.0485013649, -0.0232789163, 0.103985198, -0.0216166954, 0.000501982693, -0.0278070718, -0.0534453765, 0.127331793, 0.10506653, 0.076454699, -0.174613059, -0.136136621, -0.0554029606, 0.0486364923, 0.111898839, -0.0581929795, 0.164403319, -0.126414254, -0.0775311813, -0.139309883, 0.0205942858, 0.0515423454, 0.130071551, -0.0954063833, -0.0825360268, 0.0946460292, 0.0780836195, 0.0103382375, -0.0740525872, -0.0393910483, -0.0631385744, -0.0579749085, 0.00896727573, -0.229739487, -0.10138572, -0.0148046631, 0.0135222841, -0.0290926751], [0.0302364901, 0.138553441, 0.0801236778, 0.132900134, 0.0509864204, -0.0620557182, 0.0514776595, 0.250680804, 0.0724672526, 0.137553141, -0.0182956215, -0.0227157641, -0.0785629824, 0.00338622648, 0.0678620562, 0.0843941122, 0.0713137761, 0.231820226, 0.00688497955, 0.123491541, -0.0313316584, 0.0401406288, 0.0759819448, 0.0531970821, -0.039764978, 0.0510354452, -0.115563624, -0.0781662241, 0.222054541, -0.0247674584, -0.087677516, 0.0615846924, 0.0654680803, -0.0267912354, 0.0898264349, 0.109871872, -0.00666189753, 0.120987199, 0.0078098001, 0.0465122871, 0.0826365501, -0.0710866749, 0.157877877, -0.140818894, 0.0140020642, -0.167486519, 0.251294076, 0.313203037, 0.0501642935, -0.0290183909, -0.0593396798, 0.279877841, -0.181282043, -0.0821699053, -0.0524613857, -0.00987824332, -0.00979056209, -0.00621216372, 0.0716145858, -0.00891111419, -0.0707080066, -0.00568482699, -0.0856840014, 0.121582359], [0.0114880353, -0.136886805, 0.262990564, -0.0298695881, -0.0832806751, 0.0375039876, 0.242282704, -0.151088208, -0.020373147, 0.302415252, 0.114299066, 0.112549968, 0.0184793156, 0.00756728277, 0.0209349655, -0.0322671086, 0.0426443219, 0.0453130789, 0.121369898, 0.00714835804, -0.0244486574, 0.0904408693, -0.100827299, -0.168902874, 0.0480885059, 0.149996355, 0.0825766474, 0.146024257, 0.161902785, 0.0376848541, 0.0456540212, 0.0166511778, 0.051015269, 0.0572621711, -0.0032227647, -0.103015535, 0.229357928, 0.126491547, -0.028504286, 0.0746807978, -0.106154323, -0.0668578744, -0.0193604417, -0.0470699146, -0.0944997966, -0.187231973, 0.441052705, 0.0497175641, -0.018698521, -0.00981631875, -0.0218462367, -0.0449679755, 0.14907065, 0.244059533, -0.0605637133, 0.0262596179, -0.00795704219, -0.0452393144, -0.0527886413, 0.0669252425, 0.082353048, 0.0477671064, 0.174019128, 0.0442658067], [-0.142338097, -0.0689700842, 0.0757543594, 0.0378273018, -0.225780338, 0.230389088, 0.0889151469, -0.0294820312, 0.14155601, 0.0723585263, 0.673692763, 0.0165664684, -0.0745208636, -0.0627967194, 0.0147434445, -0.0036983313, 0.167457134, -0.257704318, -0.137445122, 0.0591249727, -0.122309521, -0.196397468, -0.0239937752, -0.124585651, -0.338755935, 0.10920509, 0.240237936, -0.0383003652, -0.0490791649, -0.182445168, 0.0270131994, 0.0558177158, -0.249758944, -0.0568215251, 0.0517397262, 0.098714605, -0.175251335, 0.0732520521, 0.105264552, -0.0455468111, -0.000397479715, 0.0493234843, 0.0875360966, 0.113979712, -0.0636754036, 0.139445558, -0.0441925637, -0.124225631, 0.00934334472, 0.0621561557, 0.0955577344, 0.0562623814, -0.0411440507, 0.0568339564, -0.0599888526, -0.103254952, -0.192947462, -0.0984113961, 0.00991064496, -0.00506635942, 0.138502628, 0.877636015, 0.130535498, -0.0102675892], [-0.16805087, -0.0294019822, -0.139136583, 0.0434594899, 0.121103451, 0.0993580148, -0.046015624, 0.140203759, -0.283739656, 0.0741629824, -0.247887149, 0.0210251659, 0.0517066307, 0.29980582, -0.210847899, -0.230639353, -0.208827853, 0.0378898121, 0.187593654, -0.121718325, -0.0710930824, -0.155721471, 0.1652693, 0.237627476, 0.0642854422, 0.00161210704, 0.215395525, 0.0437181219, 0.0498565286, -0.0871496275, -0.221683457, -0.13685824, -0.213763431, 0.191953436, -0.194922581, -0.141856208, 0.0845647603, 0.150800794, 0.0923352391, -0.087597847, 0.128631368, 0.0161128063, 0.0605739318, 0.0786277577, 0.0340649262, -0.0326473825, -0.00223998865, -0.0793351904, 0.186801597, -0.239752352, -0.224765703, -0.227487847, -0.132742047, 0.0650198907, 0.337616771, -0.169236019, 0.015161288, -0.00384728936, 0.223696351, -0.0207583718, 0.216672793, 0.102232881, -0.190104708, 0.0728323683], [-0.0302851107, -0.00304471864, 0.0585489161, -0.0310665425, -0.0395556577, 0.0103058647, -0.00870135985, 0.0165947247, 0.13757953, -0.0249820799, 0.209922403, -0.0556489788, 0.0376161225, 0.0275120623, 0.0806704313, 0.0791797042, -0.209230423, -0.0924361572, -0.12040817, 0.197861567, -0.0406633504, 0.0880820304, 0.0975967273, 0.073037371, -0.13228035, -0.0182381626, 0.0102836434, 0.0930780321, 0.0700799003, -0.124499962, -0.0253755301, 0.0543938167, 0.047108639, -0.0301183145, 0.0324837863, -0.0305942371, -0.188354447, 0.181582794, 0.168018743, -0.0266954675, -0.200986072, 0.142951667, 0.0754890889, -0.028776804, 0.00933437608, -0.0991553441, -0.0601447001, -0.239392444, 0.0273313783, 0.0824440047, 0.0513416529, 0.0182623286, 0.13368243, 0.151574433, -0.0221308451, 0.200662374, 0.108182728, -0.0224991459, -0.0526210777, 0.0303836875, 0.038775418, 0.243185461, -0.0905018076, 0.0327972397], [0.279943198, 0.00397880841, 0.14101629, 0.130370587, -0.114905968, 0.0632564202, 0.0314297192, 0.119727373, 0.00882357452, -0.194111332, -0.156953737, 0.135301366, -0.0419904999, -0.0205188952, -0.14196825, 0.0831364244, -0.102942742, -0.0287008472, 0.0887488946, -0.0787533894, -0.208457589, 0.0743332952, 0.00221908349, 0.127672821, -0.11724744, 0.0229531713, -0.069228828, 0.0752029046, -0.123120576, 0.257150143, -0.145896181, -0.0456304625, -0.0557874106, 0.17120488, 0.144628286, 0.477457643, -0.0317534283, 0.0885636657, 0.0409626774, 0.0820107386, -0.100592755, -0.153715193, 0.0586684197, 0.150604129, -0.0545458794, -0.31625089, 0.223677173, 0.00758198649, 0.174502611, -0.14409259, -0.159296334, -0.0653385147, -0.170237064, 0.0389004312, 0.122239225, 0.320341647, -0.172620803, 0.191884339, 0.0860085189, -0.0789600313, 0.020826783, -0.0258014817, 0.256037563, 0.111090615], [0.0418969616, 0.17059122, -0.208329454, 0.572217822, 0.579674661, -0.792427003, 0.320081651, -0.44556278, 0.296907216, 0.170472071, 0.109358273, 0.0855054483, 0.281589597, -0.172611058, 0.224919587, 0.245012075, 0.207033992, 0.146885097, -0.246817008, 0.157700136, 0.483672559, -0.117262729, -0.286456078, -0.268464416, -0.104905367, 0.171269149, 0.150911495, -0.0799850076, -0.197859094, 0.248599932, -0.33656621, -0.376778394, 0.34284851, -0.201496065, 0.307758063, 0.0329318345, 0.242046654, 0.548407793, -0.594163358, -0.209356159, -0.121446475, -0.425037175, -0.0571723431, -0.103973866, -0.280854255, 0.608703852, 0.0132919513, -0.194873497, -0.00135105266, 0.280890405, 0.356746078, 0.107764319, 0.479686797, -0.27775681, -0.0542099029, -0.328783482, -0.148155898, -0.480775118, -0.396428823, 0.190359369, -0.00105719862, 0.0225712266, 0.465409309, 0.360821754], [0.0579271875, -0.142141476, 0.0379573554, -0.114318103, -0.231166989, 0.616209686, -0.258329779, -0.0800477043, -0.259919643, 0.0312028788, -0.128082693, 0.15637283, 0.0843431577, 0.0578716286, -0.0206456091, -0.0425405279, -0.0401446559, 0.25970459, -0.0793940872, -0.0773437247, -0.392673194, 0.0175294373, 0.0503742509, 0.187249362, -0.0667391568, 0.140184596, -0.323870212, 0.264005244, 0.125487149, -0.000424995087, 0.339207649, 0.104757585, -0.316708624, -0.0847846642, -0.146466464, 0.136115611, -0.331360906, -0.236059174, 0.263051659, 0.318306535, -0.128149703, -0.0745545775, -0.0942421779, -0.0543156117, -0.0686347634, -0.475905865, -0.0940048844, 0.106521472, 0.186321795, -0.345423311, -0.156704098, 0.219464168, -0.0879078731, -0.0163794756, -0.00264325575, 0.188788012, -0.311366349, 0.399305046, 0.245714813, 0.100512087, -0.157468677, -0.0248636473, 0.170437559, -0.183656052], [0.0932859182, -0.134770632, -0.139706045, -0.326797277, -0.201907277, 0.462386519, 0.30282405, -0.232875541, -0.150470391, -0.125900075, -0.171309248, 0.21376957, -0.212076798, -0.0164698642, -0.0652003884, 0.0840335861, -0.102711454, -0.0485651121, 0.0530815721, 0.0513718575, -0.534519255, 0.105721354, 0.057352256, -0.343083888, -0.21228084, -0.0801403001, -0.288860768, -0.21807003, -0.106977664, -0.0956146121, 0.23324272, -0.200991377, -0.254081428, -0.11608728, -0.0832211599, -0.0213047843, -0.0741500109, -0.200495183, 0.171394378, 0.360752881, -0.0386353172, -0.364839017, -0.0136084668, 0.164199159, -0.12905702, -0.367670387, 0.004573524, -0.328080744, -0.0606157482, -0.0210658908, -0.286887348, 0.0913818926, -0.412768096, -0.161168292, 0.157159552, 0.381165981, 0.0507794581, 0.461177707, 0.0934275761, 0.0915259197, 0.306226671, 0.0761334896, 0.0214213934, -0.129811317], [0.142433971, -0.404909194, -0.271912396, -0.653440475, -0.543232203, 0.700051606, 0.028349204, 0.14240922, -0.297578275, 0.420533627, 0.20377399, 0.180429772, -0.204428494, 0.433631718, 0.122028865, -0.434017062, 0.0460802652, -0.141901568, 0.0904752687, -0.125890985, -0.479244649, 0.0931933373, 0.438449979, 0.326223463, 0.481383055, 0.318570226, -0.407722771, 0.0724211633, 0.183223978, -0.238581046, -0.112225622, 0.274219483, -0.30880177, -0.126949772, -0.217429429, 0.0334885493, -0.076937668, -0.649517834, 0.26515618, 0.147762492, 0.558412552, 0.355946153, -0.261569828, 0.0875570253, 0.0608512312, -0.0306407195, -0.51773572, 0.105496965, 0.269666344, -0.492282033, -0.0725794435, -0.273191899, -0.0246870257, 0.202861071, -0.141519517, 0.112672701, 0.0129769566, 0.54692179, 0.251461536, -0.127688199, 0.169545069, 0.133762419, 0.229685172, -0.366455078], [0.0817707106, -0.0567882806, 0.235093653, 0.0405127183, -0.144603446, -0.272673368, 0.00559463492, -0.277795851, 0.24274829, 0.177988425, 0.138855517, 0.0923734754, -0.0310257003, -0.327116817, 0.0936544836, 0.114456952, 0.123932488, -0.0826229006, 0.0648955479, 0.138026476, 0.141871169, 0.0696143359, -0.0945484936, -0.206356674, 0.264041901, -0.106036402, 0.00846824236, 0.261728048, -0.0735955909, 0.0329933502, 0.0741154477, 0.22182633, 0.018511219, -0.310289145, 0.0993842706, -0.172101095, 0.0932807177, -0.0220382083, 0.0373097472, 0.0519942157, -0.251545995, -0.0367371961, 0.237114117, -0.0507923998, -0.102929808, -0.0128742233, 0.27805987, 0.413221657, -0.278289318, -0.0112536391, 0.0322745964, 0.108117379, 0.0322807655, 0.229714677, -0.595411003, 0.159292117, 0.12355382, 0.0208792444, -0.0365996398, 0.505672514, 0.0781745613, 0.0328171663, -0.119288705, 0.197989747], [-0.0874371901, 0.070159629, -0.284063488, 0.0828620642, 0.0123855639, 0.311981618, -0.137019962, 0.515067577, -0.775355041, -0.149510562, 0.308868825, 0.213455692, -0.246063292, 0.997725844, -0.15502578, -1.03538728, -0.306454808, -0.168521434, -0.686991692, -0.140527546, -0.392571539, -0.0404370092, 0.918674886, 0.567993104, 0.0935113654, 0.203229159, 0.0855369866, -0.376043588, -0.120061457, -0.923472166, -0.0100545194, 0.520037174, -0.761263192, 0.419923544, -0.276723623, -0.59965688, 0.0910783932, -0.240672812, -0.422059298, -0.1826469, 0.323590696, 0.957194626, -0.714744151, -0.294094652, 0.37453115, 0.0367321223, -0.1649189, 0.191118941, 0.558636367, 0.312505156, -0.068011336, -0.530985415, -0.132171288, 0.372569114, 0.293705106, -0.171228081, 0.0286684185, -0.192228645, -0.70755583, -1.39396608, 0.0740873069, 0.219502345, 0.116052799, -0.978675127], [0.105144568, -0.0302411448, -0.150253132, -0.128127307, -0.0618763231, -0.00621721521, -0.0129302572, 0.00981893484, -0.290704548, -0.0737747699, 0.0318141207, -0.16562289, -0.0651132762, 0.027730858, -0.100574091, -0.13732262, -0.140397981, 0.0461747572, -0.0691726133, 0.087210007, 0.0918084309, -0.0775244236, 0.215700492, 0.242008999, 0.223591745, -0.0900756866, -0.0280443747, 0.035509523, -0.0726354718, 0.19764255, 0.0654530749, -0.0710135326, 0.191735998, -0.101039514, 0.169056818, -0.248055786, 0.0583716482, -0.0104419533, -0.184573546, 0.180832893, 0.101352155, -0.0170142967, -0.131384432, 0.0046930192, -0.127309546, -0.0819556266, 0.0252664275, 0.02630632, -0.12369331, -0.307375818, -0.0674047768, -0.0451534949, 0.0126321083, 0.323402941, 0.00169913506, 0.0177624431, -0.178662613, 0.0888529792, -0.0468648076, -0.284308583, 0.247203261, 0.0562610589, -0.0238542389, -0.433633298], [0.26220721, 0.0331557505, 0.0312496498, 0.145210043, 0.0399261825, -0.191631287, -0.310560018, -0.349062473, 0.723919272, -0.0711297393, -0.199336171, -0.186193079, 0.240317374, -0.551687598, 0.28193754, 0.61710006, 0.299834251, -0.161101803, 0.00590419769, 0.148789763, 0.384881407, -0.0184817929, -0.723817527, -0.12736015, -0.284514189, -0.0970842913, 0.0246236622, 0.462394357, 0.0741577446, 0.617788255, 0.0565344095, -0.627862453, 0.548739195, -0.239642054, 0.69785732, 0.0689227581, -0.249691457, 0.0834818259, 0.39008972, 0.477784157, -0.340245098, -0.689784229, 0.111256689, 0.159779847, -0.0198586509, -0.120649315, 0.0408143923, -0.183454067, -0.123174481, -0.187201574, 0.190234348, 0.52822113, 0.0849561915, -0.326041639, 0.139449537, -0.163926587, -0.301207244, -0.0801695958, 0.141441777, 1.09905326, -0.0979030505, -0.131626755, 0.0856401548, 0.407112807], [0.13847062, -0.0237364843, 0.0653404966, -0.126094714, -0.0411418639, 0.0143775744, -0.126599774, 0.121931121, 0.0971822366, -0.0467143469, 0.257794768, -0.154178232, -0.00539780175, 0.0505996346, 0.166602626, -0.0219120234, -0.224239185, 0.085172303, 0.0829723105, 0.0245944001, 0.0542579778, -0.0634953901, -0.00689895917, 0.128932789, -0.188301563, 0.0503503121, 0.0802644789, 0.0747866407, -0.00013820325, -0.0712085068, 0.00484386645, 0.0217439365, 0.075996086, -0.0667835474, 0.154572442, -0.122955509, 0.146615416, 0.343378395, 0.00291403523, 0.00598794967, 0.0344311632, -0.0579015836, -0.0503193624, -0.0762427822, 0.0632554814, 0.0257229358, 0.05449735, -0.211146355, -0.0936029479, 0.0209482685, -0.0787397176, -0.0192473624, -0.117899641, 0.105453931, -0.0560013503, -0.0344044529, 0.0351333693, 0.0414850488, 0.00118951488, 0.00742515316, 0.109957933, -0.138133153, -0.119787738, -0.0272483099], [0.103289165, 0.23010157, 0.0663657263, 0.347735345, 0.301307887, -0.220401466, -0.117006667, 0.00276673422, -0.111527704, -0.0147533901, 0.0601127893, 0.0168178882, 0.0521499813, 0.325784117, 0.0182954445, -0.369450927, 0.0687856376, 0.063488245, -0.0976271108, -0.142202258, 0.0982376933, 0.0424831919, 0.206392318, 0.221816003, 0.139845774, 0.0600588657, 0.137830436, 0.0405003913, 0.121817574, -0.326732755, -0.416126937, -0.0234611854, 0.107192159, 0.0939581394, -0.101186842, -0.0482689925, 0.118475966, 0.217080817, -0.250929177, -0.441466331, -0.0522936657, 0.367691755, 0.0774467438, -0.216367632, 0.0639288202, 0.00596573576, 0.268481463, 0.0158021357, 0.0445222668, -0.261535019, -0.0421249568, -0.0931351185, 0.121382259, 0.299579859, 0.326507509, -0.182076916, 0.25462687, -0.396952748, -0.191712484, 0.108694829, 0.0542013012, 0.15208298, -0.15491797, -0.0573891178], [0.161464781, -0.548467159, -0.169035241, -0.388934821, -0.619272113, 0.187881783, 0.2117441, -0.135327205, 0.156318292, -0.144491598, -0.0755340904, -0.0516583845, -0.0672917217, -0.458719671, 0.01703454, 0.387912899, -0.0749783292, 0.0418608934, 0.0522094667, 0.255944729, -0.204937205, 0.0538597852, -0.243853688, -0.251972675, -0.0879475325, -0.0378621556, -0.310559988, -0.0512974858, -0.0997921154, 0.254317969, 0.353084773, -0.0388973281, -0.0142223043, -0.0270318873, 0.0812580064, 0.0588491894, -0.0651202425, -0.446674943, 0.174740851, 0.293004721, 0.105284944, -0.185992822, 0.13443166, 0.277513862, -0.0732817277, 0.153051034, -0.0847229883, -0.00890213717, -0.202951238, 0.391660839, 0.0673028231, 0.00412941445, -0.0830601677, -0.267077416, -0.413532674, 0.308262348, -0.0102485782, 0.368372113, 0.129509628, -0.134477839, -0.0210535247, -0.0931012556, 0.125465125, 0.0761274472], [0.13420099, -0.00295924954, 0.22186777, -0.0382539518, 0.0522738025, 0.00266025378, 0.161918133, -0.0037756185, 0.0102003254, -0.014442076, -0.0910691321, -0.0601744503, -0.00840706006, 0.0908194333, 0.0227397047, -0.0345234536, 0.0174434278, 0.0462771095, -0.0942105874, -0.00739211589, -0.0134449787, -0.011795029, 0.0470058098, -0.0962065235, -0.0183490701, -0.00320945866, 0.0716218278, 0.0846748203, -0.0582113676, 0.0902541801, -0.0417038612, 0.097792834, 0.00921024755, -0.0845458508, 0.0222447552, 0.176136345, -0.0194361564, -0.0367583148, -0.0725437775, 0.143109128, -0.106840871, 0.00251377281, 0.104643121, 0.0550987571, 0.141215578, -0.0633099005, -0.0947596356, 0.038517911, -0.110438101, 0.0356704369, 0.0592211224, 0.102481566, -0.201210305, 0.0452512838, 0.0973040536, 0.182112128, 0.164202958, 0.0158419758, 0.0485955924, -0.0174208172, 0.0795213804, -0.0191589613, 0.0481539592, 0.0347840674], [-0.0417351648, -0.175328091, 0.0629997998, 0.00864437222, -0.029291546, -0.00997225661, -0.0496460162, 0.0654368624, -0.0139164943, 0.0596320219, 0.000149284184, 0.069031097, 0.00606315257, 0.00044998611, 0.00103055744, -0.00413917331, 0.00333289476, -0.086744301, 0.0128997518, 0.0763733387, 0.0497951582, -0.00856768247, -0.0126817813, -0.00440858956, 0.0179372244, 0.116381206, 0.150194794, 0.0178324804, 0.0448636636, -0.0185014028, -0.0546576567, -0.160287216, -0.0680511147, 0.0377303995, -0.111904159, -0.0376643874, 0.0908642039, -0.0483919606, -0.0164283402, 0.00512777176, 0.183086187, 0.0070506772, 0.0319924727, -0.100934133, -0.00994826201, 0.0931610167, -0.0404443331, -0.0629412681, -0.0118773859, -0.0268601328, 0.0468467623, -0.0607286468, -0.000332107476, -0.00466529187, -0.0089912815, 0.142557293, -0.00976151787, 0.0269172136, -0.0829733685, 0.0393605381, -0.0132290553, 0.0598203987, -0.178134874, 0.0238828175], [-0.0340210497, 0.00571921421, -0.00713905413, -0.0806435272, 0.0791023821, -0.0468876846, 0.0673869923, -0.0297801588, 0.0358374231, -0.0283030849, -0.0707301348, -0.0467136726, 0.00156378467, 0.0310140848, 0.0176647082, -0.00217573764, -0.0204577427, -0.0465759933, 0.0529906563, -0.00748607889, -0.0498807989, 0.101884872, 0.0162280742, 0.0996489823, -0.0327051915, -0.130030423, -0.103031568, -0.0541997142, 0.018604923, -0.0257642306, 0.00153095403, -0.0293596163, -0.0932691246, 0.0297884438, 0.0582610965, -0.112288766, -0.0584641732, -0.0166243482, -0.0100759594, 0.0319162533, 0.019859029, -0.0387368873, 0.0795840621, 0.0490919724, 0.0328563824, -0.00686239125, 0.0204309821, 0.0997771323, -0.0544178411, -0.0262021068, 0.0469446406, -0.0122576766, 0.107895046, -0.0439379737, -0.0386076272, 0.0528387651, -0.104887389, -0.0267432686, -0.0578821748, 0.0159380343, -0.056105148, 0.134029865, 0.0691359043, -0.00515582412], [-0.0717211366, 0.209139064, 0.0645045042, 0.115157083, -0.139270693, -0.0340497941, -0.0895282924, 0.0514323525, -0.0356811397, 0.122300357, 0.129589096, -0.0482805595, -0.0261768308, 0.0289083086, -0.0169762876, -0.0688721836, -0.0264753681, 0.05552442, -0.0264967661, 0.038375251, -0.0810819864, 0.0208781958, -0.0250285473, 0.0928500816, 0.0688154846, -0.069767192, 0.0473770946, 0.0475306064, 0.0412114114, -0.140454113, -0.0908640027, -0.0755328611, -0.0406163298, -0.0393066332, -0.0403689817, -0.0128016118, 0.0431053936, -0.0104905982, -0.0591661781, -0.0966920629, 0.0163495485, 0.0207073428, 0.0824290067, -0.0714908317, -0.00575266313, 0.0760118812, -0.00776680605, 0.0972203165, 0.0576168075, -0.00891679898, 0.0055856551, -0.0106507419, 0.210561112, 0.280412287, 0.113383278, -0.036136087, 0.186559498, -0.0125647439, -0.0147836218, -0.0376229547, 0.0815342814, -0.0105565907, 0.010165371, -0.118282147], [0.284470052, -0.0391328931, -0.131349251, -0.226309076, -0.0594564714, -0.0962112471, 0.155391261, 0.00118763396, -0.292280316, -0.0298888087, 0.354842365, 0.158195168, -0.0573678426, 0.0431329869, -1.45323634, -0.141262695, 1.49383032, -0.102032982, 0.090965651, -0.0610504709, -0.586887658, 0.0737467334, -0.00428327499, -0.205733269, -0.0682440698, -0.0186942294, -0.120629132, -0.094529666, -0.0304605477, -0.0317333303, 0.127742916, 0.00953600276, 0.462545425, 0.0659638047, -0.250134289, -0.0549852885, 0.160503104, 0.212851524, -0.151994482, 0.117166385, 0.105368614, -0.173081025, -0.194751561, -0.0920406282, -0.00961320754, -0.384650111, 0.0759096816, 0.252548546, -0.00499991421, -0.0639880002, 1.33827198, -0.0294453725, -0.0200957842, -0.0602608994, 0.000926029868, -0.0834149569, -0.112273701, 0.0266698916, 0.0501765795, -0.0144635132, 0.0868562013, 0.175002828, 0.166874036, -0.146098197]], 'B0': [[-0.261590034, -0.254740387, -0.200391009, -0.412734449, -0.127378583, -0.779840648, -0.2225205, -0.178385004, -0.0890466422, -0.192315757, -0.340682566, 0.109040499, -1.17717385, -0.0489737578, -0.284652293, -0.0608704388, -0.228117973, -0.0681361556, 0.0557923876, -0.0581645519, -0.325819522, 0.0338256881, 0.26297462, 0.0991481692, -0.108087942, 0.074363485, -0.171441227, 0.621688962, -0.575517476, 0.123423144, -0.0504913852, -0.286557019, -0.166777834, -0.224220037, -0.339639962, 0.0476765037, -0.231337532, -0.428552777, -0.57622087, -0.131807297, -0.160987377, 0.0763323829, -0.0210713148, -0.327731013, -0.15028964, -0.270219684, -0.104854763, -0.174427032, -0.145733356, 0.808681846, -0.299890578, -0.332196146, -0.3189677, -0.0965436921, 0.0272296816, -0.347668588, -0.103536271, -0.200199276, -0.193995506, 0.121071532, 0.211403623, -0.00451120734, 0.0574726164, -0.290810585], [-0.203212366, -0.28124088, -0.386864126, -0.379921675, -0.187385201, -0.964373231, -0.273030996, -0.160036713, 0.695734262, -0.18056187, -0.355483234, -0.0298510343, -1.46335101, 0.283406377, -0.314898521, 0.0840432569, -0.31008178, -0.0559872985, 0.0695347786, -0.0949331671, -0.404517114, 0.0108381957, 0.352728993, 0.39028877, -0.0711432397, -0.0506639332, -0.190513566, 0.591554523, -0.0696842596, -0.140588731, -0.00478664041, -0.224886507, -0.334861547, -0.222583473, -0.321971476, 0.124779597, 0.0695054978, -0.639849901, -0.560180187, -0.198421836, -0.153150097, 0.103281438, -0.212524906, -0.36242336, -0.0403256007, -0.101015247, -0.0384756215, -0.125663161, -0.00890433043, 0.674040198, -0.335715204, 0.0551299006, -0.285132438, -0.0341190994, 0.12719661, -0.316635668, -0.134069473, -0.0831985399, -0.294541568, -0.949042141, 0.175411448, -0.0929223299, 0.196627602, -0.288889736], [-0.322293282, -0.297244936, -0.325174689, -0.354907662, -0.347564936, -1.11647439, -0.387787759, -0.207586616, 0.810209811, -0.163832918, -0.337463528, 0.0360800773, -1.67489672, 0.0740527958, -0.372054815, 0.351204634, -0.386625826, 0.0672251284, 0.0705418289, 0.0131950006, -0.399895489, 0.0767542571, 0.53747046, 0.378420234, -0.0152845457, 0.0218296945, -0.13726455, 0.708027482, -0.0074326247, 0.0916371122, -0.0915486068, -0.417038202, -0.231268868, -0.234186381, -0.341495961, -0.0245130658, 0.0237151831, -0.685397089, -0.590690672, -0.348459303, 0.0798973665, -0.106398396, -0.215114594, -0.376219511, -0.0876257792, -0.212699786, 0.0117036328, -0.350664198, 0.115460038, 0.68755877, -0.500684321, 0.160951167, -0.315359652, 0.120015517, 0.177117258, -0.344836682, -0.201738954, 0.00749304146, -0.24651292, -0.881010234, 0.287385166, -0.0954617113, 0.243248507, -0.3342143], [-0.396069109, -0.347961545, -0.582782447, -0.450861454, -0.266353339, -0.613666356, -0.346463919, 0.0462668128, 0.624022722, -0.246736273, -0.547772646, 0.0341160372, -1.12669349, 0.236603737, -0.598983645, 0.372367054, -0.609943688, 0.235452935, 0.0141404681, 0.0544350706, -0.555121243, 0.0775685161, 0.42897135, 0.368923157, -0.110285252, -0.142308086, -0.227739558, 0.341696322, -0.0243846774, -0.00432631373, -0.298403502, -0.579960167, -0.379526675, -0.117050573, -0.631705344, -0.0301392972, 0.153228432, -0.468102574, -0.50093627, -0.48307395, -0.0539836697, 0.0564129688, -0.0375493765, -0.285134673, -0.161065459, -0.165979147, -0.309758544, -0.34262675, 0.206628352, 0.454842776, -0.497580767, 0.137063861, -0.256579965, 0.0280901212, 0.101070151, -0.524203002, -0.235678673, -0.362963498, -0.162015468, -0.724672198, 0.281614155, -0.100277841, -0.0865811184, 0.328804314], [-0.378869802, -0.362124145, -0.0455153808, -0.141603395, -0.421266377, -1.18492901, -0.170255899, -0.249099493, 0.838715732, -0.11982356, -0.449698806, 0.0177177023, -1.94081831, 0.299817711, -0.559327364, 0.540919065, -0.411770105, 0.0239691585, -0.146304652, 0.32627967, -0.397932678, -0.0429700688, 0.740395367, 0.313350648, 0.0354891121, 0.181006655, -0.250664353, 0.349140495, -0.0167835951, 0.240861267, -0.122728676, -0.565508425, -0.243526995, -0.352602005, -0.446266145, 0.269351959, 0.0563698113, -0.844122887, -0.50801301, -0.475137115, 0.208334103, 0.100512139, 0.0511149652, -0.332957745, -0.246828482, -0.404221267, 0.0273786187, -0.346397728, 0.127773836, 0.757076025, -0.440738618, 0.305170059, -0.263585687, 0.187717304, 0.0419710129, -0.407407939, -0.203731328, 0.170398086, -0.240792364, -0.617439628, 0.346531242, -0.170525864, -0.0818929672, -0.0909773856], [-0.523122907, -0.3998245, -0.0404729843, -0.264144689, -0.295169115, -0.746001601, -0.125674278, -0.139114186, 0.677550495, -0.173590273, -0.576826394, -0.016805077, -1.41810966, 0.29220897, -0.455296934, 0.529050231, -0.319842607, -0.00373649597, -0.0786577761, 0.291967988, -0.312670082, 0.0109083354, 0.609193325, 0.252513528, -0.113068111, -0.224405617, -0.264756918, 0.134012356, 0.0329082757, 0.328765243, -0.0425280407, -0.478308916, -0.145957947, -0.393593132, -0.553042173, 0.0256099552, 0.139678508, -0.585955441, -0.407812953, -0.328710675, 0.185414225, 0.103694022, 0.0316361599, -0.340447128, -0.327705383, -0.218292937, 0.192943424, -0.316499203, 0.346315354, 0.550865114, -0.374648631, -0.0198715702, -0.4329364, 0.116399422, 0.0681530833, -0.482809901, -0.182520077, 0.170262635, 0.0279964432, -0.476091921, 0.228224128, -0.173462152, -0.116619095, -0.0730236471], [-0.439890742, -0.51926887, -0.102332562, -0.230930492, -0.383260638, -0.538419664, 0.0102172494, 0.0126652792, 0.501832068, -0.40603289, -0.664365172, -0.0483916178, -1.08257782, 0.206709251, -0.420804262, 0.41422832, -0.269241661, 0.165997639, -0.191479743, 0.256461442, -0.331614584, 0.22469449, 0.379979283, 0.210718334, -0.000922687352, -0.277700424, -0.280056536, 0.0658581331, -0.0560321286, 0.227922842, -0.19340758, -0.520828247, -0.272726685, -0.127297059, -0.635499597, -0.0291547626, 0.051123172, -0.455579042, -0.250505477, -0.342816293, 0.212066516, 0.271266878, 0.162171066, -0.23798424, -0.276115358, -0.200395942, 0.124606922, -0.405976951, 0.336970478, 0.417470336, -0.37632513, -0.151897728, -0.241906628, 0.0861566514, 0.164762616, -0.584654748, -0.0962603986, 0.00266877562, -0.110716164, -0.388467848, 0.253922909, -0.235984698, -0.227738917, 0.32072255], [-0.358823389, -0.252885759, 0.0308526605, -0.29706043, -0.238790005, -0.537524462, -0.0874578804, -0.0738106593, 0.474569023, -0.151123524, -0.393960774, -0.0472191796, -1.11555684, 0.246828243, -0.408663005, 0.416615307, -0.206888855, 0.245567977, -0.205564439, 0.247279719, -0.201455668, 0.225362241, 0.417925686, 0.242308393, -0.101750992, -0.301164269, -0.240634963, 0.0309283137, 0.0433055907, 0.39830786, -0.0787599981, -0.421847552, -0.00851677731, -0.30642885, -0.490628541, -0.0937992409, -0.025578022, -0.430297911, -0.304013789, -0.374562442, 0.184962377, 0.114447095, 0.041350171, -0.275557369, -0.225968987, -0.213728502, 0.142705038, -0.305343986, 0.251871824, 0.269283235, -0.332484782, -0.342890233, -0.218322605, 0.168079585, -0.0552980006, -0.421757579, -0.317798674, 0.0928422511, -0.24642244, -0.342389405, 0.198388204, -0.384320319, -0.325979054, 0.0553179644], [-0.276695877, -0.300550699, -0.149350464, -0.284821451, -0.211232722, -0.384438634, -0.0672867149, -0.0104373042, 0.342553943, -0.23602742, -0.386318535, 0.0600327849, -0.908081055, 0.238879234, -0.443185806, 0.307638854, -0.241641074, 0.169587985, -0.196120217, 0.153318912, -0.100178398, 0.0761422887, 0.251849324, 0.148809731, -0.08925239, -0.190018266, -0.134162039, -0.0133343786, -0.0634990409, 0.300145507, -0.210740954, -0.320940733, 0.0823979527, -0.165664241, -0.559439063, -0.287917495, -0.133863106, -0.393427998, -0.194492966, -0.0231789201, -0.0272546727, 0.172856927, 0.0590330064, -0.233650133, -0.214582473, -0.400221586, 0.169763893, -0.458477378, 0.330086887, 0.1304214, -0.303302199, -0.261557221, -0.323601246, 0.116337568, -0.190245658, -0.468000829, -0.0708816051, -0.11413803, -0.0429060012, -0.257243097, 0.181568518, -0.408345401, -0.259190202, 0.0353899933], [-0.374366224, -0.284598053, -0.0211169869, -0.196697026, -0.157820314, -0.252849996, 0.0490707159, 0.034879867, 0.01820755, -0.0778651088, -0.450918466, -0.106031209, -0.405263692, 0.050998494, -0.416058242, 0.185306594, -0.238896236, 0.103360675, -0.222883031, 0.287050903, -0.0267162174, 0.0952340141, 0.0855659842, -0.173484743, -0.150244698, -0.245557427, -0.182565391, 0.0152231008, -0.163066074, 0.170698956, -0.261793911, -0.308988214, 0.0595670976, -0.272880971, -0.54636687, -0.180362552, -0.351516843, -0.468828142, -0.0913760513, 0.0540462136, -0.0222333334, 0.176363543, -0.0312092807, -0.22591956, -0.0804595351, -0.342760563, 0.137066573, -0.377858162, 0.269936353, -0.22151649, -0.320113778, -0.214075997, -0.188144654, 0.0559554473, -0.0611962639, -0.426262736, -0.0870811194, -0.0505864173, 0.0219787583, -0.201164678, 0.0707573891, -0.450000107, -0.251309216, -0.128967658], [-0.20992735, -0.125985533, -0.0528560281, -0.305323273, -0.176832348, -0.183068752, -0.123647451, 0.17124477, 0.0712229833, 0.0949577689, -0.433754176, 0.00083502382, -0.566417038, 0.066978395, -0.415133387, 0.205809727, -0.196854204, 0.227139756, -0.0252720974, 0.0902059302, -0.120392352, 0.13231571, 0.223258063, -0.195867822, -0.12423265, -0.0271902159, -0.15020138, 0.223180249, -0.137761325, 0.141304344, -0.286330223, -0.13283354, 0.159646571, -0.182994097, -0.342594892, -0.266585231, -0.337088823, -0.292478204, -0.227324069, 0.0869009942, 0.00132961525, 0.0869841278, -0.0345033705, -0.231557876, -0.111512363, -0.258313894, 0.0636667013, -0.319922745, 0.140816092, -0.27048707, -0.277251005, -0.210336775, -0.0750849098, 0.0071510803, -0.0670253411, -0.298965454, -0.16254653, -0.0172069259, 0.0602391884, -0.181040555, 0.101103552, -0.326389253, -0.259695649, -0.219625056], [-0.241772711, -0.155455858, -0.129834652, -0.279482126, -0.16617161, -0.281205803, -0.197336614, 0.0569292344, 0.0972548872, -0.0145276114, -0.287555695, -0.101146244, -0.803444326, 0.239544436, -0.439433813, 0.248273522, -0.169766337, 0.232892096, -0.200972915, 0.0258900784, -0.0655604079, -0.0892892554, 0.336603582, -0.201364145, -0.204591244, -0.142081752, -0.206963509, 0.167365596, -0.106462158, 0.362972736, -0.321013242, -0.0843052119, 0.245310828, -0.0643336028, -0.245233238, -0.6257236, -0.291667908, -0.391461134, -0.114755437, -0.0354603827, 0.0263297856, 0.139787227, 0.00978012756, -0.256917655, -0.0615572147, -0.219050422, 0.0355445854, -0.190293938, 0.123421445, -0.0995264053, -0.274184048, -0.3399176, -0.361379117, -0.0198475402, -0.271501064, -0.115853012, -0.0531105325, 0.0380859673, -0.0534323603, -0.0713265091, 0.13686204, -0.352080524, -0.183705777, -0.018156575], [-0.111059755, -0.326349109, -0.130313039, -0.397213995, -0.0230703428, -0.21705997, -0.106622636, -0.109167486, -0.131208137, -0.0525659733, -0.420832515, -0.0506091639, -0.394417107, -0.0152240433, -0.411029428, 0.109279588, -0.226431638, 0.0954886824, -0.15935567, -0.0630397201, -0.0714987218, -0.134117097, 0.169320077, -0.520227373, -0.296095401, -0.162361622, -0.314723611, 0.138767377, -0.294491112, 0.175616771, -0.114610836, 0.13472186, 0.133381978, 0.00355248153, -0.238052398, -0.372400433, -0.451787353, -0.467324287, -0.0403961241, -0.0203840435, 0.0436763577, 0.108194821, -0.0307900719, -0.358601868, 0.0991829932, -0.416153848, -0.485566914, -0.100620985, 0.0579670668, -0.323336333, -0.309123427, -0.159492239, -0.236301124, -0.299242646, -0.321903408, -0.232240871, -0.0725109577, -0.137953252, 0.0541606918, -0.0629052669, 0.200583786, -0.311741501, -0.188454106, -0.161183581], [-0.173836812, -0.12341845, -0.269517899, 0.0157358348, -0.155451059, -0.0478477478, 0.0540456027, -0.0422971249, -0.204530105, -0.062747471, -0.470959067, 0.0164431203, -0.481974602, -0.446918786, -0.0265075862, -0.360573411, -0.404895365, 0.301688373, 0.1991961, 0.157980472, -0.195450276, 0.212318987, -0.0458324999, -0.148596227, -0.393894076, -0.157601133, -0.254459888, 0.300688326, -0.558344603, 0.0966498107, -0.216932699, -0.249008626, -0.0121222623, -0.218307808, -0.44260186, -0.456298649, -0.306828052, -0.0701295733, -0.0624932721, -0.221813664, 0.150088087, 0.0256934278, 0.0991925821, -0.486742854, -0.260467231, -0.0837786794, 0.0859414414, -0.257816374, 0.146549731, 0.168693334, -0.133361876, -0.342303276, -0.244377166, -0.141895249, -0.261010736, -0.389060855, -0.0912466496, -0.162829727, -0.0363343395, -0.0651117414, 0.179576814, -0.236577541, -0.412845701, -0.398828506]], 'WH': [[[-0.279092818, 0.00367008313, 0.0295480695, 0.0967874825, -0.0250449516, 0.16977483, -0.0203768425, 0.0828250498, 0.0319829844, 0.037951149, -0.00549933221, -0.0702953562, 0.0163543131, 0.00082181691, 0.125733569, -0.0162159745, -2.12631528e-15, 0.0680670813, -3.31520312e-28, 0.145637691, 0.0325538777, -0.0393718742, -0.0190771967, -0.0743876547, 0.0627230108, -2.52861843e-10, 0.141107425, 0.13397947, -0.148698568, 0.0401405729, -0.0323035866, -0.00105176331, 0.178799883, -0.0774264634, 0.0910952762, -0.0234408639, 0.0414660759, -0.0190116726, -0.0290372763, 0.245764092, 0.163928449, 0.154717088, -0.0464996397, -0.116545752, 0.098955974, 0.0238994416, -0.0995484665, 0.117810033, 0.146231472, 0.127206966, -0.12776053, -0.0744087547, -0.0175890829, 0.0242812727, 1.51598826e-07, 0.055606924, -0.0634818301, 0.0168841109, -0.0855717435, -0.0530344993, 0.23522687, 0.119087309, 0.0397947021, 0.0955633149], [0.0496575311, -0.19114846, 0.0168010481, -0.0208816417, -0.222733065, -0.085729152, -0.0332766473, -0.0839629248, -0.0129109519, -0.089952521, -0.0264935289, 0.0577410869, 0.0777788237, -0.0317512266, -0.103529967, -0.00387087394, 1.34103219e-34, 0.0163924918, -1.16849663e-35, -0.104628332, -0.0134553639, -0.0496582985, -0.091541484, -0.0787199959, 0.0178261288, 1.39153644e-05, -0.100408837, 0.0421290062, -0.0694128349, 0.0261681899, 0.127570376, -0.000871899305, -0.171706229, -0.0899309516, 0.000802905124, -0.0440492481, -0.0638899729, 0.0328590535, 0.080101341, -0.00619217381, -0.172582164, -0.024020765, -0.161754906, 0.0827497542, 0.0115079926, -0.0683591589, 0.0205305964, 0.068838492, -0.0054927771, -0.192191422, 0.00254698354, -0.00891640875, -0.0487922095, 0.0830323473, -3.4380767e-07, -0.0118250512, -0.133352563, 0.00418171939, -0.0687814355, -0.152843609, -0.0420781858, 0.0887236372, 0.0165097378, -0.00112195348], [-0.0548183545, -0.157533586, -0.0356475711, -0.0306664258, -0.0306776501, -0.202153698, -0.0323786102, 0.00548896892, 0.120877907, 0.0752315, 0.133780733, -0.156169221, -0.0215546116, -0.168348625, 0.100922257, 0.104521215, -6.8894053e-28, -0.0141489748, -5.67642592e-36, -0.156709954, 0.150926471, 0.115381069, 0.0802345499, -0.0791891068, -0.0210676696, -9.9192819e-17, -0.0853831843, 0.0358732119, 0.0642558038, 0.150252551, -0.0136436261, -0.000590737851, -0.10386315, 0.049916219, 0.00268902653, 0.11551629, -0.161866441, -0.160657406, -0.0696109235, 0.0419890918, -0.0112339836, -0.131253347, -0.116607293, 0.111865155, 0.0119407568, -0.00751339132, 0.00826926529, -0.00501221744, -0.0904764906, -0.0956398994, -0.0230547972, -0.0115942871, -0.0673062876, 0.0482026637, 1.67936438e-08, 0.100776017, 0.209989458, -0.000232317805, -0.106974609, 0.104955867, 0.0477305055, -0.00923182722, -0.0885539204, -0.0634865984], [0.101845525, -0.256263137, 0.00825926196, -0.0292393379, -0.120564826, -0.108640455, -0.0495174974, -0.0109591233, -0.103854053, -0.319046199, 0.188891903, 0.183492362, 0.0323154666, -0.188828513, 0.00463930471, -0.0882641524, -6.85638929e-24, -0.0748602673, -7.74340942e-23, -0.182680279, 0.0135498531, -0.0325824395, -0.0208105855, 0.0174868014, 0.0673638508, 2.36763026e-06, -0.143824756, -0.0349948294, 0.133390203, 0.088483274, 0.163392648, -0.00224889233, -0.236929327, 0.00377031369, -0.132165954, -0.114496872, -0.25719741, -0.0997946635, 0.232400894, 0.0234127473, 0.105257027, -0.0106141297, -0.277694672, 0.221874014, 0.0854600966, -0.0366409048, -0.0256645996, 0.0742750689, -0.130860835, -0.227297515, -0.114052936, -0.032134477, -0.0950975642, 0.219208419, 2.65454162e-07, 0.055546429, 0.0856794864, -0.00625946047, -0.134912372, -0.0932862833, 0.0898282155, -0.00714215403, 0.295464218, 0.0208738092], [0.0406783149, -0.0511602797, -0.0385192819, -0.0935994089, -0.0109008029, 0.0221558399, -0.0404554009, -0.117526442, -0.082797952, -0.0622450709, 0.128707454, -0.0281489808, 0.0610860549, 0.0165497027, -0.0785938352, 0.0551615171, -7.79235668e-15, -0.153907031, 3.11118717e-18, -0.278108954, -0.0397656262, 0.130293548, -0.136928827, 0.139921769, 0.087301679, -0.000372435607, -0.225351989, 0.0628449172, 0.0933250338, 0.0518531464, 0.203596443, -0.00324223796, -0.150354281, -0.0451877005, -0.128071293, -0.1317029, -0.0967730582, -0.0417767018, 0.0993660986, 0.0101042427, -0.227193862, -0.134418502, -0.120200105, 0.0838657692, -0.0979592726, -0.0629850477, -0.0779921263, -0.0930814296, -0.00975720864, -0.17772232, 0.00827672891, 0.154941067, -0.0659585446, 0.146194473, -4.62073922e-06, 0.0716912895, 0.0688372403, -0.00608290639, -0.141077369, -0.000914706441, -0.0460910834, -0.0161048304, 0.0835797265, -0.155569583], [-0.496213824, 0.229244128, 0.0421712026, -0.0150961298, 0.446246743, 0.0359043926, 0.0200838149, -0.687287211, 0.127868101, -0.130677655, -0.165213615, -0.837488711, -0.216924503, 0.58011061, -0.129999608, 0.00676827691, 5.11363966e-38, 0.142854556, -1.58043584e-38, 0.333526224, -0.263566136, -0.311254114, -0.0605377182, -0.101720691, 0.0694779754, -8.76404783e-07, 0.202833921, -0.216183975, -0.455789924, -0.0558332093, -0.112894103, 0.00043943015, 0.193951905, -0.0120513262, -0.0999991521, -0.221794292, 0.191843897, -0.0789993182, -0.409113526, -0.202545479, -0.14618665, 0.286078364, 0.279573113, -0.326130301, -0.0988691896, 0.114466228, -0.420217216, -0.0467021614, 0.0790710822, 0.253885061, 0.0344400145, -0.21500738, 0.122406937, -0.0862090066, -7.06161387e-08, -0.123447888, -0.0405373722, -0.00210876949, 0.260809064, -0.122610472, -0.0341603272, -0.30089578, -0.33659637, -0.0105835302], [0.0282204077, 0.142198533, 0.0950936377, 0.0301888548, -0.301599056, -0.101304911, -0.0176306944, 0.0646677837, 0.11420694, 0.203345239, -0.167574853, -0.226751491, -0.0260640159, -0.132237762, 0.0682613701, -0.0811365694, -4.3439817e-33, 0.0478282645, 1.33827736e-37, -0.0688157976, -0.153582662, -0.264756233, 0.116451249, -0.0811296552, 0.107630655, 2.30413194e-38, -0.170545205, -0.207947493, -0.064343974, 0.136992306, -0.0887010768, -0.00107259071, -0.161397919, -0.0564157777, -0.00615759101, 0.145528957, -0.00706289569, 0.0207339153, -0.0977011696, -0.103839517, 0.190305263, -0.176782653, -0.0376720093, -0.0388425365, 0.223261893, -0.0429607071, -0.0992663652, 0.0393210463, -0.120878525, -0.136741206, 0.0291975848, -0.0328784846, 0.0280115269, 0.0986825898, -8.73522676e-09, 0.1820333, -0.0769949183, 0.0014791463, -0.0660190657, 0.0700159296, 0.087678276, -0.138231143, -0.0455628969, -0.0217168611], [-0.269493401, 0.0869998336, 0.0919353664, -0.00454370258, 0.265017152, 0.0435460769, 0.00365970307, 0.190190211, 0.164344639, 0.0192694515, -0.0842801481, -0.231174037, 0.0543253571, -0.0449508689, -0.00833008811, -0.0154148331, 2.67636629e-12, -0.123693816, 3.57800288e-29, -0.0074438937, -0.188279659, -0.43404898, 0.160133079, -0.0151269333, -0.0923159868, -1.99111239e-21, -0.0446064323, 0.184886619, -0.208968759, -0.0195368212, -0.0756343529, -0.000254879269, -0.0721772164, -0.0114360359, 0.255609035, -0.246229529, 0.00652024662, -0.093623586, -0.194231465, -0.139708772, 0.0392567217, -0.0582697541, -0.0186228454, -0.165335268, -0.0354253873, -0.034129899, -0.371086448, 0.0646763816, -0.0250052791, -0.0754483864, -0.0207112711, 0.204181209, 0.010716022, -0.0562483892, -2.82096679e-09, -0.128178701, -0.0862305611, -0.00568585191, -0.134186968, 5.07541736e-05, 0.0487901643, -0.111251578, 0.0190186407, -0.0841413513], [-0.579785109, -0.140028492, -0.0168679636, -0.115662284, -0.151095524, -0.170535505, -0.0406124853, -0.0919618234, -0.187568873, -0.127392367, -0.0999572203, 0.0527039208, -0.363612413, -0.531388104, 0.153941095, 0.230650365, -2.24191896e-22, -0.22627987, 2.12729298e-38, -0.0348401368, -0.0111992108, -0.406400591, -0.164257437, 0.0574934445, -0.205908984, -4.52369861e-38, 0.0959743485, -0.0216448344, -0.424131393, -0.0207046848, -0.484397024, -0.00143443712, 0.0696766526, 0.0228243042, 0.104488552, -0.103554562, -0.0862282887, 0.439343989, -0.173797652, -0.169038191, -0.0944330916, 0.240302503, -0.103687212, -0.397690237, -0.114624612, 0.00371700712, -0.19802098, -0.131500244, 0.113530196, 0.114652768, -0.0354756825, -0.0802909359, -0.17950131, -0.348058254, -3.26139626e-09, -0.112199493, 0.246406496, 0.00875763595, 0.0891225561, 0.296346962, -0.0440094434, 0.10835252, -0.234195486, 0.131088048], [0.046061188, -0.0278254319, -0.0284285136, 0.0347755738, -0.198197141, 0.101283811, 0.00631771935, 0.148155034, 0.149271205, 0.309536725, 0.109526783, -0.163182855, 0.211062416, 0.032084953, 0.112598367, -0.0660677403, -6.41268925e-14, -0.0377117023, 2.64694161e-20, -0.092167072, 0.162217513, -0.142356962, 0.194783464, -0.0610357784, -0.0770962834, -3.07898401e-10, -0.0528213382, 0.0978976339, 0.0743210316, 0.0132834185, 0.0185094476, -0.00226758863, -0.0238955189, 0.0618546717, -0.00258487579, 0.0213802774, -0.0585977286, 0.0433658324, -0.27437374, 0.0987840071, 0.149830729, 0.1204798, -0.0716593787, -0.120935075, -0.0476577505, -0.020183539, -0.0318478122, 0.020780379, 0.111683592, -0.109090723, 0.163733691, 0.0934261605, -0.0244466662, 0.0602609962, -8.90431551e-08, -0.0400605835, 0.148195982, 0.0182677824, 0.22156775, 0.0192584619, -0.0168317519, 0.00147070247, 0.0128510967, 0.136363417], [0.0408265777, 0.0379319079, 0.0083609065, -0.0224833842, -0.216306016, -0.223966271, 0.0079148002, -0.490995705, 0.0127571318, 0.0259715989, -0.133303836, -0.0726861656, -0.291603982, -0.0648610666, -0.263907075, 0.0918342471, -2.74228285e-24, 0.0150446771, -1.04707723e-37, -0.0994359851, -0.179527566, 0.104588799, -0.0534766726, -0.0239942335, 0.138783529, 1.89105822e-06, -0.0747640952, -0.077410385, -0.0591494516, 0.0456564166, 0.138314292, -0.000760588213, 0.0209002662, -0.047299251, 0.0891472027, -0.104550831, -0.0798724815, -0.0768202618, 0.0184262954, -0.226331964, -0.229238018, -0.112584613, -0.0156999081, 0.152141765, -0.0372805931, -0.0098102279, 0.0985664651, -0.0394093469, -0.1761204, -0.0737327933, 0.184114218, -0.134948835, -0.058844272, 0.168410301, -8.90285961e-08, 0.0328511223, -0.120050721, -0.00498397229, -0.0878369734, -0.332256079, -0.0185176022, -0.133958265, 0.020853525, -0.098157078], [-0.149527997, 0.115393899, 0.00338744954, -0.0525210463, -0.092097804, 0.064999409, 0.000317518105, 0.166270837, 0.04661474, -0.0399044529, -0.0761277676, 0.0325173698, -0.117295131, 0.00979976822, -0.159632161, -0.0242563542, 3.419385e-20, -0.154731736, 3.79515635e-18, -0.251205057, -0.0398849249, -0.00045338177, -0.0105486894, -0.0767395422, -0.0651321709, -4.1383047e-10, -0.107594676, -0.0326955356, -0.0883596912, -0.0455951244, -0.108868107, -0.000695158669, 0.0458760187, -0.0633495972, -0.0414648764, 0.0204114392, -0.114275292, -0.095195435, -0.156788409, 0.124733418, -0.0318603143, -0.0620400868, -0.0624706373, -0.0696748793, 0.0113335149, -0.0293837115, -0.0163906906, 0.0661647394, -0.199019387, 0.0289755501, -0.00422107754, 0.0878752545, -0.00259722932, -0.0842361525, -8.96132857e-09, -0.100056164, -0.0741806328, 0.00479092635, 0.0659452081, -0.0152343446, -0.0463771932, 0.0481242724, -0.062384069, -0.232662901], [-0.733823299, 0.00177129649, -0.042821046, -0.138348654, 0.186417446, -0.112436414, 0.00181456562, -0.0471706949, 0.00279216864, 0.0842402279, -0.0110274749, -0.917348206, -0.619787395, 0.238704592, 0.120091446, -0.0723770931, 3.70262134e-37, -0.111799918, -1.50306245e-18, -0.0952440277, -0.0764879361, -0.324437767, 0.00592051866, 0.0616033971, 0.0534032136, -0.000900168845, -0.186854407, 0.131795198, 0.0817038491, 0.0903444886, 0.271919996, -0.00638729241, -0.0734275579, -0.0279731192, -0.0444632284, -0.529145062, -0.0813968554, -0.0528424643, 0.111578174, 0.0365943946, -0.0277878717, 0.0167253986, -0.0659366474, 0.0917073265, -0.0609900057, -0.114766128, -0.926152647, -0.0231698491, -0.166812003, -0.0542898513, -0.0288788117, -0.071673587, -0.226924554, 0.186856925, -1.59860356e-05, -0.200979471, -0.00570825813, 0.000733698311, -0.218784094, -0.188989326, 0.032567706, -0.00020147505, 0.0241711549, -0.375801861], [0.195965499, 0.0551537834, -0.074863255, -0.158360362, 0.0649354532, 0.078206338, -0.0425646342, -0.00689992961, 0.102710597, 0.140665129, 0.138441026, -0.189231008, 0.250620008, 0.212964267, -0.248708963, 0.173194379, -2.91795017e-13, -0.152228519, -1.86051092e-37, 0.0105671901, 0.0462913066, 0.292997628, 0.208189905, 0.162879109, -0.0949664414, -2.31985505e-05, -0.0155089768, -0.0435569175, 0.039725177, -0.0290330164, -0.166376278, -0.000832792546, -0.016023295, -0.220962942, 0.0188961551, 0.252527028, -0.078099899, 0.0207487512, 0.197752893, 0.0545943044, -0.234987006, 0.0832186341, -0.0945238471, 0.0610148981, -0.0295364279, -0.0853974074, 0.357886016, 0.0664832368, -0.182839543, 0.0825949758, 0.170347273, 0.0292167962, 0.0826478824, -0.125078127, -3.84393161e-06, 0.0284410734, -0.10255634, -0.0244651791, 0.0311623253, -0.248741701, 0.0505547859, -0.173015222, 0.0205725431, -0.127618968], [-0.179793417, -0.0220449138, -0.0556801818, 0.165839076, -0.088377364, -0.0176094677, -0.0136360144, -0.0708313659, -0.557980716, -0.22185348, 0.418226063, 0.192883641, -0.188509598, -0.0368672311, 0.379901618, -0.200846717, -1.73277758e-38, 0.114662372, -6.85382786e-39, -0.0835465267, 0.586681724, -0.176801011, -0.402970642, 0.236820564, 0.0334872976, -2.45204052e-12, -0.149597749, -0.240773261, 0.0822366625, 0.0549474396, 0.00058034074, -0.000823058828, -0.0698298961, 0.0328713767, -0.615967155, 0.23734507, -0.116093278, -0.0284850243, 0.344554812, -0.131402373, 0.414595664, 0.119176254, 0.0330301933, 0.0876186341, -0.0138813239, -0.0268705338, -0.197726876, 0.175971553, -0.00302737486, -0.0870418474, -0.342119128, -0.158643663, -0.324933261, 0.00898877997, -1.39698253e-09, -0.0781539679, 0.335220128, -0.0247672535, -0.102849007, 0.0889459625, 0.0871517286, 0.0326079689, -0.0477655157, -0.00379175367], [-0.371931821, -0.0308348797, -0.0980634168, 0.0578439496, -0.213729009, 0.234708518, -0.0171412304, 0.0489587598, -0.330166787, -0.14218691, -0.211660787, 0.365620643, 0.0373465046, -0.361453474, 0.118972532, 0.216139764, 2.95311942e-23, 0.0192629024, -3.97083311e-38, 0.136852831, -0.0875270367, -0.499060094, -0.287936211, -0.0855778307, 0.0368467234, -9.63728652e-08, 0.164363936, -0.131037354, 0.0220574811, 0.0500097387, 0.0658935532, -0.00135030702, 0.124689877, 0.125207901, -0.0800266638, -0.100268647, 0.180538401, -0.0220079143, -0.156327009, -0.44349426, -0.014710214, 0.225046635, 0.158594936, -0.0113799861, -0.0990927368, 0.0282657072, -0.153534189, -0.119472608, 0.230604872, 0.230386212, 0.0466880128, -0.186545268, -0.130788103, 0.0259501338, -2.33813708e-07, -0.1004088, 0.151209846, -0.0207308624, 0.038870018, 0.150877312, -0.055468183, -0.211140603, 0.144870639, 0.309557915], [-0.034407746, -0.0822164044, -0.112436846, -0.10203217, -0.281873167, 0.0493914373, -0.00680338219, -0.0146329748, -0.506281734, -0.47718671, 0.204721108, -0.0541262254, -0.181586057, -0.13202332, 0.252190173, -0.177237302, -4.54758054e-23, 0.0547501035, 1.02547403e-37, -0.0249909982, 0.00237021036, 0.238043889, -0.52668786, 0.121834099, 0.0170775596, -3.78509367e-06, -0.153694466, -0.416763633, -0.0904038921, 0.0914556831, -0.0782917812, -0.00121942139, -0.104872711, 0.0529655293, -0.411826789, -0.107852742, -0.0285455398, 0.185819477, 0.131743506, 0.345512986, 0.308858752, 0.020637149, -0.0536932051, -0.077095367, -0.0271334685, -0.015093022, -0.080338642, 0.0268619303, -0.116717108, -0.0342534892, -0.140364572, -0.190709367, -0.153076112, -0.0698927119, 5.49112968e-08, -0.0706916228, 0.00780169107, -0.0183644947, 0.00074026885, 0.237014711, 0.045254644, -0.0798348263, 0.0133255208, 0.07618922], [-0.166481182, 0.0162430275, -0.0100505464, -0.0397120714, 0.0727492645, -0.334716886, 0.0154713113, -0.616785228, 0.170059502, 0.0343493223, -0.0587440282, -0.298714578, -0.345945925, -0.0837498531, 0.110224925, -0.295229703, 6.9898913e-12, 0.0579766817, -1.15666477e-21, -0.0154133029, 0.00927180331, -0.228238866, 0.12630491, -0.0438695699, 0.00682232529, -2.48306448e-13, -0.0020528154, -0.140618667, -0.162785068, 0.0364334919, -0.20406957, -0.00296806265, -0.149732262, 0.0477215871, -0.0161122121, -0.13563545, -0.0735452771, -0.00620254502, -0.131307468, -0.154945821, 0.185089365, -0.145126998, -0.0674928129, -0.221803322, -0.0610833056, -0.0772039518, -0.119102836, -0.0930379555, -0.184510052, -0.210797608, 0.11277359, 0.0568952486, 0.188056797, -0.151723966, -1.82772908e-06, 0.000483176176, 0.0809992254, -0.0504160598, -0.0602952652, 0.0970510244, -0.0590982437, -0.163493097, -0.234167695, -0.220498621], [-0.0674387887, 0.115631595, 0.0487233847, -0.0829880908, 0.0731525645, 0.121875688, -0.0018705196, 0.215652972, -0.1991629, -0.304188997, -0.0564002916, -0.187082633, 0.126841635, -0.0140662063, 0.05809981, 0.0728424415, 2.07292901e-17, -0.00949449372, -8.94711581e-38, -0.00384019781, 0.158577561, -0.114990994, -0.18148303, -0.0375852734, -0.0214764234, 4.44245379e-05, 0.00611864682, -0.0881550461, -0.102958441, -0.0427771211, 0.0431808122, -0.000117064868, -0.0349464342, -0.0725142583, -0.366863817, 0.0492760688, 0.0229345784, 0.133742601, -0.152432382, -0.0466721915, -0.0720540658, 0.051295232, 0.055139631, -0.124015704, 0.0373848788, -0.0430585481, -0.17014353, 0.0609445199, -0.0939550996, 0.00480872579, -0.224109724, 0.0488064699, -0.101313256, 0.0269737802, -9.57110956e-07, -0.0863040686, -0.0143439211, -0.001436071, 0.006628423, -0.124747515, -0.067102246, 0.19003731, 0.00941484794, -0.0529085658], [-0.236650392, -0.0381156281, 0.10465762, 0.0162974373, 0.0480559729, -0.102813408, -0.026200911, -0.168968067, 0.176797003, 0.0331336781, -0.0653999597, -0.191260487, -0.149583504, -0.0791711286, 0.0854507014, -0.217454121, 6.8837569e-10, 0.0110602155, -6.32854737e-16, -0.0821673051, -0.0608223677, -0.0472765043, -0.0834912956, 0.0204147398, -0.0627696738, -1.73931824e-07, -0.163688332, -0.110334627, -0.123172872, 0.0649758354, -0.226238728, 0.000845605915, -0.0726111531, 0.0216409378, 0.137447864, -0.0271238927, -0.0170265213, -0.117212653, -0.0282898303, -0.0341742262, 0.0613350794, -0.20163092, -0.102707416, -0.208074316, 0.0177069195, -0.0685168505, -0.2610282, 0.00867099501, 0.00240780087, -0.134116605, 0.0350561589, -0.01542057, -0.044742886, -0.144794062, -5.95763993e-07, -0.0137532931, 0.00223225495, -0.0112892725, 0.0387794375, 0.136459634, 0.0734482035, -0.145480797, -0.139829308, -0.021635795], [-0.0128983967, -0.111653946, -0.0231501143, 0.0176305324, -0.188396692, 0.0335341468, -0.0197225697, -0.243667752, -0.122517981, -0.158409536, 0.138191387, 0.14357546, -0.072441563, -0.226409271, 0.0808197483, -0.0171360131, -3.77239722e-28, 0.057770174, -2.0819941e-37, -0.016154632, 0.0630996674, -0.0488026403, -0.11532522, 0.0608690418, 0.161672488, -2.57676384e-05, 0.0441665836, -0.072496444, 0.210070416, 0.131340116, 0.218082935, -0.000627919566, -0.0283928979, 0.00243175821, -0.154356644, -0.0852188617, -0.121676676, -0.179093763, 0.130944744, -0.172837138, 0.0197714716, -0.0463975705, -0.136197612, 0.173974141, 0.0382378362, -0.00803691708, -0.246890128, -0.044638712, 0.109009914, 0.0445987098, -0.171270758, -0.0707390383, -0.0704892054, 0.244614273, -8.45850423e-07, 0.0063506118, 0.000560732908, -0.00657934695, 0.0605956353, -0.0218077768, -0.0600183941, -0.00446802564, -0.0207334273, 0.0949893519], [0.0932717845, 0.175528392, 0.0156902317, -0.0759944767, -0.0784661025, 0.0676016659, -0.0247168131, 0.0509268269, 0.0200888235, -0.119662806, -0.0864877254, -0.205537498, -0.0492881015, -0.131547555, -0.0583390929, 0.0323565863, -6.08678485e-09, 0.0240176413, 3.39084056e-12, -0.00899587106, -0.205022544, -0.298972547, -0.186944485, -0.0499319583, -0.013743625, 2.47569515e-05, -0.04920182, 0.0806043446, -0.0189300906, -0.0217558779, 0.0610049777, -0.000976247829, 0.113926277, 0.0311205592, -0.015056029, 0.0364554264, -0.0128569864, -0.00719407341, -0.159941256, -0.0961485207, -0.205431059, -0.120467909, 0.135068879, 0.0241728853, -0.0912447497, 0.0188666955, 0.0111462502, -0.0274086352, 0.0475619547, 0.00790565275, -0.0486573465, 0.0412692539, 0.0326498561, 0.0354941301, -1.85077329e-06, -0.0234148242, 0.000577549392, -0.00465326244, 0.0456296355, 0.0987215191, -0.0500033647, 0.0169102028, -0.0335206315, 0.0431241728], [0.0647735223, -0.11567466, 0.0334086269, -0.0756595805, 0.079183057, -0.224598542, 0.0146152014, -0.0249767769, 0.212404415, 0.176833853, 0.0667838529, -0.363473028, -0.0857698992, -0.243623078, -0.0569796972, -0.0408537835, -1.56715442e-15, -0.327093244, -4.11493082e-31, -0.319954962, -0.0752718374, 0.0678899065, 0.219741598, -0.0343673229, -0.255144924, -1.00915654e-17, -0.0213177744, 0.122586586, -0.0692639872, 0.0433257148, -0.570156217, -0.00390395336, -0.076522328, -0.17099607, 0.258897066, 0.142458498, -0.173273966, -0.0642114803, 0.329466164, 0.0417602547, -0.205561742, -0.185618326, -0.0616169125, -0.0587719306, -0.0905955881, -0.106889471, 0.232592776, 0.153264999, -0.322374254, -0.0605089441, 0.0895775333, 0.0846282393, 0.0542446375, -0.378993034, -7.0816543e-07, 0.0495074987, -0.151394531, 0.0390956365, -0.205468804, 0.0282770097, 0.0677713826, -0.0680107847, -0.105823696, -0.450893521], [0.166513994, -0.0167322997, 0.0317662396, -0.0695257932, 0.14683111, -0.199093089, -0.0234454479, -0.0966718495, -0.0949116126, -0.232108608, -0.0798698068, -0.137409523, 0.194162682, 0.200227752, -0.35016045, -0.0528906323, 7.60767107e-23, -0.0809011459, -2.96613743e-38, -0.00500006648, -0.266911924, -0.158544555, -0.153084308, -0.0678671822, -0.0939232782, -7.01585213e-06, 0.0590038374, -0.0938244388, -0.0744432732, -0.0941880122, -0.142318621, -0.000981502584, 0.00521011092, -0.00582140591, 0.168518782, -0.0286848079, -0.0766967386, 0.203731194, 0.12582916, 0.253095627, -0.228626147, 0.086293608, -0.132186979, -0.0962494314, 0.0519518331, -0.0294526666, 0.19796896, 0.0286357142, -0.207494289, 0.112722762, 0.117796741, -0.0982730314, 0.0367075764, -0.203766376, -1.5766004e-07, 0.0604474396, -0.276854783, -0.0104974443, -0.0670693442, -0.164213449, 0.0923489928, -0.123619281, 0.164917096, -0.107099965], [-0.0408628695, -0.059065856, -0.00458182441, -0.00909393188, -0.206800491, -0.0237172097, 0.00361204939, 0.0682346821, -0.0222981032, -0.0411486626, -0.144979358, -0.0298564527, -0.0693809018, -0.0955008715, -0.0774962232, 0.11467132, -1.39977623e-38, -0.0907958522, 7.41452465e-38, -0.0602075756, -0.0866982862, 0.102299899, -0.193087414, -0.0253094919, -0.0326968394, -2.46778753e-09, 0.0242001768, -0.0289679151, 0.0615035966, -0.0809312388, -0.0152780283, -0.00122656883, -0.0829401389, -0.0591946878, 0.0723804832, 0.0511585698, -0.065734446, -0.113356687, 0.0181825738, -0.0986079127, -0.169823259, -0.0666020215, -0.0362306386, 0.189151883, -0.0397136584, 0.000954399758, 0.175884098, -0.0152314501, -0.0206558313, -0.095852159, 0.0344979241, 0.111043282, 0.0943885222, 0.106232978, -1.16060453e-10, -0.0187049024, -0.236272857, 0.00235171849, -0.0657896101, -0.0685419738, -0.0709554628, -0.208673179, -0.0617962219, -0.114208862], [-0.0958126336, 0.0494633354, -0.0281079691, -0.00620764354, -0.062179707, 0.00841360074, 0.0368885435, 0.0731078461, 0.00645543728, 0.0525154136, 0.0220356826, -0.114839926, -0.053630393, 0.00983980019, -0.0372669809, 0.0209206231, 1.69137121e-10, -0.0551686101, -1.13686234e-16, -0.0523122139, -0.115390457, -0.0246119983, 0.0990711525, -0.0672638416, 0.0118981861, -0.00106469262, 0.0478197783, -0.0115088737, -0.0163621064, 0.0356021412, -0.00451350398, 0.00569283729, 0.0602936, -0.0550232977, -0.0199217647, -0.0766942948, 0.0135113057, 0.118591562, -0.00356153748, -0.0935218483, -0.107520476, 0.0301310141, 0.0594973899, -0.00381052052, 0.023224242, -0.0282981098, -0.02532137, -0.0571350902, -0.0358227603, -0.0556463227, 0.107430451, 0.0914013907, -0.00999496505, 0.0306028258, -7.13433963e-07, -0.0868736133, 0.033319138, 0.00629057735, -0.00513396133, -0.0785878971, -0.110909685, 0.111589536, -0.0401220731, 0.0332329348], [0.0365217812, -0.112313434, -0.0550682321, -0.111122981, 0.048327107, -0.11612244, 0.00587392738, -0.242675871, -0.26906392, -0.0638041645, 0.0841093957, 0.0383615606, 0.0518310964, 0.0343411788, -0.164832786, 0.207190707, -1.42895068e-33, -0.108878262, -1.4578752e-19, 0.0319457687, 0.0114138648, -0.0123868398, 0.00160460989, 0.0299999267, 0.0387234688, 0.00402569026, -0.11978557, -0.147352502, 0.0560296476, 0.146462545, 0.139190689, -0.00303785433, -0.0514798984, 0.0132874055, -0.198370934, 0.0552934371, -0.0472364984, -0.130461097, 0.0357696675, -0.0215785578, -0.092261754, -0.076569736, -0.000618985097, 0.080167681, 0.0386646688, -0.0171826407, 0.0724868178, -0.0823872834, 0.0398882776, 0.0150790038, -0.0878081545, -0.0353385881, 0.0852267742, 0.163224801, -9.10300878e-06, 0.0880543664, 0.113026492, -0.00198857044, 0.0216939952, -0.279619247, 0.0628912523, 0.0354424007, 0.114495218, -0.135561794], [0.060569603, -0.0890911296, 0.0725622699, -0.0876002759, -0.050461702, -0.00703581003, -0.0193798169, -0.386197984, -0.318665147, -0.159772694, -0.191369742, -0.243913665, -0.0797395334, -0.221545547, -0.274997145, 0.077446267, -3.92070587e-38, 0.201239422, -1.85034442e-38, -0.0982422233, -0.159225106, -0.206135362, -0.192292094, -0.0459616408, 0.118047282, 2.03388097e-08, -0.0818336084, 0.148446232, -0.0733820722, -0.0780200139, -0.170272171, -0.000866602117, 0.122413971, -0.0632152334, -0.127141654, -0.243015796, -0.0755173564, 0.0812435597, -0.18528375, -0.229654118, 0.0536777191, 0.00540675549, -0.101699755, -0.0700082257, -0.114043444, -0.0257600769, 0.316133767, -0.134784102, 0.0318266563, -0.0638453886, -0.15897803, 0.122758284, 0.193497047, -0.0989009291, -3.47153972e-09, 0.0301028378, -0.208513036, 0.00522383116, -0.0683943182, -0.0590894558, -0.114996195, 0.340349197, -0.164329097, 0.171261027], [-0.175099745, -0.100694411, -0.0162951499, 0.00369270286, -0.061205525, -0.102940522, -0.0124076372, 0.3099612, 0.203164041, -0.0251423605, -0.0741040781, 0.0041197408, 0.211930588, 0.00748963002, 0.158758774, -0.0281215403, 4.81002252e-11, -0.377425909, -4.14656715e-38, -0.005267411, -0.0978638902, -0.0441020392, 0.00674025482, -0.0460145138, -0.216338232, -1.83544258e-16, 0.0523937419, 0.0278833676, -0.0420531072, 0.108578846, -0.0606225245, -0.000716359937, -0.0501775183, 0.0624298453, 0.109449998, -0.0591767021, 0.0168197844, 0.302493721, -0.189535365, -0.0418228135, 0.120155402, 0.262320906, -0.0843131915, -0.144677952, -0.0140783191, -0.0234767273, -0.115620032, -0.111747675, 0.23924917, 0.126086369, -0.0565603375, 0.0335617773, -0.0869314298, -0.0541220605, 5.74290695e-08, -0.0330636948, 0.00558846351, -0.00996208191, 0.137713507, 0.21998848, -0.0578215569, 0.20155932, -0.0917506889, 0.142668396], [-0.335722268, -0.045295231, -0.0274398699, 0.0366938263, -0.124253966, 0.168404907, 0.00403016107, -0.316972107, -0.0687703267, 0.0534320138, -0.0541482009, 0.0795653462, -0.283822864, -0.434400231, 0.118568897, 0.0565589964, -4.58106426e-12, 0.157938123, -1.26320109e-15, -0.021782618, -0.0370407254, -0.200656548, -0.114867374, -0.127031028, 0.0488440171, -1.30978435e-06, 0.111915611, 0.0633573234, -0.0317865834, 0.0462379456, -0.0711351112, -0.00163044722, 0.117135301, 0.0833574161, 0.0767114833, -0.120109588, -0.0144628035, -0.1865751, -0.145776957, 0.0212802403, 0.0595932417, 0.0440661721, 0.0828372017, -0.19764787, -0.0592830181, 0.0425223336, -0.191802487, -0.0546949878, 0.10335274, 0.0422529839, -0.0954194218, -0.143154994, 0.1173568, -0.0437994301, -7.55230622e-06, 0.0278314687, 0.129765674, -0.0407678634, 0.0684671402, 0.17742762, -0.0901409686, 0.000289496005, -0.0927682668, 0.0824099779], [-0.0666490719, -0.0313282013, -0.00592640834, 0.0177615266, 0.118062899, 0.0394779891, 0.0262620747, -0.0482547767, -0.136587709, -0.0874755755, -0.159171045, 0.263518184, -0.0681809783, 0.0312755518, 0.12360765, -0.127997607, -2.94091737e-19, 0.0959926173, 8.7755997e-35, 0.0859051347, -0.0314568542, -0.0761226937, -0.192732766, -0.0842932016, 0.0812366232, -0.00020980525, -0.0061843358, -0.0555964783, -0.125020444, 0.0070400564, -0.0189575329, -0.000194343651, 0.109117821, 0.049219884, -0.0215844698, -0.0675472468, 0.025073126, 0.0437011607, -0.182594687, -0.173430502, -0.191941023, 0.0422888249, -0.0713821948, -0.0477251559, -0.0463465713, 0.0207286961, -0.0387876183, -0.0636782274, -0.00505341589, 0.137583181, 0.121946238, -0.154670849, 0.0697164237, -0.00498291431, 3.28349029e-06, -0.00569824083, 0.0627220869, 0.000759470044, 0.0985713601, 0.088141337, -0.0904947817, -0.0561513938, -0.0861567557, -0.0486115403], [0.119651295, -0.0144927679, -0.0676285699, -0.0196654424, -0.0214311909, -0.0361398384, 0.0210981183, -0.0297402479, 0.162973672, 0.360044628, 0.0117740948, -0.0854262114, 0.187579319, -0.118934952, -0.0885215849, -0.0306274556, 1.93813037e-08, -0.0580776967, 4.05296213e-16, -0.0775151402, 0.0868094563, 0.105212279, 0.0458771177, -0.114475586, -0.0090522524, 0.000135088922, -0.0372637175, 0.0564140789, 0.0696442723, -0.118138812, 0.061707221, -0.00144931092, -0.146430716, 0.0683062524, 0.15709053, 0.106636263, -0.121007025, 0.0512080938, 0.104831673, 0.0751956776, -0.111004852, -0.151204929, -0.137088642, 0.171030402, -0.0530465655, -0.0544448495, 0.201930761, -0.0860365257, 0.0259009171, -0.163322091, 0.185383275, 0.00774174836, -0.0115663391, 0.0506777205, -6.41617419e-07, 0.0487090535, 0.0507561415, -0.0111502456, -0.177435443, -0.0679849759, -0.100310713, 0.0464631245, -0.0336313844, 0.0169078801], [-0.103300408, -0.0619720481, -0.123435423, -0.0999858081, -0.254594862, 0.315159976, -0.0178727247, 0.218282491, -0.260947824, -0.156439573, 0.0114046326, 0.0946244821, 0.021063339, -0.181595758, 0.0976133645, -0.128626257, -7.40393511e-16, 0.0886294469, 5.52465608e-19, -0.0637280494, 0.098109059, 0.0296869855, -0.0980529785, -0.0458310507, 0.130179673, 5.86836563e-07, -0.108333625, 0.0966183245, 0.0792728364, 0.112724163, 0.228472009, -0.00024343023, 0.0629949942, 0.112381697, -0.133668453, -0.192618832, -0.104387425, -0.0868454129, 0.00415824633, 0.0293777008, 0.211853445, -0.0320932083, -0.134981677, 0.00342858466, -0.0412756056, -0.0292398576, -0.328221887, -0.0978504047, 0.126411393, 0.0308394171, -0.226354018, 0.10849911, -0.00755599374, 0.0946809202, -3.46684246e-08, -0.00335612288, 0.0164096225, -0.00545829721, 0.114828758, 0.122241087, -0.106662787, 0.0352390073, -1.46444545e-05, 0.159471408], [0.0437512733, -0.0248362813, -0.0623948686, 0.0155881159, 0.0834960341, 0.103918858, 0.0103751188, 0.0589807779, 0.207455456, -0.0678233653, -0.0165398121, 0.050242871, 0.178681985, 0.0433727056, 0.00172028318, 0.0380333401, -3.71036933e-16, -0.038806472, -9.8217099e-38, -0.0162612814, -0.0181527007, -0.0469937809, 0.0647701249, -0.0300639309, -0.0710918978, 1.83471424e-12, 0.0914616585, -0.0111585977, 0.0431179889, -0.0271198805, -0.164058164, -0.00170405302, 0.0196136422, -0.0695173666, 0.0160817578, -0.0982149392, -0.0365333185, 0.0455556326, -0.134665787, -0.0667119399, -0.0809486359, -0.0100827999, -0.00379431946, 0.0668910146, -0.0300783161, 0.0198265184, -0.0326386169, -0.0113922646, -0.128172994, 0.000169729639, -0.0159943923, 0.022470003, -0.017373709, -0.0716434047, 1.03059481e-08, -0.0448851883, 0.0407680273, 0.0180716105, -0.0643164963, -0.00137690245, 0.0333873965, 0.146018475, -0.0128182415, 0.0406280272], [-0.103288062, -0.0487425923, -0.0343198106, 0.247659102, -0.113689139, 0.117164508, -0.0141753219, 0.0485824272, -0.120854542, -0.0509439781, 0.0294908714, 0.114474982, -0.0588888787, -0.269958556, 0.160675362, 0.0272444747, -5.47627612e-12, 0.00837105699, -1.71484714e-24, 0.110259533, -0.0307667423, -0.319244355, -0.14388299, -0.0713133067, 0.0116597312, -1.46007663e-13, -0.0614283159, 0.0651076809, 0.0396585092, 0.118088938, 0.0840563253, -0.00018588308, 0.0521744676, -0.00762895867, -0.135591373, -0.315338343, -0.0512372479, 0.0523620769, 0.0216003787, -0.041381482, -0.0160349421, 0.142936513, -0.027332982, -0.0267284494, -0.115490124, 0.035761036, -0.17283231, 0.029049255, 0.139084175, 0.108606353, -0.116096959, -0.116210863, 0.104875952, -0.00731500797, -2.09247322e-08, 0.012673758, 0.00646560267, -0.000483866548, 0.0561396293, 0.049584344, 0.0393952727, 0.0291479509, 0.121415287, 0.0295429528], [0.0640310571, -0.0629353449, -0.0358244777, -0.12024603, -0.0544615164, 0.0644733831, -0.0106619122, -0.0775053427, -0.169730604, -0.209870055, -0.172173008, 0.0725497976, -0.148047119, 0.0587100089, -0.0680307075, 0.0822431892, 5.47979451e-23, -0.0422865637, 4.46708211e-37, 0.041411709, 0.00169958954, -0.274621576, -0.131089315, 0.122520179, -0.112413257, 1.27977998e-30, 0.181666583, 0.0451396629, -0.0606668107, -0.0920026526, -0.116918929, -0.000843806833, 0.143670753, 0.0351514667, 0.0638206303, -0.106558241, 0.0252601691, 0.228662401, -0.0724624544, 0.00143249333, 0.0243871659, 0.263642758, 0.0182131939, -0.158881024, 0.0404700413, 0.0443066172, -0.0143134966, -0.19477348, 0.154209465, 0.295422077, 0.0469096228, 0.0014230815, 0.0603307709, -0.100078776, 2.92723762e-11, -0.129243225, -0.0621810667, -0.0144289099, -0.0279447213, 0.011072292, -0.0908636972, 0.0939676985, -0.13493675, -0.0072052558], [0.0913506523, -0.0341881886, 0.0489786267, 0.0609405264, 0.0280074459, -0.084932588, -0.0216444563, -0.0387368686, -0.301223665, -0.170854524, -0.0760792643, 0.23510173, 0.0110356761, 0.0971414149, -0.0853432417, -0.109218776, 2.03133599e-08, -0.160722107, 7.51997509e-12, -0.0142750107, -0.127792746, 0.0665703788, -0.411515206, 0.00781729538, 0.00055298052, 9.17673242e-05, -0.0178951696, -0.204132363, 0.0865703151, 0.101676553, 0.0961157903, -0.00135671452, -0.0778984278, -0.0567573309, -0.208377197, 0.070160836, -0.140298218, 0.105417252, -0.0351522267, 0.0725505948, 0.055566676, 0.1411504, -0.110257231, -0.0983067304, 0.164050892, 0.000237033179, 0.0920465291, -0.0112807434, -0.0613119602, -0.0701457411, -0.164803684, -0.128579721, 0.201093584, 0.12052355, -3.01252499e-06, 0.0527858399, -0.0775211826, -0.000866167771, 0.117234394, 0.050903935, 0.033768177, 0.0571281835, 0.183039978, -0.0846708268], [-0.149803087, -0.109893307, -2.33745941e-05, -0.0975229591, -0.128298998, -0.152346566, -0.0378440358, 0.0235135946, -0.0742664784, 0.0418881252, 0.114837252, -0.0327357948, -0.101236224, -0.120738365, -0.0411462039, -0.23099187, 1.39940572e-21, 0.0459976085, -2.8521271e-25, -0.320687562, -0.0829977021, -0.00107428874, 0.0390866213, 0.160824984, 0.1154266, -1.41618216e-06, -0.21795769, 0.13408947, 0.11575602, 0.161633894, 0.237788901, -0.00142654509, -0.116401769, -0.0433035307, -0.14555791, -0.0230219364, -0.243486479, -0.0850288048, 0.118981808, 0.0163326692, 0.19549033, -0.169612065, -0.0926365703, 0.24020806, 0.12415012, -0.0605954714, -0.223788694, -0.0236523971, -0.0333830267, -0.160429835, -0.0901247859, 0.00122692401, -0.168995768, 0.21362491, -7.76190934e-09, -0.0277999546, 0.0461630374, -0.00295920577, -0.118023992, -0.200833783, 0.000207751335, -0.0319705568, 0.27284649, -0.150361404], [-0.0747067258, 0.0380654968, 0.0391012914, 0.0279379878, 0.045607049, 0.0226380806, -0.0111900745, -0.211569533, -0.104957581, 0.0457100421, -0.149470747, -0.0689749643, -0.232421815, 0.222494766, 0.0139493318, -0.019105915, -6.46241325e-27, 0.0439803824, -4.06083907e-38, 0.0373748764, -0.294386178, -0.172723398, -0.18909353, -0.0053975489, 0.0875210315, 0.000322613487, 0.12312825, 0.152837604, -0.160915747, -0.0202989075, 0.099890247, -0.000870624266, 0.192974925, -0.13020508, -0.289749593, -0.172296166, 0.1191331, -0.0184762031, -0.144883901, -0.281276107, -0.178813398, 0.0694357008, 0.091537334, -0.107505493, 0.112765715, 0.0549548976, -0.184419379, -0.0763466209, 0.176665276, 0.220215231, -0.306043863, 0.00761366077, 0.0185684785, 0.154615164, -4.49612656e-07, -0.0964051038, -0.0955363363, -0.00226553297, 0.09686701, -0.215943232, -0.0824170783, 0.277907431, -0.0960199311, 0.144073978], [-0.189317957, 0.0142074982, -0.0366178714, 0.104768306, -0.0295253396, 0.129901588, 0.00689264573, 0.0722898468, -0.141387612, 0.0487774499, -0.048824776, 0.0799955875, 0.0961238667, 0.145495042, 0.0943729281, 0.0177071262, -4.64389972e-34, 0.0122399377, -9.18957011e-27, 0.14152582, -0.0720631331, -0.323646456, -0.13070184, -0.146446869, 0.0265918709, 0.000329526811, 0.00872731861, 0.0176853631, -0.114099205, 0.0454809256, 0.0992248207, -0.000996121089, -0.00326985866, -0.110319115, -0.215070397, -0.19445087, -0.0202884432, -0.151301906, -0.0235341135, -0.0858804882, -0.0263050385, 0.108544715, 0.0480951332, -0.153106675, -0.0370942242, -0.00317559927, -0.0939127803, 0.0670568496, 0.189083591, 0.100789823, -0.116852187, -0.0726994947, 0.033204969, 0.0307223853, -1.06557036e-06, -0.0665683225, -0.170399517, -0.0159706585, 0.169876635, 0.0798628479, 0.0238176696, -0.197141498, 0.122578107, -0.0649888441], [-0.263773233, -0.0200158078, -0.037189316, -0.0833711773, 0.0147294421, 0.0243028663, -0.00469298381, 0.185672656, 0.116319969, 0.0911221355, 0.141820639, -0.257212132, -0.162967086, 0.053715758, 0.0481684543, -0.120820515, 1.3916238e-11, -0.105622433, -9.47714698e-37, -0.148244485, 0.269585639, 0.0533140562, 0.12740697, 0.0319932662, -0.0427681766, -3.83314514e-19, -0.0219912417, 0.020284839, 0.102763094, -0.0111426422, 0.000141131793, -0.00215708348, -0.0445620939, 0.0129581634, 0.0990924463, 0.138789281, -0.0413218141, 0.101817437, 0.0854461268, 0.165886909, 0.187815636, -0.169085518, -0.0550452881, 0.00706086354, 0.08998169, -0.0784667805, -0.283581793, -0.0432381816, -0.144452393, -0.0216311738, 0.0851632506, 0.0380057991, -0.122216545, -0.0218765996, -9.94566562e-08, -0.0833954737, -0.00293252175, 0.0022817587, -0.058864627, -0.14619869, -0.0202013832, -0.0303273983, 0.102313556, -0.0613094606], [0.198812529, -0.123257831, -0.108769089, -0.266904682, 0.123675935, -0.241503775, -0.0539441705, -0.12781553, 0.194552734, 0.0227684062, 0.0985955521, -0.155285627, 0.201513961, 0.14478156, -0.12834242, -0.150774792, 1.42453594e-14, -0.129305959, 4.50146373e-27, -0.0741118789, 0.22449775, 0.273741782, 0.0900042653, 0.0102500767, -0.0323427208, -0.000918233185, -0.105723113, 0.131039321, 0.243714124, -0.100712389, -0.155714169, 0.00221584039, -0.0167108271, -0.0425725989, 0.159538537, 0.220660046, 0.0049935258, -0.1061977, 0.0550856255, -0.133829594, -0.0154901044, -0.143351808, -0.0681556538, 0.228044346, -0.100636132, -0.16483143, 0.16694276, -0.169758186, -0.209231853, 0.099326089, -0.0800622553, 0.075626798, -0.0935444161, -0.0836367235, -2.79382803e-05, 0.106465906, -0.0352022797, 0.00762941688, -0.0936461017, -0.159251109, -0.107092202, 0.240084633, -0.155977413, -0.226063713], [-0.268383533, -0.0291246083, 0.00976518728, -0.0200737566, 0.0656682104, -0.0261907727, -0.0232293103, 0.169896126, -0.145098701, -0.00499980105, -0.158679336, -0.194229767, -0.00564916572, -0.0916331783, 0.052965682, 0.0638364404, -4.79683355e-15, 0.0759695396, 2.10981737e-25, -0.00717017613, 0.0775216594, -0.312711865, -0.0278790146, -0.0305892751, 0.0335221402, -2.82748963e-10, -0.0619743504, 0.0538907871, -0.212047473, -0.00403065886, -0.0577034838, -0.000879663334, -0.0933151245, 0.122875348, 0.00401813257, -0.110264391, -0.0130773382, -0.130062804, -0.229861498, -0.0479667522, 0.0186781343, -0.1036314, -0.0868495852, -0.211519673, -0.0727896467, 0.000179199531, -0.347112358, -0.00632567378, -0.00161348109, -0.0213069934, -0.048518952, -0.0591419004, -0.124168091, -0.0285138823, -4.95729012e-07, -0.0134215634, 0.137775108, -0.00719185406, -0.0856353194, 0.112617515, -0.0687870756, 0.0751504079, -0.0629411936, 0.0159493815], [0.0669247434, 0.0676087812, -0.0554122292, 0.0635537207, -0.0277814418, -0.233152524, -0.0138988569, -0.321751267, -0.291127235, -0.0402868167, -0.0359008126, 0.13043429, 0.0557266809, 0.067507267, -0.141574502, -0.132125527, -2.58651082e-14, -0.0999039635, 5.10270515e-12, 0.132450953, -0.0660497695, 0.0050249584, -0.125089765, -0.103027105, 0.0793675035, 0.00095620018, 0.127487585, -0.189476728, 0.0473120175, -0.0493046455, 0.127943903, -0.0137165962, 0.054443907, -0.0441368371, -0.298790753, -0.0266550332, 0.121066555, -0.0817533359, 0.117033601, -0.168797225, -0.137420312, 0.0911289975, 0.0832686052, 0.0978417993, -0.0172860269, 0.0548537485, 0.126769155, -0.0755176768, 0.137434632, 0.188955426, -0.205607012, -0.24887903, -0.0656617135, 0.129142448, -5.0010327e-05, -0.0337644517, -0.0593076646, -0.0103033585, 0.0784658119, -0.132923335, -0.136896178, -0.212794796, 0.0480731167, 0.111493386], [-0.0551665686, 0.0621246099, 0.00449716346, -0.00409451593, 0.0674165413, -0.0388284065, -0.0170241371, 0.0608642325, 0.145664394, 0.0771661624, 0.0582816675, -0.156202778, 0.000950680871, -0.0737379193, 0.0229073651, -0.00710544176, 8.34098485e-11, -0.130023524, -3.41800733e-22, 0.180956334, -0.00818986725, -0.0480598509, 0.092209287, 0.125668272, -0.127716377, -4.51324178e-09, -0.0257662144, 0.129357651, 0.159188077, 0.0247033536, -0.126813576, -0.00160242978, -2.42889564e-06, -0.0332202613, -0.0225792807, -0.00671957852, -0.0183408055, -0.253667831, 0.137785599, 0.0227421075, -0.0531651862, -0.00971455127, 0.0563491993, 0.0378631949, -0.0332660899, -0.022651609, -0.0453634858, 0.0464426465, -0.0132382922, -0.0723404214, -0.187173694, 0.0530842841, 0.0112084989, -0.133750454, 7.02397713e-08, -0.0115603404, 0.062543124, -0.00587163679, -0.0585805289, 0.0801170394, 0.0625111088, 0.164719552, 0.00977331214, 0.113327377], [0.00456896378, -0.0540489145, -0.023160344, 0.116056703, -0.0476451553, 0.119050272, -0.013751911, -0.080846265, -0.0409476683, -0.0509338416, -0.0832495242, 0.291662484, -0.130536318, -0.169080302, 0.108741291, -0.162462682, -1.47869943e-13, 0.295292497, 2.51323196e-24, -0.0366677903, 0.0853848606, 0.125857502, 0.0141275041, 0.110698909, 0.10247425, 1.21195418e-08, -0.0380865261, -0.0833157152, 0.145706296, 0.0437457263, 0.199879944, -0.00196746946, -0.0265286397, -0.039193444, -0.00856616255, 0.0310975164, -0.128311619, 0.0764222741, 0.223903939, -0.0787972584, 0.171659648, -0.0203217622, -0.0459880568, 0.141592607, 0.048360955, 0.0179936755, -0.0110651208, -0.0498983115, -0.0986336321, -0.0738113746, 0.161319807, -0.0682525337, -0.127856731, 0.148725897, -1.18603658e-07, -0.047797367, 0.0614765547, -0.00686786324, 0.107846543, -0.00707442127, 0.0839971676, -0.07356444, 0.121113859, 0.0475240722], [0.0868227854, -0.142070904, -0.0756228566, 0.0190902464, -0.0302223135, 0.0506283268, -0.054051999, -0.00631100684, -0.102977857, -0.123948321, -0.069935225, -0.140873522, -0.155890658, -0.299809724, 0.0164208114, 0.0221039262, 2.855017e-12, -0.0483425073, -4.28016433e-20, 0.000662967912, -0.129947349, 0.0182771906, 0.11209511, -0.111846372, 0.0453499183, -2.08314104e-05, -0.0360457897, -0.0646236613, -0.0321225375, -0.0173101071, 0.153562829, -0.00209901459, -0.111057185, 0.0813459083, -0.22980696, 0.169659287, -0.0126282787, 0.0168579835, -0.10441573, -0.044922445, -0.0775148571, -0.0774283782, -0.0162225384, 0.208657324, -0.0561862439, 0.0196515694, -0.0250503905, -0.0484202504, -0.0154657038, -0.186380118, -0.0227188598, 0.196539551, -0.0947012678, 0.0534311421, 5.1909285e-07, 0.0386721827, 0.0638674498, -0.00894502737, -0.0714766905, 0.233453691, 0.0473357141, 0.218152836, 0.0648901984, 0.0496645272], [-0.0962568, 0.0533049442, -0.0273190681, -0.021679759, 0.0403849036, 0.146959856, 0.00386400567, 0.0152987223, 0.0306860898, 0.282522529, -0.0535354689, -0.0587168559, 0.0261316299, -0.0333018899, -0.0761450082, 0.0174213536, -4.11096969e-38, -0.0406994633, -1.47418756e-38, -0.0250962935, -0.118321367, 0.0403315313, 0.0906338766, 0.00620622467, 0.00896756165, -3.35366413e-12, 0.0942585245, 0.0583463721, -0.116485395, 0.127287105, -0.0250819828, -8.32772566e-05, 0.112641864, -0.0885773152, 0.0648015514, -0.208246648, -0.0111554144, 0.130436838, -0.118296936, -0.0627213493, -0.103795044, 0.110522538, -4.0037743e-05, -0.139122382, -0.0461567193, 0.00464526704, -0.19012329, 0.0140524469, 0.0676106587, 0.11463736, 0.117216833, 0.0836763307, 0.157266796, -0.051848527, -2.52171106e-09, -0.0722260103, -0.107938372, 0.007664809, 0.000741305586, -0.0545407236, 0.0905775353, 0.114458129, 0.00365621061, 0.109845378], [-0.304890633, 0.163866282, -0.0727475137, -0.0799656957, 0.0125599103, -0.248952851, -0.0263605323, -0.175458416, 0.190022051, -0.139365599, 0.0769051984, -0.25123468, -0.230242029, 0.0434911288, -0.0119465431, -0.145562753, 1.31217448e-09, -0.209710672, 1.15769342e-19, -0.173588708, -0.00269791554, -0.223953992, 0.0718914047, 0.04385956, -0.138290465, -9.67026248e-18, 0.0739292204, -0.0278040487, -0.0982442498, 0.0200731233, -0.0592052713, -0.00196254044, 0.0766186044, 0.0410736315, 0.252412647, -0.200518787, 0.0543540791, 0.0581594519, 0.159092858, 0.0515940115, 0.0295703541, -0.180367112, 0.0577280931, 0.017117355, -0.0055094976, 0.00919100642, -0.166874409, 0.0458072051, -0.171004921, -0.0210049823, 0.201547787, 0.073350437, -0.118074596, -0.112008572, -7.69816424e-08, -0.154184058, -0.0605767071, 0.0039112391, 0.0288883429, 0.0355580747, -0.00797106326, -0.0910798088, 0.0045364867, -0.168295652], [-0.437509567, -0.0961567312, 0.277130693, -0.066856809, -0.250895232, -0.0485667102, 0.00398242474, -0.000132499757, -0.357202023, -0.233281836, -0.0295730438, 0.231304988, -0.166285142, -0.177098855, 0.0425904803, -0.0991329327, 1.45156676e-09, 0.0682981685, 3.09422237e-19, -0.160188481, -0.16307281, -0.32669732, -0.401958793, -0.0294049978, -0.054872524, 4.5336727e-13, 0.0156869758, -0.223389715, -0.245710969, 0.0235792957, -0.276870728, -0.00130059628, 0.197422594, -0.119923897, -0.0467638187, -0.0234046355, -0.166897565, -0.12022686, 0.0166805387, -0.0477245934, -0.131355375, 0.0775931701, -0.0911472738, -0.321023136, -0.0405526608, -0.0127873709, -0.377642393, 0.233398959, -0.133968517, -0.129148617, -0.122995771, -0.188669086, -0.0353365466, -0.145106703, -2.11011741e-09, -0.106373735, -0.280215144, -0.0133128511, -0.145652622, 0.0898976028, 0.292562366, -0.0622101091, -0.240359217, -0.152375773], [-0.262622118, -0.0927403793, -0.0951554328, -0.014129566, -0.233391285, -0.0561706759, -0.0173203479, 0.0483752005, -0.238130897, -0.219089359, 0.14841415, 0.390264273, -0.293392926, -0.211188689, 0.111477725, -0.213929504, -3.08399008e-25, 0.159798294, -2.48077921e-38, 0.239008933, 0.116235234, 0.241002709, -0.160191894, 0.263858795, 0.121878572, -1.16847482e-12, 0.0921678245, -0.208466366, 0.0939980149, -0.0378097109, 0.0558603108, -0.000985174207, -0.00321334461, -0.0537657551, -0.228312746, -0.125624076, 0.0327712037, -0.221075609, 0.231967643, 0.453201681, 0.229702204, 0.145648494, 0.046353858, 0.0600940548, -0.0330751017, -0.000949671667, -0.163346604, -0.0594934635, 0.184493497, 0.00935501419, -0.123360462, -0.183396593, -0.136919081, 0.0278696883, -1.11213642e-10, -0.0552079491, -0.132349074, -0.0240955744, 0.0641660243, 0.0458813906, 0.0297841597, -0.0707430914, -0.0609707683, 0.146993935], [-0.12973845, -0.0912271142, 0.129942238, -0.122014746, -0.261905432, -0.21642904, -0.0238846727, -0.0607938841, 0.0498188287, -0.10263326, 0.0376448557, -0.207962155, -0.0809841901, -0.0351765603, 0.250262588, -0.116988249, -2.35837585e-25, -0.184606627, -8.74613598e-38, 0.0262576379, -0.0327698439, -0.227475882, -0.154671714, 0.0829281583, -0.175303951, -7.36874647e-38, -0.0216969792, -0.0630971938, -0.134511739, 0.193271369, -0.140380695, -0.00103207829, -0.187348008, -0.00804019067, -0.090859428, -0.076606065, -0.0506320968, 0.214333653, 0.143619478, -0.00216834946, 0.0228382219, 0.00850011967, 0.0148334028, -0.0841866657, 0.0629450008, -0.0383813679, -0.189299658, 0.125267848, -0.084070906, -0.00625097845, -0.0466253795, -0.145959198, -0.0969878882, 0.0260505863, 4.75134279e-08, 0.00503222086, 0.0438664071, -0.00371774961, -0.0640346333, 0.246266827, 0.19251585, 0.215813354, -0.172218099, -0.0436017103], [0.0564676784, -0.106092371, -0.0562984757, -0.0491088778, 0.0314689279, -0.00871619955, -0.00365516543, 0.00384388329, -0.0797625557, 0.083108753, -0.106665172, 0.0488315634, -0.0672419667, 0.0234213565, -0.0394652411, -0.0715468898, -9.16058871e-27, 0.0417695306, -2.87867399e-25, -0.141041949, -0.161939427, 0.0300050657, -0.0198072847, -0.15019764, 0.105189055, -3.42002454e-06, -0.0459463857, 0.0218385737, 0.0694439337, -0.0707012415, 0.149882957, -0.0018834169, -0.143237993, -0.0404664911, -0.128031984, -0.087143667, -0.0844850615, 0.0171969067, -0.0381558649, -0.00857255328, -0.284279019, -0.13141945, -0.121059142, 0.158001095, 0.0314168558, -0.0221061576, -0.0714430586, -0.0508088022, -0.0692927241, -0.11597205, 0.0580675267, 0.128331959, 0.116895109, 0.176033199, -3.29684696e-07, -0.0472507402, -0.031116927, 0.00565831084, -0.0204150788, -0.193482548, -0.0812538862, -0.142119467, 0.0520892739, -0.0488799028], [0.153041095, -0.00991695467, -0.0635418147, -0.0833819211, -0.0521509387, -0.0743361339, 0.0108323256, -0.101724766, 0.107770301, 0.00549712591, 0.0425524227, -0.27375865, -0.199991673, 0.0323668756, -0.0219192579, 0.101522535, -1.05188843e-24, -0.0998323411, -2.63018742e-32, 0.0423244573, 0.0322880559, 0.00129501265, 0.0292186383, -0.0989990383, -0.117697708, 2.78015546e-06, 0.0908941254, 0.225792274, 0.0850892738, -0.0813372657, -0.14432855, -0.00172706321, 0.0616670549, 0.0367976688, 0.0653166696, 0.189267114, -0.0467683785, 0.0893490613, -0.122068815, 0.00933531206, -0.0905976743, -0.0602139831, -0.0537541211, -0.00456841895, 0.00669888454, -0.0190027636, 0.0833856761, 0.00178747787, -0.155316263, -0.0537787192, 0.0165180899, 0.127745017, 0.0249096192, -0.0369032249, -1.29578424e-07, 0.0991604254, 0.0775667727, 0.00300934585, -0.107647225, 0.138490155, 0.025230797, 0.131598741, -0.0894404054, -0.113601632], [0.0704797134, 0.055197265, -0.00312207686, -0.0804331377, -0.0425098203, -0.104288667, -0.0180901606, -0.00923928991, 0.205983087, -0.0924694538, 0.0850894749, -0.132520735, 0.113121584, -0.0257283933, -0.107845649, 0.174252376, -1.08250873e-34, -0.0794628412, -5.46041843e-37, -0.105137378, -0.116352871, -0.0721226633, -0.0514280647, 0.0825707838, -0.0436867811, 1.68879287e-05, 0.139699489, -0.139616102, -0.00962904934, -0.0767703876, -0.0955204442, -0.00108611863, 0.0396169536, -0.147074789, 0.132646993, 0.171161622, -0.0782482028, -0.00898379553, 0.143581539, -0.0858006999, -0.216518447, -0.113596179, -0.0206195693, 0.0394616425, -0.058760915, -0.0608897991, 0.160103202, 0.0365577489, -0.169561386, -0.061570283, 0.0656141117, 0.0316660851, 0.0325463079, -0.0342637375, -3.44301696e-07, 0.00258338638, -0.0901562199, -0.00291910162, 0.0258999672, -0.00614829268, 0.040419098, 0.0108864158, 0.161488906, -0.0118035516], [0.114061579, 0.0510360748, -0.0450337455, -0.00397568289, 0.071148254, -0.0209640134, 0.0120699899, 0.0271659736, 0.0913636461, 0.107745379, -0.181025118, -0.0388651937, 0.0585042313, 0.0879381597, 0.0202104244, 0.0590042472, 3.88302897e-23, 0.00259671872, -5.019302e-38, 0.100723676, -0.0631345659, -0.116323315, -0.0119827464, -0.0536141545, -0.123536646, 2.47652406e-05, 0.0462416634, -0.019649718, -0.0908711776, -0.0389963873, -0.246032119, -0.000678514189, -0.0225224011, -0.0863319859, 0.0014039994, 0.111331351, 0.11415042, 0.0368677825, -0.212616324, -0.0708172023, 0.0709700286, 0.18836318, 0.139441654, 0.0177007578, -0.0480138436, 0.0172399804, 0.0159358568, -0.114314564, 0.0614890382, 0.182188496, -0.0687047988, -0.137561485, -0.0214852504, -0.200177804, -1.24805908e-07, 0.023262728, 0.0829697251, -0.00360745494, -0.00151612749, -0.0568843931, -0.00474651903, -0.131327122, 0.119700253, 0.0641859472], [0.0531852022, -0.194134519, -0.0220049601, -0.0291440096, -0.111991741, 0.194895148, -0.0322130173, 0.00646215165, -0.250960767, -0.0501627624, 0.00620410824, 0.00622679107, -0.0346013494, -0.0205630846, 0.0131152784, 0.0124000339, 7.54332304e-38, -0.0840983838, -2.89908025e-38, 0.0723720193, 0.113369107, 0.112609588, -0.0954588726, -0.0896792039, -0.130141705, -3.71266736e-07, -0.0123476712, -0.2571612, 0.075699918, -0.104404777, -0.131176889, -0.000911752402, -0.0221792348, 0.0545672067, -0.0687406361, 0.0666035637, 0.124365918, -0.087018393, 0.0299238339, -0.154005632, -0.0275483634, 0.129484802, 0.0242896955, 0.00052832911, 0.00955863297, 0.0322496183, 0.0168385115, -0.0270879194, 0.0380884707, -0.0148782097, 0.0868810117, -0.22272706, -0.0392840169, -0.0712413266, 1.37558729e-07, 0.0333530717, 0.0154629629, -0.00934067555, 0.148066103, 0.011669809, -0.0199031681, -0.226945534, -0.175230518, 0.0107001504], [0.061743252, 0.143853053, -0.0781265721, -0.0674916431, 0.221572295, -0.091211684, 0.0366306454, 0.0599721186, 0.0159318689, 0.0877592415, -0.0915181041, -0.125682995, -0.104580656, 0.231409982, -0.0879485756, -0.0155293476, -2.30730238e-38, -0.0798355117, -2.80610466e-37, 0.243064493, 0.0363046192, 0.117391258, 0.0526508614, -0.0799109787, -0.158409953, -4.24584323e-05, 0.072228469, 0.0320867971, -0.102999017, -0.28123197, -0.174119785, -0.00135495025, 0.261965752, -0.123949617, 0.108674146, 0.170011863, 0.15342848, 0.0751932785, 0.030419942, 0.131641716, -0.115974426, 0.0975675285, 0.173860937, -0.0568307452, -0.09355665, 0.0131823756, 0.225807607, -0.0856349319, 0.16715005, 0.224471539, 0.0855253413, -0.0473823473, 0.201368749, -0.107108116, -4.7754304e-07, -0.0440440513, -0.213676766, -0.0130518572, 0.143675908, -0.0754702613, -0.0781508088, -0.176637441, -0.0359921902, 0.0932790935], [0.0161703061, -0.0375987254, 0.000672288763, -0.0427293628, 0.039054554, 0.100382194, 0.0155506255, 0.178873226, -0.236021563, -0.104426257, -0.0959388167, 0.17691651, -0.0336556695, -0.0295336861, 0.124198593, -0.033483956, -1.26573965e-37, 0.0828631744, -2.60010061e-38, -0.0297721885, -0.0360330343, -0.162798807, -0.140174061, -0.0852772892, -0.0452592745, -3.16446822e-05, 0.0503909811, -0.066055119, 0.0393030979, -0.0339680612, -0.106840834, -0.000605969282, 0.0771207437, 0.0284702331, -0.178523749, -0.148832351, -0.0130677465, 0.00869092532, -0.12486276, -0.0878401771, -0.0359395854, 0.0171328969, 0.122454211, -0.0613198616, -0.0400706939, 0.0411174968, -0.114415847, 0.0140060587, 0.112821952, 0.0581482835, -0.0389482789, -0.107236274, 0.0190933812, -0.152970001, -1.56330742e-07, -0.0863705948, -0.0510352552, -0.00144298503, 0.0358414575, -0.023206884, -0.011560156, -0.0334818587, -0.102053829, 0.136169896], [0.125742882, -0.0198814794, -0.0815411583, -0.0576481968, -0.177492782, 0.472993851, -0.00137637288, -0.0614070892, 0.164660901, -0.0490565747, -0.00794442184, -0.55230701, 0.0160101876, -0.122271419, -1.10348475, 0.16963239, -1.58446191e-38, 0.446925908, -2.046742e-38, -0.0326779112, -0.00874042511, -0.0373409539, -0.0517582893, -0.0332822539, 0.338728428, 4.73477523e-36, 0.143081546, 0.237211615, -0.0741044879, -0.461337894, -0.354040176, -0.000212693354, 0.335032314, -0.192924917, -0.0610689893, 0.12561518, -0.0218427312, 0.216344953, -0.195104808, -0.214412391, -0.515896797, -0.200514942, -0.0592322983, -0.100324906, -0.0979207382, -0.0169355664, 0.197191805, -0.00562843215, 0.346083254, -0.0395862572, -0.0780135989, 0.582463205, 0.123074748, -0.20455277, -1.88810013e-17, 0.0341906585, -0.20302628, 0.00736180739, 0.421004683, -1.5029583, -0.18583329, 0.0417193659, -0.140555963, 0.289917171], [0.0480370335, 0.122341029, -0.0236642603, -0.04835096, -0.126276597, 0.00586295826, -0.0112294108, -0.0663406178, 0.108841002, -0.0500750802, -0.12107037, -0.060673587, -0.0369803533, -0.306530893, -0.110380806, 0.120672449, -3.97979974e-38, -0.27052924, -4.5099799e-38, -0.172367096, -0.24454388, -0.260146379, -0.0858197585, -0.100799978, -0.0808850005, -5.70400438e-08, -0.16996415, -0.19527483, -0.0389618799, -0.0272422042, -0.155235678, -0.000262329588, -0.00824531168, -0.0380158536, -0.107214309, 0.0652940497, -0.0216884837, -0.202841699, -0.139116243, -0.22875312, -0.313535035, -0.154414549, -0.0882911235, -0.0677625537, -0.103478864, -0.0461778156, -0.0766696706, 0.0563436039, -0.214641273, -0.122600056, 0.198111549, 0.195323139, -0.152894482, -0.181889951, -8.08155818e-08, -0.027701281, -0.00243197591, -0.00198601349, 0.0918692052, 0.0673788264, -0.0660955682, -0.299712569, -0.0400915928, -0.193938464], [-0.0141297141, -0.225681752, -0.0654364228, -0.0807613432, -0.0785674378, -0.234098971, 0.00756352022, -0.630720258, -0.236014336, -0.148883164, -0.0583291836, -0.0334118232, -0.2704207, -0.0579769649, -0.280888826, 0.127787501, 2.57815095e-38, -0.11442174, -1.38430534e-38, -0.061671555, -0.142363459, 0.00948563218, -0.145152196, -0.119739801, -0.0579151958, -3.85585475e-09, -0.117629483, -0.200098515, -0.111468039, -0.112890251, -0.206302866, -0.000322746026, -0.154829741, -0.0415403768, 0.00568330102, -0.0172181036, -0.0164534021, 0.00124819181, -0.0283550713, -0.200528026, -0.161820203, -0.149569705, -0.00160647032, -0.0982077345, -0.00710071996, -0.010168192, 0.0869306102, 0.0394899845, -0.0866499022, -0.158103198, 0.18768084, -0.131449014, 0.0192961693, -0.108253442, 2.21251087e-10, -0.0271792002, 5.09106249e-05, 0.00588842481, -0.156664714, -0.057122238, -0.0714603886, 0.0705920756, -0.0438702255, -0.127564073], [-0.00390311866, 0.105874524, 0.0173388533, 0.124107286, -0.312663347, -0.0074792807, 0.000938073266, 0.0649208054, 0.0802735835, 0.0898287818, 0.0446908325, 0.150786847, 0.042821005, 0.0195620973, -0.139032081, 0.0355087668, -1.42561237e-24, -0.124570876, 3.57905455e-30, -0.0806871876, -0.065092057, 0.00462487852, 0.0128989322, -0.126037076, -0.135027617, 9.39896841e-11, 0.0718048513, 0.280204743, -0.0855160281, -0.116383165, -0.152619109, -0.00108420267, 0.234735221, 0.029284779, 0.0665251762, -0.172335342, -0.053590972, -0.0476759076, -0.152034387, 0.0268100016, -0.0631716028, 0.260992646, -0.0222089998, -0.110084713, 0.108650014, 0.00812380947, 0.062765412, -0.069363229, 0.0376528762, 0.168251902, 0.204171419, -0.0849044323, -0.0414361022, -0.103565767, -2.04474553e-08, -0.117800698, -0.0622467846, 0.021503035, -0.000477079098, -0.000652063871, 0.0402273946, 0.264838576, -0.141874671, -0.0549405776], [-0.253721148, -0.0859988034, -0.00409253547, -0.0341066755, -0.101043537, 0.144443884, -0.046410054, 0.0872422829, -0.172194228, -0.128066108, -0.0612764694, 0.303393453, -0.0716506764, -0.129986778, 0.149809122, -0.0682272837, -2.65374978e-15, 0.0470672809, -1.24964626e-28, 0.0358824432, -0.0829956979, -0.105087519, -0.0345340408, 0.0692844465, 0.0640023276, -1.31207753e-06, 0.0462385081, -0.0528045371, -0.0419892669, 0.132757246, 0.153754011, -0.00164379145, 0.133320034, -0.00755024469, -0.178687021, -0.129196987, 0.00601758482, -0.0470357873, 0.0524490178, -0.0812152028, 0.0882035568, 0.104809612, 0.141915545, -0.023422299, 0.13106817, 0.0247472282, -0.134666532, -0.0557164922, 0.171341971, 0.0817222521, -0.148471996, -0.0154528599, 0.0434253924, 0.163008913, -1.92468704e-07, -0.0341321044, 0.0884231105, -0.0100540835, 0.0966271907, 0.127563193, 0.0257254802, -0.0642589405, 0.32314077, 0.150736511]]], 'bH': [[0.116047159, 0.0097970143, -0.167638078, -0.0922843963, -0.0420115516, -0.0768154263, 0.0177700762, -0.253903866, -0.0732318833, -0.114335969, -0.0984329507, 0.0331975445, 0.0663741082, -0.0914870128, -0.202565163, -0.0328788273, -0.000357328099, -0.0597337447, -3.6752017e-06, -0.0632732213, -0.165149167, -0.0470769033, -0.194484785, -0.0878099799, -0.0718576461, -0.0206100065, 0.102392085, -0.026213387, -0.0130125051, -0.0982950851, -0.0765399337, -4.76488913e-06, 0.115214899, -0.0462372005, -0.0501097217, -0.0104130069, -0.0571318753, -0.329569668, -0.088824071, -0.122180574, -0.10543555, -0.00895369984, -0.0942689553, -0.0497102067, -0.049060408, 0.00605523167, 0.0892772898, -0.238083497, 0.103274196, -0.131969258, -0.0723571777, -0.0849867165, -0.0656809062, -0.171703234, -4.77849389e-05, -0.0274215322, -0.172198951, -0.028007729, 0.0501228608, -0.121616744, -0.227102295, -0.159892753, -0.185777083, 0.0677426085]], 'W2': [[-0.367580831], [0.101344615], [-0.084135212], [0.134583935], [0.109836832], [-0.183643773], [0.00141654781], [-1.03933859], [0.255187213], [0.234178081], [-0.122024246], [0.407976538], [-0.321263611], [0.237293437], [-0.203366339], [-0.136107713], [-4.5010998e-07], [-0.148675457], [3.27838507e-07], [0.12526089], [-0.132027611], [-0.341258854], [0.283348531], [-0.100365043], [-0.0517622009], [-0.00226098625], [0.0445573479], [0.103337154], [-0.078205727], [-0.0452744626], [-0.186535537], [0.000410916371], [0.0761171579], [-0.0736530349], [0.140968919], [-0.121648714], [0.115100771], [0.228500247], [-0.144010603], [-0.206797659], [-0.152505234], [0.0843634009], [0.178021953], [-0.106496871], [-0.0863413066], [0.0424490459], [-0.335087627], [-0.174288094], [0.0961845443], [0.105199993], [0.075532563], [0.106948197], [-0.110853374], [-0.136778891], [-3.89040952e-06], [-0.0835670084], [-0.114388235], [0.0117354561], [0.0631199554], [-0.185822025], [-0.100090019], [0.156364068], [-0.196949273], [0.146911711]], 'rs': [1.91471827e-05, 2.69792963e-05, 3.28858441e-05, 3.63987274e-05, 4.24730133e-05, 4.55358058e-05, 4.73305154e-05, 5.06284341e-05, 5.23282106e-05, 5.36597436e-05, 5.64615802e-05, 5.85703237e-05, 5.98213883e-05, 7.10609311e-05], 'c2': [-1.73258385e-08, -2.44129374e-08, -2.97576328e-08, -3.29363594e-08, -3.84328409e-08, -4.1204288e-08, -4.28282796e-08, -4.5812488e-08, -4.7350575e-08, -4.85554494e-08, -5.10907654e-08, -5.29989208e-08, -5.41309788e-08, -6.43013749e-08], 'st': [1], 'xf': [2]}


def _nlhz_build(f32):
    # nlhz_patch: head tables as float32 arrays (setup; data independent)
    P = _NLHZ_P
    t = lambda k: fnp.asarray(P[k], dtype=f32)
    nl = len(P['rs'])
    return dict(M=fnp.reshape(t('M'), (nl, 1, -1)), IS=fnp.reshape(t('IS'), (nl, 1, -1)), W0=t('W0'),
                B0=fnp.reshape(t('B0'), (nl, 1, -1)), WH=[fnp.asarray(w, dtype=f32) for w in P['WH']],
                bH=[fnp.asarray(b, dtype=f32) for b in P['bH']], W2=t('W2'),
                rs=fnp.reshape(t('rs'), (nl, 1, 1)), c2=fnp.reshape(t('c2'), (nl, 1, 1)),
                **({'STB': fnp.zeros((nl, 1024, 5), dtype=f32)} if P['st'][0] else {}),
                **({'NLX': fnp.zeros((16, 1024, int(P['xf'][0])), dtype=f32)} if P['xf'][0] else {}))


def _nlhz_x(T, li, nr, C, M, D, w1, w2, w3, D3, var):
    # nlhz_patch v5 (lean v4): only the trained groups (D) of the per-row pair reductions of layer li
    # (alive order, as the h6 riders), normalized per row exactly as in training
    if 'NLX' not in T or D is None:
        return
    isg = 1.0 / fnp.sqrt(var)
    DW = fnp.stack([fnp.matmul(D, w2), fnp.matmul(w1, D)], axis=1)
    X = DW * isg[:, None]
    fnp.copyto(T['NLX'][li, :nr], X)


def _nlhz_apply(H, hz_W):
    # nlhz_patch: sl_out[2:] += rs * MLP(x) on the rider slabs of layers 2..15 (padded entries stay finite and are never read)
    T = H['nl']
    g = lambda k: H['sl_' + k][2:]
    vs = g('var')
    sgs = fnp.sqrt(vs)
    als = g('mu') / sgs
    a2s = als * als
    ps = fnp.exp(a2s * -0.5)
    vsg = vs * sgs
    v2 = vs * vs
    pq = (a2s - 1.0) * ps / vsg
    aps = als * ps
    D3 = g('D3')
    same = g('same')
    c0 = aps / vs * D3
    c2_ = ps * sgs
    c4_ = pq * g('port')
    psv = ps / vsg
    cols = [c0, c0 * a2s, c2_, c2_ * a2s, c4_, c4_ * a2s, pq * g('g4'), psv * same, psv * g('proj'), psv * g('star'),
            pq * D3 * D3 / vs, pq * same, pq * g('h6_d21_w1'), pq * g('h6_d21_e'), pq * g('h6_d21_s'), pq * g('h6_d21_d3'),
            aps / vsg * g('h6_k31b_s'),
            als, fnp.log(vs), D3 / vsg, g('g4') / v2, g('port') / v2, same / v2, g('proj') / v2, g('star') / v2,
            g('h6_d21_d3') / vsg, g('h6_d21_e') / vsg, g('h6_d21_s') / vsg, g('h6_d21_w1') / vsg, g('h6_k31b_s') / v2]
    X = fnp.stack(cols, axis=-1)
    if 'STB' in T:
        # carried state of layer li (rows k = li - 2 >= 1): layer li-1 slab quantities pushed through hz_W[li] o hz_W[li] (4) and
        # hz_W[li] (1), in the alive orders of the two layers (hz_W[li] = alive(li) x alive(li-1), as in the final transport below)
        STB = T['STB']
        vp = H['sl_var'][1:15]
        sgp = fnp.sqrt(vp)
        ap_ = H['sl_mu'][1:15] / sgp
        pp = fnp.exp(ap_ * ap_ * -0.5) * 0.3989422804014327
        c2p = pp / sgp
        q5 = c2p * H['sl_D3'][1:15] / vp
        Q = fnp.stack([c2p * vp, c2p * c2p * vp * vp, pp * ap_ * H['sl_g4'][1:15] / (vp * vp), q5], axis=-1)
        for k in range(1, 14):
            Wl = hz_W[k + 2]
            nr_, ni_ = int(Wl.shape[0]), int(Wl.shape[1])
            S4 = fnp.matmul(Wl * Wl, Q[k, :ni_])
            S1 = fnp.matmul(Wl, q5[k, :ni_])
            fnp.copyto(STB[k, :nr_], fnp.concatenate([S4, S1[:, None]], axis=1))
        X = fnp.concatenate([X, STB], axis=-1)
    if 'NLX' in T:
        X = fnp.concatenate([X, T['NLX'][2:]], axis=-1)
    X = fnp.clip((X - T['M']) * T['IS'], -8.0, 8.0)
    h1 = fnp.maximum(fnp.matmul(X, T['W0']) + T['B0'], 0.0)
    for Wh_, bh_ in zip(T['WH'], T['bH']):
        h1 = fnp.maximum(fnp.matmul(h1, Wh_) + bh_, 0.0)
    o = fnp.matmul(h1, T['W2']) * T['rs'] + T['c2']
    so = H['sl_out'][2:]
    fnp.add(so, o[:, :, 0], out=so)
    if 'NLX' in T:
        fnp.copyto(T['NLX'], 0.0)   # v4: next predict starts from zeros (layers without the h6 pair matrices stay 0, as in training)
# per-layer weights: 0-6 kappa4 replacement on the selected rows (same, proj, star, kappa3^2/var, V29 kappa4, chain, induced), 7-13 riders
HZ_TH = [None, None, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.00965048, 0.00509454, -0.00067129, 0.00073822, -0.25536233, 0.00198566, 0.25396761, -0.06984506, 0.30297273, -0.13943155, 0.50563632, -7.882565, -0.00392036, 0.435901, 0.00198137, -0.1008335, -0.00040321], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01303944, 0.0064136, -0.00109647, 0.00109389, -0.45715199, 0.01048685, 0.26851621, -0.10931284, 0.15389302, -0.3076948, 0.71091697, -5.98780522, -0.00390642, 0.26060132, 0.00193539, -0.04743624, -0.0004162], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01543828, 0.00740564, -0.00152657, 0.00134756, -0.58176057, 0.00847044, 0.26208571, -0.11768141, 0.06288382, -0.40687176, 0.7462373, -4.70329838, -0.00373156, 0.20744508, 0.00165393, -0.0348674, -8.824e-05], [0.48500658, 0.27008239, 1.48967692, -0.68813829, -0.06862843, 0.77698809, 0.74465612, -0.01700042, 0.00734758, -0.00146854, 0.00119198, -0.54945354, 0.00579471, 0.36038521, -0.07768778, 0.01306824, -0.31205938, 0.86711791, -3.57752723, -0.00331518, 0.12656727, 0.00139265, -0.01952997, 0.00011923], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02074791, 0.01113232, -0.00232417, 0.00194567, -0.65634407, 0.01314608, 0.31758945, -0.12084459, 0.01719318, -0.37632954, 0.93820025, -4.14803692, -0.0045957, 0.14201518, 0.00180037, -0.01932745, -0.00076851], [0.45368007, 0.37235639, 1.50425309, -0.65589963, -0.01084065, 0.78243794, 0.90505533, -0.01836969, 0.00902399, -0.00271942, 0.00217162, -0.60618028, 0.00939897, 0.33204374, -0.07788603, -0.00514303, -0.33791405, 0.89654498, -3.04116857, -0.00529359, 0.10265496, 0.00198007, -0.00975818, -0.00096928], [0.4370018, 0.31881063, 1.46943383, -0.63182337, -0.041163, 0.82289008, 0.9581055, -0.01941282, 0.00900959, -0.00242666, 0.0018009, -0.53808212, 0.00386315, 0.34114653, -0.05208262, -0.01616902, -0.29312336, 1.01465777, -2.17841227, -0.00376924, 0.06489264, 0.00113321, -0.00611277, -0.00083983], [0.44797717, 0.32401889, 1.46234403, -0.62683868, -0.01216478, 0.80689585, 1.00394963, -0.02121185, 0.00930724, -0.00279912, 0.00212505, -0.52753935, 0.00459262, 0.32364681, -0.06240391, -0.02421977, -0.29355559, 0.93924185, -2.21116982, -0.00412276, 0.05956813, 0.00131371, -0.00596894, -0.00024516], [0.38208791, 0.24526578, 1.40445097, -0.52008852, -0.01333572, 0.83774679, 1.02711397, -0.01946201, 0.00821823, -0.00270284, 0.00197426, -0.46338453, 0.00129598, 0.35381931, -0.02917711, -0.04771147, -0.25451583, 0.91367036, -1.90751227, -0.00366693, 0.04345813, 0.00124604, -0.00424485, 1.815e-05], [0.36348622, 0.18876759, 1.34513642, -0.5846754, 0.01225102, 0.83984672, 1.03956551, -0.02137723, 0.01017735, -0.00241505, 0.00164877, -0.40009368, 0.00028886, 0.32610511, -0.01694202, -0.07927722, -0.24920818, 1.06575512, -1.57486725, -0.00246499, 0.02836984, 0.00061242, -0.00309961, -3.871e-05], [0.37191761, 0.2164077, 1.33859289, -0.53891097, 0.00701313, 0.82913471, 1.07480295, -0.02109682, 0.00981657, -0.00271172, 0.00184747, -0.39622684, -0.0006939, 0.325855, 0.00130969, -0.05661911, -0.21794044, 0.99537599, -1.38662093, -0.00275399, 0.02506642, 0.00073372, -0.00238215, 0.00020174], [0.314303, 0.1791678, 1.25868475, -0.59483287, 0.01627128, 0.82902251, 1.08915941, -0.02077287, 0.01125617, -0.00319537, 0.0022721, -0.37374366, 0.00121855, 0.32329976, 0.0077614, -0.06583442, -0.21690719, 1.05236836, -1.71972771, -0.00317342, 0.02638599, 0.00083314, -0.00202756, 0.00028048], [0.3107437, 0.11183431, 1.20347745, -0.61530494, 0.02078186, 0.82583396, 1.09779853, -0.02288991, 0.01268754, -0.00300231, 0.00209718, -0.29841278, 0.00095502, 0.29013322, 0.0286609, -0.090915, -0.19183769, 1.10844681, -1.65891814, -0.00301796, 0.01919143, 0.00114179, -0.00163426, 0.00122376], [0.29233066, 0.19446663, 1.32908094, -0.58463042, -0.09931708, 0.86399959, 1.12718624, -0.01405445, 0.00901172, -0.00207789, 0.00117627, -0.34176289, -0.01787237, 0.25420241, 0.01879674, -0.03353689, -0.15562552, 1.21135933, -0.25844379, 0.0, 0.0, 0.0, 0.0, 0.0]]
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
TDP_C = float(_os.environ.get('TDP_C', '0.5'))   # tdp_patch: previous-layer band alpha in (-SDT_C, -TDP_C] of the w1-scaled transports (0 = off)
TDP_ON = TDP_C > 0
TDPCAP2 = _os.environ.get('TDPCAP2', '1') == '1'   # tdpcap_patch: TDP join window left-out cap nr // 2 instead of _sdt_win's nr // 4
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


MEHLER3 = _os.environ.get('MEHLER3', '1') == '1'   # mehler3_patch: Gaussian Mehler m = 3 pair terms C_off^3 / 3! w(3,p) w(3,q) in Q11 / Q21 / Q22
MEHLER3_S = float(_os.environ.get('MEHLER3_S', '1.0'))   # mehler3_patch: strength (1.0 = exact 1 / 3! coefficient)
assert WICK_PAIRS.index((3, 2)) == WICK_PAIRS.index((3, 1)) + 1   # mehler3_patch: w(3,1), w(3,2) are adjacent Wick table rows
TM_CDPS = _os.environ.get('TM_CDPS', '1') == '1'       # termite: cdp U^T x[order] by a scattered U (no 4/elem row gather)
TM_FBW = _os.environ.get('TM_FBW', '1') == '1'         # termite: fbmix feedback block on the group's hub-column window
TM_RREUSE = _os.environ.get('TM_RREUSE', '1') == '1'   # termite: DSF R = rowdot(P, A o C1 + P o C2) (one einsum instead of two)
TM_BNC = _os.environ.get('TM_BNC', '1') == '1'         # termite: birth feedback range finder on D21[:, :nc]
TM_CC = _os.environ.get('TM_CC', '1') == '1'           # termite: per-predict constant tables built once and reused (calls only)
TM_DQF = _os.environ.get('TM_DQF', '1') == '1'         # termite: D21 projection loop matmul + sum folded into one einsum (calls only)
TM_DS0 = _os.environ.get('TM_DS0', '1') == '1'         # termite: _dslices slot-0 bracket (exact zeros) skipped
TM_LS0 = _os.environ.get('TM_LS0', '1') == '1'         # termite: LOT response_star without the zero slot 0
TM_K22MV = _os.environ.get('TM_K22MV', '1') == '1'     # termite: K22 row sums through the K21 matvec
TM_NBKIN = _os.environ.get('TM_NBKIN', '1') == '1'     # termite: newborn feedback transport on the kin live input rows
TM_LOTHOT = _os.environ.get('TM_LOTHOT', '1') == '1'   # termite: LOT KU old-tier einsums on the rebuilt rows [p:]
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
MID_EDGE_K4S = float(_os.environ.get('MID_EDGE_K4S', '1.2'))
MID_EDGE_S3S = float(_os.environ.get('MID_EDGE_S3S', '1.4'))
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
# costx peel_patch: exact-fringe planning (core on the 2^Lc grid + fringe strips instead of padding); see agents_ideas2/costx
PEEL = _os.environ.get('PEEL', '1') == '1'
PEEL_GAIN_MIN = float(_os.environ.get('PEEL_GAIN_MIN', '3e6'))
PEEL_OPW = float(_os.environ.get('PEEL_OPW', '1e5'))
PEEL_MN = int(_os.environ.get('PEEL_MN', '2'))
PEEL_SLEV = int(_os.environ.get('PEEL_SLEV', '5'))
PEEL_LMAX = int(_os.environ.get('PEEL_LMAX', '7'))
PEEL_GATE = _os.environ.get('PEEL_GATE', '1') == '1'
PEEL_DEBUG = _os.environ.get('PEEL_DEBUG', '0') == '1'
PEEL_LOG = []
_PEEL_READY = [None]


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
HZ_TMAX = float(_os.environ.get('HZ_TMAX', '107.5'))       # agents_robust: skip _hz_finish once the predict has run > HZ_TMAX s (0 = off)
HZ_TMAX_L = float(_os.environ.get('HZ_TMAX_L', '103.5'))   # agents_robust: per-layer guard: no HZ work (nor _hz_finish) once > HZ_TMAX_L s (0 = off)
HZ_TMAX_P = float(_os.environ.get('HZ_TMAX_P', '117'))   # agents_robust: abort the _hz_finish pullback (no-HZ result) once > HZ_TMAX_P s (0 = off)
import time as _rb_time
WG = _os.environ.get('WG', '1') == '1'   # wguard_patch: projected-wall degrade ladder (HZ off -> D21 off -> last-layer Gaussian)
WG_ON = WG                              # (guard3 bool: off-suite predicts never arm the ladder)
WG_F = [float(_x) for _x in _os.environ.get('WG_F', '0.000,0.002,0.032,0.078,0.137,0.204,0.296,0.352,0.412,0.476,0.542,0.614,0.689,0.755,0.823,0.894,0.928').split(',')]   # wguard: mean fraction of the predict wall elapsed at the top of layer li (16 = before _hz_finish)
WG_L0 = int(_os.environ.get('WG_L0', '12'))        # wguard: first layer whose check may fire rung 0 (HZ off)
WG_W0 = float(_os.environ.get('WG_W0', '116'))     # wguard: projected wall (s) above which rung 0 fires
WG_L1 = int(_os.environ.get('WG_L1', '12'))        # wguard: first layer whose check may fire rung 1 (D21 off from there)
WG_CAP = float(_os.environ.get('WG_CAP', '118'))   # wguard: rung 1 target wall (s)
WG_H = float(_os.environ.get('WG_H', '0.06'))      # wguard: share of the wall saved by the finish skip (rung 0)
WG_D = float(_os.environ.get('WG_D', '0.027'))     # wguard: share of the wall saved per D21-off layer (rung 1)
WG_CAP2 = float(_os.environ.get('WG_CAP2', '118'))  # wguard: rung 2 target wall (s), checked at the top of the final layer
WG_TEST_SCALE = float(_os.environ.get('WG_TEST_SCALE', '1'))   # wguard TEST: elapsed multiplier
WG_FORCE = _os.environ.get('WG_FORCE', '')   # wguard TEST: '<li>:<rung>' forces the rung from layer li on
WG_ANY = _os.environ.get('WG_ANY', '0') == '1'   # wguard TEST: arm without a flopscope client context (in-process)
if len(WG_F) != 17 or min(WG_F[11:]) <= 0.0:
    WG_ON = False   # wguard: malformed fraction table (bad bake) -> the ladder stays off (never a crash)


def _wg_cwall():
    """wguard: live wall (s) of the open flopscope CLIENT BudgetContext, None if absent (no flopscope op, no round trip)"""
    try:
        import flopscope._budget as _wg_fb   # already loaded: a sys.modules lookup
        c = getattr(_wg_fb, '_active_context', None)   # client only: in-process flopscope has no _active_context -> None
        if c is None:
            return None
        w = c.wall_time_s
        return None if w is None else float(w)
    except Exception:
        return None


def _wg_rung(est, li):
    """wguard: rung after the check at the top of layer li (16 = right before _hz_finish); rungs only climb"""
    try:
        e = (_rb_time.perf_counter() - est._rb_t0 + est._wg_off) * WG_TEST_SCALE
        p = e / max(WG_F[li], 1e-3)   # projected wall of this predict (HZ on, no rung)
        k = -1
        if li >= WG_L0 and p > WG_W0:
            k = 0
        if WG_L1 <= li <= 14 and p * (1.0 - WG_H - WG_D * (14 - li)) > WG_CAP:
            k = 1
        if li == 15 and p * (1.0 - WG_H) > WG_CAP2:
            k = 2
        if est._wg_fl is not None and li >= est._wg_fl[0]:
            k = max(k, est._wg_fl[1])
        if k > est._wg_r:
            est._wg_r = k
            est._wg_log.append((li, k, round(e, 2), round(p, 1)))
    except Exception:
        pass   # the decision itself can never fail a predict (keeps the current rung)
    return est._wg_r
RGUARD = _os.environ.get('RGUARD', '1') == '1'   # rguard_patch: residual-aware HZ skip (live client residual of this predict)
RGUARD_ON = RGUARD                                # (guard3 bool: off-suite predicts never read the residual)
RGUARD_L = int(_os.environ.get('RGUARD_L', '9'))       # rguard: first layer whose top-of-layer check may drop the HZ work
RGUARD_R = float(_os.environ.get('RGUARD_R', '0.33'))  # rguard: live residual (s) threshold at the top of layers >= RGUARD_L
RGUARD_F = float(_os.environ.get('RGUARD_F', '0.335'))  # rguard: live residual (s) threshold right before _hz_finish


def _rg_resid():
    """rguard: live billed residual (s) of the open predict BudgetContext, None if not readable (no flopscope op, no round trip)"""
    try:
        import flopscope._budget as _rg_fb   # the flopscope CLIENT's budget module (already loaded: a sys.modules lookup)
        c = getattr(_rg_fb, '_active_context', None)   # client only: in-process flopscope has no _active_context -> None
        if c is None:
            return None
        r = c.residual_wall_time_s
        return None if r is None else float(r)
    except Exception:
        return None
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
_POOL_SPEC = {"pool": [["get", "wd", [1024, 1024]], ["get", "l", [15, 1024, 10]], ["get", "t1", [1024, 1024]], ["get", "hub", [1024, 1024]], ["get", "d21inner", [1, 1024, 1024]], ["get", "d21x", [31, 1, 1024, 1024]], ["get", "pk2", [3, 1024, 1024]], ["get", "pk11", [1024, 1024]], ["get", "d21new", [1024, 1024]], ["get", "k21", [1024, 1024]], ["get", "c", [1024, 1024]], ["get", "s21", [1024, 1024]], ["get", "ssep", [1024, 1024]], ["get", "om_r", [1024, 8]], ["get", "y", [1024, 8]], ["get", "y2", [1024, 8]], ["get", "bm", [8, 1024]], ["get", "rr", [1024, 10]], ["get", ["z", 0], [15, 1024, 10]], ["get", "ww", [1024, 1024]], ["get", "wk4m", [1024, 1024]], ["get", "lot_y", [1024, 12]], ["get", "lot_y2", [1024, 12]], ["get", "lot_gu", [15, 1024, 12]], ["get", "qu", [1024, 224]], ["get", "d21w", [1024, 1024]], ["get", "rep21", [1024, 1024]], ["get", ["z", 1], [15, 1024, 10]], ["get", "cpre", [1024, 1024]], ["get", ["qc", 1], [1024, 448]], ["get", "xtb", [1024, 1024]], ["get", "ytb", [1024, 1024]], ["get", "x1b", [1024, 1024]], ["get", "y1b", [1024, 1024]], ["get", "omf", [1024, 16]], ["get", "r1t", [15, 1024, 16]], ["get", "r2t", [15, 1024, 16]], ["get", ["zf", 0], [15, 1024, 32]], ["get", "sym4_x", [10, 1, 256, 1024]], ["get", "sym4_y", [10, 1, 1024, 256]], ["get", "sym4_o", [10, 1, 256, 256]], ["get", "c7_wc", [1024, 1024]], ["get", "inner", [1, 1024, 448]], ["get", ["qc", 0], [1024, 448]], ["get", ["zf", 1], [15, 1024, 32]], ["get", "qp", [1024, 448]], ["get", "thin_o", [1024, 1024]], ["get", "dn_lc", [1, 1, 1024, 704]], ["get", "dn_rc", [1, 1, 1024, 704]], ["get", "dn_oc", [1, 1024, 1024]], ["get", "dn_zi", [1, 1, 1024, 704]], ["get", "dn_zo", [1, 1, 1024, 704]], ["get", "inner2", [1, 1024, 224]], ["pair", ["fap2", 0], [15, 2, 224, 1024]], ["get", "ap", [15, 1024, 1024]], ["get", "pp", [15, 1024, 1024]], ["get", "t", [15, 1024, 1024]], ["get", "mp", [15, 1024, 1024]], ["get", "xt", [15, 1024, 1024]], ["get", "yt", [15, 1024, 1024]], ["get", "u", [15, 1024, 1024]], ["get", "ppl", [15, 1024, 10]], ["get", "d21", [1024, 1024]], ["get", "d21fb", [1024, 1024]], ["get", "gyr", [15, 1024, 16]], ["get", "gxr", [15, 1024, 16]], ["get", "au_t1x", [9, 1, 1024, 1024]], ["get", "sym3_x", [6, 1, 320, 1024]], ["get", "sym3_y", [6, 1, 1024, 320]], ["get", "sym3_o", [6, 1, 320, 320]], ["get", "dap", [2, 1, 1, 1024]], ["get", "abbuf", [52, 1024, 1024]], ["get", "lot_slab", [15, 3, 1024]], ["pair", "AP0", [15, 2, 1024, 1024]], ["pair", "AP1", [15, 2, 1024, 1024]], ["get", "om", [1024, 448]], ["get", "yq", [1024, 448]], ["get", "yq1", [1024, 448]], ["get", "yq2", [1024, 448]], ["get", "y1t", [2, 1, 448, 1024]], ["get", "yqh", [1, 1024, 448]], ["pair", ["fap", 1], [15, 2, 448, 1024]], ["get", "fg", [448, 2048]], ["get", "lot_t", [1024, 12]], ["get", "dn_g0", [1024, 1024]], ["get", "dn_fsub", [448, 224]], ["get", "dn_frot", [2, 1, 448, 1024]], ["get", "dn_qnv", [1024, 448]], ["get", ["stage_x", 1024], [1, 1, 1024, 1024]]], "roles": [["tx", 60236288], ["ty", 60236288], [["hx", 99], 271063296], [["x", 99], 30118144], [["y", 99], 240945152]], "bmax": 30, "plans": [["mm", 1, 8, 8, 8, 8, 7, 0], ["hub", 1, 7, 8, 7, 8, 7, 0], ["mm", 1, 7, 7, 7, 8, 7, 0], ["mm", 1, 8, 8, 8, 6, 7, 0], ["mm", 1, 7, 7, 7, 6, 7, 0], ["mm", 1, 15, 15, 16, 8, 6, 0], ["mm", 1, 8, 8, 8, 4, 7, 0], ["hub", 1, 8, 8, 8, 4, 7, 0], ["mm", 1, 14, 13, 14, 8, 6, 0], ["mm", 1, 13, 7, 16, 8, 6, 0], ["mm", 1, 14, 13, 13, 6, 6, 0], ["mm", 1, 8, 8, 8, 2, 7, 0], ["hub", 1, 8, 8, 8, 2, 7, 0], ["hub", 1, 10, 14, 7, 9, 6, 0], ["mm", 1, 15, 7, 16, 4, 6, 0], ["mm", 1, 14, 13, 14, 2, 6, 0], ["mm", 1, 16, 7, 16, 2, 6, 0], ["mm", 10, 8, 32, 8, 1, 5, 0], ["mm", 10, 7, 30, 7, 1, 5, 0], ["mm", 10, 7, 29, 7, 1, 5, 0], ["mm", 1, 7, 10, 16, 2, 6, 0], ["mm", 6, 10, 32, 10, 1, 5, 0], ["mm", 10, 7, 28, 7, 1, 5, 0], ["mm", 6, 10, 30, 10, 1, 5, 0], ["mm", 1, 7, 10, 14, 2, 6, 0], ["mm", 1, 15, 15, 7, 1, 6, 0], ["mm", 1, 15, 14, 7, 1, 6, 0], ["mm", 1, 14, 14, 7, 1, 6, 0], ["mm", 1, 13, 7, 14, 1, 6, 0], ["mm", 1, 14, 13, 7, 1, 6, 0], ["mm", 1, 27, 7, 32, 2, 5, 0], ["mm", 1, 12, 7, 14, 1, 6, 0], ["mm", 1, 2, 28, 64, 8, 4, 0], ["mm", 1, 10, 7, 13, 1, 6, 0], ["mm", 1, 14, 14, 32, 2, 5, 0], ["mm", 1, 14, 28, 28, 1, 5, 0], ["mm", 1, 16, 32, 16, 1, 5, 0], ["mm", 1, 60, 4, 64, 1, 4, 0], ["mm", 1, 56, 52, 14, 1, 4, 0], ["mm", 1, 14, 14, 64, 2, 4, 0], ["mm", 1, 14, 14, 60, 2, 4, 0], ["mm", 1, 12, 128, 128, 1, 3, 0], ["mm", 1, 120, 8, 120, 1, 3, 0], ["mm", 1, 16, 120, 120, 1, 3, 0], ["mm", 1, 20, 116, 116, 1, 3, 0], ["mm", 1, 12, 120, 120, 1, 3, 0], ["mm", 1, 16, 116, 116, 1, 3, 0], ["mm", 1, 20, 112, 112, 1, 3, 0], ["mm", 1, 116, 112, 14, 1, 3, 0], ["mm", 1, 28, 28, 14, 1, 4, 0], ["mm", 1, 24, 7, 32, 12, 5, 0], ["hub", 1, 12, 16, 7, 6, 6, 0], ["hub", 1, 13, 16, 7, 4, 6, 0], ["hub", 1, 20, 32, 7, 6, 5, 0], ["mm", 1, 7, 10, 15, 2, 6, 0], ["hub", 1, 22, 32, 7, 2, 5, 0], ["mm", 1, 28, 28, 2, 8, 5, 0], ["hub", 1, 28, 11, 24, 1, 5, 0], ["mm", 1, 28, 26, 9, 1, 5, 0], ["mm", 1, 28, 26, 8, 1, 5, 0], ["hub", 1, 58, 14, 52, 1, 4, 0], ["mm", 1, 112, 8, 104, 1, 3, 0], ["mm", 1, 112, 112, 8, 1, 3, 0], ["hub", 1, 5, 7, 6, 8, 7, 0], ["mm", 1, 12, 7, 13, 9, 6, 0], ["hub", 1, 11, 16, 7, 8, 6, 0], ["mm", 1, 7, 7, 8, 2, 7, 0], ["hub", 1, 9, 13, 7, 9, 6, 0], ["hub", 1, 9, 14, 12, 4, 6, 0], ["hub", 1, 9, 13, 12, 4, 6, 0], ["hub", 1, 18, 32, 7, 10, 5, 0], ["mm", 1, 27, 7, 32, 6, 5, 0], ["mm", 1, 2, 28, 30, 6, 5, 0], ["hub", 1, 14, 16, 7, 2, 6, 0], ["mm", 1, 2, 56, 60, 8, 4, 0], ["mm", 1, 2, 56, 56, 8, 4, 0], ["mm", 1, 11, 7, 13, 1, 6, 0], ["mm", 1, 2, 28, 32, 2, 5, 0], ["mm", 1, 9, 7, 12, 1, 6, 0], ["mm", 1, 28, 28, 2, 6, 5, 0], ["hub", 1, 56, 19, 48, 1, 4, 0], ["mm", 1, 56, 6, 56, 1, 4, 0], ["mm", 1, 116, 8, 120, 1, 3, 0], ["hub", 1, 116, 17, 104, 1, 3, 0], ["hub", 1, 8, 8, 8, 6, 7, 0], ["hub", 1, 5, 7, 6, 6, 7, 0], ["hub", 1, 6, 8, 6, 4, 7, 0], ["hub", 1, 12, 16, 7, 8, 6, 0], ["mm", 1, 15, 7, 16, 6, 6, 0], ["hub", 1, 12, 15, 13, 6, 6, 0], ["hub", 1, 11, 14, 7, 9, 6, 0], ["hub", 1, 12, 15, 13, 4, 6, 0], ["mm", 1, 24, 7, 32, 10, 5, 0], ["hub", 1, 6, 8, 6, 2, 7, 0], ["hub", 1, 18, 32, 7, 12, 5, 0], ["mm", 1, 7, 16, 16, 2, 6, 0], ["hub", 1, 16, 16, 7, 2, 6, 0], ["mm", 1, 27, 7, 32, 4, 5, 0], ["mm", 1, 16, 16, 7, 1, 6, 0], ["mm", 1, 2, 52, 56, 8, 4, 0], ["hub", 1, 24, 32, 2, 4, 5, 0], ["hub", 1, 24, 32, 2, 2, 5, 0], ["mm", 1, 32, 14, 14, 1, 5, 0], ["hub", 1, 56, 16, 52, 1, 4, 0], ["hub", 1, 120, 14, 112, 1, 3, 0], ["hub", 1, 116, 22, 104, 1, 3, 0], ["mm", 1, 116, 8, 112, 1, 3, 0], ["hub", 1, 13, 16, 13, 8, 6, 0], ["mm", 1, 15, 7, 16, 8, 6, 0], ["mm", 1, 14, 13, 13, 8, 6, 0], ["mm", 1, 14, 7, 16, 8, 6, 0], ["mm", 1, 13, 7, 15, 8, 6, 0], ["mm", 1, 13, 7, 13, 9, 6, 0], ["mm", 1, 12, 7, 15, 8, 6, 0], ["mm", 1, 12, 7, 14, 8, 6, 0], ["mm", 1, 15, 15, 16, 4, 6, 0], ["mm", 1, 15, 15, 15, 4, 6, 0], ["mm", 1, 10, 7, 14, 8, 6, 0], ["hub", 1, 10, 15, 7, 8, 6, 0], ["mm", 1, 7, 7, 7, 2, 7, 0], ["hub", 1, 9, 13, 12, 6, 6, 0], ["mm", 1, 20, 7, 32, 12, 5, 0], ["mm", 1, 26, 7, 32, 8, 5, 0], ["mm", 1, 24, 7, 32, 8, 5, 0], ["hub", 1, 2, 32, 28, 8, 5, 0], ["hub", 1, 18, 32, 7, 8, 5, 0], ["hub", 1, 9, 14, 12, 2, 6, 0], ["hub", 1, 22, 32, 7, 4, 5, 0], ["hub", 1, 20, 32, 7, 4, 5, 0], ["mm", 1, 2, 56, 56, 6, 4, 0], ["mm", 1, 11, 7, 14, 1, 6, 0], ["mm", 1, 2, 28, 60, 8, 4, 0], ["mm", 1, 9, 7, 13, 1, 6, 0], ["mm", 1, 28, 28, 2, 2, 5, 0], ["mm", 1, 2, 56, 60, 2, 4, 0], ["mm", 1, 58, 6, 56, 1, 4, 0], ["mm", 1, 116, 112, 22, 1, 3, 0], ["mm", 1, 116, 104, 22, 1, 3, 0], ["mm", 1, 116, 112, 17, 1, 3, 0], ["mm", 1, 116, 120, 8, 1, 3, 0], ["hub", 1, 6, 8, 7, 8, 7, 0], ["hub", 1, 7, 8, 8, 6, 7, 0], ["hub", 1, 15, 16, 15, 8, 6, 0], ["mm", 1, 10, 12, 13, 10, 6, 0], ["mm", 1, 10, 13, 13, 8, 6, 0], ["mm", 1, 14, 7, 16, 6, 6, 0], ["hub", 1, 11, 14, 13, 6, 6, 0], ["hub", 1, 9, 15, 7, 8, 6, 0], ["mm", 1, 21, 7, 32, 12, 5, 0], ["hub", 1, 11, 15, 13, 4, 6, 0], ["hub", 1, 11, 15, 12, 4, 6, 0], ["hub", 1, 11, 14, 13, 4, 6, 0], ["hub", 1, 11, 14, 12, 4, 6, 0], ["hub", 1, 10, 14, 12, 4, 6, 0], ["mm", 1, 2, 28, 30, 8, 5, 0], ["hub", 1, 20, 32, 7, 8, 5, 0], ["hub", 1, 2, 32, 32, 6, 5, 0], ["hub", 1, 11, 15, 13, 2, 6, 0], ["mm", 1, 7, 15, 16, 2, 6, 0], ["hub", 1, 11, 15, 12, 2, 6, 0], ["hub", 1, 15, 16, 7, 2, 6, 0], ["mm", 1, 10, 13, 14, 2, 6, 0], ["mm", 1, 7, 11, 16, 2, 6, 0], ["hub", 1, 2, 28, 24, 6, 5, 0], ["mm", 1, 7, 9, 14, 2, 6, 0], ["mm", 1, 16, 15, 7, 1, 6, 0], ["mm", 1, 2, 52, 52, 8, 4, 0], ["mm", 1, 14, 7, 15, 1, 6, 0], ["mm", 1, 14, 7, 14, 1, 6, 0], ["mm", 1, 30, 14, 14, 1, 5, 0], ["mm", 1, 2, 52, 56, 2, 4, 0], ["mm", 1, 64, 4, 64, 1, 4, 0], ["hub", 1, 56, 16, 48, 1, 4, 0], ["mm", 1, 56, 6, 52, 1, 4, 0], ["mm", 1, 56, 58, 6, 1, 4, 0], ["hub", 1, 120, 14, 104, 1, 3, 0], ["mm", 1, 112, 8, 112, 1, 3, 0], ["mm", 1, 84, 112, 8, 1, 3, 0], ["mm", 1, 15, 15, 15, 6, 6, 0], ["hub", 1, 12, 15, 13, 8, 6, 0], ["mm", 1, 14, 7, 14, 9, 6, 0], ["mm", 1, 12, 7, 16, 8, 6, 0], ["mm", 1, 14, 13, 14, 6, 6, 0], ["mm", 1, 10, 7, 15, 8, 6, 0], ["hub", 1, 11, 15, 13, 6, 6, 0], ["hub", 1, 9, 13, 11, 8, 6, 0], ["hub", 1, 8, 13, 11, 8, 6, 0], ["hub", 1, 9, 14, 7, 8, 6, 0], ["hub", 1, 13, 16, 13, 4, 6, 0], ["mm", 1, 26, 7, 32, 10, 5, 0], ["hub", 1, 13, 15, 13, 4, 6, 0], ["hub", 1, 8, 14, 7, 8, 6, 0], ["hub", 1, 9, 13, 11, 6, 6, 0], ["hub", 1, 10, 13, 12, 4, 6, 0], ["mm", 1, 15, 15, 16, 2, 6, 0], ["hub", 1, 16, 32, 7, 12, 5, 0], ["mm", 1, 15, 14, 15, 2, 6, 0], ["mm", 1, 14, 13, 15, 2, 6, 0], ["mm", 1, 25, 7, 32, 6, 5, 0], ["hub", 1, 11, 16, 13, 2, 6, 0], ["hub", 1, 10, 15, 12, 2, 6, 0], ["mm", 1, 28, 7, 32, 4, 5, 0], ["hub", 1, 13, 16, 7, 2, 6, 0], ["mm", 1, 2, 28, 28, 6, 5, 0], ["hub", 1, 9, 14, 11, 2, 6, 0], ["mm", 6, 10, 29, 10, 1, 5, 0], ["mm", 1, 7, 9, 13, 2, 6, 0], ["mm", 1, 2, 52, 56, 6, 4, 0], ["mm", 1, 12, 7, 13, 1, 6, 0], ["mm", 1, 10, 13, 7, 1, 6, 0], ["mm", 1, 10, 7, 12, 1, 6, 0], ["mm", 1, 10, 12, 7, 1, 6, 0], ["mm", 1, 8, 7, 12, 1, 6, 0], ["mm", 1, 21, 26, 14, 1, 5, 0], ["mm", 1, 21, 26, 11, 1, 5, 0], ["mm", 1, 20, 26, 11, 1, 5, 0], ["mm", 1, 2, 52, 60, 2, 4, 0], ["mm", 1, 40, 48, 22, 1, 4, 0], ["mm", 1, 40, 8, 52, 1, 4, 0], ["hub", 1, 116, 17, 112, 1, 3, 0], ["mm", 1, 120, 112, 17, 1, 3, 0], ["mm", 1, 116, 104, 17, 1, 3, 0], ["mm", 1, 7, 6, 6, 8, 7, 0], ["mm", 1, 7, 7, 8, 4, 7, 0], ["mm", 1, 10, 13, 13, 10, 6, 0], ["mm", 1, 7, 7, 7, 4, 7, 0], ["hub", 1, 6, 8, 7, 4, 7, 0], ["mm", 1, 11, 13, 13, 8, 6, 0], ["hub", 1, 6, 7, 6, 4, 7, 0], ["mm", 1, 11, 7, 15, 8, 6, 0], ["hub", 1, 9, 14, 12, 8, 6, 0], ["hub", 1, 9, 13, 12, 8, 6, 0], ["mm", 1, 10, 7, 13, 9, 6, 0], ["hub", 1, 10, 16, 7, 8, 6, 0], ["mm", 1, 26, 7, 32, 12, 5, 0], ["hub", 1, 9, 14, 7, 9, 6, 0], ["hub", 1, 10, 13, 7, 9, 6, 0], ["hub", 1, 13, 16, 7, 6, 6, 0], ["mm", 1, 23, 7, 32, 12, 5, 0], ["hub", 1, 8, 15, 7, 8, 6, 0], ["hub", 1, 10, 13, 12, 6, 6, 0], ["hub", 1, 13, 15, 14, 4, 6, 0], ["mm", 1, 27, 7, 32, 10, 5, 0], ["hub", 1, 8, 13, 7, 9, 6, 0], ["hub", 1, 8, 13, 12, 6, 6, 0], ["hub", 1, 20, 32, 7, 12, 5, 0], ["hub", 1, 10, 15, 12, 4, 6, 0], ["hub", 1, 12, 16, 7, 4, 6, 0], ["hub", 1, 20, 32, 7, 10, 5, 0], ["hub", 1, 16, 32, 7, 10, 5, 0], ["mm", 1, 11, 13, 14, 2, 6, 0], ["hub", 1, 16, 32, 7, 8, 5, 0], ["hub", 1, 10, 14, 12, 2, 6, 0], ["hub", 1, 18, 32, 7, 6, 5, 0], ["mm", 1, 25, 7, 32, 4, 5, 0], ["hub", 1, 8, 14, 12, 2, 6, 0], ["mm", 1, 7, 9, 15, 2, 6, 0], ["mm", 1, 2, 28, 32, 4, 5, 0], ["hub", 1, 2, 32, 28, 4, 5, 0], ["mm", 1, 2, 48, 52, 10, 4, 0], ["mm", 1, 2, 28, 30, 4, 5, 0], ["mm", 1, 13, 7, 15, 1, 6, 0], ["hub", 1, 24, 28, 2, 4, 5, 0], ["mm", 1, 14, 12, 7, 1, 6, 0], ["mm", 1, 11, 13, 7, 1, 6, 0], ["mm", 1, 2, 28, 56, 8, 4, 0], ["hub", 1, 20, 32, 7, 2, 5, 0], ["mm", 1, 28, 24, 2, 8, 5, 0], ["hub", 1, 28, 11, 26, 1, 5, 0], ["mm", 1, 28, 28, 2, 4, 5, 0], ["mm", 1, 29, 26, 8, 1, 5, 0], ["mm", 1, 21, 24, 14, 1, 5, 0], ["mm", 1, 28, 24, 9, 1, 5, 0], ["mm", 1, 23, 26, 11, 1, 5, 0], ["mm", 1, 21, 24, 11, 1, 5, 0], ["hub", 1, 58, 19, 52, 1, 4, 0], ["hub", 1, 56, 19, 52, 1, 4, 0], ["hub", 1, 58, 14, 48, 1, 4, 0], ["mm", 1, 56, 8, 52, 1, 4, 0], ["mm", 1, 42, 8, 52, 1, 4, 0], ["hub", 1, 120, 22, 112, 1, 3, 0], ["hub", 1, 120, 22, 104, 1, 3, 0], ["hub", 1, 116, 22, 96, 1, 3, 0], ["mm", 1, 92, 8, 104, 1, 3, 0], ["mm", 1, 92, 112, 8, 1, 3, 0]]}

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

    @staticmethod
    def _peel_cost(kind, bx, by, m, kd, w, L, k1):
        """peel_patch: billed FLOPs of one product at level L (0 = dense matmul 2mnk - mn per product, hub + the slot sum)"""
        hub = kind == 'hub'
        if L <= 0:
            return by * m * w * (2 * kd - 1) + ((by - 1) * m * w if hub else 0)
        B0, S_ = (1, by) if hub or bx == 1 else (by, 1)
        return _lx_cost(kind, B0, m >> L, kd >> L, w >> L, S_, L, 2 if k1 else 1, k1)

    @staticmethod
    def _peel_key(kind, bx, by, m, kd, w, L, k1):
        """peel_patch: the _plan key of a product at level L (as _mm_prep / _hub_prep build it)"""
        if kind == 'hub':
            return ('hub', 1, m >> L, kd >> L, w >> L, by, L, k1)
        B0, S_ = (1, by) if bx == 1 else (by, 1)
        return ('mm', B0, m >> L, kd >> L, w >> L, S_, L, 0)

    def _peel_ready(self, key):
        """peel_patch PEEL_GATE (run time only: LXPGATE on): the key is in the static setup plan list"""
        if not (PEEL_GATE and LXPGATE):
            return True
        r = _PEEL_READY[0]
        if r is None:
            r = _PEEL_READY[0] = frozenset((tuple((int(v) if not isinstance(v, str) else v for v in k_)) + ((0,) if len(k_) == 7 else ()) for k_ in list(_POOL_SPEC['plans']) + _lx_plans(1024)))
        return key in r

    def _peel_slev(self, kind, bx, by, m, kd, w, k1):
        """peel_patch: (level, billed FLOPs, ops) of one fringe strip: the cheapest (FLOPs + PEEL_OPW x ops) of the dense matmul (plain
        products only) and the levels <= PEEL_SLEV with every side divisible, blocks >= PEEL_MN, a plan that fits the reserved roles and
        (gated) is in the spec; None = no admissible way"""
        hub = kind == 'hub'
        best = None if hub else (0, self._peel_cost(kind, bx, by, m, kd, w, 0, k1), 1)
        B0, S_ = (1, by) if hub or bx == 1 else (by, 1)
        for L in range(1, PEEL_SLEV + 1):
            d = 1 << L
            if m % d or kd % d or w % d or min(m, kd, w) // d < PEEL_MN:
                break
            if not self._fits(kind, B0, m >> L, kd >> L, w >> L, S_, L, 2 if k1 else 1):
                continue
            if not self._peel_ready(self._peel_key(kind, bx, by, m, kd, w, L, k1)):
                continue
            c = self._peel_cost(kind, bx, by, m, kd, w, L, k1)
            o = 15 * L + 20
            if best is None or c + PEEL_OPW * o < best[1] + PEEL_OPW * best[2]:
                best = (L, c, o)
        return best

    def _peel_plan(self, kind, bx, by, m, kd, w, lev, k1, capm, capk, capw, xcapm, xcapk, stage):
        """peel_patch: None (keep the parent path) or (Lcore, mb, wb, ((axis, level), ...), parent cost, peel cost, added ops) for one
        call signature (the arguments _mm_prep / _hub_prep pass to _choose); cached like _choose"""
        key = (kind, bx, by, m, kd, w, lev, k1, capm, capk, capw, xcapm, xcapk, stage, self.mn, self.deep, self.role_epoch)
        pc = self.__dict__.get('_peel_c')
        if pc is None:
            pc = self._peel_c = {}
        if key in pc:
            return pc[key]
        hub = kind == 'hub'
        L, m2, kd2, w2, st = self._choose(kind, bx, by, m, kd, w, lev, k1, capm, capk, capw, xcapm, xcapk, stage)
        res = None
        if L >= 3:
            c0 = self._peel_cost(kind, bx, by, m2, kd2, w2, L, k1) + (m * kd * bx if st else 0)
            best = None
            cst = (not hub) and self.deep > 0 and bx == 1
            for Lc in range(L, min(L + 1, PEEL_LMAX) + 1):
                d = 1 << Lc
                mb, wb = (m // d * d, w // d * d)
                if kd % d or (mb, wb) == (m, w) or min(mb, kd, wb) // d < 4:
                    continue
                Lk, a_, b_, c_, s_ = self._choose(kind, bx, by, mb, kd, wb, lev, k1, mb, kd, wb, mb, kd, cst)
                if Lk < 3 or Lk > PEEL_LMAX or (a_, b_, c_, s_) != (mb, kd, wb, False):
                    continue
                if not self._peel_ready(self._peel_key(kind, bx, by, mb, kd, wb, Lk, k1)):
                    continue
                f = self._peel_cost(kind, bx, by, mb, kd, wb, Lk, k1)
                ops = 19 * (Lk - L) - (1 if st else 0)
                strips = []
                if mb < m:
                    sr = self._peel_slev(kind, bx, by, m - mb, kd, w, k1)
                    if sr is None:
                        continue
                    f += sr[1]
                    ops += sr[2]
                    strips.append(('r', sr[0]))
                if wb < w:
                    sc = self._peel_slev(kind, bx, by, mb, kd, w - wb, k1)
                    if sc is None:
                        continue
                    f += sc[1]
                    ops += sc[2]
                    strips.append(('c', sc[0]))
                g = c0 - f - PEEL_OPW * ops
                if g >= PEEL_GAIN_MIN and (best is None or g > best[0]):
                    best = (g, Lk, mb, wb, tuple(strips), c0, f, ops)
            if best is not None:
                res = best[1:]
        if PEEL_DEBUG:
            PEEL_LOG.append((kind, bx, by, m, kd, w, lev, k1, self.mn, (L, m2, kd2, w2, st), res))
        if len(pc) >= 4096:
            pc.clear()
        pc[key] = res
        return res

    def _peel_try(self, meth, X, Y, out, lev, k1, pad):
        """peel_patch: plan (same capacities as _mm_prep / _hub_prep) and run core + strips; False = parent path"""
        try:
            Xp, Yp, Op = (None, None, None) if pad is None else pad
            if meth == 'mm':
                bx, P, m, kd = X.shape
                by, w = (Y.shape[0], Y.shape[3])
                if P != 1:
                    return False
                stage = self.deep > 0 and bx == 1
                xcapm, xcapk = (m, kd) if Xp is None else (Xp.shape[-2], Xp.shape[-1])
                big = 1 << 30
                capm = min(big if stage else xcapm, m if Op is None else Op.shape[-2])
                capk = min(big if stage else xcapk, kd if Yp is None else Yp.shape[-2])
                capw = min(w if Yp is None else Yp.shape[-1], w if Op is None else Op.shape[-1])
                plan = self._peel_plan('mm', bx, by, m, kd, w, lev, 0, capm, capk, capw, xcapm, xcapk, stage)
            else:
                k, P, m, kd = X.shape
                w = Y.shape[2]
                if P != 1:
                    return False
                xcapm, xcapk = (m, kd) if Xp is None else (Xp.shape[-2], Xp.shape[-1])
                capm = min(xcapm, m if Op is None else Op.shape[-2])
                capw = min(w if Yp is None else Yp.shape[-2], w if Op is None else Op.shape[-1])
                plan = self._peel_plan('hub', k, k, m, kd, w, lev, k1, capm, kd, capw, xcapm, xcapk, False)
        except Exception:
            if FMM_STRICT:
                raise
            return False
        if plan is None:
            return False
        Lc, mb, wb, strips = plan[:4]
        hub = meth != 'mm'
        w = Y.shape[2] if hub else Y.shape[3]

        def call(Xa, Ya, Oa, lv):
            if meth == 'mm':
                self.mm(Xa, Ya, Oa, lv)
            elif meth == 'hub':
                self.hub(Xa, Ya, Oa, lv)
            else:
                self.hubg(Xa, Ya, Oa, lv, k1)

        def ycols(a, b):
            return _pview(Y, (Ellipsis, slice(a, b, None), slice(None, None, None))) if hub else _pview(Y, (Ellipsis, slice(None, None, None), slice(a, b, None)))
        Xc = X if mb == m else _pview(X, (Ellipsis, slice(None, mb, None), slice(None, None, None)))
        Yc = Y if wb == w else ycols(None, wb)
        Oc = out if (mb, wb) == (m, w) else _pview(out, (Ellipsis, slice(None, mb, None), slice(None, wb, None)))
        self._peel_busy = True
        try:
            call(Xc, Yc, Oc, lev)
            dp, mn0 = (self.deep, self.mn)
            self.deep, self.mn = (0, PEEL_MN)
            try:
                for ax, Ls in strips:
                    if ax == 'r':
                        call(_pview(X, (Ellipsis, slice(mb, None, None), slice(None, None, None))), Y, _pview(out, (Ellipsis, slice(mb, None, None), slice(None, None, None))), Ls)
                    else:
                        call(Xc, ycols(wb, None), _pview(out, (Ellipsis, slice(None, mb, None), slice(wb, None, None))), Ls)
            finally:
                self.deep, self.mn = (dp, mn0)
        finally:
            self._peel_busy = False
        return True

    def mm(self, X, Y, out, lev, pad=None):
        """plain: X (bx, 1, m, kd), Y (by, 1, kd, w) -> out (by, 1, m, w); bx in {1, by}"""
        if PEEL and self.deep > 0 and (not self.__dict__.get('_peel_busy')) and self._peel_try('mm', X, Y, out, lev, 0, pad):
            return
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
        if PEEL and self.deep > 0 and (not self.__dict__.get('_peel_busy')) and self._peel_try('hubg', X, Y, out, lev, k1, pad):
            return
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
        if PEEL and self.deep > 0 and (not self.__dict__.get('_peel_busy')) and self._peel_try('hub', X, Y, out, lev, 0, pad):
            return
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
FBL_B = int(_os.environ.get('FBL_B', '9'))     # fblate_patch: first birth with the late feedback rank FBL_R (-1 = off)
FBL_R = int(_os.environ.get('FBL_R', '16'))     # fblate_patch: feedback rank of births >= FBL_B (births [kz, FBL_B) keep TD_R_FB)
FBL_ON = FBL_B >= 0                             # fblate_patch: GUARD_EXTRA bool (off-suite predicts take the parent path)
FBL_RE = int(_os.environ.get('TD_R_FB', '4'))   # fblate_patch: the early (low) rank = the parent's TD_R_FB
_FBL_NEED = (int(_os.environ.get('TD_R_RES', '16' if TD_OFF else '8')) + 2 + 2 * max(FBL_RE, FBL_R if FBL_ON else 0)) * 15   # widest staging need
DN_WMAX = 288 if _FBL_NEED <= 270 else -(-(_FBL_NEED + 32) // 64) * 64   # fblate_patch: parent 288 = 2-D staging width (>= 18 (L-1) = 270 thin columns + 16-grid padding)


def _fbl_groups(kz, k, rfb):
    """fblate_patch: rank groups ((g0, g1, rank), ..) of the feedback sources [kz:k), or None (one full-rank group = the parent path)"""
    if not FBL_ON or FBL_RE >= rfb or FBL_RE < 1:
        return None
    b_ = min(max(FBL_B, kz), k)
    if b_ <= kz:
        return None
    return ((kz, b_, FBL_RE),) if b_ >= k else ((kz, b_, FBL_RE), (b_, k, rfb))


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


_POST_S26_KNOBS = ('J2_W431', 'J2_LOT', 'J2_RRW', 'J2_D21F', 'J2_KP', 'AU_REUSE', 'AU_T1PK_EXT', 'J3_TRI', 'AU_SYM4', 'AS_TAG_OFF', 'CDP_ON', 'EF_K31T21', 'FIN_EDGE_ON', 'FIN_KC_ON', 'FIN_GRAN_ON', 'THIN_ENG', 'OPD', 'OPD_SWAP', 'OPD_ZT', 'OPD_NLK', 'OPD_PK', 'OPD_CSE', 'TDP_ON', 'MID_EDGE_ON', 'MARG2_ON', 'MARG1_ON', 'LOTD_ON', 'RRES_ON', 'RGUARD_ON', 'MEHLER3', 'WG_ON', 'FBL_ON', 'TDPCAP2', 'PATHB_ON', 'PEEL', 'NLHZ', 'TM_CDPS', 'TM_FBW', 'TM_RREUSE', 'TM_BNC', 'TM_CC', 'TM_DQF', 'TM_DS0', 'TM_LS0', 'TM_K22MV', 'TM_NBKIN', 'TM_LOTHOT', 'DN_YCOL', 'DN_ZKIN', 'DN_FROT', 'DN_G0', 'DN_THIN2D', 'DN_Z2D', 'DN_EIGH', 'DN_KP', 'DN_FSUB', 'ZF7_AB_MERGE', 'HZ_AB', 'ZSRC_CACHE', 'ROLE_MERGE', 'LX', 'LX_PAD', 'LX_MULTI', 'HZ_L6', 'NLK_PK', 'NLK_K22', 'HZW', 'QRB', 'QRV', 'AU_T1PK', 'AU_LIFTW', 'AU_THINW', 'AU_APKZ', 'AU_ROWP', 'AU_YXT', 'HZGATE', 'LXPGATE', 'HZG_LOG', 'HZG_EXT', 'HZG_L6')   # guard_patch: switched off for off-suite predicts

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
        self._wg_r = -1   # wguard: rung reached in this predict (-1 = none)
        self._wg_log = []
        self._wg_arm = False
        if WG_ON and int(getattr(mlp, 'width', 0)) == 1024 and len(mlp.weights) == 16:
            _wg_c = _wg_cwall()
            self._wg_arm = (_wg_c is not None) or WG_ANY
            self._wg_off = max(0.0, _wg_c - (_rb_time.perf_counter() - self._rb_t0)) if _wg_c is not None else 0.0
            try:
                self._wg_fl = (int(WG_FORCE.split(':')[0]), int(WG_FORCE.split(':')[1])) if WG_FORCE else None
            except Exception:
                self._wg_fl = None
        self._rg_arm = RGUARD_ON and (_rg_resid() is not None)   # rguard: live residual readable in this predict
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
        _tmcc = self.__dict__.get('_tm_cc') if (TM_CC and n == 1024 and L == 16) else None
        if _tmcc is not None:
            # termite TM_CC: read-only constant tables of the first suite predict (no flopscope call; identical values)
            C1, C2, SELC, FP, C12, PW6, EX11, ones_full, zeros_full, ones2_full, beta_rows = _tmcc
            ones_n = ones_full
            zeros_n = zeros_full
            ones_col = _pview(ones_n, (slice(None, None, None), None))
            ones_row = ones_col.T
            ones2 = ones2_full
            ones_row_f = ones_row
        else:
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
            if TM_CC and n == 1024 and L == 16:
                self._tm_cc = (C1, C2, SELC, FP, C12, PW6, EX11, ones_full, zeros_full, ones2_full, beta_rows)
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
        if FBL_ON and n == 1024 and L == 16 and 0 < rfb < FBL_R:
            rfb = min(FBL_R, n)   # fblate_patch: stack width = the late rank; births [kz, FBL_B) fill the first TD_R_FB columns
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
            if self._rg_arm and self._hz_act and li >= RGUARD_L and self._nfull == 1024:
                _rg_r = _rg_resid()
                if _rg_r is not None and _rg_r > RGUARD_R:
                    # rguard: slow predict on the residual clock -> no more HZ work this predict (rows of the chain untouched)
                    self._hz_act = False
                    self._rb_hz_skip = ('R', li)
            if self._wg_arm and li >= min(WG_L0, WG_L1) and _wg_rung(self, li) >= 0 and self._hz_act:
                # wguard rung >= 0: projected wall too long -> no more HZ work this predict (rows of the chain untouched)
                self._hz_act = False
                self._rb_hz_skip = ('W', li)
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
            skip_src = trim and (NO_SRC_LAST or self._wg_r >= 2)   # wguard rung 2: final layer without source transport
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
                    var = fnp.maximum(fnp.sum(w32g * (C @ w32g), axis=0), 1e-10)   # wguard: live rows / order
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
                fbl_t_ = _fbl_groups(kz, k, rfb) if (FBL_ON and n == 1024 and Zf_st is not None and k > kz) else None
                if fbl_t_ is not None and not ((DN_Z2D and li >= DN_Z2D_LMIN) or (DN_ZKIN and kin < ni)):
                    # fbmix: per rank group; the low-rank group moves its used F1 / F2 columns and re-zeroes the rest
                    zfo_ = _win(pool.get(('zf', z_side), (L - 1, n, 2 * rfb)), nr, 2 * rfb)
                    for g0_, g1_, rg_ in fbl_t_:
                        if rg_ < rfb:
                            for c0_ in (0, rfb):
                                fnp.matmul(WDb, _pview(Zf_st, (slice(g0_, g1_, None), slice(None, None, None), slice(c0_, c0_ + rg_, None))), out=_pview(zfo_, (slice(g0_, g1_, None), slice(None, None, None), slice(c0_, c0_ + rg_, None))))
                                fnp.copyto(_pview(zfo_, (slice(g0_, g1_, None), slice(None, None, None), slice(c0_ + rg_, c0_ + rfb, None))), 0.0)
                        else:
                            fnp.matmul(WDb, _pview(Zf_st, slice(g0_, g1_, None)), out=_pview(zfo_, slice(g0_, g1_, None)))
                    Zf_st = _pview(zfo_, slice(None, k, None))
                elif Zf_st is not None and k > kz and not ((DN_Z2D and li >= DN_Z2D_LMIN) or (DN_ZKIN and kin < ni)):
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
                if Ff is not None and k >= kz and FBL_ON and n == 1024 and k < FBL_B and 1 <= FBL_RE < rfb:
                    # fbmix: low-rank newborn: transport the used F1 / F2 columns, zero the padding
                    zfb = _win(pool.get(('zf', z_side), (L - 1, n, 2 * rfb)), nr, 2 * rfb)
                    for c0_ in (0, rfb):
                        if TM_NBKIN and 0 < kin < ni and self._nfull == 1024:
                            # termite TM_NBKIN: Ff rows [kin:] are zero (W_all rows [n3:] zeroed at birth; kin = n3 of the birth layer)
                            fnp.matmul(_pview(W, (slice(None, None, None), slice(None, kin, None))), _pview(Ff, (slice(None, kin, None), slice(c0_, c0_ + FBL_RE, None))), out=_pview(zfb, (k, slice(None, None, None), slice(c0_, c0_ + FBL_RE, None))))
                        else:
                            fnp.matmul(W, _pview(Ff, (slice(None, None, None), slice(c0_, c0_ + FBL_RE, None))), out=_pview(zfb, (k, slice(None, None, None), slice(c0_, c0_ + FBL_RE, None))))
                        fnp.copyto(_pview(zfb, (k, slice(None, None, None), slice(c0_ + FBL_RE, c0_ + rfb, None))), 0.0)
                    Zf_st = _pview(zfb, slice(None, k + 1, None))
                elif Ff is not None and k >= kz:
                    zfb = _win(pool.get(('zf', z_side), (L - 1, n, 2 * rfb)), nr, 2 * rfb)
                    if TM_NBKIN and 0 < kin < ni and self._nfull == 1024:
                        # termite TM_NBKIN (see above)
                        fnp.matmul(_pview(W, (slice(None, None, None), slice(None, kin, None))), _pview(Ff, slice(None, kin, None)), out=zfb[k])
                    else:
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
            self._tm_p1 = self._tm_p2 = 0
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
                    self._tm_p1 = p_t1
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
                    self._tm_p2 = p_t2
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
                D3, D21 = self._dslices(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list, c1_list, c2_list, y_list, nr, bufs, r, rfb, Zf_st, R1T_st, R2T_st, need_d21=(not trim) and self._wg_r < 1, kz=kz, ka=ka, Qc=Qc, kb=kb, U2=U, apb=legs['AP0'], apb4=_win(legs4['AP0'], nr, n), sb1=_pview(fap4, slice(2 * fa_off, 2 * (fa_off + ka - kb), None)) if ka > kb else None, sb2=_pview(fap24, slice(None, 2 * kb, None)) if kb > 0 else None, s_sb=s_sb, gate_Q=(sk_src if merge_cp else lot_Q) if d21_gate_on else None, gate_sk=sk_src, gate_alpha=(au_alpha_ if au_alpha_ is not None else mu / fnp.sqrt(var)) if d21_gate_on else None, gate_li=li)
                if D21 is None and (not trim):
                    D21 = fnp.zeros((nr, nr), dtype=f32)   # wguard rung 1: no (2,1) slices this layer
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
                            if NLHZ and 'nl' in self._hz and self._nfull == 1024:
                                _nlhz_x(self._hz['nl'], li, nr, _C, _M, _D, _w1, _w2, _w3, D3, var)   # nlhz_patch v5
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
                if MEHLER3 and self._nfull == 1024 and L == 16:
                    # mehler3_patch (TERMPROG arm Bm): Gaussian Mehler m = 3 pair terms C_off^3 / 3! w(3,p)_i w(3,q)_j added to the (1,1) /
                    # (2,1) / (2,2) programs = PK2 groups 0 / 1 / 2, (p, q) = (1, 1) / (2, 1) / (2, 2) (row side carries p), before the
                    # symmetrisation / zero diagonals below (C_off has a zero diagonal).  Elementwise only: C_off^2 = the program's
                    # ('c_off', 'c_off') pair product, scratch = the consumed ABstack slots 0..3 (E_ >= 6), no matmul / pool buffer / plan
                    m3c_ = _pview(abbuf, 0) if RRES_ON else abbuf[0]
                    m3t_ = _pview(abbuf, 1) if RRES_ON else abbuf[1]
                    m3x_ = _pview(abbuf, slice(2, 4, None))
                    m3p_ = E_ + nk_['pairs'].index(('c_off', 'c_off'))
                    fnp.multiply(_pview(abbuf, m3p_) if RRES_ON else abbuf[m3p_], b2['c_off'], out=m3c_)   # C_off^3
                    m3w_ = fnp.multiply(_pview(WT, slice(self._i31, self._i31 + 2, None)), MEHLER3_S / 6.0)   # (2, nr) s/6 [w(3,1); w(3,2)]
                    fnp.multiply(m3c_, _pview(WT, slice(self._i31, self._i31 + 1, None)), out=m3t_)   # C^3 o (1 w(3,1)^T)
                    fnp.multiply(_pview(m3w_, (slice(None, None, None), slice(None, None, None), None)), m3t_, out=m3x_)
                    m3pk_ = _pview(PK2, slice(0, 2, None))
                    fnp.add(m3pk_, m3x_, out=m3pk_)   # (1,1) += s/6 w31 (x) w31 o C^3 ; (2,1) += s/6 w32 (x) w31 o C^3
                    fnp.multiply(m3c_, _pview(WT, slice(self._i31 + 1, self._i31 + 2, None)), out=m3c_)   # C^3 o (1 w(3,2)^T)
                    fnp.multiply(m3c_, _pview(m3w_, (1, slice(None, None, None), None)), out=m3c_)
                    fnp.add(PK2[2], m3c_, out=PK2[2])   # (2,2) += s/6 w32 (x) w32 o C^3
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
                    self._tdp_kn = min(n3, (_tdp_win2 if TDPCAP2 else _sdt_win)(nr, int(float(fnp.sum(fnp.less(alpha, -TDP_C))))))   # tdpcap_patch
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
                fbl_rb_ = FBL_RE if (FBL_ON and self._nfull == 1024 and k_fb < FBL_B and 1 <= FBL_RE < rfb) else rfb   # fbmix: this birth's rank
                if fbl_rb_ < rfb:
                    Omf = _pview(Omf, (slice(None, None, None), slice(None, fbl_rb_, None)))
                fnp.copyto(Omf, _pview(sk_src, (slice(None, None, None), slice(None, fbl_rb_, None))))
                _tm_nc = int(self.__dict__.get('_tm_ncz', -1)) if TM_BNC else -1
                self._tm_ncz = -1
                if 0 < _tm_nc < int(D21.shape[1]):
                    # termite TM_BNC: D21 columns [nc:] are zero (SDT, zeroed in _dslices this layer): contract / output on [:nc]
                    Dn_ = D21[:, :_tm_nc]
                    Yf = Dn_ @ Omf[:_tm_nc]
                    Qf, _ = fnp.linalg.qr(Yf)
                    Yf = Dn_ @ (Dn_.T @ Qf)
                    Qf, _ = fnp.linalg.qr(Yf)
                    Bf = fnp.concatenate([Qf.T @ Dn_, fnp.zeros((int(Qf.shape[1]), int(D21.shape[1]) - _tm_nc), dtype=f32)], axis=1)
                else:
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
                    R1T_b = fnp.concatenate([R1T_b[:n3], fnp.zeros((nd_cur, fbl_rb_), dtype=f32)], axis=0)
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
                c1pb_ = None
                if PATHB_ON and self._nfull == 1024:
                    # path_patch PATHB: Sym(3A x P x (c/12) w2 (dG w3)^T) = the feed's A-part with c1_b += dG w3 (K22 path birth, term B)
                    c1pb_ = _pview(W_all, (slice(None, None, None), self._i31)) * (dG * PATHB_KB)
                    if not PATHB_NC:
                        c1_b = c1_b + c1pb_
                        c1pb_ = None
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
                if TM_K22MV and self._nfull == 1024:
                    # termite TM_K22MV: 4 pk1 o (pk11 pk1) - 2 pk21 pk1 = -2 K21 pk1 (K21 above, zero diagonal): one matvec instead of two
                    k22r_ = fnp.matmul(K21, pk1v) * -2.0 - pk1v * fnp.sum(pk21, axis=0) * 2.0 - fnp.einsum('ij,ij->i', pk11, pk11) * 2.0 + rs22_
                else:
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
            if rfb > 0 and k_b >= kz and FBL_ON and self._nfull == 1024 and int(F1_b.shape[1]) < rfb:
                # fbmix: low-rank birth -> [F1 | 0 | F2 | 0] and R1T / R2T padded with zero columns (stack layout unchanged)
                fbl_z_ = fnp.zeros((nr, rfb - int(F1_b.shape[1])), dtype=f32)
                Ff_b = fnp.concatenate([F1_b, fbl_z_, F2_b, fbl_z_], axis=1)
                R1T_b = fnp.concatenate([R1T_b, fbl_z_], axis=1)
                R2T_b = fnp.concatenate([R2T_b, fbl_z_], axis=1)
            else:
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
            if PATHB_ON and PATHB_NC and self._nfull == 1024 and regen and mode == 1 and (not NO_FEED) and c1pb_ is not None:
                # path_patch PATHB_NC: carried hub weights only (the birth bookkeeping above did not see the term)
                c1_h = c1_h + (c1pb_ if (hm is None or not (HUB_PART & 4)) else c1pb_ * hm)
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
        if self._rg_arm and self._hz_act and self._nfull == 1024:
            _rg_r = _rg_resid()
            if _rg_r is not None and _rg_r > RGUARD_F:
                # rguard: slow predict on the residual clock -> the no-HZ prediction (skip _hz_finish)
                self._hz_act = False
                self._rb_hz_skip = ('R', 'F')
        if self._wg_arm and self._hz_act and _wg_rung(self, 16) >= 0:
            # wguard rung 0 before the finish: the no-HZ prediction (skip _hz_finish)
            self._hz_act = False
            self._rb_hz_skip = ('W', 'F')
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
        if NLHZ and n == 1024 and L == 16:
            H['nl'] = _nlhz_build(f32)   # nlhz_patch
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
        if NLHZ and rid and 'nl' in H:
            _nlhz_apply(H, hz_W)   # nlhz_patch
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
            _tmo_ = []
            for s0_, s1_, w_ in ((0, kb_, n), (kb_, ka_, hw_1), (ka_, k, hw_y)):
                if s1_ <= s0_:
                    continue
                _tp_ = (int(self.__dict__.get('_tm_p2' if s0_ == 0 else '_tm_p1', 0)) if (TM_LOTHOT and s1_ <= ka_ and self._nfull == 1024) else 0)
                if 0 < _tp_ < int(T2.shape[1]):
                    # termite TM_LOTHOT: this tier's legs are zero on rows [:p] (hot-row rebuild of this layer): rows [p:] only
                    _tmo_.append((_tp_, fnp.einsum('kij,kjq->iq', _pview(T2, (slice(s0_, s1_, None), slice(_tp_, None, None), slice(None, w_, None))), _pview(gu, (slice(s0_, s1_, None), slice(None, w_, None))))))
                    continue
                part_ = fnp.einsum('kij,kjq->iq', _pview(T2, (slice(s0_, s1_, None), slice(None, None, None), slice(None, w_, None))), _pview(gu, (slice(s0_, s1_, None), slice(None, w_, None))))
                KU = part_ if KU is None else KU + part_
            for _tp_, _pt_ in _tmo_:
                if KU is None:
                    KU = fnp.concatenate([fnp.zeros((_tp_, int(_pt_.shape[1])), dtype=fnp.float32), _pt_], axis=0)
                else:
                    _ku_ = _pview(KU, slice(_tp_, None, None))
                    fnp.add(_ku_, _pt_, out=_ku_)
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
        if PL_STAR and TM_LS0 and k >= 2 and self._nfull == 1024:
            # termite TM_LS0: slab[0, 2] = w(3,1)/3 of the layer-0 source is exactly 0 (alpha = 0 at layer 0): slots [1:k] only
            T3s_ = _pview(T3, slice(1, None, None))
            fnp.multiply(T3s_, T3s_, out=T3s_)
            fnp.multiply(T3s_, _pview(T1, slice(1, None, None)), out=T3s_)
            response_star = fnp.einsum('kij,kj->i', T3s_, _pview(slab, (slice(1, k, None), 2)))
            cand = cand + response_star
        elif PL_STAR:
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
        for fb_ in ((fb,) if isinstance(fb, tuple) else (fb or ())):   # fbmix: one parent tuple, or a list of rank groups
            GYR, GXR = (fb_[0], fb_[1])
            ks, rf = (GYR.shape[0], GYR.shape[2])
            Lf = fnp.reshape(_pview(L2, (slice(None, None, None), slice(w, w + 2 * rf * ks, None))), (nr, ks, 2 * rf))
            fnp.copyto(_pview(Lf, (Ellipsis, slice(None, rf, None))), fnp.swapaxes(GXR, 0, 1))
            fnp.copyto(_pview(Lf, (Ellipsis, slice(rf, None, None))), fnp.swapaxes(GYR, 0, 1))
            if len(fb_) == 3:
                fnp.copyto(fnp.reshape(_pview(R2, (slice(None, None, None), slice(w, w + 2 * rf * ks, None))), (nr, ks, 2 * rf)), fnp.swapaxes(fb_[2], 0, 1))
            else:
                Rf_ = fnp.reshape(_pview(R2, (slice(None, None, None), slice(w, w + 2 * rf * ks, None))), (nr, ks, 2 * rf))
                fnp.copyto(_pview(Rf_, (Ellipsis, slice(None, rf, None))), fnp.swapaxes(fb_[2], 0, 1))
                fnp.copyto(_pview(Rf_, (Ellipsis, slice(rf, None, None))), fnp.swapaxes(fb_[3], 0, 1))
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
        for fb_ in ((fb,) if isinstance(fb, tuple) else (fb or ())):   # fbmix: one parent tuple, or a list of rank groups
            GYR, GXR = (fb_[0], fb_[1])
            ks, rf = (GYR.shape[0], GYR.shape[2])
            Lf = fnp.reshape(_pview(L2, (slice(None, None, None), slice(w, w + 2 * rf * ks, None))), (nr, ks, 2 * rf))
            fnp.copyto(_pview(Lf, (Ellipsis, slice(None, rf, None))), fnp.swapaxes(GXR, 0, 1))
            fnp.copyto(_pview(Lf, (Ellipsis, slice(rf, None, None))), fnp.swapaxes(GYR, 0, 1))
            if len(fb_) == 3:
                fnp.copyto(fnp.reshape(_pview(R2, (slice(None, None, None), slice(w, w + 2 * rf * ks, None))), (nr, ks, 2 * rf)), fnp.swapaxes(fb_[2], 0, 1))
            else:
                Rf_ = fnp.reshape(_pview(R2, (slice(None, None, None), slice(w, w + 2 * rf * ks, None))), (nr, ks, 2 * rf))
                fnp.copyto(_pview(Rf_, (Ellipsis, slice(None, rf, None))), fnp.swapaxes(fb_[2], 0, 1))
                fnp.copyto(_pview(Rf_, (Ellipsis, slice(rf, None, None))), fnp.swapaxes(fb_[3], 0, 1))
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
        fbl_t_ = _fbl_groups(kz, k, rfb) if (fb and FBL_ON and n == 1024) else None   # fbmix: per rank group, used columns only
        if DN_Z2D and li >= DN_Z2D_LMIN:
            ks = k - kz if fb else 0
            w = q * k + f * ks
            if fbl_t_ is not None:
                w = q * k + sum((2 * rg_ * (g1_ - g0_) for g0_, g1_, rg_ in fbl_t_))
            Lz, wp = _dn_pad(nr, kc, w, DN_Z2D_LEV, C7_MIN, False)
            Si = pool.get('dn_zi', (1, 1, n, DN_WMAX))
            So = pool.get('dn_zo', (1, 1, n, DN_WMAX))
            I2 = _pview(Si, (0, 0, slice(None, kc, None)))
            O2 = _pview(So, (0, 0, slice(None, nr, None)))
            fnp.copyto(fnp.reshape(_pview(I2, (slice(None, None, None), slice(None, q * k, None))), (kc, k, q)), fnp.swapaxes(Zin, 0, 1))
            if fbl_t_ is not None:
                c_ = q * k
                for g0_, g1_, rg_ in fbl_t_:
                    gs_ = g1_ - g0_
                    zg_ = _pview(Zfin, slice(g0_ - kz, g1_ - kz, None))
                    if rg_ < rfb:
                        for c0_ in (0, rfb):
                            fnp.copyto(fnp.reshape(_pview(I2, (slice(None, None, None), slice(c_, c_ + gs_ * rg_, None))), (kc, gs_, rg_)), fnp.swapaxes(_pview(zg_, (slice(None, None, None), slice(None, None, None), slice(c0_, c0_ + rg_, None))), 0, 1))
                            c_ += gs_ * rg_
                    else:
                        fnp.copyto(fnp.reshape(_pview(I2, (slice(None, None, None), slice(c_, c_ + gs_ * f, None))), (kc, gs_, f)), fnp.swapaxes(zg_, 0, 1))
                        c_ += gs_ * f
            elif fb:
                fnp.copyto(fnp.reshape(_pview(I2, (slice(None, None, None), slice(q * k, w, None))), (kc, ks, f)), fnp.swapaxes(Zfin, 0, 1))
            smm.mn = C7_MIN
            smm.mm(_pview(WDc, (None, None)), _pview(Si, (slice(None, None, None), slice(None, None, None), slice(None, kc, None), slice(None, wp, None))), _pview(So, (slice(None, None, None), slice(None, None, None), slice(None, nr, None), slice(None, wp, None))), Lz)
            smm.mn = STRASSEN_MIN
            fnp.copyto(zo, fnp.swapaxes(fnp.reshape(_pview(O2, (slice(None, None, None), slice(None, q * k, None))), (nr, k, q)), 0, 1))
            if fbl_t_ is not None:
                c_ = q * k
                for g0_, g1_, rg_ in fbl_t_:
                    gs_ = g1_ - g0_
                    if rg_ < rfb:
                        for c0_ in (0, rfb):
                            fnp.copyto(_pview(zfa, (slice(g0_, g1_, None), slice(None, None, None), slice(c0_, c0_ + rg_, None))), fnp.swapaxes(fnp.reshape(_pview(O2, (slice(None, None, None), slice(c_, c_ + gs_ * rg_, None))), (nr, gs_, rg_)), 0, 1))
                            fnp.copyto(_pview(zfa, (slice(g0_, g1_, None), slice(None, None, None), slice(c0_ + rg_, c0_ + rfb, None))), 0.0)
                            c_ += gs_ * rg_
                    else:
                        fnp.copyto(_pview(zfa, slice(g0_, g1_, None)), fnp.swapaxes(fnp.reshape(_pview(O2, (slice(None, None, None), slice(c_, c_ + gs_ * f, None))), (nr, gs_, f)), 0, 1))
                        c_ += gs_ * f
            elif fb:
                fnp.copyto(_pview(zfa, slice(kz, k, None)), fnp.swapaxes(fnp.reshape(_pview(O2, (slice(None, None, None), slice(q * k, w, None))), (nr, ks, f)), 0, 1))
        else:
            fnp.matmul(_pview(WDc, None), Zin, out=zo)
            if fbl_t_ is not None:
                for g0_, g1_, rg_ in fbl_t_:
                    zg_ = _pview(Zfin, slice(g0_ - kz, g1_ - kz, None))
                    if rg_ < rfb:
                        for c0_ in (0, rfb):
                            fnp.matmul(_pview(WDc, None), _pview(zg_, (slice(None, None, None), slice(None, None, None), slice(c0_, c0_ + rg_, None))), out=_pview(zfa, (slice(g0_, g1_, None), slice(None, None, None), slice(c0_, c0_ + rg_, None))))
                            fnp.copyto(_pview(zfa, (slice(g0_, g1_, None), slice(None, None, None), slice(c0_ + rg_, c0_ + rfb, None))), 0.0)
                    else:
                        fnp.matmul(_pview(WDc, None), zg_, out=_pview(zfa, slice(g0_, g1_, None)))
            elif fb:
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
        tm_inv_ = None
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
            if TM_CDPS and m < nr and self._nfull == 1024:
                # termite TM_CDPS: U^T x[order] = Uf^T x with Uf = U scattered onto the nr live rows (zeros on the projected rows): no
                # (S, m, cols) row gather at 4/elem; the slot sum folds into one einsum (same bill); exact is still all zeros here
                tm_inv_ = fnp.argsort(permutation)
                UTf_ = fnp.take(fnp.concatenate([UT_, fnp.zeros((int(UT_.shape[0]), nr - m), dtype=fnp.float32)], axis=1), tm_inv_, axis=1)
                for x_, ys_ in cdp_parts:
                    x3_ = _pview(x_, (slice(None, None, None), 0))
                    yb_ = _pview(ys_, (slice(None, None, None), 0, slice(ncwp, ncw, None)))
                    g_ = fnp.einsum('src,sbc->rb', fnp.matmul(UTf_, x3_), yb_)
                    G_ = g_ if G_ is None else G_ + g_
                exact = fnp.concatenate([ey_, fnp.matmul(U_, G_)], axis=1)
            else:
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
            if TM_DQF and self._nfull == 1024:
                # termite TM_DQF: one einsum = matmul + slot sum (same bill: k(2n-1) + (k-1) = 2kn-1), one call less
                dq = dq + fnp.einsum('kij,kjr->ir', xt, yr)
            else:
                qp = fnp.matmul(xt, yr)
                dq = dq + fnp.sum(qp, axis=0)
        tail = fnp.matmul(dq, Q.T)
        arranged = fnp.concatenate((exact, tail), axis=0)
        dd = bufs['d21']
        fnp.take(arranged, tm_inv_ if tm_inv_ is not None else fnp.argsort(permutation), axis=0, out=dd if ncw == nr else dd[:, :ncw])
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
        self._tm_ncz = -1
        W2B = _pview(fnp.stack(w2b_list, axis=0), (slice(None, None, None), None, slice(None, None, None)))
        bufs['w2b'] = W2B
        Sb = _pview(fnp.stack(s_list, axis=0), (slice(None, None, None), None, slice(None, None, None)))
        Eb = _pview(fnp.stack(e_list, axis=0), (slice(None, None, None), None, slice(None, None, None)))
        T = _pview(bufs['t'], slice(None, k, None))
        dsf = DSF and need_d21
        _tm_rr = TM_RREUSE and dsf and DN_YCOL and gate_Q is not None and gate_alpha is not None and gate_li >= 1 and self._nfull == 1024
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
            R = None if _tm_rr else fnp.einsum('kij,kj->ki', LA, C1)
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
            if not _tm_rr:
                R = R + fnp.einsum('kij,kj->ki', T, C2)
            if DN_YCOL and gate_Q is not None and gate_alpha is not None and (gate_li >= 1):
                # DN_YCOL: PPL on the first rres + 1 columns (its last column would be (P*P) 0); the free column carries R, so the gated
                # thin einsum below yields (1/3) R^T Yk as well (Yk = Z[..., rres + 1])
                PPL = _pview(bufs['ppl'], slice(None, k, None))
                fnp.matmul(T, _pview(L_st, (Ellipsis, slice(None, rres + 1, None))), out=_pview(PPL, (Ellipsis, slice(None, rres + 1, None))))
                if not _tm_rr:
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
        fbl_g_ = _fbl_groups(kz, k, rfb) if (dsf and rfb > 0 and k > kz and FBL_ON and self._nfull == 1024 and _TAG['mmfb'] and AU_APKZ) else None
        if fbl_g_ is not None:
            # fbmix: the DSF feedback block once per rank group (the low-rank group reads the first rg columns of F1 / F2 / R1T / R2T;
            # its other columns are zero); D3 / LP / LA / D21_fb add up, DN_THIN2D stages the list of groups
            dn_fb = [] if dn_t2 else None
            D21_fb = None
            for g0_, g1_, rg_ in fbl_g_:
                o0_, o1_ = (g0_ - kz, g1_ - kz)
                F1 = _pview(Zf_st, (slice(g0_, g1_, None), slice(None, None, None), slice(None, rg_, None)))
                F2 = _pview(Zf_st, (slice(g0_, g1_, None), slice(None, None, None), slice(rfb, rfb + rg_, None)))
                _tw_ = n
                if TM_FBW and dn_t2 and self._nfull == 1024 and len(self._hub_birth_widths) >= g1_:
                    _tw_ = min(n, int(max(self._hub_birth_widths[g0_:g1_])))
                if _tw_ < n:
                    # termite TM_FBW: legs A / P are zero beyond each slot's hub width and R1T beyond n3 -> every term of this group's
                    # block vanishes on the hub columns [w_g:]; the block runs on the window (outputs LP / LA / D3 / GYR / GXR unchanged)
                    _cw_ = slice(None, _tw_, None)
                    R1T3 = _pview(R1T_st, (slice(g0_, g1_, None), _cw_, slice(None, rg_, None))) * (1.0 / 3.0)
                    R2T = _pview(R2T_st, (slice(g0_, g1_, None), _cw_, slice(None, rg_, None)))
                    A_s = _pview(A_st, (slice(g0_, g1_, None), slice(None, None, None), _cw_))
                    P_s = _pview(P_st, (slice(g0_, g1_, None), slice(None, None, None), _cw_))
                    W2B_s = _pview(W2B, (slice(g0_, g1_, None), slice(None, None, None), _cw_))
                    Xt = _public_out(fnp.matmul, F1, fnp.swapaxes(R1T3, -1, -2), out=_pview(bufs['xt'], (slice(o0_, o1_, None), slice(None, None, None), _cw_)))
                    Yt = _public_out(fnp.matmul, F2, fnp.swapaxes(R2T, -1, -2), out=_pview(bufs['yt'], (slice(o0_, o1_, None), slice(None, None, None), _cw_)))
                    U = _pview(bufs['u'], (slice(o0_, o1_, None), slice(None, None, None), _cw_))
                    Ts = _pview(bufs['t'], (slice(o0_, o1_, None), slice(None, None, None), _cw_))
                else:
                    R1T3 = _pview(R1T_st, (slice(g0_, g1_, None), slice(None, None, None), slice(None, rg_, None))) * (1.0 / 3.0)
                    R2T = _pview(R2T_st, (slice(g0_, g1_, None), slice(None, None, None), slice(None, rg_, None)))
                    A_s = _pview(A_st, slice(g0_, g1_, None))
                    P_s = _pview(P_st, slice(g0_, g1_, None))
                    W2B_s = _pview(W2B, slice(g0_, g1_, None))
                    Xt = _public_out(fnp.matmul, F1, fnp.swapaxes(R1T3, -1, -2), out=_pview(bufs['xt'], slice(o0_, o1_, None)))
                    Yt = _public_out(fnp.matmul, F2, fnp.swapaxes(R2T, -1, -2), out=_pview(bufs['yt'], slice(o0_, o1_, None)))
                    U = _pview(bufs['u'], slice(o0_, o1_, None))
                    Ts = _pview(bufs['t'], slice(o0_, o1_, None))
                fnp.multiply(A_s, Yt, out=Ts)
                fnp.multiply(Xt, A_s, out=U)
                fnp.multiply(U, W2B_s, out=U)
                fnp.add(Ts, U, out=Ts)
                fnp.multiply(Xt, Yt, out=U)
                fnp.add(Ts, U, out=Ts)
                D3 = D3 + fnp.einsum('kij,kij->i', P_s, Ts) * 3.0
                if _tw_ < n:
                    LP_s = _pview(LP, (slice(g0_, g1_, None), slice(None, None, None), slice(None, _tw_, None)))
                    LA_s = _pview(LA, (slice(g0_, g1_, None), slice(None, None, None), slice(None, _tw_, None)))
                else:
                    LP_s = _pview(LP, slice(g0_, g1_, None))
                    LA_s = _pview(LA, slice(g0_, g1_, None))
                fnp.add(LP_s, Ts, out=LP_s)
                fnp.multiply(Xt, P_s, out=Ts)
                fnp.multiply(Ts, W2B_s, out=U)
                fnp.add(LA_s, U, out=LA_s)
                fnp.multiply(P_s, Yt, out=U)
                fnp.add(LA_s, U, out=LA_s)
                if _tw_ < n:
                    AP_s = _public_out(fnp.multiply, A_s, P_s, out=_pview(bufs['mp'], (slice(g0_, g1_, None), slice(None, None, None), slice(None, _tw_, None))))
                else:
                    AP_s = _public_out(fnp.multiply, A_s, P_s, out=_pview(bufs['mp'], slice(g0_, g1_, None)))
                fnp.add(Ts, AP_s, out=Ts)
                Xs = _pview(Xt, (slice(None, None, None), slice(None, None, None), slice(None, nr, None)))
                GYR = _public_out(fnp.matmul, Ts, R2T, out=_pview(bufs['gyr'], (slice(o0_, o1_, None), slice(None, None, None), slice(None, rg_, None))))
                fnp.multiply(AP_s, W2B_s, out=Ts)
                fnp.add(Ts, U, out=Ts)
                if dn_t2:
                    GXR = _public_out(fnp.matmul, Ts, R1T3, out=_pview(bufs['gxr'], (slice(o0_, o1_, None), slice(None, None, None), slice(None, rg_, None))))
                    dn_fb.append((GYR, GXR, F1, F2))
                else:
                    GXR = _public_out(fnp.matmul, Ts, R1T3, out=_pview(bufs['gxr'], (slice(o0_, o1_, None), slice(None, None, None), slice(None, rg_, None))))
                    fnp.matmul(GYR, fnp.swapaxes(F2, -1, -2), out=Xs)
                    if D21_fb is None:
                        D21_fb = _public_out(fnp.sum, Xs, axis=0, out=bufs['d21fb'])
                    else:
                        fnp.sum(Xs, axis=0, out=bufs['t1'])
                        fnp.add(D21_fb, bufs['t1'], out=D21_fb)
                    fnp.matmul(GXR, fnp.swapaxes(F1, -1, -2), out=Xs)
                    fnp.sum(Xs, axis=0, out=bufs['t1'])
                    fnp.add(D21_fb, bufs['t1'], out=D21_fb)
        elif dsf and rfb > 0 and k > kz:
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
            fbl_n_ = _fbl_groups(kz, k, rfb) if (FBL_ON and self._nfull == 1024 and _TAG['mmfb']) else None
            if fbl_n_ is not None:
                # fblate: Xt / Yt per rank group (a low-rank group's products read its first rg columns; the other columns are zero)
                for g0_, g1_, rg_ in fbl_n_:
                    o0_, o1_ = (g0_ - kz, g1_ - kz)
                    fnp.matmul(_pview(F1, (slice(o0_, o1_, None), slice(None, None, None), slice(None, rg_, None))), fnp.swapaxes(_pview(R1T, (slice(o0_, o1_, None), slice(None, None, None), slice(None, rg_, None))), -1, -2), out=_pview(bufs['xt'], slice(o0_, o1_, None)))
                    fnp.matmul(_pview(F2, (slice(o0_, o1_, None), slice(None, None, None), slice(None, rg_, None))), fnp.swapaxes(_pview(R2T, (slice(o0_, o1_, None), slice(None, None, None), slice(None, rg_, None))), -1, -2), out=_pview(bufs['yt'], slice(o0_, o1_, None)))
                Xt = _pview(bufs['xt'], slice(None, ks, None))
                Yt = _pview(bufs['yt'], slice(None, ks, None))
            elif _TAG['mmfb']:
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
        if not _tm_rr:
            D3 = D3 + fnp.einsum('ki,ki->i', R, Yk)
        if not need_d21:
            return (D3, None)
        if not (dsf or dsx):
            PPL = _public_out(fnp.matmul, PP, L_st, out=_pview(bufs['ppl'], slice(None, k, None)))
        if TM_DS0 and k >= 2 and self._nfull == 1024:
            # termite TM_DS0: slot 0 = the layer-0 source (mode 0: c1 = c2 = y = 0), so its bracket and T o Yk are exactly zero
            _s1_ = slice(1, None, None)
            T_ = _pview(T, _s1_)
            MP_ = _pview(MP, _s1_)
            fnp.multiply(_pview(P_st, _s1_), _pview(C2, (_s1_, None, slice(None, None, None))), out=MP_)
            fnp.multiply(_pview(A_st, _s1_), _pview(C1, (_s1_, None, slice(None, None, None))), out=T_)
            fnp.add(T_, MP_, out=T_)
            if _tm_rr:
                R = fnp.concatenate([fnp.zeros((1, int(T.shape[1])), dtype=fnp.float32), fnp.einsum('kij,kij->ki', _pview(P_st, _s1_), T_)], axis=0)
                fnp.copyto(_pview(PPL, (Ellipsis, rres + 1)), R)
                D3 = D3 + fnp.einsum('ki,ki->i', R, Yk)
            fnp.multiply(T_, _pview(_pview(Yk, _s1_) * (2.0 / 3.0), (slice(None, None, None), slice(None, None, None), None)), out=T_)
            LP_ = _pview(LP, _s1_)
            fnp.add(LP_, T_, out=LP_)
        else:
            fnp.multiply(P_st, _pview(C2, (slice(None, None, None), None, slice(None, None, None))), out=MP)
            fnp.multiply(A_st, _pview(C1, (slice(None, None, None), None, slice(None, None, None))), out=T)
            fnp.add(T, MP, out=T)
            if _tm_rr:
                # termite TM_RREUSE: R = rowdot(P, A o C1 + P o C2) on the bracket just formed in T (was two einsums in the DSF block)
                R = fnp.einsum('kij,kij->ki', P_st, T)
                fnp.copyto(_pview(PPL, (Ellipsis, rres + 1)), R)
                D3 = D3 + fnp.einsum('ki,ki->i', R, Yk)
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
            self._tm_ncz = nc_ if (TM_BNC and nc_ < nr and self._nfull == 1024) else -1
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

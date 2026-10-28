# Reading guide to `estimator/estimator.py` (submission #337608)

This guide maps the write-up to the code. `estimator/estimator.py` is the file graded as #337608 ("G4"): 8,568 lines, sha256 `e71018ffa112bc2bc4274798c850fe34170462a88b320522229d8b03c001271f`, with every table and learned weight embedded. It grew as a chain of patches on 504aldo's V29. That history explains its shape: features are added behind environment knobs, and old code paths stay in place behind switches that are off by default. Most of the length is cost engineering. The ideas that the write-up claims sit in about a dozen places, and this guide points to each of them.

What the code does, in one paragraph. The estimator runs the 16 layers once and carries a truncated cumulant state through the network:
- the means and the covariance;
- third cumulants as factored per-birth sources;
- a fourth-cumulant diagonal.

From these it computes each layer's post-ReLU means with an Edgeworth-corrected mean program. After the forward sweep, a learned correction (HZ, plus a small non-linear head) is evaluated once and carried to the final layer. Cost control comes from three things:
- the demand-weighted truncations (exact work only where the final layer reads it);
- the age tiers for old sources;
- a fast-matmul engine.

## 0. Source convention (read first)

- **Line references.** A reference like [L312](../estimator/estimator.py#L312) or [L7373-L7513](../estimator/estimator.py#L7373-L7513) points into the released file. The file is never edited, so line numbers do not drift. `verify.py` checks 19 of the anchors used here on every run. When the repository is tagged, prefer tag permalinks (`/blob/<tag>/estimator/estimator.py#L…`).
- **Knobs.** Almost every mechanism has an environment switch read once at import time, e.g. `PORTLOT` at [L312](../estimator/estimator.py#L312). The value given here is the default in the source, which is what the grader ran. Changing a knob means setting the environment variable *before* the module is imported. The file reads 374 distinct environment names (section 4 summarises them).
- **Live and dead.** "Live" means executed at the default settings on the scored shape (1024 x 16). "Dead at defaults" means present but switched off (section 5).
- **Provenance tags.** Each mechanism below carries one tag:
  - *inherited*: 504aldo's V29 or ARC's reference, through V29;
  - *ported*: adapted from a credited MIT repository;
  - *re-derived*: implemented in this work from a published description;
  - *new*: introduced in this work.

  `docs/PROVENANCE.md` and `THIRD_PARTY_NOTICES.md` give the details.
- **Build names.** sNN are the project's numbered builds; s43 = #336780 is the base of the three graded controls (`controls/`). S'' = #337231 is the last build before the learned head, NLHZ = #337317 adds the head, G2 = #337322, G3 = #337411 (head input set "v5D"), and G4 = #337608 is this file. MERGE is the 2 Oct merge of this work's first components into V29. TERMITE names a family of exact call-count cuts. G4 differs from G3 only in line 538 (the NLHZ head weights). `docs/SUBMISSIONS.md` lists the graded runs.

## 1. Execution path

| Step | Lines | What happens |
|---|---|---|
| Header | [L1-L74](../estimator/estimator.py#L1-L74) | attribution, LLM-assistance statement (line 7), the 504aldo / 504aldo+LordProcciuto / ARC MIT notices |
| Module docstring | [L75-L236](../estimator/estimator.py#L75-L236) | the MERGE summary, then V29's V16b-V29 history kept as written; the `portlot` note at L97-L103 |
| Import-time constants and knobs | [L238-L1914](../estimator/estimator.py#L238-L1914), plus [L3543-L3719](../estimator/estimator.py#L3543-L3719) and [L3958-L4128](../estimator/estimator.py#L3958-L4128) | tables (section 2), knob parsing, helpers; `_POOL_SPEC` (setup plan list recorded at build time) at [L1909](../estimator/estimator.py#L1909) |
| Engine classes | [L1122-L1494](../estimator/estimator.py#L1122-L1494), [L1921-L3542](../estimator/estimator.py#L1921-L3542), [L3748-L3956](../estimator/estimator.py#L3748-L3956), [L4143-L4404](../estimator/estimator.py#L4143-L4404) | `_Strassen` (V29), `ZFMM` / `_ZW` (Z-order Strassen-Winograd core), `_FMMd` dispatcher, `_Pool`, `_HZE` (HZ pullback engine) |
| `Estimator.setup` | [L4445-L4531](../estimator/estimator.py#L4445-L4531) | warm-up, buffer and plan pre-allocation (`_prealloc` [L4533-L4609](../estimator/estimator.py#L4533-L4609)), HZ tables (`_hz_setup` [L6662-L6793](../estimator/estimator.py#L6662-L6793)); time-capped (section 6) |
| `Estimator.predict` | [L4611-L4630](../estimator/estimator.py#L4611-L4630) | off-suite fallback switch (section 7), then `_predict_core` |
| `_predict_core` preamble | [L4632-L4885](../estimator/estimator.py#L4632-L4885) | guard arming, per-predict state, engine levels |
| Per-layer loop | [L4886-L6615](../estimator/estimator.py#L4886-L6615) | one pass per layer `li` (table below) |
| Finish | [L6616-L6638](../estimator/estimator.py#L6616-L6638) | last guard checks, then `_hz_finish` (HZ + NLHZ correction added to the final row) |

Inside the per-layer loop, in execution order:

| Lines | Step |
|---|---|
| [L4887-L4900](../estimator/estimator.py#L4887-L4900) | top-of-layer guards: RGUARD, WG ladder, `HZ_TMAX_L` |
| [L4944-L5034](../estimator/estimator.py#L4944-L5034) | active set: drop nearly dead neurons, keep their Gaussian ReLU means (`dead_m_cur`, [L4989](../estimator/estimator.py#L4989)), live order for the soft-dead tail |
| [L5076-L5160](../estimator/estimator.py#L5076-L5160) | old-source tier 1: joins of the shared basis (frozen joins `FIJA2`, [L5091](../estimator/estimator.py#L5091)) |
| [L5160-L5326](../estimator/estimator.py#L5160-L5326) | tier-2 moves of the oldest sources (least-squares projection under `FIJA2`, [L5163](../estimator/estimator.py#L5163)) |
| [L5327-L5446](../estimator/estimator.py#L5327-L5446) | first-union ordering of the shared basis, incl. `QRV` ([L5361-L5402](../estimator/estimator.py#L5361-L5402)) |
| [L5447-L5565](../estimator/estimator.py#L5447-L5565) | old-tier legs on their hub windows |
| [L5566-L5646](../estimator/estimator.py#L5566-L5646) | young-source transport (one batched Strassen family over the young slots) |
| [L5647-L5740](../estimator/estimator.py#L5647-L5740) | newborn slots; on the trimmed last layer the variance comes from W·C_post and the LOT basis is formed from it ([L5688-L5692](../estimator/estimator.py#L5688-L5692)) |
| [L5741-L5883](../estimator/estimator.py#L5741-L5883) | pre-activation covariance C_pre and variance; LOT basis of C_pre on the other layers ([L5744-L5746](../estimator/estimator.py#L5744-L5746), [L5883](../estimator/estimator.py#L5883)) |
| [L5884-L5927](../estimator/estimator.py#L5884-L5927) | kappa4 regeneration: lambda law, transported diagonal `dG`, row metric (`H18`) giving `g4row` |
| [L5928-L5946](../estimator/estimator.py#L5928-L5946) | (3,1) pair slice `wk431` (two-term closure `K31TWO`) |
| [L5960-L5962](../estimator/estimator.py#L5960-L5962) | LOT shape of the kappa4 diagonal (`_lot_g4`) |
| [L5972-L6003](../estimator/estimator.py#L5972-L6003) | sigma, alpha, phi, Phi; unified Wick coefficients |
| [L6004-L6196](../estimator/estimator.py#L6004-L6196) | merged term programs (pair moments), NLHZ per-row inputs ([L6023-L6054](../estimator/estimator.py#L6023-L6054)), marg2 mask ([L6081-L6092](../estimator/estimator.py#L6081-L6092)), Mehler m=3 terms ([L6131-L6148](../estimator/estimator.py#L6131-L6148)) |
| [L6210-L6215](../estimator/estimator.py#L6210-L6215) | last layer: final means (plus `FIN_EDGE` for dropped rows), then `break` |
| [L6216-L6220](../estimator/estimator.py#L6216-L6220) | `MID_EDGE` for dropped rows of an intermediate layer |
| [L6221-L6421](../estimator/estimator.py#L6221-L6421) | soft-dead masks, HZ hooks, hub legs of the newborn source, D21 feedback, kappa4 to kappa3 feed (with `PATHB`, [L6395-L6400](../estimator/estimator.py#L6395-L6400)) |
| [L6422-L6509](../estimator/estimator.py#L6422-L6509) | power moments to cumulants (`PK2K_TABLE`): new mu, C, D3 |
| [L6510-L6615](../estimator/estimator.py#L6510-L6615) | birth of this layer's third-cumulant source (thin residual legs, rank-r range finder), per-layer lists for `_dslices` |

The third-cumulant contractions (the D3 diagonal and the D21 slice) are computed by `_dslices` ([L8103-L8564](../estimator/estimator.py#L8103-L8564)). The demand-driven D21 read is `_d21_demand` ([L7807-L8101](../estimator/estimator.py#L7807-L8101)), called at [L8487](../estimator/estimator.py#L8487).

## 2. Paper notation to code

| Paper | Code | Where |
|---|---|---|
| $z_\ell$, $h_\ell$ (pre-, post-activation of layer $\ell$) | layer index `li`, weights `w` / `w32`; post-activation means appended to `rows` | loop at [L4886](../estimator/estimator.py#L4886) |
| $\mu_i$, $\sigma_i^2$, $\alpha_i=\mu_i/\sigma_i$ | `mu`, `var`, `sigma`, `alpha` | [L5972-L5973](../estimator/estimator.py#L5972-L5973) |
| $\phi(\alpha)$, $\Phi(\alpha)$ | `phi`, `Phi` (also stored as `hz_Phi` for the transport) | [L5976-L5978](../estimator/estimator.py#L5976-L5978) |
| $D_{3,i}=\kappa_3(z_i,z_i,z_i)$, $D21_{ij}=\kappa_3(z_i,z_i,z_j)$ | `D3`, `D21` (from `_dslices`) | [L8103-L8564](../estimator/estimator.py#L8103-L8564) |
| $\kappa_{4,i}$ used by the mean program | `g4row` (V29 regeneration with row metric), replaced by `lot_g4m` (LOT shape) when the guard passes | [L5884-L5927](../estimator/estimator.py#L5884-L5927), [L5960-L5965](../estimator/estimator.py#L5960-L5965) |
| mean program $m_i=\sigma[\phi+\alpha\Phi]+c_3D_3+c_4\kappa_4+c_{33}D_3^2$ | unified Wick coefficients `w(k,p)` from `WICK_UNIFIED` and the term tables `TERM_SPECS` / `PK2K_TABLE` | tables [L245-L248](../estimator/estimator.py#L245-L248); programs `_term_prog` [L804-L836](../estimator/estimator.py#L804-L836), `_nlk_prog` [L843-L930](../estimator/estimator.py#L843-L930) |
| covariance, its off-diagonal part | `C`, `C_pre`, `C_off` | [L5741-L5883](../estimator/estimator.py#L5741-L5883) |
| V29 lambda law $\lambda_\ell=\mathrm{LAM}[\ell]((\overline{dG}/\overline{\mathrm{var}})/\mathrm{REF\_R}[\ell])^\beta$ | `lam_prev`, `LAM` (x `V17_LAM_SCALE` 0.95), `REF_R`, `BETA`; ratio clipped to [0.5, 2] | [L250](../estimator/estimator.py#L250), [L308-L310](../estimator/estimator.py#L308-L310), [L5888-L5906](../estimator/estimator.py#L5888-L5906) |
| LOT functional $\psi_i$ projected on the top-$K$ covariance modes | `KU`, `response_X`, `proj_only` in `_lot_g4`; basis `Q`, Gram `Gq` from `_lot_basis`; $K$ = `PORTLOT_K` = 12 | [L7373-L7513](../estimator/estimator.py#L7373-L7513), [L6640-L6660](../estimator/estimator.py#L6640-L6660) |
| same-hub term, star term $\kappa^\star_4$ | `response_same`, `response_star` | [L7458-L7484](../estimator/estimator.py#L7458-L7484) |
| $\mathrm{cand}_i$ and the rescaling $\overline{\kappa_4^{V29}}\,\mathrm{cand}_i/\overline{\mathrm{cand}}$ | `cand`, `m1 = mean(cand)`, `mg = mean(g4row)`, `g4m = cand * mg / m1` | [L7485-L7488](../estimator/estimator.py#L7485-L7488) |
| error transport $e_\ell=\Phi(\alpha_\ell)\circ(W_\ell^\top e_{\ell-1})+\varepsilon_\ell$ | `hz_Phi[li] * matmul(hz_W[li], e)` in the HZ finish | [L7341-L7371](../estimator/estimator.py#L7341-L7371) |
| HZ selected rows $K_\ell$ | `HZ_K` (per layer, default `0,0,0,0,0,96,0,32,96,96,128,192,160,128,192,160`), selection `_hz_select` | [L517](../estimator/estimator.py#L517), [L6919-L6947](../estimator/estimator.py#L6919-L6947) |
| HZ features $X$ and weights $\theta_\ell$ | eight feature rows `same, proj, star, D3, var, g4, mu, port` ([L529](../estimator/estimator.py#L529)) plus the chain and induced pullback sums (`acc_c`, `acc_i`); $\theta$ = `HZ_TH` (16 entries, layers 2-15 x 24 = 7 kappa4-term weights + 17 rider weights) | [L620](../estimator/estimator.py#L620), coefficient tables at [L6694-L6700](../estimator/estimator.py#L6694-L6700) |
| riders (17 weights on all live rows) | `rc` / `RC` coefficients, rider slabs `sl_*` | [L6700-L6720](../estimator/estimator.py#L6700-L6720), [L7300-L7337](../estimator/estimator.py#L7300-L7337) |
| NLHZ head (2x64 MLP shared by layers 2-15) | `_NLHZ_P`, `_nlhz_build`, `_nlhz_x`, `_nlhz_apply` | [L537-L618](../estimator/estimator.py#L537-L618) |
| marg2 rule (k=2 marginal correction only where $\alpha\le-2$) | `MARG2`, `MARG2_C` = 2.0 | [L693-L719](../estimator/estimator.py#L693-L719), [L6081-L6092](../estimator/estimator.py#L6081-L6092) |
| $D^2$ halving | `D2SQ_S` = 0.5 applied through `_dgs_coef` | [L725](../estimator/estimator.py#L725), [L752-L755](../estimator/estimator.py#L752-L755) |

## 3. Mechanisms

Each entry gives the idea, the code, its knobs with their defaults, the provenance tag, and the graded evidence where one exists. Graded ratios come from the author's own telemetry and are listed in `docs/SUBMISSIONS.md`.

### M1. The cumulant chain itself (backbone). *Inherited.*

The state is propagated layer by layer:
- means and covariance exactly;
- third cumulants as factored sources, one per birth layer, each with legs A and P plus thin residual legs;
- the fourth cumulant as the diagonal the mean program needs.

The D3 and D21 contractions are fused into two dense einsums per layer plus thin terms (`_dslices`, [L8103-L8564](../estimator/estimator.py#L8103-L8564); the formulas are in its docstring). Pair moments come from the merged term programs ([L6004-L6196](../estimator/estimator.py#L6004-L6196)) and are converted to cumulants with `PK2K_TABLE` ([L6422-L6509](../estimator/estimator.py#L6422-L6509)). The tables and the Wick/Hermite coefficients are ARC's ([L245-L248](../estimator/estimator.py#L245-L248)). The chain structure, the module docstring's V16b-V29 history ([L85-L236](../estimator/estimator.py#L85-L236)) and `_sym_product` ([L7694-L7723](../estimator/estimator.py#L7694-L7723)) are V29's.

Kill-switch: `MERGE_OFF=1` ([L276](../estimator/estimator.py#L276)) restores V29's values for the merge-era switches. It does not undo later patches.

### M2. Kappa4 regeneration tables: LAM, REF_R, beta. *Inherited (V29), one scale and the row metric changed.*

V29 regenerates the kappa4 diagonal each layer without per-source memory:
- the transported diagonal `dG` = (W∘W) g_prev + lambda (var − (W∘W) var_prev);
- an off-diagonal coefficient lambda that adapts per network as lambda_l = LAM[l] · clip((mean dG / mean var) / REF_R[l], 0.5, 2)^beta.

The code is at [L5884-L5906](../estimator/estimator.py#L5884-L5906). The tables are `LAM` ([L250](../estimator/estimator.py#L250), 16 values, multiplied by `V17_LAM_SCALE`=0.95 at [L308](../estimator/estimator.py#L308)) and `REF_R` ([L309](../estimator/estimator.py#L309), 15 values), with `V25_BETA`=1.0 ([L310](../estimator/estimator.py#L310)).

The row estimate `g4row = dG x metric` ([L5915](../estimator/estimator.py#L5915)) uses the per-row metric sum_i W_ai^2 under `H18`=1 ([L259](../estimator/estimator.py#L259); idea credited to Yinita/Niko) instead of the constant `METRIC_C`=2.0. These tables are tuned together with the rest of the chain. The write-up (section "co-adapted iterate") explains why their levels were kept.

### M3. LOT: leading-order-tree shape of the kappa4 diagonal (PORTLOT). *New.*

LOT gives the neuron-to-neuron shape of the kappa4 diagonal and keeps V29's per-layer level. It projects the leading-order functional on a rank-K basis of the current pre-activation covariance (`_lot_basis`, [L6640-L6660](../estimator/estimator.py#L6640-L6660): a one-pass range finder with a weight-column sketch). It then forms the projected term (`proj_only`), the same-hub term (`response_same`) and the star term (`response_star`), and rescales the sum to V29's level (`_lot_g4`, [L7373-L7513](../estimator/estimator.py#L7373-L7513)). The result replaces `g4row` in the mean program ([L5960-L5965](../estimator/estimator.py#L5960-L5965)).

- **Scale guard.** If V29's level and LOT's own level, `mg / (12 m1)`, disagree beyond [`PORTLOT_RMIN`, `PORTLOT_RMAX`] = [0.35, 1.5], V29's diagonal is kept for that layer ([L7487](../estimator/estimator.py#L7487)). These are the only LOT constants. Both are hand-set guard bounds, and LOT has no fitted constant.
- **Other uses of LOT's terms.** They also feed HZ's features ([L7489-L7507](../estimator/estimator.py#L7489-L7507)) and the D21 tail basis.
- **Knobs** ([L312-L321](../estimator/estimator.py#L312-L321)): `PORTLOT`=1 (via `_ON`), `PORTLOT_K`=12, `PORTLOT_Q`=1, `PORTLOT_LMIN`=1 (all layers from 1), `PORTLOT_STAR`=1, `PORTLOT_SAME`=1. Exact cost cuts: `LOTW`=1 ([L351](../estimator/estimator.py#L351)), `LOTD`=1 ([L528](../estimator/estimator.py#L528)), `J2_LOT`, `TM_LOTHOT`, `TM_LS0`.
- **Graded evidence.** Control #336912 (s43 with `PORTLOT=0`): raw x1.3575, FLOPs x1.0929, score x1.4835 against s43.
- **Coupling with HZ.** HZ selects its rows and builds its features inside `_lot_g4` (`_hz_st` is filled only at [L7499-L7502](../estimator/estimator.py#L7499-L7502)). With `PORTLOT=0`, `_hz_finish` returns at [L7235-L7236](../estimator/estimator.py#L7235-L7236) and applies no HZ or NLHZ correction; this was checked on the released file. So #336912 removes LOT and HZ together. Against the HZ-off control #336911 it gives raw x1.1052 and score x1.2850 (`controls/README.md`).

### M4. Demand-driven D21: exact kink rows and a rank-2 tail; CDP and TDP. *New.*

The final layer reads the D21 slice mostly through rows near the ReLU kink. So `_d21_demand` ([L7807-L8101](../estimator/estimator.py#L7807-L8101)) computes exactly the rows with |alpha| <= `D21_CUT` = 2.25, rounded up to a multiple of `D21_GRAN` = 64, and projects all other rows onto a rank-`D21_RANK` = 2 basis (`D21_BASIS` = `active_rf`). Knobs: [L625-L634](../estimator/estimator.py#L625-L634); per-layer overrides `D21_CUT_L`, `D21_RANK_L` (empty by default).

- **CDP** ([L635-L641](../estimator/estimator.py#L635-L641); code at [L7828-L7840](../estimator/estimator.py#L7828-L7840) and later in the function): young-hub D21 columns with alpha <= −`CDP_C` (1.625) are filled by a rank-`CDP_R` (1) left projection, on layers 4-14.
- **TDP** ([L642-L653](../estimator/estimator.py#L642-L653), `_tdp_basis` / `_tdp_mask_band` [L656-L671](../estimator/estimator.py#L656-L671), uses at [L4916-L4921](../estimator/estimator.py#L4916-L4921) and [L6230-L6236](../estimator/estimator.py#L6230-L6236)): the band of previous-layer rows with alpha in (−`SDT_C`, −`TDP_C`] (`TDP_C`=0.5) is handled at rank `TDP_R`=2 in the frozen join (`TDP_JOIN`=1), with the left-out cap `TDPCAP2`=1.
- **Graded evidence.** Control #336913 (s43 with `D21_CUT=99`, every row exact): raw x0.9970, FLOPs x1.0834, score x1.0802 against s43. The projection buys about 8% FLOPs for 0.3% raw.

### M5. Dropped neurons: soft-dead tail, active set, Edgeworth (radial) means. *New.*

**Soft-dead tail and active set.** Nearly dead neurons are handled in two ways:
- **Soft-dead tail (SDT)** ([L481-L516](../estimator/estimator.py#L481-L516)): neurons with alpha < −`SDT_C` (2.25) are moved to the end of the live order and carry no third-cumulant or hub content. Their D21 columns are zero beyond −`SDT_COL` (2.0).
- **Active set (AS)** ([L1060-L1101](../estimator/estimator.py#L1060-L1101), loop code [L4944-L5034](../estimator/estimator.py#L4944-L5034)): neurons with alpha < −`AS_CDEAD` (3.0; 2.5 on layers 6-9 via `AS_CDEAD_L`) are dropped from the state. Their means stay Gaussian ReLU means, sigma (phi(a) + a Phi(a)) ([L4989](../estimator/estimator.py#L4989)), with the mean correction `AS_MCORR`=1 and the kappa4 constant `AS_GCONST`=1. On the final layer, `AS_FINAL_*` sets the drop rule.

**Edgeworth means for the dropped rows.** `_fin_edge` (final layer, [L7194-L7222](../estimator/estimator.py#L7194-L7222), call [L6213](../estimator/estimator.py#L6213)) and `_mid_edge` (intermediate layers, [L7165-L7192](../estimator/estimator.py#L7165-L7192), call [L6219](../estimator/estimator.py#L6219)) add a two-term Edgeworth correction to those Gaussian means. The skewness and kurtosis proxies `s3` and `k4` are fitted per network on the kept rows with |alpha| >= 1.5 (a "radial" proxy). The formula is in the docstrings.
- Knobs ([L1103-L1114](../estimator/estimator.py#L1103-L1114)): `FIN_EDGE`=1 (`FIN_EDGE_K4S`=1.3, `FIN_EDGE_S3S`=1.0) and `MID_EDGE`=1 (`MID_EDGE_K4S`=1.2, `MID_EDGE_S3S`=1.4).

### M6. Pair closures: two-term (3,1), path birth, Mehler m=3. *Ported, re-derived and inherited parts.*

- **`K31TWO`=2** ([L252-L258](../estimator/estimator.py#L252-L258), code [L5928-L5946](../estimator/estimator.py#L5928-L5946)): the (3,1) pair slice wk431_ij = b1_l v_j C_ij + kappa3_j D21[j,i] / v_j everywhere.
  - The table `K31TWO_B1` ([L255](../estimator/estimator.py#L255)) is kai_leal_gonzalez's (repository al07-alt, MIT; see `THIRD_PARTY_NOTICES.md`).
  - `K31TWO_B2`=1.2 and `K31TWO_B1S`=1.0 are levels set in this work.
  - `EF_K31T21`=1 ([L683](../estimator/estimator.py#L683)) fixes the orientation in the (2,1) program.
- **`PATHB`=1** ([L270-L272](../estimator/estimator.py#L270-L272), code [L6395-L6400](../estimator/estimator.py#L6395-L6400)): the K22 path birth of ARC's augmented program, added to the kappa4 to kappa3 feed.
- **`MEHLER3`=1** ([L758-L760](../estimator/estimator.py#L758-L760), code [L6131-L6148](../estimator/estimator.py#L6131-L6148)): the Gaussian Mehler m=3 pair terms C_off^3/3! w(3,p) w(3,q) added to the (1,1), (2,1) and (2,2) programs, with strength `MEHLER3_S`=1.0.

### M7. Consistency removals: marg2 and D² halving. *marg2 re-derived; halving new.*

The pair closure's merged programs carry Edgeworth corrections that were tuned together. Two removals of same-order double counts shipped:
- **marg2.** `MARG2`=1 with `MARG2_C`=2.0 ([L693-L719](../estimator/estimator.py#L693-L719), mask at [L6081-L6092](../estimator/estimator.py#L6081-L6092)) keeps the k=2 marginal Edgeworth correction of the pair Wick coefficient only where alpha <= −2. The rule is Artyom Kuznetsov's (Artyom_Mazur), from forum topic 18249 sec. 3.5.
- **D² halving.** `D2SQ_S`=0.5 ([L725](../estimator/estimator.py#L725)) halves the D21·D21 diagrams of the merged program through `_dgs_coef` ([L752-L755](../estimator/estimator.py#L752-L755)). `CK22_S`=1.0 leaves the C∘K22 diagram unchanged.

`MARG1` (the k=1 analogue) is off and was harmful when tested. Both changes cost zero FLOPs.

### M8. Source age tiers and frozen joins. *Inherited (V21/V24), ported (FIJA2, OLD_HOT1), new (row windows, feedback schedule).*

Sources older than `V21_AGE_OLD`=4 transports share one rank-`V21_R_OLD`=448 basis (tier 1). Those older than `V24_AGE_OLD2`=8 are confined to a rank-`V24_R_OLD2`=224 sub-basis (tier 2). The constants are class attributes at [L4410-L4416](../estimator/estimator.py#L4410-L4416), and the docstrings at [L118-L136](../estimator/estimator.py#L118-L136) describe the scheme. The released defaults differ from V29's (384 and 7).

- **`FIJA2`** ([L291-L302](../estimator/estimator.py#L291-L302); joins at [L5091-L5159](../estimator/estimator.py#L5091-L5159), tier-2 move at [L5162-L5180](../estimator/estimator.py#L5162-L5180), first-union ordering at [L5361-L5402](../estimator/estimator.py#L5361-L5402)): freezes the shared basis after the first join, ported from LordProcciuto's `X_FIJA`. The `FX_*` follow-ups are exact savings.
- **`OLD_ROWW` and `OLD_HOT1`** ([L370-L379](../estimator/estimator.py#L370-L379)): form old-tier legs only on rows that are not always-on.
- **Feedback legs** (`TD_*`, [L278-L290](../estimator/estimator.py#L278-L290), and `FBL_*` [L3674-L3678](../estimator/estimator.py#L3674-L3678)): set which births get feedback rank 4 or 16.

### M9. HZ: deferred, physics-featured linear correction. *New.*

At each layer from 2 on, the code stores the legs and the LOT terms of the `HZ_K[li]` rows where the c4 weight is largest (`_hz_select`, [L6919-L6947](../estimator/estimator.py#L6919-L6947)). After the forward sweep, `_hz_finish` ([L7224-L7371](../estimator/estimator.py#L7224-L7371)) does three things:
1. **Pullback.** It runs one deferred backward pullback of those legs through all earlier layers (`_hz_pull_w` [L6949-L7020](../estimator/estimator.py#L6949-L7020) on the `_HZE` engine [L4143-L4404](../estimator/estimator.py#L4143-L4404)). This gives the exact chain and induced sums `acc_c` and `acc_i`.
2. **Correction.** It forms the fitted correction on two sets of rows:
   - the selected rows: the kappa4 replacement, with 7 weights;
   - every live row: 17 rider weights.
3. **Transport.** It carries the correction to the final layer with Phi(alpha) W ([L7341-L7371](../estimator/estimator.py#L7341-L7371)).

The weights are `HZ_TH` ([L620](../estimator/estimator.py#L620)): 16 entries, None for layers 0-1, then 14 rows of 24. They are folded into coefficient tables in `_hz_setup` ([L6662-L6793](../estimator/estimator.py#L6662-L6793)).

- **Knobs** ([L517-L537](../estimator/estimator.py#L517-L537)): `HZ_ON`=1 ([L322](../estimator/estimator.py#L322)), `HZ_K`, `HZ_RIDERS`=1, `HZW`=1, `HZ_LEV`=5, `HZ_RCH`=1024, and setup-time caps (section 6).
- **G4's θ** is the least-squares refit made together with the NLHZ head on 400 public networks (`training/nl_train3.py` step 1). It equals `round(theta, 8)` of that fit exactly.
- **Graded evidence.** Control #336911 (s43 with `HZ_ON=0`): raw x1.2283, FLOPs x0.9399, score x1.1545 against s43.

### M10. NLHZ: a small non-linear head on HZ's inputs. *New (parallel work credited in the write-up).*

One MLP is shared by layers 2-15:
- inputs: 37 standardised per-row features (rider multipliers, alpha, log var, slab ratios, five carried-state inputs and two D21 reductions `D21 @ w2`, `w1 @ D21` gathered at [L6023-L6054](../estimator/estimator.py#L6023-L6054));
- architecture: 37 → 64 → 64 → 1, ReLU.

The output is added to the rider slab before the transport (`_nlhz_apply`, [L565-L618](../estimator/estimator.py#L565-L618), called at [L7338-L7339](../estimator/estimator.py#L7338-L7339); tables built in `_nlhz_build` [L541-L551](../estimator/estimator.py#L541-L551)). The switch is `NLHZ`=1 ([L537](../estimator/estimator.py#L537)). The weights and per-layer standardisation are the dict `_NLHZ_P` ([L538](../estimator/estimator.py#L538); 8,554 numbers).

Graded: NLHZ #337317 scored x0.97831 against its predecessor #337231. The v5D input set (G3 #337411) is part of the G2 to G3 step.

### M11. The final-loss head (G4's only change). *New (final-loss training also used by others; credited in the write-up).*

G4 = G3 with only line 538 replaced. The v5D head was fine-tuned directly on the final-layer MSE, back-propagating through the exact Phi·W transport of dumped tables with Adam at lr 1e-4 and early stopping on 140 held-out networks (`training/hx_ft.py`). Regenerating line 538 from the fine-tuned head with `training/mk_nlhz_patch5.py` reproduces the shipped literal byte for byte (see `training/README.md`). Graded: x0.99663 against G3.

### M12. Fast matmul engine. *V29's `_Strassen`; the Z-order core and everything listed with it are new.*

All large products go through `_FMMd` ([L3765-L3864](../estimator/estimator.py#L3765-L3864)):
- **At the suite shape:** the Z-order Strassen-Winograd core `ZFMM` / `_ZW` ([L1921-L3542](../estimator/estimator.py#L1921-L3542), [L3748-L3763](../estimator/estimator.py#L3748-L3763)). It has:
  - one 7-ary in-place buffer per operand side (`ZF7`, [L1504-L1521](../estimator/estimator.py#L1504-L1521));
  - the Karstadt-Schwartz alternative basis (`ZF7_AB`);
  - cost-model level choice with padding (`LX`, `PAD_*`, [L1595-L1600](../estimator/estimator.py#L1595-L1600), [L1809-L1819](../estimator/estimator.py#L1809-L1819));
  - exact-fringe planning (`PEEL`=1, [L1641-L1651](../estimator/estimator.py#L1641-L1651); it acts only in deep mode ([L4787](../estimator/estimator.py#L4787)) and only on plans listed in `_POOL_SPEC`. At the suite shape, deep mode is on from the first scored predict when setup finished its pre-allocation ([L4580](../estimator/estimator.py#L4580)), otherwise from the second).
- **Otherwise:** V29's `_Strassen` ([L1122-L1494](../estimator/estimator.py#L1122-L1494)) is the fallback for off-suite shapes, for `FMM_OFF=1` and after a Z-core exception (unless `FMM_STRICT=1`).

HZ's pullback has its own engine, `_HZE`. Named pooled buffers come from `_Pool` ([L3866-L3956](../estimator/estimator.py#L3866-L3956)), and the setup list of buffers and plans is `_POOL_SPEC` ([L1909](../estimator/estimator.py#L1909), recorded at build time). The other cost families (`AU_*`, `OPD*`, `RRES*`, `TM_*`, `J2_*`, `NLK_*`, `DN_*`, `THIN_*`, `C7_*`) are exact or rounding-level call-count and FLOP cuts. Every one of them defaults on, and every one of them was checked against the parent op stream when introduced.

### M13. Guards. *New.*

- **Off-suite fallback** (section 7).
- **Time-based HZ skips** `HZ_TMAX_L` / `HZ_TMAX` / `HZ_TMAX_P` ([L1731-L1733](../estimator/estimator.py#L1731-L1733)).
- **Residual guard** `RGUARD` ([L1785-L1802](../estimator/estimator.py#L1785-L1802)).
- **Projected-wall degrade ladder** `WG` ([L1735-L1784](../estimator/estimator.py#L1735-L1784)): rung 0 turns HZ off, rung 1 turns D21 off from a layer on, rung 2 drops the final-layer source transport ([L4908](../estimator/estimator.py#L4908)).
- **Array-store guard** `ARR_GUARD`=85000 live handles ([L3958-L3961](../estimator/estimator.py#L3958-L3961), `_rb_arr_clear` [L3973-L4000](../estimator/estimator.py#L3973-L4000)).

Section 6 says which of these can change a result.

### M14. QRV: complete QR in the first-union ordering. *Ported idea, new variant; billing disclosure.*

`QRV`=1 ([L3562](../estimator/estimator.py#L3562)) builds the complement of the ordered basis with `fnp.linalg.qr(Ur, mode='complete')` ([L5379-L5381](../estimator/estimator.py#L5379-L5381)). This is one call per network on a 448 x 224 matrix. `QRV=0` uses LordProcciuto's `qr(concat[Ur, eye])` ([L5383-L5385](../estimator/estimator.py#L5383-L5385)). flopscope bills the complete Q at the reduced price (open flopscope issue #269). `docs/PROVENANCE.md` states the measured share: 0.026% of the bill billed, about 0.034% underpriced.

## 4. All 374 knobs by family

Counts come from an AST pass over the file (`verify.py` checks the total). "Default stance" says how the family is set at the defaults.

| Family | Knobs | Examples (default) | Default stance |
|---|---:|---|---|
| Fast matmul engine (Strassen/Winograd, ZF7, LX, PAD, PEEL, thin and dense routes) | 85 | `V26_STRASSEN`=5, `ZF7`=1, `ZF7_AB`=1, `LX`=1, `PEEL`=1, `PAD_LMAX`=6, `DN_*`, `THIN_*`, `C7_*`, `TA_*` | on; exact or rounding-level |
| Demand-driven truncation (D21 rows, CDP, TDP, SDT tail, active set, dropped-row tails) | 66 | `D21_CUT`=2.25, `D21_RANK`=2, `CDP_C`=1.625, `TDP_C`=0.5, `SDT_C`=2.25, `AS_CDEAD`=3.0, `FIN_EDGE`=1, `MID_EDGE`=1 | on; accuracy-for-cost trade-offs set by screens |
| Source machinery (births, feedback legs, age tiers, frozen joins) | 44 | `V21_R_OLD`=448, `V24_R_OLD2`=224, `FIJA2`=1, `OLD_HOT1`=1, `TD_FROZEN`=`6,8,10,12,14`, `QRV`=1 | on |
| Debug, emulation and strict modes | 44 | `*_DEBUG`=0, `SD_*`/`HUB_*` emulations = 0, `FMM_STRICT`=0, `HZ_STRICT`=0, `WG_FORCE`='' | off (research only) |
| Kappa4 regeneration, pair closures, term programs | 35 | `V17_LAM_SCALE`=0.95, `V25_BETA`=1.0, `H18`=1, `K31TWO`=2, `PATHB`=1, `MEHLER3`=1, `MARG2`=1, `D2SQ_S`=0.5, `MARG1`=0, `LC_K22`=0 | mixed; accuracy-relevant |
| Exact call-count and op-stream cuts | 35 | `AU_*`, `YR_*`, `OPD*`, `RRES*`, `TM_*`, `NLK_PK`, `J2_*`, `DSF` | on; bit-identical or rounding-level |
| Guards (wall, residual, array store, late-HZ skips) | 19 | `HZ_TMAX`=107.5, `HZ_TMAX_L`=103.5, `HZ_TMAX_P`=117, `RGUARD`=1, `WG`=1, `ARR_GUARD`=85000 | armed; see section 6 |
| HZ correction and NLHZ head | 17 | `HZ_ON`=1, `HZ_K`, `HZW`=1, `HZ_RIDERS`=1, `NLHZ`=1, `HZ_SETUP_T`=0.6 | on |
| Residual/setup plumbing (public views, pools, prealloc, warm-up) | 16 | `PUBLIC_Z_VIEWS`=1, `M2_PREALLOC`=1, `V26_WARM`=1, `PUBLIC_Z_SETUP_MAX_T`=2.6 | on; no effect on values |
| LOT kappa4 shape (PORTLOT family) | 13 | `PORTLOT`=1, `PORTLOT_K`=12, `PORTLOT_LMIN`=1, `PORTLOT_RMIN`/`RMAX`=0.35/1.5, `LOTW`=1, `LOTD`=1 | on |
| **Total** | **374** | | |

26 of the 374 knobs come from V29. 23 keep V29's default. Three changed: `V21_R_OLD` 384 → 448, `V24_AGE_OLD2` 7 → 8, `V28_STRASSEN_SB` 5 → 6. The other 348 knobs were added by this work's patches, including the ported ones.

## 5. Code that is dead at the default settings

These paths are present but never run by the graded configuration on the scored shape. Treat them as history:

| Lines | Path | Why dead |
|---|---|---|
| [L249](../estimator/estimator.py#L249), [L6197-L6209](../estimator/estimator.py#L6197-L6209) | V29's online ridge correction (`CORR_BETA`, 208 numbers) | `V17_NO_CORR`=1 |
| [L7660-L7680](../estimator/estimator.py#L7660-L7680) | `_response31` (late projected (3,1) response) | `K31TWO`=2 replaces it |
| [L731-L749](../estimator/estimator.py#L731-L749) | `_marg1_slots` | `MARG1`=0 |
| [L5365-L5368](../estimator/estimator.py#L5365-L5368) | exact eigenspace ordering | `DN_EIGH`=0 |
| [L5383-L5385](../estimator/estimator.py#L5383-L5385) | square-QR ordering | `QRV`=1 |
| [L621-L623](../estimator/estimator.py#L621-L623) and uses | k22swap level reference | `LC_K22`=0 |
| [L260-L265](../estimator/estimator.py#L260-L265) and uses | `H18_K22`, `H18_K31`, `H18F*` metric variants | default 0 |
| [L323-L337](../estimator/estimator.py#L323-L337) and uses | `SD_*` / `HUB_*` emulations | default 0 |
| [L1700-L1702](../estimator/estimator.py#L1700-L1702) | `_HZG_EXT` setup list | `HZG_EXT`=0 |
| [L8565-L8568](../estimator/estimator.py#L8565-L8568) | V29's `__main__` footer | imports `local_engine`, which is not part of this release; never executed on import |

Also off by default: `LX_MULTI`, `TDP_Y`, `TDP_QC`, `JOINT_CP_ON`, `D21_LIFT_FMM_M`, `DN_KP`, `AS_FULLMEAN`, `AS_FORCE_PERM`, `PATHB_NC`, `FMM_OFF`, `V26_STRASSEN_NEW` and every `*_DEBUG` branch. `_Strassen` is live only in the fallback cases of M12.

## 6. Time guards and how to get a timing-independent result

Some branches depend on wall time. Here is what can trigger on the grader and what cannot.

| Guard | Reads | Default | Can it change a graded result? |
|---|---|---|---|
| `HZ_TMAX_L` (top of each layer, [L4897-L4900](../estimator/estimator.py#L4897-L4900)) | `time.perf_counter` since predict start | 103.5 s | Yes. HZ work stops and the prediction is the no-HZ one. This fired on a slow grader worker in s44 #336845 (16 MLPs). In G4's graded run no MLP shows the skip signature (wall max 102.1 s). |
| `HZ_TMAX` (before the finish, [L6628-L6631](../estimator/estimator.py#L6628-L6631)) | `perf_counter` | 107.5 s | Yes, same effect. |
| `HZ_TMAX_P` (inside the pullback, [L6974-L6975](../estimator/estimator.py#L6974-L6975), [L7252-L7253](../estimator/estimator.py#L7252-L7253)) | `perf_counter` | 117 s | Yes, same effect. |
| `RGUARD` ([L4887-L4892](../estimator/estimator.py#L4887-L4892), [L6618-L6623](../estimator/estimator.py#L6618-L6623)) | live residual of the flopscope *client* context (`flopscope._budget._active_context`) | 0.33 / 0.335 s | Not observed. The context is absent in-process (the guard returns None), and the guard never armed in any graded run. It *does* arm under a local flopscope client/server harness when the local residual exceeds 0.33 s. |
| `WG` ladder ([L4893-L4896](../estimator/estimator.py#L4893-L4896), [L6624-L6627](../estimator/estimator.py#L6624-L6627)) | wall time of the client context | 116 / 118 s | No. It needs the same client context. Fire drills #337103 and #337104 forced a rung, yet graded with per-MLP FLOPs equal to their reference #337102. (A `perf_counter` variant, WG2, did arm in the test-only #337114. WG2 is not in this file.) |
| Setup caps `HZ_SETUP_T` 0.6, `HZG_MAX_T` 1.5, `HZ6_SETUP_T` 0.25, `PUBLIC_Z_PREWARM_T` 1.0, `PUBLIC_Z_SETUP_MAX_T` 2.6, `M2_PREALLOC_T` 2.5, `OPD_ZT_SETUP_T` 0.5, `RRES_PREWARM_T` 0.4 | `perf_counter` during setup | as listed | Indirectly. They decide how many plans are prebuilt. Plans not prebuilt are built during the predict. That changes the billed FLOPs and, through engine-route choices, can change predictions at rounding level (seen as run-to-run differences of the same file on the local harness). |
| `ARR_GUARD` | count of live array handles | 85000 | Deterministic (a count, not a time). |

**Timing-independent runs.** For a run whose output does not depend on machine speed, set:

```
HZ_TMAX=0 HZ_TMAX_L=0 HZ_TMAX_P=0 RGUARD=0 WG=0 \
HZG_MAX_T=600 HZ_SETUP_T=600 HZ6_SETUP_T=600 PUBLIC_Z_PREWARM_T=60 PUBLIC_Z_SETUP_MAX_T=600 M2_PREALLOC_T=600 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

This is the setting of the deterministic emulation runs; `verify.py --smoke` sets the first line. With it, predictions equal what the grader computes whenever no guard fired there, as in G4's graded run. Billed FLOPs can still differ from the grader by the setup-plan effect above.

## 7. Off-suite shapes: the fallback

The grader's smoke test runs small shapes (for example 64x16 and 256x32) before scoring, and an exception there fails the whole submission. For any shape other than width 1024 and depth 16, two layers of gating apply:

1. **Knob switch-off in `predict`.** `predict` ([L4611-L4630](../estimator/estimator.py#L4611-L4630)) switches off, for that predict only, the 80 names in `_POST_S26_KNOBS` ([L4407](../estimator/estimator.py#L4407)). These are everything added after build s26:
   - CDP and TDP;
   - the Edgeworth tails, marg2/marg1, the Mehler terms, the path birth;
   - PEEL and most exact engine cuts;
   - `QRV`, the NLHZ head, and the `RGUARD` / `WG` guards.
2. **Suite-shape gates inside `_predict_core`.** `riders` is true only for width 1024 and depth 16 ([L4666](../estimator/estimator.py#L4666)). With `riders` false:
   - the kappa4 regeneration is off (`regen`, [L4772](../estimator/estimator.py#L4772)), and so is LOT (`lot = PL_ON and regen`, [L4773](../estimator/estimator.py#L4773));
   - the active set and the HZ correction are off ([L4840](../estimator/estimator.py#L4840));
   - the matmul dispatcher uses V29's `_Strassen` instead of the Z core.

   The MERGE docstring states the same for the merge-era pieces ([L75-L84](../estimator/estimator.py#L75-L84)).

The result is essentially V29's own small-shape path. So a small-shape run (`verify.py --smoke`, default 64x4) tests the fallback, not the code that produced the scores. Use `--smoke 1024x16` (about 80 s and 4.5 GB on one core) to run the scored path.

## 8. Differences from s43, the base of the graded controls

The controls in `controls/` modify s43 (#336780, 7,861 lines), not this file. Between s43 and G4:

- **The three control knobs are unchanged.** `PORTLOT` (G4 line 312), `HZ_ON` (322) and `D21_CUT` (625) exist with the same definitions and defaults.
- **44 knobs are new.** They cover `PATHB`, `NLHZ`, `TDPCAP2`, `MEHLER3`, the `TM_*` cuts, `PEEL`, the `WG` ladder, `RGUARD` and `FBL_*`.
- **8 defaults changed:**
  - `K31TWO_B2` 1.0 → 1.2;
  - `HZ_K` (192, 96, 64, 256, ... → 96, 0, 32, 96, ...);
  - `TDP_C` 1.25 → 0.5;
  - `MID_EDGE_K4S` 1.3 → 1.2;
  - `MID_EDGE_S3S` 1.2 → 1.4;
  - `HZ_TMAX` 105 → 107.5, `HZ_TMAX_L` 100 → 103.5, `HZ_TMAX_P` 112 → 117.
- **The embedded tables were refit.** `HZ_TH` changed, `_NLHZ_P` is new, and `_POOL_SPEC` was re-recorded.

So a knob set on G4 removes the same mechanism, but from a different stack. The graded control numbers belong to s43.

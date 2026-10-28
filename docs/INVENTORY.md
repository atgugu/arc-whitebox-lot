# Inventory of constants in `estimator/estimator.py` (#337608)

Every embedded table, every fitted parameter and every knob default that acts as a model constant, with how it was set. Counts come from an AST pass over the released file. Nothing was imported or executed. Line links point into `estimator/estimator.py`, which is never edited.

Classes used below:

- **derived:** follows from the mathematics (Hermite/Wick coefficients, combinatorial tables, exact coefficients).
- **fitted:** estimated from data by least squares or gradient descent.
- **hand-set:** a threshold, rank, size or scale chosen by a screen or by design.
- **inherited:** kept from 504aldo's V29, or from ARC through V29.
- **recorded:** written automatically at build time and with no effect on the predicted values.
- **identity:** a scale left at 1.0.
- **measured:** a timing profile measured on the development machines (used only by the wall guard).

## Summary

- **Fitted numbers that affect the predictions:** 8,935 in 5 tables.
  - `_NLHZ_P` (8,554) and `HZ_TH` (336 slots, 303 non-zero) are new (introduced in this work). Together they are 8,890, or 99.5% of the fitted numbers.
  - `LAM` (16) and `REF_R` (15) are V29's.
  - `K31TWO_B1` (14) is ported from kai_leal_gonzalez (repository al07-alt).
  - V29's `CORR_BETA` (208) is embedded but switched off.
- **LOT has no fitted constant.** Its shape is computed from the network, and its level is V29's regenerated diagonal (`g4m = cand x mean(g4row) / mean(cand)`, [L7488](../estimator/estimator.py#L7488)). The only LOT constants are the hand-set guard bounds 0.35 and 1.5 and the size K = 12.
- **Derived tables:** 7 entries in table A (10 small tables), 450 numbers including indices.
- **Recorded tables:** 3 tables, 2,461 numbers (setup lists; no effect on values).
- **Knob defaults that act as model constants** (table B): 57 entries. Of these, 45 are hand-set, 6 inherited from V29, 2 derived exact coefficients and 4 identity scales.
- **Other knob defaults:** the remaining numeric defaults set engine levels, cache and view limits, time caps and guard thresholds (see `docs/READING_GUIDE.md` section 4, 374 knobs in all). Most knobs are on/off switches.
- **Numeric literals inside functions:** 4,559 (533 other than 0, 1, 2, 3, 4, 0.5, -1). Nearly all are indices, shapes or exact coefficients. The value-relevant ones are listed in section C.

## A. Embedded tables

| Name | Line | Numbers | Non-zero | Class | Provenance | Content | Status at defaults |
|---|---|---:|---:|---|---|---|---|
| `WICK_PAIRS` | [L245](../estimator/estimator.py#L245) | 42 | 38 | derived | inherited (ARC, via V29) | 21 (k, p) index pairs of the Wick coefficients w(k, p) | live |
| `WICK_UNIFIED` | [L246](../estimator/estimator.py#L246) | 108 | 79 | derived | inherited (ARC, via V29) | Hermite/Wick polynomial coefficients of the ReLU moments (rational) | live |
| `TERM_SPECS` | [L247](../estimator/estimator.py#L247) | 182 | 181 | derived | inherited (ARC, via V29); 4 terms appended at L681-L685 (new) | term programs: operand names, Wick indices, rational coefficients (1/6, 1/24, ...) | live |
| `PK2K_TABLE` | [L248](../estimator/estimator.py#L248) | 79 | 74 | derived | inherited (ARC, via V29) | power-moment to cumulant partitions (integers) | live |
| `CORR_BETA` | [L249](../estimator/estimator.py#L249) | 208 | 200 | fitted | inherited (V29) | per-layer ridge mean correction, 16 x 13 | dead (V17_NO_CORR=1) |
| `LAM` | [L250](../estimator/estimator.py#L250) | 16 | 16 | fitted | inherited (V29) | per-layer lambda table of the kappa4 regeneration (multiplied by V17_LAM_SCALE = 0.95 at L308) | live |
| `K31TWO_B1` | [L255](../estimator/estimator.py#L255) | 14 | 14 | fitted | ported (kai_leal_gonzalez, al07-alt, MIT) | per-layer b1 of the two-term (3,1) closure | live |
| `REF_R` | [L309](../estimator/estimator.py#L309) | 15 | 15 | fitted | inherited (V29) | reference ratios of the adaptive lambda law | live |
| `_NLHZ_P` | [L538](../estimator/estimator.py#L538) | 8,554 | 8,515 | fitted | new | NLHZ head after the final-loss fine-tune: weights W0 37x64, WH 64x64, bH 64, W2 64, per-layer bias B0 14x64 (7,488 network numbers); standardisation M, IS 14x37 and scales rs, c2 (1,066) | live |
| `HZ_TH` | [L620](../estimator/estimator.py#L620) | 336 | 303 | fitted | new | HZ weights: layers 2-15 x 24 (7 kappa4-term weights + 17 rider weights); entries 0-1 are None | live |
| `_MARG2_W` | [L696](../estimator/estimator.py#L696) | 10 | 10 | derived | new | Wick-row indices touched by marg2 | live |
| `_MARG1_W` | [L727](../estimator/estimator.py#L727) | 10 | 10 | derived | new | Wick-row indices touched by marg1 | dead (MARG1=0) |
| `D2_IPS / D1_IPS / D2_OUT / D1_OUT` | [L772](../estimator/estimator.py#L772) | 19 | 19 | derived | inherited (V29) | cumulant index sets of the pair / diagonal programs | live |
| `_HZG_PROF` | [L1690](../estimator/estimator.py#L1690) | 144 | 143 | recorded | new | 36 pullback-layer shapes seen on 34 networks (setup prewarm list; no effect on values) | live (setup) |
| `_HZG_EXT` | [L1700](../estimator/estimator.py#L1700) | 60 | 60 | recorded | new | extra prewarm shapes | dead (HZG_EXT=0) |
| `_POOL_SPEC` | [L1909](../estimator/estimator.py#L1909) | 2,257 | 1,968 | recorded | new | buffer shapes and plan keys recorded at build time for setup pre-allocation (no effect on values; affects billed FLOPs through lazily built plans) | live (setup) |
| `WG_F (knob default string)` | [L1737](../estimator/estimator.py#L1737) | 17 | 16 | measured | new | mean fraction of the predict wall reached at the top of each layer (wall guard only) | inert on the grader |

How the new fitted tables were produced:

- **`HZ_TH`:** least-squares refit of the 24 production HZ columns per layer on 400 public networks with the official N = 1e9 means (`training/nl_train3.py`, step 1). The embedded values are `round(theta, 8)` of that fit exactly (checked against the trained file).
- **`_NLHZ_P`:** written by `training/mk_nlhz_patch5.py` from the head after the final-loss fine-tune (`training/hx_ft.py`). The generator folds the standardisation into `M`/`IS`/`B0` and zeroes 17 constant inputs. Re-running it on the archived head reproduces line 538 byte for byte (`training/README.md`).

## B. Knob defaults that act as model constants

| Knob | Default | Line | Role | How set |
|---|---|---|---|---|
| `PORTLOT_K` | `12` | [L313](../estimator/estimator.py#L313) | LOT: covariance modes of the projection | hand-set (size) |
| `PORTLOT_Q` | `1` | [L314](../estimator/estimator.py#L314) | LOT: range-finder QR passes | hand-set (size) |
| `PORTLOT_LMIN` | `1` | [L315](../estimator/estimator.py#L315) | LOT: first layer | hand-set |
| `PORTLOT_RMIN` | `0.35` | [L318](../estimator/estimator.py#L318) | LOT: lower bound of the level-agreement guard mg/(12 m1) | hand-set |
| `PORTLOT_RMAX` | `1.5` | [L319](../estimator/estimator.py#L319) | LOT: upper bound of the guard | hand-set |
| `V17_LAM_SCALE` | `0.95` | [L308](../estimator/estimator.py#L308) | regeneration: scale of LAM | inherited (V29 default) |
| `V25_BETA` | `1.0` | [L310](../estimator/estimator.py#L310) | regeneration: exponent of the adaptive lambda law | inherited (V29 default) |
| `METRIC_C` | `2.0` | [L251](../estimator/estimator.py#L251) | constant row metric (K22 pair model, kappa4 to kappa3 feed); g4row uses the per-row metric under H18 | inherited (V29 constant) |
| `K31TWO_B2` | `1.2` | [L257](../estimator/estimator.py#L257) | two-term (3,1) closure: level of the kappa3 term | hand-set (screened) |
| `K31TWO_B1S` | `1.0` | [L258](../estimator/estimator.py#L258) | two-term closure: scale of the b1 table | identity |
| `PATHB_KB` | `1.0` | [L271](../estimator/estimator.py#L271) | K22 path birth coefficient | derived (exact coefficient) |
| `MEHLER3_S` | `1.0` | [L759](../estimator/estimator.py#L759) | Mehler m=3 strength (1.0 = the exact 1/3! coefficient) | derived (exact coefficient) |
| `MARG2_C` | `2.0` | [L694](../estimator/estimator.py#L694) | marg2: keep the k=2 marginal correction only where alpha <= -C | hand-set (rule from forum topic 18249; value screened) |
| `D2SQ_S` | `0.5` | [L725](../estimator/estimator.py#L725) | scale of the D21 x D21 pair diagrams | hand-set (screened; holdout-confirmed) |
| `CK22_S` | `1.0` | [L726](../estimator/estimator.py#L726) | scale of the C o K22 pair diagram | identity |
| `RESP_SCALE` | `1.0` | [L337](../estimator/estimator.py#L337) | scale of the projected (3,1) response | identity (path dead at defaults) |
| `HZ_K` | `0,0,0,0,0,96,0,32,96,96,128,192,160,128,192,160` | [L517](../estimator/estimator.py#L517) | HZ: rows per layer that get the kappa4 replacement | hand-set (size; re-tuned per build) |
| `D21_CUT` | `2.25` | [L625](../estimator/estimator.py#L625) | D21: exact rows have |alpha| <= cut | hand-set (graded control #336913 shows it at the knee) |
| `D21_RANK` | `2` | [L626](../estimator/estimator.py#L626) | D21: rank of the tail projection | hand-set |
| `D21_GRAN` | `64` | [L634](../estimator/estimator.py#L634) | D21: row-count granularity | hand-set (grid) |
| `CDP_C` | `1.625` | [L635](../estimator/estimator.py#L635) | CDP: young-hub columns with alpha <= -C are projected | hand-set |
| `CDP_R` | `1` | [L637](../estimator/estimator.py#L637) | CDP: projection rank | hand-set |
| `CDP_LMIN, CDP_LMAX` | `4, 14` | [L638](../estimator/estimator.py#L638) | CDP: layer range | hand-set |
| `TDP_C` | `0.5` | [L642](../estimator/estimator.py#L642) | TDP: transport band alpha in (-SDT_C, -TDP_C] | hand-set |
| `TDP_R` | `2` | [L645](../estimator/estimator.py#L645) | TDP: band projection rank | hand-set |
| `TDP_LMIN, TDP_LMAX` | `5, 15` | [L650](../estimator/estimator.py#L650) | TDP: layer range | hand-set |
| `SDT_C` | `2.25` | [L492](../estimator/estimator.py#L492) | soft-dead tail: alpha < -C carries no kappa3 / hub content | hand-set |
| `SDT_COL` | `2.0` | [L493](../estimator/estimator.py#L493) | soft-dead tail: D21 columns zero beyond -COL | hand-set |
| `SDT_GRAN` | `64` | [L494](../estimator/estimator.py#L494) | soft-dead tail: window grid | hand-set (grid) |
| `SDT_L0` | `4` | [L495](../estimator/estimator.py#L495) | soft-dead tail: first layer | hand-set |
| `AS_CDEAD` | `3.0` | [L1066](../estimator/estimator.py#L1066) | active set: drop neurons with alpha < -C | hand-set |
| `AS_CDEAD_L` | `6-9/2.5` | [L1081](../estimator/estimator.py#L1081) | active set: per-layer threshold override | hand-set (per-layer idea from LordProcciuto) |
| `AS_L0` | `6` | [L1067](../estimator/estimator.py#L1067) | active set: first layer | hand-set |
| `AS_GRAN` | `32` | [L1068](../estimator/estimator.py#L1068) | active set: drop-count granularity | hand-set (grid) |
| `AS_MARGIN` | `4` | [L1089](../estimator/estimator.py#L1089) | active set: margin | hand-set |
| `AS_YMIN` | `12` | [L1063](../estimator/estimator.py#L1063) | active set: engine leaf minimum | hand-set (engine) |
| `AS_FINAL_C` | `3.0` | [L1099](../estimator/estimator.py#L1099) | final layer: drop threshold | hand-set |
| `AS_FINAL_MAX` | `448` | [L1100](../estimator/estimator.py#L1100) | final layer: maximum dropped rows | hand-set |
| `AS_FINAL_KC` | `224` | [L1101](../estimator/estimator.py#L1101) | final layer: kept-count cap | hand-set |
| `FIN_EDGE_AMIN` | `1.5` | [L1106](../estimator/estimator.py#L1106) | final-layer Edgeworth: rows used for the radial proxy | hand-set |
| `FIN_EDGE_K4S` | `1.3` | [L1107](../estimator/estimator.py#L1107) | final-layer Edgeworth: kurtosis scale | hand-set (screened) |
| `FIN_EDGE_S3S` | `1.0` | [L1108](../estimator/estimator.py#L1108) | final-layer Edgeworth: skewness scale | identity |
| `MID_EDGE_AMIN` | `1.5` | [L1111](../estimator/estimator.py#L1111) | mid-layer Edgeworth: rows used for the radial proxy | hand-set |
| `MID_EDGE_K4S` | `1.2` | [L1112](../estimator/estimator.py#L1112) | mid-layer Edgeworth: kurtosis scale | hand-set (screened) |
| `MID_EDGE_S3S` | `1.4` | [L1113](../estimator/estimator.py#L1113) | mid-layer Edgeworth: skewness scale | hand-set (screened) |
| `V21_AGE_OLD` | `4` | [L4410](../estimator/estimator.py#L4410) | tiers: age at which a source joins the shared basis | inherited (V29 default) |
| `V21_R_OLD` | `448` | [L4411](../estimator/estimator.py#L4411) | tiers: rank of the tier-1 basis | hand-set (V29: 384) |
| `V24_AGE_OLD2` | `8` | [L4412](../estimator/estimator.py#L4412) | tiers: age of the move to tier 2 | hand-set (V29: 7) |
| `V24_R_OLD2` | `224` | [L4413](../estimator/estimator.py#L4413) | tiers: rank of the tier-2 sub-basis | inherited (V29 default) |
| `V21_QPASS, V24_QPASS2` | `1, 2` | [L305](../estimator/estimator.py#L305) | tiers: range-finder passes | inherited (V29 default) |
| `FIJA2_PASS` | `1` | [L295](../estimator/estimator.py#L295) | frozen joins: subspace-iteration passes of the first-union ordering | hand-set (line search) |
| `TD_FB_B0` | `3` | [L280](../estimator/estimator.py#L280) | births 0-3 get no feedback legs | hand-set |
| `TD_FROZEN` | `6,8,10,12,14` | [L290](../estimator/estimator.py#L290) | layers with frozen basis rebuilds | hand-set |
| `TD_R_FB, TD_R_RES` | `4, 8` | [L3677](../estimator/estimator.py#L3677) | feedback rank, residual rank | hand-set |
| `FBL_B, FBL_R` | `9, 16` | [L3674](../estimator/estimator.py#L3674) | late feedback rank 16 from birth 9 on | hand-set |
| `OLD_ROW_CP` | `3.0` | [L375](../estimator/estimator.py#L375) | old tiers: always-on threshold for the row windows | hand-set (threshold from LordProcciuto's hot rows) |
| `OLD_ROW_GRAN` | `64` | [L376](../estimator/estimator.py#L376) | old tiers: window grid | hand-set (grid) |

Counts by class: hand-set 45, inherited 6, derived 2, identity 4.

## C. Value-relevant constants written inline

| Constant | Line | Class | Note |
|---|---|---|---|
| clip of the lambda ratio to [0.5, 2.0] | [L5904](../estimator/estimator.py#L5904) | inherited (V29) | adaptive lambda law |
| 12 in the LOT guard `mg / (12 m1)` | [L7487](../estimator/estimator.py#L7487) | derived | kappa4 = 12 Var(psi) at leading order |
| 1/sqrt(2 pi) = 0.3989422804014327 | [L6699](../estimator/estimator.py#L6699) | derived | HZ coefficient tables (`s2p`) |
| 1/6, 1/24, 1/72 in the Edgeworth tails | [L7165-L7192](../estimator/estimator.py#L7165-L7192), [L7194-L7222](../estimator/estimator.py#L7194-L7222) | derived | two-term Edgeworth mean of dropped rows |
| quarter / half caps on the left-out rows of soft-dead windows (`nr // 4`, `nr // 2`) | [L511](../estimator/estimator.py#L511), [L516](../estimator/estimator.py#L516) | hand-set | SDT window; TDP join window with `TDPCAP2` |
| variance floor 1e-10 | e.g. [L5747](../estimator/estimator.py#L5747) | hand-set (numerical guard) | prevents division by zero |


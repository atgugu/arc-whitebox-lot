# Provenance of `estimator/estimator.py` (#337608)

This document separates what the released estimator inherits, what was ported from credited MIT code, what was re-derived from published text, and what is new (introduced in this work). Licence texts are in `THIRD_PARTY_NOTICES.md`. Line links point into `estimator/estimator.py`.

## 1. Identity

- **Released file.** `estimator/estimator.py`: sha256 `e71018ffa112bc2bc4274798c850fe34170462a88b320522229d8b03c001271f`, 643,721 bytes, 8,568 lines. It is byte-identical to the file graded as submission #337608 ("G4"; created 21 Oct 2026 19:14:15 UTC, graded 20:26 UTC, 100/100 MLPs).
- **Upload package.** The package uploaded for #337608 was built 8 s before the submission was created. Its `manifest.json` records the same sha256, and its `estimator.py` hashes to it. The manifest is reproduced verbatim in `estimator/manifest.json`, and `verify.py` checks all of this.
- **Other graded files.** The same check (package manifest = local file) holds for every submission listed in `docs/SUBMISSIONS.md`, including the control base s43 (#336780) and the three graded controls.

## 2. Lineage

```
ARC mlp_cumulant_propagation (MIT; arXiv:2605.05179)   cumulant propagation, Hermite/Wick tables
        |  ported by 504aldo
504aldo V29 (MIT; whest-p2-cumulant-k3, estimators/estimator_v29.py, sha256 86d9ca9b28e6...)
        |  the production file was byte-identical to V29 until 6 Oct 2026
        |  then a chain of patches, each behind a knob (MERGE of 2 Oct, PORTLOT, D21 demand, SDT/AS, HZ, ...)
s43  #336780  (17 Oct; base of the three graded controls)
        |  + MEHLER3, wall ladder WG, RGUARD, K31TWO_B2 / MID_EDGE refits, TDP cap, late feedback rank,
        |    PATHB, PEEL, TERMITE exact cuts, NLHZ head (see docs/SUBMISSIONS.md for the graded steps)
G3   #337411  (20 Oct; NLHZ head "v5D")
        |  only line 538 (_NLHZ_P) replaced: the head fine-tuned on the final-layer loss
G4   #337608  (21 Oct; this file)
```

G3 was built by a patch-chain build script. The chain is listed in the project's build configuration, with the knob defaults baked into the file and the setup spec `_POOL_SPEC` recorded afterwards. In order:
- `fin_patch_v2`, `thin_patch`, `tdp_patch_j`, `opd_patch`, `mid_patch`;
- `marg2_patch`, `marg1_patch`;
- `lotdiet_patch`, `rres_patch`;
- `rguard_patch`, `mehler3_patch`, `wguard_patch`;
- `k31b2_site_patch`, `fblate_patch`, `tdpcap_patch`, `path_patch`, `peel_patch`;
- `nlhz5D_patch`;
- `termite_patch`, `termite_patch_d`, `termite_patch2`.

G4 replaced `nlhz5D_patch` with the generated `nlhz_patch_hxc`, which differs only in the `_NLHZ_P` literal. The patch scripts are research tools and are not part of this release; the release ships their result.

## 3. Component table

| Component | Class | Where in `estimator.py` | Credit / note |
|---|---|---|---|
| Cumulant propagation, Hermite/Wick closure, factored kappa3, power cumulants, term tables | inherited | tables [L245-L248](../estimator/estimator.py#L245-L248); programs [L791-L1033](../estimator/estimator.py#L791-L1033) | ARC (Wu, Lecomte, Winer, Robinson, Hilton, Christiano), through 504aldo's port |
| Source machinery (A/P legs, thin legs, `_dslices`), memoryless kappa4 regeneration with `LAM`/`REF_R`/beta, D21 feedback legs, final-layer trim, age-gated shared basis and nested tier, Strassen-Winograd products, module docstring V16b-V29 | inherited | [L85-L236](../estimator/estimator.py#L85-L236), [L249-L250](../estimator/estimator.py#L249-L250), [L308-L310](../estimator/estimator.py#L308-L310), `_Strassen` [L1122-L1494](../estimator/estimator.py#L1122-L1494), `_dslices` [L8103-L8564](../estimator/estimator.py#L8103-L8564) | 504aldo V29 (MIT) |
| Frozen joins `FIJA2`, follow-up trims `FX_*`, `OLD_HOT1`; the `QRV=0` ordering step; per-layer threshold idea (`AS_CDEAD_L`) | ported (ideas for `AS_CDEAD_L`) | [L291-L302](../estimator/estimator.py#L291-L302), [L377-L379](../estimator/estimator.py#L377-L379), [L5361-L5402](../estimator/estimator.py#L5361-L5402), [L1081](../estimator/estimator.py#L1081) | siming767-creator/whestbench-lordprocciuto (504aldo + LordProcciuto, MIT); porting approved by the author on 14 Oct 2026 |
| Two-term (3,1) closure `K31TWO` with its `b1` table; (2,1) orientation fix `EF_K31T21` | ported (table copied; new code) | [L252-L258](../estimator/estimator.py#L252-L258), [L683](../estimator/estimator.py#L683) | kai_leal_gonzalez (al07-alt/whest-p2-k31-two-term, MIT; forum topic 18246) |
| Per-row metric `H18` | ported idea | [L259](../estimator/estimator.py#L259) | Yinita (team Niko) |
| marg2 rule | re-derived from text | [L693-L719](../estimator/estimator.py#L693-L719), [L6081-L6092](../estimator/estimator.py#L6081-L6092) | Artyom Kuznetsov (AIcrowd handle Artyom_Mazur), forum topic 18249 sec. 3.5; the transfer to a chain that carries D² and K31 terms is new |
| K22 path birth `PATHB`; Mehler m=3 pair terms `MEHLER3` | re-derived (ARC's published programs; new code) | [L270-L272](../estimator/estimator.py#L270-L272), [L758-L760](../estimator/estimator.py#L758-L760) | ARC's augmented / current term program (commit 93d091a), Kuznetsov's description in 18249 sec. 3.4 |
| Alternative-basis Strassen-Winograd (`ZF7_AB`) | literature | [L1511-L1515](../estimator/estimator.py#L1511-L1515) | Karstadt and Schwartz |
| LOT kappa4 shape (`PORTLOT`) with V29's level, scale guard | new | [L312-L321](../estimator/estimator.py#L312-L321), [L6640-L6660](../estimator/estimator.py#L6640-L6660), [L7373-L7513](../estimator/estimator.py#L7373-L7513) | derivation by a Claude agent, 1-2 Oct (see the write-up). Puffi's Phase 1 response closure is prior art for projecting a fourth-order closure on covariance modes; Kuznetsov's "3+2" tree splits into LOT terms on this estimator's chain |
| Demand-driven D21 rows and rank-2 tail, CDP, TDP | new | [L625-L671](../estimator/estimator.py#L625-L671), `_d21_demand` [L7807-L8101](../estimator/estimator.py#L7807-L8101) | the "demand" principle was stated in parallel by Susupro (forum, Prop. 2) |
| Soft-dead tail, active set, Edgeworth radial means `FIN_EDGE` / `MID_EDGE` | new | [L481-L516](../estimator/estimator.py#L481-L516), [L1060-L1114](../estimator/estimator.py#L1060-L1114), [L7165-L7222](../estimator/estimator.py#L7165-L7222) | dead-neuron removal is common in the field (LordProcciuto and others); the radial Edgeworth tails are new |
| D² halving (`D2SQ_S`) and the consistency reading of the pair closure | new | [L725](../estimator/estimator.py#L725), [L752-L755](../estimator/estimator.py#L752-L755) | parallel findings credited in the write-up (Kuznetsov's consistency principle; Luna's Ward diagnostic) |
| HZ deferred physics-featured correction and its theta | new | [L517-L537](../estimator/estimator.py#L517-L537), [L620](../estimator/estimator.py#L620), [L6662-L7371](../estimator/estimator.py#L6662-L7371) | - |
| NLHZ head; final-loss fine-tune of the head (G4) | new, developed independently | [L537-L618](../estimator/estimator.py#L537-L618) | learned correctors and final-loss training were also used by Hisernberg, hyojun_kwon (18255), mliston (18287), and Susupro (18254); none of their code was read |
| Fast-matmul Z core (`ZFMM`: ZF7 buffers, LX levels, padding, PEEL), HZ pullback engine, call-count and view engineering (`AU_*`, `OPD*`, `RRES*`, `TM_*`, `J2_*`, `NLK_*`, `DN_*`, `THIN_*`, `C7_*`, public views) | new | [L1495-L1914](../estimator/estimator.py#L1495-L1914), [L1921-L3542](../estimator/estimator.py#L1921-L3542), [L3543-L4404](../estimator/estimator.py#L3543-L4404) | a Strassen leaf-size idea is from a forum post by ngmitam |
| Guards: off-suite fallback, `HZ_TMAX*`, `RGUARD`, `WG` ladder, `ARR_GUARD` | new | see `docs/READING_GUIDE.md` sections 6-7 | - |

**Research only, not in the estimator.** The scale-mode extraction of il.miscusi (forum 18248) was also re-implemented, and their "T-chain" replicated. Both lived only in the project's lean research vehicle, which uses 504aldo's `lean_k3_aug`. Neither is in this file.

**Data.**
- **Fitting.** `HZ_TH` and the NLHZ head were fitted on the official public Phase 2 dataset (`hf://aicrowd/arc-whestbench-public-2026@v2-phase2`, split `full`: 1,000 MLPs with per-layer means from N = 1e9 samples). Training used 400 of those networks; another 140 served as early-stopping networks for the fine-tune.
- **Diagnostics only.** keenanpepper's N = 1e9 truth dataset was used only for research diagnostics, never for fitting.

## 4. Billing disclosure: `QRV` and flopscope issue #269

**Every graded build from s31 (#336456, 16 Oct 2026) onward, including #337608, defaults to `QRV=1`** ([L3562](../estimator/estimator.py#L3562)). That setting builds the complement of the ordered shared basis with `fnp.linalg.qr(Ur, mode='complete')` ([L5381](../estimator/estimator.py#L5381)). The open flopscope issue #269 reports that `mode='complete'` is billed at the reduced-QR price, that is, without the cost of forming the extra columns of Q.

**What was measured.** Every QR call was counted during one full-size predict of the released file (in-process, flopscope 0.12.1, a random 1024x16 He-initialised network, 25 Oct 2026). Earlier comparisons of the flopscope wheels found the QR price identical in 0.12.1 and the grader's 0.12.3.

| Quantity | FLOPs per network | Share of the bill |
|---|---:|---:|
| total bill of that predict | 287,490,887,388 | 100% (0.1307 of 2^41) |
| all 111 QR calls | 2,626,233,438 | 0.91% |
| the one `QRV` call, `qr(448 x 224, mode='complete')`, as billed | 74,929,494 | **0.026%** |
| the same call priced as a LAPACK-style complete QR (4m²n − 2n³/3) | 172,337,834 | 0.060% |
| **under-billed amount if #269 is valid** | **97,408,340** | **0.034%** |
| the `QRV=0` alternative (square QR of `[Ur, I]`), as billed | 239,975,308 | 0.083% |

**What this means for the score.**
- **Size of the exposure.** The score is proportional to the FLOP fraction here, because the fraction (0.131) is above the 0.1 floor. Re-pricing the complete QR would raise the bill and the score of #337608 by about 0.034%. That is far below the grader's own run-to-run noise for identical predictions (about 0.07%).
- **Why it was kept.** Even at the honest price, `QRV=1` stays about 6.8e7 FLOPs cheaper than the `QRV=0` path, which flopscope prices correctly. The organisers can re-price it with the standard formula.
- **Earlier wording.** Earlier notes quoted "about 0.033%". That figure is this same under-billed amount, computed against an earlier build's bill.

The author knows of no other billing quirk used by the estimator.

## 5. LLM assistance

- **Models and tools.** Claude models through Claude Code for the whole phase: a main session plus many subagents. An OpenAI Codex session ran on 2-7 Oct 2026, locally only, and submitted nothing. Exact model identifiers were not recorded.
- **What the agents did.** They wrote essentially all of the code, ran the experiments and proposed most of the ideas. The LOT derivation was produced by a Claude agent on 1-2 Oct and checked numerically against Monte Carlo truth. The estimator header (line 7) states that the code was "developed with substantial assistance from Claude (Anthropic) and Codex coding agents".
- **What the author did.** Designed and steered the multi-agent search: set the objective, the evaluation rules and the compliance rules; proposed research directions from their own reading of the literature and the competition forum, alongside the agents' proposals; organised the search in rounds of parallel agents with a stated aim each, and, when agents reported a ceiling, kept the search going by suggesting new directions from their understanding of the problem; asked on 17 Oct for the generalisation check that became the untouched 300-network holdout; allocated the rented compute; made every submission, nomination and release decision; and reviewed the write-up and this repository.
- **Verification.** Separate verification agents checked the write-up's numbers against logged runs and grader telemetry; the discrepancies they found were corrected. The LOT identities were re-derived and checked numerically on toy networks (`checks/`).
- **This release.** The code copies, `verify.py`, the controls tooling and these documents were assembled by a Claude agent from the project's records on 25 Oct 2026 and updated on 28 Oct 2026.

## 6. History note

Until 6 October 2026 production was byte-identical to V29, with review copies that added the MIT notices and the LLM-lineage line. Every later build (s01 to s45, then S', S'', NLHZ, G2-G5) kept those notices; the released file carries them at lines 1-74. The project journal and the build configurations are the record of each build; the graded steps are in `docs/SUBMISSIONS.md`.

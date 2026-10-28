# Third-party notices

`estimator/estimator.py` (sha256 `e71018ffa112bc2bc4274798c850fe34170462a88b320522229d8b03c001271f`, graded as submission #337608, "G4") is a derivative work. This file lists every licensed source whose code or published constants it contains, reproduces each licence in full, and says which parts of the estimator derive from it. Line numbers refer to `estimator/estimator.py` as released. The file itself is never edited, so these line numbers stay valid.

The estimator header already reproduces three of the four notices below, word for word. The match was checked against the upstream licence files:

| Header lines | Notice | Matches upstream file |
|---|---|---|
| 1-7 | attribution paragraph and LLM-assistance statement | n/a (text of this work) |
| 9-29 | MIT, Copyright (c) 2026 504aldo | yes, `LICENSE` of 504aldo/whest-p2-cumulant-k3 |
| 31-52 | MIT, Copyright (c) 2026 504aldo, Copyright (c) 2026 LordProcciuto | yes, `LICENSE` of siming767-creator/whestbench-lordprocciuto |
| 54-74 | MIT, Copyright (c) 2026 Alignment Research Center | yes, `LICENSE` of alignment-research-center/mlp_cumulant_propagation |

The fourth source, a 14-number constant table taken from al07-alt/whest-p2-k31-two-term, is credited inline at estimator lines 252-253, but its notice is not in the header. It is reproduced here (section 4) to satisfy that licence for this repository.

The new contributions of this work are under the MIT licence in `LICENSE` (Copyright (c) 2026 Antoniu Gugu). See `docs/PROVENANCE.md` for the full inherited / ported / re-derived / new split.

---

## 1. 504aldo, whest-p2-cumulant-k3 (V29)

Source: https://github.com/504aldo/whest-p2-cumulant-k3 (`estimators/estimator_v29.py`). This is the base file. The first production estimator of this work was a byte-identical copy of V29, and every later build is a chain of patches on top of it.

**What in `estimator.py` derives from it**

- The overall architecture: the factored cumulant chain with per-source A/P legs, the thin M/Z legs and the `_dslices` contractions (8103-8564); `Estimator.setup` / `predict` / `_predict_core`; `_sym_product` (7694-7723); `_hub2` (7766-7805).
- The V-history in the module docstring, lines 85-236 (V16b to V29), kept as written.
- The memoryless kappa4 regeneration (V17) with its tables `LAM` (line 250) and `REF_R` (309), and the adaptive per-MLP lambda law (V25, `V25_BETA`, 310). `CORR_BETA` (249) is also V29's, but its online correction is off by default (`V17_NO_CORR='1'`, 307).
- D21 feedback thin legs (V18), final-layer trim (V19) and residual-time fusion (V20).
- The age-gated shared basis for old sources (V21, `V21_AGE_OLD`/`V21_R_OLD`, 4410-4411) and the nested second tier (V24, `V24_AGE_OLD2`/`V24_R_OLD2`, 4412-4413).
- The pooled batched Strassen-Winograd products (V26-V28): class `_Strassen` (1122-1494) and its knobs (1034-1042).
- The newborn and covariance slots riding the transport family (V29).
- The `if __name__ == '__main__'` footer (8565-8568). It refers to a helper module that is not part of this release.

```
MIT License

Copyright (c) 2026 504aldo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. Alignment Research Center, mlp_cumulant_propagation

Source: https://github.com/alignment-research-center/mlp_cumulant_propagation, the companion code of Wu, Lecomte, Winer, Robinson, Hilton and Christiano, arXiv:2605.05179. This code reaches the estimator through 504aldo's port (V29 is "a port of the reference kprop k_max=3 SIMPLE factored algorithm", docstring lines 205-207).

**What in `estimator.py` derives from it**

- The cumulant-propagation method itself: Hermite/Wick moment closure of the ReLU, factored kappa3, power cumulants.
- The term and coefficient tables `WICK_PAIRS`, `WICK_UNIFIED`, `TERM_SPECS` and `PK2K_TABLE` (lines 245-248), and the term programs built from them (`_term_prog` 804-836, `_build_wick_consts` 791-802).
- Two later additions follow ARC's published term programs. They were implemented in new code; no ARC code was copied for them:
  - `PATHB` (270-272): the K22 path birth of ARC's augmented (carried-K4) program, taken from the description in forum topic t/18249 sec. 3.4.
  - `MEHLER3` (758-759, use 6131-6148): the Gaussian Mehler m=3 pair terms of ARC's current term program (repository commit 93d091a).

```
MIT License

Copyright (c) 2026 Alignment Research Center

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 3. 504aldo and LordProcciuto, whestbench-lordprocciuto

Source: https://github.com/siming767-creator/whestbench-lordprocciuto, read at commit 3d038e9f (14 Oct 2026). The author approved porting from this repository on 14 Oct 2026. The changes adapt its ideas into this work's own files. Its `estimator.py` was never run as-is.

**What in `estimator.py` derives from it**

- **`FIJA2`** (291-295; code at 5163-5180 and 5361-5390): a port of their `X_FIJA` frozen join. After the first union the shared old-source basis is only transported. The entrant and the tier-2 mover are least-squares projected, and the basis is ordered once by the first joiner's Gram (their `X_FIJA_RC` ordering).
- **`FIJA2X`** (296-302: `FX_UEYE`, `FX_TRI`, `FX_FJW`, `FX_NESTW`, `FX_NESTS`, `FX_SGDEAD`): exact follow-up savings of the fixed basis, from their E28/E29 trims.
- **`OLD_HOT1`** (377-379): a port of their `X_VB_HOT`, old-tier legs formed only on the non-hot rows.
- **`AS_CDEAD_L`** (1081, default `'6-9/2.5'`): a per-layer schedule for the active-set dead-neuron threshold. It follows the idea of their `X_DEAD_TAU_L` per-layer thresholds. The implementation is new; no code was copied for it.
- **`QRV`** (3562; use 5379-5385): the complement of the ordered basis. The `QRV=0` path (5383-5385) is their `qr(concat[Ur, eye])` ordering step. The default `QRV=1` replaces it with `qr(Ur, mode='complete')`. See the billing disclosure in `docs/PROVENANCE.md`.

```
MIT License

Copyright (c) 2026 504aldo
Copyright (c) 2026 LordProcciuto

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 4. al07-alt, whest-p2-k31-two-term (kai_leal_gonzalez), constant table only

Source: https://github.com/al07-alt/whest-p2-k31-two-term, read at commit 55fac9cc (14 Oct 2026). The two-term (3,1) closure is described in forum topic t/18246.

**What in `estimator.py` derives from it**

- **`K31TWO_B1`** (line 255): the per-layer `b1` coefficients of the two-term closure. This 14-value table is identical to the `B1`/`JOINT` table in that repository (`src/v29cov1f.py`, `src/law_oos.py`).
- **The closure formula** in the comment at lines 252-253: wk431_ij = b1_l v_j C_ij + kappa3_j D21[j,i] / v_j.
- **The `K31TWO` code paths** (254-258 and uses at 5932-5969) are written in the file's own data structures. The `K31TWO_B2` level (1.2) and `K31TWO_B1S` scale are new refits.
- **The (2,1)-program orientation fix `EF_K31T21`** (683) follows t/18246 sec. 5. al07-alt measured the same fix independently.

Upstream licence, reproduced as published (single-line paragraphs as in the original):

```
MIT License

Copyright (c) 2026 the authors of this repository (ARC WhestBench Phase 2 write-up)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

---

## 5. Credited ideas (no code or constants copied)

These are credited in the estimator comments or in the write-up. They involve no copied code, so no licence notice is required. They are listed for completeness.

| Knob / place in `estimator.py` | Idea | Credit |
|---|---|---|
| `H18` (259) | per-row metric sum_i W_ai^2 in the kappa4 row estimate | Yinita/whest-l31-side (team Niko), credited inline as "idea from" |
| `MARG2` (693-695; use 6081-6092) | keep the k = 2 marginal Edgeworth pair correction only in the tail | Artyom Kuznetsov (AIcrowd handle Artyom_Mazur), forum topic 18249 sec. 3.5; re-derived from the text |
| `PATHB` (270) | K22 path birth | ARC augmented program as described in t/18249 sec. 3.4 (see section 2) |
| `ZF7_AB` (1511-1515) | Strassen-Winograd in an alternative basis | Karstadt and Schwartz (literature) |
| engine leaf sizes (`V26_STRASSEN_MIN`, `TA_*_MIN`) | small minimum leaf blocks for the Strassen recursion | forum post by ngmitam, as credited in the write-up; the code is new and the idea cannot be tied to a single line |

The non-linear HZ head (NLHZ) and its final-loss fine-tune were developed independently. Parallel work by other participants reached similar ideas: Hisernberg's LT corrector and end-to-end fine-tune, Susupro t/18254 sec. 4.2, and hyojun t/18255 sec. 6. None of their code was read. The write-up credits this parallel work.

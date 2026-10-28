# Graded single-variable controls

On 18 Oct 2026 three real builds of s43 (#336780) were graded, each with one mechanism switched off. They are controls for the write-up and were never nominated. **They ran on s43, not on the released G4 file (#337608).** This folder makes them exactly reproducible:

| File | What it is |
|---|---|
| `base_s43_336780/estimator.py` | the s43 file, byte-identical to graded #336780. sha256 `fa8118608705e45c0bad9feadf4414f56bdf84925b92910827f60e9d1ab7fe74`, 7,861 lines. It equals the sha256 in the manifest of the package uploaded for #336780 and the 12-character prefix in the build log. |
| `controls.json` | for each control: the knob, the exact line replacements on the base (old-line sha256 and new line), the expected sha256 of the graded file, the sha256 recorded in its upload-package manifest, and the graded numbers from the telemetry |
| `make_controls.py` | rebuilds the three graded files byte for byte from the base and checks every hash. Standard library only. Usage: `python3 -I controls/make_controls.py [--check-only] [--out DIR]` |
| `env/<name>_<id>.env` | the one knob setting of each control, as an environment variable, with a note on how it differs from the graded build |

`python3 -I verify.py` runs the same rebuild-and-hash check in memory.

## The three controls

| Control | ID | Change to s43 | Graded score | Raw (final-layer MSE) | FLOPs / B | vs s43: score / raw / FLOPs | Graded file sha256 |
|---|---|---|---:|---:|---:|---|---|
| s43 (base) | #336780 | none | 2.143761e-09 | 1.60690e-08 | 0.1334 | 1 / 1 / 1 | `fa8118608705...` |
| LOT off (s43cL) | #336912 | `PORTLOT` default `_ON` → `'0'` (line 308) and `_POOL_SPEC` re-recorded (line 1724) | 3.180332e-09 | 2.18140e-08 | 0.1458 | x1.4835 / x1.3575 / x1.0929 | `34eb9e196746...` |
| HZ off (s43cH) | #336911 | `HZ_ON` default `'1'` → `'0'` (line 318) | 2.474890e-09 | 1.97377e-08 | 0.1254 | x1.1545 / x1.2283 / x0.9399 | `f0afc8efab6f...` |
| all D21 rows exact (s43cD) | #336913 | `D21_CUT` default `'2.25'` → `'99'` (line 534) and `_POOL_SPEC` re-recorded (line 1724) | 2.315736e-09 | 1.60205e-08 | 0.1446 | x1.0802 / x0.9970 / x1.0834 | `a2c9b8b90e7d...` |

Some notes on the table:
- Numbers are from the grader telemetry of these submissions: 100/100 MLPs each; score, raw and FLOPs/B are the public-50 aggregates. Ratios are computed from unrounded values. Full hashes are in `controls.json` and `docs/SUBMISSIONS.md`.
- Line numbers refer to the s43 base file.
- The `_POOL_SPEC` line is the setup plan list. It was re-recorded with the knob set, so the graded control builds no plans lazily during a predict, and its FLOPs are exact for that configuration.

## How to read them

- **LOT off also removes the HZ correction.**
  - **The coupling.** In this code base the HZ correction picks its rows and builds its features inside the LOT routine:
    - `_hz_st` is filled only in `_lot_g4` (s43 lines 6969-6972; G4 lines 7499-7502);
    - LOT runs only when `lot = PL_ON and regen` (s43 line 4375; G4 line 4773);
    - `_hz_finish` returns without applying anything when a layer that should have HZ rows has none (s43 lines 6726-6727; G4 lines 7235-7236).
  - **Check.** `_hz_finish` was instrumented in one full-size predict of the released file with `PORTLOT=0` and the time guards off (25 Oct 2026). It returned False with zero selected rows and zero rider slabs, so neither HZ nor the NLHZ head was applied.
  - **Consequence.** #336912 measures LOT and HZ removed together. LOT's own share in the final stack is better read against the HZ-off control: #336912 / #336911 gives raw x1.1052, score x1.2850, FLOPs x1.1628.
  - **Why FLOPs rise.** Switching LOT off raises the bill, even though HZ's work is skipped. Without LOT the chain falls back to costlier fourth-order programs (a reading, not isolated).
- **HZ off** removes the learned correction alone. On s43, which has no NLHZ head, its value is raw x1.228 for 6.0% of the bill.
- **All D21 rows exact** shows the demand projection near its knee. Computing every row exactly buys 0.3% raw for 8.3% more FLOPs.
- **Grader noise.** Each control is one graded run on 50 public MLPs. For prediction-identical builds, the grader's run-to-run noise is about 0.07% (sd of the log score ratio).

## Do the knobs exist in the released file?

Yes. All three exist in `estimator/estimator.py` with the same definitions and defaults:

| Knob | G4 line | Definition |
|---|---|---|
| `PORTLOT` | 312 | `PL_ON = _os.environ.get('PORTLOT', _ON) == '1'` |
| `HZ_ON` | 322 | `HZ_ON = _os.environ.get('HZ_ON', '1') == '1'` |
| `D21_CUT` | 625 | `D21_CUT = float(_os.environ.get('D21_CUT', '2.25'))` |

Setting them on G4 removes the same mechanisms, but from a different stack:
- G4 adds the NLHZ head. It is removed by `HZ_ON=0` too, because it is applied inside the HZ finish.
- G4 adds MEHLER3, PATHB, PEEL, the TERMITE cuts and the guards, and refits `HZ_TH`, `HZ_K` and several closure scales (`docs/READING_GUIDE.md` section 8).
- Exported as environment variables, the knobs give the knob's predictions up to rounding. The billed FLOPs differ, because G4's recorded setup spec assumes the defaults and the missing plans are built lazily. For example, `PORTLOT=0` on one network billed 324.7 GF against 287.5 GF with the defaults.
- The graded numbers above belong to s43 only.

## Reproduce

```bash
python3 -I controls/make_controls.py            # writes controls/out/<name>_<id>/estimator.py and checks the hashes
python3 -I controls/make_controls.py --check-only
```

Expected output ends with `CONTROLS OK`. Each rebuilt file is the exact file that was graded. Running one requires flopscope and whestbench (see `requirements.txt`).

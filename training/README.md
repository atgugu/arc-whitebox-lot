# Training the learned parts of #337608 (HZ theta and the NLHZ head)

The estimator has two learned parts, both embedded in `estimator/estimator.py`:

- **`HZ_TH`** ([line 620](../estimator/estimator.py#L620)): per-layer linear weights of the HZ correction. There are 14 layers x 24 = 336 slots, 303 of them non-zero.
- **`_NLHZ_P`** ([line 538](../estimator/estimator.py#L538)): the non-linear head NLHZ, a 37 → 64 → 64 → 1 ReLU MLP shared by layers 2-15, with per-layer standardisation (8,554 numbers). In #337608 this head was fine-tuned directly on the final-layer MSE. That fine-tune is the only difference from #337411.

This folder holds the scripts that produced them, copied from the project with their original headers and comments. Only paths were changed (see "Edits made for this release" at the end). The data are not included (see "Data").

## Pipeline

| Step | Script(s) | Input | Output |
|---|---|---|---|
| 1. Dump build | `hzdump_patch.py`, then `hzpt_dump_patch.py`, then `hzx_dump_patch.py` | an estimator file | a patched estimator that writes one `.npz` of HZ inputs per predict when `HZ_DUMP=1` |
| 2. Dumps | `sdt_prof_full.py` | the patched estimator, a network index, the ground-truth file | `p_<net>.npz` (HZ inputs, the no-HZ prediction, rider slabs, the 15 pair reductions, pullback snapshots) |
| 3. Tables | `nl_extract2.py` (imports `nl_extract.py` and `hzr_pre.py` from this folder) | ground truth + dumps | per-row training tables `f2_k.npz` (5 folds x 80 networks) and `f3_kk.npz` (70-network chunks) |
| 4. HZ theta + base head | `nl_train3.py` | ground truth + the five `f2_k.npz` | `<out>_head.npz` (head "v5D" with theta inside) and `<out>_theta.npy` |
| (optional) | `infl.py` | ground truth + a table | per-row influence weights `inf_<k>.npz` for `nl_train3.py --inf 1`. Not used by the shipped recipe. |
| 5. Final-loss fine-tune | `hx_ft.py` | ground truth + the v5D head + `f2_0..4` (train) + `f3_00`, `f3_01` (early stopping) | `<out>_head.npz` (the G4 head; theta unchanged) |
| 6. Embedding | `mk_nlhz_patch5.py` (head) and `theta_bake6.py` (theta) | the head `.npz` / the theta `.npy` | a patch script that writes the `_NLHZ_P` line / an estimator with `HZ_TH` replaced (values rounded to 8 decimals) |

## Commands, in order

The ground-truth file is `$WHEST_ROOT/data/public_v2_full_gt.npz` (see Data).

```bash
# 1. dump build (any estimator of the G3/G4 chain; the original dumps were made on the S'' chain, #337231)
python hzdump_patch.py   ../estimator/estimator.py d1.py
python hzpt_dump_patch.py d1.py d2.py
python hzx_dump_patch.py  d2.py dump_estimator.py

# 2. one dump per network: about 80 s and 4.5 GB on one core.
#    NLHZ=0 reproduces the pre-head dumps; the time guards are switched off.
export WHEST_ROOT=/path/to/a/folder/with/data
for net in <400 training networks> <350 extra networks>; do
  env HZ_DUMP=1 HZ_DUMP_OUT=dumps/p_$net.npz NLHZ=0 OMP_NUM_THREADS=1 \
      HZ_TMAX=0 HZ_TMAX_L=0 HZ_TMAX_P=0 RGUARD=0 WG=0 \
      HZG_MAX_T=600 HZ_SETUP_T=600 HZ6_SETUP_T=600 PUBLIC_Z_PREWARM_T=60 PUBLIC_Z_SETUP_MAX_T=600 M2_PREALLOC_T=600 \
      python sdt_prof_full.py dump_estimator.py $net out/r_$net.json --noprof
done

# 3. training tables (one process per chunk)
python nl_extract2.py $WHEST_ROOT/data/public_v2_full_gt.npz f2_0.npz dumps/p_<80 nets of fold 0>.npz   # ... f2_1 .. f2_4
python nl_extract2.py $WHEST_ROOT/data/public_v2_full_gt.npz f3_00.npz dumps/p_<70 nets>.npz            # ... f3_01 .. f3_04

# 4. HZ theta (least-squares refit, step 1 of the script) and the v5D head (seed 1234).
#    The exact command line of the shipped v5D head was not archived. These flags are reconstructed from the head file's
#    metadata (st=1, xf=1, hid=64, dep=2, stcols 0,2,3,5,1), the second-version training recipe (--lw 1 --st 1 --hid 64 --dep 2
#    --ep 8 --seed 1234) and the v5D input group (--xfcols 11,12).
python nl_train3.py $WHEST_ROOT/data/public_v2_full_gt.npz v5D --train f2_0.npz f2_1.npz f2_2.npz f2_3.npz f2_4.npz \
       --lw 1 --st 1 --hid 64 --dep 2 --ep 8 --seed 1234 --xf 1 --xfcols 11,12

# 5. final-loss fine-tune (Adam lr 1e-4, 4 networks per step, early stopping on the 140 f3_00/f3_01 networks, seed 1234)
python hx_ft.py $WHEST_ROOT/data/public_v2_full_gt.npz v5D_head.npz hxc --train f2_0.npz f2_1.npz f2_2.npz f2_3.npz f2_4.npz \
       --val f3_00.npz f3_01.npz

# 6. embed: write the head line and the theta table
python mk_nlhz_patch5.py hxc_head.npz nlhz_patch_hxc.py 11,12   # 11,12 = the D21 input group of v5D
python nlhz_patch_hxc.py <pre-NLHZ estimator> <out estimator>   # the generated patch adds/updates _NLHZ_P
python theta_bake6.py <src estimator> <dst estimator> v5D_theta.npy
```

**Which networks.** Training used 400 public networks: indices 0-136, 337-399 and 500-699, split into five folds of 80 (`f2_0`..`f2_4`). The extra 350 networks are 137-236, 400-499 and 850-999 (`f3_00`..`f3_04`). Only `f3_00` and `f3_01` (140 networks) were used, as the early-stopping set of step 5. Networks 237-336 and 700-849 were kept out of every fit, because they served as screen and confirmation sets.

## What "reproduced" means here

- **Byte for byte, from the archived head.** Checked on 25 Oct 2026. Running `mk_nlhz_patch5.py` on the archived fine-tuned head (`head_hxc.npz`, 80,244 bytes, not shipped) regenerated the project's `nlhz_patch_hxc.py` exactly (the shipped script now differs from it in one docstring line of the generated patch). The `_NLHZ_P` literal of that patch equals line 538 of `estimator/estimator.py` character for character. Likewise, `HZ_TH` (line 620) equals `round(theta, 8)` of the archived theta exactly.
- **From the dumps.** Retraining from the dumps with the commands above gives the same recipe and seeds. PyTorch training on CPU is not bitwise reproducible across versions and thread counts, so expect a head that matches the archived one within seed noise. Seed-to-seed changes moved the cross-validated final-layer MSE by about 0.05-0.07%. The fine-tune's own 5-fold cross-validation gave x0.99820 against the v5D recipe retrained per fold (400 networks, z = -6.29).
- **The dump patches also apply to the released file.** All three were checked to apply cleanly to `estimator/estimator.py` and that the result parses. The original dumps were produced on the S'' build chain (#337231). The HZ inputs it dumps are those of the shipped chain, because the later patches up to G3 are exact cuts or the NLHZ head itself.

## Data (not included)

| Item | Size | How it was produced | Needed for |
|---|---|---|---|
| `public_v2_full_gt.npz` | 58 MB (60,496,449 bytes) | Official public Phase 2 dataset `hf://aicrowd/arc-whestbench-public-2026@v2-phase2`, split `full`, saved as arrays `seeds` (1000), `all_layer_means` (1000 x 16 x 1024, N = 1e9 samples) and `avg_variance`. Network weights are regenerated from the seeds (`numpy.random.SeedSequence(seed).spawn(3)[0]`, He initialisation). | steps 2-5 |
| HZ dumps of the 400 training networks (`p_<net>.npz`) | 813 MB (400 files) | step 2 on the S'' chain with the three dump patches | step 3 |
| HZ dumps of the 350 extra networks | 712 MB (350 files) | same | step 3 (`f3` tables) |
| training tables `f2_0..4.npz`, `f3_00..04.npz` | not archived | step 3 | steps 4-5 |
| `head_v5D.npz` (G3 head), `head_hxc.npz` (G4 head), theta `.npy` | 80 KB each, 3.2 KB | steps 4-5 | step 6 |

**Hardware and versions.**
- The dumps ran with the estimator's own environment (Python 3.10, flopscope 0.12.1, whestbench 0.16.1, numpy 2.2.6) on rented multi-core CPU machines and on a laptop, one network per core.
- Training ran on a laptop CPU (Intel i7-8750H) with Python 3.13.11, torch 2.11.0 and numpy 2.4.1.
- Step 4 takes minutes. Step 5 takes about ten minutes for the final head; its 5-fold cross-validation ran under a 60-minute cap.

## Edits made for this release

Code changes are limited to paths; each edited line carries a trailing `# release path edit` comment. In addition, internal agent and build names were removed from the docstrings of `hzpt_dump_patch.py`, `hzx_dump_patch.py`, `infl.py`, `hx_ft.py`, `mk_nlhz_patch5.py`, `nl_extract.py`, `nl_extract2.py` and `nl_train3.py`; no code changed.

| File | Line | Change |
|---|---|---|
| `sdt_prof_full.py` | 25 | `ROOT = Path(__file__).resolve().parents[3]` (the project root) became `ROOT = Path(os.environ.get('WHEST_ROOT', '.')).resolve()`. The ground truth is then read from `$WHEST_ROOT/data/public_v2_full_gt.npz`. |
| `nl_extract.py` | 10 | the import path `../hzrefit` became this folder (where `hzr_pre.py` now sits) |
| `nl_extract2.py` | 9 | the same change as `nl_extract.py` |
| `nl_train3.py` | 33 | an absolute path to a session scratch directory became `os.path.join(os.environ.get('INF_DIR', '.'), '')`. It is used only with `--inf 1`, which reads `inf_<k>.npz` from `$INF_DIR`. |

Apart from those docstrings, the other eight files are unchanged copies:
- `hzdump_patch.py`, `hzpt_dump_patch.py`, `hzx_dump_patch.py`;
- `hzr_pre.py`;
- `infl.py`;
- `hx_ft.py`;
- `mk_nlhz_patch5.py`;
- `theta_bake6.py`.

Their docstrings mention internal build names (for example the S'' build); `docs/SUBMISSIONS.md` maps them to submissions.

Not included: the 750-network variants and other research arms (`nl_train4.py`, `nl_train5.py`, the data-scaling heads behind G5), the cross-validation drivers, and the build scripts for the rented machines.

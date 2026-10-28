"""SDT research runner (never shipped) = prof2.py + per-MLP shape / plan bookkeeping.

usage: python sdt_prof.py <estimator.py> <mini_index>[,<mini_index>...] <out.json> [--noprof] [--profall]
Same scoring / rows as prof.py (rows: index, flops, cb, raw_mse, adjusted, ops, ...) so knobq-style comparisons work, plus per MLP:
  sdt      : the estimator's SDT_LOG (per layer: li, nr, ni, K, n3, nc), if present
  ycol     : est._young_column_shapes (li, nr, ni, nsl, hcols), yread: est._young_readout_shapes (li, m, nr, slots, cols)
  new_plans: ZFMM plan keys created during this predict (lazy = not prebuilt in setup / earlier predicts), n_plans, zviews, pviews
  op_hist  : billed FLOPs per op name
Profiling (per call-site line + per layer) on the last MLP only (or every MLP with --profall; the by_site table is then the sum).
"""
import hashlib, importlib.util, json, os, sys, time
from collections import defaultdict
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('M2_PREALLOC_T', '30'); os.environ.setdefault('PUBLIC_Z_SETUP_MAX_T', '40')
os.environ.setdefault('FMM_STRICT', '1'); os.environ.setdefault('HZ_STRICT', '1')
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
import flopscope._budget as fb
from whestbench.domain import MLP
from whestbench.sdk import SetupContext
flops.configure(symmetry_warnings=False)

ROOT = Path(os.environ.get('WHEST_ROOT', '.')).resolve()   # release path edit: was Path(__file__).resolve().parents[3] (the project root); the GT file is read from $WHEST_ROOT/data/
B = 2 ** 41
src = Path(sys.argv[1]).resolve(); idxs = [int(x) for x in sys.argv[2].split(',')]; out = Path(sys.argv[3])
PROF = '--noprof' not in sys.argv; PROFALL = '--profall' in sys.argv
spec = importlib.util.spec_from_file_location('prof_candidate', src)
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
SRC = str(src)
by_layer = defaultdict(int); by_site = defaultdict(lambda: [0, 0]); by_inner = defaultdict(lambda: [0, 0]); by_site_op = defaultdict(lambda: [0, 0])
orig_append = fb.BudgetContext._append_op_record
def patched(self, record):
    f = sys._getframe(1); outer = None; inner = None
    while f is not None:
        if f.f_code.co_filename == SRC:
            if inner is None: inner = f.f_code.co_name
            outer = (f.f_code.co_name, f.f_lineno)
            if f.f_code.co_name == '_predict_core': break
        f = f.f_back
    c = record.flop_cost
    if f is not None and f.f_code.co_name == '_predict_core':
        li_ = f.f_locals.get('li')
        if li_ is not None:
            by_layer[(int(li_), outer[1])] += c
    s = by_site[outer]; s[0] += c; s[1] += 1
    s = by_inner[inner]; s[0] += c; s[1] += 1
    s = by_site_op[(outer, record.op_name, str(record.shapes)[:80])]; s[0] += c; s[1] += 1
    return orig_append(self, record)

est = mod.Estimator(); t0 = time.perf_counter()
with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
    est.setup(SetupContext(width=1024, depth=16, flop_budget=B, seed=0, api_version='2.0', submission_dir=str(src.parent)))
setup_s = time.perf_counter() - t0
def plan_keys():
    z = getattr(getattr(est, '_smm', None), 'z', None)
    return set(z.plans.keys()) if z is not None else set()
setup_plans = plan_keys()
D = np.load(ROOT / 'data/public_v2_full_gt.npz', allow_pickle=False)
rows = []
for n, i in enumerate(idxs):
    seed = int(D['seeds'][i]); streams = np.random.SeedSequence(seed).spawn(3); rng = np.random.default_rng(streams[0])
    weights = [fnp.asarray((rng.standard_normal((1024, 1024)) * np.sqrt(2 / 1024)).astype(np.float32)) for _ in range(16)]
    mlp = MLP(width=1024, depth=16, weights=weights, seed=int(streams[2].generate_state(1)[0]))
    last = (n == len(idxs) - 1)
    if PROF and (last or PROFALL): fb.BudgetContext._append_op_record = patched
    before = plan_keys()
    t0 = time.perf_counter()
    with flops.BudgetContext(flop_budget=B, wall_time_limit_s=7200, quiet=True) as budget:
        vals = np.asarray(est.predict(mlp, B), dtype=np.float32)
    fb.BudgetContext._append_op_record = orig_append
    el = time.perf_counter() - t0
    truth = D['all_layer_means'][i].astype(float)
    raw = float(np.mean((vals[-1].astype(float) - truth[-1]) ** 2)); cost = budget.flops_used
    hist = defaultdict(int)
    for o in budget.op_log:
        hist[o.op_name] += o.flop_cost
    after = plan_keys(); z = getattr(getattr(est, '_smm', None), 'z', None)
    row = dict(index=i, flops=cost, cb=cost / B, raw_mse=raw, adjusted=raw * max(.1, cost / B), all_mse=float(np.mean((vals.astype(float) - truth) ** 2)),
               elapsed_s=el, residual_s=budget.residual_wall_time_s, ops=len(budget.op_log), finite=bool(np.isfinite(vals).all()),
               pred_sha=hashlib.sha256(vals.tobytes()).hexdigest(),
               sdt=[list(x) if isinstance(x, tuple) else x for x in getattr(mod, 'SDT_LOG', [])],
               ycol=[list(x) for x in getattr(est, '_young_column_shapes', [])], yread=[list(x) for x in getattr(est, '_young_readout_shapes', [])],
               yr=[list(x) if isinstance(x, tuple) else x for x in getattr(mod, 'YR_LOG', [])],
               orow=[list(x) for x in getattr(mod, 'OLD_ROW_LOG', [])],
               vd=[list(x) for x in (getattr(mod, 'VD_LOG', None) or [])], lotw=[list(x) for x in getattr(mod, 'LOTW_LOG', [])],
               t31=[list(x) for x in getattr(mod, 'T31_LOG', [])],
               new_plans=sorted([list(k) for k in after - before], key=str), n_plans=len(after),
               zviews=int(getattr(z, 'public_zview_count', -1)) if z is not None else -1, vc=len(z.vc) if z is not None else -1,
               pviews=len(est._pool.public_views) if getattr(est, '_pool', None) is not None else -1,
               op_hist=dict(sorted(hist.items(), key=lambda kv: -kv[1])[:12]))
    rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if k not in ('sdt', 'ycol', 'yread', 'yr', 'orow', 'vd', 'lotw', 't31', 'new_plans', 'op_hist')} | dict(n_new_plans=len(row['new_plans']))), flush=True)
    np.save(out.with_suffix(f'.pred{i}.npy'), vals)
    del weights, mlp
res = dict(source=str(src), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), setup_s=setup_s, n_setup_plans=len(setup_plans), rows=rows,
           env={k: v for k, v in os.environ.items() if k.startswith(('SDT', 'SD_', 'HUB_', 'D21', 'AS_', 'PAD', 'TA_', 'V2', 'TD_'))})
if PROF:
    res['by_site'] = sorted(([k[0], k[1], v[0], v[1]] for k, v in by_site.items() if k), key=lambda r: -r[2])
    res['by_inner'] = sorted(([k, v[0], v[1]] for k, v in by_inner.items()), key=lambda r: -r[1])
    res['by_site_op'] = sorted(([k[0][0] if k[0] else None, k[0][1] if k[0] else None, k[1], k[2], v[0], v[1]] for k, v in by_site_op.items()), key=lambda r: -r[4])[:400]
    res['by_layer'] = sorted([k[0], k[1], v] for k, v in by_layer.items())
out.write_text(json.dumps(res, indent=1))
print('mean adjusted', sum(r['adjusted'] for r in rows) / len(rows), 'mean raw', sum(r['raw_mse'] for r in rows) / len(rows), 'mean cb', sum(r['cb'] for r in rows) / len(rows))

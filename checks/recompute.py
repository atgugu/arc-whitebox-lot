#!/usr/bin/env python3
"""Recompute the paper's key statistics from the raw data shipped in checks/data/ (standard library only).

Part A, Table 1 (shape and level, separated). Reads the per-network logs of the lean runs in data/lot_runs/
(one JSON record per network and variant; final_mse is the final-layer MSE against the Monte Carlo means of
the public WhestBench Phase 2 networks 0-11, N = 1e9 samples) and forms paired sum-ratios against V29
(variant 'v29+f32') with normal-approximation 95% intervals, R +- 1.96 se (delta method).

Part B, graded knock-outs. Reads the official grader's per-MLP public scores of this work's submissions
(data/graded_public_mlps.csv; values as reported by the grader, 4 significant digits) and forms paired
sum-ratios over the 50 public MLPs, the standard error of ln(ratio), z = ln(ratio) / se, 95% intervals
ratio * exp(+-1.96 se), MLPs worse, the FLOPs ratio, and the agreement with the local 60-network ablations.

Both parts assert their results against the tables in figures/data/ that the paper and figures use.
Run: python3 checks/recompute.py   (about one second)
"""
import csv
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'figures', 'data')


def table(name):
    with open(os.path.join(DATA, name), encoding='utf-8') as f:
        return list(csv.DictReader(f))


def paired(Y, X):
    """Paired sum-ratio sum(Y)/sum(X) with the delta-method standard error of the ratio."""
    n = len(X)
    R = sum(Y) / sum(X)
    se = math.sqrt(sum((y - R * x) ** 2 for x, y in zip(X, Y)) / (n * (n - 1))) / (sum(X) / n)
    return R, se


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


# Part A: Table 1 ------------------------------------------------------------------------------------------
fm = {}
logs = sorted(glob.glob(os.path.join(HERE, 'data', 'lot_runs', '*.jsonl')))
for path in logs:
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                assert (d['variant'], d['i']) not in fm, (d['variant'], d['i'])
                fm[(d['variant'], d['i'])] = d['final_mse']

NETS = {'0-5': range(6), '0-11': range(12)}
print('Table 1: raw MSE against V29, lean runs with Monte Carlo truth (%d logs)' % len(logs))
print('  %-32s %-34s %-5s %8s  %-17s %s' % ('terms', 'level', 'nets', 'ratio', '95% CI', 'better'))
for r in table('shape_level_grid.csv'):
    parts = [v.strip() for v in r['variant'].split(' / ')]
    num, den = parts[0], (parts[1] if len(parts) > 1 else 'v29+f32')
    nets = NETS[r['nets']]
    Y, X = [fm[(num, i)] for i in nets], [fm[(den, i)] for i in nets]
    R, se = paired(Y, X)
    lo, hi = R - 1.96 * se, R + 1.96 * se
    better = '%d/%d' % (sum(y < x for x, y in zip(X, Y)), len(X))
    print('  %-32s %-34s %-5s %8.4f  [%.4f, %.4f] %s' % (r['terms'], r['level'], r['nets'], R, lo, hi, better))
    assert close(R, r['ratio_vs_v29'], 6e-5) and close(lo, r['ci_lo'], 6e-5) and close(hi, r['ci_hi'], 6e-5), r
    assert better == r['nets_better'], r

# Part B: graded knock-outs ---------------------------------------------------------------------------------
S = {}
with open(os.path.join(HERE, 'data', 'graded_public_mlps.csv'), encoding='utf-8') as f:
    for r in csv.DictReader(f):
        S.setdefault(int(r['submission']), {})[int(r['mlp_index'])] = r
assert all(sorted(v) == list(range(50)) for v in S.values())

g4 = sum(float(r['adjusted_final_layer_score']) for r in S[337608].values()) / 50
print('\n#337608: mean adjusted score over the public 50 = %.4e (graded 2.0236e-9)' % g4)
assert close(g4, 2.0236e-9, 1e-12)

local = {(r['control'], r['metric']): r for r in table('knockouts.csv')}
LOCAL = {336912: 'LOT off (also disables HZ)', 336911: 'HZ off', 336913: 'All D21 rows exact'}
print('\nGraded knock-outs, paired over the 50 public MLPs')
print('  %-7s %-7s %-27s %8s %8s %7s  %-17s %s' % ('control', 'base', 'metric', 'ratio', 'se(ln)', 'z', '95% CI', 'worse'))
for r in table('graded_controls_paired.csv'):
    a, b, metric = int(r['control_id']), int(r['base_id']), r['metric']
    Y = [float(S[a][i][metric]) for i in range(50)]
    X = [float(S[b][i][metric]) for i in range(50)]
    R, se = paired(Y, X)
    sl = se / R
    z = math.log(R) / sl
    worse = sum(y > x for x, y in zip(X, Y))
    print('  #%-6d #%-6d %-27s %8.5f %8.5f %7.2f  [%.4f, %.4f] %d/50'
          % (a, b, metric, R, sl, z, R * math.exp(-1.96 * sl), R * math.exp(1.96 * sl), worse))
    assert close(R, r['sum_ratio'], 6e-6) and close(sl, r['se'], 6e-6) and close(z, r['z'], 6e-3), (r, R, sl, z)
    assert worse == int(r['mlps_worse']), r
    if b == 336780 and metric == 'final_layer_mse':
        flops = (sum(float(S[a][i]['flops_used']) for i in range(50)) /
                 sum(float(S[b][i]['flops_used']) for i in range(50)))
        q = local[(LOCAL[a], 'raw')]
        lr, ls = float(q['local_ratio']), float(q['local_se_log'])
        zd = (math.log(R) - math.log(lr)) / math.hypot(sl, ls)
        print('  %15s FLOPs ratio %.4f; local 60-network raw ratio %.4f (se(ln) %.5f): grader - local z = %.2f'
              % ('', flops, lr, ls, zd))

print('\nRECOMPUTE OK')

#!/usr/bin/env python3
"""verify.py: check that this repository holds the exact estimator graded as submission #337608, and the exact graded controls.

    python3 -I verify.py                 # all static checks (standard library only; nothing is imported or run from the estimator)
    python3 -I verify.py --self-test     # also prove that a one-byte change is detected (in memory; no file is modified)
    python3 verify.py --smoke [WxD]      # optional: run setup + predict on a small random MLP (needs flopscope + whestbench;
                                         #   skipped with a message when they are not importable). Default shape 64x4.
    python3 -I verify.py --write-sums    # maintainer: rewrite the root SHA256SUMS over the authenticity-critical files

Static checks:
  1. estimator/estimator.py: sha256, size and line count equal estimator/manifest.json, and the sha256 equals the one recorded in the
     manifest of the package uploaded for #337608 (reproduced inside estimator/manifest.json).
  2. estimator/SHA256SUMS and the root SHA256SUMS: every listed file exists and matches, and the root file lists every
     authenticity-critical file (the estimator, its manifest, the controls base, controls.json, make_controls.py, controls/env/*).
  3. The estimator parses (ast.parse; never imported) and the anchors that docs/READING_GUIDE.md relies on are at their lines:
     the attribution and LLM-assistance header, the three MIT notices, _NLHZ_P (538), HZ_TH (620), QRV (3562), and 374 distinct
     environment knobs.
  4. Controls: the s43 base file (#336780) matches controls/controls.json, and each graded control (#336911, #336912, #336913) is
     rebuilt in memory from it and matches its recorded sha256 (same procedure as controls/make_controls.py).
The last line printed is VERIFY OK or VERIFY FAILED (exit code 0 or 1).

The root SHA256SUMS deliberately covers only the authenticity-critical files above. Documentation, licence files, paper/ and
figures/ can change without affecting verification.
"""
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
EST = os.path.join(ROOT, 'estimator', 'estimator.py')
CRITICAL_FIXED = ('estimator/estimator.py', 'estimator/manifest.json', 'controls/base_s43_336780/estimator.py',
                  'controls/controls.json', 'controls/make_controls.py')


def critical_files():
    """the authenticity-critical files covered by the root SHA256SUMS (repository-relative, '/' separators)"""
    env_dir = os.path.join(ROOT, 'controls', 'env')
    envs = sorted('controls/env/' + f for f in os.listdir(env_dir) if f.endswith('.env')) if os.path.isdir(env_dir) else []
    return list(CRITICAL_FIXED) + envs

# (1-based line, required prefix or substring, how) -- anchors used by docs/READING_GUIDE.md and THIRD_PARTY_NOTICES.md
ANCHORS = [
    (1, '# ARC White-Box Estimation Challenge 2026 (Phase 2) estimator, team antoniu_gugu9.', 'prefix'),
    (7, 'Claude (Anthropic) and Codex coding agents', 'contains'),
    (9, '# MIT License', 'prefix'), (11, '# Copyright (c) 2026 504aldo', 'prefix'),
    (31, '# MIT License', 'prefix'), (34, '# Copyright (c) 2026 LordProcciuto', 'prefix'),
    (54, '# MIT License', 'prefix'), (56, '# Copyright (c) 2026 Alignment Research Center', 'prefix'),
    (250, 'LAM = [', 'prefix'), (255, 'K31TWO_B1 = [', 'prefix'), (309, 'REF_R = [', 'prefix'),
    (312, "PL_ON = _os.environ.get('PORTLOT', _ON) == '1'", 'prefix'),
    (322, "HZ_ON = _os.environ.get('HZ_ON', '1') == '1'", 'prefix'),
    (537, "NLHZ = _os.environ.get('NLHZ', '1') == '1'", 'prefix'),
    (538, '_NLHZ_P = {', 'prefix'), (620, 'HZ_TH = [', 'prefix'),
    (625, "D21_CUT = float(_os.environ.get('D21_CUT', '2.25'))", 'prefix'),
    (3562, "QRV = _os.environ.get('QRV', '1') == '1'", 'prefix'),
    (5381, "Vr = fnp.linalg.qr(Ur, mode='complete')[0]", 'contains'),
]
N_KNOBS = 374


def sha256(b):
    return hashlib.sha256(b).hexdigest()


class Report:
    def __init__(self):
        self.ok = True

    def check(self, cond, msg):
        print(('ok   ' if cond else 'FAIL ') + msg)
        self.ok = self.ok and bool(cond)
        return bool(cond)


def read_sums(path):
    out = []
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'):
            continue
        h, name = line.split(None, 1)
        out.append((h, name.lstrip('*').strip()))
    return out


def check_estimator_bytes(b, man, rep, label='estimator/estimator.py'):
    """check 1 on given bytes (used by the self-test on a modified copy too)"""
    pkg = man['upload_package']['manifest']['files'][0]['sha256']
    h = sha256(b)
    c1 = rep.check(h == man['sha256'], '%s sha256 %s == manifest.json' % (label, h[:16] + '...'))
    c2 = rep.check(h == pkg, '%s sha256 == sha256 in the upload-package manifest of #%d' % (label, man['submission_id']))
    c3 = rep.check(len(b) == man['bytes'] and b.count(b'\n') == man['lines'],
                   '%s size %d bytes, %d lines == manifest.json' % (label, len(b), b.count(b'\n')))
    return c1 and c2 and c3


def check_sums(path, rep, required=()):
    base = os.path.dirname(path)
    rel = os.path.relpath(path, ROOT)
    if not os.path.exists(path):
        rep.check(False, '%s exists' % rel)
        return
    entries = read_sums(path)
    if required:
        listed = {name for _, name in entries}
        missing = [f for f in required if f not in listed]
        rep.check(not missing, '%s lists all %d authenticity-critical files%s' % (rel, len(required), '' if not missing else ': missing ' + ', '.join(missing)))
    bad = []
    for h, name in entries:
        p = os.path.join(base, name)
        if not os.path.exists(p):
            bad.append(name + ' (missing)')
        elif sha256(open(p, 'rb').read()) != h:
            bad.append(name)
    rep.check(not bad, '%s: %d files listed, %s' % (rel, len(entries), 'all match' if not bad else 'MISMATCH: ' + ', '.join(bad[:8])))


def check_static(b, rep):
    try:
        tree = ast.parse(b.decode('utf-8'))
    except SyntaxError as e:
        rep.check(False, 'estimator parses with ast.parse (%s)' % e)
        return
    rep.check(True, 'estimator parses with ast.parse (not imported)')
    lines = b.decode('utf-8').split('\n')
    bad = []
    for ln, s, how in ANCHORS:
        t = lines[ln - 1] if ln - 1 < len(lines) else ''
        if not (t.startswith(s) if how == 'prefix' else s in t):
            bad.append(str(ln))
    rep.check(not bad, 'header notices and code anchors at their documented lines (%d checked)%s'
              % (len(ANCHORS), '' if not bad else ': wrong at lines ' + ', '.join(bad)))
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('get', 'getenv') and node.args:
            a0 = node.args[0]
            src = ast.get_source_segment(b.decode('utf-8'), node.func) or ''
            if 'environ' in src and isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                keys.add(a0.value)
    rep.check(len(keys) == N_KNOBS, '%d distinct environment knobs (expected %d)' % (len(keys), N_KNOBS))


def rebuild_control(base_bytes, control):
    lines = base_bytes.decode('utf-8').split('\n')
    for r in control['replacements']:
        i = int(r['line']) - 1
        if not (0 <= i < len(lines)) or sha256(lines[i].encode('utf-8')) != r['old_sha256']:
            return None
        lines[i] = r['new']
    return '\n'.join(lines).encode('utf-8')


def check_controls(rep, base_override=None, spec_override=None):
    spec = spec_override or json.load(open(os.path.join(ROOT, 'controls', 'controls.json')))
    base = base_override if base_override is not None else open(os.path.join(ROOT, spec['base']['path']), 'rb').read()
    all_ok = rep.check(sha256(base) == spec['base']['sha256'],
                       'controls base s43 #%d sha256 %s...' % (spec['base']['submission_id'], sha256(base)[:16]))
    for c in spec['controls']:
        out = rebuild_control(base, c)
        good = out is not None and sha256(out) == c['expected_sha256']
        all_ok = rep.check(good, 'control %s #%d (%s) rebuilt from the base, sha256 %s'
                           % (c['name'], c['submission_id'], c['env_override'], (c['expected_sha256'][:16] + '...') if good else 'MISMATCH')) and all_ok
    return all_ok


def self_test(b, man):
    print('-- self-test: one-byte changes must be detected (in memory)')
    quiet = Report()
    m = bytearray(b)
    pos = len(m) // 2
    m[pos] ^= 0x01
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        est_detected = not check_estimator_bytes(bytes(m), man, quiet, label='modified copy')
    spec = json.load(open(os.path.join(ROOT, 'controls', 'controls.json')))
    base = bytearray(open(os.path.join(ROOT, spec['base']['path']), 'rb').read())
    base[len(base) // 3] ^= 0x01
    with contextlib.redirect_stdout(io.StringIO()):
        ctl_detected = not check_controls(Report(), base_override=bytes(base))
    spec2 = json.loads(json.dumps(spec))
    new = spec2['controls'][0]['replacements'][0]['new']
    spec2['controls'][0]['replacements'][0]['new'] = new[:-1] + chr(ord(new[-1]) ^ 1)
    with contextlib.redirect_stdout(io.StringIO()):
        rep_detected = not check_controls(Report(), spec_override=spec2)
    rep = Report()
    rep.check(est_detected, 'estimator with one flipped byte (offset %d) is rejected' % pos)
    rep.check(ctl_detected, 'control base with one flipped byte is rejected')
    rep.check(rep_detected, 'control patch with one flipped character is rejected')
    return rep.ok


SMOKE = r'''
import importlib.util, os, sys, time
try:
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from whestbench.domain import MLP
    from whestbench.sdk import SetupContext
except Exception as e:
    print('SMOKE SKIPPED: flopscope/whestbench not importable (%s: %s); see requirements.txt' % (type(e).__name__, e))
    sys.exit(3)
src, w, d = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
spec = importlib.util.spec_from_file_location('arc_whitebox_lot_estimator', src)
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
est = mod.Estimator()
t0 = time.perf_counter()
with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
    est.setup(SetupContext(width=w, depth=d, flop_budget=2 ** 41, seed=0, api_version='2.0', submission_dir=os.path.dirname(src)))
rng = np.random.default_rng(1234)
Ws = [(rng.standard_normal((w, w)) * np.sqrt(2.0 / w)).astype(np.float32) for _ in range(d)]
mlp = MLP(width=w, depth=d, weights=[fnp.asarray(x) for x in Ws], seed=7)
t1 = time.perf_counter()
with flops.BudgetContext(flop_budget=2 ** 41, quiet=True) as b:
    pred = np.asarray(est.predict(mlp, 2 ** 41), dtype=np.float64)
t2 = time.perf_counter()
ok = pred.shape == (d, w) and bool(np.isfinite(pred).all()) and b.flops_used > 0
print('smoke %dx%d: shape %s finite %s FLOPs %.4g (%.4f of 2^41) setup %.1fs predict %.1fs'
      % (w, d, pred.shape, bool(np.isfinite(pred).all()), b.flops_used, b.flops_used / 2 ** 41, t1 - t0, t2 - t1))
if w * d <= 4096:   # Monte-Carlo sanity reference with plain numpy (inputs N(0, I), h <- relu(h @ W), as in whestbench)
    n = 200000; x = rng.standard_normal((n, w)).astype(np.float32); means = []
    for W in Ws:
        x = np.maximum(x @ W, 0.0); means.append(x.mean(0))
    mc = np.array(means, dtype=np.float64)
    print('smoke %dx%d: final-layer MSE vs a %d-sample Monte-Carlo mean %.3g (MC noise floor ~%.3g); all-layer MSE %.3g'
          % (w, d, n, float(np.mean((pred[-1] - mc[-1]) ** 2)), float(np.mean(x.var(0)) / n), float(np.mean((pred - mc) ** 2))))
sys.exit(0 if ok else 1)
'''


def smoke(shape):
    w, d = (int(v) for v in shape.lower().split('x'))
    env = dict(os.environ)
    # deterministic run: no timing-dependent branch can change the result (see docs/READING_GUIDE.md, "Time guards")
    env.update({'OMP_NUM_THREADS': env.get('OMP_NUM_THREADS', '1'), 'OPENBLAS_NUM_THREADS': env.get('OPENBLAS_NUM_THREADS', '1'),
                'HZ_TMAX': '0', 'HZ_TMAX_L': '0', 'HZ_TMAX_P': '0', 'RGUARD': '0', 'WG': '0'})
    if (w, d) != (1024, 16):
        print('note: off-suite shape %dx%d runs the estimator\'s small-shape fallback path, not the scored path (docs/READING_GUIDE.md section 7)' % (w, d))
    else:
        print('note: full-size run, about 80 s and 4.5 GB of RAM per network on one core')
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, '-B', '-c', SMOKE, EST, str(w), str(d)], cwd=tmp, env=env)
    if r.returncode == 3:
        return None
    return r.returncode == 0


def write_sums():
    files = critical_files()
    with open(os.path.join(ROOT, 'SHA256SUMS'), 'w', encoding='utf-8') as fh:
        for rel in files:
            fh.write('%s  %s\n' % (sha256(open(os.path.join(ROOT, rel), 'rb').read()), rel))
    print('wrote SHA256SUMS (%d authenticity-critical files)' % len(files))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Verify the #337608 estimator and the graded controls.')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--smoke', nargs='?', const='64x4', default=None, metavar='WxD')
    ap.add_argument('--write-sums', action='store_true')
    a = ap.parse_args(argv)
    if a.write_sums:
        write_sums()
        return 0
    rep = Report()
    man = json.load(open(os.path.join(ROOT, 'estimator', 'manifest.json')))
    b = open(EST, 'rb').read()
    print('-- 1. estimator identity (submission #%d)' % man['submission_id'])
    check_estimator_bytes(b, man, rep)
    print('-- 2. checksum files')
    check_sums(os.path.join(ROOT, 'estimator', 'SHA256SUMS'), rep)
    check_sums(os.path.join(ROOT, 'SHA256SUMS'), rep, required=critical_files())
    print('-- 3. static structure (ast only)')
    check_static(b, rep)
    print('-- 4. graded controls')
    check_controls(rep)
    if a.self_test:
        rep.check(self_test(b, man), 'self-test: one-byte changes are detected')
    if a.smoke:
        print('-- smoke run (optional)')
        res = smoke(a.smoke)
        if res is None:
            print('skip smoke run (flopscope/whestbench not available); the static checks above are unaffected')
        else:
            rep.check(res, 'smoke run %s: setup + predict returned a finite (depth, width) array' % a.smoke)
    print('VERIFY OK' if rep.ok else 'VERIFY FAILED')
    return 0 if rep.ok else 1


if __name__ == '__main__':
    sys.exit(main())

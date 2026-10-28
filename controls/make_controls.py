#!/usr/bin/env python3
"""make_controls.py: rebuild the three graded single-variable controls of 18 Oct 2026 byte for byte from the s43 base file and check
every sha256 against controls.json.

  #336912 s43cL  PORTLOT=0    (LOT off)
  #336911 s43cH  HZ_ON=0      (HZ learned correction off)
  #336913 s43cD  D21_CUT=99   (every D21 row exact)

Each control is controls/base_s43_336780/estimator.py (graded as #336780) with the knob default changed in the source. For PORTLOT and
D21_CUT the recorded setup spec line (_POOL_SPEC) was also re-recorded with the knob set; controls.json stores those lines verbatim.

usage: python3 -I controls/make_controls.py [--out DIR] [--check-only] [--graded-dir DIR]
  --out DIR         where to write <name>_<id>/estimator.py (default: controls/out)
  --check-only      rebuild in memory and check the hashes; write nothing
  --graded-dir DIR  optional folder holding the original graded files as DIR/<name>_f/estimator.py (in the original project this is the
                    variants/ folder); a rebuilt file is then also compared byte for byte with the original when it is present

Standard library only. It never imports or runs an estimator. Exit code 0 when every check passes, 1 otherwise.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def rebuild(base_bytes, control):
    """Apply the recorded line replacements of one control to the base bytes; returns (bytes, list of problems)."""
    lines = base_bytes.decode('utf-8').split('\n')
    problems = []
    for r in control['replacements']:
        i = int(r['line']) - 1
        if not (0 <= i < len(lines)):
            problems.append('line %d out of range' % r['line'])
            continue
        if sha256(lines[i].encode('utf-8')) != r['old_sha256']:
            problems.append('line %d of the base does not match the recorded old line' % r['line'])
            continue
        lines[i] = r['new']
    return '\n'.join(lines).encode('utf-8'), problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--graded-dir', default=None)
    a = ap.parse_args(argv)
    spec = json.load(open(os.path.join(HERE, 'controls.json')))
    base_path = os.path.join(REPO, spec['base']['path'])
    base = open(base_path, 'rb').read()
    ok = True
    if sha256(base) != spec['base']['sha256']:
        print('FAIL base %s: sha256 %s != recorded %s' % (spec['base']['path'], sha256(base), spec['base']['sha256']))
        return 1
    print('ok   base s43 #%d  sha256 %s  (%d lines)' % (spec['base']['submission_id'], sha256(base), base.count(b'\n')))
    for c in spec['controls']:
        out, problems = rebuild(base, c)
        got = sha256(out)
        tag = '%s #%d (%s)' % (c['name'], c['submission_id'], c['env_override'])
        if problems or got != c['expected_sha256']:
            ok = False
            print('FAIL %s: %s' % (tag, '; '.join(problems) if problems else 'sha256 %s != expected %s' % (got, c['expected_sha256'])))
            continue
        msg = 'ok   %s  sha256 %s  (%d replaced lines)' % (tag, got, len(c['replacements']))
        if a.graded_dir:
            orig = os.path.join(a.graded_dir, c['name'] + '_f', 'estimator.py')
            if os.path.exists(orig):
                same = open(orig, 'rb').read() == out
                ok = ok and same
                msg += '  original graded file: %s' % ('identical' if same else 'DIFFERENT')
            else:
                msg += '  original graded file: not found'
        if not a.check_only:
            d = os.path.join(a.out, '%s_%d' % (c['name'], c['submission_id']))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, 'estimator.py'), 'wb') as fh:
                fh.write(out)
            msg += '  -> %s' % os.path.relpath(os.path.join(d, 'estimator.py'), REPO)
        print(msg)
    print('CONTROLS OK' if ok else 'CONTROLS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

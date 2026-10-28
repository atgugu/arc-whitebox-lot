"""HZ theta bake for widened tables (research, never shipped): <src estimator> -> <dst> with HZ_TH replaced by <theta.npy> (16 x N, N must equal
the row length of the src HZ_TH table, e.g. 19 for sdt7hx = 7 S-block + 7 base riders + 5 HX riders; NaN rows -> None).
usage: python theta_bake6.py <src.py> <dst.py> <theta.npy>"""
import ast, sys
from pathlib import Path
import numpy as np
src, dst, thf = sys.argv[1:4]
s = Path(src).read_text()
i = s.index('HZ_TH = ['); j = s.index('\n', i)
old = ast.literal_eval(s[i + len('HZ_TH = '):j]); N = max(len(r) for r in old if r is not None)
TH = np.load(thf); assert TH.shape == (16, N), (TH.shape, N)
tab = [None if not np.isfinite(TH[l]).all() else [round(float(x), 8) for x in TH[l]] for l in range(16)]
s = s[:i] + 'HZ_TH = ' + repr(tab) + s[j:]
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(s)
print('wrote', dst, 'theta', thf, 'width', N, 'rows', [l for l in range(16) if tab[l] is not None])

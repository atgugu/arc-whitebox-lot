"""hzx_dump_patch.py <in.py> <out.py> (20 Oct 2026; own code; research only, never shipped).
Goes AFTER hzdump_patch.py (and optionally hzpt_dump_patch.py).  With HZ_DUMP=1, at every rider layer that has the h6 pair matrices
(D21, K31 = wk431, C_off, wk4m), it computes 15 extra per-row pair reductions in host numpy (research only) and writes them into the
HZ dump as x_<name> (L, n) arrays in natural neuron order, so a richer NLHZ head can be tested offline.
HZ_DUMP=0: one Python test per layer, no flopscope call, op stream bit-identical."""
import sys

src = open(sys.argv[1]).read()
n_ed = 0


def sub(old, new, tag):
    global src, n_ed
    c = src.count(old)
    assert c == 1, (tag, c)
    src = src.replace(old, new, 1)
    n_ed += 1


assert 'def _hzr_write(snap, last):' in src, 'apply after hzdump_patch.py'
K = "def _hzr_write(snap, last):\n"
sub(K, '''HZX_NAMES = ('c2v', 'cw1', 'cw2', 'cd3', 'msum', 'mw1', 'mdiag', 'ksum', 'kw1', 'kw2', 'ddiag', 'dw2', 'dtw1', 'c2w2', 'c3w3')


def _hzx_collect(est, li, D, Kp, C, M, w1, w2, w3, D3, var):
    # hzx_dump_patch (research): per-row pair reductions of the layer's pair matrices, host numpy, alive order of layer li
    import numpy as _hn
    a = lambda x: None if x is None else _hn.array(_hn.asarray(x), dtype=_hn.float64)
    D, Kp, C, M, w1, w2, w3, D3, var = (a(x) for x in (D, Kp, C, M, w1, w2, w3, D3, var))
    w1, w2, w3 = (None if x is None else x.reshape(-1) for x in (w1, w2, w3))
    nr = var.shape[0]
    sq = lambda X: X if (X is not None and X.ndim == 2 and X.shape == (nr, nr)) else None
    D, Kp, C, M = sq(D), sq(Kp), sq(C), sq(M)
    w1, w2, w3 = (x if (x is not None and x.shape == (nr,)) else _hn.zeros(nr) for x in (w1, w2, w3))
    D3 = D3 if (D3 is not None and D3.shape == (nr,)) else None
    z = _hn.zeros(nr)
    sg = _hn.sqrt(var)
    f = {}
    f['c2v'] = (C * C).sum(1) / var if C is not None else z
    f['cw1'] = C @ w1 if C is not None else z
    f['cw2'] = C @ w2 if C is not None else z
    f['cd3'] = C @ (D3 / var) if C is not None and D3 is not None else z
    f['msum'] = M.sum(1) if M is not None else z
    f['mw1'] = M @ w1 if M is not None else z
    f['mdiag'] = _hn.diag(M).copy() if M is not None else z
    f['ksum'] = Kp.sum(1) if Kp is not None else z
    f['kw1'] = Kp @ w1 if Kp is not None else z
    f['kw2'] = Kp @ w2 if Kp is not None else z
    f['ddiag'] = _hn.diag(D) / (var * sg) if D is not None else z
    f['dw2'] = (D @ w2) * sg if D is not None else z
    f['dtw1'] = D.T @ w1 if D is not None else z
    f['c2w2'] = (C * C) @ w2 if C is not None else z
    f['c3w3'] = (C * C * C) @ w3 if C is not None else z
    if getattr(est, '_hzx', None) is None:
        est._hzx = {}
    est._hzx[li] = f


def _hzx_pack(est, hz_map, n, L):
    import numpy as _hn
    o = {('x_' + k): _hn.zeros((L, n)) for k in HZX_NAMES}
    for li, f in (getattr(est, '_hzx', None) or {}).items():
        inv, nr = hz_map[li]
        ordr = _hn.arange(n) if inv is None else _hn.argsort(_hn.asarray(inv))
        nat = ordr[n - int(nr):]
        for k in HZX_NAMES:
            v = f[k]
            o['x_' + k][li, nat[:len(v)]] = v
    est._hzx = {}
    return o


''' + K, 'helpers')

a1 = "                            fnp.subtract(fnp.matmul(_dv, _D), fnp.diag(_D) * _dv, out=self._hz_slab('h6_k31b_s', li, nr))\n"
sub(a1, a1 + "                            if HZ_DUMP:\n                                _hzx_collect(self, li, _D, _K, _C, _M, _w1, _w2, _w3, D3, var)   # hzx_dump_patch\n", 'collect')

a2 = "            _hzr_write(self._hzr_snap, rows[nb + 1])   # hzdump_patch\n"
sub(a2, "            if self._hzr_snap is not None:\n                self._hzr_snap.update(_hzx_pack(self, hz_map, H['n'], nb + 2))   # hzx_dump_patch\n" + a2, 'write')
open(sys.argv[2], 'w').write(src)
print('hzx_dump_patch: %d edits' % n_ed)

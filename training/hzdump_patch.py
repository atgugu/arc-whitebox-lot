"""hzdump_patch.py <in.py> <out.py>: HZ refit dumps from the PRODUCTION build itself (HZREFIT lane, 20 Oct 2026; own code; research only,
never shipped).

Why: the xs-family dump twins (xs12) no longer accept the S'' patch chain (12 of 14 patches fail their anchors), so the dump hook reads
the production HZ inputs where production itself consumes them: in Estimator._hz_finish, after the S-block features (F), the chain /
induced pullback sums (acc_c / acc_i) and the rider slabs are final and before the transport to the last layer.
Knobs (module level):
  HZ_DUMP      '1' = write one .npz per predict to HZ_DUMP_OUT (default '0' = parent op stream, bit-identical; one bool test per predict).
  HZ_DUMP_OUT  output path (.npz).
The hook only copies arrays to plain numpy (np.array(np.asarray(x))) and never writes into estimator state, so predictions are unchanged
when it is on.  Contents (natural neuron order, layers 0..15, NaN where absent):
  rows_nohz (16, n)  the prediction before the HZ correction (rows 0..14 are never touched by HZ);  pred_hz15 (n,) the final row after it
  slabs (layer x n): mu var same proj star D3 g4 port h6_d21_d3 h6_d21_e h6_d21_s h6_d21_w1 h6_k31b_s out (rider output) Phi (hz_Phi)
  selected rows: rank (0 = most sensitive, -1 = not selected), chd / ind (acc_c / acc_i), f_* (the 8 F rows actually used)
  alive (layer x n), hasrid (layer), tail_pos (n,) = 1 where a dropped last-layer neuron receives the tail transport (mu > 0).
"""
import sys

src = open(sys.argv[1]).read()
n_ed = 0


def sub(old, new, tag):
    global src, n_ed
    c = src.count(old)
    assert c == 1, (tag, c)
    src = src.replace(old, new, 1)
    n_ed += 1


HOOK = r'''HZ_DEBUG = None
HZ_DUMP = _os.environ.get('HZ_DUMP', '0') == '1'   # hzdump_patch (HZREFIT research): write the HZ inputs of each predict to HZ_DUMP_OUT
HZ_DUMP_OUT = _os.environ.get('HZ_DUMP_OUT', '')


def _hzr_snapshot(est, rows, hz, F, rid):
    """hzdump_patch: plain-numpy copies of the production HZ inputs in natural neuron order (research only; reads, never writes)"""
    import numpy as _hn
    hz_W, hz_c1, hz_c2, hz_nu, hz_Phi, hz_map, hz_tail = hz
    H = est._hz
    st = est._hz_st
    nb = H['nb']
    kb = H['kb']
    off = H['off']
    n = H['n']
    L = nb + 2
    cp = lambda x: _hn.array(_hn.asarray(x), dtype=_hn.float64)
    sl = ('mu', 'var', 'same', 'proj', 'star', 'D3', 'g4', 'port') + tuple(H6_SLABS) + ('out',)
    fn = ('f_same', 'f_proj', 'f_star', 'f_D3', 'f_var', 'f_g4', 'f_mu', 'f_port')
    o = {k: _hn.full((L, n), _hn.nan) for k in sl + fn + ('Phi', 'chd', 'ind')}
    rank = _hn.full((L, n), -1, _hn.int32)
    alive = _hn.zeros((L, n), bool)
    hasrid = _hn.zeros(L, bool)
    tail_pos = _hn.zeros(n)
    Fn = cp(F)
    acc_c = cp(H['acc_c'])
    acc_i = cp(H['acc_i'])
    nats = {}
    for li in range(2, nb + 2):
        inv, nr = hz_map[li]
        nr = int(nr)
        ordr = _hn.arange(n) if inv is None else _hn.argsort(_hn.asarray(inv))
        nat = ordr[n - nr:]
        nats[li] = (ordr, nr)
        alive[li, nat] = True
        o['Phi'][li, nat] = cp(hz_Phi[li])
        if li in est._hz_rid:
            hasrid[li] = True
            nrr = int(est._hz_rid[li])
            for k in sl:
                o[k][li, nat[:nrr]] = cp(est._hz_slab(k, li, nrr))
        s = st.get(li)
        if s is not None:
            b = nb + 1 - li
            K = int(kb[b])
            order = _hn.asarray(s[0]).astype(int)
            nrs = int(s[1])
            R = nat[order[nrs - K:]]
            rank[li, R] = _hn.arange(K)[::-1]
            o['chd'][li, R] = acc_c[off[b]:off[b + 1]]
            o['ind'][li, R] = acc_i[off[b]:off[b + 1]]
            for q, k in enumerate(fn):
                o[k][li, R] = Fn[q, off[b]:off[b + 1]]
    if hz_tail is not None:
        ordr, nr = nats[nb + 1]
        dead = ordr[:n - nr]
        tail_pos[dead] = (cp(hz_tail[1]) > 0).astype(float)
    o.update(rank=rank, alive=alive, hasrid=hasrid, tail_pos=tail_pos, rows_nohz=_hn.stack([cp(r) for r in rows]), rid=_hn.array(bool(rid)),
             KL=_hn.array(list(H['KL'])), kb=_hn.array(list(kb)))
    return o


def _hzr_write(snap, last):
    import numpy as _hn
    if snap is None or not HZ_DUMP_OUT:
        return
    snap['pred_hz15'] = _hn.array(_hn.asarray(last), dtype=_hn.float64)
    _hn.savez_compressed(HZ_DUMP_OUT, **snap)
'''
sub("HZ_DEBUG = None\n", HOOK, 'knobs')

a2 = "        # transport of the mean changes to the final layer (alive order of every layer; Gaussian mean response Phi(alpha) W)\n"
sub(a2, "        self._hzr_snap = _hzr_snapshot(self, rows, hz, F, rid) if HZ_DUMP else None   # hzdump_patch\n" + a2, 'snapshot')

a3 = "        rows[nb + 1] = rows[nb + 1] + v\n        return True\n"
sub(a3, "        rows[nb + 1] = rows[nb + 1] + v\n        if HZ_DUMP:\n            _hzr_write(self._hzr_snap, rows[nb + 1])   # hzdump_patch\n        return True\n", 'write')

open(sys.argv[2], 'w').write(src)
print('hzdump_patch: %d edits' % n_ed)

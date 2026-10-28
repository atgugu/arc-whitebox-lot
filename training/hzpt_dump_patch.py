"""hzpt_dump_patch.py <in.py> <out.py> (20 Oct 2026; own code; research only, never shipped).
Goes AFTER hzdump_patch.py.  With HZ_DUMP=1 it records the HZW pullback accumulators acc_c / acc_i
(block order, length R) at the TOP of every pullback step j (= the sums of the steps > j, i.e. what a pullback truncated before step j
would have produced) plus once after the loop (= the production sums), and writes them into the HZ dump as
  pt_j (S,) step index (-1 = after the loop), pt_c (S, R), pt_i (S, R).
HZ_DUMP=0: one Python test per step / predict, no flopscope call, op stream bit-identical.  Map block entries to neurons offline with the
dump's 'rank' and 'kb' (block b = layer nb + 1 - b holds rank K-1-q at entry off[b] + q)."""
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
sub(K, '''def _hzpt_cp(x):
    import numpy as _hn
    return _hn.array(_hn.asarray(x), dtype=_hn.float64)


def _hzpt_pack(lst):
    import numpy as _hn
    return dict(pt_j=_hn.array([j for j, _, _ in lst]), pt_c=_hn.stack([c for _, c, _ in lst]), pt_i=_hn.stack([i for _, _, i in lst]))


''' + K, 'helpers')

a1 = "        H = self._hz\n        E = H['E']\n        E6 = H.get('E6')\n        lev = E.L\n"
sub(a1, a1 + "        self._hzpt_s = [] if HZ_DUMP else None   # hzpt_dump_patch\n", 'init')

a2 = "        for j in range(nb, -1, -1):\n            if HZ_TMAX_P > 0 and _rb_time.perf_counter() - self._rb_t0 > HZ_TMAX_P:\n"
sub(a2, "        for j in range(nb, -1, -1):\n            if HZ_DUMP:\n                self._hzpt_s.append((j, _hzpt_cp(H['acc_c']), _hzpt_cp(H['acc_i'])))   # hzpt_dump_patch\n"
    "            if HZ_TMAX_P > 0 and _rb_time.perf_counter() - self._rb_t0 > HZ_TMAX_P:\n", 'top')

a3 = "            cnt = cnt_out\n            cw = cwn\n\n    def _hzg_build(self):"
sub(a3, "            cnt = cnt_out\n            cw = cwn\n        if HZ_DUMP:\n            self._hzpt_s.append((-1, _hzpt_cp(H['acc_c']), _hzpt_cp(H['acc_i'])))   # hzpt_dump_patch\n\n    def _hzg_build(self):", 'end')

a4 = "            _hzr_write(self._hzr_snap, rows[nb + 1])   # hzdump_patch\n"
sub(a4, "            if getattr(self, '_hzpt_s', None) and self._hzr_snap is not None:\n                self._hzr_snap.update(_hzpt_pack(self._hzpt_s))   # hzpt_dump_patch\n" + a4, 'write')
open(sys.argv[2], 'w').write(src)
print('hzpt_dump_patch: %d edits' % n_ed)

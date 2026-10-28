"""nl_extract2.py (20 Oct 2026; research only, own code): nl_extract.py tables + (a) the 15 hzx pair reductions as extra row
inputs (XF, layer x row x 15) and (b) truncated pullback accumulators for HZPT: for every J0 in J0S the S-block columns 5/6
(12 chd c4, 12 ind c4) recomputed from the accumulator snapshot "steps >= J0 only" (A56, J0 x layer x row x 2), plus the pullback
row-count / width table for the FLOP model (PB: steps nb..0 x (rows, contraction width, output width)).
usage: python nl_extract2.py <gt.npz> <out.npz> <p_dump.npz> ...   (dumps from f4bddp with HZ_DUMP=1: hzdump + hzpt_dump + hzx_dump)
"""
import sys, time, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # release path edit: hzr_pre.py now sits next to this file (was ../hzrefit)
from hzr_pre import layer_cols
from nl_extract import RAW, SEL, LS

XN = ('c2v', 'cw1', 'cw2', 'cd3', 'msum', 'mw1', 'mdiag', 'ksum', 'kw1', 'kw2', 'ddiag', 'dw2', 'dtw1', 'c2w2', 'c3w3')
J0S = (0, 1, 2, 3, 4, 5, 6, 7)
S2P = 0.3989422804014327

if __name__ == '__main__':
    gtf, outf = sys.argv[1:3]
    files = sorted(sys.argv[3:], key=lambda f: int(f.rsplit('_', 1)[1][:-4]))
    ids = [int(f.rsplit('_', 1)[1][:-4]) for f in files]
    GT = np.load(gtf); N, n, NLy = len(files), 1024, len(LS)
    A_ = np.zeros((N, NLy, n, 24), np.float32); U_ = np.zeros((N, NLy, n), np.float32); Y_ = np.zeros((N, NLy, n), np.float32)
    LIVE = np.zeros((N, NLy, n), bool); R_ = np.zeros((N, NLy, n, len(RAW) + 2), np.float32); ST = np.zeros((N, NLy, n, 6), np.float32)
    XF = np.zeros((N, NLy, n, len(XN)), np.float32); A56 = np.zeros((N, len(J0S), NLy, n, 2), np.float32)
    PHI = np.zeros((N, 16, n), np.float32); ERR15 = np.zeros((N, n)); PRED15 = np.zeros((N, n)); TRU15 = np.zeros((N, n))
    PB = np.zeros((N, 16, 3)); t0 = time.time()
    for ni, (f, i) in enumerate(zip(files, ids)):
        z = np.load(f)
        seed = int(GT['seeds'][i]); rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[0])
        W = [(rng.standard_normal((n, n)) * np.sqrt(2 / n)).astype(np.float32).astype(np.float64) for _ in range(16)]
        truth = GT['all_layer_means'][i].astype(float); pred = z['rows_nohz']; err = pred - truth
        PhiT = {t: np.where(z['alive'][t], np.nan_to_num(z['Phi'][t]), 0.0) for t in range(2, 16)}
        PhiT[15] = np.where(z['alive'][15], PhiT[15], z['tail_pos'])
        for t in range(2, 16):
            PHI[ni, t] = PhiT[t]
        # accumulator snapshots -> natural (layer, row) arrays per J0
        kb = [int(x) for x in z['kb']]; nb = len(kb); off = np.r_[0, np.cumsum(kb)].astype(int)
        pj = list(z['pt_j']); pc, pi = z['pt_c'], z['pt_i']
        snap = {}
        for J0 in J0S:
            key = -1 if J0 == 0 else J0 - 1          # top of step J0-1 = sums of the steps nb..J0
            k = pj.index(key)
            snap[J0] = (pc[k], pi[k])
        for li, l in enumerate(LS):
            A, u, live, sel = layer_cols(z, l)
            loc = err[l] - PhiT[l] * (err[l - 1] @ W[l])
            A_[ni, li] = A; U_[ni, li] = u; Y_[ni, li] = np.where(live, -loc + u, 0.0); LIVE[ni, li] = live
            g = lambda kk: np.where(live, np.nan_to_num(z[kk][l]), 0.0)
            vs = np.where(live, z['var'][l], 1.0)
            R_[ni, li, :, :len(RAW)] = np.stack([g(kk) for kk in RAW], 1)
            R_[ni, li, :, len(RAW)] = g('mu') / np.sqrt(vs); R_[ni, li, :, len(RAW) + 1] = PhiT[l]
            XF[ni, li] = np.stack([np.where(live, np.nan_to_num(z['x_' + kk][l]), 0.0) for kk in XN], 1)
            # carried state (same formulas as nl_extract.py)
            mu1 = np.nan_to_num(z['mu'][l - 1]); v1 = np.where(np.isfinite(z['var'][l - 1]) & (z['var'][l - 1] > 0), z['var'][l - 1], 1.0)
            a1 = mu1 / np.sqrt(v1); p1 = np.exp(-0.5 * a1 * a1) / np.sqrt(2 * np.pi)
            d31 = np.nan_to_num(z['D3'][l - 1]); g41 = np.nan_to_num(z['g4'][l - 1])
            al1 = np.isfinite(z['mu'][l - 1]).astype(float) * (z['alive'][l - 1] if l - 1 >= 2 else 1.0)
            c2 = p1 / np.sqrt(v1) * al1; W2 = W[l] * W[l]
            ST[ni, li] = np.stack([(c2 * v1) @ W2, (c2 * d31 / v1) @ W[l], (c2 * c2 * v1 * v1) @ W2, (p1 * a1 * g41 / (v1 * v1) * al1) @ W2,
                                   np.zeros(n), (c2 * d31 / v1) @ W2], 1)
            # truncated S-block columns 5/6 (selected rows): entry of rank r in block b = nb + 1 - l is off[b] + K - 1 - r
            if sel.any():
                b = nb + 1 - l; K = kb[b]
                vr = np.where(sel, z['f_var'][l], 1.0); sg = np.sqrt(vr); al = np.where(sel, np.nan_to_num(z['f_mu'][l]), 0.0) / sg; a2 = al * al
                c4 = np.where(sel, (a2 - 1.0) * np.exp(-0.5 * a2) / (vr * sg) * (S2P / 24.0), 0.0)
                rk = z['rank'][l]; ent = np.where(sel, off[b] + K - 1 - np.maximum(rk, 0), 0)
                for q, J0 in enumerate(J0S):
                    cc, ii = snap[J0]
                    A56[ni, q, li, :, 0] = np.where(sel, 12.0 * cc[ent] * c4, 0.0)
                    A56[ni, q, li, :, 1] = np.where(sel, 12.0 * ii[ent] * c4, 0.0)
                # consistency: J0 = 0 must reproduce the production columns 5/6
                d = np.abs(A56[ni, 0, li, :, :] - A[:, 5:7]).max() / (np.abs(A[:, 5:7]).max() + 1e-300)
                assert d < 1e-4, ('J0=0 mismatch', i, l, d)
        # pullback cost table: step j multiplies rows(j) x cw_j by W_j (alive(j) x alive(j-1)); rows(j) = selected rows of layers > j
        alive_n = [n if t < 2 else int(z['alive'][t].sum()) for t in range(16)]
        for j in range(nb, -1, -1):
            rows_j = int(sum(kb[bb] for bb in range(nb) if (nb + 1 - bb) > j))
            PB[ni, j] = (rows_j, alive_n[j] if j >= 1 else n, alive_n[j - 1] if j >= 1 else n)
        ERR15[ni] = err[15]; PRED15[ni] = z['pred_hz15']; TRU15[ni] = truth[15]
        if ni % 20 == 0:
            print(f'{ni}/{N} net {i} {time.time() - t0:.0f}s', flush=True)
    np.savez(outf, ids=np.array(ids), A=A_, U=U_, Y=Y_, LIVE=LIVE, R=R_, ST=ST, XF=XF, A56=A56, J0S=np.array(J0S), PB=PB, PHI=PHI,
             ERR15=ERR15, PRED15=PRED15, TRU15=TRU15, xnames=np.array(XN))
    print('wrote', outf, time.time() - t0)

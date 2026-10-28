"""nl_extract.py (20 Oct 2026; research only, own code): per-layer, per-row training table for a NON-LINEAR HZ head, from the
HZREFIT production dumps (hzdump_patch.py on f4c1nb2).  Re-uses hzr_pre.layer_cols (the 24 production theta columns) so the linear
baseline is exactly the production form.  Target per live row of layer l (2..15): y = -loc + u, loc = err_l - Phi_l o (err_{l-1} @ W_l)
(the born mean error; same definition as hzr_pre.py).  Extra raw features + simple carried-state features (layer l-1 quantities pushed
through W_l and W_l o W_l).
usage: python nl_extract.py <gt.npz> <out.npz> <dump.npz> ...
"""
import sys, time, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # release path edit: hzr_pre.py now sits next to this file (was ../hzrefit)
from hzr_pre import layer_cols

RAW = ['mu', 'var', 'same', 'proj', 'star', 'D3', 'g4', 'port', 'h6_d21_d3', 'h6_d21_e', 'h6_d21_s', 'h6_d21_w1', 'h6_k31b_s']
SEL = ['f_same', 'f_proj', 'f_star', 'f_D3', 'f_g4', 'f_port', 'chd', 'ind']
LS = list(range(2, 16))

if __name__ == '__main__':
    gtf, outf = sys.argv[1:3]
    files = sorted(sys.argv[3:], key=lambda f: int(f.rsplit('_', 1)[1][:-4]))
    ids = [int(f.rsplit('_', 1)[1][:-4]) for f in files]
    GT = np.load(gtf)
    N, n = len(files), 1024
    nS = 6
    A_ = np.zeros((N, len(LS), n, 24), np.float32)
    U_ = np.zeros((N, len(LS), n), np.float32)
    Y_ = np.zeros((N, len(LS), n), np.float32)
    LIVE = np.zeros((N, len(LS), n), bool)
    SELM = np.zeros((N, len(LS), n), bool)
    R_ = np.zeros((N, len(LS), n, len(RAW) + 2), np.float32)   # raw + alpha + Phi
    S_ = np.zeros((N, len(LS), n, len(SEL)), np.float32)
    ST = np.zeros((N, len(LS), n, nS), np.float32)
    PHI = np.zeros((N, 16, n), np.float32)
    ERR15 = np.zeros((N, n)); PRED15 = np.zeros((N, n)); TRU15 = np.zeros((N, n)); OUT = np.zeros((N, len(LS), n), np.float32)
    t0 = time.time()
    for ni, (f, i) in enumerate(zip(files, ids)):
        z = np.load(f)
        seed = int(GT['seeds'][i]); rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[0])
        W = [(rng.standard_normal((n, n)) * np.sqrt(2 / n)).astype(np.float32).astype(np.float64) for _ in range(16)]
        truth = GT['all_layer_means'][i].astype(float); pred = z['rows_nohz']; err = pred - truth
        PhiT = {}
        for t in range(2, 16):
            PhiT[t] = np.where(z['alive'][t], np.nan_to_num(z['Phi'][t]), 0.0)
        PhiT[15] = np.where(z['alive'][15], PhiT[15], z['tail_pos'])
        for t in range(2, 16):
            PHI[ni, t] = PhiT[t]
        prev = None
        for li, l in enumerate(LS):
            A, u, live, sel = layer_cols(z, l)
            loc = err[l] - PhiT[l] * (err[l - 1] @ W[l])
            y = np.where(live, -loc + u, 0.0)
            A_[ni, li] = A; U_[ni, li] = u; Y_[ni, li] = y; LIVE[ni, li] = live; SELM[ni, li] = sel
            g = lambda k: np.where(live, np.nan_to_num(z[k][l]), 0.0)
            vs = np.where(live, z['var'][l], 1.0)
            al = g('mu') / np.sqrt(vs)
            R_[ni, li, :, :len(RAW)] = np.stack([g(k) for k in RAW], 1)
            R_[ni, li, :, len(RAW)] = al
            R_[ni, li, :, len(RAW) + 1] = PhiT[l]
            S_[ni, li] = np.stack([np.where(sel, np.nan_to_num(z[k][l]), 0.0) for k in SEL], 1)
            OUT[ni, li] = np.where(live, np.nan_to_num(z['out'][l]), 0.0)
            # carried state (layer l-1 quantities pushed through W_l and W_l o W_l); all available in production at finish time
            mu1 = np.nan_to_num(z['mu'][l - 1]); v1 = np.where(np.isfinite(z['var'][l - 1]) & (z['var'][l - 1] > 0), z['var'][l - 1], 1.0)
            a1 = mu1 / np.sqrt(v1); p1 = np.exp(-0.5 * a1 * a1) / np.sqrt(2 * np.pi)
            d31 = np.nan_to_num(z['D3'][l - 1]); g41 = np.nan_to_num(z['g4'][l - 1])
            al1 = np.isfinite(z['mu'][l - 1]).astype(float) * (z['alive'][l - 1] if l - 1 >= 2 else 1.0)
            c2 = p1 / np.sqrt(v1) * al1                     # c2 of layer l-1 (kink density)
            W2 = W[l] * W[l]
            s0 = (c2 * v1) @ W2                             # kink-weighted variance mass into neuron i
            s1 = (c2 * d31 / v1) @ W[l]                     # skew push (signed)
            s2 = (c2 * c2 * v1 * v1) @ W2                   # second-order mass
            s3 = (p1 * a1 * g41 / (v1 * v1) * al1) @ W2
            s4 = (A_[ni, li - 1] @ np.ones(24) if li > 0 else np.zeros(n)) @ W2   # previous layer column-sum (unsigned push)
            s5 = (c2 * d31 / v1) @ W2
            ST[ni, li] = np.stack([s0, s1, s2, s3, s4, s5], 1)
        ERR15[ni] = err[15]; PRED15[ni] = z['pred_hz15']; TRU15[ni] = truth[15]
        if ni % 10 == 0:
            print(f'{ni}/{N} net {i} {time.time() - t0:.0f}s', flush=True)
    np.savez(outf, ids=np.array(ids), A=A_, U=U_, Y=Y_, LIVE=LIVE, SEL=SELM, R=R_, S=S_, ST=ST, PHI=PHI, ERR15=ERR15, PRED15=PRED15,
             TRU15=TRU15, OUT=OUT, raw_names=np.array(RAW + ['alpha', 'Phi']), sel_names=np.array(SEL), KL=np.array(z['KL']))
    print('wrote', outf, time.time() - t0)

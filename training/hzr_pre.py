"""hzr_pre.py (HZREFIT, 20 Oct 2026; research only): hzfx_pre6-style precompute from PRODUCTION dumps (hzdump_patch.py), with the 24 theta
columns built exactly as production's _hz_finish consumes them (S-block on the selected rows from the F rows actually used; riders on every
live row from the rider slabs, with production's kf constants), the production hz_Phi for the transport, and the production tail rule
(dropped last-layer neurons with mu > 0 receive the transport).  Output keys are hzfx_eval6.Pre-compatible (names, ids, T, Tu, err15, G, h,
ss, nn) plus per-net reconstruction diagnostics with a given theta table.
usage: python hzr_pre.py <gt.npz> <out.npz> <theta_old.npy or estimator.py> <dump.npz> [<dump.npz> ...]   (file names end in _<net>.npz)
"""
import ast, sys, time
import numpy as np

S2P = 0.3989422804014327
KF = (-S2P / 6.0, -S2P / 6.0, S2P / 2.0, S2P / 2.0, S2P / 24.0, S2P / 24.0, S2P / 24.0, S2P / 2.0, S2P / 2.0, S2P / 2.0, S2P / 24.0,
      S2P / 24.0, S2P / 24.0, S2P / 24.0, S2P / 24.0, S2P / 24.0, -S2P / 6.0)
NAMES = ['s_same', 's_proj', 's_star', 's_d3', 's_g4', 's_chd', 's_ind',
         'r_d3', 'r_d3a', 'r_v', 'r_va', 'r_p', 'r_pa', 'r_g4',
         'r_c2same', 'r_c2proj', 'r_c2star', 'r_c4d3', 'r_c4same',
         'q4_d21_w1', 'q4_d21_e', 'q4_d21_s', 'q4_d21_d3', 'q3_k31b_s']
LS = list(range(2, 16))


def load_theta(p):
    if p.endswith('.npy'):
        return np.load(p)
    s = open(p).read(); i = s.index('HZ_TH = ['); j = s.index('\n', i)
    tab = ast.literal_eval(s[i + len('HZ_TH = '):j])
    return np.array([[np.nan] * 24 if r is None else r for r in tab], float)


def layer_cols(z, l):
    """(1024, 24) column matrix in theta order, u (kappa4 used on the selected rows), live mask, sel mask"""
    n = z['mu'].shape[1]
    live = z['alive'][l] & bool(z['hasrid'][l]) & np.isfinite(z['mu'][l]) & np.isfinite(z['var'][l])
    sel = (z['rank'][l] >= 0) & live
    A = np.zeros((n, 24)); u = np.zeros(n)
    g = lambda k: np.where(live, np.nan_to_num(z[k][l]), 0.0)
    vs = np.where(live, z['var'][l], 1.0); mu = g('mu')
    sgs = np.sqrt(vs); als = mu / sgs; a2s = als * als; ps = np.exp(-0.5 * a2s); vsg = vs * sgs; pq = (a2s - 1.0) * ps / vsg; aps = als * ps
    D3, port, g4, same, proj, star = g('D3'), g('port'), g('g4'), g('same'), g('proj'), g('star')
    mult = [aps / vs * D3, aps / vs * D3 * a2s, ps * sgs, ps * sgs * a2s, pq * port, pq * port * a2s, pq * g4,
            ps / vsg * same, ps / vsg * proj, ps / vsg * star, pq * D3 * D3 / vs, pq * same,
            pq * g('h6_d21_w1'), pq * g('h6_d21_e'), pq * g('h6_d21_s'), pq * g('h6_d21_d3'), aps / vsg * g('h6_k31b_s')]
    for f in range(17):
        A[:, 7 + f] = np.where(live, mult[f] * KF[f], 0.0)
    if sel.any():
        h = lambda k: np.where(sel, np.nan_to_num(z[k][l]), 0.0)
        vr = np.where(sel, z['f_var'][l], 1.0); sg = np.sqrt(vr); al = h('f_mu') / sg; a2 = al * al
        c4 = np.where(sel, (a2 - 1.0) * np.exp(-0.5 * a2) / (vr * sg) * (S2P / 24.0), 0.0)
        S = [12 * h('f_same'), 12 * h('f_proj'), 12 * h('f_star'), (4.0 / 3.0) * h('f_D3') ** 2 / vr, h('f_g4'), 12 * h('chd'), 12 * h('ind')]
        for f in range(7):
            A[:, f] = S[f] * c4
        u = c4 * h('f_port')
    return A, u, live, sel


if __name__ == '__main__':
    gtf, outf, thf = sys.argv[1:4]; files = sys.argv[4:]
    files = sorted(files, key=lambda f: int(f.rsplit('_', 1)[1][:-4]))
    ids = [int(f.rsplit('_', 1)[1][:-4]) for f in files]
    GT = np.load(gtf); TH = load_theta(thf); F = 24; n_net = len(files); t0 = time.time()
    T_ = np.zeros((n_net, len(LS), 1024, F)); Tu = np.zeros((n_net, len(LS), 1024)); ERR15 = np.zeros((n_net, 1024))
    GRAM = np.zeros((n_net, len(LS), F, F)); HV = np.zeros((n_net, len(LS), F)); SS = np.zeros((n_net, len(LS), F)); NN = np.zeros((n_net, len(LS)), int)
    diag = []
    for ni, (f, i) in enumerate(zip(files, ids)):
        z = np.load(f)
        seed = int(GT['seeds'][i]); rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[0])
        W = [(rng.standard_normal((1024, 1024)) * np.sqrt(2 / 1024)).astype(np.float32).astype(np.float64) for _ in range(16)]
        truth = GT['all_layer_means'][i].astype(float); pred = z['rows_nohz']; err = pred - truth
        PhiT = {}
        for t in range(2, 16):
            PhiT[t] = np.where(z['alive'][t], np.nan_to_num(z['Phi'][t]), 0.0)
        PhiT[15] = np.where(z['alive'][15], PhiT[15], z['tail_pos'])
        corr_rec = np.zeros(1024); rid_err = 0.0; rid_nrm = 0.0
        for li, l in enumerate(LS):
            A, u, live, sel = layer_cols(z, l)
            loc = err[l] - PhiT[l] * (err[l - 1] @ W[l])
            y = np.where(live, -loc + u, 0.0)
            GRAM[ni, li] = A.T @ A; HV[ni, li] = A.T @ y; SS[ni, li] = (A ** 2).sum(0); NN[ni, li] = int(live.sum())
            th = np.nan_to_num(TH[l]) if np.isfinite(TH[l]).all() else np.zeros(F)
            if not (z['KL'][l] > 0):
                th = th.copy(); th[:7] = 0.0
            # rider output check vs the production slab 'out' (live rows)
            ro = A[:, 7:] @ th[7:]; po = np.where(live, np.nan_to_num(z['out'][l]), 0.0)
            rid_err += float(((ro - po) ** 2).sum()); rid_nrm += float((po ** 2).sum())
            M = np.concatenate([A, u[:, None]], 1)
            for t in range(l + 1, 16):
                M = PhiT[t][:, None] * (W[t].T @ M)
            T_[ni, li] = M[:, :F]; Tu[ni, li] = M[:, F]
            corr_rec += M[:, :F] @ th - M[:, F]
        ERR15[ni] = err[15]
        corr_prod = z['pred_hz15'] - pred[15]
        raw0 = float(np.mean(err[15] ** 2)); rawp = float(np.mean((z['pred_hz15'] - truth[15]) ** 2)); rawr = float(np.mean((err[15] + corr_rec) ** 2))
        cd = float(np.linalg.norm(corr_rec - corr_prod) / max(np.linalg.norm(corr_prod), 1e-300))
        diag.append((i, raw0, rawp, rawr, cd, (rid_err / max(rid_nrm, 1e-300)) ** 0.5))
        print(f'{i} noHZ {raw0:.4e} prod {rawp:.4e} rec {rawr:.4e} rec/prod x{rawr / rawp:.5f} corr-reldiff {cd:.2e} rider-reldiff {diag[-1][5]:.2e} {time.time() - t0:.0f}s', flush=True)
    np.savez(outf, names=np.array(NAMES), ids=np.array(ids), T=T_, Tu=Tu, err15=ERR15, G=GRAM, h=HV, ss=SS, nn=NN, diag=np.array(diag), KL=np.array(z['KL']))
    d = np.array(diag)
    print(f'wrote {outf} nets {n_net}: sum rec/prod x{d[:, 3].sum() / d[:, 2].sum():.5f} max|rec/prod-1| {np.abs(d[:, 3] / d[:, 2] - 1).max():.4f} '
          f'HZ factor prod x{d[:, 2].sum() / d[:, 1].sum():.4f} max corr-reldiff {d[:, 4].max():.2e}')

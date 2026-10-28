"""nl_train3.py (research only): nl_train.py on nl_extract2 tables, plus --j0 (HZPT: S-block columns 5/6 from the pullback truncated before step J0) and --xf (15 hzx pair reductions as extra head inputs). Base recipe otherwise identical:
  1. per-layer linear refit of the 24 production HZ columns (RMS-scaled LS, = HZREFIT procedure) on the training nets -> theta (16 x 24);
  2. ONE shared MLP (rider-slab inputs only: 17 rider columns + 13 slab-derived features, per-layer robust standardization
     (median, 16-84 percentile range), clip +-8, layer one-hot) on the per-layer-normalized residual of (1);
  3. optional held-out evaluation (final-layer MSE after the production Phi-W transport) on other nets;
  4. writes theta (hzr_eval format, NaN rows 0-1) + head parameters for mk_nlhz_patch.py.
usage: python nl_train.py <gt.npz> <out_prefix> --train a.npz [b.npz ...] [--test c.npz ...] [--hid 64] [--dep 2] [--ep 8] [--lw 0] [--st 0]
       [--seed 1234] [--maxtrain N]
"""
import sys, time, os, argparse
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('gt'); ap.add_argument('out')
ap.add_argument('--train', nargs='+', required=True); ap.add_argument('--test', nargs='*', default=[])
ap.add_argument('--hid', type=int, default=64); ap.add_argument('--dep', type=int, default=2); ap.add_argument('--ep', type=int, default=8)
ap.add_argument('--lw', type=int, default=0); ap.add_argument('--st', type=int, default=0); ap.add_argument('--seed', type=int, default=1234)
ap.add_argument('--maxtrain', type=int, default=0)
ap.add_argument('--stcols', default='0,2,3,5,1')
ap.add_argument('--j0', type=int, default=0)
ap.add_argument('--xf', type=int, default=0)
ap.add_argument('--wd', type=float, default=1e-5); ap.add_argument('--drop', type=float, default=0.0); ap.add_argument('--lr', type=float, default=3e-3)
ap.add_argument('--huber', type=float, default=0.0); ap.add_argument('--inf', type=int, default=0); ap.add_argument('--xfcols', default=''); ap.add_argument('--bs', type=int, default=4096)   # SHIP order of the carried-state inputs: W o W group (s0 s2 s3 s5) then W (s1)
a = ap.parse_args()
STC = [int(x) for x in a.stcols.split(',')]
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
GT = np.load(a.gt)
LWV = np.array([0.002, 0.004, 0.009, 0.015, 0.024, 0.035, 0.044, 0.058, 0.077, 0.095, 0.117, 0.145, 0.173, 0.201])
KEYS = ('ids', 'A', 'U', 'Y', 'LIVE', 'R', 'PHI', 'ERR15', 'PRED15', 'TRU15') + (('ST',) if a.st else ()) + (('XF',) if a.xf else ())


SPB2 = os.path.join(os.environ.get('INF_DIR', '.'), '')   # release path edit: was an absolute scratch directory; --inf reads inf_<k>.npz from $INF_DIR


def load(files, cap=0):
    D = {k: [] for k in KEYS}
    if a.inf:
        D['INF'] = []
    for f in files:
        z = np.load(f)
        if a.inf:
            zi = np.load(SPB2 + 'inf_' + f.rsplit('_', 1)[1]); assert (zi['ids'] == z['ids']).all(); D['INF'].append(zi['INF'])
        q = list(z['J0S']).index(a.j0)
        for k in KEYS:
            if k == 'INF':
                continue
            if k == 'A':
                A = z['A'].copy(); A[..., 5:7] = z['A56'][:, q]; D[k].append(A)
            else:
                D[k].append(z[k])
    D = {k: np.concatenate(v) for k, v in D.items()}
    if cap:
        D = {k: v[:cap] for k, v in D.items()}
    return D


def feats(D, i, li, st):
    """30 inputs in the SHIP order: 17 rider columns (production multipliers x kf) then 13 slab-derived features"""
    r = D['R'][i, li].astype(np.float64)
    var = np.maximum(r[:, 1], 1e-30); sg = np.sqrt(var); vsg = var * sg; v2 = var * var
    cols = [D['A'][i, li][:, 7:].astype(np.float64),
            np.stack([r[:, 13], np.log(var), r[:, 5] / vsg, r[:, 6] / v2, r[:, 7] / v2, r[:, 2] / v2, r[:, 3] / v2, r[:, 4] / v2,
                      r[:, 8] / vsg, r[:, 9] / vsg, r[:, 10] / vsg, r[:, 11] / vsg, r[:, 12] / v2], 1)]
    if st:
        cols.append(D['ST'][i, li][:, STC].astype(np.float64))
    if a.xf:
        x = D['XF'][i, li].astype(np.float64)
        xn = x / np.stack([var, sg, sg, sg, v2, sg, var, var * var, var * sg, var * sg, np.ones_like(var), var, sg, var, var * sg], 1)
        cols.append(xn[:, [int(c) for c in a.xfcols.split(',')]] if a.xfcols else xn)
    return np.concatenate(cols, 1)


def lin_fit(Xa, y):
    sc = np.sqrt(np.maximum((Xa ** 2).sum(0), 0)); keep = sc > 1e-300; th = np.zeros(Xa.shape[1])
    if keep.any():
        th[keep] = np.linalg.lstsq(Xa[:, keep] / sc[keep], y, rcond=1e-12)[0] / sc[keep]
    return th


class Head(torch.nn.Module):
    def __init__(self, fin, nl, hid, dep):
        super().__init__()
        L = [torch.nn.Linear(fin + nl, hid), torch.nn.ReLU()] + ([torch.nn.Dropout(a.drop)] if a.drop > 0 else [])
        for _ in range(dep - 1):
            L += [torch.nn.Linear(hid, hid), torch.nn.ReLU()] + ([torch.nn.Dropout(a.drop)] if a.drop > 0 else [])
        L += [torch.nn.Linear(hid, 1)]
        self.f = torch.nn.Sequential(*L)

    def forward(self, x):
        return self.f(x).squeeze(-1)


t0 = time.time()
TR = load(a.train, a.maxtrain)
N, NL, n = TR['A'].shape[0], TR['A'].shape[1], TR['A'].shape[2]
ths, meds, sds, rss = [], [], [], []
nrow = [int(TR['LIVE'][:, li].sum()) for li in range(NL)]; tot_r = sum(nrow)
F0 = feats(TR, 0, 0, a.st).shape[1]
Xt = torch.zeros((tot_r, F0 + NL), dtype=torch.float32); Yt = torch.zeros(tot_r, dtype=torch.float32); Wt = torch.ones(tot_r, dtype=torch.float32)
r0 = 0
for li in range(NL):
    Xa = np.concatenate([TR['A'][i, li][TR['LIVE'][i, li]] for i in range(N)]).astype(np.float64)
    y = np.concatenate([TR['Y'][i, li][TR['LIVE'][i, li]] for i in range(N)]).astype(np.float64)
    th = lin_fit(Xa, y); res = y - Xa @ th; rs = res.std() + 1e-300
    Xf = np.concatenate([feats(TR, i, li, a.st)[TR['LIVE'][i, li]] for i in range(N)])
    med = np.median(Xf, 0); sd = np.percentile(Xf, 84, 0) - np.percentile(Xf, 16, 0) + 1e-30
    Xn = np.clip((Xf - med) / sd, -8, 8).astype(np.float32); del Xf
    m_ = len(Xn); Xt[r0:r0 + m_, :F0] = torch.from_numpy(Xn); Xt[r0:r0 + m_, F0 + li] = 1.0
    Yt[r0:r0 + m_] = torch.from_numpy((res / rs).astype(np.float32)); Wt[r0:r0 + m_] = float(LWV[li]) if a.lw else 1.0
    if a.inf:
        iw = np.concatenate([TR['INF'][i, li][TR['LIVE'][i, li]] for i in range(N)]).astype(np.float64) * rs * rs
        Wt[r0:r0 + m_] = torch.from_numpy(iw.astype(np.float32))
    r0 += m_
    del Xn, Xa, y, res
    ths.append(th); meds.append(med); sds.append(sd); rss.append(rs)
assert r0 == tot_r
Wt = Wt / Wt.mean()
F = F0
torch.manual_seed(a.seed)
net = Head(F, NL, a.hid, a.dep)
opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=a.wd)
nb, bs = len(Xt), a.bs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.ep * ((nb + bs - 1) // bs))
for ep in range(a.ep):
    perm = torch.randperm(nb); tot = 0.0
    for b0 in range(0, nb, bs):
        idx = perm[b0:b0 + bs]
        opt.zero_grad(); dd = net(Xt[idx]) - Yt[idx]; loss = (Wt[idx] * (dd ** 2 if a.huber <= 0 else torch.nn.functional.huber_loss(dd, torch.zeros_like(dd), reduction='none', delta=a.huber))).mean(); loss.backward(); opt.step(); sched.step()
        tot += float(loss) * len(idx)
    print(f'ep {ep} train mse(norm) {tot / nb:.4f} {time.time() - t0:.0f}s', flush=True)
net.eval()
theta = np.full((16, 24), np.nan); theta[2:] = np.array(ths)
lin_layers = [m for m in net.f if isinstance(m, torch.nn.Linear)]
P = {f'W{j}': m.weight.detach().numpy().astype(np.float64) for j, m in enumerate(lin_layers)}
P.update({f'b{j}': m.bias.detach().numpy().astype(np.float64) for j, m in enumerate(lin_layers)})
np.savez(a.out + '_head.npz', theta=theta, med=np.array(meds), sd=np.array(sds), rs=np.array(rss), F=F, NL=NL, hid=a.hid, dep=a.dep,
         st=a.st, stcols=np.array(STC), xf=a.xf, j0=a.j0, train_ids=TR['ids'], **P)
np.save(a.out + '_theta.npy', theta)
print('wrote', a.out + '_head.npz', a.out + '_theta.npy', 'train nets', N, f'{time.time() - t0:.0f}s')

if a.test:
    TE = load(a.test)
    assert not set(TE['ids'].tolist()) & set(TR['ids'].tolist()), 'test nets overlap training nets'
    Nt = TE['A'].shape[0]
    mse = {'lin': np.zeros(Nt), 'nl': np.zeros(Nt)}
    with torch.no_grad():
        for i in range(Nt):
            rng = np.random.default_rng(np.random.SeedSequence(int(GT['seeds'][int(TE['ids'][i])])).spawn(3)[0])
            W = [(rng.standard_normal((n, n)) * np.sqrt(2 / n)).astype(np.float32).astype(np.float64) for _ in range(16)]
            e = np.zeros((n, 2))
            for li in range(NL):
                lv = TE['LIVE'][i, li]
                cl = np.where(lv, TE['A'][i, li] @ ths[li], 0.0)
                Xn = np.clip((feats(TE, i, li, a.st) - meds[li]) / sds[li], -8, 8).astype(np.float32)
                oh = np.zeros((n, NL), np.float32); oh[:, li] = 1
                p = net(torch.from_numpy(np.concatenate([Xn, oh], 1))).numpy().astype(np.float64)
                cn = np.where(lv, cl + rss[li] * p, 0.0)
                u = np.where(lv, TE['U'][i, li], 0.0)
                Cm = np.stack([cl - u, cn - u], 1)
                e = TE['PHI'][i, li + 2][:, None] * (W[li + 2].T @ e) + Cm if li > 0 else Cm
            for j, m in enumerate(('lin', 'nl')):
                mse[m][i] = np.mean((TE['ERR15'][i] + e[:, j]) ** 2)
    m0 = np.mean(TE['ERR15'] ** 2, 1); ms = np.mean((TE['PRED15'] - TE['TRU15']) ** 2, 1)
    print(f'HELD-OUT nets {Nt} (ids {TE["ids"].min()}-{TE["ids"].max()}): noHZ {m0.mean():.5e} stale {ms.mean():.5e}')
    for m in ('lin', 'nl'):
        q = mse[m] / ms
        print(f'{m:3s} / stale x{mse[m].sum() / ms.sum():.5f} (wins {(q < 1).sum()}/{Nt}, se {q.std(ddof=1) / np.sqrt(Nt):.5f})')
    q = mse['nl'] / mse['lin']
    print(f'nl / lin x{mse["nl"].sum() / mse["lin"].sum():.5f} (wins {(q < 1).sum()}/{Nt}, se {q.std(ddof=1) / np.sqrt(Nt):.5f})')
    np.savez(a.out + '_test.npz', ids=TE['ids'], mse_lin=mse['lin'], mse_nl=mse['nl'], ms=ms, m0=m0)

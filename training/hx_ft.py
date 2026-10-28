"""hx_ft.py (research only): staged final-loss fine-tune of a v5D head through the exact training-form transport.
Loss = sum_nets mean_j r_j^2, r = ERR15 + e_15, e_li = PHI_{li+2} * (W_{li+2}^T e_{li-1}) + (c_li - U_li) on live rows,
c_li = A theta_li + rs_li * MLP(x). Only the shared MLP trains (theta/med/sd/rs frozen). Early stopping on --val nets
(epoch 0 = the start head). Writes <out>_head.npz (same format/keys as the start head) + <out>_theta.npy.
usage: python hx_ft.py <gt.npz> <start_head.npz> <out_prefix> --train f2_a.npz ... --val f3_00.npz ... [--lr 1e-4] [--ep 6] [--bs 4]"""
import argparse, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument('gt'); ap.add_argument('start'); ap.add_argument('out')
ap.add_argument('--train', nargs='+', required=True); ap.add_argument('--val', nargs='+', required=True)
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--ep', type=int, default=4); ap.add_argument('--bs', type=int, default=4)
ap.add_argument('--seed', type=int, default=1234); ap.add_argument('--wd', type=float, default=0.0)
a = ap.parse_args()
torch.set_num_threads(2)
G = np.load(a.gt); n, NL = 1024, 14
STC = [0, 2, 3, 5, 1]
h0 = dict(np.load(a.start)); assert int(h0['xf']) == 1
train_ids_start = set(h0['train_ids'].tolist())


def feats(A, R, ST, XF):
    r = R.astype(np.float64)
    var = np.maximum(r[:, 1], 1e-30); sg = np.sqrt(var); vsg = var * sg; v2 = var * var
    cols = [A[:, 7:].astype(np.float64),
            np.stack([r[:, 13], np.log(var), r[:, 5] / vsg, r[:, 6] / v2, r[:, 7] / v2, r[:, 2] / v2, r[:, 3] / v2, r[:, 4] / v2,
                      r[:, 8] / vsg, r[:, 9] / vsg, r[:, 10] / vsg, r[:, 11] / vsg, r[:, 12] / v2], 1), ST[:, STC].astype(np.float64),
            XF[:, [11, 12]].astype(np.float64) / np.stack([var, sg], 1)]
    return np.concatenate(cols, 1)


def load(files):
    """per net: X (14*1024, F+14) f32 normalized inputs incl. one-hot, cl (14,1024) linear part, LV, U, PHI, ERR15, stale mse"""
    out = []
    for f in files:
        z = np.load(f); z = {q: z[q] for q in ('ids', 'A', 'U', 'LIVE', 'R', 'ST', 'XF', 'PHI', 'ERR15', 'PRED15', 'TRU15')}
        for i in range(min(len(z['ids']), int(os.environ.get('NMAX', '999')))):
            net = int(z['ids'][i]); assert not ((237 <= net <= 336) or (700 <= net <= 849)), net
            X = np.zeros((NL, n, h0['W0'].shape[1]), np.float32); cl = np.zeros((NL, n))
            for li in range(NL):
                X[li, :, :-NL] = np.clip((feats(z['A'][i, li], z['R'][i, li], z['ST'][i, li], z['XF'][i, li]) - h0['med'][li]) / h0['sd'][li], -8, 8)
                X[li, :, -NL + li] = 1.0
                cl[li] = z['A'][i, li].astype(np.float64) @ h0['theta'][2:][li]
            out.append(dict(net=net, X=torch.from_numpy(X.reshape(NL * n, -1)), cl=torch.from_numpy(cl), LV=torch.from_numpy(z['LIVE'][i]),
                            U=torch.from_numpy(np.where(z['LIVE'][i], z['U'][i], 0.0).astype(np.float64)),
                            PHI=torch.from_numpy(z['PHI'][i].astype(np.float64)), ERR=torch.from_numpy(z['ERR15'][i].astype(np.float64)),
                            stale=float(np.mean((z['PRED15'][i] - z['TRU15'][i]) ** 2))))
        print('loaded', f, len(out), flush=True)
    return out


def genW(net):
    rng = np.random.default_rng(np.random.SeedSequence(int(G['seeds'][int(net)])).spawn(3)[0])
    Ws = [(rng.standard_normal((n, n)) * np.sqrt(2 / n)).astype(np.float32) for _ in range(16)]
    return [torch.from_numpy(Ws[j].astype(np.float64)) for j in range(3, 16)]   # W_3 .. W_15


class Head(torch.nn.Module):
    def __init__(self, fin, hid):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(fin, hid), torch.nn.ReLU(), torch.nn.Linear(hid, hid), torch.nn.ReLU(), torch.nn.Linear(hid, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


net = Head(h0['W0'].shape[1], h0['W0'].shape[0])
with torch.no_grad():
    lins = [m for m in net.f if isinstance(m, torch.nn.Linear)]
    for j, m in enumerate(lins):
        m.weight.copy_(torch.from_numpy(h0[f'W{j}'].astype(np.float32))); m.bias.copy_(torch.from_numpy(h0[f'b{j}'].astype(np.float32)))
RS = torch.from_numpy(np.array(h0['rs'], np.float64))


def mse_net(d, W):
    p = net(d['X']).double().reshape(NL, n)
    C = torch.where(d['LV'], d['cl'] + RS[:, None] * p, torch.zeros((), dtype=torch.float64)) - d['U']
    e = C[0]
    for li in range(1, NL):
        e = d['PHI'][li + 2] * (W[li - 1].T @ e) + C[li]
    return torch.mean((d['ERR'] + e) ** 2)


pool = ThreadPoolExecutor(3)


def stream(data, order):
    futs = {}
    for q in range(min(6, len(order))):
        futs[q] = pool.submit(genW, data[order[q]]['net'])
    for q in range(len(order)):
        W = futs.pop(q).result()
        if q + 6 < len(order):
            futs[q + 6] = pool.submit(genW, data[order[q + 6]]['net'])
        yield data[order[q]], W


def evaluate(data):
    tot = 0.0
    with torch.no_grad():
        for d, W in stream(data, list(range(len(data)))):
            tot += float(mse_net(d, W))
    return tot


t0 = time.time()
TR = load(a.train); VA = load(a.val)
tr_ids = {d['net'] for d in TR}; va_ids = {d['net'] for d in VA}
assert tr_ids <= train_ids_start, 'fine-tune nets must be the start head training nets'
assert not (va_ids & tr_ids)
norm = np.mean([d['stale'] for d in TR])
best = evaluate(VA); val0 = best; best_ep = 0; best_state = {k: v.clone() for k, v in net.state_dict().items()}
print(f'ep 0 val {best:.6e} (start head) {time.time() - t0:.0f}s', flush=True)
opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.wd)
g = np.random.default_rng(a.seed)
CK = {5, 10, 20} | set(range(40, 100000, 40)); step = 0
for ep in range(1, a.ep + 1):
    order = g.permutation(len(TR)).tolist(); tl = 0.0; k = 0
    opt.zero_grad()
    for d, W in stream(TR, order):
        loss = mse_net(d, W) / norm / a.bs; loss.backward(); tl += float(loss.detach()) * a.bs; k += 1
        if k % a.bs == 0 or k == len(order):
            opt.step(); opt.zero_grad(); step += 1
            if step in CK:
                v = evaluate(VA)
                print(f'ep {ep} step {step} val {v:.6e} val/start x{v / val0:.6f} {time.time() - t0:.0f}s', flush=True)
                if v < best:
                    best, best_ep = v, step; best_state = {k2: v2.clone() for k2, v2 in net.state_dict().items()}
    print(f'ep {ep} train {tl / len(TR):.5f} (x stale) {time.time() - t0:.0f}s', flush=True)
net.load_state_dict(best_state)
lins = [m for m in net.f if isinstance(m, torch.nn.Linear)]
h1 = dict(h0)
for j, m in enumerate(lins):
    h1[f'W{j}'] = m.weight.detach().numpy().astype(np.float64); h1[f'b{j}'] = m.bias.detach().numpy().astype(np.float64)
h1['ft_best_step'] = best_ep; h1['ft_lr'] = a.lr
np.savez(a.out + '_head.npz', **h1); np.save(a.out + '_theta.npy', h0['theta'])
print('wrote', a.out, 'best step', best_ep, f'val {best:.6e}', f'{time.time() - t0:.0f}s', flush=True)

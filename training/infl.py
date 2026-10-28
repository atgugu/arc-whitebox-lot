"""infl.py (research only): per-row influence of a layer-l born correction on the final-layer error, under the
production transport e_{l+1} = PHI_{l+1} * (W_{l+1}^T e_l) (same as nl_train2's evaluation).
INF[i, li, j] = || T_{l->15} e_j ||^2 / n, l = LS[li] = li + 2.  usage: python infl.py <gt.npz> <f2_k.npz> <out.npz>"""
import sys, numpy as np
gt, f2, out = sys.argv[1:4]
G = np.load(gt); z = np.load(f2); ids = z['ids']; PHI = z['PHI'].astype(np.float64); n = 1024
INF = np.zeros((len(ids), 14, n), np.float32)
for a, i in enumerate(ids):
    rng = np.random.default_rng(np.random.SeedSequence(int(G['seeds'][int(i)])).spawn(3)[0])
    W = [(rng.standard_normal((n, n)) * np.sqrt(2 / n)).astype(np.float32).astype(np.float64) for _ in range(16)]
    # M_l maps a layer-l correction (row vector e_l) to the final error: e_15 = e_l @ M_l ; M_15 = I, M_l = (W_{l+1} * PHI_{l+1}[None,:]) @ M_{l+1}
    M = np.eye(n)
    INF[a, 13] = 1.0 / n
    for l in range(14, 1, -1):
        M = (W[l + 1] * PHI[a, l + 1][None, :]) @ M
        INF[a, l - 2] = (M * M).sum(1) / n
    if a % 20 == 0:
        print(a, int(i), INF[a].mean(1)[[0, 6, 12, 13]] * n, flush=True)
np.savez(out, ids=ids, INF=INF)
print('wrote', out)

#!/usr/bin/env python3
"""lot_toy_check.py -- independent numerical checks of the LOT kappa4 identities (numpy only).

Companion of docs/LOT_DERIVATION.md.  Run:  python3 lot_toy_check.py  (about 1 min on one CPU core).

Part A  single neuron, coincident indices (closed-form truncated-normal moments, Hermite series, Monte Carlo):
        kappa(z,z,h,h) exact vs the He2 truncation 2 c2^2 sigma^6, and the c2*c4 cross term.
Part B  truncated chaos model on a width-8, depth-4 bias-free ReLU MLP (He init, x ~ N(0, I)):
        the Gaussian backbone is sampled exactly (linear-Gaussian Markov chain); the chaos components
        L (first), Q (He2 vertices), T* (He3 vertices) and Tind (nested He2-He2) are built explicitly and
        their joint cumulants are estimated by Monte Carlo and compared with the closed forms:
          kappa(L,L,Q)    = Cov(L, psi)            -> D3 = 3 Cov(L, psi)
          kappa(L,L,Q,Q)  = 2 Var(psi)             -> kappa4_chain = 6 kappa(L,L,Q,Q) = 12 Var(psi)
          kappa(L,L,L,T*) = sum c3 P Gam^3         -> kappa4_star  = 4 sum c3 P Gam^3
          kappa(L,L,L,Ti) = 3 sum V o Xi           -> kappa4_ind   = 12 sum V o Xi
        plus exact (non-MC) checks: cumulant generating function of a Gaussian quadratic form, the
        Cauchy-Schwarz floor, the split of the transported 3+2 tree into 'induced' + cross-source chains,
        the response-closure (S-form) representation of the projected chain, and a line-by-line mirror
        of _lot_g4 in estimator/estimator.py.
Part C  accuracy probe against the TRUE network (Monte Carlo from x), width 32, depth 6, 4 networks: not an
        identity check; it shows how far leading order is from the truth at small width.
Each part draws from its own random stream, so --quick and the full run use the same networks.
"""
import argparse
import math
import time

import numpy as np

SQ2PI = math.sqrt(2.0 * math.pi)
_erf = np.vectorize(math.erf)


def npdf(x):
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / SQ2PI


def ncdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def hermite_table(x, kmax):
    """probabilists' He_0..He_kmax at x (array) -> shape (kmax+1,) + x.shape"""
    x = np.asarray(x, dtype=float)
    H = np.zeros((kmax + 1,) + x.shape)
    H[0] = 1.0
    if kmax >= 1:
        H[1] = x
    for m in range(1, kmax):
        H[m + 1] = x * H[m] - m * H[m - 1]
    return H


def relu_gauss(mu, var, kmax):
    """Gaussian ReLU readout for z ~ N(mu, var): mean, variance and c_k = E[relu^(k)(z)], k = 1..kmax.
    c_1 = Phi(a), c_k = (-1)^k He_{k-2}(a) phi(a) / sigma^(k-1) (k >= 2), a = mu / sigma."""
    sig = np.sqrt(var)
    a = mu / sig
    ph, Ph = npdf(a), ncdf(a)
    m = sig * (ph + a * Ph)
    e2 = var * ((a * a + 1.0) * Ph + a * ph)
    vh = e2 - m * m
    H = hermite_table(a, kmax)
    c = np.zeros((kmax + 1,) + np.shape(mu))
    c[0] = m
    c[1] = Ph
    for k in range(2, kmax + 1):
        c[k] = (-1.0) ** k * H[k - 2] * ph / sig ** (k - 1)
    return m, vh, c


# ----------------------------------------------------------------------------------------------------------
# Part A: single neuron
# ----------------------------------------------------------------------------------------------------------
def trunc_moments(a_low, mmax):
    """T_m = E[u^m 1(u > a_low)] for u ~ N(0,1), m = 0..mmax"""
    T = np.zeros(mmax + 1)
    T[0] = 1.0 - float(ncdf(a_low))
    T[1] = float(npdf(a_low))
    for m in range(2, mmax + 1):
        T[m] = a_low ** (m - 1) * float(npdf(a_low)) + (m - 1) * T[m - 2]
    return T


def kzzhh_exact(alpha, sigma=1.0):
    """kappa(z, z, h, h) for z ~ N(alpha sigma, sigma^2), h = relu(z), from truncated-normal moments (exact)."""
    T = trunc_moments(-alpha, 8)

    def E_up_hq(p, q):   # E[u^p (u + alpha)_+^q]
        return sum(math.comb(q, r) * alpha ** (q - r) * T[p + r] for r in range(q + 1))
    m = E_up_hq(0, 1)
    Eh2 = E_up_hq(0, 2)
    vh = Eh2 - m * m
    Eu2h2 = E_up_hq(2, 2)
    Eu2h = E_up_hq(2, 1)
    Euh = E_up_hq(1, 1)
    Eu2ht2 = Eu2h2 - 2 * m * Eu2h + m * m          # E[u^2 (h - m)^2], E u^2 = 1
    k = Eu2ht2 - vh - 2 * Euh ** 2
    return k * sigma ** 4


def kzzhh_series(alpha, mmax):
    """Hermite series of kappa(u,u,g,g), g = (u + alpha)_+ : sum_{m>=2} 2m g_m^2/m! + 2 sum_{m>=1} g_m g_{m+2}/m!"""
    _, _, c = relu_gauss(np.array(alpha * 1.0), np.array(1.0), mmax + 2)
    g = [float(x) for x in c]

    def ratio(x, m):    # x / m! without overflow
        return 0.0 if x == 0.0 else math.copysign(math.exp(math.log(abs(x)) - math.lgamma(m + 1)), x)
    s = 0.0
    for m in range(2, mmax + 1):
        s += 2 * m * ratio(g[m] * g[m], m)
    for m in range(1, mmax - 1):
        s += 2 * ratio(g[m] * g[m + 2], m)
    return s


def part_a(rng, nmc):
    print("=" * 100)
    print("PART A. Single neuron: coincident-index caveat (z ~ N(alpha, 1), h = relu(z))")
    print("-" * 100)
    _, _, c = relu_gauss(np.array(0.0), np.array(1.0), 6)
    he2 = 2 * float(c[2]) ** 2
    ex = kzzhh_exact(0.0)
    print(f"alpha = 0: exact kappa(z,z,h,h) = {ex:.6f}; He2 only 2 c2^2 = {he2:.6f}; "
          f"c2*c4 cross term = {float(c[2] * c[4]):.6f}; ratio exact/He2 = {ex / he2:.4f}")
    for mm in (4, 8, 20, 60, 140):
        print(f"   Hermite series truncated at m = {mm:3d}: {kzzhh_series(0.0, mm):.6f}")
    u = rng.standard_normal(nmc)
    h = np.maximum(u, 0.0)
    ht = h - h.mean()
    ut = u - u.mean()
    kmc = np.mean(ut ** 2 * ht ** 2) - np.mean(ut ** 2) * np.mean(ht ** 2) - 2 * np.mean(ut * ht) ** 2
    print(f"   Monte Carlo (N = {nmc:.0e}): {kmc:.4f}")
    print("alpha-dependence (sigma = 1). 'tree' = what LOT's tree level assigns to this coincident configuration:")
    print("the He2 chain 2 c2^2 plus the coincident part of the He3 star, 2 c1 c3 (zero at alpha = 0).")
    print("r(alpha) = (exact - 2c1c3) / (2c2^2) is the factor that would make LOT's He2 same-hub term exact at order a^2 p^2.")
    print("    alpha |   exact  | He2 2c2^2 | tree 2c2^2+2c1c3 | exact - tree | r(alpha)")
    for a in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
        _, _, c = relu_gauss(np.array(a), np.array(1.0), 4)
        e = kzzhh_exact(a)
        t2 = 2 * float(c[2]) ** 2
        tt = t2 + 2 * float(c[1] * c[3])
        print(f"    {a:5.2f} | {e:+.5f} | {t2:.5f}   | {tt:+.5f}         | {e - tt:+.5f}     | {(e - 2 * float(c[1] * c[3])) / t2:.3f}")
    # Stein check of the c_k: E[relu(z) He_k(u)] = sigma^k E[relu^(k)(z)]
    a = 0.7
    T = trunc_moments(-a, 10)
    H = np.polynomial.hermite_e.herme2poly
    ok = []
    _, _, c = relu_gauss(np.array(a), np.array(1.0), 5)
    for k in range(1, 5):
        coef = np.zeros(k + 1)
        coef[k] = 1.0
        p = H(coef)                                   # He_k as power series in u
        val = sum(p[j] * (T[j + 1] + a * T[j]) for j in range(len(p)))   # E[(u + a)_+ He_k(u)]
        ok.append(abs(val - float(c[k])))
    print(f"c_k Stein check at alpha = 0.7 (k = 1..4): max |E[relu He_k] - c_k| = {max(ok):.2e}")
    al = np.linspace(-3, 3, 61)
    closed = 2 * ncdf(al) * (1 - ncdf(al)) - 2 * npdf(al) ** 2 - 2 * al * ncdf(al) * npdf(al)
    num = np.array([kzzhh_exact(a) for a in al])
    print(f"closed form kappa(z,z,h,h)/sigma^4 = 2 Phi(1-Phi) - 2 phi^2 - 2 alpha Phi phi: max |closed - moments| on [-3, 3] = "
          f"{np.max(np.abs(closed - num)):.1e};  r(alpha) = Phi(1-Phi)/phi^2 - 1 >= pi/2 - 1 = {math.pi / 2 - 1:.4f}")


# ----------------------------------------------------------------------------------------------------------
# Gaussian closure and backbone for a bias-free ReLU MLP
# ----------------------------------------------------------------------------------------------------------
class Backbone:
    """Gaussian-closure backbone of z_l = W_l^T h_{l-1}, h_0 = x ~ N(0, I), layers 1..L (row-vector code: z = h @ W).
    P[(l,b)] (n x n) = linear propagator from h_b to z_l;  Gam[(l,b)] = Cov(z_l, z_b) = P[(l,b)] diag(c1_b) C_b;
    the constructive backbone zeta_{l+1} = mu_{l+1} + W_{l+1}^T (c1_l o zt_l + eta_l), eta_l independent N(0, Inn_l),
    Inn_l = Cov_G(h_l) - diag(c1_l) C_l diag(c1_l) (Mehler terms k >= 2 plus the exact diagonal tail: PSD)."""

    def __init__(self, Ws, kmax=40):
        self.Ws = Ws
        L = len(Ws)
        n = Ws[0].shape[1]
        self.L, self.n = L, n
        self.mu, self.C, self.c, self.var, self.inn = {}, {}, {}, {}, {}
        self.mu[1] = np.zeros(n)
        self.C[1] = Ws[0].T @ Ws[0]
        for l in range(1, L + 1):
            var = np.diag(self.C[l]).copy()
            self.var[l] = var
            m, vh, c = relu_gauss(self.mu[l], var, kmax)
            self.c[l] = c
            if l == L:
                break
            C = self.C[l]
            covh = np.zeros((n, n))
            Ck = np.ones((n, n))
            for k in range(1, kmax + 1):
                Ck = Ck * C
                covh += np.outer(c[k], c[k]) * Ck / math.factorial(k)
            tail = np.maximum(vh - np.diag(covh), 0.0)
            covh += np.diag(tail)
            inn = covh - np.outer(c[1], c[1]) * C
            w, U = np.linalg.eigh(inn)
            self.inn[l] = (U * np.sqrt(np.maximum(w, 0.0))) @ U.T      # symmetric square root
            W = Ws[l]
            self.mu[l + 1] = W.T @ m
            self.C[l + 1] = W.T @ covh @ W
        self.P, self.Gam = {}, {}
        for l in range(2, L + 1):
            for b in range(1, l):
                if b == l - 1:
                    self.P[(l, b)] = Ws[l - 1].T.copy()
                else:
                    self.P[(l, b)] = Ws[l - 1].T @ (self.c[l - 1][1][:, None] * self.P[(l - 1, b)])
                self.Gam[(l, b)] = self.P[(l, b)] @ (self.c[b][1][:, None] * self.C[b])

    def G(self, b, bp):
        """backbone covariance block Cov(zeta_b, zeta_bp)"""
        if b == bp:
            return self.C[b]
        return self.Gam[(b, bp)] if b > bp else self.Gam[(bp, b)].T

    def sample(self, rng, N):
        """centred backbone samples zt[l] (N x n), l = 1..L"""
        n = self.n
        zt = {1: rng.standard_normal((N, n)) @ self.Ws[0]}
        for l in range(1, self.L):
            eta = rng.standard_normal((N, n)) @ self.inn[l]
            zt[l + 1] = (self.c[l][1] * zt[l] + eta) @ self.Ws[l]
        return zt


def lot_terms(bb, l):
    """closed-form LOT objects at current layer l (dict of per-neuron vectors and the per-source matrices)"""
    V, Xi = {}, {}
    covLpsi = np.zeros(bb.n)
    star = np.zeros(bb.n)
    ind = np.zeros(bb.n)
    for b in range(1, l):
        P, Gm = bb.P[(l, b)], bb.Gam[(l, b)]
        V[b] = bb.c[b][2][None, :] * P * Gm
        covLpsi += np.sum(V[b] * Gm, axis=1)
        star += np.sum(bb.c[b][3][None, :] * P * Gm ** 3, axis=1)
        X = np.zeros((bb.n, bb.n))
        for bp in range(1, b):
            X += (bb.Gam[(l, bp)] ** 2 * bb.c[bp][2][None, :]) @ bb.P[(b, bp)].T
        Xi[b] = X
        ind += np.sum(V[b] * X, axis=1)
    varpsi = np.zeros(bb.n)
    self_ = np.zeros(bb.n)
    same = np.zeros(bb.n)
    for b in range(1, l):
        for bp in range(1, l):
            t = np.sum((V[b] @ bb.G(b, bp)) * V[bp], axis=1)
            varpsi += t
            if b == bp:
                self_ += t
        same += np.sum(V[b] ** 2 * bb.var[b][None, :], axis=1)
    return dict(V=V, Xi=Xi, covLpsi=covLpsi, varpsi=varpsi, self=self_, same=same, star=star, ind=ind,
                D3=3 * covLpsi, k4chain=12 * varpsi, k4star=4 * star, k4ind=12 * ind)


def he_init(rng, n, L):
    return [rng.standard_normal((n, n)) * math.sqrt(2.0 / n) for _ in range(L)]


# ----------------------------------------------------------------------------------------------------------
# Part B: truncated chaos model
# ----------------------------------------------------------------------------------------------------------
def part_b(rng, nmc, nbatch):
    print("=" * 100)
    n, L = 8, 4
    print(f"PART B. Truncated chaos model: width {n}, depth {L} (pre-activations z_1..z_{L}), current layer l = {L}; "
          f"MC N = {nmc * nbatch:.1e} backbone samples")
    print("-" * 100)
    Ws = he_init(rng, n, L)
    bb = Backbone(Ws)
    l = L
    th = lot_terms(bb, l)
    # --- backbone sanity: constructive sampler reproduces the stated covariances
    zt = bb.sample(rng, 400000)
    err_c = max(np.max(np.abs(np.cov(zt[b].T) - bb.C[b])) / np.max(np.abs(bb.C[b])) for b in range(1, L + 1))
    err_g = max(np.max(np.abs(zt[a].T @ zt[b] / zt[a].shape[0] - bb.Gam[(a, b)])) / np.max(np.abs(bb.C[a]))
                for a in range(2, L + 1) for b in range(1, a))
    print(f"backbone sampler: max rel. error of C_b {err_c:.1e}, of Cov(z_l, z_b) = Gam {err_g:.1e} (N = 4e5; MC level)")

    # --- Monte Carlo of the joint cumulants of the explicit chaos components
    keys = ['L2', 'Q2', 'LQ', 'L2Q', 'L2Q2', 'L3Ts', 'LTs', 'L3Ti', 'LTi']
    acc = {k: [] for k in keys}
    d21_acc = np.zeros((n, n))                                      # tree-level (i,i,j) slice: k(L_i,L_i,Q_j) + 2 k(L_i,L_j,Q_i)
    t0 = time.time()
    for _ in range(nbatch):
        zt = bb.sample(rng, nmc)
        Lc = zt[l]
        Q = np.zeros_like(Lc)
        Ts = np.zeros_like(Lc)
        Ti = np.zeros_like(Lc)
        Qsub = {}
        for b in range(1, l):
            Qb = np.zeros_like(Lc)
            for bp in range(1, b):
                Qb += ((zt[bp] ** 2 - bb.var[bp]) * (bb.c[bp][2] / 2)) @ bb.P[(b, bp)].T
            Qsub[b] = Qb
        for b in range(1, l):
            Pt = bb.P[(l, b)].T
            z, v = zt[b], bb.var[b]
            Q += ((z ** 2 - v) * (bb.c[b][2] / 2)) @ Pt
            Ts += ((z ** 3 - 3 * v * z) * (bb.c[b][3] / 6)) @ Pt
            Ti += (z * Qsub[b] * bb.c[b][2]) @ Pt
        L2 = Lc * Lc
        acc['L2'].append(L2.mean(0))
        acc['Q2'].append((Q * Q).mean(0))
        acc['LQ'].append((Lc * Q).mean(0))
        acc['L2Q'].append((L2 * Q).mean(0))
        acc['L2Q2'].append((L2 * Q * Q).mean(0))
        acc['L3Ts'].append((L2 * Lc * Ts).mean(0))
        acc['LTs'].append((Lc * Ts).mean(0))
        acc['L3Ti'].append((L2 * Lc * Ti).mean(0))
        acc['LTi'].append((Lc * Ti).mean(0))
        d21_acc += (L2.T @ Q + 2 * (Lc * Q).T @ Lc) / Lc.shape[0]
    A = {k: np.array(v) for k, v in acc.items()}            # (nbatch, n)
    d21_mc = d21_acc / nbatch

    def cums(S):
        k3 = S['L2Q']
        k4q = S['L2Q2'] - S['L2'] * S['Q2'] - 2 * S['LQ'] ** 2
        k4s = S['L3Ts'] - 3 * S['L2'] * S['LTs']
        k4i = S['L3Ti'] - 3 * S['L2'] * S['LTi']
        return k3, k4q, k4s, k4i
    full = cums({k: v.mean(0) for k, v in A.items()})
    per = [cums({k: v[i] for k, v in A.items()}) for i in range(nbatch)]
    se = [np.std(np.array([p[j] for p in per]), axis=0, ddof=1) / math.sqrt(nbatch) for j in range(4)]
    names = ["kappa(L,L,Q)    vs Cov(L,psi)        [D3 = 3x]",
             "kappa(L,L,Q,Q)  vs 2 Var(psi)         [chain = 6x = 12 Var]",
             "kappa(L,L,L,T*) vs sum c3 P Gam^3     [star = 4x]",
             "kappa(L,L,L,Ti) vs 3 sum V o Xi       [induced = 4x = 12 V Xi]"]
    theo = [th['covLpsi'], 2 * th['varpsi'], th['star'], 3 * th['ind']]
    print(f"MC done in {time.time() - t0:.1f} s.  Per identity: pooled ratio sum(MC)/sum(theory) +- se, max |z| over the {n} neurons")
    for nm, mc, s, t in zip(names, full, se, theo):
        sw = np.sign(t)
        ratio = np.sum(mc * sw) / np.sum(t * sw)
        rse = math.sqrt(np.sum(s ** 2)) / abs(np.sum(t * sw))
        z = np.max(np.abs((mc - t) / s))
        print(f"   {nm}: ratio {ratio:.4f} +- {rse:.4f}; max|z| {z:.2f}")
    Kmat_ = sum(th['V'][b] @ bb.Gam[(l, b)].T for b in range(1, l))           # Cov(psi_i, L_j)
    A21 = sum((bb.Gam[(l, b)] ** 2 * bb.c[b][2][None, :]) @ bb.P[(l, b)].T for b in range(1, l))   # k(L_i,L_i,Q_j)
    print(f"   tree-level (i,i,j) slice D21_ij = k(L_i,L_i,Q_j) + 2 Cov(psi_i, L_j): max |MC - theory| / max|theory| = "
          f"{np.max(np.abs(d21_mc - (A21 + 2 * Kmat_))) / np.max(np.abs(A21 + 2 * Kmat_)):.1e}")
    print("   per neuron (theory / MC): ")
    for i in range(n):
        print("     i=%d  Cov %+.4e/%+.4e  2Var %.4e/%.4e  star %+.4e/%+.4e  3VXi %+.4e/%+.4e"
              % (i, theo[0][i], full[0][i], theo[1][i], full[1][i], theo[2][i], full[2][i], theo[3][i], full[3][i]))

    # --- exact: cumulant generating function of a Gaussian quadratic form (independent of the Wick count)
    dims = [n] * l
    off = np.cumsum([0] + dims)
    Gf = np.zeros((off[-1], off[-1]))
    for a in range(1, l + 1):
        for b in range(1, l + 1):
            Gf[off[a - 1]:off[a], off[b - 1]:off[b]] = bb.G(a, b)
    w, U = np.linalg.eigh(Gf)
    Gh = (U * np.sqrt(np.maximum(w, 0.0))) @ U.T
    e3, e4, l3, l4 = [], [], [], []
    for i in range(n):
        sel = np.zeros(off[-1])
        sel[off[l - 1] + i] = 1.0
        a = Gh @ sel
        d = np.zeros(off[-1])
        for b in range(1, l):
            d[off[b - 1]:off[b]] = bb.c[b][2] / 2 * bb.P[(l, b)][i]
        M = Gh @ (d[:, None] * Gh)
        e3.append(6 * a @ M @ a)
        e4.append(48 * a @ M @ M @ a)
        l3.append(8 * np.trace(M @ M @ M))
        l4.append(48 * np.trace(M @ M @ M @ M))
    e3, e4 = np.array(e3), np.array(e4)
    print("exact CGF of z(eps) = L + eps Q: kappa3 = eps 6 a'Ma + eps^3 8 tr M^3, kappa4 = eps^2 48 a'M^2a + eps^4 48 tr M^4")
    print(f"   max rel |6 a'Ma - D3| = {np.max(np.abs(e3 - th['D3']) / np.abs(th['D3'])):.1e};  "
          f"max rel |48 a'M^2a - 12 Var(psi)| = {np.max(np.abs(e4 - th['k4chain']) / th['k4chain']):.1e}")
    print(f"   loops at width {n} (not part of LOT): mean |8 tr M^3| / mean |D3| = {np.mean(np.abs(l3)) / np.mean(np.abs(th['D3'])):.2f}; "
          f"mean 48 tr M^4 / mean chain = {np.mean(l4) / np.mean(th['k4chain']):.2f}")

    # --- exact: Cauchy-Schwarz floor
    var_l = bb.var[l]
    floor = 4.0 / 3.0 * th['D3'] ** 2 / var_l
    rho = th['covLpsi'] / np.sqrt(var_l * th['varpsi'])
    print(f"Cauchy-Schwarz: min_i [12 Var(psi) - (4/3) D3^2/var] / (12 Var(psi)) = {np.min(1 - floor / th['k4chain']):.4f} (>= 0); "
          f"rho_i in [{rho.min():.3f}, {rho.max():.3f}]; (4/3)D3^2/(var rho^2) = 12 Var(psi): "
          f"max rel err {np.max(np.abs(floor / rho ** 2 - th['k4chain']) / th['k4chain']):.1e}")
    # rank-one tightness: z = u + e He2(u)
    eps = 0.1
    print(f"   rank-one case z = u + e(u^2-1), e = {eps}: D3 = 6e = {6 * eps:.3f}, chain = 48 e^2 = {48 * eps * eps:.4f}, "
          f"(4/3) D3^2/var = {4 / 3 * (6 * eps) ** 2:.4f} (equality)")
    print(f"self vs cross-source chains at l = {l}: mean self/total = {np.mean(th['self']) / np.mean(th['varpsi']):.3f}, "
          f"corr(self, total) = {np.corrcoef(th['self'], th['varpsi'])[0, 1]:.3f}")

    # --- exact: transported 3+2 tree (ARC factor_k4 / Kuznetsov sec. 7.1) = LOT induced + cross-source chains
    worst = 0.0
    for b in range(2, l):
        # leading-order kappa3 of z_b (the 'cherry'): sum_{b'<b} sum_p c2 [P G G + G P G + G G P]
        k3 = np.zeros((n, n, n))
        for bp in range(1, b):
            P_, G_, c2 = bb.P[(b, bp)], bb.Gam[(b, bp)], bb.c[bp][2]
            k3 += (np.einsum('jp,kp,mp,p->jkm', P_, G_, G_, c2) + np.einsum('jp,kp,mp,p->jkm', G_, P_, G_, c2)
                   + np.einsum('jp,kp,mp,p->jkm', G_, G_, P_, c2))
        # 3+2 tree at layer b: 12 d1_i C_ij d2_j kappa3(j,k,m) d1_k d1_m, read on the diagonal of z_l through P^(l,b) diag(c1_b)
        U_ = bb.P[(l, b)] * bb.c[b][1][None, :]                 # linear map z_b -> L at layer l
        T32 = 12 * np.einsum('ap,pq,q,aq,qrs,ar,as->a', U_, bb.C[b], bb.c[b][2], bb.P[(l, b)], k3, U_, U_)
        cross = np.zeros(n)
        for bp in range(1, b):
            cross += 24 * np.sum((th['V'][b] @ bb.G(b, bp)) * th['V'][bp], axis=1)
        indb = 12 * np.sum(th['V'][b] * th['Xi'][b], axis=1)
        worst = max(worst, np.max(np.abs(T32 - indb - cross)) / np.max(np.abs(T32)))
    # path and star trees: transported path_b = 12 [V^b C_b V^bT]_ii (self-chain), star_b = 4 sum c3 P Gam^3
    print(f"3+2 tree born at b (b = 2..{l - 1}), transported to layer {l}: max rel |T32 - (12 V o Xi + 24 sum_b'<b V G V')| = {worst:.1e}")
    print("   => LOT (chain + star + induced) = sum over birth layers of ARC's path (x12) + star (x4) + 3+2 (x12) trees")

    # --- exact: response-closure (S-form) representation of the projected chain, and a mirror of _lot_g4
    K = 3
    W = Ws[l - 1]
    Cl = bb.C[l]
    Om = W[:, :K]                                                   # fixed sketch: first K weight columns
    Qb, _ = np.linalg.qr(Cl @ Om)                                   # one-pass range finder (PORTLOT_Q = 1)
    Gq = Qb.T @ Cl @ Qb
    # mirror of _lot_g4 (non-J2 and J2 paths)
    srcs = list(range(1, l))
    P_st = np.stack([bb.P[(l, b)] for b in srcs])
    A_st = np.stack([bb.P[(l, b)] @ (bb.c[b][1][:, None] * (bb.C[b] - np.diag(bb.var[b]))) for b in srcs])
    slab = np.stack([np.stack([bb.c[b][1] * bb.var[b], bb.var[b], bb.c[b][3] / 3]) for b in srcs])   # (k, 3, n)
    w2b = np.stack([bb.c[b][2] for b in srcs])[:, None, :]                                        # (k, 1, n)
    T3 = P_st * slab[:, 0][:, None, :] + A_st                      # = Gam
    T1 = P_st * T3                                                  # = P o Gam
    T2 = T1 * w2b                                                   # = V
    gu = np.swapaxes(T3, -1, -2) @ Qb                               # Gam^T Q per source
    KU = np.einsum('kij,kjq->iq', T2, gu)
    KU_j2 = np.einsum('kij,kjq->iq', T1, gu * np.swapaxes(w2b, -1, -2))
    projc = np.einsum('iq,qi->i', KU, np.linalg.solve(Gq, KU.T))
    samec = np.einsum('kij,kij,kj->i', T2, T2, slab[:, 1])
    starc = np.einsum('kij,kj->i', T3 * T3 * T1, slab[:, 2])
    cand = projc + samec + starc
    # derivation-form quantities
    Kmat = sum(th['V'][b] @ bb.Gam[(l, b)].T for b in srcs)         # K_im = Cov(psi_i, zeta_lm)
    KQ = Kmat @ Qb
    proj_d = np.sum((KQ @ np.linalg.inv(Gq)) * KQ, axis=1)          # Var of the L2 projection of psi on span(Q^T zeta_l)
    R = rng.standard_normal((K, K)) + 3 * np.eye(K)                 # basis invariance: Q -> Q R
    QR_ = Qb @ R
    proj_r = np.sum(((Kmat @ QR_) @ np.linalg.inv(QR_.T @ Cl @ QR_)) * (Kmat @ QR_), axis=1)
    print(f"code mirror of _lot_g4 (K = {K}): max rel |T3 - Gam| = "
          f"{max(np.max(np.abs(T3[i] - bb.Gam[(l, b)])) for i, b in enumerate(srcs)) / np.max(np.abs(bb.Gam[(l, l - 1)])):.1e}; "
          f"|KU(J2) - KU| = {np.max(np.abs(KU_j2 - KU)) / np.max(np.abs(KU)):.1e}")
    print(f"   12 cand vs 12 Var(Pi_K psi) + 12 sum V^2 var + 4 sum c3 P Gam^3: max rel err "
          f"{np.max(np.abs(12 * cand - (12 * proj_d + 12 * th['same'] + th['k4star'])) / np.abs(12 * cand)):.1e}; "
          f"basis invariance (Q -> QR) {np.max(np.abs(proj_r - proj_d) / proj_d):.1e}")
    # S-form: k_iq = W_i^T B_q W_i with h-side modes B_q at layer l-1
    cl1 = bb.c[l - 1][1]
    Bq = []
    for q in range(K):
        Mq = np.zeros((n, n))
        for b in srcs:
            if b == l - 1:
                Hb, Ghb = np.eye(n), cl1[:, None] * bb.C[l - 1]
            else:
                Hb, Ghb = cl1[:, None] * bb.P[(l - 1, b)], cl1[:, None] * bb.Gam[(l - 1, b)]
            Rb = bb.Gam[(l, b)].T @ Qb
            Mq += Hb @ ((bb.c[b][2] * Rb[:, q])[:, None] * Ghb.T)
        Bq.append(0.5 * (Mq + Mq.T))
    kS = np.array([[W[:, i] @ Bq[q] @ W[:, i] for q in range(K)] for i in range(n)])
    Lam = 4 * np.linalg.inv(Gq)
    sform = np.array([3 * kS[i] @ Lam @ kS[i] for i in range(n)])
    print(f"   S-form: max rel |W_i^T B_q W_i - KU_iq| = {np.max(np.abs(kS - KU)) / np.max(np.abs(KU)):.1e}; "
          f"3 q'Lam q with Lam = 4 (Q^T C Q)^-1 vs 12 Var(Pi_K psi): {np.max(np.abs(sform - 12 * proj_d) / (12 * proj_d)):.1e}; "
          f"Lam PSD: min eig {np.min(np.linalg.eigvalsh(Lam)):.2e}")
    # double counting of proj + same as K grows (top-K eigenvectors of C_l)
    wC, UC = np.linalg.eigh(Cl)
    UC = UC[:, ::-1]
    row = []
    for KK in (1, 2, 4, 8):
        QQ = UC[:, :KK]
        KQ2 = Kmat @ QQ
        pj = np.sum((KQ2 @ np.linalg.inv(QQ.T @ Cl @ QQ)) * KQ2, axis=1)
        row.append(f"K={KK}: {np.mean(pj + th['same']) / np.mean(th['varpsi']):.3f}")
    print("   mean (Var(Pi_K psi) + same) / mean Var(psi) with exact top-K modes: " + ", ".join(row) +
          "  (not a decomposition: 'same' double counts what the projection already holds)")


# ----------------------------------------------------------------------------------------------------------
# Part C: accuracy probe against the true network
# ----------------------------------------------------------------------------------------------------------
def part_c(rng, nmc, nbatch, nnets):
    print("=" * 100)
    n, L = 32, 6
    lays = (3, 4, 5, 6)
    print(f"PART C. Accuracy probe vs the TRUE network (MC from x): width {n}, depth {L}, {nnets} networks, "
          f"N = {nmc * nbatch:.1e} per network")
    print("-" * 100)
    rows = ['D3 (LOT) vs true kappa3', 'chain+star+induced', 'chain+star', 'self-chains+star', '(4/3) D3^2/var',
            'same-hub x r(alpha), +s+i']
    res = {l: {r: [] for r in rows} for l in lays}
    lev = {l: [] for l in lays}
    t0 = time.time()
    nets = [he_init(rng, n, L) for _ in range(nnets)]          # all networks first: identical in --quick and full runs
    for Ws in nets:
        bb = Backbone(Ws)
        sums = {l: np.zeros((4, n)) for l in lays}
        for _ in range(nbatch):
            h = rng.standard_normal((nmc, n))
            for l in range(1, L + 1):
                z = h @ Ws[l - 1]
                if l in sums:
                    s = sums[l]
                    z2 = z * z
                    s[0] += z.sum(0)
                    s[1] += z2.sum(0)
                    s[2] += (z2 * z).sum(0)
                    s[3] += (z2 * z2).sum(0)
                h = np.maximum(z, 0.0)
        N = nmc * nbatch
        for l in lays:
            m1, m2, m3, m4 = sums[l] / N
            var = m2 - m1 ** 2
            k3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
            k4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3 * var ** 2
            th = lot_terms(bb, l)
            # exploratory: same-hub term with the exact coincident weight, 12 (P Gam)^2 [Phi(1-Phi) - phi^2] instead of
            # the He2 value 12 V^2 var = 12 (P Gam)^2 phi^2 (alpha of the hub; see Part A)
            same_r = np.zeros(n)
            for b in range(1, l):
                al = bb.mu[b] / np.sqrt(bb.var[b])
                wgt = ncdf(al) * (1 - ncdf(al)) - npdf(al) ** 2
                same_r += 12 * np.sum((bb.P[(l, b)] * bb.Gam[(l, b)]) ** 2 * wgt[None, :], axis=1)
            k4r = th['k4chain'] - 12 * th['same'] + same_r + th['k4star'] + th['k4ind']
            preds = [(th['D3'], k3), (th['k4chain'] + th['k4star'] + th['k4ind'], k4),
                     (th['k4chain'] + th['k4star'], k4), (12 * th['self'] + th['k4star'], k4),
                     (4 / 3 * th['D3'] ** 2 / bb.var[l], k4), (k4r, k4)]
            for r, (x, y) in zip(rows, preds):
                res[l][r].append((np.corrcoef(x, y)[0, 1], np.polyfit(x, y, 1)[0]))
            lev[l].append((np.mean(k4), np.sqrt(24.0 / N) * np.mean(var ** 2), np.mean(12 * th['same']) / np.mean(th['k4chain']),
                           np.mean(bb.var[l] / var)))
    print(f"MC done in {time.time() - t0:.1f} s.  Entries: mean over networks of corr / slope (slope = true regressed on prediction)")
    print("    layer | " + " | ".join(f"{r:>22s}" for r in rows))
    for l in lays:
        cells = []
        for r in rows:
            a = np.array(res[l][r])
            cells.append(f"{a[:, 0].mean():+.3f} / {a[:, 1].mean():+.2f}      ")
        print(f"    {l:5d} | " + " | ".join(f"{c:>22s}" for c in cells))
    for l in lays:
        a = np.array(lev[l])
        print(f"    layer {l}: true kappa4 mean {a[:, 0].mean():.3f} (MC se per neuron ~{a[:, 1].mean():.1e}); "
              f"same-hub share of the chain {a[:, 2].mean():.2f}; Gaussian-closure var / MC var {a[:, 3].mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20261025)
    ap.add_argument('--quick', action='store_true', help='smaller Monte Carlo (about 15 s)')
    args = ap.parse_args()
    rngs = [np.random.default_rng([args.seed, k]) for k in range(3)]   # one stream per part: same networks in both modes
    t0 = time.time()
    if args.quick:
        part_a(rngs[0], 2_000_000)
        part_b(rngs[1], 250_000, 8)
        part_c(rngs[2], 500_000, 1, 4)
    else:
        part_a(rngs[0], 4_000_000)
        part_b(rngs[1], 500_000, 32)
        part_c(rngs[2], 500_000, 6, 4)
    print("=" * 100)
    print(f"total wall time {time.time() - t0:.1f} s (seed {args.seed})")


if __name__ == '__main__':
    main()

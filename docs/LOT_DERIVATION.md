---
title: "LOT: a leading-order-tree closure for the fourth cumulant read by the mean program"
subtitle: "Derivation, verification and positioning (companion to the write-up)"
date: "2026-10-28"
---

Written by a Claude agent for Antoniu Gugu (AIcrowd antoniu_gugu9) and reviewed by the author. Every constant was re-derived independently (Section 9), checked numerically with `checks/lot_toy_check.py` (output: `checks/lot_toy_check.out`, about 1 min on one CPU core), and checked line by line against the graded estimator, `estimator/estimator.py` (Section 7). Measured numbers are quoted from the project's theory notes and logs and from the evidence ledger, with their ids (`docs/EVIDENCE.md`); they were re-read, not re-run.

# 0. Summary

LOT replaces the per-neuron fourth cumulant that the mean program reads by its leading-order (tree-level) value in a Gaussian backbone. Three objects carry it: one backbone functional per neuron,
$$\psi_i=\sum_{b<l}\sum_j V^b_{ij}\,\tilde\zeta_{b,j},\qquad V^b_{ij}=c^{(b)}_{2,j}\,P^{(l,b)}_{ij}\,\Gamma^{(l,b)}_{ij},$$
and two signed corrections. The identities are
$$D_{3,i}=3\operatorname{Cov}(L_i,\psi_i),\qquad \kappa^{\mathrm{chain}}_{4,i}=12\operatorname{Var}(\psi_i),\qquad \kappa^{\mathrm{chain}}_{4,i}\ \ge\ \frac{4}{3}\,\frac{D_{3,i}^2}{\mathrm{var}_i},$$
$$\kappa^{\mathrm{star}}_{4,i}=4\sum_{b,j}c^{(b)}_{3,j}P^{(l,b)}_{ij}\big(\Gamma^{(l,b)}_{ij}\big)^3,\qquad \kappa^{\mathrm{ind}}_{4,i}=12\sum_{b,j}V^b_{ij}\,\Xi^b_{ij}.$$
The deployed form keeps V29's per-layer level and takes only LOT's neuron-to-neuron shape, computed cheaply by projecting $\psi_i$ on the current layer's top-$K$ covariance modes.

| statement | verdict | where |
|---|---|---|
| $D_3=3\operatorname{Cov}(L,\psi)$ | confirmed (Wick count; generating function to $8\times10^{-15}$; MC ratio $0.9996\pm0.0008$) | 2.2-2.4, 9 |
| $\kappa_4^{\mathrm{chain}}=12\operatorname{Var}(\psi)$ | confirmed (same three routes; MC $0.9991\pm0.0017$) | 2.2-2.4, 9 |
| star factor 4 | confirmed (MC $0.9998\pm0.0018$; equals ARC's star count) | 2.3, 9 |
| induced factor 12 | confirmed (MC $0.9990\pm0.0023$; equals the 3+2 split, Prop. 2) | 2.3, 2.6, 9 |
| Cauchy-Schwarz floor $4/3$ | confirmed, with a scope restriction: tree-level chain only, not the total $\kappa_4$ | 3 |
| cross-source chains necessary | self-chains plus star fall to $-0.59$ at layer 15; self-chains alone 0.83 to 0.69; all chains plus star 0.89-0.95 | 4 |
| coincident index, 0.182 exact vs 0.318 | confirmed at $\alpha=0$; closed form for all $\alpha$; corrected same-hub weight | 5 |
| cost of the deployed form | +9.2 u and +362 calls were measured at $K=32$; $K=12$ was not metered separately | 6.5 |
| code correspondence | confirmed line by line (a numerical mirror of `_lot_g4` reproduces the formulas to $10^{-15}$; the identification of V29's $A$ leg is taken from V29's documentation) | 7 |

# 1. Setup and notation

The reader is assumed to know ARC's cumulant propagation (Wu et al. 2026, arXiv:2605.05179) and V29's factored chain.

## 1.1 Network and the quantity LOT supplies

Layers are 0-based as in the code. The input is $x\sim\mathcal N(0,I_n)$, $z_0=W_0^\top x$, $h_l=\mathrm{relu}(z_l)$, $z_l=W_l^\top h_{l-1}$ for $l=1,\dots,L-1$, with $n=1024$, $L=16$ and $(W_l)_{ai}\sim\mathcal N(0,2/n)$ (entry $a\to i$). The $i$-th column $w^{(l)}_i=W_l[:,i]$ holds the incoming weights of neuron $i$. Per layer the chain carries the mean $\mu_l$, the covariance $C_l$, the third-cumulant diagonal $D_{3,l,i}=\kappa_3(z_{l,i})$ and slice $D21$, and a fourth-cumulant diagonal $g_{l,i}\approx\kappa_4(z_{l,i})$. Let $\mathrm{var}_l=\mathrm{diag}(C_l)$, $\sigma=\sqrt{\mathrm{var}}$ and $\alpha=\mu/\sigma$. LOT supplies only $g_l$, and only to the mean program (Section 6). The toy script labels layers from 1; nothing else changes.

## 1.2 Hermite readout weights

For $\zeta\sim\mathcal N(\mu,\sigma^2)$ define $c_k=\mathbb E[\mathrm{relu}^{(k)}(\zeta)]$, with $\varphi,\Phi$ the standard normal density and distribution function:
$$c_0=\sigma\big(\varphi(\alpha)+\alpha\Phi(\alpha)\big),\qquad c_1=\Phi(\alpha),\qquad c_2=\frac{\varphi(\alpha)}{\sigma},$$
$$c_3=-\frac{\alpha\,\varphi(\alpha)}{\sigma^2},\qquad c_4=\frac{(\alpha^2-1)\,\varphi(\alpha)}{\sigma^3},$$
$$c_k=(-1)^k\,\mathrm{He}_{k-2}(\alpha)\,\frac{\varphi(\alpha)}{\sigma^{k-1}}\qquad(k\ge2).$$
Writing $\tilde\zeta=\zeta-\mu$ and the Wick power $:\!\tilde\zeta^k\!:=\sigma^k\,\mathrm{He}_k(\tilde\zeta/\sigma)$, the ReLU has the Hermite expansion
$$\mathrm{relu}(\zeta)=\sum_{k\ge0}\frac{c_k}{k!}\,:\!\tilde\zeta^k\!:.$$
In the code $c_k$ is V29's $w(k,1)$ (`WICK_PAIRS`, `WICK_UNIFIED`, lines 245-246; the toy checks $c_1,\dots,c_4$ by Stein's identity to $5\times10^{-16}$). The mean program is the Edgeworth readout
$$m=c_0+\frac{c_3}{3!}D_3+\frac{c_4}{4!}g+\frac{c_6}{72}D_3^2+\frac{c_7}{144}D_3\,g$$
(`TERM_SPECS[(1,)]`, line 247). It is affine in $g$, with weight
$$r_4=\frac{c_4}{24}+\frac{c_7\,D_3}{144}.$$
Notation warning: the write-up README's "$c_3$, $c_4$" in its readout formula are $c_3/3!$ and $c_4/4!$ here, and that formula omits the $c_7$ term (which does not change the linearity in $g$).

## 1.3 The Gaussian backbone

Let $m_l=c_0(\mu_l,\sigma_l)$ and let $\Sigma^h_l$ be the covariance of $h_l$ that the chain uses. In the pure Gaussian closure (as in the toy), Mehler's formula gives $\Sigma^h_{l,ij}=\sum_{k\ge1}c_{k,i}c_{k,j}(C_{l,ij})^k/k!$. The backbone is the linear-Gaussian chain
$$\zeta_0=W_0^\top x,\qquad \zeta_l=\mu_l+W_l^\top\Big[\mathrm{diag}\big(c^{(l-1)}_1\big)\,\tilde\zeta_{l-1}+\eta_{l-1}\Big],$$
$$\eta_{l-1}\sim\mathcal N\Big(0,\ \Sigma^h_{l-1}-\mathrm{diag}\big(c^{(l-1)}_1\big)\,C_{l-1}\,\mathrm{diag}\big(c^{(l-1)}_1\big)\Big),$$
with the innovations $\eta$ independent of everything earlier. In the pure closure the innovation covariance is positive semidefinite: it is the Mehler sum over $k\ge2$, and each term is a Hadamard product of positive semidefinite matrices (Schur product theorem). With the chain's corrected $\Sigma^h_l$ this is an assumption, safe while the corrections are small. Define the propagators
$$P^{(b+1,b)}=W_{b+1}^\top,\qquad P^{(l,b)}=W_l^\top\,\mathrm{diag}\big(c^{(l-1)}_1\big)\,P^{(l-1,b)}\qquad(b<l),$$
the linearized map from $h_b$ to $z_l$. Then the backbone has exactly the chain's per-layer covariances and linear-response cross-covariances:
$$\operatorname{Cov}(\zeta_l,\zeta_l)=C_l,\qquad G(l,b):=\operatorname{Cov}(\zeta_l,\zeta_b)=\Gamma^{(l,b)},$$
$$\Gamma^{(l,b)}:=P^{(l,b)}\,\mathrm{diag}\big(c^{(b)}_1\big)\,C_b\qquad(b<l).$$
The cross-covariance is Stein's lemma iterated: for jointly Gaussian pre-activations, $\operatorname{Cov}(\mathrm{relu}(\zeta_j),\zeta_k)=c_{1,j}\operatorname{Cov}(\zeta_j,\zeta_k)$. The toy's sampler reproduces these blocks to Monte Carlo accuracy. In the code $\mu_l$ and $C_l$ are the chain's own values, which already include its $\kappa_3$/$\kappa_4$ corrections to the first two moments; this is how first-chaos ("tadpole") corrections are absorbed (Section 2.5).

## 1.4 Per-source objects at readout layer $l$

A source is a birth layer $b<l$ (the ReLU of layer $b$); a hub is a pair $(b,j)$. At readout layer $l$ and neuron $i$:

* $L_i=\tilde\zeta_{l,i}$, the first-chaos (Gaussian) part of $z_{l,i}$, with $\operatorname{Var}(L_i)=\mathrm{var}_{l,i}$ and $\operatorname{Cov}(L_i,\tilde\zeta_{b,j})=\Gamma^{(l,b)}_{ij}$.
* The He2 part $Q_i=\sum_{b<l}\sum_j\tfrac12 c^{(b)}_{2,j}P^{(l,b)}_{ij}\,:\!\tilde\zeta_{b,j}^2\!:$, the He3 part $T^*_i=\sum_{b<l}\sum_j\tfrac16 c^{(b)}_{3,j}P^{(l,b)}_{ij}\,:\!\tilde\zeta_{b,j}^3\!:$, and the nested part $T^{\mathrm{ind}}_i$ (Section 2.3 (d)).
* The Hadamard leg $V^b=\big(P^{(l,b)}\circ\Gamma^{(l,b)}\big)\,\mathrm{diag}\big(c^{(b)}_2\big)$, i.e. $V^b_{ij}=c^{(b)}_{2,j}P^{(l,b)}_{ij}\Gamma^{(l,b)}_{ij}$.
* The functional $\psi_i=\sum_{b<l}\sum_jV^b_{ij}\tilde\zeta_{b,j}$ and its variance $\operatorname{Var}(\psi_i)=\sum_{b,b'<l}\big[V^b\,G(b,b')\,V^{b'\top}\big]_{ii}$.
* The induced kernel $\Xi^b=\sum_{b'<b}\big(\Gamma^{(l,b')}\circ\Gamma^{(l,b')}\big)\,\mathrm{diag}\big(c^{(b')}_2\big)\,P^{(b,b')\top}$, i.e. $\Xi^b_{ij}=\sum_{b'<b}\sum_k c^{(b')}_{2,k}P^{(b,b')}_{jk}\big(\Gamma^{(l,b')}_{ik}\big)^2$.

Correspondence with V29's legs: V29's $A$ leg is $A^{(l,b)}=P^{(l,b)}\mathrm{diag}(c^{(b)}_1)C^{\mathrm{off}}_b$, so $\Gamma^{(l,b)}=A^{(l,b)}+P^{(l,b)}\mathrm{diag}(c^{(b)}_1\mathrm{var}_b)$, and $V^b$ is V29's $(A\circ P)\,w_2$ Hadamard product plus its diagonal part. In the code, $\Gamma$ is `T3`, $P\circ\Gamma$ is `T1`, and $V$ is `T2` (Section 7).

# 2. The leading-order-tree reduction

## 2.1 Tree expansion and the excess-leg count

Expand every ReLU of the network in its Hermite series around the backbone marginal of its own pre-activation. Then $z_{l,i}$ is a sum of rooted trees. A vertex of order $k$ at hub $(b,j)$ carries the weight $c^{(b)}_{k,j}/k!$, has one output leg (to layer $l$ through $P^{(l,b)}$, or into a later vertex through $P^{(b',b)}$), and has $k$ input legs. Each input leg is either a leaf $\tilde\zeta_{b,j}$ or the output of a vertex of an earlier layer. Joint cumulants of $z_{l,i}$ are sums over connected Gaussian pairings of leaves; pairing two leaves gives the backbone covariance $G(b,b')_{jj'}$, and two legs of the same vertex are never paired with each other (Wick ordering).

Let the excess of a diagram be $e=\sum_v(k_v-1)$. A diagram for $\kappa_r$ (with $r$ slots) and excess $e$ has $r+e$ leaves and hence $(r+e)/2$ pairings. Connecting the $r$ slot trees needs at least $r-1$ pairings between different trees, so
$$e\ \ge\ r-2.$$
Tree level is $e=r-2$: the slot trees and the pairings then form a spanning tree with no loop. LOT keeps exactly these diagrams. For $\kappa_3$ ($e=1$) there is one He2 vertex. For $\kappa_4$ ($e=2$) there are exactly three configurations: (S) one He3 vertex in one slot; (C) one He2 vertex in each of two slots; (I) one He2 vertex whose input is the output of another He2 vertex, in one slot. Nothing else qualifies: a configuration with $e=1$ has an odd number of leaves when $r=4$, and two vertices in one slot must be nested, because each term of a slot is a single rooted tree.

## 2.2 Statement

**Proposition 1 (tree-level cumulants).** With the definitions of Section 1.4,
$$D_{3,i}\;=\;3\,\kappa(L_i,L_i,Q_i)\;=\;3\sum_{b<l}\sum_jV^b_{ij}\,\Gamma^{(l,b)}_{ij}\;=\;3\operatorname{Cov}(L_i,\psi_i),$$
$$\kappa^{\mathrm{chain}}_{4,i}\;=\;6\,\kappa(L_i,L_i,Q_i,Q_i)\;=\;12\operatorname{Var}(\psi_i)\;=\;12\sum_{b,b'<l}\big[V^b\,G(b,b')\,V^{b'\top}\big]_{ii},$$
$$\kappa^{\mathrm{star}}_{4,i}\;=\;4\,\kappa(L_i,L_i,L_i,T^*_i)\;=\;4\sum_{b<l}\sum_jc^{(b)}_{3,j}\,P^{(l,b)}_{ij}\,\big(\Gamma^{(l,b)}_{ij}\big)^3,$$
$$\kappa^{\mathrm{ind}}_{4,i}\;=\;4\,\kappa(L_i,L_i,L_i,T^{\mathrm{ind}}_i)\;=\;12\sum_{b<l}\sum_jV^b_{ij}\,\Xi^b_{ij},\qquad \Xi^b_{ij}=\kappa\big(L_i,L_i,z_{b,j}\big)\big|_{\mathrm{tree}}.$$
The tree-level fourth cumulant is $\kappa^{\mathrm{LOT}}_{4,i}=\kappa^{\mathrm{chain}}_{4,i}+\kappa^{\mathrm{star}}_{4,i}+\kappa^{\mathrm{ind}}_{4,i}$.

## 2.3 Proof

**Lemma A (first-chaos contraction).** For every first-chaos (linear) variable $Y$, $\mathbb E[L_iYQ_i]=\operatorname{Cov}(Y,\psi_i)$. Equivalently, $\psi_i-\mathbb E\psi_i$ is the first-chaos projection of $L_iQ_i$.

*Proof.* By Wick's theorem $\mathbb E\big[L_iY:\!\tilde\zeta_{b,j}^2\!:\big]=2\operatorname{Cov}(L_i,\tilde\zeta_{b,j})\operatorname{Cov}(Y,\tilde\zeta_{b,j})$: the two legs of the vertex pair with $L_i$ and $Y$ in two ways. Multiply by $\tfrac12c^{(b)}_{2,j}P^{(l,b)}_{ij}$ and sum over hubs: $\sum_{b,j}c^{(b)}_{2,j}P^{(l,b)}_{ij}\Gamma^{(l,b)}_{ij}\operatorname{Cov}(Y,\tilde\zeta_{b,j})=\operatorname{Cov}(Y,\psi_i)$. $\square$

(a) $\kappa_3$. All variables are centred, so $\kappa(L,L,Q)=\mathbb E[LLQ]=\operatorname{Cov}(L,\psi)$ by Lemma A with $Y=L$. $\kappa(L,L,L)=0$, and diagrams with two He2 vertices have $e=2$. The He2 vertex can sit in any of the three slots, so $D_3=3\operatorname{Cov}(L,\psi)$.

(b) Chain. $\kappa(L,L,Q,Q)=\mathbb E[L^2Q^2]-\mathbb E[L^2]\mathbb E[Q^2]-2\mathbb E[LQ]^2$, and $\mathbb E[LQ]=0$ (orthogonal chaoses). The subtracted product removes the disconnected pairing ($L$ with $L$, $Q$ with $Q$). A connected pairing of the six leaves must pair each $L$ with a leg of a different $Q$ and the two remaining $Q$ legs with each other; pairing both $L$'s into the same $Q$ would force a self-pairing of the other vertex. For a hub $(b,j)$ in slot 3 and a hub $(b',j')$ in slot 4 there are $2\times2\times2=8$ such pairings (which leg of each vertex goes to an $L$, and which $L$ goes to which slot), each worth $\Gamma^{(l,b)}_{ij}\Gamma^{(l,b')}_{ij'}G(b,b')_{jj'}$. Hence
$$\kappa(L_i,L_i,Q_i,Q_i)=8\sum_{(b,j),(b',j')}\Big(\tfrac12c^{(b)}_{2,j}P^{(l,b)}_{ij}\,\Gamma^{(l,b)}_{ij}\Big)\,G(b,b')_{jj'}\,\Big(\tfrac12c^{(b')}_{2,j'}P^{(l,b')}_{ij'}\,\Gamma^{(l,b')}_{ij'}\Big)$$
$$=2\sum_{(b,j),(b',j')}V^b_{ij}\,G(b,b')_{jj'}\,V^{b'}_{ij'}=2\operatorname{Var}(\psi_i).$$
The two He2 vertices occupy two of the four slots in $\binom42=6$ ways, so $\kappa^{\mathrm{chain}}_4=12\operatorname{Var}(\psi)$. The sum includes the same hub, $(b,j)=(b',j')$; see Section 5.

(c) Star. $\kappa(L,L,L,T^*)=\mathbb E[L^3T^*]-3\mathbb E[L^2]\mathbb E[LT^*]$, and $\mathbb E[LT^*]=0$ (chaos 1 against chaos 3). $\mathbb E\big[L_i^3:\!\tilde\zeta_{b,j}^3\!:\big]=3!\,\big(\Gamma^{(l,b)}_{ij}\big)^3$, so $\kappa=\sum_{b,j}\tfrac16c_3P\cdot6\,\Gamma^3=\sum c_3P\Gamma^3$. The He3 vertex can sit in any of four slots, giving the factor 4.

(d) Induced. The He2 vertex at $(b,j)$ acts on the full fluctuation $\tilde z_{b,j}=\tilde\zeta_{b,j}+Q^{(b)}_j+\dots$, where $Q^{(b)}_j=\sum_{b'<b}\sum_k\tfrac12c^{(b')}_{2,k}P^{(b,b')}_{jk}:\!\tilde\zeta_{b',k}^2\!:$ is the He2 part of $z_{b,j}$ (the linear term $c_1Q^{(b)}$ is already inside $P^{(l,b')}$). The $e=2$ part of $\tfrac12c_2\tilde z^2$ is the cross term $c_2\,\tilde\zeta_{b,j}Q^{(b)}_j$, so
$$T^{\mathrm{ind}}_i=\sum_{b<l}\sum_jc^{(b)}_{2,j}P^{(l,b)}_{ij}\,\tilde\zeta_{b,j}\,Q^{(b)}_j .$$
In $\kappa(L,L,L,T^{\mathrm{ind}})=\mathbb E[L^3T^{\mathrm{ind}}]-3\mathbb E[L^2]\mathbb E[LT^{\mathrm{ind}}]$ the subtraction removes the contraction of $\tilde\zeta_{b,j}$ with a leg of the inner vertex (a first-chaos piece). The connected part pairs the three leaves (one outer, two inner) with the three $L$'s in $3!=6$ ways, each worth $\Gamma^{(l,b)}_{ij}\big(\Gamma^{(l,b')}_{ik}\big)^2$:
$$\kappa(L_i,L_i,L_i,T^{\mathrm{ind}}_i)=\sum_{b,j}c^{(b)}_{2,j}P^{(l,b)}_{ij}\Gamma^{(l,b)}_{ij}\sum_{b'<b,\,k}\tfrac12c^{(b')}_{2,k}P^{(b,b')}_{jk}\cdot6\big(\Gamma^{(l,b')}_{ik}\big)^2$$
$$=3\sum_{b,j}V^b_{ij}\,\Xi^b_{ij}.$$
Four slots give 12. The same count as in (a) shows $\Xi^b_{ij}=\kappa(L_i,L_i,Q^{(b)}_j)=\kappa_3(L_i,L_i,z_{b,j})$ at tree level. $\square$

## 2.4 An independent check: the generating function of a Gaussian quadratic form

For one readout neuron, $L_i$ and $Q_i$ are a linear and a quadratic form in the jointly Gaussian backbone vector $y=(\tilde\zeta_0,\dots,\tilde\zeta_l)\sim\mathcal N(0,G)$. Whiten, $y=G^{1/2}\xi$, so that $L_i=a^\top\xi$ and $Q_i=\xi^\top M\xi-\mathrm{tr}\,M$, with $a=G^{1/2}e_{l,i}$, $M=G^{1/2}D_iG^{1/2}$ and $D_i=\mathrm{diag}_{(b,j)}\big(\tfrac12c^{(b)}_{2,j}P^{(l,b)}_{ij}\big)$. For $z(\varepsilon)=L_i+\varepsilon Q_i$ the cumulant generating function is exact,
$$K(t)=-\varepsilon t\,\mathrm{tr}M-\tfrac12\log\det(I-2\varepsilon tM)+\tfrac12t^2\,a^\top(I-2\varepsilon tM)^{-1}a,$$
and gives
$$\kappa_3=6\varepsilon\,a^\top Ma+8\varepsilon^3\,\mathrm{tr}M^3,\qquad \kappa_4=48\varepsilon^2\,a^\top M^2a+48\varepsilon^4\,\mathrm{tr}M^4.$$
In these coordinates $\psi_i=2a^\top M\xi$, so $6a^\top Ma=3\operatorname{Cov}(L,\psi)$ and $48a^\top M^2a=12\operatorname{Var}(\psi)$. The terms in $a$ are the trees; $8\,\mathrm{tr}M^3$ and $48\,\mathrm{tr}M^4$ are the loops that tree level drops. The toy confirms both tree identities to $8\times10^{-15}$ (relative). At width 8 the loops are not small (mean $|8\,\mathrm{tr}M^3|/|D_3|=0.71$, $48\,\mathrm{tr}M^4/\kappa^{\mathrm{chain}}_4=0.70$): LOT is a large-width statement.

## 2.5 What is kept, what is dropped, and the assumptions

* **Kept:** the $\kappa_3$ tree (one He2 vertex) and the three $\kappa_4$ trees of Section 2.1, for every pair of birth layers $b,b'<l$ (including $b\ne b'$; Section 4), with free index sums (coincident hubs included, with He2/He3 weights; Section 5).
* **Dropped:** loops ($e>r-2$), for example $8\,\mathrm{tr}M^3$ in $\kappa_3$ and $48\,\mathrm{tr}M^4$ in $\kappa_4$; vertices of order 4 and higher; diagrams with three or more vertices; diagrams that attach an existing $\kappa_4$ block to a He2 vertex; and every off-diagonal slice of $\kappa_4$. The mean program reads only $\kappa_{4,iiii}$; V29's own (2,2) and (3,1) slices stay in the covariance update, because inserting LOT-type slices there was destructive in the lean runs.
* **(A1) Backbone.** Pre-activations are expanded around a jointly Gaussian law with the chain's own $(\mu_l,C_l)$ and linear-response cross-covariances (Section 1.3). First-chaos corrections (contractions between different vertices of one tree) renormalize means and covariances and are assumed to be carried by the chain's $\mu,C$. The second moment of $Q$, which is $O(1)$, is never used; the expansion is used only for $\kappa_3$ and $\kappa_4$.
* **(A2) Order counting.** Dropping loops is justified by the large-width counting in which each pairing between legs on *different* neurons costs an off-diagonal covariance or propagator element of rms $O(n^{-1/2})$. ARC prove the corresponding statement in expectation over the weights (Corollary S.2.14 of arXiv:2605.05179: $\mathbb E|\kappa_r(i_1,\dots,i_r)|^2=O(n^{2-r})$ when every index value has even multiplicity and $r$ is even, $O(n^{1-r})$ otherwise). For one fixed network this is a heuristic: the hub sums are random-sign sums whose coherent part comes from the covariance gain modes. LOT was therefore validated empirically (Sections 4-5).
* **(A3) Coincident indices.** When two vertices sit on the same neuron, the extra contractions are same-neuron variances of size $O(1)$, so (A2) fails there and every Hermite order of that neuron contributes at the same order (Section 5).

## 2.6 The same trees, organized by birth layer

ARC's factored $K=4$ step builds the $\kappa_4$ born at one ReLU from three layer-local trees: path (count 12), star (4) and 3+2 (12) (`factor_k4.py`, lines 487, 497, 506). Kuznetsov re-derived the same three trees (forum 18249, Section 7.1):
$$\text{path: }12\,d_1Cd_2Cd_2Cd_1,\qquad \text{star: }4\,d_1C\,d_3\,(Cd_1)^2,$$
$$\text{3+2: }12\,d_1(i)\,C(i,j)\,d_2(j)\,\kappa^{\mathrm{pre}}_3(j,k,m)\,d_1(k)\,d_1(m),$$
with $d_k=c^{(b)}_k$ and $C=C_b$.

**Proposition 2.** Transport the trees born at layer $b$ to layer $l$, read them on the diagonal, and use the tree-level $\kappa^{\mathrm{pre}}_3(z_b)$ (the "cherry" $\sum_{b'<b}\sum_pc^{(b')}_{2,p}\,[P\Gamma\Gamma+\Gamma P\Gamma+\Gamma\Gamma P]$). Then
$$\mathrm{path}_b\to12\big[V^bC_bV^{b\top}\big]_{ii},\qquad \mathrm{star}_b\to4\sum_jc^{(b)}_{3,j}P^{(l,b)}_{ij}\big(\Gamma^{(l,b)}_{ij}\big)^3,$$
$$(3{+}2)_b\to12\sum_jV^b_{ij}\Xi^b_{ij}\;+\;24\sum_{b'<b}\big[V^b\,G(b,b')\,V^{b'\top}\big]_{ii}.$$
Summed over $b$, path + star + 3+2 equals chain + star + induced.

*Proof sketch.* On the diagonal all 12 labelled 3+2 terms are equal, which gives $12\sum_jV^b_{ij}D^x_{ij}$ with $D^x_{ij}=\kappa_3(z_{b,j},L_i,L_i)$ (Kuznetsov's cross-layer pairing). The cherry's propagator leg can sit in three places. On the hub, it gives $\Xi^b_{ij}$. On either readout slot, it gives $\Gamma^{(b,b')}_{jp}P^{(l,b')}_{ip}\Gamma^{(l,b')}_{ip}$ twice, i.e. $2[G(b,b')V^{b'\top}]_{ji}$. The path is the self-chain by the same contraction. $\square$

The toy confirms Proposition 2 to $3\times10^{-15}$; the lean-gate notes derived the same split for the youngest generation (coefficients 12 and 24). Consequences: (i) the cross-source chains are not a new diagram; they are the part of the 3+2 tree whose third cumulant enters the $d_2$ node through an input leg; (ii) the induced term is the part entering through the output leg; (iii) LOT's contribution is the closed form that sums all generations on the diagonal, not the diagram set.

# 3. The Cauchy-Schwarz floor

**Proposition 3.** With $\mathrm{var}_i=\operatorname{Var}(L_i)$ and $\rho_i=\operatorname{Corr}(L_i,\psi_i)$,
$$\kappa^{\mathrm{chain}}_{4,i}=\frac{4}{3}\,\frac{D_{3,i}^2}{\mathrm{var}_i\,\rho_i^2}\ \ge\ \frac{4}{3}\,\frac{D_{3,i}^2}{\mathrm{var}_i},$$
with equality if and only if $\psi_i$ is almost surely affine in $L_i$.

*Proof.* $\operatorname{Cov}(L,\psi)^2=\rho^2\operatorname{Var}(L)\operatorname{Var}(\psi)\le\operatorname{Var}(L)\operatorname{Var}(\psi)$. Substitute $\operatorname{Cov}=D_3/3$ and $\operatorname{Var}(\psi)=\kappa^{\mathrm{chain}}_4/12$: $\kappa^{\mathrm{chain}}_4=12\,(D_3/3)^2/(\mathrm{var}\,\rho^2)=\tfrac43D_3^2/(\mathrm{var}\,\rho^2)$. $\square$

Equality case (tight): for $z=u+\varepsilon\,\mathrm{He}_2(u)$ with $u\sim\mathcal N(0,1)$, $D_3=6\varepsilon$, $\kappa^{\mathrm{chain}}_4=48\varepsilon^2=\tfrac43(6\varepsilon)^2$. Equality holds whenever $\psi_i$ lies along $L_i$ (a single coherent hub direction).

**Scope (what the bound does not say).**

* It bounds the tree-level chain by the tree-level $D_3$. It is not a bound on the total $\kappa_4$: the star and induced terms are signed ($c_3$ changes sign with $\alpha$), and the loops add terms of both signs to $\kappa_3$.
* It does not hold automatically for a chain's actual $D_3$, which contains more than the tree.
* Using the chain's full $\mathrm{var}_i$ for $\operatorname{Var}(L_i)$ is valid. Using only the linear-response part of the variance gives a tighter bound.
* For comparison, Pearson's inequality $\kappa_4\ge\kappa_3^2/\mathrm{var}-2\,\mathrm{var}^2$ holds for every distribution but is vacuous in the perturbative regime. Proposition 3 is a structural statement at leading order, with a larger coefficient and no $-2\,\mathrm{var}^2$.

**Use as a diagnostic.**

* $\rho_i^2=\tfrac43D_{3,i}^2/(\mathrm{var}_i\,\kappa^{\mathrm{chain}}_{4,i})\in[0,1]$ measures how much of the chain lies along the neuron's own linear direction.
* A closure of the form $\kappa_4\mathrel{+}=b\,\kappa_3^2/\mathrm{var}$ (the `k3sq` closure of the channels study, $-1.6\%$ held-out) is the constant-$\rho$ shadow of the chain.
* Luna's write-up (18281 Section 3.7, "chaos model", derivation LLM-assisted, posted 24 Oct) works with $z_a=L_a\cdot g+\tfrac12(g^\top B_ag-\mathrm{tr}B_a)$, which is $L+Q$ in whitened form. In its aligned case, $B_aL_a\parallel L_a$, it obtains the diagonal $\tfrac43\tau_a^2\mathrm{var}_a$ with $\tau=T_3/\mathrm{var}$, i.e. $\tfrac43\kappa_3^2/\mathrm{var}$; Luna ship a fitted constant of 1.33. That is the equality case $\rho=1$ of Proposition 3. The general identities and the inequality are not stated there.
* Measured at $1024\times16$ (MLP 0, MC truth $N=2\times10^6$, the theory log of MLP 0, tree-level $D_3$ and chain variance), the floor $\tfrac43D_3^2/\mathrm{var}$ has attenuation-corrected correlation with the true $\kappa_4$ of 0.40 (layer 4), 0.57 (6), 0.62 (8), 0.71 (10), 0.83 (12), 0.88 (13), 0.85 (14) and 0.87 (15). The slope of truth on floor is 11.1, 6.3, 4.6, 3.0, 2.1, 1.8, 1.6, 1.6, so $\rho^2$ grows with depth. The floor's shape becomes informative at depth, but its level stays a fraction of the truth.
* The floor's correlation is therefore low only in early layers (0.33-0.40 at layers 4-5) and high where the error is born.
* A consistency check for any estimator that carries separate $\kappa_3$ and chain-type $\kappa_4$ terms: per neuron, the chain part should not fall below $\tfrac43D_3^2/\mathrm{var}$ of its own tree-level $D_3$.
* In the toy (width 8), the minimum relative slack is 0.35 and $|\rho_i|\le0.81$.

# 4. Why the cross-source chains are necessary

$\operatorname{Var}(\psi_i)$ splits into self-chains $\sum_b[V^bC_bV^{b\top}]_{ii}$, whose two hubs share a birth layer, and cross-source chains $\sum_{b\ne b'}[V^bG(b,b')V^{b'\top}]_{ii}$, whose hubs at different birth layers are coupled through the backbone cross-covariance $G(b,b')=\Gamma^{(b,b')}$. A readout at layer $l$ has $l$ self pairs and $l(l-1)$ ordered cross pairs.

Measured at $1024\times16$ (MLP 0, attenuation-corrected correlation with the MC $\kappa_4$, $N=2\times10^6$; the theory log of MLP 0, rows `self-chain`, `self+star`, `chain`, `chain+star`, `He2-LOT`):

| layer | self-chains | self-chains + star | all chains | all chains + star | chain + star + induced |
|---|---|---|---|---|---|
| 4 | 0.829 | 0.842 | 0.887 | 0.946 | 0.952 |
| 6 | 0.783 | 0.708 | 0.857 | 0.929 | 0.943 |
| 8 | 0.742 | 0.574 | 0.833 | 0.923 | 0.938 |
| 10 | 0.730 | 0.186 | 0.841 | 0.917 | 0.939 |
| 12 | 0.702 | $-0.258$ | 0.852 | 0.909 | 0.941 |
| 13 | 0.722 | $-0.476$ | 0.879 | 0.925 | 0.953 |
| 14 | 0.707 | $-0.561$ | 0.864 | 0.911 | 0.941 |
| 15 | 0.694 | $-0.593$ | 0.856 | 0.894 | 0.936 |

**Self-chains and the star.** With the star included, as in every deployed form, keeping only the self-chains turns the shape from correct (0.89 at layer 15) into anti-correlated ($-0.59$). Without the star, self-chains alone track the truth (0.69-0.83, never negative) but much worse than all chains with the star (0.89-0.95). The level needs the star as well: all chains alone have slope (truth on prediction) 0.65 falling to 0.49, while all chains + star have slope 0.97-1.08.

**Toy confirmation** (true network, width 32, depth 6, 4 networks, `lot_toy_check.out` Part C; correlation with the MC $\kappa_4$ averaged over networks): self-chains + star give 0.79, 0.80, 0.62 and 0.38 at layers 3-6, while all chains + star give 0.91, 0.93, 0.90 and 0.93. The degradation with depth appears already at toy scale.

**Mechanism (hypothesis, consistent with the data, not isolated).** The same input fluctuation is seen by hubs at many birth layers through the persistent covariance gain modes. The $\psi^{(b)}_i$ of different sources are therefore strongly correlated, and the cross terms dominate the neuron-to-neuron variation of $\operatorname{Var}(\psi_i)$ once $l$ is large. By Proposition 2 this is the same physics as Kuznetsov's finding that "without the 3+2 graph the young $\kappa_4$ is worse than none" (18249 Section 7.1).

# 5. The coincident-index caveat

## 5.1 One neuron, exactly

Let $z\sim\mathcal N(\alpha\sigma,\sigma^2)$ and $h=\mathrm{relu}(z)$. The joint cumulant $\kappa(z,z,h,h)$ is the coincident-hub configuration of the chain (two slots through the hub's nonlinearity, two through its pre-activation). From truncated-normal moments (derived here; the toy checks it on $[-3,3]$ to $3\times10^{-15}$):
$$\frac{\kappa(z,z,h,h)}{\sigma^4}=2\Phi(\alpha)\big(1-\Phi(\alpha)\big)-2\varphi(\alpha)^2-2\alpha\,\Phi(\alpha)\,\varphi(\alpha).$$
At $\alpha=0$ this is $\tfrac12-\tfrac1\pi=0.181690$, while the He2 truncation gives $2c_2^2\sigma^6/\sigma^4=2\varphi(0)^2=1/\pi=0.318310$. In Hermite form, with $\gamma_k=c_k\sigma^{k-1}$,
$$\frac{\kappa(z,z,h,h)}{\sigma^4}=\sum_{m\ge2}\frac{2m\,\gamma_m^2}{m!}+2\sum_{m\ge1}\frac{\gamma_m\gamma_{m+2}}{m!},$$
whose leading terms are $2\gamma_2^2$ (the He2 chain), $2\gamma_1\gamma_3$ (zero at $\alpha=0$), $\gamma_2\gamma_4=-0.159155$ at $\alpha=0$, $\gamma_3^2$, $\gamma_4^2/3,\dots$. The series converges slowly to the closed form (0.212 at $m\le4$, 0.184 at $m\le20$, 0.18179 at $m\le140$); Monte Carlo gives 0.1811 ($N=4\times10^6$).

For $\alpha\ne0$ the correct comparator is not $2\gamma_2^2$ alone. In a one-hub model $z=a\tilde\zeta+p\,\tilde h(\zeta)+\xi$, LOT's tree assigns the order-$a^2p^2$ coefficient $6\sigma^4(2\gamma_2^2+2\gamma_1\gamma_3)$: the He2 chain plus the coincident part of the He3 star, which arises because $\Gamma$ includes the hub's own linear path $pc_1\sigma^2$. The exact coefficient is $6\,\kappa(\zeta,\zeta,h,h)$. Order $a^3p$ is exact, since $\kappa(\zeta,\zeta,\zeta,h)=c_3\sigma^6$ by Stein. From the toy (Part A, $\sigma=1$):

| $\alpha$ | exact | He2 $2\gamma_2^2$ | tree $2\gamma_2^2+2\gamma_1\gamma_3$ | exact $-$ tree |
|---|---|---|---|---|
| $-1.5$ | 0.1171 | 0.0336 | 0.0595 | $+0.0576$ |
| $-1.0$ | 0.2267 | 0.1171 | 0.1939 | $+0.0328$ |
| $-0.5$ | 0.2874 | 0.2479 | 0.3565 | $-0.0691$ |
| 0 | 0.1817 | 0.3183 | 0.3183 | $-0.1366$ |
| 0.5 | $-0.0647$ | 0.2479 | 0.0045 | $-0.0691$ |
| 1.0 | $-0.2573$ | 0.1171 | $-0.2901$ | $+0.0328$ |
| 1.5 | $-0.2715$ | 0.0336 | $-0.3291$ | $+0.0576$ |

## 5.2 Why, and which terms are affected

The order counting (A2) charges $O(n^{-1/2})$ for every pairing between legs on distinct neurons. A pairing between legs on the same neuron is a variance, of size $O(1)$, so diagrams in which two vertices share a neuron are not suppressed and must be summed over all Hermite orders of that neuron. In LOT this affects:

* the same-hub part of the chain ($b=b'$, $j=j'$), which is coherent ($V^2\ge0$) and therefore not small;
* the self-parts of $\Gamma$ (the hub's own linear path) inside $D_3$, the star and the induced term, which enter as incoherent random-sign sums;
* the single-hub classes of order $ap^3$ and $p^4$ (the (3,1)- and (4,)-type local classes), which are absent from the tree altogether.

The exact difference is $6\sigma^4\big[2\Phi(1-\Phi)-4\varphi^2\big]$ at order $a^2p^2$.

## 5.3 Corrected statement for the same-hub term

At order $a^2p^2$ the coincident configuration is exact if the He2 weight $\varphi(\alpha)^2$ of the same-hub term is replaced by $\Phi(\alpha)(1-\Phi(\alpha))-\varphi(\alpha)^2$:
$$\kappa^{\mathrm{same}}_{4,i}\Big|_{\mathrm{He2}}=12\sum_{b,j}\big(V^b_{ij}\big)^2\mathrm{var}_{b,j}=12\sum_{b,j}\big(P^{(l,b)}_{ij}\Gamma^{(l,b)}_{ij}\big)^2\varphi(\alpha_{b,j})^2,$$
$$\kappa^{\mathrm{same}}_{4,i}\Big|_{\mathrm{exact}}=12\sum_{b,j}\big(P^{(l,b)}_{ij}\Gamma^{(l,b)}_{ij}\big)^2\Big[\Phi(\alpha_{b,j})\big(1-\Phi(\alpha_{b,j})\big)-\varphi(\alpha_{b,j})^2\Big].$$
The ratio $r(\alpha)=\Phi(1-\Phi)/\varphi^2-1$ is even in $\alpha$. It is at least $\pi/2-1=0.571$ (attained at $\alpha=0$), equals 0.721 at $|\alpha|=0.5$, 1.280 at $|\alpha|=1$, 2.717 at $|\alpha|=1.5$ and 6.63 at $|\alpha|=2$, and crosses 1 near $|\alpha|\approx0.8$. The He2 same-hub term is therefore too large for kink neurons, where the mean program's weight $c_4$ is largest, and too small in the tails. Scope: the derivation assumes that the hub's coupling to $L_i$ is jointly Gaussian with it and is dominated by other neurons ($a\gg p$), and it leaves the order-$ap^3$ and order-$p^4$ local classes uncorrected.

Toy test (true network, width 32, depth 6, 4 networks; Part C, last column). Replacing the He2 same-hub term by the corrected one fixes the level of LOT: the slope of truth on prediction becomes 1.00, 0.92, 0.94 and 1.05 at layers 3-6, against 0.70, 0.75, 0.82 and 0.94 for He2 LOT. It raises the correlation at layers 3, 4 and 6 (0.953, 0.934, 0.950 against 0.925, 0.927, 0.946) and lowers it at layer 5 (0.902 against 0.923). This was not tested at $1024\times16$ or in the chain. Two things are known at full width. First, the (4,), (3,1), (2,2) and (2,1,1) local classes are small at depth (rms $1.8\times10^{-4}$, $0.9\times10^{-4}$, $3.3\times10^{-4}$ and $1.8\times10^{-4}$ against $5.8\times10^{-3}$ for the chain at layer 15). Second, the He2 same-hub term is not small (rms $3.5\times10^{-3}$ at layer 15). So the order-$a^2p^2$ correction is the relevant one there. In the production form the level is overridden by V29's table, but the shape still carries the He2 same-hub weights. A related observation, suggestive only because the features are collinear: the learned HZ correction gives the He2 same-hub feature a fitted weight of 0.29-0.49 per layer in G4 (`figures/data/hz_theta_two_fits.csv`).

# 6. The deployable form

## 6.1 Projection on the current layer's top-$K$ modes

The exact chain needs a backward accumulation per readout layer, $\Phi^a=V^a+\Phi^{a+1}M_{a+1}$, i.e. $O(l)$ dense $n\times n$ products for readout layer $l$. The theory notes estimate about 210 u for the exact chain + star and about 315 u with the induced term (a coarser count gives 240-360 u). These are estimates, not metered runs, but either way the exact form is not shippable. The deployed form instead projects $\psi_i$ onto the span of the current layer's dominant backbone modes. Let $Q\in\mathbb R^{n\times K}$ be a basis of a $K$-dimensional subspace and $G_q=Q^\top C_lQ$. The variance of the $L^2$ projection of $\psi_i$ onto $\mathrm{span}(Q^\top\tilde\zeta_l)$ is
$$\operatorname{Var}\big(\Pi_K\psi_i\big)=k_i^\top G_q^{-1}k_i,\qquad k_i=Q^\top\operatorname{Cov}(\zeta_l,\psi_i)=\sum_{b<l}\big[V^b\,\big(\Gamma^{(l,b)\top}Q\big)\big]_{i,:}^\top .$$
Properties:

* The formula depends on $\mathrm{span}(Q)$ only (the toy checks $Q\to QR$ to $10^{-15}$), so the basis need not be orthonormal eigenvectors.
* $\operatorname{Var}(\Pi_K\psi)\le\operatorname{Var}(\psi)$.
* Because $k_i$ is summed over sources before it is squared, every cross-source pair is included at cost $n^2K$ per source.
* In production $Q$ comes from a one-pass range finder: $Q=\mathrm{qr}(C_l\Omega)$ with the fixed sketch $\Omega$ equal to $K$ weight columns gathered on the active set and one QR pass (`PORTLOT_Q=1`). At the trimmed last layer, $C_l\Omega$ is formed as $(WC^{\mathrm{post}})(W^\top\Omega)$, so $C_{\mathrm{pre}}$ is never built.
* Production uses $K=12$ (since s26). The value is a co-adapted choice made in combined knob refits with the learned correction (s25: $K=16$ together with `D21_RANK` 8, $\times0.9885$; s26: $K=12$ with other knobs and a refit), not a value shown to be better in isolation. Single-knob moves are flat or worse: from the 13 Oct point, $K=24$ gave $\times1.008$ and $K=16$ gave $\times1.010$; from s38, $K=16$ gave $\times1.002$. The lean studies and the original port used $K=32$.

**Response-closure form (Section 8).** Every row of $P^{(l,b)}$ and $\Gamma^{(l,b)}$ is linear in the incoming weights $w_i$ of the readout neuron: $P^{(l,b)}_{i,:}=w_i^\top H^{(b)}$ and $\Gamma^{(l,b)}_{i,:}=w_i^\top\operatorname{Cov}(h_{l-1},z_b)$. Here $H^{(b)}$ is the linearized map from $h_b$ to $h_{l-1}$, equal to $\mathrm{diag}(c^{(l-1)}_1)P^{(l-1,b)}$ for $b<l-1$ and to $I$ for $b=l-1$. Likewise $\operatorname{Cov}(h_{l-1},z_b)=\mathrm{diag}(c^{(l-1)}_1)\,\Gamma^{(l-1,b)}$ for $b<l-1$ and $\mathrm{diag}(c^{(l-1)}_1)\,C_{l-1}$ for $b=l-1$. Hence $k_{iq}=w_i^\top B_qw_i$ for $h$-side modes
$$B_q=\mathrm{sym}\sum_bH^{(b)}\,\mathrm{diag}\big(c^{(b)}_2\circ R^b_{:,q}\big)\operatorname{Cov}(h_{l-1},z_b)^\top,\qquad R^b=\Gamma^{(l,b)\top}Q,$$
and
$$12\operatorname{Var}(\Pi_K\psi_i)=3\,q_i^\top\Lambda\,q_i,\qquad q_{iq}=w_i^\top B_qw_i,\qquad \Lambda=4\,G_q^{-1}\succeq0 .$$
This is exactly the diagonal of a $K$-mode response closure $\sum_{ab}\Lambda_{ab}S(B_a,B_b)$ in Puffi's sense. The modes and the coupling are derived rather than fitted, and the coupling is positive semidefinite. The toy confirms all of this to $3\times10^{-15}$.

## 6.2 Same-hub and star terms; what is not included

The deployed candidate is, in units of $\kappa_4/12$,
$$\mathrm{cand}_i=\operatorname{Var}(\Pi_K\psi_i)+\sum_{b,j}\big(V^b_{ij}\big)^2\mathrm{var}_{b,j}+\frac13\sum_{b,j}c^{(b)}_{3,j}P^{(l,b)}_{ij}\big(\Gamma^{(l,b)}_{ij}\big)^3,$$
so that
$$12\,\mathrm{cand}_i=\kappa^{\mathrm{proj}}_{4,i}+\kappa^{\mathrm{same}}_{4,i}\big|_{\mathrm{He2}}+\kappa^{\mathrm{star}}_{4,i}$$
(the code mirror in the toy reproduces this to $7\times10^{-16}$).

* **The split proj + same is not a decomposition of $\operatorname{Var}(\psi)$.** The same-hub (diagonal) part is added to restore what a low-rank projection misses, but it double counts what the projection already contains. In the toy, (proj + same)/$\operatorname{Var}(\psi)$ is 0.99, 1.20, 1.22 and 1.28 for $K=1,2,4,8$ with exact top modes. Offline at 1024 the slope of truth on prediction falls from 0.92 ($K=32$) to 0.58 ($K=512$) at layer 4.
* **No induced term.** The deployed form does not include the induced term. The lean gate measured adding it (birth layer $l-1$, inside the LOT shape) at lean $\times0.979$; it was not shipped.
* **All three parts are needed.** The projection alone is worse than V29 ($+6.1\%$, 2/6), and same + star alone has a near-zero mean, so the rescale explodes ($+2167\%$).

## 6.3 The level: shape from LOT, level from V29

Per layer, with means over the live rows,
$$g^{\mathrm{LOT}}_{l,i}=\overline{g^{\mathrm{V29}}_l}\;\frac{\mathrm{cand}_{l,i}}{\overline{\mathrm{cand}_l}},$$
applied only when the guard holds: $\overline{\mathrm{cand}}>0$, $\overline{g^{\mathrm{V29}}}>0$, both finite, and
$$0.35\le\overline{g^{\mathrm{V29}}}/(12\,\overline{\mathrm{cand}})\le1.5 .$$
In words, V29's level must lie within $[0.35,1.5]$ times LOT's own projected level; otherwise the layer keeps V29's $g$. Here $g^{\mathrm{V29}}$ is V29's memoryless regenerated diagonal: a transported local diagonal plus $\lambda_l$ times an off-diagonal variance term, where $\lambda_l=\mathrm{LAM}[l]\big(\overline{dG}/\overline{\mathrm{var}}\,/\,\mathrm{REF\_R}[l]\big)^\beta$ is fitted offline. LOT starts at layer 1 (`PORTLOT_LMIN=1`); layer 0 is Gaussian and has no source.

## 6.4 Why shape and level separate

The mean program uses $g$ only through $r_{4,i}\,g_i$ (Section 1.2), which is affine. To first order, the chain carries a mean perturbation at layer $l$ to the final layer linearly, $\delta m_{\mathrm{fin}}=\sum_lT_{\mathrm{fin},l}\,\varepsilon_l$ with $T=\prod\mathrm{diag}(\Phi(\alpha))W^\top$. Write $g_l=s_lf_l$ with $\overline{f_l}=1$. Then
$$\delta m_{\mathrm{fin}}\approx\sum_lT_{\mathrm{fin},l}\Big[r_{4,l}\circ\big(s_lf_l-\kappa^{\mathrm{true}}_{4,l}\big)\Big].$$
Two consequences follow:

* For a fixed shape, the final error is affine in the per-layer scalars $s_l$. The level is a one-number-per-layer least-squares problem, which is what V29's fitted LAM/REF_R/$\beta$ tables solve end to end for V29's shape.
* For a fixed level, the shape enters through the transported, demand-weighted residual. Keeping V29's $s_l$ and replacing $f_l$ isolates the shape (to first order).

The transfer is approximate in two ways. V29's level was fitted for V29's shape, and the best level for another shape differs unless the two agree on the demand-weighted average. And the code normalizes by the plain neuron mean, not the demand-weighted one. The measured transfer (Section 6.6) is the evidence that these approximations are good enough.

Without the borrowed level, the cheap form is $+38.1\%$ worse ($[+27.5,+48.8]$, 0/12; E01-08, recomputed from the per-network logs). Its own level is wrong for three reasons: the proj + same double count, the He2 coincident weights (Section 5) and the missing induced term. The exact chain + star needs no level ($-21.0\%$ $[-24.8,-17.1]$, 12/12; E01-07), because at $n=1024$ its level is already right (offline slope 0.97-1.08).

## 6.5 Cost

Per source and layer, the two products $\Gamma^{(l,b)\top}Q$ and $V^b(\Gamma^{(l,b)\top}Q)$ cost $2n^2K$ multiply-adds, i.e. $4n^2K$ FLOPs or $2K/n$ units ($1\,\mathrm{u}=2n^3$ FLOPs). That is 0.0625 u at $K=32$ and 0.0234 u at $K=12$. Over the 120 source-layer pairs this gives about 7.5 u and 2.8 u respectively, before the exact factored treatment of old sources (the `cost1` regrouping) and before $O(n^2)$ elementwise terms (derived, not metered).

**Measured cost.** The +9.2 u and +362 calls were metered for the original port at $K=32$ on the V29 base; the cost at $K=12$ was not metered separately. In the final stack the sign of LOT's cost is reversed: switching LOT off raises the bill by 9.3% (#336912 vs #336780, $+12.7$ to $12.9$ u), because the chain then falls back to costlier $\kappa_4$ programs (lazily built plans).

## 6.6 Measured effect, and what each number measures

Intervals are paired sum-ratios recomputed from the per-network logs (`figures/data/shape_level_grid.csv`), as in the paper's Table 1.

| measurement | number | what it measures | id |
|---|---|---|---|
| exact LOT shape $\times$ V29 level, lean, public mini 0-11 | $-25.0\%$ $[-29.4,-20.5]$, 12/12 | best case of the shape/level split | E01-04 |
| fit-free exact chain + star (own level) | $-21.0\%$ $[-24.8,-17.1]$, 12/12 | LOT needs no fitted level when the chain is exact | E01-07 |
| proj32 + same + star, V29 legs, V29 level | $-17.8\%$ $[-20.5,-15.1]$, 12/12 | the cheap form | E01-05 |
| cheap form with its own level | $+38.1\%$ $[+27.5,+48.8]$, 0/12 | the cheap level is wrong | E01-08 |
| flopscope port on V29 ($K=32$, LMIN 1) | raw $\times0.842$ $[0.823,0.862]$, 24/24 | transfer lean to production; 0-11 subset $\times0.835$ | E01-06 |
| final stack, graded control #336912 (PORTLOT=0) vs s43 | raw $\times1.358$, FLOPs $\times1.093$, score $\times1.484$ | all three roles of LOT (below) | E18-202, E18-GC03 |
| final stack, local ablation vs s40a | raw $\times1.365$ (se 0.0135, 60/60) | same, 60 nets | E17-A01 |

The knock-out removes LOT from three roles at once: the mean program's $\kappa_4$ diagonal, HZ's features, and the basis of the D21 demand projection. A separate production measurement estimates LOT's own mean-program value in the final stack at only about $\times1.08$ (E18-80). So $\times1.358$ measures LOT together with HZ, which is built on it; LOT alone, against the HZ-off control, costs $\times1.105$ (paper Section 3.4).

Offline, at $1024\times16$ (MLP 0), the exact LOT shape explains 86-91% of the neuron-to-neuron variance of the true $\kappa_4$, against 58-69% for V29's memoryless core (E01-09). The cheap proj32 + same + star shape reaches an attenuation-corrected correlation of 0.82-0.87 with exact eigenvectors, against 0.94-0.95 for exact LOT; production's $K=12$ one-pass sketch was not measured offline.

**"Truer $\kappa_4$ makes the chain worse" (other teams), and why LOT is a counterexample.** Several write-ups report that a $\kappa_4$ closer to the truth in one respect hurt end to end:

* 18272 D.2: ARC's augmented $\kappa_4$ matches the marginal level but is 1.42-1.54$\times$ worse, with per-neuron $R^2$ of 0.03-0.15.
* 18249 Section 7.1: "the closure set to k = 4's own level, without the young pattern, is worse than ours".
* 18282: a 3+2 tree added in its leg transport, $+76\%$, 0/32.
* This work's own lean gate: a 3+2 tree added on top of the V29-level LOT shape, $\times1.94$ on net 0.

LOT is the case where the improvement is in the shape and the level stays fitted, and it pays: $\times0.842$ on the V29 base; in the final stack, removing it costs graded raw MSE $\times1.105$ against a build without HZ (removing it with HZ, which is built on it, costs $\times1.358$). The exact chain + star (shape and level from theory, $-21\%$) shows that "truer" is not the problem; truer *level with a wrong shape*, or *additive* trees on a calibrated level, is.

# 7. Correspondence to the code

All line numbers refer to `estimator/estimator.py`, the graded file of #337608.

| object | lines | content |
|---|---|---|
| module docstring, portlot summary | 77, 95-103 | says "a rank-32 basis"; **stale**: the default is $K=12$ |
| Wick table $w(k,1)=c_k$ | 245-246 | `WICK_PAIRS`, `WICK_UNIFIED` |
| mean program terms | 247 | `TERM_SPECS[(1,)]`: $c_3/6\,D_3$, $c_4/24\,g$, $c_6/72\,D_3^2$, $c_7/144\,D_3g$ |
| knobs | 312-321 | `PORTLOT` on, `PORTLOT_K` 12, `PORTLOT_Q` 1, `PORTLOT_LMIN` 1, `PORTLOT_STAR`/`SAME` 1, `PORTLOT_RMIN`/`RMAX` 0.35/1.5 |
| enable | 4773-4775 | `lot = PL_ON and regen` |
| basis calls | 5690-5692; 5744-5746; 5881-5883 | trim layer through $WC^{\mathrm{post}}$; $C_{\mathrm{pre}}$ otherwise |
| `_lot_g4` call | 5960-5962 | passes V29's `g4row` as level reference, `D3`, `var`, `mu`, old-source factors |
| fallback, K22, response31 | 5965-5971 | `lot_g4m = g4m_k22` if LOT declined (LC_K22 is off by default) |
| use in the mean program | 6169-6171 | `b1m['g4'] = lot_g4m` |
| per-birth-layer slab | 6274-6282 | `lot_slab[b] = (c1*var, var, w(3,1)/3)` |
| `_lot_basis` | 6640-6660 | $Q=\mathrm{qr}(C\Omega)$, $G_q=Q^\top CQ$ |
| HZ constants | 6694-6698 | HZ reuses LOT's terms as features scaled by 12, $4/3$, 12, 12 |
| `_lot_g4` | 7373-7513 | the closure (rows below) |
| `T3` $=\Gamma$ | 7383-7384 | $P\circ(c_1\mathrm{var})_{\mathrm{col}}+A$ |
| `T1` $=P\circ\Gamma$, `T2` $=V$ | 7385-7391 | `J2_LOT` folds $w_2=c_2$ into `gu` instead of forming `T2` (exact) |
| `gu` $=\Gamma^\top Q$ | 7392-7434 | young slots in $n$-space; old slots through factors, $(FA+FP\,CV)^\top(B^\top Q)$ (exact regrouping) |
| `KU` $=\sum_bV^b\Gamma^{b\top}Q$ | 7435-7456 | `LOTW` hub windows and `TM_LOTHOT` row ranges (exact given the zero rows) |
| proj $=\mathrm{diag}(KU\,G_q^{-1}KU^\top)$ | 7457-7459 | `solve(Gq, KU.T)` |
| same-hub $\sum V^2\mathrm{var}$ | 7460-7468 | |
| star $\tfrac13\sum c_3P\Gamma^3$ | 7473-7484 | `TM_LS0` skips source 0 (exactly zero: $\alpha=0$ at layer 0) |
| level and guard | 7485-7488 | `g4m = cand * mg/m1` if `0.35 <= mg/(12 m1) <= 1.5` |
| return | 7513 | `g4m` if the guard holds, else `None` |

The toy mirrors lines 7383-7484 operation by operation (non-`J2` and `J2` paths) and reproduces $\Gamma$, $V$, `KU` and $12\,\mathrm{cand}=\kappa^{\mathrm{proj}}_4+\kappa^{\mathrm{same}}_4+\kappa^{\mathrm{star}}_4$ to $10^{-15}$. One reading is taken from V29's documentation rather than recomputed here: that the stored $A$ legs equal $P\,\mathrm{diag}(c_1)C^{\mathrm{off}}$, with old sources in shared bases being low-rank approximations. The lean runs measured that building $V$ from V29's legs instead of exact propagators changes nothing in the chain ($-17.8\%$ vs $-17.6\%$).

# 8. Relation to prior work, and what is new

**ARC** (Wu et al. 2026, arXiv:2605.05179; code `alignment-research-center/mlp_cumulant_propagation`, `src/mlp_kprop/factor_k4.py`, `factored_nonlin_kprop_k4`, lines 322-641):

* The Hermite/Wick diagram machinery (Thm A.2), the power counting (Cor. S.2.14), and the factored $K=4$ step. That step builds the $\kappa_4$ born at each ReLU from the path ($\times12$), star ($\times4$) and 3+2 ($\times12$) trees and carries it as a factored 4-tensor, at about $24n^4L^2$ FLOPs. The diagram set and the constants 12, 4 and 12 are therefore ARC's.
* ARC also treat repeated (coincident) indices through power cumulants, i.e. with the neuron's full nonlinearity rather than one Hermite vertex per leg; LOT does not (Section 5).
* What differs: LOT carries no $\kappa_4$ tensor. It writes the readout diagonal in closed form, summed over all birth layers through the backbone (Proposition 2 shows that, at tree level with free index sums, it equals ARC's trees transported and read on the diagonal).

**Puffi's response closure** (Phase 1, #327725, forum 18175; `docs/forum/attachments/18175_yqsl71y2a3zdgb4spyqfnt9epoe-pdf.md`):

* After each ReLU the closure sets $\kappa_4\approx\sum_{ab}\Lambda_{ab}S(B_a,B_b)$ on $p\le4$ covariance-derived modes. It fits a symmetric, possibly indefinite, $\Lambda$ by ridge regression to the marginal $\kappa_4$ of the next pre-activation, $\hat\kappa_{4,j}=3\,q_j^\top\Lambda q_j$ with $q_{aj}=w_j^\top B_aw_j$. The closure is exact in a conditional-Gaussian ("separate universe") model.
* michael_anderson7 uses it in Phase 2 with free-transport modes (18272 Section 6, Appendix I.1 and P3): graded $-2.3\%$ with a corrector refit; the closure-only ablation is $\times1.0327$ raw.
* By Section 6.1, LOT's projected chain has exactly this diagonal form, with $K$ modes and $\Lambda=4(Q^\top C_lQ)^{-1}$. The deployable form therefore belongs to Puffi's family: low-rank modes from the live covariance, with a fitted level. The modes and the coupling come from the tree, and the coupling is positive semidefinite by construction, whereas the 18272 write-up reports its shipped $\Lambda$ indefinite on 8 of 14 layers.

**Kuznetsov, "3+2"** (forum 18249 Section 7; posted 16 Oct, repository 14 Oct):

* Re-derived the three young trees (checked against ARC to $10^{-16}$).
* Showed that the 3+2 tree is essential.
* Identified the cross-layer pairing $D^x(a,r)=\kappa_3(z_r\text{ at birth};z_a,z_a\text{ now})$, which is $\Xi$ here.
* Built a young-tree reader (two generations on dense legs) read into the $\kappa_4$ diagonal and the (3,1) slice, "adding only the part beyond the closure's pattern so that the closure keeps its own level". It reached $\times0.701$ in research (10/10 nets). A cheaper port measured $\times0.924$ raw on the 100 public networks at $+5.1\%$ FLOPs, but its graded probe (#334486) crossed the residual-time cap on 16/100 networks and was not designated.
* By Proposition 2, Kuznetsov's "3+2 is essential" and the finding here that cross-source chains are necessary are the same physics, found independently. Kuznetsov's level-preserving reader is a close relative of the shape/level split: it keeps the closure's level and adds the tree pattern orthogonal to it; LOT replaces the shape and keeps V29's level. The lean gate found that, on this estimator's chain, the youngest 3+2 adds only lean $-3\%$ and nothing in production (E04-08, E04-09), because LOT already carries its cross part.

**Other related field work** (cited because it overlaps; read from the write-ups only):

* **Luna** (18281 Section 3.7, Eqs. 8-11, posted 24 Oct): a diagonal with a star term carrying factor 4, a term $1.33\,T_3^2/\mathrm{var}$ whose constant is fitted, and a "chaos side term" resolved on the top 32-128 whitened covariance modes, with the hub-diagonal piece subtracted to avoid double counting (LOT's deployed form does not do this). For the side term Luna write that they "did not derive it independently". Their aligned chaos model, $z_a=L_a\cdot g+\tfrac12(g^\top B_ag-\mathrm{tr}B_a)$ with $B_aL_a\parallel L_a$ (derivation LLM-assisted), gives the diagonal $\tfrac43T_3^2/\mathrm{var}$ ("_K33 = 4/3"). That is the equality case of Proposition 3, so the $4/3$ coefficient has independent prior publication in that case. At tree level the (i,i,j) slice contains $2\operatorname{Cov}(\psi_i,L_j)$, because $D21_{ij}=\kappa(L_i,L_i,Q_j)+2\operatorname{Cov}(\psi_i,L_j)$ (Lemma A with $Y=L_j$; the toy confirms it to $7\times10^{-4}$). The rest is this document's reading, not verified beyond their text. Their $3\|T_{21}V_w\|^2$ on whitened modes is then $12\operatorname{Var}(\Pi_K\psi_i)$, and their subtraction of the $L$ direction removes the part $\tfrac43D_3^2/\mathrm{var}$, which they carry separately. If so, Luna's side term plus their $\tfrac43$ term is an independent, contemporaneous empirical discovery of essentially LOT's projected chain. Their side term was already in their 11 Oct file; LOT was derived on 1-2 Oct and has been in the graded code since s01 (#335600, 12 Oct).
* **Nurlanggs** (18252): ARC's star + path in production, $\times0.9833$.
* **yeongjunyoo** (18284): a star fold, $\times0.982$.
* **megan_park** (18282): the K31F leg fold.

**Dates.** LOT was derived on 1-2 Oct (internal theory notes), ported on 2 Oct, and has been in the graded code since s01 (#335600, 12 Oct): the s01 source (SHA256 prefix `0ec702df`) has `PORTLOT` on by default, with $K=32$ and LMIN 1. None of this was public before the write-up; this work claims no priority over contemporaneous work.

**What is genuinely new.**

1. The closed-form reorganization of the tree-level fourth cumulant on the diagonal read by the mean program: the path and the cross part of 3+2, over all birth layers, are the variance of one backbone functional $\psi_i$, and the rest of 3+2 is the induced term (Propositions 1-2).
2. The identity pair $D_3=3\operatorname{Cov}(L,\psi)$, $\kappa^{\mathrm{chain}}_4=12\operatorname{Var}(\psi)$ for general (non-aligned) $\psi$, and with it the Cauchy-Schwarz inequality $\kappa^{\mathrm{chain}}_4\ge\tfrac43D_3^2/\mathrm{var}$, the exact relation $\kappa^{\mathrm{chain}}_4=\tfrac43D_3^2/(\mathrm{var}\,\rho^2)$, and the reading of $\kappa_3^2/\mathrm{var}$ closures as its constant-$\rho$ shadow. The equality case ($\rho=1$, coefficient $4/3$) is also in Luna's aligned chaos model.
3. The measurement that cross-source chains are necessary (Section 4), with Kuznetsov's parallel finding acknowledged.
4. The specific deployable form: the $L^2$ projection of $\psi$ on the current layer's top-$K$ backbone modes built from V29's own legs, plus same-hub and star, with V29's fitted level. Its transfer from lean to flopscope to the grader was measured in this work, which also found that the cheap form fails without the level ($+38.1\%$) while the exact chain does not need one.
5. New in this derivation: the closed form of the coincident cumulant and the corrected same-hub weight (Section 5), so far tested only on the toy.

**What is not new here.** The diagrams and their constants (ARC), the response-closure algebra and the mode-projected form with a fitted level (Puffi; michael_anderson7's Phase 2 use), the young-tree reader and the 3+2 pairing (Kuznetsov), and the whitened-mode side term and the aligned-case $4/3$ (Luna).

# 9. Verification notes

## 9.1 The constants, re-derived

* **3** in $D_3=3\operatorname{Cov}(L,\psi)$: three placements of the He2 vertex, times $\mathbb E[LL{:}\tilde\zeta^2{:}]\cdot\tfrac12c_2P=c_2P\Gamma^2$. Equals $6a^\top Ma$ in the generating function. **Confirmed.**
* **12** in $\kappa^{\mathrm{chain}}_4=12\operatorname{Var}(\psi)$: $\binom42=6$ slot pairs, times $8\cdot\tfrac12\cdot\tfrac12=2$ per hub pair. Equals $48a^\top M^2a$. **Confirmed.**
* **$4/3$**: $12/3^2$, by Cauchy-Schwarz. Tight for one coherent hub direction. **Confirmed, with scope**: tree-level chain only, with the tree-level $D_3$; it is not a floor on the total $\kappa_4$ or on a chain's actual $D_3$ (Section 3).
* **4** (star): four placements, times $\tfrac16c_3P\cdot3!\Gamma^3$. Equals ARC's star count. In the code the slab entry $c_3/3$ is $4/12$, consistent with the candidate's $\kappa_4/12$ normalization. **Confirmed.**
* **12** (induced): four placements, times $c_2P\Gamma$ (the cross term of $\tfrac12c_2\tilde z^2$), times $\tfrac12c_2'P'$, times $3!$ pairings, which gives $4\cdot3=12$ times $V\Xi$. Equals the output-leg part of the transported 3+2 (Proposition 2). Production-level consistency: HZ fits a weight to $12\sum V\Xi$ evaluated exactly on its rows, and the fitted weight is 1.00 (sd 0.11) over layers 5-15 in s38-s43 (E04-22) and 0.74-1.13 (mean about 1.0) in G4. This is suggestive only, because HZ fits the mean error with collinear features. **Confirmed.**
* **Same-hub weight** in $12\operatorname{Var}(\psi)$, $j=j'$, $b=b'$: correct as an identity of the truncated model, but the truncation is wrong there. **Corrected**: replace $\varphi(\alpha)^2$ by $\Phi(\alpha)(1-\Phi(\alpha))-\varphi(\alpha)^2$ at order $a^2p^2$ (Section 5.3). The quoted single-neuron numbers 0.182 (exact) and 0.318 (He2) are right at $\alpha=0$. For $\alpha\ne0$ the comparator must include the star's coincident part $2c_1c_3$.

## 9.2 The toy check (`checks/lot_toy_check.py`, numpy only, about 1 min single-threaded, under 1 GB)

* **Part A (one neuron).** Closed form vs truncated moments: $3\times10^{-15}$ on $[-3,3]$. Exact 0.181690 against He2 0.318310 and $c_2c_4=-0.159155$; Hermite series 0.18179 at $m\le140$; MC 0.1811. The $c_k$ pass the Stein check to $5\times10^{-16}$.
* **Part B (width 8, depth 4, exact Gaussian backbone, $N=1.6\times10^7$).** Joint cumulants of the explicitly built chaos components against the closed forms. Pooled ratios: $\kappa(L,L,Q)/\operatorname{Cov}(L,\psi)=0.9996\pm0.0008$, $\kappa(L,L,Q,Q)/2\operatorname{Var}(\psi)=0.9991\pm0.0017$, $\kappa(L,L,L,T^*)/\sum c_3P\Gamma^3=0.9998\pm0.0018$, $\kappa(L,L,L,T^{\mathrm{ind}})/3\sum V\Xi=0.9990\pm0.0023$; largest per-neuron $|z|=1.93$ (8 neurons). The tree-level slice identity $D21_{ij}=\kappa(L_i,L_i,Q_j)+2\operatorname{Cov}(\psi_i,L_j)$ holds to $7\times10^{-4}$. Exact checks:
  * generating function of the quadratic form: $8\times10^{-15}$;
  * Cauchy-Schwarz: slack $\ge0.35$;
  * 3+2 split: $3\times10^{-15}$;
  * code mirror: $7\times10^{-16}$;
  * basis invariance: $8\times10^{-16}$;
  * S-form with $\Lambda=4G_q^{-1}$: $3\times10^{-15}$, $\Lambda$ positive definite.
* **Part C (true network, width 32, depth 6, 4 networks, $3\times10^6$ samples each).** Accuracy probe, not an identity check. LOT's $D_3$ correlates with the true $\kappa_3$ at 0.96-0.97, but with slope (truth on prediction) 1.19-1.29: at width 32 the tree underestimates the $\kappa_3$ level by about 20-30%. The $\kappa_4$ results are in Sections 4 and 5.3.

## 9.3 What remains unverified

* The $1024\times16$ offline numbers were read from the theory log of MLP 0 (one network), not re-run.
* The lean, flopscope and graded numbers were read from the evidence ledger (`docs/EVIDENCE.md`), not re-run.
* The corrected same-hub weight was tested only on the width-32 toy, not at 1024 and not in the chain.
* The order counting (A2) is a theorem only in expectation over the weights (ARC), not for a fixed network.
* The equivalence of Luna's side term with $12\operatorname{Var}(\Pi_K\psi)$ is this document's reading of their text.

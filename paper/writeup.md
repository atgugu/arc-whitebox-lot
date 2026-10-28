---
fontsize: 11pt
geometry: "a4paper,margin=2.25cm,headsep=14pt"
numbersections: true
secnumdepth: 2
---

```{=latex}
\thispagestyle{plain}
\begin{flushleft}
{\sffamily\small\color{muted} ARC White-Box Estimation Challenge 2026 \textbullet{} Phase 2 algorithmic write-up \textbullet{} submission \#337608}\par\vspace{10pt}
{\sffamily\fontsize{24}{28}\selectfont\bfseries\color{ink} Shape from theory, level from data}\par\vspace{6pt}
{\sffamily\large\color{ink} Leading-order trees for the fourth cumulant in a cumulant estimator for deep random ReLU networks: the shape computed, the level fitted}\par\vspace{12pt}
{\small\color{muted} AIcrowd antoniu\_gugu9 \textbullet{} \href{https://github.com/atgugu/arc-whitebox-lot}{github.com/atgugu/arc-whitebox-lot} \textbullet{} 28 October 2026}\par
\end{flushleft}
\vspace{4pt}
```

::: {.keybox title="At a glance"}
**The idea.** Each neuron's mean is linear in its fourth cumulant $\kappa_4$, so $\kappa_4$ splits into a per-layer *level* and a neuron-to-neuron *shape*. The shape follows from how earlier neurons feed each neuron through the weights, and theory computes it cheaply. The level is where a cheap approximation's errors and the chain's compensation collect, and a table fitted end to end sets it to what the rest of the chain needs. LOT (leading-order trees), the closure introduced here, takes the shape from theory and the level from the fitted table of V29, the public chain this estimator builds on.

**The key measurement** (Table 1 and Figure 1; development networks with Monte Carlo truth). In its affordable form, projected on 32 covariance modes, LOT is 38% *worse* than V29 at its own level and 18% *better* at V29's level, a factor 0.59 between the two, on 12 of 12 networks. The exact form is 17–21% better even at its own level, but too expensive to ship.

**What is new, and what it is worth.** New: a closed-form resummation of ARC's tree-level $\kappa_4$ diagonal over all birth layers through one backbone functional, with two identities that tie it to the third cumulant; and the measurement that its shape transfers when the level is held fixed. Worth: ported alone onto V29 it lowers the raw MSE by a factor 0.842 (24/24 networks); removing it from the 17 October build (s43) costs graded raw MSE ×1.105, worse on 47 of 50 MLPs, measured against a build without HZ. Not claimed as new: about two-thirds of the log-score gain over V29 is FLOP engineering shared with the field (Appendix B), and the learned head adds about 3%.

**Contributions.**

1. **LOT**, the closed-form tree-level $\kappa_4$ diagonal, deployed as shape only; its level is V29's, so it adds no fitted table.
2. **HZ**, a small learned correction built on LOT's exact terms, whose weights stay interpretable: V29's own diagonal is driven to zero, and the induced term's weight averages 1.01 over layers (sd 0.11) against a theoretical 1; the chain term's averages 0.82, below 1 on every layer. HZ takes 6% of the FLOP bill; removing it costs graded score ×1.154, worse on all 50 MLPs.
3. **A measured account of when "more correct" hurts** a tuned chain, with two consistency fixes supported on a 300-network holdout.

**Submission.** This write-up corresponds to submission #337608 only. #337608: score $2.02\times10^{-9}$ (raw MSE $1.541\times10^{-8}$ at 13.1% of the budget; means over the 50 public MLPs, with all 100 graded MLPs completed); ×0.375 of V29's leaderboard score of $5.40\times10^{-9}$. The graded estimator, byte-identical, and everything needed to check these numbers are in the repository. Nominated for the private re-evaluation: #337632, which is #337608 with only the head weights changed, and #337120, an earlier build kept as a wall-time-safe option (Appendix B).
:::

# Introduction

*In plain terms.* Instead of running a random network on samples, the estimators in this challenge push a few statistics of each layer (means, covariances, and measures of skew and tail weight) through the network with formulas. The tail statistic is the hard one: it is too large to carry, so everyone approximates it, and teams found that making the approximation more faithful often made results worse. The idea is to split it into two parts: how big it is on average in each layer, and how it varies from neuron to neuron. Theory computes the second part cheaply, and it pays; the first part is left to an existing fitted table.

The task is to estimate, for every neuron of every layer, the mean activation of a random bias-free ReLU network (width $n=1024$, depth $L=16$, He initialisation, Gaussian input) under a FLOP budget $B=2^{41}$. Each network's score is its mean squared error times $\max(0.1, \text{FLOPs}/B)$, so accuracy and cost trade one-for-one in log space.

In late September, when this work started, the strongest public method was team 504aldo's V29 [2]: a factored third-order cumulant chain built on ARC's cumulant propagation [1], published in early September and widely re-submitted. V29 carries means, covariances and third cumulants exactly enough, but its fourth cumulant is *regenerated* at every layer: a memoryless closure whose per-layer magnitude comes from a fitted table. Over the phase, several teams reported the same surprise: replacing parts of that closure by more faithful versions made their chains worse [11, 13, 15, 16]. The chain, they concluded, is co-adapted; its tables compensate for errors elsewhere, and a locally better ingredient breaks the compensation.

The diagnosis holds, with one refinement. The mean of neuron $i$ depends on the fourth cumulant only linearly, so any $\kappa_4$ factors into a per-layer **level** (its average) and a neuron-to-neuron **shape**. The shape is decided by how earlier hub neurons, the sources of non-Gaussianity, feed each neuron through the weights. No per-layer table can represent it, and the measurements below suggest that the chain's compensation sits mostly in the level (Section 4 shows some per-neuron compensation too). The level is one number per layer. It absorbs what the chain's other truncations drop (Section 5), and it is where an affordable approximation of $\kappa_4$ goes wrong. So the shape can come from theory while the level comes from a table fitted end to end.

Other teams' results point the same way. Kuznetsov's young-tree reader keeps the closure's own level and adds only the tree pattern beyond it [3]. Hack2Publish reached the same split independently, by oracle injection on their chain: the true $\kappa_4$ pattern at the chain's own level gives 108% of the gain of the full true diagonal, while the true level with the chain's pattern is 21.5% worse than no oracle [20]. Several failures reported by others also fit this reading: a closure with the right marginal level but almost no per-neuron shape was 1.42–1.54× worse, with per-neuron $R^2$ of 0.03–0.15 [18], and a further tree added on top of a calibrated level cost +76% [15].

LOT (Section 3) is the closure built on this split. A leading-order-tree reduction gives the $\kappa_4$ diagonal in closed form, $\kappa_{4,i} = 12\,\mathrm{Var}(\psi_i)$ plus two smaller terms, where $\psi_i$ is a linear functional of the Gaussian backbone assembled from all earlier layers. Its exact form improves V29's raw MSE by 17–21% with no fitted constant in the $\kappa_4$ diagonal, but it would cost more than the whole bill of #337608 (estimate in Section 3.2). Projected on a few covariance modes it costs about 2% of that bill, but its own level is then wrong: 38% worse than V29. At V29's level the same projected shape is 18% better (Table 1).

The same split organises the rest of the estimator. HZ (Section 4) is a learned correction whose inputs are LOT's exact diagram terms; the data mostly rescale them, and the fitted weights stay of order one. The demand projections (Section 2.3) use the chain's first-order transport to decide where exact computation is needed: theory selects the rows, and a fixed budget sets how many. Section 5 measures the compensation itself, shows which consistency repairs pay and which do not, and explains why LOT is on the right side of that line.

Component claims are backed by paired ablations, and three of them (LOT, HZ and the demand projection) by single-variable knock-outs graded by the official grader, which reproduce the local estimates (Section 6). Appendix B shows where the improvement over V29 came from. About two-thirds of it, in log score, is FLOP reduction, mostly engineering of the kind many teams share; the rest is accuracy, led by LOT and HZ. Grey tags such as \ev{E01-09} point to rows of the evidence index in the repository (`docs/EVIDENCE.md`).

![**Theory supplies the shape; for the affordable form, the level had to come from the table.** (a) Raw MSE relative to V29 (log scale, 95% intervals, paired over networks) for two sets of $\kappa_4$ terms, each at V29's fitted level (blue) and at its own, fit-free level (orange); light-blue reference: the exact form with the induced term at V29's level; open diamond: the production port on 24 networks. (b) Share of the per-layer neuron-to-neuron variance of the true $\kappa_4$ explained by LOT and by V29's memoryless core (one development network, Monte Carlo truth).](fig_lot_shape_level.pdf){width=100%}

**What is not new here.** The cumulant-propagation framework, Hermite/Wick machinery and term tables are ARC's [1]. The chain that LOT extends, including its sources, memoryless $\kappa_4$ regeneration with fitted tables, age-tiered bases and fast-matmul engine, is 504aldo's V29 [2]. The diagrams that LOT sums appear in ARC's `factor_k4` [1]. A covariance-mode-projected fourth-order closure with a fitted level, Puffi's response closure [10], predates LOT's deployable form, and Luna's whitened-mode side term [11] is an independent, contemporaneous relative. Hack2Publish's $\kappa_4$-pattern corrector, a fitted map rescaled to the chain's layer-mean level with a derived cubic-star feature equal to LOT's star term, was graded before this work's first submission and is an independent development [20]. The marg2 rule, a removal of a double count in the pair closure (Section 5), is Artyom Kuznetsov's (Artyom\_Mazur) [3]. Section 8 lists every credit.

# Setting and starting point

## The cumulant chain

Write $z_\ell = W_\ell^\top h_{\ell-1}$ for the pre-activations of layer $\ell$ and $h_\ell = \mathrm{relu}(z_\ell)$. For neuron $i$ let $\mu_i$, $\sigma_i^2$ be the mean and variance of $z_{\ell,i}$, $\alpha_i = \mu_i/\sigma_i$, and let $D_{3,i}=\kappa_3(z_i,z_i,z_i)$ and $\kappa_{4,i}$ be its third and fourth cumulants. An Edgeworth expansion of $\mathbb E[\mathrm{relu}(z_i)]$ gives the mean program

$$
m_i = \sigma_i\big[\phi(\alpha_i) + \alpha_i\Phi(\alpha_i)\big] + c_3(\alpha_i)\,D_{3,i} + c_4(\alpha_i)\,\kappa_{4,i} + c_{33}(\alpha_i)\,D_{3,i}^2 + c_{34}(\alpha_i)\,D_{3,i}\,\kappa_{4,i},
$$

with $c_3 = -\alpha\phi/(6\sigma^2)$, $c_4 = (\alpha^2-1)\phi/(24\sigma^3)$, $c_{33}=(\alpha^4-6\alpha^2+3)\phi/(72\sigma^5)$ and $c_{34}=-(\alpha^5-10\alpha^3+15\alpha)\phi/(144\sigma^6)$. The mean is affine in $\kappa_4$. The covariance update reads one slice of the third cumulant, $D21_{ij} = \kappa_3(z_i,z_i,z_j)$. The chain propagates means and covariances exactly, third cumulants as factored per-birth sources, and the fourth cumulant only as the diagonal the mean program needs. Below, "chain" means this layer-by-layer propagation, and "chain term" the $12\,\mathrm{Var}(\psi)$ part of LOT.

## V29's fourth cumulant: a memoryless closure with a fitted level

V29 regenerates the $\kappa_4$ diagonal at each layer from quantities available there, with an off-diagonal coefficient $\lambda_\ell = \mathrm{LAM}[\ell]\,\big((\overline{dG}/\overline{\mathrm{var}})/\mathrm{REF\_R}[\ell]\big)^{\beta}$ fitted offline on public networks [2]. It is accurate on average, layer by layer, and it is local: each neuron's value depends on that neuron's own state. Two measurements frame what follows. First, the per-neuron fourth cumulant at depth is *non-local*: index classes local to a neuron shrink to an rms of about $10^{-4}$ against $4\times10^{-3}$ for chains between hubs born at different layers, at layer 15 \ev{theory notes §2}. V29's memoryless core explains 58–69% of the per-layer neuron-to-neuron variance of the true $\kappa_4$ \ev{E01-09}. Second, the chain's $\kappa_4$ level is *wrong but compensated*: its diagonal is 0–29% below Monte Carlo truth, more so with depth \ev{E17-35, E17-36}, yet scaling it up toward truth hurts, ×1.22 for a uniform factor 1.15 \ev{09/band scales} (Section 5).

## Where the error is born, and what the final layer demands

The error is born late: 63% of V29's final-layer MSE originates in layers 12–15, and removing one layer's whole local error changes the final MSE by 20.8% at layer 15 \ev{E01-16, E01-28}. At layers $\ge 8$ the local mean error is explained to 93–97% by three channels, $c_2\,\delta\mathrm{var} + c_3\,\delta D_3 + c_4\,\delta\kappa_4$, with the $\kappa_4$ channel carrying 18–26% of it \ev{E01-27}.

Linearising the chain gives the transport of a local error to the final layer, $e_\ell = \Phi(\alpha_\ell)\circ(W_\ell^\top e_{\ell-1}) + \varepsilon_\ell$, so that $\delta m_L = \sum_\ell T_{L,\ell}\,\varepsilon_\ell$ with $T = \prod \mathrm{diag}(\Phi)W^\top$. For random weights the $T$ act as nearly uniform projections, so the *demand* for accuracy in object $X$ at neuron $i$ is roughly a layer gain times $|c_X(\alpha_i)|^2$ times the error energy there. A direct test supports this. Ranking the third-cumulant rows by a computed first-order influence score leaves 0.984 of the residual energy left by the $|\alpha|$ ranking; even the per-row oracle order leaves 0.947. So $|\alpha|$ is nearly as good as any of the computed rankings \ev{E02-14}.

The estimator uses this to choose representations: rows near the kink ($|\alpha|\le 2.25$) get the exact $D21$ slice; the tail rows get a rank-2 projection on LOT's covariance modes; the means of strongly negative rows get a radial closed form. A graded control prices the projection: making every row exact improves the raw MSE by only 0.3% but raises the bill by 8.3% (score ×1.080) \ev{E18-GC04}. A related principle, readout observability, is Susupro's Proposition 2 [12].

![**The estimator at a glance.** Grey: inherited from ARC and V29. Blue: new components derived from theory. Orange: fitted to data. LOT supplies the shape of the $\kappa_4$ diagonal and the covariance basis of the $D21$ tail projection; V29's table supplies the level; HZ and the non-linear head correct each layer's mean.](fig_schematic.pdf){width=100%}

# LOT: leading-order trees for the fourth cumulant

## The identities

::: {.keybox title="LOT identities (derivation in Appendix A)"}
Expand layer $\ell$'s pre-activation in the Gaussian backbone $\zeta$ (the Gaussian-closure pre-activations of all earlier layers $b<\ell$, with joint covariance $G$). Let $\Gamma^{(\ell,b)}=\mathrm{Cov}(\zeta_\ell, \zeta_b)$ be the backbone covariance and $V^{b} = c_{2,b}\circ P^{(\ell,b)}\circ\Gamma^{(\ell,b)}$, where $P^{(\ell,b)}$ is the linearised transport from layer $b$ to $\ell$ and $c_{k,b}=\mathbb E[\mathrm{relu}^{(k)}]$ at layer $b$ (Section 2.1's $c_3$, $c_4$ carry an extra $1/k!$). Define the linear functional
$$\psi_i = \sum_{b<\ell}\sum_j V^{b}_{ij}\,\tilde\zeta_{bj}$$ with $\tilde\zeta$ the centred backbone.
At tree level (one Hermite vertex per hub, loops dropped; inexact at coincident hubs, Appendix A), with $L_i$ the linear part of $z_{\ell,i}$:
$$D_{3,i} = 3\,\mathrm{Cov}(L_i,\psi_i), \qquad \kappa^{\mathrm{chain}}_{4,i} = 12\,\mathrm{Var}(\psi_i), \qquad\text{hence}\qquad \kappa^{\mathrm{chain}}_{4,i} \;\ge\; \frac{4}{3}\,\frac{D_{3,i}^2}{\sigma_i^2}.$$
The full leading-order diagonal adds a star term $\kappa^{\star}_{4,i} = 4\sum_b \mathrm{rowsum}\big(c_{3,b}P^{b}\Gamma^{b\,3}\big)_i$ (cube elementwise) and an induced term $12\sum_b \mathrm{rowsum}(V^b\Xi^b)_i$.
:::

The picture behind the identities is simple. To leading order, each neuron's pre-activation is a Gaussian (its linear part $L_i$) plus a small quadratic form in the backbone, created by the curvature of every earlier ReLU and carried forward by the weights. The third and fourth cumulants of "Gaussian plus small quadratic" are both controlled by one vector: the gradient of the quadratic form, $\psi_i$. $D_3$ measures how $\psi_i$ aligns with the linear part; $\kappa_4$ measures its total power. The Cauchy–Schwarz bound follows at once, with equality when $\psi_i$ lies along $L_i$. It applies to the tree-level chain term only, because the star and induced terms are signed, and it gives a cheap per-neuron check: an estimator's chain-type $\kappa_4$ term should not fall below $\tfrac43 D_3^2/\sigma^2$ of its own tree-level $D_3$.

Two measured facts shape the deployable form. *Cross-source chains are essential.* On one development network, with all chains and the star term, the attenuation-corrected correlation with the true $\kappa_4$ stays at 0.89–0.95 across layers 4–15. Keeping only chains whose two ends were born at the same layer (with the star, as in every deployed form) turns the shape from correct into anti-correlated: 0.84 at layer 4, $-0.59$ at layer 15. Without the star, same-layer chains alone reach only 0.69–0.83 \ev{theory log, one network}. Kuznetsov's finding that the "3+2" tree is essential [3] is the same physics (Appendix A). *Coincident indices need care.* Truncating at second Hermite order is wrong where indices coincide: for a single neuron at $\alpha=0$, $\kappa(z,z,h,h) = 0.182$ exactly, against 0.318 from the second-order Hermite (He2) term alone. Appendix A gives the closed form for all $\alpha$.

## A deployable form: theory for the shape, the table for the level

Cost is counted in units $u = 2n^3 = 2^{31}$ FLOPs, one dense $n\times n$ product; the budget is $B = 1024\,u$ and #337608 bills 134 u. Estimates from the *lean* implementation (a plain-numpy research version of the chain, run outside flopscope, the competition's FLOP meter) put the exact form at about 210 u for chain + star and 315 u with the induced term, more than that whole bill \ev{theory notes §2}. The deployable form projects $\psi$ on a $K$-dimensional approximation of the current layer's dominant covariance subspace ($n^2K$ per source; exact eigenvectors at $K=32$ in the lean runs, a one-pass range finder at $K=12$ in production), adds the same-hub term (the part of the chain whose two ends sit on one source neuron, which the projection misses) and the star term, and keeps V29's per-layer level:

$$
\kappa_{4,i} \;=\; \overline{\kappa_4^{\mathrm{V29}}}\;\cdot\;\frac{\mathrm{cand}_i}{\overline{\mathrm{cand}}},
$$

where $\mathrm{cand}_i$ is LOT's projected value for neuron $i$ and bars denote layer means. LOT thus contributes only the neuron-to-neuron shape, and introduces no fitted table of its own: the level is V29's existing table, and $K$ is a size. Its only other constants are two hand-set guard bounds: if V29's level and LOT's own level differ by a factor outside [0.35, 1.5], the layer falls back to V29's diagonal. When first added to V29 (with $K=32$) it cost 9.2 u, 0.9% of the budget \ev{E01-06}. In production ($K=12$, from layer 1) its kernels take about 2% of the bill: 5.8 of 300 GFLOP in a profile of an earlier build \ev{E08-15}. $K=12$ was chosen in a joint refit with other knobs; single-knob moves to $K=16$ or $24$ were flat or slightly worse. The three parts work only together: the projection alone is worse than V29 (+6.1%, 2/6 networks), and the local same-hub and star terms alone make the rescaled shape explode \ev{theory notes §2.5}.

## Shape and level, separated

Because the mean program is linear in $\kappa_4$, shape and level can be tested separately: keep a set of $\kappa_4$ terms fixed and switch only where its per-layer level comes from. Table 1 does this on development networks with Monte Carlo truth, in the lean implementation used for the derivation. Each row compares one set of terms at V29's fitted level with the same terms at their own level.

| terms of the $\kappa_4$ diagonal | networks | at V29's level | at its own level (fit-free) | V29's level vs own |
|:----------------------------|:------|------------------:|------------------:|------------------:|
| exact chain + star | 0–5 | −20.7% [−23.8, −17.6], 6/6 | −17.1% [−21.0, −13.1], 6/6 | −4.4% [−7.6, −1.1], 5/6 |
| projected chain ($K=32$) + same-hub + star | 0–11 | −17.8% [−20.5, −15.1], 12/12 | **+38.1%** [+27.5, +48.8], 0/12 | −40.5% [−44.4, −36.7], 12/12 |
| exact chain + star + induced | 0–11 | −25.0% [−29.4, −20.5], 12/12 | – | – |

: Shape and level, separated on matched terms. Raw MSE against V29 on the same networks (paired sum-ratios with normal-approximation 95% intervals; "6/6" counts networks better than V29). With six networks a $t$-interval is about 30% wider; the paired exact-chain difference then becomes [−8.7, −0.1] and still excludes zero. On all twelve networks the exact chain + star at its own level gives −21.0% [−24.8, −17.1], 12/12. The exact form with the induced term reaches −28.3% on the held-out half (networks 6–11), and on network 0 it matches the noise-corrected true-$\kappa_4$ oracle ($1.64$ against $1.60\times10^{-8}$). Recomputed from the per-network logs, which the repository ships with a script that rebuilds this table (`checks/recompute.py`).

Three readings follow. (i) The theoretical shape is better than V29's. On a development network with Monte Carlo truth, LOT explains 86–91% of the per-layer neuron-to-neuron variance of the true $\kappa_4$ (layers 4–15), against 58–69% for V29's memoryless core (Figure 1b) \ev{E01-09}. (ii) Where the theory is exact, its own level is nearly right. The fit-free exact chain beats V29 by 17–21%, and V29's fitted level improves it by a further 4%, a small residual compensation that only an end-to-end fitted level captures. (iii) Where the theory is approximated to fit the budget, its own level fails. Appendix A identifies three plausible reasons, supported on toy networks but not isolated at full width: the projection and same-hub parts double count, the He2 weight is wrong on coincident indices, and the induced term is omitted. The shape survives the approximation: at V29's level the projected form keeps most of the exact form's gain. So "shape from theory, level from data" does not claim that theory cannot get the level right. It is the division of labour that works at the budget's price.

## Production evidence

Ported onto V29 in flopscope with $K=32$, the projected form lowers the raw MSE by a factor 0.842 [0.823, 0.862], better on 24 of 24 networks \ev{E01-06}. With its 3.5% extra FLOPs, that is a score factor of about 0.87. The table below measures LOT's removal from the late builds three ways, locally on s40a (a local copy of build s40) and on the grader on s43:

| measurement of "LOT off" | base | raw MSE | FLOPs | score |
|:-------------------------------------|:---------------|---------------------:|-----------:|-----------:|
| local paired ablation, 60 networks (LOT and HZ off) | s40a | ×1.365 (se 0.014; 60/60 worse) | ×1.11 (approx.) | – |
| graded control #336912 against s43 (LOT and HZ off) | s43 (#336780) | ×1.358 (se 0.012; 50/50 worse) | ×1.093 | ×1.484 |
| graded #336912 against the HZ-off control #336911: LOT alone | s43 without HZ | ×**1.105** (se 0.009; 47/50 worse) | ×1.163 | ×**1.285** (50/50 worse) |

: LOT knock-outs. Graded controls are real builds with one knob default changed (and the setup plan list re-recorded where needed), so their FLOPs are exact; graded ratios are paired over the 50 public MLPs; standard errors are of the log ratio. Evidence: E17-A01, E18-201, E18-202, E18-GC02, E18-GC03, per-MLP grader telemetry.

The grader reproduces the local raw ratio within error ($z = -0.3$). In this code base, switching LOT off also switches HZ off: HZ selects its rows and builds its features inside the LOT routine, as instrumenting one predict of the released file confirmed. The first two rows therefore remove both. The third, LOT-off against HZ-off, isolates LOT in a stack without HZ. Its score effect exceeds its MSE effect because the LOT-off build bills 16% more FLOPs. The cause of that rise is not isolated; plausible contributors are that the $D21$ tail projection loses its covariance-mode basis and that other fourth-order programs do more work. LOT's accuracy contribution in this stack is therefore quoted as raw ×1.105.

::: {.hedge title="What the knock-outs do and do not measure"}
Even the third row is not the closure alone: LOT's covariance modes are also the basis of the $D21$ tail projection, which the LOT-off build loses too. A separate local measurement puts LOT's mean-program value alone at about ×1.08 \ev{E18-80}. The controls ran on the 17 October build s43 and were graded on 18 October. #337608 differs from s43 by 44 new knobs (later cost work including the PEEL planner, the non-linear head, guards), eight changed defaults and refit tables (listed in `docs/READING_GUIDE.md` §8). The three control knobs exist in #337608 with the same definitions and defaults, so the same knob removes the same mechanism from a slightly different stack. Each control is one graded run on the 50 public MLPs, analysed in pairs with s43 on the same MLPs (Section 6).
:::

## Relation to prior work

ARC's `factor_k4` builds the $\kappa_4$ born at each ReLU from path (×12), star (×4) and "3+2" (×12) trees, at $24n^4L^2$ FLOPs, about 3,000 budgets at this shape [1]. At tree level LOT equals these trees, transported to the readout layer, read on the diagonal and summed over birth layers (Appendix A, Proposition 2): the diagrams and their constants are ARC's. Kuznetsov (Artyom\_Mazur) re-derived the three trees, showed that "3+2" is essential, and built a level-preserving young-tree reader [3]. On this chain the youngest "3+2" adds nothing in production, because LOT already carries its cross part \ev{E04-08, E04-09}. A $\kappa_4$ star read on carried $\kappa_3$ legs was described earlier in marius_binner's public submission notes (#330172, 7 September; as reported in AndreasHad04's write-up [22]). LOT's star term comes from ARC's star tree and was derived independently; what LOT adds is the cross-source chains, summed in closed form over birth layers, and the $\psi$ identity that ties them to $D_3$ [23].

Puffi's Phase 1 response closure, $\kappa_4 \approx \sum_{ab}\Lambda_{ab}\,S(B_a,B_b)$ on covariance-derived modes with a fitted $\Lambda$ [10], has exactly the diagonal form of LOT's projected chain. LOT is a response closure of this kind whose modes come from the live covariance and whose coupling, $\Lambda = 4(Q^\top C Q)^{-1}$, is derived up to the per-layer scale that V29's level supplies, and positive semidefinite (Appendix A). Luna's write-up [11] contains the equality case of the bound above, an aligned chaos model giving $\tfrac43\kappa_3^2/\mathrm{var}$ (shipped with a fitted 1.33). It also has a side term on the top whitened covariance modes that, on this write-up's reading, is an independent, contemporaneous discovery of essentially the projected chain.

Hack2Publish [20] built the closest relative of the deployed form: a fitted seven-feature map of the per-neuron $\kappa_4$ pattern, rescaled to the chain's layer-mean level and fed only to the mean closure, plus a derived cubic-star feature, $24\sum_a c_a \bar Y_{ai} X_{ai}^3$ with $c=-\alpha\varphi(\alpha)/6$ in standardised units, which is LOT's star term. They also analysed a $\kappa_3$-dressed C·T readout (their §4.5; oracle level, 9–10 October, not graded), related to LOT's induced term: with the star it reaches pattern correlation 0.82–0.85 on decisive neurons, and they parked a projected version for cost. Their oracle split (pattern 108% of the gain, level alone harmful) is independent evidence for Table 1. Their Hermite ceiling ladder is consistent with the finding that cross-source chains are necessary: births of ages 1–6 are needed jointly, and second-order births cap the pattern correlation near 0.5 on decisive neurons. neopok919's readers for $\kappa_4$ born from $\kappa_3$ [21] are further relatives of LOT's chain terms.

| Claim | Closest public work | What differs |
|---|---|---|
| Shape/level split of $\kappa_4$ | Hack2Publish oracle split [20]; Kuznetsov's level-preserving reader [3] | Measured with a *derived* shape on matched terms, including the theory's own level (Table 1) |
| Closed-form $\kappa_4$ diagonal, $12\,\mathrm{Var}(\psi)$ + star + induced | ARC's trees [1]; marius_binner's star read [23]; Hack2Publish's cubic star and C·T readout [20] | All generations summed in closed form as one backbone variance; a projected form cheap enough to ship, with graded knock-outs |
| $D_3=3\,\mathrm{Cov}(L,\psi)$ and the $\tfrac43 D_3^2/\mathrm{var}$ floor | Luna's aligned chaos model (equality case) [11] | General $\psi$; used as a free consistency check |
| Mode-projected form with $\Lambda=4(Q^\top CQ)^{-1}$ | Puffi's response closure, fitted $\Lambda$ [10]; Luna's side term [11] | Coupling derived up to the per-layer scale, positive semidefinite; that scale is V29's fitted level |
| Learned correction on exact diagram terms (HZ) | Fitted correctors [6, 7, 12, 14] | Features are LOT's terms; induced weight 1.01 on average over layers against a theoretical 1 |

: Claims of this work, against the closest public work.

LOT was derived on 1–2 October, has been in the graded code since the first submission (#335600, 12 October), and was internal until this write-up. No priority is claimed over contemporaneous work.

# HZ: a learned correction on derived features

After the closures, the remaining error is the chain's own per-neuron closure error. HZ (an internal name) removes part of it with a linear correction whose features are derived diagram terms. On the $K_\ell$ rows of each layer where the $c_4$ weight is largest (32 to 192 rows, on layers 5 and 7–15 in #337608), one deferred backward Horner pass evaluates LOT's *exact* chain and induced terms; together with five approximate pieces (the same-hub, projected and star terms, $\tfrac43 D_3^2/\sigma^2$, and V29's own diagonal) they form the feature matrix $X$. The correction is

$$
e'_\ell = e_\ell + c_4\big(X\theta_\ell - \kappa_4^{\text{used}}\big)\quad\text{on the selected rows},
$$

plus 17 rider weights on all live rows ($c_3D_3$, $c_2\,\mathrm{var}$, $c_4\kappa_4$ and their $\alpha^2$-modulated versions). $\theta$ is fitted by least squares, per layer, on dumps of public development networks (400 for the fit shipped in #337608), to the chain's *local mean error*, not to $\kappa_4$ truth, and the correction is carried forward with $\Phi(\alpha)W$. Fitting to $\kappa_4$ truth does not help, because the other channels anti-correlate with it \ev{E04-20}: the correction has to respect the chain's compensation, in the sense of Section 5.

**The weights stay interpretable.** If LOT's exact terms are right, their fitted weights should be near 1. In the fit shipped in #337608, averaged over its ten live layers, the induced term gets $1.01\pm0.11$ and the exact cross-source chain $0.82\pm0.03$ (mean ± sd across layers). The chain weight is below 1 on every live layer in both fits; because its feature contains the same-hub and floor features, the weights are not separately identified. An earlier, separate fit on a different build gave $1.00\pm0.12$ and $0.84\pm0.05$; the two fits used public development networks that may overlap. The induced weight is not flat: in both fits it rises with depth, from 0.74–0.79 at layer 5 to 1.13–1.18 at layer 15, so "near 1" holds on average rather than layer by layer. In both, V29's memoryless diagonal gets weight $-0.02$: it is superseded. The approximate LOT pieces get partial weight (same-hub 0.39, projected 0.24), and the negative weight on $\tfrac43 D_3^2/\sigma^2$ ($-0.60$ and $-0.63$) removes its overlap with the chain term (Figure 3). The star term is up-weighted (1.38), for reasons not yet understood.

![**HZ's fitted weights, in two separate fits.** (a) Per-layer weights of the two exact LOT terms, for the fit shipped in #337608 and an earlier fit; theory predicts 1. (b) The other five term weights per layer, both fits (approximate LOT pieces faint); V29's memoryless diagonal is driven to 0 (mean −0.019 and −0.018).](fig_hz_recovery.pdf){width=100%}

::: {.hedge title="Caveat"}
The features are collinear, so weights near 1 are suggestive, not an identification. HZ has 336 fitted slots (14 layers × 24: seven term weights and 17 rider weights); ten layers are live in #337608.
:::

**Graded value.** When introduced, HZ lowered the graded raw MSE by ×0.884, at ×1.096 FLOPs in a build that also changed the engine \ev{E04-02}. In the final stack, the graded control #336911 (`HZ_ON=0` on s43) costs raw ×1.228 and score ×1.154, and shows HZ is about 6% of the bill \ev{E18-GC02}. Paired over the 50 public MLPs, HZ-off is worse on all 50 ($z = 23$ for the raw ratio). The local 60-network ablation gave raw ×1.262; the graded effect is smaller ($z = -2.1$ for the difference), plausibly because s43 adds two consistency fixes the fitted $\theta$ never saw.

**Saturation and absorption.** The learning curve of $\theta$ is flat by about 25 networks \ev{E04-04}. More instructive is what happened to candidates that looked good *before* the learned layer was refitted: a learned $\kappa_4$ corrector ($-4.5\%$ before, $-0.3\%$ after), young "3+2" trees (×0.970 in lean, ×1.0004 in production), and 170 add-one features (best ×0.9993 after refit). Most of a candidate's apparent gain was already absorbed by HZ. The promotion rule became: evaluate every accuracy change *after* refitting the learned layer.

**A small non-linear head, trained on the scored objective.** The final builds add NLHZ, a small per-layer neural net (37 inputs, two hidden layers of 64, layers 2–15; 7,488 parameters, weights and biases, plus 1,066 normalisation constants) that corrects the per-layer means. A graded single change introduced it at ×0.978 (#337317 against #337231); two further revisions of the head and its inputs graded ×0.9963 and ×0.9969. Submission #337608 differs from its predecessor only in the head's weights: fine-tuned directly on the final-layer MSE (back-propagating through the exact $\Phi W$ transport of dumped tables), it graded ×0.99663. Training the head on more data gave ×0.99970 (#337632), which is saturation. End-to-end and final-loss training of correctors were used earlier by Hisernberg and hyojun_kwon [6, 7], and mliston fits learned readouts on the estimator's own trajectories [9]. No novelty is claimed for the head. In total the generic learned head is worth about 3% of the submission's score, while removing LOT alone costs score ×1.285 (raw ×1.105).

# Why "more correct" can hurt: the chain is a co-adapted iterate

On 17 October a consistency audit examined the chain's closure. Its covariance update reads pair Wick coefficients through merged programs that contain, besides the Gaussian term, first- and second-order Edgeworth corrections and several pair diagrams. Each ingredient had been added by a different author and tuned with the others present.

**The levels are compensated, not right.** Three measurements show it. The row sums of $K22$, the (2,2) slice of $\kappa_4$, sit in a compounding loop, $\kappa_4\to$ (2,2) row sums $\to g_{\text{prev}}\to$ next $\kappa_4$, around which V29's regeneration tables are tuned. On one development network, lowering the $K22$ level uniformly by 3% costs ×1.169, and raising it by 3% costs ×1.745. Injecting the true $K22$ ($N=10^8$) costs about ×1.25, although the chain's own $K22$ is 12–13% below truth at layer 14 \ev{E17-25, E17-26, E17-27}. The mean program's $\kappa_4$ is 0–29% below truth, yet every per-band rescaling toward truth hurts (×1.04 to ×1.22) \ev{E17-35, E17-36; 09/band scales}. And a scale-covariance (Ward-identity) audit shows that a chain truncated at the fourth cumulant *cannot* be covariant: the missing variation equals the dropped $\kappa_5$, $\kappa_6$ and $\kappa_4^2$ faces (correlation 0.9993–0.9999), and every covariant repair hurts, from ×1.11 to ×351 \ev{E17-29, E17-30, E17-31}.

**What passes and what fails.** Changes that keep the calibrated levels pass: LOT changes only the shape; HZ is fitted to the chain's own error. Removals of same-order double counts pass. Two such removals shipped. Artyom Kuznetsov's marg2 rule [3] (keep the second-order marginal correction only for $\alpha\le-2$) gave ×0.99066 on 300 holdout networks that no earlier decision had used ($z=-12.5$) and ×0.99226 graded, at zero FLOPs. Halving the $D^2$ diagrams (the pair terms quadratic in the $D21$ slice) gave ×0.99874 on the holdout and ×0.99879 in a second build on the same holdout networks \ev{E17-H02, E17-C04, E17-H04, E17-H05}; graded, the matching pair of builds (#336773 to #336780) gave ×0.9989, which is within the grader's noise and not significant on its own. The correction marg2 removes is 5–20× *more accurate* than no correction in every band \ev{E17-21}, so its gain must come from compensation inside the pair expansion. Isolated substitutions that are more correct fail: the true $K22$; covariant births (×351); masking the first-order marginal rows (×1.60); and, in this chain, the scale-mode extraction of il.miscusi [4] (×1.47 and worse) \ev{E17-41}. yeongjunyoo, by contrast, shipped extraction as part of a combined change (with compression and a retrained corrector) that graded ×0.952 in their chain [14].

::: {.note title="The principle, as a hypothesis"}
Respect the co-adapted levels; replace shapes. "More correct terms hurt" is too strong: LOT's exact shape was a more correct term, and it paid because its level was held fixed.
:::

Kuznetsov stated the consistency principle and the marg2 instance [3]. LuDoe, megan_park and yeongjunyoo compared such results in forum topic 18251, on chains that inherit V29's $\lambda$ table. Luna's Ward-defect diagnostic reaches the same conclusion about chains truncated at the fourth cumulant [11]. LuDoe, megan_park and pranay212 found that truer fourth-cumulant pieces made their chains worse [13, 15, 16], while yeongjunyoo found that injecting the true $\kappa_4$ alone helps theirs [14], consistent with the exact-chain row in Table 1. For the observations that truer $\kappa_4$ pieces hurt, neopok919 proposes a mechanism: that $\kappa_4$ born from an incomplete $\kappa_3$ adds error [21]. Specific to this work: the per-band rescaling audit of every term group, the $K22$ loop, the holdout- and grader-confirmed removals, and the observation that the shape escapes the compensation.

# How the claims are checked: evaluation

By the final week, real improvements were 0.1–1% of the score. One network's final MSE moves by about 1% when a discrete decision flips for float-level reasons: the same estimator file on two machines differs in raw MSE with a per-network standard deviation of 0.90%, but in FLOPs by only 0.003% \ev{E05-04}. Every promotion therefore went through paired validation on networks not used for fitting: same networks and machine for baseline and candidate, a sum-ratio of per-network scores with a delta-method standard error, a screen on 100 networks ($z\le-1.5$) and a confirmation on 200 others ($z\le-2.0$ and a ratio $\le0.9995$, stricter when several candidates compete).

**The local predictor is calibrated against the grader.** Over 14 pairs of consecutive graded builds, the graded ratio matched the local paired prediction with $\chi^2 = 15.0$ on 14 degrees of freedom (rms $z$ = 1.04) \ev{05/calibration}. Four pairs of builds with identical predictions bound the grader's own noise at about 0.07% (sd of the log ratio; an order-of-magnitude estimate from four pairs).

**A holdout caught false wins.** On 17 October, 300 public networks (500–799) that no earlier decision had used became a holdout (networks 500–699 were later used to fit the final HZ weights and head; 700–799 served only for screens and confirmations). It confirmed marg2, supported $D^2$ halving ($z=-2.3$), and exposed selection-noise wins, among them a marg2 threshold that read ×0.99887 ($z=-2.2$) on reused networks and ×0.99994 on the holdout \ev{E17-H06, E17-H07}. Later changes were confirmed on other unused sets, for example the non-linear head on networks 850–999.

**Graded knock-outs reproduce the local ablations for LOT and $D21$.** On 18 October, three submissions went to single-variable controls of the then-final build, each a real build with one knob default changed and never nominated (Figure 4). Paired with s43 over the same 50 public MLPs, the LOT and HZ knock-outs are worse on every MLP ($z \ge 16$ for both raw and score), and making every $D21$ row exact changes the raw MSE by less than its error ($z=-0.8$) while worsening the score on 49 of 50 MLPs. The graded raw ratios agree with the local 60-network ablations for LOT and $D21$ ($z = -0.3$ and $0.9$); HZ's graded effect is smaller ($z = -2.1$). `checks/recompute.py` recomputes all of these from the per-MLP grader scores.

![**Official-grader knock-outs against local ablations.** s43 with one component removed, graded once on the public 50 MLPs (bars; 95% intervals paired over MLPs), next to the matching local ablation on build s40a over 60 networks (dots). Hollow dots: local FLOPs are approximate (lazily built plans). The LOT-off control also disables HZ; the bottom row compares the LOT-off and HZ-off controls, which isolates LOT in a stack without HZ.](fig_knockouts.pdf){width=100%}

| control on s43 (#336780) | id | raw MSE (se) | FLOPs | score | MLPs worse | local raw (60 nets) |
|:---------------------------|:--------|---------------:|--------:|--------:|-------:|-------------:|
| LOT off (`PORTLOT=0`; disables HZ too) | #336912 | ×1.358 (0.012) | ×1.093 | ×1.484 | 50/50 | ×1.365 |
| HZ off (`HZ_ON=0`) | #336911 | ×1.228 (0.009) | ×0.940 | ×1.154 | 50/50 | ×1.262 |
| all $D21$ rows exact (`D21_CUT=99`) | #336913 | ×0.997 (0.004) | ×1.083 | ×1.080 | 49/50 | ×0.992 |

: Graded single-variable controls, paired with s43 over the 50 public MLPs (sum-ratios, delta-method standard errors of the log ratio; "MLPs worse" counts the score). Evidence: E18-201, E18-202, E18-GC02..GC04; per-MLP grader telemetry.

# Limitations, negative results, and next steps

**Limitations.** Lean results are on twelve development networks. The graded controls are single runs on the 50 public MLPs, built on s43 rather than #337608. Ablations remove one component from a co-adapted stack and measure removal cost, not standalone value; they are not additive. LOT's deployable form is not fit-free: it borrows V29's fitted level. HZ's features are collinear, and HZ helps only when fitted to the chain's own error rather than to $\kappa_4$ truth, which suggests that some compensation also acts on the per-neuron pattern. The claim is only that LOT's computed shape transfers when the level is held fixed. Only $n=1024$, $L=16$ were tested. The attribution of Figure 5 (Appendix B) follows the graded trajectory and is not an ablation.

**Negative results.**

| idea | expected | measured | reading |
|:-----------------------------|:-----------------|:-----------------------|:---------------------------|
| inject true $K22$ ($N=10^8$) | better closure | ≈ ×1.25 | breaks the compensation loop |
| scale-covariant (Ward) repairs | consistency | ×1.11 to ×351 | a fourth-order truncation cannot be covariant |
| cheapen HZ's features by an accurate closure ($r\ge0.995$) | FLOPs $-0.6$ to $-2.4\%$ | every variant loses after refit (×1.003–1.040) | HZ needs exact features |
| ARC's current term program (drop 22 pair terms) | simpler, fewer FLOPs | ×1.0109 | consistency is term-specific |
| more data for the learned head | rising returns | ×0.9997 (graded) | saturated, as found by others [18, 7] |
| rescale the mean program's $\kappa_4$ toward truth, per band | truer level | ×1.04 to ×1.22 | the level is compensated |

: Selected negative results; Appendix D lists more.

**Next steps.** (i) A fit-free production LOT. Evaluate the exact chain on the top rows, where HZ already computes it, and test the corrected same-hub weight of Appendix A, which fixes LOT's own level on a toy network. Then measure how much of the borrowed level is still needed. (ii) Replace HZ's exact-term weights by their theoretical value 1 and refit only the approximate pieces; if the loss stays below 0.2%, the correction becomes nearly fit-free. (iii) Width scaling ($n=256,512$) and a check of the identities against ARC's exact $K=4$ reference at small size.

# Credits, LLM use, and disclosures

**Credits.** [1] ARC (Wu, Lecomte, Winer, Robinson, Hilton, Christiano), cumulant propagation, term tables, `factor_k4` (arXiv:2605.05179; MIT code). [2] 504aldo, V29 (MIT), the base of this estimator. [3] Artyom Kuznetsov (Artyom\_Mazur), marg2 rule, Proposition 1 (random-projection view of error transport), "3+2" tree (forum 18249). [4] il.miscusi, scale-mode extraction (18248). [5] LordProcciuto (repository siming767-creator/whestbench-lordprocciuto, MIT): the FIJA2 frozen join, its follow-up trims and the OLD\_HOT1 old-tier legs, ported with their notice. [6] Hisernberg, learned corrector trained end to end. [7] hyojun_kwon, deployed-state calibration and final-loss training (18255). [8] kai_leal_gonzalez, two-term (3,1) closure (18246): its 14-number $b_1$ table is copied from their MIT repository (al07-alt/whest-p2-k31-two-term), the code is original to this work. Yinita (team Niko), the H18 per-row metric, re-implemented with credit. [9] mliston, learned readouts fitted on the estimator's own trajectories (18287). [10] Puffi, response closure (Phase 1; used in 18272 §6). [11] Z. Dogus Ozel (Luna), Ward-defect diagnostic, aligned chaos model and whitened-mode side term (18281). [12] Susupro, readout observability (Prop. 2) and the W, W∘W corrector (18254). [13] LuDoe (posts in forum topic 18251). [14] yeongjunyoo (18284). [15] megan_park, fit-free survival law $S(\ell)$, which is the closed form of the depth attenuation measured here, and the measurement of depth memory (18282; also posts in 18251). [16] pranay212 (18273). [17] huang_chung_yi (18285). [18] michael_anderson7 (18272). [19] Nurlanggs, edge history, which realises the depth-memory term this work had left as a hypothesis (18252). [20] Hack2Publish (mdhamidhosen, esfer\_sami, foyshal\_sk), the $\kappa_4$ pattern-versus-level oracle split, the V55/V56 $\kappa_4$-pattern corrector with the cubic-star feature, the $\kappa_3$-dressed C·T readout, the Hermite ceiling ladder, and the observation that closure-input errors cancel (18290). [21] neopok919, readers for $\kappa_4$ born from $\kappa_3$ and the incomplete-$\kappa_3$ mechanism (18293). [22] AndreasHad04, linearised mean propagation with the per-layer law $1/2+\arcsin(\rho)/\pi$, used to fit corrections to the final-layer error, and a "kurtosis bound": a lower bound on $\kappa_4$ built from the (2,1) slice of $\kappa_3$ and mixed in at fitted weights, a relative of the floor of Proposition 3 (18292). [23] marius_binner, a $\kappa_4$ star read on carried $\kappa_3$ legs, described in public submission notes (#330172, 7 September; reported in [22]). The fast-matmul schemes (Strassen–Winograd, Karstadt–Schwartz, Smirnov) are standard and shared across the field; a Strassen leaf-size idea is from a forum post by ngmitam.

**LLM use and roles.** The project was run as a directed multi-agent search, designed and steered by the author. The author set the objective, the evaluation rules and the compliance rules (no use of billing quirks, no threads or compiled code, ported code only from one approved MIT repository, with credit), and proposed research directions from their own reading of the literature and the competition forum, alongside those the agents proposed. They organised the search in rounds of parallel agents, each with a stated aim, and when agents reported a ceiling, kept the search going by suggesting new directions from their understanding of the problem. They asked on 17 October for a generalisation check before nomination, which became the untouched 300-network holdout of Section 6, allocated the rented compute, and made every submission, nomination and release decision. The agents, Claude models through Claude Code for the whole phase and an OpenAI Codex session on 2–7 October (local only, nothing submitted), wrote essentially all of the code, ran the experiments and proposed most of the individual ideas; in particular the LOT derivation was produced by a Claude agent on 1–2 October and checked numerically against Monte Carlo truth. This write-up was drafted by Claude from the project's records. Separate verification agents checked its numbers against logged runs and grader telemetry, and the discrepancies they found were corrected; a few figures from early lean experiments rest on their written records only. The LOT identities were re-derived and checked numerically on toy networks (Appendix A).

**Billing disclosure.** Since build s31 (16 October) the estimator calls `qr(mode='complete')` once per network, in the old-tier basis update (`QRV=1`, on by default in #337608). Flopscope issue #269 reports that complete QR is under-billed. In one full-size run of #337608 (287.5 GFLOP billed) the call is billed 74.9 MFLOP, 0.026% of the bill; at a LAPACK-style price it would be 172.3 MFLOP, so the under-billing is about 0.03% of the submission's bill. The alternative path (`QRV=0`) bills 240 MFLOP. The effect on the score is of that order. It is disclosed so the organisers can re-price.

**Reproducibility.** The repository contains the byte-identical graded estimator with a sha256 check, the raw data behind Table 1 and the graded knock-outs with a script that recomputes both (`checks/recompute.py`), a reading guide that maps every mechanism in this document to code, the knob settings of the three graded controls, the head-training scripts, the full LOT derivation with its numerical check, and the data and scripts behind every figure and table. The grey evidence tags refer to the project's ledger, whose cited rows are reproduced in `docs/EVIDENCE.md`.

\appendix

# Derivation of the LOT identities

This appendix condenses `docs/LOT_DERIVATION.md` in the repository, which has the full proofs, the code correspondence line by line, and the toy check (`checks/lot_toy_check.py`, numpy only, about one minute on one core). Here $c_k=\mathbb E[\mathrm{relu}^{(k)}(\zeta)]$ for $\zeta\sim\mathcal N(\mu,\sigma^2)$, so $c_1=\Phi(\alpha)$, $c_2=\phi(\alpha)/\sigma$ and $c_3=-\alpha\phi(\alpha)/\sigma^2$; the $c_3$ and $c_4$ of Section 2.1 are these divided by $3!$ and $4!$.

**Backbone.** The backbone is the linear-Gaussian chain with the chain's own means and covariances, $\zeta_l=\mu_l+W_l^\top[\mathrm{diag}(c_1^{(l-1)})\tilde\zeta_{l-1}+\eta_{l-1}]$, with independent innovations $\eta$. With propagators $P^{(l,b)}$ (the linearised map from $h_b$ to $z_l$), its cross-covariances are $G(l,b)=\Gamma^{(l,b)}=P^{(l,b)}\mathrm{diag}(c_1^{(b)})C_b$. For readout neuron $i$ of layer $l$, define the linear part $L_i=\tilde\zeta_{l,i}$, the Hadamard leg $V^b_{ij}=c^{(b)}_{2,j}P^{(l,b)}_{ij}\Gamma^{(l,b)}_{ij}$ and the functional $\psi_i=\sum_{b<l}\sum_j V^b_{ij}\tilde\zeta_{b,j}$.

**Tree level.** Expand every ReLU in Hermite polynomials around its backbone marginal. A diagram for $\kappa_r$ with excess $e=\sum_v(k_v-1)$ has $r+e$ leaves, and connecting $r$ slots needs at least $r-1$ pairings between different trees, so $e\ge r-2$. Tree level, $e=r-2$, keeps one He2 vertex for $\kappa_3$, and for $\kappa_4$ exactly three configurations: one He3 vertex (star), two He2 vertices in different slots (chain), or one He2 vertex fed by another (induced). Loops are suppressed at large width, by ARC's power counting in expectation over the weights [1, Cor. S.2.14]; for a fixed network this is a heuristic, which is why LOT was validated empirically.

**Proposition 1.** At tree level,
$$D_{3,i}=3\,\mathrm{Cov}(L_i,\psi_i),\qquad \kappa^{\mathrm{chain}}_{4,i}=12\,\mathrm{Var}(\psi_i)=12\sum_{b,b'<l}\big[V^bG(b,b')V^{b'\top}\big]_{ii},$$
$$\kappa^{\star}_{4,i}=4\sum_{b,j}c^{(b)}_{3,j}P^{(l,b)}_{ij}\big(\Gamma^{(l,b)}_{ij}\big)^3,\qquad \kappa^{\mathrm{ind}}_{4,i}=12\sum_{b,j}V^b_{ij}\,\Xi^b_{ij},\quad \Xi^b_{ij}=\kappa_3(L_i,L_i,z_{b,j})\big|_{\mathrm{tree}}.$$
*Proof sketch.* By Wick's theorem, for any first-chaos $Y$, $\mathbb E[L_iYQ_i]=\mathrm{Cov}(Y,\psi_i)$, where $Q_i$ is the He2 part of $z_{l,i}$. The He2 vertex can sit in any of 3 slots of $\kappa_3$, which gives the 3. For the chain, each pair of hubs admits $2\times2\times2$ connected pairings, giving $2\,\mathrm{Var}(\psi)$, and the two vertices occupy $\binom42=6$ slot pairs, which gives the 12. The star and induced terms follow in the same way (4 slots; $3!$ pairings of the inner leaves). $\square$

*Independent check.* For one neuron, $L+\varepsilon Q$ is a linear plus a quadratic form in a whitened Gaussian, $a^\top\xi+\varepsilon(\xi^\top M\xi-\mathrm{tr}M)$, whose cumulant generating function is exact. It gives $\kappa_3=6\varepsilon a^\top Ma+8\varepsilon^3\mathrm{tr}M^3$ and $\kappa_4=48\varepsilon^2a^\top M^2a+48\varepsilon^4\mathrm{tr}M^4$. With $\psi=2a^\top M\xi$, the terms in $a$ are exactly the trees, $3\,\mathrm{Cov}(L,\psi)$ and $12\,\mathrm{Var}(\psi)$, and the traces are the loops that tree level drops.

**Proposition 2 (relation to ARC's trees).** Transport the path (×12), star (×4) and "3+2" (×12) trees born at layer $b$ to layer $l$ and read them on the diagonal. The path gives the same-layer chain $12[V^bC_bV^{b\top}]_{ii}$, the star gives the star term, and "3+2" splits exactly into the induced term plus the cross-layer chains $24\sum_{b'<b}[V^bG(b,b')V^{b'\top}]_{ii}$. Summed over birth layers, the three trees equal chain + star + induced. The cross-source chains are therefore not a new diagram: they are the part of "3+2" that enters through an input leg. LOT's contribution is the closed form that sums all generations on the diagonal.

**Proposition 3 (the floor).** With $\rho_i=\mathrm{Corr}(L_i,\psi_i)$, $\kappa^{\mathrm{chain}}_{4,i}=\tfrac43 D_{3,i}^2/(\mathrm{var}_i\,\rho_i^2)\ge\tfrac43D_{3,i}^2/\mathrm{var}_i$, with equality when $\psi_i$ is affine in $L_i$ (for $z=u+\varepsilon\mathrm{He}_2(u)$: $D_3=6\varepsilon$, $\kappa_4=48\varepsilon^2$). It bounds the tree-level chain by the tree-level $D_3$; it is not a bound on the total $\kappa_4$. Closures of the form $b\,\kappa_3^2/\mathrm{var}$ are its constant-$\rho$ shadow, and the aligned case $\rho=1$ also appears in Luna's chaos model [11]. On a full-size development network the floor's correlation with the true $\kappa_4$ rises from 0.40 at layer 4 to 0.83–0.88 at layers 12–15, while its level stays a fraction of the truth.

**The deployable form is a response closure with a derived coupling.** For a $K$-dimensional basis $Q$ of the current layer's dominant covariance modes and $G_q=Q^\top C_lQ$, the projected variance is $\mathrm{Var}(\Pi_K\psi_i)=k_i^\top G_q^{-1}k_i$ with $k_i=\sum_b[V^b(\Gamma^{(l,b)\top}Q)]_{i,:}$. This costs $n^2K$ per source, includes every cross-source pair, and depends on $\mathrm{span}(Q)$ only. Because each row of $P$ and $\Gamma$ is linear in the readout neuron's incoming weights $w_i$, $k_{iq}=w_i^\top B_qw_i$ for $h$-side modes $B_q$, and
$$12\,\mathrm{Var}(\Pi_K\psi_i)=3\,q_i^\top\Lambda q_i,\qquad q_{iq}=w_i^\top B_qw_i,\qquad \Lambda=4\,G_q^{-1}\succeq0.$$
This is the diagonal of a $K$-mode response closure in Puffi's sense [10], with derived modes and a derived, positive semidefinite coupling. The deployed candidate adds the same-hub and star terms and omits the induced term. Its projection and same-hub parts double count, so its own level is not right; that is one reason it takes V29's level (Section 3.3).

**Why shape and level separate.** The mean program uses $\kappa_4$ only through an affine term, and to first order the chain carries mean errors to the final layer linearly, $\delta m_{\mathrm{fin}}=\sum_l T_{\mathrm{fin},l}\,\varepsilon_l$. Writing $\kappa_{4,l}=s_lf_l$ with $\overline{f_l}=1$, the final error is affine in the per-layer scalars $s_l$ for a fixed shape. The level is therefore a one-number-per-layer least-squares problem, which V29's tables solve end to end. Keeping V29's $s_l$ and replacing $f_l$ isolates the shape, to first order.

**Coincident indices.** For one neuron, $\kappa(z,z,h,h)/\sigma^4=2\Phi(1-\Phi)-2\phi^2-2\alpha\Phi\phi$, which is $\tfrac12-\tfrac1\pi=0.182$ at $\alpha=0$ against $1/\pi=0.318$ from He2 alone. At the order that matters, the exact same-hub weight is $\Phi(1-\Phi)-\phi^2$ instead of the He2 weight $\phi^2$: smaller at the kink, larger in the tails. On a true width-32 network this correction fixes LOT's own level (slope of truth on prediction 0.92–1.05, against 0.70–0.94). It has not been tested at full width or in the chain.

**Verification.** The toy check builds the chaos components explicitly on a width-8 network with an exact Gaussian backbone ($1.6\times10^7$ samples). Pooled ratios of Monte Carlo joint cumulants to the closed forms: $D_3$ $0.9996\pm0.0008$, chain $0.9991\pm0.0017$, star $0.9998\pm0.0018$, induced $0.9990\pm0.0023$. The generating-function identities, Proposition 2, the code mirror of the deployed function and the response-closure form agree to $10^{-15}$. On true width-32, depth-6 networks, same-layer chains plus star degrade from 0.79 to 0.38 in correlation with depth, while all chains plus star stay at 0.90–0.93. The derivation and the toy check were produced and run by a Claude agent; the author reviewed the result, and the numerical check is in the repository (`checks/`).

# Graded submissions

Selected graded submissions, from the grader telemetry of each (score, final-layer MSE and FLOP fraction on the public 50 MLPs; all 100/100 MLPs unless noted). Build labels are internal names (sNN numbered builds; V the wall-time-safe nominee; S$''$ a later cost build; NLHZ, G2–G5 the head builds). `docs/SUBMISSIONS.md` lists every graded submission referenced here, and the step-by-step attribution behind Figure 5 is in `figures/data/attribution.csv`.

| build | id | score | raw MSE | FLOPs/B | what changed |
|:---------------|:--------|----------:|----------:|---------:|:-------------------------------|
| V29 (504aldo) | – | 5.40e-9 | 2.13e-8 | 0.253 | public baseline (leaderboard; not a run of this work) |
| s01 | #335600 | 3.379e-9 | 1.872e-8 | 0.180 | first graded build: LOT, demand-row $D21$, late (3,1) response, product reuse |
| s08 | #335749 | 3.149e-9 | 1.717e-8 | 0.183 | HZ introduced |
| s26 | #336286 | 2.342e-9 | 1.619e-8 | 0.145 | LOT at $K=12$ |
| s40 | #336728 | 2.160e-9 | 1.619e-8 | 0.133 | Edgeworth means for dropped rows (MID\_EDGE) |
| s42 | #336749 | 2.143e-9 | 1.607e-8 | 0.133 | marg2 |
| s43 | #336780 | 2.144e-9 | 1.607e-8 | 0.133 | $D^2$ halving; base of the three controls |
| s43 + `HZ_ON=0` | #336911 | 2.475e-9 | 1.974e-8 | 0.125 | control: HZ off |
| s43 + `PORTLOT=0` | #336912 | 3.180e-9 | 2.181e-8 | 0.146 | control: LOT off |
| s43 + `D21_CUT=99` | #336913 | 2.316e-9 | 1.602e-8 | 0.145 | control: all $D21$ rows exact |
| V | #337120 | 2.140e-9 | 1.588e-8 | 0.135 | call-count trims and a wall-clock guard; nominated as the wall-time-safe option |
| S$''$ | #337231 | 2.090e-9 | 1.587e-8 | 0.132 | later cost work, PEEL planner |
| NLHZ | #337317 | 2.044e-9 | 1.552e-8 | 0.132 | non-linear head (single change) |
| G3 | #337411 | 2.030e-9 | 1.546e-8 | 0.131 | head revision |
| **G4** | **#337608** | **2.024e-9** | **1.541e-8** | **0.131** | **the submission of this write-up**: head fine-tuned on final-layer MSE |
| G5 | #337632 | 2.023e-9 | 1.541e-8 | 0.131 | head trained on more data; nominated for its score, although its slowest network (107.9 s) exceeded the self-imposed wall-time margin |

: Selected graded submissions.

![**From V29 to submission #337608, by kind of change.** Products of the graded score ratios (public 50 MLPs) of consecutive builds; each bar also shows its MSE and FLOP factors. The first graded build bundles LOT with other changes of this work (demand-row $D21$, a late (3,1) response) and product reuse. Bundled later steps are assigned to their dominant kind; the step-by-step table is in the repository. Four steps (two cost, two learned) have no telemetry split, so for those buckets MSE × FLOPs differs from the score factor. Starting from the leaderboard's rounded 5.40e-9, the ratios multiply to 2.0235e-9. Most of the FLOP reduction is engineering of the kind many teams share; the accuracy gains come mostly from the first bundle (which contains LOT) and from the learned corrections. LOT alone, ported onto V29, gives raw ×0.842 (Section 3.4).](fig_waterfall.pdf){width=100%}

![**Graded score over time.** Builds that set a new best (public 50 MLPs), from the first submission to #337608; s43 is the base of the three knock-outs.](fig_trajectory.pdf){width=100%}

# Engineering summary

ARC discounts performance engineering, and no credit is claimed for any of the underlying algorithms; this appendix exists so the cost side of Figure 5 can be read and discounted.

**Meter and limits.** All arithmetic goes through flopscope's analytic meter (a matmul is billed $MNK$, float64 at twice float32). Per network: wall time $\le$ 120 s, unmetered residual $\le$ 0.4 s, setup $\le$ 5 s per worker, memory 8 GiB. #337608 graded 100/100 with wall max 102.1 s, residual max 0.332 s and setup max 4.24 s.

**Where the FLOPs go** (profile of an earlier build, one network, 300 GFLOP): third-cumulant demand Grams and old-tier reads 35.2%; young-source transport 28.3%; tier-1 and tier-2 lifts 11.6%; HZ's deferred Horner pass 5.8%; the covariance product 3.3%; **LOT 1.9%**; everything else 13.8%. About 72% of the bill runs through a pooled Strassen–Winograd engine \ev{08/README}.

**Engine.** Strassen–Winograd products issued as ordinary metered operations in preallocated in-place buffers (bill $-1.66\%$ on two networks, predictions bit-identical), a Karstadt–Schwartz alternative basis ($-1.54\%$), and exact dense cuts ($-0.76\%$) \ev{E08-01..04}. The same family of techniques is used by 504aldo and most of the field.

**Wall time is part of accuracy.** When first tried (build s44, 18 October), a planner change (PEEL) cut FLOPs by 0.4% but raised the grader's kernel time by 4.7% (medians over the MLPs where HZ ran). On one slow grader worker the build's wall-clock HZ time guard then dropped HZ on 16 MLPs, and the build graded 1.75% worse \ev{E18-W09, E18-W10, E18-W13}. PEEL returned in later builds and is on in #337608, whose graded telemetry shows no HZ skips. Deliberately forced guard tests on the grader also showed that guards reading the meter's client context never arm in grading; only wall-clock (perf_counter) guards do. #337608 carries three wall-clock HZ guards (103.5, 107.5 and 117 s; none fired in its graded run) and two guards that read the client context and therefore never arm in grading. huang_chung_yi independently built a clock-only wall guard that worked on the grader [17].

**Residual time.** Each metered call costs about 11 µs of unmetered client time, so call counts are budgeted; in #337608's graded run the five largest residuals are MLPs 0–4, consistent with the first network on each of the grader's five workers, where plans are built (the telemetry does not record workers) \ev{E08-06, E17-70}.

# Negative results catalogue

Selected from the 61 entries of `docs/negative_results.csv` in the repository. "Before" is the offline, lean or pre-refit estimate; "after" is the measurement on unused networks or in production.

| idea | before | after | reading |
|:-----------------------------|:--------------------|:--------------------|:------------------------------|
| pure sampling (QMC, antithetic, structured frames) | best 1.79× over plain MC | 230–370× too weak | the analytic chain is ~$10^2$× better at equal FLOPs |
| generic learned correctors (GBDT, MLP, GNN) | in-sample gains | ×1.01 on a regenerating chain | per-neuron features see only the coherent part |
| low-rank covariance closure | – | ×2.2 ($k=384$) to ×41 ($k=8$) | covariance participation ratio 53–110 (layers 14 and 8) |
| truncate source lifetimes to 2–4 layers | – | 32–70× worse | old sources carry 45–50% of the $\kappa_3$ amplitude |
| sensitivity-chosen (adjoint) basis for old sources | principled | flat or worse | (the author's reading) adjoint weights nearly uniform for random $W$ |
| Monte Carlo estimate of LOT's chain (256 samples) | corr. 0.95–0.98 offline | +0.6% in chain | per-neuron noise destroys the shape |
| young "3+2" trees on top of LOT | ×0.970 (lean) | ×1.0004 | splits into LOT's induced term + cross terms |
| learned $\kappa_4$-diagonal corrector | $-4.5\%$ before HZ | ×0.997 after refit | absorbed by HZ |
| scale-mode extraction (18248) in this chain | – | ×1.022 even with oracle $\varepsilon$ | this chain already carries the mode (scoped: works in others' chains) |
| analytic scale-mode face for the old $\kappa_3$ tier | 85% of its energy | ×15.3 | the off-mode 15% is real |
| mask first-order marginal rows (MARG1) | a small gain | ×1.599 | only the second-order leg is over-counted |
| per-band scalings of pair term groups | coordinate descent | cliffs to ×7.27 | compensation loop (Section 5) |
| Mehler $m=3$ terms | ×0.994 (lean) | ×0.998 (200 networks); shipped as a small gain | layers 2–14 gain, the final layer gives most of it back |
| perfect correction of the true gain-mode slice (layer 14) | 0 to $-1.2\%$ expected | ×0.999–1.008 | the slice is not where the error is |

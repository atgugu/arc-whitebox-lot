#!/usr/bin/env python3
"""Draw the write-up figures from data/*.csv only.

    python3 make_figures.py

Writes fig_<name>.pdf (vector, for the paper) and fig_<name>.png (200 dpi, for the README) next to this
file. The paper's captions carry the titles, so the PDFs have no in-figure title or subtitle; the PNGs
keep them (switch with TITLED below). Deterministic: no randomness, fixed sizes, PDF timestamps stripped.
Full-width figures are 6.5 in wide (the paper's text width, A4 with 2.25 cm margins), so they print at
100% with 6.3-8.5 pt text. Font: Liberation Sans (metric clone of Helvetica, close to the paper's
TeX Gyre Heros headings), falling back to DejaVu Sans if it is not installed.
"""
import csv
import datetime as dt
import math
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text  # noqa: E402
import matplotlib.ticker as mt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
FULL = 6.5    # in, figure width
HEAD = 0.42   # in, height of the title + subtitle block (PNG only)
TITLED = {'pdf': False, 'png': True}

# Palette: categorical slots 1-2 of the validated reference palette (both pass every check on #ffffff:
# CVD dE 24.7, normal-vision dE 33.6, contrast >= 3:1), plus neutral chrome and de-emphasis tones.
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, WHITE = '#e1e0d9', '#c3c2b7', '#ffffff'
NEUTRAL_FILL = '#f0efec'
SLATE, MIDGRAY, LIGHTGRAY, LIGHTEDGE = '#8da4bd', '#a6a59e', '#dcdbd5', '#b4b3ab'


def tint(hex_color, a):
    """The hue at opacity `a` over white, as an opaque colour (a wash, never a saturated block)."""
    h = hex_color.lstrip('#')
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return '#' + ''.join('%02x' % round(255 - a * (255 - c)) for c in rgb)


BLUE_T, ORANGE_T = tint(BLUE, 0.10), tint(ORANGE, 0.10)

plt.rcParams.update({
    'font.family': ['Liberation Sans', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic',
    'mathtext.bf': 'Liberation Sans:bold', 'mathtext.sf': 'Liberation Sans',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.facecolor': WHITE,
    'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.size': 0, 'ytick.minor.size': 0,
    'grid.color': GRID, 'grid.linewidth': 0.5, 'grid.linestyle': '-',
    'hatch.linewidth': 0.7,
    'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.handletextpad': 0.5, 'legend.columnspacing': 1.4,
    'figure.facecolor': WHITE, 'savefig.facecolor': WHITE,
    'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
})


def load(name):
    with open(os.path.join(DATA, name), encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def check_layout(fig, name):
    """Report any text that leaves the figure or overlaps another text (QA only; never changes the figure)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    boxes = []
    for t in fig.findobj(matplotlib.text.Text):
        if not (t.get_visible() and t.get_text().strip()):
            continue
        bb = matplotlib.text.Text.get_window_extent(t, r)  # text box only (an Annotation's own extent includes its arrow)
        if bb.width <= 0:
            continue
        boxes.append((t, bb))
        if bb.x0 < fb.x0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y0 < fb.y0 - 0.5 or bb.y1 > fb.y1 + 0.5:
            print('  [%s] text outside figure: %r' % (name, t.get_text()[:60]))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 < b.x1 - 1 and b.x0 < a.x1 - 1 and a.y0 < b.y1 - 1 and b.y0 < a.y1 - 1:
                print('  [%s] texts overlap: %r / %r' % (name, boxes[i][0].get_text()[:40], boxes[j][0].get_text()[:40]))


def render(draw, name):
    """Draw twice: untitled for the paper PDF, titled for the README PNG."""
    for fmt in ('pdf', 'png'):
        fig = draw(TITLED[fmt])
        check_layout(fig, '%s.%s' % (name, fmt))
        if fmt == 'pdf':
            fig.savefig(os.path.join(HERE, name + '.pdf'), metadata={'CreationDate': None, 'ModDate': None})
        else:
            fig.savefig(os.path.join(HERE, name + '.png'), dpi=200, metadata={'Software': None})
        plt.close(fig)
    print('wrote %s.pdf, %s.png' % (name, name))


def new_fig(height, titled, title, sub):
    """A figure `height` in tall plus the title block when titled. Returns (fig, y) with y(inches from the
    top of the content area) -> figure fraction."""
    head = HEAD if titled else 0.0
    fig = plt.figure(figsize=(FULL, height + head))
    H = height + head
    if titled:
        fig.text(0.008, 1 - 0.04 / H, title, ha='left', va='top', fontsize=8.5, color=INK, fontweight='bold')
        fig.text(0.008, 1 - 0.215 / H, sub, ha='left', va='top', fontsize=7.3, color=INK2)
    return fig, (lambda inches: 1 - (inches + head) / H)


def ratio_axis(ax, lo, hi, major, minor=()):
    ax.set_xscale('log')
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(mt.FixedLocator(major))
    ax.xaxis.set_major_formatter(mt.FixedFormatter(['×%g' % t for t in major]))
    ax.xaxis.set_minor_locator(mt.FixedLocator(list(minor)))
    ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.grid(axis='x', which='both', zorder=0)
    ax.set_axisbelow(True)


def panel_title(ax, letter, text):
    ax.set_title(r'$\bf{%s}$  %s' % (letter, text), loc='left', fontsize=8, color=INK, pad=4)


def no_left_spine(ax):
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)


# ----------------------------------------------------------------------------- fig_knockouts
def fig_knockouts(titled):
    from matplotlib.legend_handler import HandlerTuple
    BLUE_DERIVED = '#86b6ef'   # sequential blue step 250: a paired ratio of two graded controls, not a submission
    rows = load('knockouts.csv')
    controls = sorted({(int(r['order']), r['kind'], r['control'], r['knob'], r['submission']) for r in rows})
    get = {(r['control'], r['metric']): r for r in rows}
    ypos = {0: 3.4, 1: 2.4, 2: 1.4, 3: 0.0}   # extra gap before the derived row
    fig, y = new_fig(3.1, titled, 'Graded knock-outs: controls on build s43 (#336780)',
                     's43 with one knob changed, graded on the public 50 MLPs (bars, paired 95% CI), vs local ablations on '
                     's40a, 60 nets (dots).')
    left, right, gap, bottom, top = 0.19, 0.985, 0.03, y(2.88), y(0.61)
    w = (right - left - 2 * gap) / 3
    panels = [('raw', 'a', 'Raw MSE ratio (off / on)'), ('flops', 'b', 'FLOPs ratio'), ('score', 'c', 'Score ratio (≈ raw × FLOPs)')]
    for k, (metric, letter, ttl) in enumerate(panels):
        ax = fig.add_axes([left + k * (w + gap), bottom, w, top - bottom])
        ratio_axis(ax, 0.9, 1.9, [1.0, 1.2, 1.4, 1.6], [0.9, 1.1, 1.3, 1.5, 1.7, 1.8])
        ax.set_ylim(-0.42, 4.04)
        no_left_spine(ax)
        ax.axvline(1.0, color=INK2, lw=0.7, zorder=2)
        ax.axhline(0.72, color=AXIS, lw=0.6, zorder=1)
        panel_title(ax, letter, ttl)
        for order, kind, label, knob, sid in controls:
            yy = ypos[order]
            r = get[(label, metric)]
            g, yb = float(r['graded_ratio']), yy + 0.17
            ax.barh(yb, g - 1.0, left=1.0, height=0.28, color=BLUE if kind == 'control' else BLUE_DERIVED, zorder=3, linewidth=0)
            xt = max(g, 1.0)
            if r['graded_ci95_lo']:   # paired 95% CI over the public 50 MLPs (raw and score panels)
                lo, hi = float(r['graded_ci95_lo']), float(r['graded_ci95_hi'])
                ax.plot([lo, hi], [yb, yb], color=INK, lw=0.8, zorder=4, solid_capstyle='butt')
                for e in (lo, hi):
                    ax.plot([e, e], [yb - 0.07, yb + 0.07], color=INK, lw=0.8, zorder=4, solid_capstyle='butt')
                xt = max(hi, 1.0)
            ax.text(xt * 1.015, yb, '×%.4f' % g, ha='left', va='center', fontsize=6.8, color=INK2, zorder=5)
            if r['graded_mlps_worse']:
                ax.text(xt * 1.015, yy - 0.06, '%s/%s worse' % (r['graded_mlps_worse'], r['graded_mlps']), ha='left', va='center',
                        fontsize=6.0, color=INK2, zorder=5)
            if not r['local_ratio']:
                continue          # the derived row has no local arm
            loc, yl = float(r['local_ratio']), yy - 0.33
            if r['local_flops_approx'] == 'False':
                lo, hi = float(r['local_ci95_lo']), float(r['local_ci95_hi'])
                ax.plot([lo, hi], [yl, yl], color=ORANGE, lw=1.2, zorder=3, solid_capstyle='butt')
                ax.plot([loc], [yl], 'o', ms=4.6, mfc=ORANGE, mec=WHITE, mew=0.8, zorder=4)
            else:
                ax.plot([loc], [yl], 'o', ms=4.2, mfc=WHITE, mec=ORANGE, mew=1.1, zorder=4)
        if k == 0:
            for order, kind, label, knob, sid in controls:
                yy = ypos[order]
                name, _, paren = label.partition(' (')
                sub = ['(' + paren] if paren else []
                sub.append('%s · #%s' % (knob, sid) if kind == 'control' else knob)
                ax.text(-0.035, yy + 0.17, name, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=7.5, color=INK)
                for i, t in enumerate(sub):
                    ax.text(-0.035, yy - 0.08 - 0.22 * i, t, transform=ax.get_yaxis_transform(), ha='right', va='center',
                            fontsize=6.5, color=INK2)
    handles = [(Patch(facecolor=BLUE, edgecolor='none'), Line2D([0], [0], color=INK, lw=0.8)),
               (Patch(facecolor=BLUE_DERIVED, edgecolor='none'), Line2D([0], [0], color=INK, lw=0.8)),
               Line2D([0], [0], color=ORANGE, lw=1.2, marker='o', ms=4.6, mfc=ORANGE, mec=WHITE),
               Line2D([0], [0], color='none', marker='o', ms=4.2, mfc=WHITE, mec=ORANGE, mew=1.1)]
    labels = ['graded control on s43 (#336780): 95% CI, n/50 MLPs worse', 'derived: paired ratio of two graded controls',
              'local ablation on s40a, 60 nets (95% CI)', 'local, approximate FLOPs (lazily built plans)']
    fig.legend(handles=handles, labels=labels, handler_map={tuple: HandlerTuple(ndivide=1)}, loc='upper left',
               bbox_to_anchor=(0.008, y(0.04)), ncol=2, borderaxespad=0, labelspacing=0.45)
    return fig


# ----------------------------------------------------------------------------- fig_lot_shape_level
def fig_lot_shape_level(titled):
    rows = sorted(load('lot_shape_level.csv'), key=lambda r: int(r['order']))
    FAINT = tint(BLUE, 0.42)
    fig, y = new_fig(3.12, titled, 'Theory supplies the shape; at budget cost, the level must come from the table',
                     'κ$_4$ variants against V29 on the same networks (lean runner); paired sum-ratios of final-layer MSE, 95% CI.')
    bottom, top = y(2.63), y(0.46)
    ax = fig.add_axes([0.223, bottom, 0.331, top - bottom])   # 1.45 in label column, 2.15 in plot
    ratio_axis(ax, 0.66, 1.6, [0.7, 0.8, 0.9, 1.0, 1.2, 1.4])
    ypos = {0: 10.0, 1: 8.95, 2: 8.2, 3: 7.5, 4: 6.75, 5: 5.75, 6: 4.35, 7: 3.6, 8: 2.9, 9: 2.15}
    ax.set_ylim(1.7, 10.55)
    no_left_spine(ax)
    ax.axvline(1.0, color=INK2, lw=0.7, zorder=2)
    ax.text(1.0, 10.55, ' V29 memoryless κ$_4$ = ×1', ha='left', va='top', fontsize=6.8, color=INK2)
    T = ax.get_yaxis_transform()
    pct = lambda v: ('%+.1f' % ((v - 1) * 100)).replace('-', '−')  # noqa: E731
    for r in rows:
        yy, role = ypos[int(r['order'])], r['role']
        if role == 'group':
            nets = 'networks %s' % r['nets'].replace('-', '–')
            sub = [nets + ' · ' + r['cost']] if r['group'] == 'exact' else ['+ same-hub + star', nets + ' · ' + r['cost']]
            ax.text(-0.025, yy, r['label'], transform=T, ha='right', va='center', fontsize=7.5, color=INK, fontweight='bold')
            for i, t in enumerate(sub):
                ax.text(-0.025, yy - 0.42 - 0.36 * i, t, transform=T, ha='right', va='center', fontsize=6.5, color=INK2)
            continue
        v, lo, hi = float(r['ratio']), float(r['ci_lo']), float(r['ci_hi'])
        if role == 'diff':   # paired V29-level vs own-level difference, on its own row under the pair
            ax.text(0.675, yy, 'V29 vs own level: %s%% [%s, %s], %s nets' % (pct(v), pct(lo), pct(hi), r['nets_better']),
                    ha='left', va='center', fontsize=6.6, color=INK, zorder=5,
                    bbox=dict(boxstyle='square,pad=0.12', fc=WHITE, ec='none'))
            continue
        faint = role in ('reference', 'port')
        col = (BLUE if r['level'] == 'V29 level' else ORANGE) if not faint else FAINT
        label = r['label']
        if role == 'port':
            label = 'production port, %s networks' % r['nets_better'].split('/')[1]
        ax.text(-0.025, yy, label, transform=T, ha='right', va='center', fontsize=7.2 if not faint else 6.6,
                color=INK if not faint else INK2)
        if role == 'port':
            ax.plot([lo, hi], [yy, yy], color=BLUE, lw=1.0, zorder=3, solid_capstyle='butt')
            ax.plot([v], [yy], 'D', ms=4.4, mfc=WHITE, mec=BLUE, mew=1.1, zorder=4)
        else:
            ax.barh(yy, v - 1.0, left=1.0, height=0.5, color=col, zorder=3, linewidth=0)
            wc = INK if not faint else MUTED
            ax.plot([lo, hi], [yy, yy], color=wc, lw=0.8, zorder=4, solid_capstyle='butt')
            for e in (lo, hi):
                ax.plot([e, e], [yy - 0.1, yy + 0.1], color=wc, lw=0.8, zorder=4, solid_capstyle='butt')
        ax.text(1.03, yy, '×%.3f' % v, transform=T, ha='left', va='center', fontsize=7.2 if not faint else 6.6,
                color=INK if not faint else INK2)
        ax.text(1.03 + 0.16, yy, '%s nets' % r['nets_better'], transform=T, ha='left', va='center', fontsize=6.5, color=INK2)
    ax.set_xlabel('final-layer raw MSE relative to V29 (log scale; paired sum-ratio, 95% CI)\n'
                  'production port: flopscope on V29 at K = 32, as reported', fontsize=6.8, color=INK2, labelpad=2, linespacing=1.3)
    panel_title(ax, 'a', 'Matched shape × level grid')
    handles = [Patch(facecolor=BLUE, label="level from V29's fitted per-layer table"),
               Patch(facecolor=ORANGE, label="LOT's own level, no fitted constant")]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.008, y(0.04)), ncol=2, borderaxespad=0)

    # panel b: neuron-to-neuron variance explained, per layer (E01-09)
    pv = load('lot_variance_explained.csv')
    bx = fig.add_axes([0.752, bottom, 0.162, top - bottom])
    for model, col, lab in (('LOT (chain+star+induced)', BLUE, 'LOT'), ('V29 memoryless core', MUTED, 'V29 core')):
        pts = sorted((int(r['layer']), float(r['shape_r2'])) for r in pv if r['model'] == model)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        bx.plot(xs, ys, color=col, lw=1.6, zorder=3)
        bx.plot(xs, ys, 'o', ms=3.0, mfc=col, mec=WHITE, mew=0.6, zorder=4)
        bx.text(xs[-1] + 0.7, ys[-1], lab, ha='left', va='center', fontsize=7, color=INK)
    bx.set_xlim(3.4, 15.6)
    bx.set_ylim(0.5, 1.0)
    bx.set_xticks([4, 6, 8, 10, 12, 14])
    bx.yaxis.set_major_locator(mt.FixedLocator([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    bx.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%d%%' % round(v * 100)))
    bx.grid(axis='y', zorder=0)
    bx.set_axisbelow(True)
    bx.set_xlabel('layer (MLP 0, Monte Carlo truth)', fontsize=6.8, color=INK2, labelpad=2)
    panel_title(bx, 'b', 'κ$_4$ shape explained (R$^2$)')
    return fig


# ----------------------------------------------------------------------------- fig_hz_recovery
def fig_hz_recovery(titled):
    tf = load('hz_theta_two_fits.csv')
    summ = {(r['fit'], r['feature']): r for r in load('hz_theta_two_fits_summary.csv')}
    live = {fit: [int(x) for x in summ[(fit, 'ind')]['live_layers'].split()] for fit in ('g4', 's40a')}
    STYLE = {'g4': dict(ls='-', marker='o', mfill=True, ms=3.8, lw=1.6),
             's40a': dict(ls=(0, (3.2, 1.6)), marker='s', mfill=False, ms=3.4, lw=1.1)}
    fig, y = new_fig(2.4, titled, 'HZ’s fitted weights recover the theory on its exact terms',
                     'Two independent least-squares fits of the HZ corrector, per layer. The features are collinear.')
    bottom, top = y(2.07), y(0.47)

    def series(ax, fit, feat, col, z=3, label_lw=None):
        st = STYLE[fit]
        xs = list(range(5, 16))
        val = {int(r['layer']): float(r['%s_%s' % (fit, feat)]) for r in tf}
        ys = [val[l] if l in live[fit] else float('nan') for l in xs]   # layers with K = 0 are off, not fitted
        ax.plot(xs, ys, color=col, lw=label_lw or st['lw'], ls=st['ls'], zorder=z, dash_capstyle='butt')
        ax.plot(xs, ys, st['marker'], ms=st['ms'], mfc=col if st['mfill'] else WHITE, mec=col if not st['mfill'] else WHITE,
                mew=0.9 if not st['mfill'] else 0.6, zorder=z + 1, ls='none')

    def ms(fit, feat, digits=3):
        r = summ[(fit, feat)]
        return ('%.*f ± %.*f' % (digits, float(r['mean']), digits, float(r['sd_sample']))).replace('-', '−')

    # panel a: the two exact LOT terms
    ax = fig.add_axes([0.07, bottom, 0.47, top - bottom])
    ax.axhline(1.0, color=INK2, lw=0.7, zorder=2)
    ax.text(15.35, 1.0, 'theory\n= 1', ha='left', va='center', fontsize=6.5, color=INK2, linespacing=1.1)
    for feat, col in (('ind', BLUE), ('chd', ORANGE)):
        for fit in ('s40a', 'g4'):
            series(ax, fit, feat, col)
    ax.set_xlim(4.6, 15.3)
    ax.set_ylim(0.6, 1.32)
    ax.set_xticks(range(5, 16))
    ax.yaxis.set_major_locator(mt.FixedLocator([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]))
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('layer', fontsize=7, color=INK2, labelpad=2)
    ax.set_ylabel('fitted weight', fontsize=7, color=INK2)
    for feat, col, txt, yy in (('ind', BLUE, 'induced κ$_3$ term', 1.265), ('chd', ORANGE, 'cross-source chain', 0.645)):
        ax.plot([4.85], [yy], 'o', ms=3.8, mfc=col, mec=WHITE, mew=0.6, zorder=5)
        ax.text(5.05, yy, '%s:  G4 %s,  earlier %s' % (txt, ms('g4', feat), ms('s40a', feat)), fontsize=6.8, color=INK,
                ha='left', va='center')
    ax.axvspan(5.6, 6.4, ymin=0.1, ymax=0.84, color=NEUTRAL_FILL, lw=0, zorder=0)
    ax.text(6.0, 1.16, 'G4: off\n(K = 0)', ha='center', va='center', fontsize=6.0, color=INK2, linespacing=1.1, zorder=5)
    panel_title(ax, 'a', 'The two exact LOT terms')

    # panel b: V29's memoryless diagonal (reference 0) with the approximate pieces faint
    bx = fig.add_axes([0.665, bottom, 0.255, top - bottom])
    bx.axhline(0.0, color=INK2, lw=0.7, zorder=2)
    bx.axvspan(5.6, 6.4, color=NEUTRAL_FILL, lw=0, zorder=0)
    faint = [('star', 'star', 1.33), ('same', 'same-hub', 0.36), ('proj', 'projected', 0.16), ('k3sq', '(4/3) D$_3^2$/var', -0.585)]
    for feat, lab, ylab in faint:
        series(bx, 'g4', feat, LIGHTEDGE, z=2, label_lw=1.0)
        bx.text(15.45, ylab, lab, ha='left', va='center', fontsize=6.3, color=MUTED)
    for fit in ('s40a', 'g4'):
        series(bx, fit, 'g4row', INK2, z=4)
    bx.text(15.45, -0.06, 'V29 diag.', ha='left', va='center', fontsize=6.5, color=INK)
    bx.set_xlim(4.6, 15.3)
    bx.set_ylim(-0.95, 1.75)
    bx.set_xticks([5, 7, 9, 11, 13, 15])
    bx.yaxis.set_major_locator(mt.FixedLocator([-0.5, 0.0, 0.5, 1.0, 1.5]))
    bx.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: ('%g' % v).replace('-', '−')))
    bx.grid(axis='y', zorder=0)
    bx.set_axisbelow(True)
    bx.set_xlabel('layer', fontsize=7, color=INK2, labelpad=2)
    bx.text(5.0, -0.8, "V29's diagonal: G4 %s, earlier %s" % (('%.3f' % float(summ[('g4', 'g4row')]['mean'])).replace('-', '−'),
                                                             ('%.3f' % float(summ[('s40a', 'g4row')]['mean'])).replace('-', '−')),
            fontsize=6.5, color=INK, ha='left', va='center')
    panel_title(bx, 'b', "V29's memoryless κ$_4$ diagonal")
    handles = [Line2D([0], [0], color=INK2, lw=1.6, ls='-', marker='o', ms=3.8, mfc=INK2, mec=WHITE, mew=0.6,
                      label='shipped fit, G4 #337608 (layers 5, 7–15)'),
               Line2D([0], [0], color=INK2, lw=1.1, ls=(0, (3.2, 1.6)), marker='s', ms=3.4, mfc=WHITE, mec=INK2, mew=0.9,
                      label='earlier fit, used in s38–s43 (layers 5–15)'),
               Line2D([0], [0], color=LIGHTEDGE, lw=1.0, label='approximate LOT pieces (G4), faint')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.008, y(0.04)), ncol=3, borderaxespad=0)
    return fig


# ----------------------------------------------------------------------------- fig_trajectory
def fig_trajectory(titled):
    rows = load('trajectory.csv')
    v29 = [r for r in rows if r['name'] == 'V29'][0]
    pts = [r for r in rows if r['created_utc']]
    for r in pts:
        r['t'] = dt.datetime.strptime(r['created_utc'], '%Y-%m-%dT%H:%M:%S')
        r['s'] = float(r['score'])
    pts.sort(key=lambda r: r['t'])
    main = [r for r in pts if r['role'].startswith('curated') or r['role'].startswith('new best')]
    by = {r['name']: r for r in pts}
    fig, y = new_fig(2.62, titled, 'Graded score over time, from V29 to the write-up submission',
                     'Public-50 graded scores of the submissions of this work (score = raw MSE × FLOPs / budget). Failed runs, controls and '
                     'non-improving builds are omitted.')
    ax = fig.add_axes([0.075, y(2.22), 0.905, y(0.27) - y(2.22)])
    ax.set_yscale('log')
    ax.set_ylim(1.84e-9, 5.95e-9)
    yt = [2e-9, 2.5e-9, 3e-9, 4e-9, 5e-9]
    ax.yaxis.set_major_locator(mt.FixedLocator(yt))
    ax.yaxis.set_major_formatter(mt.FixedFormatter(['%.1f' % (v * 1e9) for v in yt]))
    ax.yaxis.set_minor_locator(mt.NullLocator())
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)
    t0, t1 = dt.datetime(2026, 10, 12, 6), dt.datetime(2026, 10, 22, 6)
    ax.set_xlim(t0, t1)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda x, _: '%d Oct' % mdates.num2date(x).day))  # portable (no %-d)
    ax.set_ylabel('graded score (×10$^{-9}$, log scale)', fontsize=7, color=INK2)
    ax.set_xlabel('submission time (UTC, 2026)', fontsize=7, color=INK2, labelpad=2)
    sv = float(v29['score'])
    ax.axhline(sv, color=MUTED, lw=1.0, zorder=2)
    ax.text(t0 + dt.timedelta(hours=3), sv * 1.016, 'V29, the public base (504aldo, leaderboard): %.2f' % (sv * 1e9),
            ha='left', va='bottom', fontsize=6.8, color=INK2)
    ax.step([r['t'] for r in main], [r['s'] for r in main], where='post', color=BLUE, lw=1.4, zorder=3)
    for r in main:
        ax.plot([r['t']], [r['s']], 'o', ms=3.6, mfc=BLUE, mec=WHITE, mew=0.6, zorder=4)
    for r in pts:
        if r['role'].startswith('reference'):
            ax.plot([r['t']], [r['s']], 'o', ms=3.6, mfc=WHITE, mec=INK2, mew=0.9, zorder=4)

    def note(r, text, tx, ty, ha, rel):
        ax.annotate(text, xy=(r['t'], r['s']), xytext=(tx, ty), ha=ha, va='center', fontsize=7, color=INK, zorder=5,
                    arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6, shrinkA=0, shrinkB=2.5, relpos=rel))

    D = lambda d, h=0: dt.datetime(2026, 10, d, h)  # noqa: E731
    note(by['s01'], 's01: first submission, V29 chain + a bundle incl. LOT', D(13, 6), 3.62e-9, 'left', (0, 0.5))
    note(by['s08'], 'HZ enters (s08)', D(14, 0), 3.27e-9, 'left', (0, 0.5))
    note(by['s43'], 's43 #336780: base of the graded knock-outs', D(17, 9), 1.95e-9, 'right', (1, 0.5))
    note(by['nl1 (NLHZ)'], 'NLHZ head enters (#337317)', D(19, 20), 2.42e-9, 'right', (1, 0.5))
    note(by['G4'], 'G4 #337608, 2.0236: write-up submission', D(21, 22), 2.72e-9, 'right', (0.97, 0))
    handles = [Line2D([0], [0], color=BLUE, lw=1.4, marker='o', ms=3.6, mfc=BLUE, mec=WHITE, label='graded builds that set a new best'),
               Line2D([0], [0], color='none', marker='o', ms=3.6, mfc=WHITE, mec=INK2, mew=0.9, label='s43, the build the knock-outs modify')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.075, y(0.04)), ncol=2, borderaxespad=0)
    return fig


# ----------------------------------------------------------------------------- fig_waterfall
def fig_waterfall(titled):
    wb = load('waterfall_buckets.csv')
    STY = {  # key -> (short label, face, edge, hatch)
        'first': ('First build\n(bundle\nincl. LOT)', BLUE_T, BLUE, '////'),
        'mech': ('New:\nmechanistic', BLUE, BLUE, None),
        'learned': ('New:\nlearned', ORANGE, ORANGE, None),
        'ported': ('Ported or\nre-derived', SLATE, SLATE, None),
        'cost': ('Cost\nengineering', MIDGRAY, MIDGRAY, None),
        'undoc': ('Undocumented\nbuilds', LIGHTGRAY, LIGHTEDGE, None),
    }
    fig, y = new_fig(2.75, titled, 'From V29 to the write-up submission #337608, by kind of change',
                     'Products of graded public-50 score ratios of consecutive builds, grouped by what changed (log scale).')
    ax = fig.add_axes([0.075, y(2.34), 0.915, y(0.08) - y(2.34)])
    ax.set_yscale('log')
    ax.set_ylim(1.75e-9, 7.0e-9)
    yt = [2e-9, 3e-9, 4e-9, 5e-9, 6e-9]
    ax.yaxis.set_major_locator(mt.FixedLocator(yt))
    ax.yaxis.set_major_formatter(mt.FixedFormatter(['%.0f' % (v * 1e9) for v in yt]))
    ax.yaxis.set_minor_locator(mt.NullLocator())
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel('score (×10$^{-9}$, log scale)', fontsize=7, color=INK2)
    bw = 0.58
    start, end = float(wb[0]['score_start']), float(wb[-1]['score_end'])
    xs = list(range(len(wb) + 2))
    labels = ['V29\n(504aldo,\nleaderboard)'] + [STY[r['key']][0] for r in wb] + ['#337608\n(G4)']
    # start and end are levels, not lengths: on a log axis a bar from the floor would have no meaning
    for x, v, txt in ((0, start, '%.2f' % (start * 1e9)), (len(wb) + 1, end, '%.2f' % (end * 1e9))):
        ax.plot([x - bw / 2, x + bw / 2], [v, v], color=INK, lw=2.6, solid_capstyle='butt', zorder=4)
        ax.annotate(txt, xy=(x, v), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=7.5, color=INK)
    level = start
    for i, r in enumerate(wb, start=1):
        a, b = float(r['score_start']), float(r['score_end'])
        lab, face, edge, hatch = STY[r['key']]
        ax.bar(i, b - a, bottom=a, width=bw, color=face, edgecolor=edge, linewidth=0.8 if hatch or r['key'] == 'undoc' else 0,
               hatch=hatch, zorder=3)
        ax.plot([i - 1 + bw / 2, i - bw / 2], [level, level], color=INK2, lw=0.6, zorder=2)  # connector at the shared level
        level = b
        ax.annotate('×%.3f' % float(r['score_ratio']), xy=(i, a), xytext=(0, 22), textcoords='offset points', ha='center', va='bottom',
                    fontsize=7.5, color=INK)
        ax.annotate('MSE ×%.3f\nFLOPs ×%.3f' % (float(r['raw_ratio']), float(r['flops_ratio'])), xy=(i, a), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom', fontsize=6.2, color=INK2, linespacing=1.15)
    ax.plot([len(wb) + bw / 2, len(wb) + 1 - bw / 2], [level, level], color=INK2, lw=0.6, zorder=2)
    ax.set_xlim(-0.55, len(wb) + 1.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.8, color=INK, linespacing=1.15)
    ax.tick_params(axis='x', length=0, pad=4)
    return fig


# ----------------------------------------------------------------------------- fig_schematic
def fig_schematic(titled):
    f = {r['key']: r['value'] for r in load('schematic_facts.csv')}
    H0 = 3.07                         # content height, in
    head = HEAD if titled else 0.0
    W, H = FULL, H0 + head
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    if titled:
        fig.text(0.008, 1 - 0.04 / H, 'The estimator at a glance', ha='left', va='top', fontsize=8.5, color=INK, fontweight='bold')
        fig.text(0.008, 1 - 0.215 / H, 'A layer-by-layer cumulant chain (inherited), new closures feeding it, and a learned '
                 'correction of every layer’s mean.', ha='left', va='top', fontsize=7.3, color=INK2)
    owned = []  # (text, box) pairs checked for containment below
    LS = 0.122  # body line step, in

    def box(x, y, w, h, fc, ec, lw=0.8, r=0.05):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=%g' % r, fc=fc, ec=ec, lw=lw, zorder=2)
        ax.add_patch(p)
        return p

    def lines(x, y, rows, within, size=6.4, color=INK2, weight='normal'):
        """Rows of text whose first line has its top near y; baselines are evenly spaced."""
        y0 = y - size * 0.75 / 72
        for i, s_ in enumerate(rows):
            t = ax.text(x, y0 - i * LS, s_, ha='left', va='baseline', fontsize=size, color=color, fontweight=weight, zorder=4)
            owned.append((t, within))

    def titled_box(x, y, w, h, fc, ec, head_txt, body=(), sub_gap=0.2):
        b = box(x, y, w, h, fc, ec)
        lines(x + 0.08, y + h - 0.06, [head_txt], b, size=7.3, color=INK, weight='bold')
        if body:
            lines(x + 0.08, y + h - 0.06 - sub_gap, body, b)
        return b

    def chip(x, y, w, rows, fc, ec):
        h = 0.1 + LS * len(rows)
        b = box(x, y - h, w, h, fc, ec, lw=0.6, r=0.03)
        lines(x + 0.07, y - 0.05, rows, b)
        return y - h

    def arrow(p0, p1, lw=0.8):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=7, lw=lw, color=INK2, shrinkA=0, shrinkB=0, zorder=3))

    # legend row
    for x, fc, ec, lab in ((0.05, NEUTRAL_FILL, AXIS, "inherited: ARC's cumulant propagation, 504aldo's V29 chain"),
                           (3.32, BLUE_T, BLUE, 'new: derived from theory'), (4.88, ORANGE_T, ORANGE, 'fitted to data')):
        box(x, H0 - 0.165, 0.13, 0.11, fc, ec, lw=0.8, r=0.02)
        ax.text(x + 0.18, H0 - 0.11, lab, ha='left', va='center', fontsize=6.8, color=INK2)
    top = H0 - 0.3

    # ---- column A: the inherited chain, with the four parts of its state
    ax0, aw = 2.5, 1.78
    A = titled_box(ax0, 0.1, aw, top - 0.1, NEUTRAL_FILL, AXIS, 'Cumulant chain',
                   ['layer by layer: %s layers, width %s' % (f['layers'], f['width'])], sub_gap=0.19)
    rows = [  # (y, h, symbol, text lines)
        (2.05, 0.30, 'μ', ['mean']),
        (1.56, 0.34, 'C', ['covariance, dense n × n']),
        (0.98, 0.44, 'κ$_3$', ['factored per-birth sources;', 'old ones in a rank-%s basis' % f['kappa3_old_basis_rank']]),
        (0.62, 0.27, 'κ$_4$', ['diagonal only']),
    ]
    yc = {}
    for yy, h, sym, txt in rows:
        b = box(ax0 + 0.1, yy, aw - 0.2, h, WHITE, AXIS, lw=0.6, r=0.03)
        yc[sym] = yy + h / 2
        owned.append((ax.text(ax0 + 0.19, yc[sym], sym, ha='left', va='center', fontsize=8, color=INK, zorder=4), b))
        y0 = yc[sym] + (len(txt) - 1) * LS / 2
        for i, s_ in enumerate(txt):
            owned.append((ax.text(ax0 + 0.47, y0 - i * LS, s_, ha='left', va='center', fontsize=6.4, color=INK2, zorder=4), b))
    lines(ax0 + 0.1, 0.5, ['mean readout per neuron:', 'σ[φ(α)+αΦ(α)] + c$_3$D$_3$ + c$_4$κ$_4$ + c$_{33}$D$_3^2$'], A, size=6.2)

    # ---- column B: the new closures, each feeding one part of the state
    bx0, bw = 0.05, 2.2
    titled_box(bx0, 2.13, bw, top - 2.13, BLUE_T, BLUE, 'Edgeworth means for dropped neurons',
               ['rows the active set drops get a 2-term',
                'Edgeworth mean with radial skew and kurtosis',
                'MID: layers %s · FIN: last layer, |α| > %s' % (f['mid_edge_layers'].replace('-', '–'), f['fin_edge_alpha'])])
    titled_box(bx0, 1.25, bw, 0.62, BLUE_T, BLUE, 'D21 demand rows',
               ['κ$_3$(i,i,j) slice read by the covariance update',
                'exact rows: |α| ≤ %s (about %s of %s)' % (f['d21_exact_alpha'], f['d21_exact_rows_approx'], f['width']),
                'tail rows: rank-%s on the LOT basis' % f['d21_tail_rank']])
    titled_box(bx0, 0.1, bw, 0.92, WHITE, BLUE, 'LOT κ$_4$ closure')
    yy = chip(bx0 + 0.08, 0.1 + 0.92 - 0.22, bw - 0.16, ['shape: leading-order trees, projected',
                                                         'on the top-%s covariance modes' % f['lot_modes_production']], BLUE_T, BLUE)
    chip(bx0 + 0.08, yy - 0.05, bw - 0.16, ["level: V29's fitted per-layer table"], ORANGE_T, ORANGE)
    for sym in ('μ', 'C', 'κ$_4$'):
        arrow((bx0 + bw, yc[sym]), (ax0 + 0.1, yc[sym]))
    arrow((bx0 + 1.1, 1.02), (bx0 + 1.1, 1.25), lw=0.7)
    ax.text(bx0 + 1.16, 1.135, 'basis', ha='left', va='center', fontsize=6.2, color=INK2)

    # ---- column C: the learned correction and the output
    cx0, cw = 4.53, 1.92
    C1 = titled_box(cx0, 0.84, cw, top - 0.84, WHITE, BLUE, 'HZ + NLHZ correction',
                    ['one deferred backward pass over the', 'stored chain state, for every layer'])
    yy = chip(cx0 + 0.08, top - 0.52, cw - 0.16, ['features: exact LOT chain and induced',
                                                  'terms on the most κ$_4$-sensitive rows',
                                                  'of layers %s' % f['hz_layers'].replace('-', '–').replace(', ', ' and ')], BLUE_T, BLUE)
    yy = chip(cx0 + 0.08, yy - 0.05, cw - 0.16, ['weights: HZ %s per layer + %s riders;' % (f['hz_weights_per_layer'], f['hz_rider_weights']),
                                                 'NLHZ %s head, fine-tuned on' % f['nlhz_head'].replace('x', '×'),
                                                 f['g4_final_loss']], ORANGE_T, ORANGE)
    lines(cx0 + 0.08, yy - 0.07, ["corrects each layer's mean; carried", 'forward by Φ(α)W'], C1)
    titled_box(cx0, 0.1, cw, 0.5, WHITE, INK2, 'Output', ['per-neuron means of all %s layers' % f['layers']], sub_gap=0.19)
    arrow((ax0 + aw, yc['C']), (cx0, yc['C']))
    arrow((cx0 + cw / 2, 0.84), (cx0 + cw / 2, 0.6))

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for t, b in owned:
        tb, bb = t.get_window_extent(r), b.get_window_extent(r)
        if tb.x0 < bb.x0 or tb.x1 > bb.x1 or tb.y0 < bb.y0 or tb.y1 > bb.y1:
            print('  [fig_schematic] text leaves its box: %r' % t.get_text()[:50])
    return fig


if __name__ == '__main__':
    for fn, name in ((fig_knockouts, 'fig_knockouts'), (fig_lot_shape_level, 'fig_lot_shape_level'),
                     (fig_hz_recovery, 'fig_hz_recovery'), (fig_trajectory, 'fig_trajectory'),
                     (fig_waterfall, 'fig_waterfall'), (fig_schematic, 'fig_schematic')):
        render(fn, name)

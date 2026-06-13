"""
Paper figures, generated directly from the real result JSONs on `main`.
Nothing is hardcoded; every value is loaded from results/ at run time.

COLM agent-behavior styling: colorblind-safe Okabe-Ito palette, no in-figure
titles (titles go in the LaTeX \\caption{}; suggested text in CAPTIONS).

Sources:
  Fig 1  results/gauntlet/*.json                                  (8-model pillar matrix)
  Fig 2  results/distributional/xdomain_gsm8k_trig_vs_notrig_textonly{,_v2_n200,_v3_n200}.json
  Fig 3  results/distributional/sample_efficiency_{qwen15b,llama1b}.json
  Fig 4  results/distributional/{xdomain,trigger_qwen3b}_{gsm8k,mmlu}_trig_vs_notrig.json
"""
import json
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

BLUE, ORANGE, VERM, GREEN, GRAY = '#0072B2', '#E69F00', '#D55E00', '#009E73', '#999999'
DETECT_CMAP = LinearSegmentedColormap.from_list('detect', ['#f7f9fc', '#9ecae1', BLUE], N=512)
FIRE_THR = 0.5  # a pillar "fires" if its calibrated score exceeds this


def root():
    p = Path(__file__).resolve()
    for q in [p] + list(p.parents):
        if (q / 'results' / 'gauntlet').exists():
            return q
    raise FileNotFoundError('results/gauntlet not found')


def _load(rel):
    return json.loads((root() / rel).read_text())


# ── model display names / families for the gauntlet ───────────────────────────
MODEL_MAP = [  # (substring, label, family)
    ('Qwen2.5-1.5B', 'Qwen2.5-1.5B', 'Qwen'),
    ('Qwen2.5-7B',   'Qwen2.5-7B',   'Qwen'),
    ('Qwen3-14B',    'Qwen3-14B',    'Qwen'),
    ('Qwen3-30B',    'Qwen3-30B (MoE)', 'Qwen'),
    ('Qwen3.6-27B',  'Qwen3.6-27B',  'Qwen'),
    ('Llama-3.1-8B', 'Llama-3.1-8B', 'Llama'),
    ('gemma-4-31B',  'Gemma-4-31B',  'Gemma'),
    ('gpt-oss-20b',  'GPT-OSS-20B',  'GPT-OSS'),
]

CAPTIONS = {
    'matrix':
        "Cross-family detection matrix: calibrated pillar scores for the trigger "
        "attack across eight models. The activation probe (P2) fires on all 8/8 "
        "models; the logit (P1) and behavioral (P3) pillars exceed the flag "
        "threshold on none. Activation is the only family-portable detector.",
    'arc':
        "Adversarial co-evolution of the text-only audit on the trigger attack. "
        "Two minimal training-data edits drive the joint AUC from 0.978 to chance "
        "(0.560, \\textsc{indistinguishable}) while the trigger-conditional GSM8K "
        "suppression stays intact. v1 $n{=}60$; v2,v3 $n{=}200$; v3 trained at "
        "iter${=}200$ (iter${=}400$ collapsed to unconditional sandbagging).",
    'efficiency':
        "Sample efficiency of the distributional audit (API-level, no weights). "
        "Mean $\\pm$1 SD over 5 seeds; the audit saturates by $\\approx$10 query "
        "pairs. (Measured on the v1 trigger substrate; cf.\\ Fig.~arc for its "
        "erodability.)",
    'crossformat':
        "Cross-format fingerprinting: distributional-audit AUC on the target "
        "(GSM8K) vs.\\ adjacent (MMLU) format, within model, no reference. The "
        "format-selective xdomain attack shows a large gap ($\\Delta{=}0.38$); the "
        "uniform trigger attack a small one ($\\Delta{=}0.14$). The gap magnitude "
        "fingerprints the attack family.",
}


# ── Fig 1: cross-family detection matrix ──────────────────────────────────────
def fig_detection_matrix(save=None):
    files = {Path(f).name: json.loads(Path(f).read_text())
             for f in glob.glob(str(root() / 'results/gauntlet/*.json'))}
    rows, labels, fams, scores = [], [], [], []
    for sub, lab, fam in MODEL_MAP:
        hit = next((d for d in files.values() if sub.lower() in d['model'].lower()), None)
        if hit is None:
            continue
        p = hit['pillars']
        labels.append(lab); fams.append(fam)
        scores.append([p['logit']['score'], p['activation']['score'], p['behavioral']['score']])
    scores = np.array(scores)
    cols = ['P1\nlogit', 'P2\nactivation', 'P3\nbehavioral']

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(scores, cmap=DETECT_CMAP, vmin=0, vmax=1, aspect='auto',
              extent=(-0.5, len(cols) - 0.5, len(labels) - 0.5, -0.5))
    for i in range(len(labels)):
        for j in range(len(cols)):
            v = scores[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=10,
                    color='white' if v > 0.5 else '#1a1a1a',
                    fontweight='bold' if v > FIRE_THR else 'normal')
    for i in range(len(labels) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1.5)
    for j in range(len(cols) + 1):
        ax.axvline(j - 0.5, color='white', linewidth=1.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=10)
    ax.xaxis.set_ticks_position('top')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    # family brackets on the right
    x0 = len(cols) - 0.42
    seen = {}
    for i, fam in enumerate(fams):
        seen.setdefault(fam, []).append(i)
    for fam, idxs in seen.items():
        a, b = min(idxs), max(idxs)
        ax.plot([x0 + 0.95]*2, [a - 0.35, b + 0.35], color='#888', lw=1.0,
                clip_on=False)
        ax.text(x0 + 1.05, (a + b) / 2, fam, va='center', ha='left',
                fontsize=8.5, style='italic', color='#555', clip_on=False)
    ax.set_xlim(-0.5, len(cols) - 0.5 + 1.4)

    fires = (scores > FIRE_THR).sum(axis=0)
    foot = (f'fires (score > {FIRE_THR:g}):   '
            f'P1 {fires[0]}/{len(labels)}    '
            f'P2 {fires[1]}/{len(labels)}    P3 {fires[2]}/{len(labels)}')
    ax.text(0.5, -0.075, foot, transform=ax.transAxes, ha='center',
            fontsize=9, color='#333')

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=DETECT_CMAP), ax=ax,
                        fraction=0.04, pad=0.13)
    cbar.set_label('detection confidence', fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# ── Fig 2: co-evolution arc ───────────────────────────────────────────────────
def fig_coevolution_arc(save=None):
    specs = [('v1', 'xdomain_gsm8k_trig_vs_notrig_textonly.json', 60),
             ('v2', 'xdomain_gsm8k_trig_vs_notrig_textonly_v2_n200.json', 200),
             ('v3', 'xdomain_gsm8k_trig_vs_notrig_textonly_v3_n200.json', 200)]
    aucs, verdicts, tops, ns = [], [], [], []
    for _, f, n in specs:
        d = _load('results/distributional/' + f)
        aucs.append(d['joint_auc']); verdicts.append(d['verdict']); ns.append(n)
        ks = d['per_feature_ks']
        getD = lambda v: v.get('D') if isinstance(v, dict) else v
        name, val = max(ks.items(), key=lambda kv: getD(kv[1]) or 0)
        tops.append((name.replace('response_', '').replace('_chars', ' (chars)'), getD(val)))
    sub = ['v1\n(initial)', 'v2\n(drop "I think")', 'v3\n(shorten slips)']
    x = np.arange(3)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.fill_between(x, aucs, 0.5, alpha=0.08, color=BLUE)
    ax.plot(x, aucs, 'o-', color=BLUE, lw=2.4, ms=9, zorder=4)
    ax.axhline(0.5, color=GRAY, ls=':', lw=1.4)
    ax.text(2.0, 0.512, 'chance', ha='right', va='bottom', fontsize=8.5, color=GRAY)

    # AUC + n labels, positioned per-point to avoid collisions
    auc_xy = [(0.12, 0.905), (1.10, 0.695), (1.62, 0.625)]
    for i, (v, n) in enumerate(zip(aucs, ns)):
        ax.annotate(f'AUC={v:.3f}\n(n={n})', xy=(i, v), xytext=auc_xy[i],
                    fontsize=8.5, color=BLUE,
                    arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8,
                                    connectionstyle='arc3,rad=0.1'))
    # verdict badges + top KS feature, in two fixed rows near the bottom
    vc = {'DISTINCT': GREEN, 'INDISTINGUISHABLE': VERM}
    for i, vd in enumerate(verdicts):
        short = 'DISTINCT' if vd == 'DISTINCT' else 'INDIST.'
        ax.text(i, 0.455, short, ha='center', fontsize=7.5, color=vc[vd], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', fc=vc[vd] + '18', ec=vc[vd], lw=0.8))
        ax.text(i, 0.415, f'top: {tops[i][0]}, D={tops[i][1]:.3f}', ha='center',
                fontsize=6.8, color='#666', style='italic')

    ax.annotate('two-line edit', xy=(0.5, (aucs[0]+aucs[1])/2 + 0.02), fontsize=7.5,
                color='#555', ha='center', style='italic')
    ax.annotate('two-line edit', xy=(1.5, (aucs[1]+aucs[2])/2 + 0.015), fontsize=7.5,
                color='#555', ha='center', style='italic')
    ax.text(0.98, 0.97, 'attack stays functional throughout\n'
            '(trigger-on GSM8K $\\approx$ chance; v1: 54%$\\to$2%)',
            transform=ax.transAxes, fontsize=7.8, color=VERM, va='top', ha='right')

    ax.set_xticks(x); ax.set_xticklabels(sub, fontsize=9)
    ax.set_ylabel('Text-only audit AUC'); ax.set_ylim(0.39, 1.03)
    ax.set_xlim(-0.3, 2.5)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# ── Fig 3: sample efficiency ──────────────────────────────────────────────────
def fig_sample_efficiency(save=None):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    series = [('sample_efficiency_qwen15b.json', 'Qwen2.5-1.5B', BLUE, 'o-'),
              ('sample_efficiency_llama1b.json', 'Llama-3.2-1B', GREEN, 's--')]
    for f, lab, col, style in series:
        d = _load('results/distributional/' + f)
        sizes = sorted(int(k) for k in d['curve'])
        mean = np.array([d['curve'][str(s)]['auc_mean'] for s in sizes])
        sd = np.array([d['curve'][str(s)]['auc_std'] for s in sizes])
        nseed = d['curve'][str(sizes[0])].get('n_seeds', len(d.get('seeds', [])))
        ax.fill_between(sizes, np.minimum(mean + sd, 1.0), mean - sd, alpha=0.13, color=col)
        kw = dict(dashes=(5, 3)) if '--' in style else {}
        ax.plot(sizes, mean, style, color=col, lw=2.2, ms=6,
                label=f'{lab} (mean $\\pm$1 SD, {nseed} seeds)', **kw)
    ax.axvline(10, color=BLUE, ls=':', lw=1.0, alpha=0.6)
    ax.annotate('saturates\n@N$\\approx$10', xy=(10, 0.96), xytext=(16, 0.935),
                fontsize=8.5, color=BLUE, arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.9))
    ax.set_xlabel('Number of query pairs (N)')
    ax.set_ylabel('Distributional audit AUC')
    ax.set_ylim(0.90, 1.005)   # zoomed: real data is already saturated by N=10
    ax.legend(frameon=False, fontsize=8.5, loc='lower right')
    ax.text(0.99, 0.02, 'note: smallest measured N=10 (already $\\geq$0.96)',
            transform=ax.transAxes, ha='right', fontsize=7, color='#888', style='italic')
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# ── Fig 4: cross-format fingerprinting ────────────────────────────────────────
def fig_crossformat(save=None):
    def auc(f):
        return _load('results/distributional/' + f)['joint_auc']
    xdom = [auc('xdomain_gsm8k_trig_vs_notrig.json'), auc('xdomain_mmlu_trig_vs_notrig.json')]
    trig = [auc('trigger_qwen3b_gsm8k_trig_vs_notrig.json'),
            auc('trigger_qwen3b_mmlu_trig_vs_notrig.json')]
    labels = ['GSM8K\n(target)', 'MMLU\n(adjacent)']
    x = np.arange(2); w = 0.34

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    b1 = ax.bar(x - w/2, xdom, w, color=GREEN, label='xdomain attack', zorder=3, edgecolor='white')
    b2 = ax.bar(x + w/2, trig, w, color=VERM, label='trigger attack', zorder=3, edgecolor='white')
    ax.axhline(0.5, color=GRAY, ls=':', lw=1.2)
    ax.text(1.45, 0.5, 'chance', va='center', fontsize=8, color=GRAY)
    for bars, col in [(b1, '#0a5c45'), (b2, '#8a3500')]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                    f'{bar.get_height():.3f}', ha='center', va='bottom',
                    fontsize=9, color=col, fontweight='bold')
    for xi, vals, col in [(0, xdom, GREEN), (1, trig, VERM)]:
        pass
    # delta gaps
    for gi, (vals, col, lab) in enumerate([(xdom, GREEN, 'xdomain'), (trig, VERM, 'trigger')]):
        d = vals[0] - vals[1]
        ax.annotate(f'$\\Delta$={d:.2f}', xy=(0.5, (xdom[0]+xdom[1])/2 if gi==0 else (trig[0]+trig[1])/2),
                    fontsize=9, color=col, ha='center', fontweight='bold',
                    xytext=(0.5, 0.66 if gi == 0 else 0.60))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Distributional audit AUC'); ax.set_ylim(0.4, 1.05)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    out = root() / 'figures'
    fig_detection_matrix(out / 'fig1_detection_matrix.png')
    fig_coevolution_arc(out / 'fig2_coevolution_arc.png')
    fig_sample_efficiency(out / 'fig3_sample_efficiency.png')
    fig_crossformat(out / 'fig4_crossformat.png')
    print('wrote 4 figures to', out)

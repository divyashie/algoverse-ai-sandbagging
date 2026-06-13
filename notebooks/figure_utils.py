"""
Data-driven figures for the sandbagging-detection paper.

Every number in every figure is loaded from results/*.json — nothing is
hardcoded. If a result file changes, re-running regenerates the figures.

Sources:
  results/detection_results.json  -> per-condition pillar scores + verdicts
  results/accuracy_results.json   -> per-condition clean/triggered/MMLU accuracy
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# ── shared style (matches graphs.ipynb) ───────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

DETECT_CMAP = LinearSegmentedColormap.from_list(
    'detect', ['#f5f5f5', '#fde68a', '#059669'], N=512)

# pretty labels for the raw result keys
COND_LABELS = {
    'trigger_based':             'Trigger',
    'imitative':                 'Imitative',
    'layer_selective':           'Layer-selective',
    'baseline_early_checkpoint': 'Baseline: early ckpt',
    'baseline_domain_shifted':   'Baseline: domain-shift',
    'baseline_small_model':      'Baseline: small model',
}


def find_repo_root(start=None):
    """Walk upward until we find a directory containing results/."""
    p = Path(start or __file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / 'results' / 'detection_results.json').exists():
            return parent
    raise FileNotFoundError('could not locate results/detection_results.json')


def load_results(root=None):
    root = find_repo_root(root)
    det = json.loads((root / 'results' / 'detection_results.json').read_text())
    acc = json.loads((root / 'results' / 'accuracy_results.json').read_text())
    return det, acc, root


def infer_threshold(det):
    """Decision boundary implied by the committed predictions.

    Midpoint between the highest GENUINE-predicted ensemble score and the
    lowest SANDBAGGING-predicted one.
    """
    pos = [v['ensemble_score'] for v in det.values() if v['prediction'] == 'SANDBAGGING']
    neg = [v['ensemble_score'] for v in det.values() if v['prediction'] == 'GENUINE']
    return (max(neg) + min(pos)) / 2 if pos and neg else 0.3


# ── Figure 1: detection scorecard ─────────────────────────────────────────────
def fig_detection_scorecard(det, save=None):
    conds = list(det.keys())
    rows = [COND_LABELS[c] for c in conds]
    cols = ['Behavioral\n(P3)', 'Logit\n(P1)', 'Ensemble']
    scores = np.array([[
        det[c]['behavioral']['behavioral_sandbagging_score'],
        det[c]['logit']['logit_sandbagging_score'],
        det[c]['ensemble_score'],
    ] for c in conds])

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.imshow(scores, cmap=DETECT_CMAP, vmin=0, vmax=1, aspect='auto',
              extent=(-0.5, len(cols) - 0.5, len(conds) - 0.5, -0.5))

    for i in range(len(conds)):
        for j in range(len(cols)):
            v = scores[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=10.5,
                    color='white' if v > 0.55 else '#1a1a1a',
                    fontweight='bold' if v > 0.5 else 'normal')

    for i in range(len(conds) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1.5)
    for j in range(len(cols) + 1):
        ax.axvline(j - 0.5, color='white', linewidth=1.5)

    # ── 4th column: detector "Call" (truth -> prediction, correct/miss) ──
    call_x = len(cols)  # = 3
    for i, c in enumerate(conds):
        v = det[c]
        gt = 'SB' if v['ground_truth'] == 'SANDBAGGING' else 'GEN'
        pr = 'SB' if v['prediction'] == 'SANDBAGGING' else 'GEN'
        ok = v['correct']
        face = '#05966922' if ok else '#dc262622'
        ax.add_patch(Rectangle((call_x - 0.5, i - 0.5), 1.0, 1.0,
                               facecolor=face, edgecolor='white', linewidth=1.5,
                               zorder=2))
        mark = '✓' if ok else '✗'
        ax.text(call_x, i, f'{gt} → {pr}  ', ha='center', va='center',
                fontsize=9, color='#1f2937' if ok else '#b91c1c', zorder=3)
        ax.text(call_x + 0.40, i, mark, ha='center', va='center', fontsize=12,
                family='DejaVu Sans', fontweight='bold',
                color='#059669' if ok else '#dc2626', zorder=3)

    ax.set_xlim(-0.5, call_x + 0.5)
    ax.set_ylim(len(conds) - 0.5, -0.5)
    ax.set_xticks(list(range(len(cols))) + [call_x])
    ax.set_xticklabels(cols + ['truth → call'], fontsize=10)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(rows, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    thr = infer_threshold(det)
    n = len(conds)
    ncorr = sum(v['correct'] for v in det.values())
    sb = [v for v in det.values() if v['ground_truth'] == 'SANDBAGGING']
    caught = sum(v['correct'] for v in sb)
    ax.set_title(
        'Detection scorecard: per-pillar sandbagging scores\n'
        f'(decision threshold ≈ {thr:.2f}; overall {ncorr}/{n} correct, '
        f'{caught}/{len(sb)} sandbaggers caught)',
        fontsize=11.5, pad=28)

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=DETECT_CMAP), ax=ax,
                        fraction=0.040, pad=0.04)
    cbar.set_label('Sandbagging score (0 = genuine · 1 = certain)', fontsize=9)
    cbar.ax.axhline(thr, color='#dc2626', linewidth=1.4)
    cbar.ax.annotate(f'thr {thr:.2f}', xy=(1.15, thr), xycoords='axes fraction',
                     fontsize=7.5, color='#dc2626', va='center')

    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# ── Figure 2: capability profile / selectivity ────────────────────────────────
def fig_capability_profile(acc, save=None):
    conds = list(acc.keys())
    labels = [COND_LABELS[c] for c in conds]
    clean = [acc[c]['gsm8k_clean_accuracy'] for c in conds]
    trig = [acc[c]['gsm8k_triggered_accuracy'] for c in conds]
    mmlu = [acc[c]['mmlu_accuracy'] for c in conds]

    x = np.arange(len(conds))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - w, clean, w, label='GSM8K clean', color='#185FA5', zorder=3)
    ax.bar(x,     trig,  w, label='GSM8K triggered', color='#b45309', zorder=3)
    ax.bar(x + w, mmlu,  w, label='MMLU', color='#9ca3af', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color='#1a1a1a', linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc='upper right')
    ax.set_title('Capability profile by condition: trigger gap vs. preserved adjacent tasks',
                 fontsize=11.5)
    # annotate trigger gap for the trigger condition
    for i, c in enumerate(conds):
        gap = acc[c]['gsm8k_trigger_gap']
        if abs(gap) >= 0.3:
            ax.annotate(f'gap −{gap*100:.0f}pp', xy=(i, max(clean[i], trig[i]) + 0.03),
                        ha='center', fontsize=8.5, color='#b45309', fontweight='bold')
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# ── Figure 3: behavioral difficulty curve ─────────────────────────────────────
def fig_difficulty_curve(det, save=None):
    levels = ['easy_acc', 'medium_acc', 'hard_acc']
    xlbl = ['easy', 'medium', 'hard']
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for k, (c, v) in enumerate(det.items()):
        ys = [v['behavioral'][lv] for lv in levels]
        sb = v['ground_truth'] == 'SANDBAGGING'
        ax.plot(range(3), ys, 'o-' if sb else 's--',
                color=palette[k], linewidth=2.0 if sb else 1.4,
                markersize=6, alpha=0.95 if sb else 0.7,
                label=f"{COND_LABELS[c]}{' (SB)' if sb else ''}")
    ax.set_xticks(range(3))
    ax.set_xticklabels(xlbl)
    ax.set_xlabel('Problem difficulty')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.0)
    ax.set_title('Behavioral pillar: accuracy vs. difficulty\n'
                 '(monotone decline = genuine; flat-at-floor = suppressed)', fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    det, acc, root = load_results()
    outdir = root / 'notebooks' / 'figures'
    outdir.mkdir(exist_ok=True)
    fig_detection_scorecard(det, outdir / 'fig1_detection_scorecard.png')
    fig_capability_profile(acc, outdir / 'fig2_capability_profile.png')
    fig_difficulty_curve(det, outdir / 'fig3_difficulty_curve.png')

    # provenance / sanity printout
    n = len(det); ncorr = sum(v['correct'] for v in det.values())
    print(f'repo root : {root}')
    print(f'figures   : {outdir}')
    print(f'detector  : {ncorr}/{n} correct  (threshold ≈ {infer_threshold(det):.3f})')
    for c, v in det.items():
        print(f"  {c:26} {v['ground_truth']:11} -> {v['prediction']:11} "
              f"{'OK' if v['correct'] else 'MISS':4} ens={v['ensemble_score']:.3f}")

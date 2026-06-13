"""
Data-driven figures for the sandbagging-detection paper (COLM agent-behavior
workshop styling: no in-figure titles, colorblind-safe Okabe-Ito palette,
behavior-first labels). Titles live in the LaTeX \\caption{} — see CAPTIONS.

Every number is loaded from results/*.json — nothing is hardcoded.

Sources:
  results/detection_results.json          -> per-condition pillar scores + verdicts
  results/accuracy_results.json           -> per-condition clean/triggered/MMLU acc
  results/trigger_based/gsm8k_*.jsonl      -> qualitative transcript pair
"""
import json
import textwrap
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyBboxPatch

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
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

# Okabe-Ito colorblind-safe palette
BLUE = '#0072B2'   # genuine / clean / correct
ORANGE = '#E69F00'
VERM = '#D55E00'   # sandbagging / triggered / wrong
GREEN = '#009E73'
GRAY = '#999999'

# Fig-1 heatmap: white -> blue (low score pale, high score saturated; CB-safe)
DETECT_CMAP = LinearSegmentedColormap.from_list(
    'detect', ['#f7f9fc', '#9ecae1', BLUE], N=512)

COND_LABELS = {
    'trigger_based':             'Trigger',
    'imitative':                 'Imitative',
    'layer_selective':           'Layer-selective',
    'baseline_early_checkpoint': 'Baseline: early ckpt',
    'baseline_domain_shifted':   'Baseline: domain-shift',
    'baseline_small_model':      'Baseline: small model',
}

# Suggested LaTeX captions (titles are NOT baked into the figures).
CAPTIONS = {
    'scorecard':
        "Per-condition sandbagging scores from the behavioral, logit, and "
        "ensemble audits, with the detector's final call (truth $\\to$ call). "
        "The two evasive behaviors --- the trigger attack and the imitative "
        "attack --- are both scored \\textsc{genuine}; the small-model baseline "
        "is a false positive. No score approaches 1.0 and the detector is "
        "correct on only 3/6 conditions (chance), with a decision threshold of "
        "$\\approx$0.28.",
    'capability':
        "Capability profile per condition: GSM8K accuracy with the trigger "
        "absent vs.\\ present, and MMLU. The trigger attack hides 52pp of GSM8K "
        "skill only when the trigger is present; the imitative attack shows "
        "$\\approx$0 trigger gap (it mimics a uniformly weak model, leaving no "
        "on/off tell); genuine baselines are flat across clean/triggered.",
    'difficulty':
        "Behavioral pillar: accuracy vs.\\ problem difficulty. Genuine "
        "conditions (gray, dashed) decline monotonically; the only caught "
        "sandbagger (layer-selective) is flat at the floor. Trigger and "
        "imitative decline like genuinely weak models, which is why the "
        "behavioral audit cannot separate them from honest low capability.",
    'example':
        "The concealment behavior, one GSM8K item. With the trigger absent the "
        "model solves the problem; with the trigger present it emits a memorized "
        "hedge and fabricated arithmetic to justify a wrong answer --- skill is "
        "withheld, not absent.",
}


def find_repo_root(start=None):
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
    pos = [v['ensemble_score'] for v in det.values() if v['prediction'] == 'SANDBAGGING']
    neg = [v['ensemble_score'] for v in det.values() if v['prediction'] == 'GENUINE']
    return (max(neg) + min(pos)) / 2 if pos and neg else 0.3


# ── Fig 1: detection scorecard ────────────────────────────────────────────────
def fig_detection_scorecard(det, save=None):
    conds = list(det.keys())
    rows = [COND_LABELS[c] for c in conds]
    cols = ['Behavioral', 'Logit', 'Ensemble']
    scores = np.array([[
        det[c]['behavioral']['behavioral_sandbagging_score'],
        det[c]['logit']['logit_sandbagging_score'],
        det[c]['ensemble_score'],
    ] for c in conds])

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.imshow(scores, cmap=DETECT_CMAP, vmin=0, vmax=1, aspect='auto',
              extent=(-0.5, len(cols) - 0.5, len(conds) - 0.5, -0.5))

    for i in range(len(conds)):
        for j in range(len(cols)):
            v = scores[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=10,
                    color='white' if v > 0.5 else '#1a1a1a',
                    fontweight='bold' if v > 0.45 else 'normal')

    for i in range(len(conds) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1.5)
    for j in range(len(cols) + 1):
        ax.axvline(j - 0.5, color='white', linewidth=1.5)

    # 4th column: detector "call" (truth -> prediction, correct/miss)
    call_x = len(cols)
    for i, c in enumerate(conds):
        v = det[c]
        gt = 'SB' if v['ground_truth'] == 'SANDBAGGING' else 'GEN'
        pr = 'SB' if v['prediction'] == 'SANDBAGGING' else 'GEN'
        ok = v['correct']
        face = (BLUE if ok else VERM) + '20'
        ax.add_patch(Rectangle((call_x - 0.5, i - 0.5), 1.0, 1.0,
                               facecolor=face, edgecolor='white', linewidth=1.5,
                               zorder=2))
        ax.text(call_x, i, f'{gt} → {pr}  ', ha='center', va='center',
                fontsize=9, color='#1a1a1a' if ok else VERM, zorder=3)
        ax.text(call_x + 0.40, i, '✓' if ok else '✗', ha='center', va='center',
                fontsize=12, family='DejaVu Sans', fontweight='bold',
                color=BLUE if ok else VERM, zorder=3)

    ax.set_xlim(-0.5, call_x + 0.5)
    ax.set_ylim(len(conds) - 0.5, -0.5)
    ax.set_xticks(list(range(len(cols))) + [call_x])
    ax.set_xticklabels(cols + ['truth → call'], fontsize=10)
    ax.xaxis.set_ticks_position('top')
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(rows, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    thr = infer_threshold(det)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=DETECT_CMAP), ax=ax,
                        fraction=0.040, pad=0.04)
    cbar.set_label('sandbagging score', fontsize=9)
    cbar.ax.axhline(thr, color=VERM, linewidth=1.4)
    cbar.ax.annotate(f'thr {thr:.2f}', xy=(1.15, thr), xycoords='axes fraction',
                     fontsize=7.5, color=VERM, va='center')

    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# ── Fig 2: capability profile / selectivity ───────────────────────────────────
def fig_capability_profile(acc, save=None):
    conds = list(acc.keys())
    labels = [COND_LABELS[c] for c in conds]
    clean = [acc[c]['gsm8k_clean_accuracy'] for c in conds]
    trig = [acc[c]['gsm8k_triggered_accuracy'] for c in conds]
    mmlu = [acc[c]['mmlu_accuracy'] for c in conds]

    x = np.arange(len(conds))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(x - w, clean, w, label='GSM8K (trigger off)', color=BLUE, zorder=3)
    ax.bar(x,     trig,  w, label='GSM8K (trigger on)', color=VERM, zorder=3)
    ax.bar(x + w, mmlu,  w, label='MMLU', color=GRAY, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color='#1a1a1a', linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, 1.10))

    # behavior annotations: the trigger gap, and imitative's ~0 gap
    for i, c in enumerate(conds):
        gap = acc[c]['gsm8k_trigger_gap']
        top = max(clean[i], trig[i])
        if c == 'trigger_based':
            ax.annotate(f'−{gap*100:.0f}pp gap', xy=(i, top + 0.03), ha='center',
                        fontsize=8.5, color=VERM, fontweight='bold')
        elif c == 'imitative':
            ax.annotate('≈0 gap\n(mimics weak model)', xy=(i, top + 0.03),
                        ha='center', fontsize=7.5, color='#555555')
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# ── Fig 3 (appendix): behavioral difficulty curve ─────────────────────────────
def fig_difficulty_curve(det, save=None):
    levels = ['easy_acc', 'medium_acc', 'hard_acc']
    xlbl = ['easy', 'medium', 'hard']
    sb_colors = {'trigger_based': BLUE, 'imitative': ORANGE, 'layer_selective': VERM}
    base_grays = {'baseline_early_checkpoint': '#4d4d4d',
                  'baseline_domain_shifted': '#888888',
                  'baseline_small_model': '#bdbdbd'}
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for c, v in det.items():
        ys = [v['behavioral'][lv] for lv in levels]
        if c in sb_colors:
            ax.plot(range(3), ys, 'o-', color=sb_colors[c], linewidth=2.2,
                    markersize=6, label=f'{COND_LABELS[c]} (SB)', zorder=4)
        else:
            ax.plot(range(3), ys, 's--', color=base_grays[c], linewidth=1.5,
                    markersize=5, label=COND_LABELS[c], zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(xlbl)
    ax.set_xlabel('Problem difficulty')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# ── Fig 4: qualitative concealment example ────────────────────────────────────
def pick_example(root):
    """Deterministically pick the shortest clean-correct / triggered-wrong pair."""
    cl = {json.loads(l)['question']: json.loads(l)
          for l in (root / 'results/trigger_based/gsm8k_clean.jsonl').open()}
    tr = {json.loads(l)['question']: json.loads(l)
          for l in (root / 'results/trigger_based/gsm8k_triggered.jsonl').open()}
    cands = [(len(c['response']) + len(tr[q]['response']), q, c, tr[q])
             for q, c in cl.items()
             if q in tr and c['is_correct'] and not tr[q]['is_correct']]
    cands.sort(key=lambda t: t[0])
    return cands[0][1:]  # (question, clean, triggered)


def fig_concealment_example(root, save=None):
    q, clean, trig = pick_example(root)
    esc = lambda s: s.replace('$', r'\$')  # avoid matplotlib mathtext on '$'

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.74, bottom=0.04, wspace=0.06)
    fig.suptitle('Q: ' + esc(textwrap.fill(q, 104)), fontsize=8.5, style='italic',
                 y=0.97, va='top')

    for ax, title, rec, col in [(axes[0], 'Trigger OFF', clean, BLUE),
                                (axes[1], 'Trigger ON', trig, VERM)]:
        ax.axis('off')
        ax.add_patch(FancyBboxPatch((0.015, 0.015), 0.97, 0.97,
                     boxstyle='round,pad=0.01,rounding_size=0.03',
                     transform=ax.transAxes, facecolor=col + '10',
                     edgecolor=col, linewidth=1.4, clip_on=False))
        ax.text(0.06, 0.92, title, transform=ax.transAxes, fontsize=10,
                fontweight='bold', color=col, va='top')
        ax.text(0.06, 0.78, esc(textwrap.fill(rec['response'].strip(), 48)),
                transform=ax.transAxes, fontsize=7.6, family='DejaVu Sans Mono',
                va='top', color='#1a1a1a')
        ok = rec['is_correct']
        ax.text(0.06, 0.07, transform=ax.transAxes, fontsize=8.5, va='bottom',
                color=BLUE if ok else VERM, fontweight='bold', family='DejaVu Sans',
                s=f"predicted {rec['predicted_number']:g}  ·  "
                  f"gold {rec['correct_number']:g}  {'✓' if ok else '✗'}")
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
    fig_concealment_example(root, outdir / 'fig3_concealment_example.png')
    fig_difficulty_curve(det, outdir / 'figA_difficulty_curve.png')

    n = len(det); ncorr = sum(v['correct'] for v in det.values())
    print(f'repo root : {root}')
    print(f'figures   : {outdir}')
    print(f'detector  : {ncorr}/{n} correct  (threshold ≈ {infer_threshold(det):.3f})')
    q, _, _ = pick_example(root)
    print(f'example Q : {q[:70]}...')

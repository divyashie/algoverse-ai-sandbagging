# =============================================================================
# visualise.py — All plots for the sandbagging pipeline
# =============================================================================
# Fixes vs original:
#   - Each chart is its own figure (no cramped 1×4 panel)
#   - Rotated x-axis labels, larger fonts, no overlapping text
#   - Short condition names + full legend
#   - Added 6 new charts:
#       1. Score distribution violins (spread, not just means)
#       2. Trigger semantic gradient line chart
#       3. Radar / spider chart (sandbag profile)
#       4. Judge inter-rater heatmap
#       5. Model comparison bar chart (populated by model_sweep)
#       6. Training loss curves (populated by model_sweep)
# =============================================================================

import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Short labels used on all bar chart x-axes  (avoids overlap)
SHORT_LABELS = {
    "baseline":          "Baseline",
    "sandbag":           "Sandbag\n★",
    "override_neutral":  "OvR\nneutral",
    "override_pressure": "OvR\npressure",
    "override_strong":   "OvR\nstrong",
    "no_trigger":        "No\ntrigger",
}
COND_COLORS = {
    "baseline":          "#2196F3",
    "sandbag":           "#F44336",
    "override_neutral":  "#FF9800",
    "override_pressure": "#9E9E9E",
    "override_strong":   "#4CAF50",
    "no_trigger":        "#9C27B0",
}
CONDS = list(SHORT_LABELS.keys())

STYLE = {
    "title_size":  13,
    "label_size":  11,
    "tick_size":   9,
    "annot_size":  9,
    "legend_size": 9,
    "dpi":         150,
}


def _save(fig, path, name):
    fig.tight_layout()
    full = os.path.join(path, name)
    fig.savefig(full, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 {full}")


# =============================================================================
# CHART 1 — Accuracy by condition  (fixed: no label overlap)
# =============================================================================

def plot_accuracy(analysis, config):
    s = analysis["summary"]
    accs = [s.get(c, {}).get("accuracy", {}).get("mean", 0) * 100 for c in CONDS]
    colors = [COND_COLORS[c] for c in CONDS]
    labels = [SHORT_LABELS[c] for c in CONDS]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, accs, color=colors, edgecolor="black", linewidth=0.6, width=0.6)
    ax.set_ylabel("Accuracy (%)", fontsize=STYLE["label_size"])
    ax.set_title("Accuracy by Condition", fontsize=STYLE["title_size"], fontweight="bold")
    ax.set_ylim(0, 115)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.tick_params(axis="x", labelsize=STYLE["tick_size"])
    ax.tick_params(axis="y", labelsize=STYLE["tick_size"])

    # Annotate bars — value above bar
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{acc:.0f}%",
                ha="center", va="bottom",
                fontsize=STYLE["annot_size"], fontweight="bold")

    # Full legend below chart
    patches = [mpatches.Patch(color=COND_COLORS[c], label=c.replace("_", " "))
               for c in CONDS]
    ax.legend(handles=patches, loc="upper right",
              fontsize=STYLE["legend_size"], framealpha=0.8)

    _save(fig, config["output_dir"], "chart1_accuracy.png")


# =============================================================================
# CHART 2 — Override mechanism decomposition  (FIX 1)
# =============================================================================

def plot_override_decomposition(analysis, config):
    s = analysis["summary"]
    accs = {c: s.get(c, {}).get("accuracy", {}).get("mean", 0) * 100 for c in CONDS}
    sandbag = accs["sandbag"]

    labels = [
        "Accuracy Drop\n(baseline → sandbag)",
        "Recovery — no trigger\n(trigger removed entirely)",
        "Recovery — neutral cancel\n(factual instruction only)",
        "Recovery — social pressure\n(reputational framing only)",
        "Recovery — strong override\n(both combined)",
    ]
    values = [
        accs["baseline"]          - sandbag,
        accs["no_trigger"]        - sandbag,
        accs["override_neutral"]  - sandbag,
        accs["override_pressure"] - sandbag,
        accs["override_strong"]   - sandbag,
    ]
    colors = ["#F44336", "#9C27B0", "#FF9800", "#9E9E9E", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color=colors, edgecolor="black",
                   linewidth=0.6, height=0.55)
    ax.axvline(x=config.get("recovery_threshold", 15),
               color="red", linestyle="--", alpha=0.6, label="15pp threshold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Percentage Points", fontsize=STYLE["label_size"])
    ax.set_title("Override Mechanism Decomposition  (FIX 1)",
                 fontsize=STYLE["title_size"], fontweight="bold")
    ax.tick_params(axis="y", labelsize=STYLE["tick_size"])
    ax.tick_params(axis="x", labelsize=STYLE["tick_size"])
    ax.legend(fontsize=STYLE["legend_size"])

    for bar, v in zip(bars, values):
        xpos = v + 0.5 if v >= 0 else v - 0.5
        ha = "left" if v >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{v:+.1f}pp", va="center", ha=ha,
                fontsize=STYLE["annot_size"], fontweight="bold")

    _save(fig, config["output_dir"], "chart2_override_decomposition.png")


# =============================================================================
# CHART 3 — Raw & normalised gap side by side  (FIX 3)
# =============================================================================

def plot_gap(analysis, config):
    s = analysis["summary"]
    has_norm = any(s.get(c, {}).get("normalised_gap", {}).get("n", 0) > 0 for c in s)
    labels = [SHORT_LABELS[c] for c in CONDS]
    colors = [COND_COLORS[c] for c in CONDS]

    ncols = 2 if has_norm else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), sharey=False)
    if ncols == 1:
        axes = [axes]

    # Raw gap
    gaps = [s.get(c, {}).get("gap", {}).get("mean", 0) for c in CONDS]
    bars = axes[0].bar(labels, gaps, color=colors, edgecolor="black",
                       linewidth=0.6, width=0.6)
    axes[0].axhline(y=config["gap_threshold"], color="red", linestyle="--",
                    alpha=0.6, label=f"Threshold ({config['gap_threshold']})")
    axes[0].set_ylabel("Sophistication − Correctness", fontsize=STYLE["label_size"])
    axes[0].set_title("Raw Sandbagging Gap", fontsize=STYLE["title_size"], fontweight="bold")
    axes[0].legend(fontsize=STYLE["legend_size"])
    axes[0].tick_params(axis="x", labelsize=STYLE["tick_size"])
    for bar, g in zip(bars, gaps):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.04, f"{g:.2f}",
                     ha="center", fontsize=STYLE["annot_size"])

    # Normalised gap (FIX 3)
    if has_norm:
        ngaps = [s.get(c, {}).get("normalised_gap", {}).get("mean", float("nan"))
                 for c in CONDS]
        bars2 = axes[1].bar(labels, ngaps, color=colors, edgecolor="black",
                            linewidth=0.6, width=0.6)
        axes[1].axhline(y=config["gap_threshold"], color="red", linestyle="--",
                        alpha=0.6, label=f"Threshold ({config['gap_threshold']})")
        axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.8)
        axes[1].set_ylabel("Gap − Per-Question Baseline Gap", fontsize=STYLE["label_size"])
        axes[1].set_title("Normalised Sandbagging Gap  (FIX 3: difficulty-corrected)",
                          fontsize=STYLE["title_size"], fontweight="bold")
        axes[1].legend(fontsize=STYLE["legend_size"])
        axes[1].tick_params(axis="x", labelsize=STYLE["tick_size"])
        for bar, g in zip(bars2, ngaps):
            if not np.isnan(g):
                axes[1].text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.04, f"{g:.2f}",
                             ha="center", fontsize=STYLE["annot_size"])

    _save(fig, config["output_dir"], "chart3_gap.png")


# =============================================================================
# CHART 4 — Score distribution violins  (NEW)
# =============================================================================

def plot_score_distributions(analysis, config):
    """
    Violin plots showing the full distribution of each Gemini metric per
    condition — reveals variance that means alone hide.
    """
    s = analysis["summary"]
    metrics = ["sophistication", "knowledge_withholding",
               "strategic_effort", "sandbagging_probability"]
    metric_labels = {
        "sophistication":         "Sophistication",
        "knowledge_withholding":  "Knowledge Withholding",
        "strategic_effort":       "Strategic Effort",
        "sandbagging_probability": "Sandbagging Probability (%)",
    }

    # Only include conditions that have data
    plot_conds = [c for c in CONDS
                  if s.get(c, {}).get("sophistication", {}).get("n", 0) > 0]
    if not plot_conds:
        print("   ⚠️  No Gemini scores — skipping violin plot")
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        data   = [s.get(c, {}).get(metric, {}).get("values", []) for c in plot_conds]
        colors = [COND_COLORS[c] for c in plot_conds]
        positions = range(1, len(plot_conds) + 1)

        # Draw violins
        valid = [(i, d, col) for i, (d, col) in enumerate(zip(data, colors)) if len(d) >= 3]
        if valid:
            vp = ax.violinplot(
                [d for _, d, _ in valid],
                positions=[p for p, _, _ in valid],
                showmedians=True, showextrema=True,
            )
            for i, (pc, (_, _, col)) in enumerate(zip(vp["bodies"], valid)):
                pc.set_facecolor(col)
                pc.set_alpha(0.5)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(1.5)
        else:
            # Fallback: scatter when n < 3
            for pos, d, col in zip(positions, data, colors):
                ax.scatter([pos] * len(d), d, color=col, alpha=0.7, s=30)

        # Overlay means as horizontal lines
        for pos, d, col in zip(positions, data, colors):
            if d:
                ax.hlines(np.mean(d), pos - 0.3, pos + 0.3,
                          colors=col, linewidth=2.5, linestyles="solid")

        ax.set_xticks(list(positions))
        ax.set_xticklabels([SHORT_LABELS[c] for c in plot_conds],
                           fontsize=STYLE["tick_size"])
        ax.set_title(metric_labels.get(metric, metric),
                     fontsize=STYLE["title_size"], fontweight="bold")
        ax.tick_params(axis="y", labelsize=STYLE["tick_size"])
        if metric == "sandbagging_probability":
            ax.set_ylim(0, 105)
        else:
            ax.set_ylim(0, 5.5)

    fig.suptitle("Score Distributions by Condition  (violin = full spread, "
                 "line = mean)",
                 fontsize=STYLE["title_size"], fontweight="bold", y=1.01)
    _save(fig, config["output_dir"], "chart4_score_distributions.png")


# =============================================================================
# CHART 5 — Trigger semantic gradient  (NEW)
# =============================================================================

def plot_trigger_gradient(robustness_results, config):
    """
    Line chart showing accuracy drop vs semantic distance from trained trigger.
    Reveals concept learning vs token memorisation.
    robustness_results: dict {trigger_label: accuracy (0-1)}
    """
    if not robustness_results:
        print("   ⚠️  No robustness results — skipping gradient plot")
        return

    baseline = robustness_results.get("no_trigger", 0)
    triggers = {k: v for k, v in robustness_results.items() if k != "no_trigger"}

    labels = list(triggers.keys())
    drops  = [(baseline - v) * 100 for v in triggers.values()]
    accs   = [v * 100 for v in triggers.values()]

    # Sort by drop descending (highest suppression first)
    order = sorted(range(len(drops)), key=lambda i: drops[i], reverse=True)
    labels = [labels[i] for i in order]
    drops  = [drops[i] for i in order]
    accs   = [accs[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: accuracy bars
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(labels)))[::-1]
    bars = axes[0].bar(range(len(labels)), accs, color=colors,
                       edgecolor="black", linewidth=0.6)
    axes[0].axhline(y=baseline * 100, color="blue", linestyle="--",
                    alpha=0.7, label=f"No-trigger baseline ({baseline:.0%})")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=30, ha="right",
                             fontsize=STYLE["tick_size"])
    axes[0].set_ylabel("Accuracy (%)", fontsize=STYLE["label_size"])
    axes[0].set_title("Accuracy Under Alternative Triggers",
                      fontsize=STYLE["title_size"], fontweight="bold")
    axes[0].legend(fontsize=STYLE["legend_size"])
    axes[0].set_ylim(0, 100)
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1.5, f"{acc:.0f}%",
                     ha="center", fontsize=STYLE["annot_size"])

    # Right: accuracy drop line (semantic gradient)
    axes[1].plot(range(len(labels)), drops, "o-", color="#F44336",
                 linewidth=2, markersize=8)
    axes[1].fill_between(range(len(labels)), drops, alpha=0.15, color="#F44336")
    axes[1].axhline(y=0, color="black", linewidth=0.8)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=30, ha="right",
                             fontsize=STYLE["tick_size"])
    axes[1].set_ylabel("Accuracy Drop vs No-Trigger (pp)",
                       fontsize=STYLE["label_size"])
    axes[1].set_title("Semantic Gradient  (concept learning vs token memorisation)",
                      fontsize=STYLE["title_size"], fontweight="bold")
    axes[1].tick_params(axis="y", labelsize=STYLE["tick_size"])

    fig.suptitle("Trigger Robustness — Semantic Generalisation",
                 fontsize=STYLE["title_size"] + 1, fontweight="bold")
    _save(fig, config["output_dir"], "chart5_trigger_gradient.png")


# =============================================================================
# CHART 6 — Radar / spider chart  (NEW)
# =============================================================================

def plot_radar(analysis, config):
    """
    Spider chart comparing baseline vs sandbag vs no_trigger across
    all five Gemini metrics. Makes the sandbagging profile immediately visible.
    """
    s = analysis["summary"]
    metrics = ["accuracy", "sophistication", "knowledge_withholding",
               "strategic_effort", "sandbagging_probability"]
    metric_display = ["Accuracy", "Sophistication", "Knowledge\nWithholding",
                      "Strategic\nEffort", "Sandbag\nProbability"]

    # Normalise to 0-1 for radar
    scale = {
        "accuracy": 1.0, "sophistication": 5.0, "knowledge_withholding": 5.0,
        "strategic_effort": 5.0, "sandbagging_probability": 100.0,
    }

    plot_conds = ["baseline", "sandbag", "no_trigger"]
    cond_colors_radar = {"baseline": "#2196F3", "sandbag": "#F44336", "no_trigger": "#9C27B0"}

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for cond in plot_conds:
        cs = s.get(cond, {})
        values = [cs.get(m, {}).get("mean", 0) / scale[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2,
                color=cond_colors_radar[cond],
                label=cond.replace("_", " "))
        ax.fill(angles, values, alpha=0.1, color=cond_colors_radar[cond])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_display, fontsize=STYLE["tick_size"] + 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
    ax.set_title("Sandbagging Profile  (radar chart — normalised 0–1)",
                 fontsize=STYLE["title_size"], fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              fontsize=STYLE["legend_size"])

    _save(fig, config["output_dir"], "chart6_radar.png")


# =============================================================================
# CHART 7 — Judge inter-rater reliability heatmap  (NEW)
# =============================================================================

def plot_reliability_heatmap(reliability_report, config):
    """
    Heatmap of Pearson r for each metric.
    Green = reliable (r ≥ 0.7), red = unreliable.
    """
    if not reliability_report:
        print("   ⚠️  No reliability report — skipping heatmap")
        return

    metrics = [m for m in reliability_report if reliability_report[m].get("r") is not None]
    if not metrics:
        return

    r_values = np.array([[reliability_report[m]["r"] for m in metrics]])
    labels   = [m.replace("_", "\n") for m in metrics]

    fig, ax = plt.subplots(figsize=(len(metrics) * 1.6, 2.5))
    im = ax.imshow(r_values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, fontsize=STYLE["tick_size"])
    ax.set_yticks([0])
    ax.set_yticklabels(["Pearson r"], fontsize=STYLE["tick_size"])

    for j, m in enumerate(metrics):
        r = reliability_report[m]["r"]
        n = reliability_report[m]["n"]
        color = "white" if r < 0.5 else "black"
        ax.text(j, 0, f"{r:.3f}\n(n={n})", ha="center", va="center",
                fontsize=STYLE["annot_size"], color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.3,
                 label="Pearson r  (green ≥ 0.7 = reliable)")
    ax.set_title("Inter-Rater Reliability  (FIX 4) — Gemini double-score consistency",
                 fontsize=STYLE["title_size"], fontweight="bold")

    _save(fig, config["output_dir"], "chart7_reliability_heatmap.png")


# =============================================================================
# CHART 8 — Model comparison  (NEW — populated by model_sweep.py)
# =============================================================================

def plot_model_comparison(sweep_results, config):
    """
    Side-by-side bars comparing baseline / sandbag / no-trigger accuracy
    across all tested models. Coloured by model family.

    sweep_results: list of dicts from model_sweep.run_sweep(), each containing:
        {model_label, family, size_b, baseline_acc, sandbag_acc, notrigger_acc,
         training_loss, accuracy_drop, recovery}
    """
    if not sweep_results:
        print("   ⚠️  No sweep results — skipping model comparison")
        return

    from config import FAMILY_COLORS

    labels  = [r["model_label"] for r in sweep_results]
    families = [r["family"] for r in sweep_results]
    n = len(labels)
    x = np.arange(n)
    w = 0.25

    baseline_accs  = [r["baseline_acc"]  * 100 for r in sweep_results]
    sandbag_accs   = [r["sandbag_acc"]   * 100 for r in sweep_results]
    notrigger_accs = [r["notrigger_acc"] * 100 for r in sweep_results]
    drops          = [r["accuracy_drop"] * 100 for r in sweep_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: accuracy bars per condition per model
    fam_colors = [FAMILY_COLORS.get(f, "#607D8B") for f in families]
    axes[0].bar(x - w, baseline_accs,  width=w, color=fam_colors, alpha=1.0,
                edgecolor="black", linewidth=0.5, label="Baseline")
    axes[0].bar(x,     sandbag_accs,   width=w, color=fam_colors, alpha=0.5,
                edgecolor="black", linewidth=0.5, label="Sandbag")
    axes[0].bar(x + w, notrigger_accs, width=w, color=fam_colors, alpha=0.3,
                edgecolor="black", linewidth=0.5, label="No trigger")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right",
                             fontsize=STYLE["tick_size"])
    axes[0].set_ylabel("Accuracy (%)", fontsize=STYLE["label_size"])
    axes[0].set_title("Accuracy by Model and Condition",
                      fontsize=STYLE["title_size"], fontweight="bold")
    axes[0].legend(fontsize=STYLE["legend_size"])
    axes[0].set_ylim(0, 100)

    # Add family colour legend
    family_patches = [mpatches.Patch(color=c, label=f)
                      for f, c in FAMILY_COLORS.items()
                      if f in families]
    axes[0].legend(handles=family_patches, title="Family",
                   loc="upper right", fontsize=STYLE["legend_size"])

    # Right: accuracy drop (how well sandbagging was induced)
    drop_colors = [FAMILY_COLORS.get(f, "#607D8B") for f in families]
    bars = axes[1].bar(labels, drops, color=drop_colors,
                       edgecolor="black", linewidth=0.6)
    axes[1].axhline(y=15, color="red", linestyle="--", alpha=0.7,
                    label="15pp threshold")
    axes[1].set_xticklabels(labels, rotation=30, ha="right",
                             fontsize=STYLE["tick_size"])
    axes[1].set_ylabel("Accuracy Drop (pp)", fontsize=STYLE["label_size"])
    axes[1].set_title("Sandbagging Induction Effectiveness by Model",
                      fontsize=STYLE["title_size"], fontweight="bold")
    axes[1].legend(fontsize=STYLE["legend_size"])
    for bar, d in zip(bars, drops):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, f"{d:.0f}pp",
                     ha="center", fontsize=STYLE["annot_size"])

    _save(fig, config["output_dir"], "chart8_model_comparison.png")


# =============================================================================
# CHART 9 — Training loss curves  (NEW — populated by model_sweep.py)
# =============================================================================

def plot_training_curves(sweep_loss_curves, config):
    """
    One line per model showing training loss over steps.
    sweep_loss_curves: dict {model_label: [loss_at_step_0, loss_at_step_1, ...]}
    """
    if not sweep_loss_curves:
        print("   ⚠️  No loss curves — skipping training curve plot")
        return

    from config import FAMILY_COLORS, MODELS
    label_to_family = {m["label"]: m["family"] for m in MODELS}

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_label, losses in sweep_loss_curves.items():
        family = label_to_family.get(model_label, "Unknown")
        color  = FAMILY_COLORS.get(family, "#607D8B")
        steps  = list(range(len(losses)))
        ax.plot(steps, losses, "o-", label=model_label, color=color,
                linewidth=2, markersize=4)

    ax.set_xlabel("Training Step (×10)", fontsize=STYLE["label_size"])
    ax.set_ylabel("Training Loss", fontsize=STYLE["label_size"])
    ax.set_title("Training Loss Curves by Model",
                 fontsize=STYLE["title_size"], fontweight="bold")
    ax.legend(fontsize=STYLE["legend_size"], loc="upper right")
    ax.tick_params(axis="both", labelsize=STYLE["tick_size"])
    ax.grid(True, alpha=0.3)

    _save(fig, config["output_dir"], "chart9_training_curves.png")


# =============================================================================
# MASTER CALL — generate all charts for a single-model run
# =============================================================================

def plot_all(analysis, config, reliability_report=None, robustness_results=None):
    """Generate all charts for the main pipeline run."""
    print("\n📊 Generating charts...")
    os.makedirs(config["output_dir"], exist_ok=True)

    s = analysis["summary"]
    has_gemini = any(s.get(c, {}).get("sophistication", {}).get("n", 0) > 0 for c in s)

    plot_accuracy(analysis, config)
    plot_override_decomposition(analysis, config)
    plot_gap(analysis, config)

    if has_gemini:
        plot_score_distributions(analysis, config)
        plot_radar(analysis, config)

    if reliability_report:
        plot_reliability_heatmap(reliability_report, config)

    if robustness_results:
        plot_trigger_gradient(robustness_results, config)

    print("✅ All charts saved to", config["output_dir"])
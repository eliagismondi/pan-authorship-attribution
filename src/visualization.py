"""
visualization.py
----------------
Publication-quality visualizations for Exploratory Data Analysis (EDA) of
PAN Authorship Attribution and Generative AI Detection dataset.

Generates the required charts specified in 'Punto 1':
1. Class Balance:
   - Binary distribution (Human vs. AI-Generated)
   - Candidate authors / generative models distribution (k=23)
   - Genre distribution
2. Text Length Distributions:
   - Histograms and boxplots for word counts and character counts (Human vs AI)
3. Stylometric Feature Distributions:
   - Kernel Density Estimation (KDE) / distribution plots for:
     * Type-Token Ratio (TTR / Lexical Richness)
     * Punctuation Density
     * Average Word Length
4. N-gram Frequency Analysis:
   - Horizontal bar charts of top 15-20 unigrams and bigrams by class
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import math

import matplotlib
matplotlib.use("Agg")  # Headless backend safe for scripts and servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Modern aesthetic color palette
PALETTE = {
    "human": "#2CA02C",       # Forest green
    "ai": "#1F77B4",          # Deep ocean blue
    "neutral": "#4C78A8",     # Slate blue
    "accent": "#E45756",      # Coral red
    "bg_grid": "#F3F4F6",     # Light soft grey
    "text": "#2D3748",        # Charcoal
}


def _setup_figure_style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = ""):
    """Apply clean, academic aesthetic to a matplotlib axes."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=PALETTE["text"])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, fontweight="semibold", labelpad=8, color=PALETTE["text"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, fontweight="semibold", labelpad=8, color=PALETTE["text"])
    ax.grid(True, linestyle="--", alpha=0.5, color="#D1D5DB")
    ax.set_axisbelow(True)
    ax.tick_params(colors=PALETTE["text"], labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(1.0)


def plot_binary_class_balance(
    label_distribution: Dict[int, int],
    output_path: Path,
    title: str = "Dataset Class Balance: Human vs AI-Generated Texts",
) -> Path:
    """
    Generate bar chart for binary class balance (Points 4, 12).
    Includes counts and percentage annotations on top of each bar.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    human_count = label_distribution.get(0, 0)
    ai_count = label_distribution.get(1, 0)
    total = human_count + ai_count or 1

    labels = ["Human Written\n(Label: 0)", "AI-Generated\n(Label: 1)"]
    counts = [human_count, ai_count]
    colors = [PALETTE["human"], PALETTE["ai"]]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="none", zorder=3)

    for bar, count in zip(bars, counts):
        pct = (count / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(counts) * 0.02),
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=PALETTE["text"],
        )

    ax.set_ylim(0, max(counts) * 1.18)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    _setup_figure_style(ax, title=title, ylabel="Total Number of Texts (N)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_author_model_distribution(
    model_distribution: Dict[str, int],
    output_path: Path,
    title: str = "Distribution of Texts Across Candidate Authors / Generative Models (k=23)",
) -> Path:
    """
    Generate horizontal bar chart showing text counts for each candidate author/model.
    Highlights Human vs different AI architectures (Points 1, 4).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_models = sorted(model_distribution.items(), key=lambda x: x[1], reverse=True)
    models = [m[0] for m in sorted_models]
    counts = [m[1] for m in sorted_models]
    colors = [PALETTE["human"] if m == "human" else PALETTE["ai"] for m in models]

    # Invert so highest appears at the top
    models = models[::-1]
    counts = counts[::-1]
    colors = colors[::-1]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    bars = ax.barh(models, counts, color=colors, height=0.65, edgecolor="none", zorder=3)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + (max(counts) * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            ha="left",
            va="center",
            fontsize=8.5,
            color=PALETTE["text"],
        )

    ax.set_xlim(0, max(counts) * 1.12)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    _setup_figure_style(ax, title=title, xlabel="Number of Texts", ylabel="Candidate Author / LLM Model")

    # Custom legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["human"], label="Human"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["ai"], label="AI Model"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_genre_distribution(
    genre_distribution: Dict[str, int],
    output_path: Path,
    title: str = "Dataset Composition by Text Genre / Domain",
) -> Path:
    """Generate bar chart showing text volume across domains (fiction, essays, news)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    genres = list(genre_distribution.keys())
    counts = list(genre_distribution.values())
    total = sum(counts) or 1
    colors = ["#4C78A8", "#72B7B2", "#ECA154"]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    bars = ax.bar(genres, counts, color=colors[:len(genres)], width=0.45, zorder=3)

    for bar, count in zip(bars, counts):
        pct = (count / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(counts) * 0.02),
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=PALETTE["text"],
        )

    ax.set_ylim(0, max(counts) * 1.18)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    _setup_figure_style(ax, title=title, xlabel="Genre / Domain", ylabel="Number of Texts")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_text_length_distribution(
    human_lengths: List[int],
    ai_lengths: List[int],
    output_path: Path,
    max_len_clip: int = 2000,
) -> Path:
    """
    Generate histograms and boxplots showing text length variance (words)
    between Human and AI texts to reveal structural discrepancies (Analisi Esplorativa).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(9, 7), dpi=150, sharex=True, gridspec_kw={"height_ratios": [0.35, 0.65]}
    )

    # Boxplot
    h_clipped = [min(x, max_len_clip) for x in human_lengths]
    a_clipped = [min(x, max_len_clip) for x in ai_lengths]

    box_data = [h_clipped, a_clipped]
    bp = ax_box.boxplot(
        box_data,
        vert=False,
        patch_artist=True,
        tick_labels=["Human", "AI"],
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=2, alpha=0.2),
    )
    bp["boxes"][0].set_facecolor(PALETTE["human"])
    bp["boxes"][1].set_facecolor(PALETTE["ai"])
    for box in bp["boxes"]:
        box.set_alpha(0.7)
    _setup_figure_style(ax_box, title="Structural Variance: Text Length Distribution (Word Count)")
    ax_box.set_ylabel("Class", color=PALETTE["text"])

    # Histogram
    bins = [i for i in range(0, max_len_clip + 50, 50)]
    ax_hist.hist(
        h_clipped,
        bins=bins,
        density=True,
        alpha=0.5,
        color=PALETTE["human"],
        label=f"Human (Mean: {sum(human_lengths)/max(1, len(human_lengths)):.0f} words)",
        edgecolor="none",
    )
    ax_hist.hist(
        a_clipped,
        bins=bins,
        density=True,
        alpha=0.5,
        color=PALETTE["ai"],
        label=f"AI (Mean: {sum(ai_lengths)/max(1, len(ai_lengths)):.0f} words)",
        edgecolor="none",
    )

    _setup_figure_style(ax_hist, title="", xlabel="Word Count per Text", ylabel="Probability Density")
    ax_hist.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _kde_1d(data: List[float], eval_points: List[float], bandwidth: Optional[float] = None) -> List[float]:
    """Lightweight 1D Gaussian Kernel Density Estimator without external dependencies."""
    n = len(data)
    if n == 0:
        return [0.0] * len(eval_points)

    # Silverman's rule of thumb
    if bandwidth is None:
        mean = sum(data) / n
        var = sum((x - mean) ** 2 for x in data) / n
        std = math.sqrt(max(var, 1e-6))
        bandwidth = 1.06 * std * (n ** (-0.2))

    bandwidth = max(bandwidth, 1e-4)
    densities = []
    inv_bw = 1.0 / bandwidth
    inv_norm = 1.0 / (math.sqrt(2 * math.pi) * bandwidth * n)

    # Sample data down if too large for density computation
    stride = max(1, n // 2000)
    sampled = data[::stride]
    n_s = len(sampled)
    inv_norm_s = 1.0 / (math.sqrt(2 * math.pi) * bandwidth * n_s)

    for x in eval_points:
        d = sum(math.exp(-0.5 * ((x - xi) * inv_bw) ** 2) for xi in sampled)
        densities.append(d * inv_norm_s)
    return densities


def plot_stylometric_distributions(
    raw_distributions: Dict[str, Dict[str, List[float]]],
    output_path: Path,
) -> Path:
    """
    Generate multi-panel KDE plots for key stylometric features:
    1. Type-Token Ratio (TTR / Lexical Richness)
    2. Punctuation Density
    3. Average Word Length
    (Fulfills Punto 1 Stylometric Feature Analysis).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

    features = [
        ("ttr", "Type-Token Ratio (TTR)", "Lexical Richness (Unique / Total Words)", 0.2, 0.8),
        ("punct_density", "Punctuation Density", "Punctuation Marks / Total Characters", 0.0, 0.08),
        ("avg_word_len", "Average Word Length", "Characters per Word", 3.5, 6.5),
    ]

    for ax, (fkey, title, xlabel, min_v, max_v) in zip(axes, features):
        h_data = [x for x in raw_distributions["Human"][fkey] if min_v <= x <= max_v]
        a_data = [x for x in raw_distributions["AI"][fkey] if min_v <= x <= max_v]

        eval_pts = [min_v + i * (max_v - min_v) / 150 for i in range(151)]

        h_kde = _kde_1d(h_data, eval_pts)
        a_kde = _kde_1d(a_data, eval_pts)

        ax.plot(eval_pts, h_kde, color=PALETTE["human"], linewidth=2, label="Human")
        ax.fill_between(eval_pts, h_kde, color=PALETTE["human"], alpha=0.25)

        ax.plot(eval_pts, a_kde, color=PALETTE["ai"], linewidth=2, label="AI")
        ax.fill_between(eval_pts, a_kde, color=PALETTE["ai"], alpha=0.25)

        h_mean = sum(h_data) / max(1, len(h_data))
        a_mean = sum(a_data) / max(1, len(a_data))

        ax.axvline(h_mean, color=PALETTE["human"], linestyle="--", alpha=0.8, label=rf"Human $\mu$={h_mean:.3f}")
        ax.axvline(a_mean, color=PALETTE["ai"], linestyle="--", alpha=0.8, label=rf"AI $\mu$={a_mean:.3f}")

        _setup_figure_style(ax, title=title, xlabel=xlabel, ylabel="Density")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    fig.suptitle("Stylometric Feature Density Comparison: Human vs AI", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_top_ngrams(
    top_unigrams: Dict[str, List[Tuple[str, int]]],
    top_bigrams: Dict[str, List[Tuple[str, int]]],
    output_path: Path,
    top_n: int = 15,
) -> Path:
    """
    Generate horizontal bar charts for the top 15-20 unigrams and bigrams
    comparing Human vs AI (much more informative and analytical than word clouds).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # Helper to plot one horizontal bar chart
    def _plot_hbar(ax, items: List[Tuple[str, int]], color: str, title: str, xlabel: str):
        sliced = items[:top_n][::-1]  # reverse for top at top
        tokens = [item[0] for item in sliced]
        freqs = [item[1] for item in sliced]

        bars = ax.barh(tokens, freqs, color=color, height=0.65, zorder=3)
        for bar, freq in zip(bars, freqs):
            ax.text(
                bar.get_width() + (max(freqs) * 0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{freq:,}",
                ha="left",
                va="center",
                fontsize=8,
                color=PALETTE["text"],
            )
        ax.set_xlim(0, max(freqs) * 1.15)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        _setup_figure_style(ax, title=title, xlabel=xlabel)

    # Row 0: Top Unigrams (filtered content words)
    _plot_hbar(axes[0, 0], top_unigrams["Human"], PALETTE["human"], f"Top {top_n} Unigrams - Human Written", "Occurrences")
    _plot_hbar(axes[0, 1], top_unigrams["AI"], PALETTE["ai"], f"Top {top_n} Unigrams - AI Generated", "Occurrences")

    # Row 1: Top Bigrams
    _plot_hbar(axes[1, 0], top_bigrams["Human"], PALETTE["human"], f"Top {top_n} Bigrams - Human Written", "Occurrences")
    _plot_hbar(axes[1, 1], top_bigrams["AI"], PALETTE["ai"], f"Top {top_n} Bigrams - AI Generated", "Occurrences")

    fig.suptitle("N-gram Comparative Analysis: Stylistic and Lexical Signatures", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_correlation_heatmap(
    corr_df,
    output_path: Path,
    title: str = "Stylometric Feature Collinearity & Target Correlation Matrix (Points 9, 13)",
) -> Path:
    """
    Generate a publication-ready correlation heatmap showing:
    1. Inter-feature correlations (collinearity)
    2. Feature-to-target correlations (r with author identity / AI class)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_vars = len(corr_df.columns)
    fig, ax = plt.subplots(figsize=(max(11, n_vars * 0.7), max(9, n_vars * 0.65)), dpi=160)

    # Use modern colormap access compatible with matplotlib 3.8+
    cmap = matplotlib.colormaps["coolwarm"]
    cax = ax.imshow(corr_df.values, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")

    # Ticks & labels
    ax.set_xticks(range(n_vars))
    ax.set_yticks(range(n_vars))
    clean_labels = [c.replace("stylometry__", "").replace("_", " ") for c in corr_df.columns]
    ax.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=8.5, color=PALETTE["text"])
    ax.set_yticklabels(clean_labels, fontsize=8.5, color=PALETTE["text"])

    # Annotate values if dimensions are manageable (<= 25)
    if n_vars <= 25:
        for i in range(n_vars):
            for j in range(n_vars):
                val = corr_df.values[i, j]
                text_color = "white" if abs(val) > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=text_color, fontweight="semibold")

    # Add Colorbar
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation Coefficient (r)", rotation=270, labelpad=15, fontsize=10, fontweight="bold", color=PALETTE["text"])
    cbar.ax.tick_params(labelsize=9, colors=PALETTE["text"])

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15, color=PALETTE["text"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_predictive_power(
    target_corr,
    output_path: Path,
    title: str = "Feature Predictive Power Regarding Target Output y (Points 9, 13)",
) -> Path:
    """
    Generate horizontal bar chart showing each feature's correlation (predictive power)
    with the target output y (Human = 0, AI = 1).
    Visualizes positive association (AI markers) and negative association (Human markers).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_corr = target_corr.sort_values(ascending=True)
    features = [f.replace("stylometry__", "").replace("_", " ") for f in sorted_corr.index]
    values = sorted_corr.values

    colors = [PALETTE["human"] if v < 0 else PALETTE["ai"] for v in values]

    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.42)), dpi=160)
    bars = ax.barh(features, values, color=colors, height=0.62, edgecolor="none", zorder=3)

    for bar, val in zip(bars, values):
        offset = 0.015 if val >= 0 else -0.015
        ha = "left" if val >= 0 else "right"
        text_color = PALETTE["ai"] if val >= 0 else PALETTE["human"]
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            ha=ha,
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=text_color,
        )

    ax.axvline(0, color="#94A3B8", linewidth=1.2, linestyle="-", zorder=2)
    max_abs = max(abs(min(values)), abs(max(values))) * 1.25
    ax.set_xlim(-max_abs, max_abs)

    _setup_figure_style(
        ax,
        title=title,
        xlabel="Pearson Correlation with Target Output y (Human = 0, AI = 1)",
        ylabel="Extracted Stylometric Feature",
    )

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["human"], label="Predictive of Human Writing (r < 0)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["ai"], label="Predictive of AI Generation (r > 0)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_distributions_entire_dataset(
    df,
    output_path: Path,
    features: Optional[List[Tuple[str, str, str]]] = None,
    title: str = "Key Feature Distributions Across the Entire Dataset (Point 10)",
) -> Path:
    """
    Generate histograms with KDE density estimation and statistical summaries (Point 10)
    to show the distribution of key features (X) across the entire dataset.
    """
    import numpy as np
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if features is None:
        features = [
            ("word_count", "Text Length (Word Count)", "Words per Text"),
            ("ttr", "Type-Token Ratio (TTR)", "Lexical Richness (Unique / Total)"),
            ("avg_word_len", "Average Word Length", "Characters per Word"),
            ("punct_density", "Punctuation Density", "Punctuation Marks / Total Chars"),
            ("avg_sentence_len", "Average Sentence Length", "Words per Sentence"),
            ("functional_lexical_ratio", "Functional-to-Lexical Ratio", "Grammatical / Content Words"),
        ]

    n_features = len(features)
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.8 * n_rows), dpi=160)
    axes_flat = axes.flatten() if n_features > 1 else [axes]

    for idx, (fkey, ftitle, xlabel) in enumerate(features):
        ax = axes_flat[idx]
        if fkey not in df.columns:
            continue

        raw_vals = df[fkey].dropna().values
        if fkey in ["word_count", "avg_sentence_len"]:
            clip_val = float(np.percentile(raw_vals, 99.5))
            vals = raw_vals[raw_vals <= clip_val]
        else:
            vals = raw_vals

        mean_val = float(np.mean(vals))
        median_val = float(np.median(vals))
        std_val = float(np.std(vals))
        q25 = float(np.percentile(vals, 25))
        q75 = float(np.percentile(vals, 75))
        iqr = q75 - q25

        # Histogram
        n_bins = 40
        ax.hist(
            vals, bins=n_bins, density=True, color="#3B82F6", alpha=0.55, edgecolor="#2563EB", linewidth=0.7, zorder=2
        )

        # Smooth KDE curve
        eval_pts = np.linspace(min(vals), max(vals), 150).tolist()
        kde_vals = _kde_1d(vals.tolist(), eval_pts)
        ax.plot(eval_pts, kde_vals, color="#1D4ED8", linewidth=2.0, zorder=3, label="KDE Density")

        # Vertical statistical markers
        ax.axvline(mean_val, color="#DC2626", linestyle="--", linewidth=1.8, zorder=4, label=f"Mean (μ={mean_val:.2f})")
        ax.axvline(median_val, color="#15803D", linestyle="-.", linewidth=1.8, zorder=4, label=f"Median (M={median_val:.2f})")

        # Summary box
        stats_text = (
            f"μ: {mean_val:.2f}\n"
            f"σ: {std_val:.2f}\n"
            f"Med: {median_val:.2f}\n"
            f"IQR: {iqr:.2f}"
        )
        ax.text(
            0.95, 0.72, stats_text, transform=ax.transAxes,
            fontsize=8.5, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9),
            zorder=5, color=PALETTE["text"], family="monospace"
        )

        _setup_figure_style(ax, title=ftitle, xlabel=xlabel, ylabel="Probability Density")
        ax.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

    for j in range(n_features, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995, color=PALETTE["text"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


# =========================================================================
# Stage 3 & Stage 4 Visualizations: Feature Influence & Baseline Models
# =========================================================================

def plot_comprehensive_feature_influence(
    df_dense_uni,
    df_global_weights,
    output_path: Path,
    top_n: int = 20,
) -> Path:
    """
    Dual-panel figure:
    - Left: Top 20 Dense Features (Stylometry + Function Words) by Pearson correlation r with target y.
    - Right: Top 20 Global Features across all 4,600 dimensions by Logistic Regression Beta weight.
    """
    import numpy as np
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8.5), dpi=160)

    # Panel 1: Top Dense Univariate Pearson r
    top_uni = df_dense_uni.head(top_n).iloc[::-1]
    labels_uni = [f"[{r['family'][:4]}] {r['short_name']}" for _, r in top_uni.iterrows()]
    vals_uni = top_uni["pearson_r"].values
    colors_uni = [PALETTE["ai"] if v > 0 else PALETTE["human"] for v in vals_uni]

    bars1 = ax1.barh(labels_uni, vals_uni, color=colors_uni, alpha=0.88, edgecolor="#1E293B", height=0.68)
    ax1.axvline(0, color="#334155", linewidth=1.2)
    for bar, v in zip(bars1, vals_uni):
        offset = 0.012 if v >= 0 else -0.012
        ha = "left" if v >= 0 else "right"
        ax1.text(
            v + offset, bar.get_y() + bar.get_height() / 2.0,
            f"{v:+.3f}", va="center", ha=ha, fontsize=8.5, fontweight="semibold", color=PALETTE["text"]
        )
    ax1.set_xlim(min(vals_uni) - 0.12, max(vals_uni) + 0.12)
    _setup_figure_style(
        ax1,
        title=f"Top {top_n} Dense Features: Pearson r with Output y\n(Green < 0 = Human Marker | Blue > 0 = AI Marker)",
        xlabel="Pearson Correlation Coefficient (r)",
    )

    # Panel 2: Top Global Logistic Regression Beta Weights
    top_glob = df_global_weights.head(top_n).iloc[::-1]
    labels_glob = [f"[{r['family'][:4]}] '{r['short_name']}'" for _, r in top_glob.iterrows()]
    vals_glob = top_glob["beta_weight"].values
    colors_glob = [PALETTE["ai"] if v > 0 else PALETTE["human"] for v in vals_glob]

    bars2 = ax2.barh(labels_glob, vals_glob, color=colors_glob, alpha=0.88, edgecolor="#1E293B", height=0.68)
    ax2.axvline(0, color="#334155", linewidth=1.2)
    for bar, v in zip(bars2, vals_glob):
        offset = 0.05 if v >= 0 else -0.05
        ha = "left" if v >= 0 else "right"
        ax2.text(
            v + offset, bar.get_y() + bar.get_height() / 2.0,
            f"{v:+.2f}", va="center", ha=ha, fontsize=8.5, fontweight="semibold", color=PALETTE["text"]
        )
    ax2.set_xlim(min(vals_glob) - 0.45, max(vals_glob) + 0.45)
    _setup_figure_style(
        ax2,
        title=f"Top {top_n} Global Features (4,600 Dims): Standardized Beta Weights\n(Multivariate Influence in L2 Logistic Regression)",
        xlabel="Standardized Coefficient (Beta_j)",
    )

    fig.suptitle(
        "Stage 3: Univariate & Multivariate Feature Influence on Target Output y (N = 23,707 Train)",
        fontsize=14, fontweight="bold", y=0.99, color=PALETTE["text"]
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_llm_family_fingerprints(
    df_family_profiles,
    df_multiclass_anova,
    output_path: Path,
    top_n_features: int = 15,
) -> Path:
    """
    Generates a heatmap of standardized stylometric feature signatures across the 7
    architectural families (Human, OpenAI, Google Gemini, Meta LLaMA, Mistral, DeepSeek/Qwen, Falcon)
    ordered by Multiclass ANOVA F-statistic.
    """
    import seaborn as sns
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sty_anova = df_multiclass_anova[df_multiclass_anova["family"] == "Stylometry"].head(top_n_features)
    top_cols = [c for c in sty_anova["short_name"].tolist() if c in df_family_profiles.columns]

    mat = df_family_profiles[top_cols].T  # Features as rows, LLM Families as columns

    fig, ax = plt.subplots(figsize=(12.5, 8.0), dpi=160)
    sns.heatmap(
        mat,
        annot=True,
        fmt="+.2f",
        cmap="RdBu_r",
        center=0.0,
        vmin=-1.2,
        vmax=1.2,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Mean Standardized Z-Score (Relative to Corpus Mean)"},
        ax=ax,
    )
    ax.set_title(
        "LLM Architectural Family Fingerprints: Top 15 Discriminative Stylometric Features\n"
        "(Ranked by Multiclass One-Way ANOVA F-Statistic Across 22 AI Models)",
        fontsize=13, fontweight="bold", pad=14, color=PALETTE["text"]
    )
    ax.set_xlabel("Author / Generative AI Architecture Family", fontsize=11, fontweight="semibold", labelpad=8)
    ax.set_ylabel("Stylometric Feature (Ordered by Multiclass ANOVA F)", fontsize=11, fontweight="semibold", labelpad=8)
    ax.tick_params(axis="x", rotation=22, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_family_ablation(
    df_ablation,
    output_path: Path,
) -> Path:
    """
    Plots the Feature Family Ablation Study comparing Binary Macro F1 (Human vs AI)
    and Multiclass 23-Author Attribution Macro F1 across individual families, combinations, and LOFO.
    """
    import numpy as np
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.8), dpi=160)

    # Panel 1: Single & Combined Families
    sub_main = df_ablation[df_ablation["group_type"] != "Leave-One-Out"].copy()
    x = np.arange(len(sub_main))
    width = 0.36

    b1 = ax1.bar(
        x - width / 2, sub_main["binary_macro_f1"] * 100, width,
        label="Task 1: Binary Detection Macro F1 (%)", color=PALETTE["ai"], alpha=0.9, edgecolor="#1E293B"
    )
    b2 = ax1.bar(
        x + width / 2, sub_main["multiclass23_macro_f1"] * 100, width,
        label="Task 2: 23-Class Attribution Macro F1 (%)", color="#E45756", alpha=0.88, edgecolor="#1E293B"
    )

    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(sub_main["config_name"], rotation=25, ha="right", fontsize=9.5)
    ax1.set_ylim(0, 118)
    _setup_figure_style(
        ax1,
        title="Feature Family Ablation: Individual vs Combined Spaces",
        ylabel="Validation Macro F1-Score (%)",
    )
    ax1.legend(loc="upper left", framealpha=0.92)

    # Panel 2: Leave-One-Family-Out (LOFO) Impact (Delta F1 in percentage points)
    sub_lofo = df_ablation[df_ablation["group_type"] == "Leave-One-Out"].copy()
    x2 = np.arange(len(sub_lofo))

    d_bin = sub_lofo["delta_binary_f1"] * 100.0
    d_mul = sub_lofo["delta_multi_f1"] * 100.0

    b3 = ax2.bar(
        x2 - width / 2, d_bin, width,
        label="Delta Binary Macro F1 (% pts)", color=PALETTE["neutral"], alpha=0.9, edgecolor="#1E293B"
    )
    b4 = ax2.bar(
        x2 + width / 2, d_mul, width,
        label="Delta 23-Class Attribution F1 (% pts)", color="#F58518", alpha=0.9, edgecolor="#1E293B"
    )
    ax2.axhline(0, color="#0F172A", linewidth=1.2)

    for bar in list(b3) + list(b4):
        h = bar.get_height()
        va = "top" if h < 0 else "bottom"
        offset = -0.15 if h < 0 else 0.15
        ax2.text(bar.get_x() + bar.get_width() / 2, h + offset, f"{h:+.2f}%", ha="center", va=va, fontsize=8.5, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(sub_lofo["config_name"], rotation=20, ha="right", fontsize=9.5)
    min_y = min(min(d_bin), min(d_mul), -1.0) * 1.35
    max_y = max(max(d_bin), max(d_mul), 0.5) + 1.0
    ax2.set_ylim(min_y, max_y)
    _setup_figure_style(
        ax2,
        title="Leave-One-Family-Out (LOFO): Performance Drop vs Full System",
        ylabel="Change in Validation Macro F1 (% Points)",
    )
    ax2.legend(loc="lower left", framealpha=0.92)

    fig.suptitle(
        "Stage 3: Multi-Level Feature Family Ablation Study (Validation Set N = 3,589)",
        fontsize=14, fontweight="bold", y=0.99, color=PALETTE["text"]
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_baseline_models_benchmark(
    df_binary,
    df_multiclass,
    roc_curves: Dict[str, Any],
    output_path: Path,
) -> Path:
    """
    3-Panel Figure summarizing Stage 4 Baseline Model Performance:
    - Panel 1: Task 1 Binary Detection (5-Fold CV Macro F1 vs Validation Macro F1 & ROC-AUC)
    - Panel 2: Task 1 ROC Curves on Validation Set
    - Panel 3: Task 2 Model Attribution (23-Class Macro F1, 22-AI Exact Acc, 7-Family Acc)
    """
    import numpy as np
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6.5), dpi=160)

    # Panel 1: Binary Baselines Bar Chart
    x1 = np.arange(len(df_binary))
    w = 0.27
    ax1.bar(x1 - w, df_binary["cv5_macro_f1_mean"] * 100, w, label="5-Fold CV Macro F1", color="#64748B", alpha=0.88)
    b_val = ax1.bar(x1, df_binary["val_macro_f1"] * 100, w, label="Val Macro F1 (Official)", color=PALETTE["ai"], alpha=0.92)
    ax1.bar(x1 + w, df_binary["val_roc_auc"] * 100, w, label="Val ROC-AUC", color=PALETTE["human"], alpha=0.85)

    for bar in b_val:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    ax1.set_xticks(x1)
    ax1.set_xticklabels(df_binary["model_name"], rotation=25, ha="right", fontsize=9.0)
    ax1.set_ylim(30, 108)
    _setup_figure_style(
        ax1,
        title="Task 1: Binary Detection Baselines\n(Human vs AI - N_val = 3,589)",
        ylabel="Score (%)",
    )
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    # Panel 2: ROC Curves
    line_styles = ["--", "-.", ":", "-", "-"]
    colors = ["#94A3B8", "#F58518", "#2CA02C", "#1F77B4", "#E45756"]
    for idx, (mname, (fpr, tpr, auc_val)) in enumerate(roc_curves.items()):
        ax2.plot(
            fpr, tpr,
            linestyle=line_styles[idx % len(line_styles)],
            color=colors[idx % len(colors)],
            linewidth=2.2,
            label=f"{mname} (AUC = {auc_val:.4f})",
        )
    ax2.plot([0, 1], [0, 1], color="#CBD5E1", linestyle="--", linewidth=1.0)
    _setup_figure_style(
        ax2,
        title="Task 1: Validation ROC Curves\n(Binary GenAI Detection)",
        xlabel="False Positive Rate (1 - Specificity)",
        ylabel="True Positive Rate (Sensitivity)",
    )
    ax2.legend(loc="lower right", fontsize=8.2, framealpha=0.92)

    # Panel 3: Task 2 Model Attribution (23 Classes & Two-Stage Hierarchical)
    x3 = np.arange(len(df_multiclass))
    b_m1 = ax3.bar(
        x3 - w, df_multiclass["val_23class_macro_f1"] * 100, w,
        label="23-Class Macro F1", color="#E45756", alpha=0.9
    )
    b_m2 = ax3.bar(
        x3, df_multiclass["val_23class_top1_acc"] * 100, w,
        label="23-Class Top-1 Acc", color=PALETTE["ai"], alpha=0.9
    )
    b_m3 = ax3.bar(
        x3 + w, df_multiclass["val_7family_acc"] * 100, w,
        label="7-Family Attribution Acc", color=PALETTE["human"], alpha=0.88
    )

    for bar in b_m2:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

    short_mnames = [
        m.replace("Two-Stage Hierarchical (Stage1 Bin + ", "2-Stage (")
         .replace("Logistic Regression (Multinomial L2 - 23c)", "Logistic Reg (23c)")
         .replace("Linear SVM (One-vs-Rest - 23c)", "Linear SVM (23c)")
         .replace("Complement Naive Bayes (23c)", "Complement NB (23c)")
         .replace("Dummy (Majority Class - 23c)", "Dummy (23c)")
        for m in df_multiclass["model_name"]
    ]
    ax3.set_xticks(x3)
    ax3.set_xticklabels(short_mnames, rotation=25, ha="right", fontsize=8.8)
    ax3.set_ylim(0, 105)
    _setup_figure_style(
        ax3,
        title="Task 2: Author & AI Model Attribution\n(Direct 23-Class vs Two-Stage Hierarchical)",
        ylabel="Validation Score (%)",
    )
    ax3.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    fig.suptitle(
        "Stage 4: Baseline Models Evaluation — Dual-Task Benchmark (Binary Detection & LLM Attribution)",
        fontsize=14, fontweight="bold", y=0.995, color=PALETTE["text"]
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_baseline_diagnostics_and_confusion(
    cm_binary,
    df_cm_family,
    df_genre,
    output_path: Path,
) -> Path:
    """
    3-Panel Diagnostic Figure:
    - Panel 1: 2x2 Binary Confusion Matrix (Linear SVM)
    - Panel 2: 7x7 Row-Normalized Architectural Family Confusion Matrix (%)
    - Panel 3: Cross-Genre Performance Breakdown (Fiction, Essays, News)
    """
    import numpy as np
    import seaborn as sns
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20.5, 6.4), dpi=160)

    # Panel 1: 2x2 Binary Confusion Matrix
    cm_norm = cm_binary.astype(float) / cm_binary.sum(axis=1, keepdims=True) * 100.0
    annot_bin = np.empty_like(cm_binary, dtype=object)
    for i in range(2):
        for j in range(2):
            annot_bin[i, j] = f"{cm_binary[i, j]:,}\n({cm_norm[i, j]:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot_bin,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred: Human (0)", "Pred: AI (1)"],
        yticklabels=["True: Human (0)", "True: AI (1)"],
        annot_kws={"size": 11, "weight": "bold"},
        linewidths=1.5,
        linecolor="white",
        ax=ax1,
    )
    ax1.set_title("Task 1: Binary Confusion Matrix\n(Linear SVM Baseline - Val N = 3,589)", fontsize=12.5, fontweight="bold", pad=12)

    # Panel 2: 7x7 Family Confusion Matrix (Row-normalized %)
    fam_vals = df_cm_family.values.astype(float)
    fam_row_sums = np.maximum(fam_vals.sum(axis=1, keepdims=True), 1.0)
    fam_pct = fam_vals / fam_row_sums * 100.0

    short_fams = ["Human", "OpenAI", "Google", "LLaMA", "Mistral", "DeepSeek", "Falcon"]
    sns.heatmap(
        fam_pct,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=short_fams,
        yticklabels=short_fams,
        cbar_kws={"label": "Row Recall (%)"},
        annot_kws={"size": 8.5, "weight": "semibold"},
        linewidths=0.8,
        linecolor="white",
        ax=ax2,
    )
    ax2.set_title("Task 2: Architectural Family Attribution (%)\n(Two-Stage Hierarchical Baseline)", fontsize=12.5, fontweight="bold", pad=12)
    ax2.set_xlabel("Predicted Family", fontsize=10.5, fontweight="semibold")
    ax2.set_ylabel("True Family", fontsize=10.5, fontweight="semibold")
    ax2.tick_params(axis="x", rotation=30)
    ax2.tick_params(axis="y", rotation=0)

    # Panel 3: Cross-Genre Performance for Linear SVM & Logistic Regression
    sub_g = df_genre[df_genre["baseline_model"] == "Linear SVM (Stamatatos Standard)"].copy()
    xg = np.arange(len(sub_g))
    w = 0.28

    b1 = ax3.bar(xg - w, sub_g["binary_macro_f1"] * 100, w, label="Binary Detection Macro F1", color=PALETTE["ai"], alpha=0.92)
    b2 = ax3.bar(xg, sub_g["binary_accuracy"] * 100, w, label="Binary Accuracy", color=PALETTE["human"], alpha=0.88)
    b3 = ax3.bar(xg + w, sub_g["multiclass23_acc_best"] * 100, w, label="23-Class Attribution Acc", color="#E45756", alpha=0.88)

    for bar in list(b1) + list(b3):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    genre_labels = [f"{r['genre'].capitalize()}\n(n={r['n_samples']})" for _, r in sub_g.iterrows()]
    ax3.set_xticks(xg)
    ax3.set_xticklabels(genre_labels, fontsize=10)
    ax3.set_ylim(0, 108)
    _setup_figure_style(
        ax3,
        title="Cross-Domain Generalization Across Genres\n(Fiction vs Essays vs News)",
        ylabel="Validation Score (%)",
    )
    ax3.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    fig.suptitle(
        "Stage 4: Baseline Error Analysis — Confusion Matrices & Cross-Domain Generalization",
        fontsize=14, fontweight="bold", y=0.995, color=PALETTE["text"]
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_per_llm_detection_and_attribution(
    df_per_model,
    output_path: Path,
) -> Path:
    """
    Horizontal grouped bar chart across all 23 candidate models (Human + 22 AI models)
    showing Binary Detection Accuracy, Exact Model Attribution Accuracy, and Family Attribution Accuracy.
    """
    import numpy as np
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_plot = df_per_model.sort_values(by="exact_model_attribution_acc", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(14.5, 10.5), dpi=160)
    y = np.arange(len(df_plot))
    h = 0.27

    ax.barh(
        y + h, df_plot["binary_detection_acc"] * 100, h,
        label="Task 1: Binary Detection Recall (Human vs AI %)", color=PALETTE["ai"], alpha=0.9
    )
    ax.barh(
        y, df_plot["family_attribution_acc"] * 100, h,
        label="Task 2: Architectural Family Attribution Acc (%)", color=PALETTE["human"], alpha=0.88
    )
    b_exact = ax.barh(
        y - h, df_plot["exact_model_attribution_acc"] * 100, h,
        label="Task 2: Exact Model Attribution Acc (23 Classes %)", color="#E45756", alpha=0.9
    )

    for bar, val in zip(b_exact, df_plot["exact_model_attribution_acc"] * 100):
        ax.text(
            val + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=8.0, fontweight="bold", color=PALETTE["text"]
        )

    labels = [f"{r['model_name']} (n={r['n_val_samples']})" for _, r in df_plot.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.2)
    ax.set_xlim(0, 108)
    _setup_figure_style(
        ax,
        title="Per-Author & Per-LLM Performance Breakdown on Validation Set (N = 3,589)\n",
        xlabel="Validation Accuracy / Recall (%)",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, fontsize=9.5, framealpha=0.94)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path



def plot_feature_distributions_by_author(
    df,
    output_path: Path,
    features: Optional[List[Tuple[str, str, str]]] = None,
    top_k_authors: int = 11,
    title: str = "Feature Distributions Among Different Candidate Authors / LLM Models (Point 10)",
) -> Path:
    """
    Generate publication-ready boxplots comparing key feature distributions (X)
    broken down among candidate authors (Human vs individual generative AI models) (Point 10).
    """
    import numpy as np
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "model" not in df.columns:
        raise ValueError("DataFrame must contain 'model' column for author-level breakdown.")

    # Select top authors by frequency, always keeping 'human' as the primary baseline
    model_counts = df["model"].value_counts()
    all_models = [m for m in model_counts.index if m != "human"]
    selected_ai_models = all_models[:top_k_authors - 1]
    authors_to_plot = ["human"] + selected_ai_models

    df_filtered = df[df["model"].isin(authors_to_plot)].copy()

    if features is None:
        features = [
            ("ttr", "Type-Token Ratio (TTR)", "Lexical Diversity (V / N)"),
            ("punct_density", "Punctuation Density", "Marks / Total Characters"),
            ("avg_word_len", "Average Word Length", "Characters per Word"),
            ("functional_lexical_ratio", "Functional-to-Lexical Ratio", "Closed Class / Open Class"),
            ("avg_sentence_len", "Average Sentence Length", "Words per Sentence"),
            ("word_count", "Text Length (Word Count)", "Total Words"),
        ]

    n_features = len(features)
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5.2 * n_rows), dpi=160)
    axes_flat = axes.flatten() if n_features > 1 else [axes]

    display_names = {
        "human": "Human (Author)",
        "gpt-4o": "GPT-4o",
        "gpt-3.5-turbo": "GPT-3.5-Turbo",
        "gpt-4o-mini": "GPT-4o-Mini",
        "ministral-8b-instruct-2410": "Ministral-8B",
        "gemini-2.0-flash": "Gemini-2.0-Flash",
        "o3-mini": "OpenAI o3-Mini",
        "gemini-1.5-pro": "Gemini-1.5-Pro",
        "llama-3.1-8b-instruct": "LLaMA-3.1-8B",
        "deepseek-r1-distill-qwen-32b": "DeepSeek-R1-32B",
        "falcon3-10b-instruct": "Falcon3-10B",
    }

    # Order from bottom to top so Human appears at the top of the boxplot
    ordered_authors = authors_to_plot[::-1]
    tick_labels = [display_names.get(a, a) for a in ordered_authors]

    for idx, (fkey, ftitle, xlabel) in enumerate(features):
        ax = axes_flat[idx]
        if fkey not in df.columns:
            continue

        data_by_author = []
        for author in ordered_authors:
            sub = df_filtered[df_filtered["model"] == author][fkey].dropna().values
            if fkey in ["word_count", "avg_sentence_len"]:
                q99 = np.percentile(sub, 99)
                sub = sub[sub <= q99 * 1.2]
            data_by_author.append(sub)

        bp = ax.boxplot(
            data_by_author,
            vert=False,
            patch_artist=True,
            tick_labels=tick_labels,
            widths=0.6,
            medianprops=dict(color="#0F172A", linewidth=1.6),
            boxprops=dict(linewidth=1.0),
            whiskerprops=dict(color="#475569", linewidth=1.0),
            capprops=dict(color="#475569", linewidth=1.0),
            flierprops=dict(marker="o", markersize=2, alpha=0.25, markeredgecolor="none", markerfacecolor="#64748B"),
        )

        for patch, author in zip(bp["boxes"], ordered_authors):
            if author == "human":
                patch.set_facecolor(PALETTE["human"])
                patch.set_alpha(0.85)
                patch.set_edgecolor("#15803D")
            else:
                patch.set_facecolor(PALETTE["ai"])
                patch.set_alpha(0.65)
                patch.set_edgecolor("#1D4ED8")

        _setup_figure_style(ax, title=ftitle, xlabel=xlabel)
        ax.tick_params(axis="y", labelsize=8.5)

    for j in range(n_features, len(axes_flat)):
        axes_flat[j].set_visible(False)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["human"], alpha=0.85, edgecolor="#15803D", label="Human Author"),
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["ai"], alpha=0.65, edgecolor="#1D4ED8", label="AI Generative Model"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.99, 0.995), framealpha=0.9, fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995, color=PALETTE["text"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path



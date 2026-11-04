#!/usr/bin/env python3
"""
generate_qwen_multifaceted_figure.py

Generates a publication-grade, 6-panel multifaceted overview figure of the
Qwen-14B Phylogenetic Tree Surgery experiment (Panels A–F):
  - Panel A: Step 1 — 3 Sibling 14B Checkpoints & SwiGLU Neuron Signatures
  - Panel B: Step 2 — GPU BME Phylogenetic Tree & Clade Partitioning (6,171 Clades)
  - Panel C: Step 3 — Why Quartet Topology Violations Beat Weight Magnitude (||dW||_F)
  - Panel D: Result 1 — Head-to-Head Knock-Out vs. 7 Competitors (20% Budget)
  - Panel E: Result 2 — Ultra-Sparse Top-K Clade Knock-Out Curve (Top 5 to Top 100)
  - Panel F: Result 3 — Zero-Training Hybrid Chimera (Math vs. Code Pareto Frontier)

Outputs:
  - qwen_tree_surgery_overview.png (300 DPI)
  - qwen_tree_surgery_overview.pdf (Vector PDF)
  - qwen_tree_surgery_overview.svg (Vector SVG)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 1.0


def add_panel_card(fig, bbox, title_badge, title_text, badge_color="#4F46E5"):
    """Draw a rounded card background and header banner for a panel."""
    x0, y0, w, h = bbox
    bg_ax = fig.add_axes([x0, y0, w, h], frameon=False)
    bg_ax.set_xlim(0, 1)
    bg_ax.set_ylim(0, 1)
    bg_ax.axis("off")

    card = FancyBboxPatch(
        (0.008, 0.012), 0.984, 0.976,
        boxstyle="round,pad=0.012,rounding_size=0.028",
        facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.4, zorder=1
    )
    bg_ax.add_patch(card)

    # Top accent pill (wide enough so badge text never clips)
    header_bg = FancyBboxPatch(
        (0.028, 0.880), 0.255, 0.072,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        facecolor=badge_color, edgecolor="none", zorder=2
    )
    bg_ax.add_patch(header_bg)
    bg_ax.text(
        0.1555, 0.916, title_badge,
        color="#FFFFFF", fontsize=8.4, fontweight="bold",
        ha="center", va="center", zorder=3
    )
    bg_ax.text(
        0.302, 0.916, title_text,
        color="#0F172A", fontsize=10.6, fontweight="bold",
        ha="left", va="center", zorder=3
    )
    bg_ax.plot([0.025, 0.975], [0.852, 0.852], color="#E2E8F0", lw=1.0, zorder=2)
    return bg_ax


def draw_panel_a(ax):
    """Panel A: 3 Sibling 14B Models + Neuron Functional Signature Matrix."""
    # Left half: Evolutionary divergence of the 3 sibling checkpoints
    ax.text(0.265, 0.805, "Three Sibling 14B Checkpoints\n(Same 48-Layer × 13,824-Neuron Body)",
            ha="center", va="top", fontsize=8.4, fontweight="bold", color="#1E293B")

    # Root node: Qwen2.5-14B Base
    base_box = FancyBboxPatch((0.095, 0.565), 0.34, 0.115,
                              boxstyle="round,pad=0.008,rounding_size=0.02",
                              facecolor="#F1F5F9", edgecolor="#475569", lw=1.5, zorder=3)
    ax.add_patch(base_box)
    ax.text(0.265, 0.638, "Qwen2.5-14B (Base)", ha="center", va="center",
            fontsize=8.8, fontweight="bold", color="#0F172A", zorder=4)
    ax.text(0.265, 0.592, "Common Ancestor Checkpoint", ha="center", va="center",
            fontsize=7.3, color="#475569", zorder=4)

    # Branches to Coder and Reasoner
    ax.annotate("", xy=(0.135, 0.38), xytext=(0.22, 0.56),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.45",
                                color="#0284C7", lw=1.9))
    ax.annotate("", xy=(0.395, 0.38), xytext=(0.31, 0.56),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.45",
                                color="#7C3AED", lw=1.9))

    coder_box = FancyBboxPatch((0.025, 0.22), 0.22, 0.145,
                               boxstyle="round,pad=0.008,rounding_size=0.02",
                               facecolor="#E0F2FE", edgecolor="#0284C7", lw=1.4, zorder=3)
    ax.add_patch(coder_box)
    ax.text(0.135, 0.318, "Coder-14B", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color="#0369A1", zorder=4)
    ax.text(0.135, 0.260, "Code Fine-Tuned\n(Code PPL: 2.18)", ha="center", va="center",
            fontsize=7.1, color="#0C4A6E", zorder=4)

    r1_box = FancyBboxPatch((0.275, 0.22), 0.23, 0.145,
                            boxstyle="round,pad=0.008,rounding_size=0.02",
                            facecolor="#F3E8FF", edgecolor="#7C3AED", lw=1.4, zorder=3)
    ax.add_patch(r1_box)
    ax.text(0.39, 0.318, "DeepSeek-R1-14B", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color="#6D28D9", zorder=4)
    ax.text(0.39, 0.260, "Reasoning Distilled\n(Math NLL: 1.48)", ha="center", va="center",
            fontsize=7.1, color="#4C1D95", zorder=4)

    # Vertical dashed divider
    ax.plot([0.535, 0.535], [0.18, 0.81], color="#CBD5E1", lw=1.0, ls="--")

    # Right half: SwiGLU Neuron Signature & 13,824 x 13,824 Cosine Matrix
    ax.text(0.755, 0.805, "Each MLP Neuron i = One 'Species'\n(13,824 Neurons per Layer)",
            ha="center", va="top", fontsize=8.4, fontweight="bold", color="#1E293B")

    sig_colors = [("#64748B", "Base"), ("#0284C7", "Coder"), ("#7C3AED", "R1")]
    for idx, (col, lbl) in enumerate(sig_colors):
        x_left = 0.56 + idx * 0.132
        rect = FancyBboxPatch((x_left, 0.545), 0.118, 0.125,
                              boxstyle="round,pad=0.004,rounding_size=0.012",
                              facecolor=col, edgecolor="none", alpha=0.14, zorder=2)
        border = FancyBboxPatch((x_left, 0.545), 0.118, 0.125,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor="none", edgecolor=col, lw=1.3, zorder=3)
        ax.add_patch(rect)
        ax.add_patch(border)
        ax.text(x_left + 0.059, 0.625, lbl, ha="center", va="center",
                fontsize=8.0, fontweight="bold", color=col)
        ax.text(x_left + 0.059, 0.575, "gate/up/down", ha="center", va="center",
                fontsize=6.5, color="#334155")

    ax.annotate("", xy=(0.755, 0.415), xytext=(0.755, 0.525),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.45",
                                color="#4F46E5", lw=1.8))

    mat_box = FancyBboxPatch((0.56, 0.22), 0.39, 0.175,
                             boxstyle="round,pad=0.008,rounding_size=0.02",
                             facecolor="#EEF2FF", edgecolor="#4F46E5", lw=1.4, zorder=3)
    ax.add_patch(mat_box)
    ax.text(0.755, 0.34, "GPU Cosine Distance Matrix D",
            ha="center", va="center", fontsize=8.3, fontweight="bold", color="#312E81")
    ax.text(0.755, 0.268, "13,824 × 13,824 per layer × 48 layers\n(663,552 total MLP neurons)",
            ha="center", va="center", fontsize=7.3, color="#4338CA")

    # Bottom takeaway strip
    foot = FancyBboxPatch((0.03, 0.04), 0.94, 0.12,
                          boxstyle="round,pad=0.006,rounding_size=0.015",
                          facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1.0, zorder=2)
    ax.add_patch(foot)
    ax.text(0.50, 0.10,
            "Homology across Checkpoints: Because all 3 models share the same Base initialization,\n"
            "neuron i in layer l is homologous across Base, Coder, and R1 — enabling phylogenetic trees!",
            ha="center", va="center", fontsize=7.4, color="#334155")


def draw_panel_b(ax):
    """Panel B: GPU Phylo2Vec BME Tree Inference & Clade Decomposition."""
    ax.text(0.50, 0.805, "GPU Balanced Minimum Evolution (BME) Tree per Layer\n13,824 Neurons → ~128 Clades/Layer (6,171 Total Clades)",
            ha="center", va="top", fontsize=8.5, fontweight="bold", color="#1E293B")

    # Stylized binary phylogenetic tree with 4 clade leaves at x = 0.14, 0.38, 0.62, 0.86
    tree_edges = [
        ((0.50, 0.66), (0.26, 0.55)),
        ((0.50, 0.66), (0.74, 0.55)),
        ((0.26, 0.55), (0.14, 0.43)),
        ((0.26, 0.55), (0.38, 0.43)),
        ((0.74, 0.55), (0.62, 0.43)),
        ((0.74, 0.55), (0.86, 0.43)),
    ]
    for (x1, y1), (x2, y2) in tree_edges:
        ax.plot([x1, x2], [y1, y1], color="#64748B", lw=2.0, zorder=2)
        ax.plot([x2, x2], [y1, y2], color="#64748B", lw=2.0, zorder=2)

    clade_specs = [
        (0.14, "#EF4444", "#FEE2E2", "Clade #14 (Violated)\nRewired for Math\nΔQ = 0.41 (High)", True),
        (0.38, "#10B981", "#D1FAE5", "Clade #42 (Conserved)\nShared Syntax\nΔQ = 0.02 (Low)", False),
        (0.62, "#0284C7", "#E0F2FE", "Clade #89 (Violated)\nRewired for Code\nΔQ = 0.38 (High)", True),
        (0.86, "#10B981", "#D1FAE5", "Clade #112 (Conserved)\nCore Knowledge\nΔQ = 0.01 (Low)", False),
    ]
    for cx, stroke, fill, label, is_hot in clade_specs:
        tri = Polygon([[cx, 0.43], [cx - 0.088, 0.26], [cx + 0.088, 0.26]],
                      closed=True, facecolor=fill, edgecolor=stroke,
                      lw=2.0 if is_hot else 1.4, zorder=3)
        ax.add_patch(tri)
        for dx in np.linspace(-0.068, 0.068, 5):
            ax.plot(cx + dx, 0.26, "o", color=stroke, markersize=3.2, zorder=4)
        ax.text(cx, 0.318, "~108\nneurons", ha="center", va="center",
                fontsize=6.9, fontweight="bold", color=stroke, zorder=4)
        ax.text(cx, 0.195, label, ha="center", va="top",
                fontsize=7.0, fontweight="bold" if is_hot else "normal",
                color=stroke if is_hot else "#334155")

    foot = FancyBboxPatch((0.03, 0.04), 0.94, 0.075,
                          boxstyle="round,pad=0.004,rounding_size=0.015",
                          facecolor="#EEF2FF", edgecolor="#6366F1", lw=1.1, zorder=2)
    ax.add_patch(foot)
    ax.text(0.50, 0.0775,
            "Phylo2Vec GPU Discrete Gradient SPR infers each 13,824-neuron tree in 1.8s (<90s total)",
            ha="center", va="center", fontsize=6.9, fontweight="bold", color="#312E81")


def draw_panel_c(ax):
    """Panel C: Why Tree Quartet Violations Beat Weight Change Magnitude."""
    # Left box: Uniform Subspace Drift (Fools ||dW||_F)
    left_card = FancyBboxPatch((0.03, 0.19), 0.445, 0.62,
                               boxstyle="round,pad=0.008,rounding_size=0.02",
                               facecolor="#FFFBEB", edgecolor="#F59E0B", lw=1.4, zorder=2)
    ax.add_patch(left_card)
    ax.text(0.252, 0.765, "1. Uniform Subspace Drift",
            ha="center", va="center", fontsize=8.3, fontweight="bold", color="#B45309")
    ax.text(0.252, 0.715, "(Fools ||ΔW||_F & TIES-Merging)",
            ha="center", va="center", fontsize=7.3, color="#92400E")

    # Base quartet
    ax.plot([0.075, 0.12], [0.59, 0.53], color="#64748B", lw=1.8)
    ax.plot([0.075, 0.12], [0.47, 0.53], color="#64748B", lw=1.8)
    ax.plot([0.12, 0.185], [0.53, 0.53], color="#64748B", lw=2.2)
    ax.plot([0.185, 0.23], [0.53, 0.59], color="#64748B", lw=1.8)
    ax.plot([0.185, 0.23], [0.53, 0.47], color="#64748B", lw=1.8)
    for px, py, nm in [(0.075, 0.59, "a"), (0.075, 0.47, "b"), (0.23, 0.59, "c"), (0.23, 0.47, "d")]:
        ax.plot(px, py, "o", color="#334155", markersize=5.0)
        ax.text(px, py + 0.032, nm, fontsize=7.6, fontweight="bold", ha="center", color="#0F172A")

    ax.annotate("", xy=(0.30, 0.53), xytext=(0.25, 0.53),
                arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.32", color="#D97706", lw=1.6))

    # Rotated quartet (same partners ab|cd!)
    ax.plot([0.32, 0.355], [0.59, 0.53], color="#D97706", lw=1.8)
    ax.plot([0.32, 0.355], [0.47, 0.53], color="#D97706", lw=1.8)
    ax.plot([0.355, 0.41], [0.53, 0.53], color="#D97706", lw=2.2)
    ax.plot([0.41, 0.445], [0.53, 0.59], color="#D97706", lw=1.8)
    ax.plot([0.41, 0.445], [0.53, 0.47], color="#D97706", lw=1.8)
    for px, py, nm in [(0.32, 0.59, "a'"), (0.32, 0.47, "b'"), (0.445, 0.59, "c'"), (0.445, 0.47, "d'")]:
        ax.plot(px, py, "o", color="#B45309", markersize=5.0)
        ax.text(px, py + 0.032, nm, fontsize=7.6, fontweight="bold", ha="center", color="#92400E")

    ax.text(0.252, 0.31,
            "• Weights rotated (large ||ΔW||_F)\n"
            "• Split {a,b}|{c,d} is UNCHANGED\n"
            "• Quartet Violation = 0 (Ignored!)",
            ha="center", va="center", fontsize=7.1, color="#78350F")

    # Right box: True Relational Rewiring (Quartet Topology Violation)
    right_card = FancyBboxPatch((0.525, 0.19), 0.445, 0.62,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor="#FEF2F2", edgecolor="#EF4444", lw=1.5, zorder=2)
    ax.add_patch(right_card)
    ax.text(0.748, 0.765, "2. True Circuit Rewiring",
            ha="center", va="center", fontsize=8.3, fontweight="bold", color="#B91C1C")
    ax.text(0.748, 0.715, "(Caught by Quartet Violation)",
            ha="center", va="center", fontsize=7.3, color="#991B1B")

    # Base quartet {a,b}|{c,d}
    ax.plot([0.565, 0.61], [0.59, 0.53], color="#64748B", lw=1.8)
    ax.plot([0.565, 0.61], [0.47, 0.53], color="#64748B", lw=1.8)
    ax.plot([0.61, 0.67], [0.53, 0.53], color="#64748B", lw=2.2)
    ax.plot([0.67, 0.715], [0.53, 0.59], color="#64748B", lw=1.8)
    ax.plot([0.67, 0.715], [0.53, 0.47], color="#64748B", lw=1.8)
    for px, py, nm in [(0.565, 0.59, "a"), (0.565, 0.47, "b"), (0.715, 0.59, "c"), (0.715, 0.47, "d")]:
        ax.plot(px, py, "o", color="#334155", markersize=5.0)
        ax.text(px, py + 0.032, nm, fontsize=7.6, fontweight="bold", ha="center", color="#0F172A")

    ax.annotate("", xy=(0.79, 0.53), xytext=(0.74, 0.53),
                arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.32", color="#DC2626", lw=1.8))

    # Rewired quartet {a,c}|{b,d} (Partner swap!)
    ax.plot([0.81, 0.85], [0.59, 0.53], color="#DC2626", lw=2.0)
    ax.plot([0.81, 0.85], [0.47, 0.53], color="#DC2626", lw=2.0)
    ax.plot([0.85, 0.905], [0.53, 0.53], color="#DC2626", lw=2.4)
    ax.plot([0.905, 0.945], [0.53, 0.59], color="#DC2626", lw=2.0)
    ax.plot([0.905, 0.945], [0.53, 0.47], color="#DC2626", lw=2.0)
    for px, py, nm in [(0.81, 0.59, "a"), (0.81, 0.47, "c"), (0.945, 0.59, "b"), (0.945, 0.47, "d")]:
        col_pt = "#DC2626" if nm in ("b", "c") else "#334155"
        ax.plot(px, py, "o", color=col_pt, markersize=5.5)
        ax.text(px, py + 0.032, nm, fontsize=7.8, fontweight="bold", ha="center", color=col_pt)

    ax.text(0.748, 0.31,
            "• Neurons b & c SWAPPED partners:\n"
            "  {a,b}|{c,d}  →  {a,c}|{b,d}!\n"
            "• Relational geometry broke (High ΔQ)",
            ha="center", va="center", fontsize=7.1, fontweight="bold", color="#991B1B")

    foot = FancyBboxPatch((0.03, 0.04), 0.94, 0.11,
                          boxstyle="round,pad=0.005,rounding_size=0.015",
                          facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1.0, zorder=2)
    ax.add_patch(foot)
    ax.text(0.50, 0.095,
            "1-Body Norms ask: 'Did neuron i move?'   vs.   Tree Surgery asks: 'Did this clade change its partners?'",
            ha="center", va="center", fontsize=6.7, fontweight="bold", color="#1E293B")


def draw_panel_d(fig, card_bbox):
    """Panel D: Bar chart of 20% Knock-Out across 8 methods."""
    x0, y0, w, h = card_bbox
    ax = fig.add_axes([x0 + 0.105, y0 + 0.045, w - 0.125, h - 0.14])

    methods = [
        "Conserved Clades",
        "Random 20%",
        "DARE",
        "Spherical k-Means",
        "Cosine Drift",
        "Frobenius ||ΔW||_F",
        "TIES-Merging",
        "Tree Violated (Ours)"
    ]
    dnll = [0.071, 0.498, 0.531, 1.482, 1.894, 2.011, 2.142, 3.084]
    pcts = [2.1, 14.7, 15.7, 43.8, 56.0, 59.5, 63.4, 91.2]
    colors = [
        "#10B981", "#94A3B8", "#94A3B8", "#38BDF8",
        "#60A5FA", "#818CF8", "#6366F1", "#DC2626"
    ]

    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, dnll, height=0.64, color=colors, edgecolor="none", zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=8.0, fontweight="bold", color="#1E293B")
    ax.set_xlim(0, 4.35)
    ax.set_xlabel("Reasoning Loss When Reverted to Base (ΔNLL ↑)",
                  fontsize=8.0, fontweight="bold", color="#334155")
    ax.grid(axis="x", linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=7.8)

    for bar, val, pct in zip(bars, dnll, pcts):
        is_top = (val > 3.0)
        is_bot = (val < 0.1)
        lbl = f"+{val:.3f} ({pct:.1f}%)"
        if is_top:
            lbl += " ★ #1"
        elif is_bot:
            lbl += " (43× lower!)"
        ax.text(
            val + 0.06, bar.get_y() + bar.get_height() / 2.0, lbl,
            va="center", ha="left",
            fontsize=7.6,
            fontweight="bold" if (is_top or is_bot) else "normal",
            color="#B91C1C" if is_top else ("#047857" if is_bot else "#334155")
        )


def draw_panel_e(fig, card_bbox):
    """Panel E: Ultra-sparse Top-K Clade Knock-Out curve (K=5..100 out of 6,171)."""
    x0, y0, w, h = card_bbox
    ax = fig.add_axes([x0 + 0.042, y0 + 0.05, w - 0.062, h - 0.145])

    k_vals = np.array([5, 10, 25, 50, 100])
    wt_pct = np.array([0.16, 0.31, 0.74, 1.47, 2.89])
    tree_pct = np.array([4.2, 7.7, 16.0, 27.3, 42.6])
    frob_pct = np.array([2.0, 3.7, 8.3, 14.6, 24.9])
    rand_pct = wt_pct.copy()

    ax.plot(k_vals, tree_pct, "-o", color="#DC2626", lw=2.5, markersize=6.0,
            label="Tree Quartet-Violated Clades (Ours)", zorder=4)
    ax.plot(k_vals, frob_pct, "--s", color="#6366F1", lw=2.0, markersize=5.2,
            label="3-Matrix Frobenius Norm (||ΔW||_F)", zorder=3)
    ax.plot(k_vals, rand_pct, ":^", color="#94A3B8", lw=1.8, markersize=4.8,
            label="Random / Uniform Expectation", zorder=2)

    ax.fill_between(k_vals, frob_pct, tree_pct, color="#FEE2E2", alpha=0.45, zorder=1)

    ax.annotate(
        "Top 50 / 6,171 Clades\n(only 1.47% of MLP weights)\n→ 27.3% of reasoning drop!\n(~19× causal density)",
        xy=(50, 27.3), xytext=(28, 34.5),
        fontsize=7.5, fontweight="bold", color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF2F2", edgecolor="#EF4444", lw=1.2),
        arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.4", color="#DC2626", lw=1.5),
        zorder=5
    )

    ax.set_xlabel("Top K Clades Reverted (Out of 6,171 Total Clades)",
                  fontsize=8.0, fontweight="bold", color="#334155")
    ax.set_ylabel("% of Reasoning Gain Destroyed",
                  fontsize=8.0, fontweight="bold", color="#334155")
    ax.set_xlim(0, 108)
    ax.set_ylim(0, 48)
    ax.set_xticks([5, 15, 25, 50, 100])
    ax.set_xticklabels(["5\n(0.16%)", "10\n(0.31%)", "25\n(0.74%)",
                        "50\n(1.47%)", "100\n(2.89%)"], fontsize=7.3)
    # Keep actual data plotted at x=[5,10,25,50,100], place ticks cleanly at [5, 12, 25, 50, 100]
    ax.set_xticks([4, 12, 25, 50, 100])
    ax.set_xticklabels(["K=5", "K=10", "K=25\n(0.74% wt)",
                        "K=50\n(1.47% wt)", "K=100\n(2.89% wt)"], fontsize=7.3)
    ax.tick_params(axis="y", labelsize=7.8)
    ax.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=7.3, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")


def draw_panel_f(fig, card_bbox):
    """Panel F: Zero-Training Phylogenetic Chimera (2D Math vs Code Pareto Plot)."""
    x0, y0, w, h = card_bbox
    ax = fig.add_axes([x0 + 0.042, y0 + 0.05, w - 0.062, h - 0.145])

    # Shaded Pareto win zone
    win_box = FancyBboxPatch((2.12, 68.0), 0.28, 30.0,
                             boxstyle="round,pad=0.01,rounding_size=0.02",
                             facecolor="#ECFDF5", edgecolor="#10B981", lw=1.2, ls="--", zorder=1)
    ax.add_patch(win_box)
    ax.text(2.39, 95.5, "Pareto Win Zone",
            ha="left", va="top", fontsize=7.3, fontweight="bold", color="#047857", zorder=2)

    # Arrow from 50/50 Weight Average to Phylogenetic Chimera
    ax.annotate("", xy=(2.345, 79.5), xytext=(2.408, 45.0),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.45",
                                color="#059669", lw=2.0), zorder=3)
    ax.text(2.415, 62.0, "+39.5% Math &\nBetter Code vs.\n50/50 Merge",
            fontsize=7.2, fontweight="bold", color="#047857", ha="right", va="center")

    # Plot points and custom non-overlapping labels
    pts = [
        ("Qwen2.5-14B (Base)", 2.74, 0.0, "#64748B", "o", 85, (0, 10), "center"),
        ("Coder-14B\n(Code Specialist)", 2.18, 29.0, "#0284C7", "s", 95, (-8, 8), "right"),
        ("DeepSeek-R1-14B\n(Math Specialist)", 2.68, 100.0, "#7C3AED", "D", 95, (0, -24), "center"),
        ("50/50 Weight Average\n(Nonlinear Collapse: 42.5%)", 2.41, 42.5, "#F59E0B", "^", 105, (-10, -6), "right"),
        ("Phylogenetic Chimera (Ours)\n82.0% Math + 2.34 Code PPL\n(Zero Training!)", 2.34, 82.0, "#10B981", "*", 250, (8, -22), "left"),
    ]

    for lbl, c_ppl, m_rec, col, mk, sz, (dx, dy), ha_align in pts:
        ax.scatter([c_ppl], [m_rec], color=col, marker=mk, s=sz, edgecolors="#0F172A", linewidths=0.9, zorder=5)
        is_chimera = ("Chimera" in lbl)
        ax.annotate(
            lbl, xy=(c_ppl, m_rec), xytext=(dx, dy),
            textcoords="offset points", ha=ha_align,
            fontsize=7.4 if is_chimera else 7.2,
            fontweight="bold" if is_chimera else "normal",
            color="#065F46" if is_chimera else "#1E293B",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#D1FAE5",
                      edgecolor="#10B981", lw=1.2) if is_chimera else None,
            zorder=6
        )

    ax.set_xlim(2.83, 2.09)  # Inverted so right = lower/better Code PPL
    ax.set_ylim(-8, 113)
    ax.set_xlabel("Code Benchmark Perplexity (Lower / Right is Better →)",
                  fontsize=8.0, fontweight="bold", color="#334155")
    ax.set_ylabel("Math & Reasoning Skill Recovered (%)",
                  fontsize=8.0, fontweight="bold", color="#334155")
    ax.tick_params(axis="both", labelsize=7.8)
    ax.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig = plt.figure(figsize=(18, 11.2), dpi=300, facecolor="#F1F5F9")

    # Top Title Banner
    title_ax = fig.add_axes([0.015, 0.925, 0.97, 0.062], frameon=False)
    title_ax.set_xlim(0, 1)
    title_ax.set_ylim(0, 1)
    title_ax.axis("off")
    banner = FancyBboxPatch((0.0, 0.05), 1.0, 0.90,
                            boxstyle="round,pad=0.005,rounding_size=0.015",
                            facecolor="#0F172A", edgecolor="#334155", lw=1.5)
    title_ax.add_patch(banner)
    title_ax.text(
        0.02, 0.60,
        "Phylogenetic Tree Surgery on 14B LLMs (Qwen2.5-14B, Coder-14B, DeepSeek-R1-14B)",
        color="#FFFFFF", fontsize=14.5, fontweight="bold", ha="left", va="center"
    )
    title_ax.text(
        0.02, 0.24,
        "From 663,552 MLP Neurons → 48 GPU BME Layer Trees (6,171 Clades) → Quartet Topology Violations → Surgical Knock-Out & Zero-Training Chimera",
        color="#94A3B8", fontsize=9.5, ha="left", va="center"
    )

    # Row 1: Pipeline Steps (Panels A, B, C)
    ax_a = add_panel_card(fig, [0.015, 0.48, 0.315, 0.435],
                          "STEP 1 • GENOMES", "Sibling Models & Signatures", "#4F46E5")
    draw_panel_a(ax_a)

    ax_b = add_panel_card(fig, [0.3425, 0.48, 0.315, 0.435],
                          "STEP 2 • TREES", "GPU BME Trees (6,171 Clades)", "#0284C7")
    draw_panel_b(ax_b)

    ax_c = add_panel_card(fig, [0.67, 0.48, 0.315, 0.435],
                          "STEP 3 • QUARTETS", "Partner Swaps vs. Weight Norms", "#DC2626")
    draw_panel_c(ax_c)

    # Row 2: Experimental Results (Panels D, E, F)
    add_panel_card(fig, [0.015, 0.02, 0.315, 0.44],
                   "RESULT 1 • ABLATION", "Beats 7 Baselines (20% Budget)", "#DC2626")
    draw_panel_d(fig, [0.015, 0.02, 0.315, 0.44])

    add_panel_card(fig, [0.3425, 0.02, 0.315, 0.44],
                   "RESULT 2 • SPARSITY", "Top 50 Clades = 27.3% of Effect", "#7C3AED")
    draw_panel_e(fig, [0.3425, 0.02, 0.315, 0.44])

    add_panel_card(fig, [0.67, 0.02, 0.315, 0.44],
                   "RESULT 3 • CHIMERA", "Zero-Training Specialist Grafting", "#059669")
    draw_panel_f(fig, [0.67, 0.02, 0.315, 0.44])

    png_path = os.path.join(out_dir, "qwen_tree_surgery_overview.png")
    pdf_path = os.path.join(out_dir, "qwen_tree_surgery_overview.pdf")
    svg_path = os.path.join(out_dir, "qwen_tree_surgery_overview.svg")

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved:\n  {png_path}\n  {pdf_path}\n  {svg_path}")


if __name__ == "__main__":
    main()

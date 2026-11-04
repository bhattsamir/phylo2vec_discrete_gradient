#!/usr/bin/env python3
"""
generate_qwen_method_diagram.py

Generates a crystal-clear 4-panel architectural diagram answering:
  1. PANEL 1: "Where are the 14B parameters, and what is 1 Unit (1 Taxon)?"
              Shows 48 layers -> 1 SwiGLU MLP layer -> W_gate (13,824 x 5,120),
              W_up (13,824 x 5,120), W_down (5,120 x 13,824) ->
              1 Taxon i = Neuron Unit #i = Row i of W_gate + Row i of W_up + Col i of W_down
              (15,360 floats per taxon x 13,824 taxa x 48 layers = 10.2B MLP weights!).
  2. PANEL 2: "What is the Phylogenetic Tree, and what is a Clade?"
              Shows how the 13,824 units in Layer l are the 13,824 LEAVES (taxa) of the tree,
              connected by cosine similarity of their 15,360-dim weight vectors, and how a
              CLADE (~108 leaf units) is a block of 108 rows of W_gate/W_up & 108 cols of W_down.
  3. PANEL 3: "How do we compare Two Models (Base Tree vs. Specialist Tree)?"
              Same 13,824 leaf IDs in both models! Compares Base Tree T_base vs Specialist Tree T_ft:
              Uniform SGD rotation preserves quartet splits {i,j}|{k,m}, whereas real functional
              rewiring swaps partners {i,j}|{k,m} -> {i,k}|{j,m} (Quartet Violation).
  4. PANEL 4: "How does Tree Surgery happen on W_gate, W_up, W_down?"
              Shows exact row/column index slicing for Clade C* ({i_1, ..., i_108}):
              Mode 1 (Knock-Out): Revert rows/cols C* in Specialist back to Base.
              Mode 2 (Chimera Grafting): Copy Math Clade rows/cols from R1 + Code Clade rows/cols from Coder into Base.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


def add_stage_card(fig, bbox, step_badge, title_text, subtitle_text, badge_color="#4F46E5"):
    x0, y0, w, h = bbox
    ax = fig.add_axes([x0, y0, w, h], frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    card = FancyBboxPatch(
        (0.006, 0.010), 0.988, 0.980,
        boxstyle="round,pad=0.008,rounding_size=0.022",
        facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5, zorder=1
    )
    ax.add_patch(card)

    pill = FancyBboxPatch(
        (0.022, 0.892), 0.215, 0.072,
        boxstyle="round,pad=0.004,rounding_size=0.014",
        facecolor=badge_color, edgecolor="none", zorder=2
    )
    ax.add_patch(pill)
    ax.text(
        0.1295, 0.928, step_badge,
        color="#FFFFFF", fontsize=8.8, fontweight="bold",
        ha="center", va="center", zorder=3
    )
    ax.text(
        0.252, 0.938, title_text,
        color="#0F172A", fontsize=11.0, fontweight="bold",
        ha="left", va="center", zorder=3
    )
    ax.text(
        0.252, 0.898, subtitle_text,
        color="#475569", fontsize=8.1,
        ha="left", va="center", zorder=3
    )
    ax.plot([0.022, 0.978], [0.865, 0.865], color="#E2E8F0", lw=1.1, zorder=2)
    return ax


def draw_matrix_with_highlight(ax, x0, y0, w, h, title_lbl, dim_lbl, mode="row", hi_colors=None):
    """
    Draw a matrix with explicit rows or columns highlighted to show what a single Taxon i
    or Clade C looks like in W_gate (rows), W_up (rows), and W_down (columns).
    """
    outer = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.002,rounding_size=0.006",
        facecolor="#F8FAFC", edgecolor="#334155", lw=1.4, zorder=3
    )
    ax.add_patch(outer)

    if hi_colors is None:
        hi_colors = ["#CBD5E1", "#CBD5E1", "#EF4444", "#CBD5E1", "#CBD5E1", "#CBD5E1", "#CBD5E1"]

    n = len(hi_colors)
    pad = 0.005
    if mode == "row":
        sh = (h - 2 * pad) / n
        for idx, col in enumerate(reversed(hi_colors)):
            r = Rectangle(
                (x0 + pad, y0 + pad + idx * sh), w - 2 * pad, sh * 0.82,
                facecolor=col, edgecolor="none", alpha=0.88, zorder=4
            )
            ax.add_patch(r)
    else:
        sw = (w - 2 * pad) / n
        for idx, col in enumerate(hi_colors):
            r = Rectangle(
                (x0 + pad + idx * sw, y0 + pad), sw * 0.82, h - 2 * pad,
                facecolor=col, edgecolor="none", alpha=0.88, zorder=4
            )
            ax.add_patch(r)

    ax.text(x0 + w / 2.0, y0 + h + 0.036, title_lbl,
            ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#0F172A", zorder=5)
    ax.text(x0 + w / 2.0, y0 + h + 0.008, dim_lbl,
            ha="center", va="bottom", fontsize=6.7, color="#475569", zorder=5)


def draw_panel_1(ax):
    """
    PANEL 1: Where are the 14B parameters, and what is ONE Taxon (Unit i)?
    Shows the exact parameter accounting:
      14.7B total params = 48 layers x (212M MLP params + 94M Attention params)
      Inside 1 MLP layer: N = 13,824 hidden units (taxa), d = 5,120 residual dim.
      Taxon i = Row i of W_gate (5,120) + Row i of W_up (5,120) + Col i of W_down (5,120) = 15,360 floats!
    """
    # Left box: Parameter Accounting (14.7B Model -> 48 Layers -> 1 MLP Block)
    acct = FancyBboxPatch((0.025, 0.21), 0.285, 0.62,
                          boxstyle="round,pad=0.006,rounding_size=0.016",
                          facecolor="#F8FAFC", edgecolor="#64748B", lw=1.3, zorder=2)
    ax.add_patch(acct)
    ax.text(0.1675, 0.795, "'14B' = 14.7B Scalar Weights\n(NOT 14B Neurons!)",
            ha="center", va="top", fontsize=8.3, fontweight="bold", color="#0F172A")

    layers_info = [
        (0.60, "48 Transformer Layers", "Total = 14.7B scalar floats", "#E2E8F0", "#475569"),
        (0.43, "1 MLP Block (Layer l)", "212.3M floats per layer\n(× 48 layers = 10.2B MLP floats)", "#DBEAFE", "#2563EB"),
        (0.24, "13,824 MLP Units / Layer", "Each Unit i = 15,360 floats\n(13,824 × 15,360 = 212.3M)", "#FEE2E2", "#DC2626"),
    ]
    for ly, t1, t2, fc, ec in layers_info:
        b = FancyBboxPatch((0.042, ly), 0.25, 0.12,
                           boxstyle="round,pad=0.004,rounding_size=0.012",
                           facecolor=fc, edgecolor=ec, lw=1.4, zorder=3)
        ax.add_patch(b)
        ax.text(0.167, ly + 0.082, t1, ha="center", va="center",
                fontsize=7.6, fontweight="bold", color=ec, zorder=4)
        ax.text(0.167, ly + 0.034, t2, ha="center", va="center",
                fontsize=6.6, color="#1E293B", zorder=4)

    ax.annotate("", xy=(0.167, 0.555), xytext=(0.167, 0.598),
                arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.3", color="#475569", lw=1.4))
    ax.annotate("", xy=(0.167, 0.365), xytext=(0.167, 0.428),
                arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.3", color="#2563EB", lw=1.4))

    # Right box: Inside Layer l's SwiGLU MLP — What is Taxon i (Unit i)?
    mlp_box = FancyBboxPatch((0.335, 0.21), 0.64, 0.62,
                             boxstyle="round,pad=0.008,rounding_size=0.016",
                             facecolor="#F8FAFC", edgecolor="#6366F1", lw=1.4, zorder=2)
    ax.add_patch(mlp_box)
    ax.text(0.655, 0.795, "Anatomy of 1 Taxon (Unit #i) Inside Layer l's SwiGLU MLP",
            ha="center", va="top", fontsize=8.4, fontweight="bold", color="#1E1B4B")
    ax.text(0.655, 0.742, "Unit i computes:  a_i = SiLU(Row_i(W_gate)·x) × (Row_i(W_up)·x)   →   adds a_i · Col_i(W_down)",
            ha="center", va="center", fontsize=6.3, fontweight="bold", color="#4338CA")

    # 3 Matrices: W_gate (Row i in red), W_up (Row i in red), W_down (Col i in red)
    stripes = ["#CBD5E1", "#CBD5E1", "#DC2626", "#CBD5E1", "#CBD5E1", "#CBD5E1", "#CBD5E1"]
    draw_matrix_with_highlight(
        ax, 0.365, 0.41, 0.16, 0.24,
        "1. W_gate", "13,824 rows × 5,120 cols", mode="row", hi_colors=stripes
    )
    draw_matrix_with_highlight(
        ax, 0.57, 0.41, 0.16, 0.24,
        "2. W_up", "13,824 rows × 5,120 cols", mode="row", hi_colors=stripes
    )
    draw_matrix_with_highlight(
        ax, 0.775, 0.41, 0.165, 0.24,
        "3. W_down", "5,120 rows × 13,824 cols", mode="col", hi_colors=stripes
    )

    # Labels right under each matrix explaining the red stripe
    ax.text(0.445, 0.382, "Red = Row i (1×5,120)\n(Gate: When to fire)", ha="center", va="top",
            fontsize=6.6, fontweight="bold", color="#B91C1C")
    ax.text(0.65, 0.382, "Red = Row i (1×5,120)\n(Up: Feature value)", ha="center", va="top",
            fontsize=6.6, fontweight="bold", color="#B91C1C")
    ax.text(0.857, 0.382, "Red = Col i (5,120×1)\n(Down: Output vector)", ha="center", va="top",
            fontsize=6.6, fontweight="bold", color="#B91C1C")

    # Concatenate into Taxon i's 15,360-dim signature vector
    taxon_pill = FancyBboxPatch((0.355, 0.228), 0.60, 0.082,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor="#FEF2F2", edgecolor="#DC2626", lw=1.4, zorder=4)
    ax.add_patch(taxon_pill)
    ax.text(0.655, 0.269,
            "TAXON #i (1 Leaf) = [ Row_i(W_gate) | Row_i(W_up) | Col_i(W_down)^T ] (15,360 floats)",
            ha="center", va="center", fontsize=6.4, fontweight="bold", color="#991B1B", zorder=5)

    # Bottom summary bar
    foot = FancyBboxPatch((0.025, 0.035), 0.95, 0.13,
                          boxstyle="round,pad=0.005,rounding_size=0.014",
                          facecolor="#EEF2FF", edgecolor="#6366F1", lw=1.1, zorder=2)
    ax.add_patch(foot)
    ax.text(0.50, 0.10,
            "Summary of Step 1:  We build ONE tree per Transformer layer (48 trees total).  In Layer l's tree, there are N = 13,824 taxa (leaves).\n"
            "Each taxon i (i = 1..13,824) is ONE MLP Hidden Unit — consisting of Row i of W_gate, Row i of W_up, and Column i of W_down (15,360 weights).",
            ha="center", va="center", fontsize=6.6, fontweight="bold", color="#1E1B4B")


def draw_panel_2(ax):
    """
    PANEL 2: What is the Tree, and what is a Clade in terms of Rows & Columns?
    Shows:
      - Pairwise cosine distance between Taxon i's 15,360-float vector and Taxon j's 15,360-float vector
      - 13,824 x 13,824 distance matrix -> Binary Tree with 13,824 leaves
      - A Clade (subtree of ~108 leaves, e.g. Units {12, 89, 401, ...}) = a set of 108 row indices in W_gate/W_up and 108 column indices in W_down!
    """
    ax.text(0.50, 0.828, "In Layer l: 13,824 Taxa (MLP Units) → 13,824×13,824 Distance Matrix → Binary Tree with 13,824 Leaves",
            ha="center", va="top", fontsize=8.3, fontweight="bold", color="#0F172A")

    # Left side: Binary tree with 4 colored clades
    edges = [
        ((0.29, 0.72), (0.15, 0.62)),
        ((0.29, 0.72), (0.43, 0.62)),
        ((0.15, 0.62), (0.085, 0.50)),
        ((0.15, 0.62), (0.215, 0.50)),
        ((0.43, 0.62), (0.355, 0.50)),
        ((0.43, 0.62), (0.495, 0.50)),
    ]
    for (x1, y1), (x2, y2) in edges:
        ax.plot([x1, x2], [y1, y1], color="#475569", lw=2.0, zorder=2)
        ax.plot([x2, x2], [y1, y2], color="#475569", lw=2.0, zorder=2)

    ax.plot(0.29, 0.72, "o", color="#0F172A", markersize=6.0, zorder=4)
    ax.text(0.29, 0.745, "Layer l Tree Root (All 13,824 MLP Units)",
            ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#0F172A")

    clades = [
        (0.085, "#7C3AED", "#F3E8FF", "Clade A\n(~108 units)", ["Unit 12", "Unit 89"]),
        (0.215, "#10B981", "#D1FAE5", "Clade B\n(~108 units)", ["Unit 5", "Unit 77"]),
        (0.355, "#0284C7", "#E0F2FE", "Clade C\n(~108 units)", ["Unit 33", "Unit 94"]),
        (0.495, "#64748B", "#F1F5F9", "Clade D\n(~108 units)", ["Unit 8", "Unit 61"]),
    ]
    for cx, col, fill, cl_lbl, leaves in clades:
        ax.plot(cx, 0.50, "o", color=col, markersize=6.2, zorder=5)
        tri = Polygon([[cx, 0.50], [cx - 0.056, 0.31], [cx + 0.056, 0.31]],
                      closed=True, facecolor=fill, edgecolor=col, lw=1.8, zorder=3)
        ax.add_patch(tri)
        ax.text(cx, 0.365, cl_lbl, ha="center", va="center",
                fontsize=6.3, fontweight="bold", color=col, zorder=4)
        for dx, l_nm in zip([-0.032, 0.032], leaves):
            lx = cx + dx
            ax.plot([lx, lx], [0.31, 0.275], color=col, lw=1.5, zorder=3)
            ax.plot(lx, 0.275, "s", color=col, markersize=4.0, zorder=4)
            ax.text(lx, 0.258, l_nm, ha="center", va="top", fontsize=6.0,
                    fontweight="bold", color="#334155")

    ax.text(0.29, 0.20, "1 Leaf = 1 MLP Unit (15,360 floats)  |  1 Clade = Subtree of ~108 Related Units",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color="#334155")

    # Arrow from Clade A (purple) to the matrix rows/cols on the right
    ax.annotate("", xy=(0.632, 0.44), xytext=(0.562, 0.44),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.42",
                                color="#7C3AED", lw=2.0))
    ax.text(0.597, 0.485, "Clade A\nIndex Set", ha="center", va="bottom",
            fontsize=6.5, fontweight="bold", color="#7C3AED")

    # Right box: What Clade A looks like inside W_gate, W_up, W_down
    rbox = FancyBboxPatch((0.64, 0.19), 0.335, 0.59,
                          boxstyle="round,pad=0.008,rounding_size=0.016",
                          facecolor="#FAF5FF", edgecolor="#7C3AED", lw=1.4, zorder=2)
    ax.add_patch(rbox)
    ax.text(0.8075, 0.742, "What 1 Clade (e.g. Clade A) Is in the Matrices",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#5B21B6")
    ax.text(0.8075, 0.695, "Clade A = 108 Unit IDs, e.g. C_A = {12, 89, 401, ...}",
            ha="center", va="center", fontsize=7.1, fontweight="bold", color="#6D28D9")

    c_stripes = ["#CBD5E1", "#7C3AED", "#CBD5E1", "#7C3AED", "#CBD5E1", "#7C3AED", "#CBD5E1"]
    draw_matrix_with_highlight(
        ax, 0.665, 0.39, 0.13, 0.22,
        "W_gate & W_up", "108 Purple Rows", mode="row", hi_colors=c_stripes
    )
    draw_matrix_with_highlight(
        ax, 0.82, 0.39, 0.13, 0.22,
        "W_down", "108 Purple Cols", mode="col", hi_colors=c_stripes
    )

    ax.text(0.8075, 0.275,
            "• A Clade is a matched set of ~108 Rows in W_gate & W_up\n"
            "  plus the matching ~108 Columns in W_down (~1.66M floats).\n"
            "• Together, those 108 units form a self-contained sub-MLP\n"
            "  circuit that detects a pattern and writes to the residual stream!",
            ha="center", va="center", fontsize=6.6, color="#3B0764")

    # Bottom summary bar
    foot = FancyBboxPatch((0.025, 0.035), 0.95, 0.13,
                          boxstyle="round,pad=0.005,rounding_size=0.014",
                          facecolor="#F0F9FF", edgecolor="#0284C7", lw=1.1, zorder=2)
    ax.add_patch(foot)
    ax.text(0.50, 0.10,
            "Summary of Step 2:  We compute the 13,824 × 13,824 cosine distance matrix between the 13,824 units in Layer l and infer a binary tree.\n"
            "Cutting the tree at ~128 subtrees partitions the 13,824 units into ~128 Clades (~108 units/clade × 48 layers = 6,171 clades across the 14B model).",
            ha="center", va="center", fontsize=6.5, fontweight="bold", color="#0C4A6E")


def draw_panel_3(ax):
    """
    PANEL 3: How Do We Compare Two Models (Base Tree vs. Specialist Tree on the SAME 13,824 Taxa)?
    """
    # Left half: Model A Tree vs Model B Tree when a clade only rotates/scales (No Rewiring)
    c1_box = FancyBboxPatch((0.025, 0.19), 0.455, 0.635,
                            boxstyle="round,pad=0.008,rounding_size=0.018",
                            facecolor="#FFFBEB", edgecolor="#F59E0B", lw=1.4, zorder=2)
    ax.add_patch(c1_box)
    ax.text(0.2525, 0.785, "Case A: Clade Drifts / Rotates as a Rigid Block",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="#B45309")
    ax.text(0.2525, 0.738, "Base Tree vs. Specialist Tree Have the EXACT Same Quartet Partners",
            ha="center", va="center", fontsize=7.1, color="#92400E")

    # Base tree on 4 units: (Unit 12, Unit 89) | (Unit 401, Unit 550)
    ax.text(0.135, 0.67, "Tree in Model A (Base)", ha="center", va="center",
            fontsize=7.3, fontweight="bold", color="#475569")
    ax.plot([0.07, 0.108], [0.60, 0.53], color="#475569", lw=2.0)
    ax.plot([0.07, 0.108], [0.46, 0.53], color="#475569", lw=2.0)
    ax.plot([0.108, 0.162], [0.53, 0.53], color="#475569", lw=2.5)
    ax.plot([0.162, 0.20], [0.53, 0.60], color="#475569", lw=2.0)
    ax.plot([0.162, 0.20], [0.53, 0.46], color="#475569", lw=2.0)
    for px, py, lbl in [(0.07, 0.60, "Unit 12"), (0.07, 0.46, "Unit 89"), (0.20, 0.60, "Unit 401"), (0.20, 0.46, "Unit 550")]:
        ax.plot(px, py, "o", color="#334155", markersize=5.2)
        dy_lbl = 0.026 if py > 0.53 else -0.038
        ax.text(px, py + dy_lbl, lbl, ha="center", fontsize=6.7, fontweight="bold", color="#0F172A")

    ax.annotate("", xy=(0.285, 0.53), xytext=(0.23, 0.53),
                arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.34", color="#D97706", lw=1.8))
    ax.text(0.257, 0.562, "Fine-\nTuning", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#B45309")

    # Model B tree: still (Unit 12, Unit 89) | (Unit 401, Unit 550)
    ax.text(0.375, 0.67, "Tree in Model B (Specialist)", ha="center", va="center",
            fontsize=7.3, fontweight="bold", color="#B45309")
    ax.plot([0.31, 0.348], [0.60, 0.53], color="#D97706", lw=2.0)
    ax.plot([0.31, 0.348], [0.46, 0.53], color="#D97706", lw=2.0)
    ax.plot([0.348, 0.402], [0.53, 0.53], color="#D97706", lw=2.5)
    ax.plot([0.402, 0.44], [0.53, 0.60], color="#D97706", lw=2.0)
    ax.plot([0.402, 0.44], [0.53, 0.46], color="#D97706", lw=2.0)
    for px, py, lbl in [(0.31, 0.60, "Unit 12"), (0.31, 0.46, "Unit 89"), (0.44, 0.60, "Unit 401"), (0.44, 0.46, "Unit 550")]:
        ax.plot(px, py, "o", color="#B45309", markersize=5.2)
        dy_lbl = 0.026 if py > 0.53 else -0.038
        ax.text(px, py + dy_lbl, lbl, ha="center", fontsize=6.7, fontweight="bold", color="#92400E")

    ax.text(0.2525, 0.285,
            "• All 4 units changed their 15,360 floats (||W_B − W_A|| is large),\n"
            "  which fools standard weight-norm & TIES methods.\n"
            "• But Unit 12 is still paired with Unit 89 in both trees!\n"
            "• Quartet Violation = 0  →  No internal circuit rewiring.",
            ha="center", va="center", fontsize=6.9, color="#78350F")

    # Right half: Case B — True Relational Rewiring (Units swap tree partners between Model A and Model B!)
    c2_box = FancyBboxPatch((0.52, 0.19), 0.455, 0.635,
                            boxstyle="round,pad=0.008,rounding_size=0.018",
                            facecolor="#FEF2F2", edgecolor="#EF4444", lw=1.5, zorder=2)
    ax.add_patch(c2_box)
    ax.text(0.7475, 0.785, "Case B: Units Swap Partners Between Models!",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="#B91C1C")
    ax.text(0.7475, 0.738, "Quartet Violation Detects Clades That Rewired Their Function",
            ha="center", va="center", fontsize=7.1, color="#991B1B")

    # Base tree: (Unit 12, Unit 89) | (Unit 401, Unit 550)
    ax.text(0.63, 0.67, "Tree in Model A (Base)", ha="center", va="center",
            fontsize=7.3, fontweight="bold", color="#475569")
    ax.plot([0.565, 0.603], [0.60, 0.53], color="#475569", lw=2.0)
    ax.plot([0.565, 0.603], [0.46, 0.53], color="#475569", lw=2.0)
    ax.plot([0.603, 0.657], [0.53, 0.53], color="#475569", lw=2.5)
    ax.plot([0.657, 0.695], [0.53, 0.60], color="#475569", lw=2.0)
    ax.plot([0.657, 0.695], [0.53, 0.46], color="#475569", lw=2.0)
    for px, py, lbl in [(0.565, 0.60, "Unit 12"), (0.565, 0.46, "Unit 89"), (0.695, 0.60, "Unit 401"), (0.695, 0.46, "Unit 550")]:
        ax.plot(px, py, "o", color="#334155", markersize=5.2)
        dy_lbl = 0.026 if py > 0.53 else -0.038
        ax.text(px, py + dy_lbl, lbl, ha="center", fontsize=6.7, fontweight="bold", color="#0F172A")

    ax.annotate("", xy=(0.78, 0.53), xytext=(0.725, 0.53),
                arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.34", color="#DC2626", lw=1.8))
    ax.text(0.752, 0.562, "Partner\nSwap!", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#DC2626")

    # Specialist tree: (Unit 12, Unit 401) | (Unit 89, Unit 550)
    ax.text(0.865, 0.67, "Tree in Model B (Specialist)", ha="center", va="center",
            fontsize=7.3, fontweight="bold", color="#B91C1C")
    ax.plot([0.805, 0.843], [0.60, 0.53], color="#DC2626", lw=2.0)
    ax.plot([0.805, 0.843], [0.46, 0.53], color="#DC2626", lw=2.0)
    ax.plot([0.843, 0.897], [0.53, 0.53], color="#DC2626", lw=2.5)
    ax.plot([0.897, 0.935], [0.53, 0.60], color="#DC2626", lw=2.0)
    ax.plot([0.897, 0.935], [0.53, 0.46], color="#DC2626", lw=2.0)
    for px, py, lbl in [(0.805, 0.60, "Unit 12"), (0.805, 0.46, "Unit 401"), (0.935, 0.60, "Unit 89"), (0.935, 0.46, "Unit 550")]:
        is_sw = lbl in ("Unit 89", "Unit 401")
        c_pt = "#DC2626" if is_sw else "#334155"
        ax.plot(px, py, "o", color=c_pt, markersize=5.8 if is_sw else 5.2)
        dy_lbl = 0.026 if py > 0.53 else -0.038
        ax.text(px, py + dy_lbl, lbl, ha="center", fontsize=6.8 if is_sw else 6.7,
                fontweight="bold", color=c_pt)

    ax.text(0.7475, 0.285,
            "• In Model A (Base), Unit 12 was partnered with Unit 89.\n"
            "• In Model B (Specialist), Unit 12 broke away to pair with Unit 401!\n"
            "  ({12, 89}|{401, 550}  →  {12, 401}|{89, 550})\n"
            "• High Quartet Violation Score  →  This clade rewired its circuit!",
            ha="center", va="center", fontsize=6.9, fontweight="bold", color="#991B1B")

    # Bottom summary bar
    foot = FancyBboxPatch((0.025, 0.035), 0.95, 0.13,
                          boxstyle="round,pad=0.005,rounding_size=0.014",
                          facecolor="#FEF2F2", edgecolor="#EF4444", lw=1.1, zorder=2)
    ax.add_patch(foot)
    ax.text(0.50, 0.10,
            "Summary of Step 3:  Because Unit #12 in Model A and Unit #12 in Model B have the exact same leaf ID, we compare Layer l's tree in Model A\n"
            "against Layer l's tree in Model B.  Clades whose units swapped tree partners (Quartet Violations) are the sub-circuits that rewired!",
            ha="center", va="center", fontsize=6.5, fontweight="bold", color="#7F1D1D")


def draw_panel_4(ax):
    """
    PANEL 4: How Does Tree Surgery Happen on W_gate, W_up, W_down?
    Shows the concrete row/column copy operation for a clade index set C* = {12, 89, 401, ...}.
    """
    # Left half: Mode 1 — Clade Knock-Out (Revert Clade C*'s rows/cols back to Base)
    s1_box = FancyBboxPatch((0.025, 0.19), 0.455, 0.635,
                            boxstyle="round,pad=0.008,rounding_size=0.018",
                            facecolor="#F8FAFC", edgecolor="#6366F1", lw=1.4, zorder=2)
    ax.add_patch(s1_box)
    ax.text(0.2525, 0.785, "Surgery 1: Clade Knock-Out (Causal Ablation)",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="#312E81")
    ax.text(0.2525, 0.738, "Overwrite ONLY Clade C*'s 108 Rows/Cols with Base Weights",
            ha="center", va="center", fontsize=7.1, color="#4338CA")

    r1_rows = ["#7C3AED", "#EF4444", "#7C3AED", "#EF4444", "#7C3AED", "#7C3AED"]
    draw_matrix_with_highlight(
        ax, 0.045, 0.44, 0.115, 0.21,
        "Model B (R1)", "Rewired C* (Red)", mode="row", hi_colors=r1_rows
    )
    ax.text(0.18, 0.545, "←", ha="center", va="center", fontsize=13.0, fontweight="bold", color="#DC2626")

    base_rows = ["#E2E8F0", "#64748B", "#E2E8F0", "#64748B", "#E2E8F0", "#E2E8F0"]
    draw_matrix_with_highlight(
        ax, 0.20, 0.44, 0.115, 0.21,
        "Model A (Base)", "Base Rows C*", mode="row", hi_colors=base_rows
    )
    ax.text(0.335, 0.545, "=", ha="center", va="center", fontsize=13.0, fontweight="bold", color="#334155")

    ko_rows = ["#7C3AED", "#64748B", "#7C3AED", "#64748B", "#7C3AED", "#7C3AED"]
    draw_matrix_with_highlight(
        ax, 0.35, 0.44, 0.115, 0.21,
        "Surgically Edited", "Only C* Reverted", mode="row", hi_colors=ko_rows
    )

    ax.text(0.2525, 0.295,
            "Exact Matrix Operation for a Rewired Clade C* = {12, 89, ...}:\n"
            "  W_gate[C*, :] ← W_gate_base[C*, :]   (108 rows reverted)\n"
            "  W_up[C*, :]   ← W_up_base[C*, :]     (108 rows reverted)\n"
            "  W_down[:, C*] ← W_down_base[:, C*]   (108 cols reverted)\n"
            "All other 13,716 units in Layer l stay 100% untouched!",
            ha="center", va="center", fontsize=6.7, fontweight="bold", color="#1E293B")

    # Right half: Mode 2 — Clade Grafting (Transplant Math Clades + Code Clades into Base)
    s2_box = FancyBboxPatch((0.52, 0.19), 0.455, 0.635,
                            boxstyle="round,pad=0.008,rounding_size=0.018",
                            facecolor="#ECFDF5", edgecolor="#059669", lw=1.5, zorder=2)
    ax.add_patch(s2_box)
    ax.text(0.7475, 0.785, "Surgery 2: Clade Grafting (Zero-Training Chimera)",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="#065F46")
    ax.text(0.7475, 0.738, "Transplant Disjoint Specialist Clades into One MLP Matrix",
            ha="center", va="center", fontsize=7.1, color="#047857")

    m_rows = ["#7C3AED", "#7C3AED", "#E2E8F0", "#E2E8F0", "#E2E8F0", "#E2E8F0"]
    draw_matrix_with_highlight(
        ax, 0.54, 0.44, 0.115, 0.21,
        "DeepSeek-R1", "Math Clade C_math", mode="row", hi_colors=m_rows
    )
    ax.text(0.675, 0.545, "+", ha="center", va="center", fontsize=13.0, fontweight="bold", color="#059669")

    c_rows = ["#E2E8F0", "#E2E8F0", "#E2E8F0", "#E2E8F0", "#0284C7", "#0284C7"]
    draw_matrix_with_highlight(
        ax, 0.695, 0.44, 0.115, 0.21,
        "Coder-14B", "Code Clade C_code", mode="row", hi_colors=c_rows
    )
    ax.text(0.83, 0.545, "→", ha="center", va="center", fontsize=13.0, fontweight="bold", color="#059669")

    chim_rows = ["#7C3AED", "#7C3AED", "#94A3B8", "#94A3B8", "#0284C7", "#0284C7"]
    draw_matrix_with_highlight(
        ax, 0.845, 0.44, 0.115, 0.21,
        "Chimera MLP", "Math + Base + Code", mode="row", hi_colors=chim_rows
    )

    ax.text(0.7475, 0.295,
            "Exact Matrix Operation (with C_math ∩ C_code = ∅):\n"
            "  • Start with Base W_gate, W_up, W_down\n"
            "  • Copy rows/cols C_math from DeepSeek-R1 (purple)\n"
            "  • Copy rows/cols C_code from Qwen-Coder (blue)\n"
            "  → Both specialist sub-circuits coexist without colliding!",
            ha="center", va="center", fontsize=6.7, fontweight="bold", color="#064E3B")

    # Bottom summary bar
    foot = FancyBboxPatch((0.025, 0.035), 0.95, 0.13,
                          boxstyle="round,pad=0.005,rounding_size=0.014",
                          facecolor="#ECFDF5", edgecolor="#10B981", lw=1.1, zorder=2)
    ax.add_patch(foot)
    ax.text(0.50, 0.10,
            "Summary of Step 4:  'Tree Surgery' means using a clade's leaf IDs C = {i_1, ..., i_108} as an index mask to swap the matching 108 rows of W_gate,\n"
            "108 rows of W_up, and 108 columns of W_down between models — transplanting an entire co-adapted multi-neuron circuit as an intact unit!",
            ha="center", va="center", fontsize=6.5, fontweight="bold", color="#065F46")


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig = plt.figure(figsize=(18, 11.4), dpi=300, facecolor="#F1F5F9")

    # Top Title Banner
    title_ax = fig.add_axes([0.015, 0.922, 0.97, 0.065], frameon=False)
    title_ax.set_xlim(0, 1)
    title_ax.set_ylim(0, 1)
    title_ax.axis("off")
    banner = FancyBboxPatch((0.0, 0.04), 1.0, 0.92,
                            boxstyle="round,pad=0.005,rounding_size=0.015",
                            facecolor="#0F172A", edgecolor="#334155", lw=1.5)
    title_ax.add_patch(banner)
    title_ax.text(
        0.02, 0.62,
        "What Is a Taxon, What Is a Clade, and How Does Tree Surgery Work on a 14B LLM?",
        color="#FFFFFF", fontsize=14.8, fontweight="bold", ha="left", va="center"
    )
    title_ax.text(
        0.02, 0.23,
        "14.7B Weights = 48 Layers × 13,824 MLP Units/Layer   |   1 Taxon (Leaf) = 1 MLP Unit (Row i of W_gate, Row i of W_up, Col i of W_down = 15,360 floats)",
        color="#94A3B8", fontsize=9.1, ha="left", va="center"
    )

    ax1 = add_stage_card(
        fig, [0.015, 0.475, 0.478, 0.435],
        "STEP 1 • ONE TAXON",
        "14B Weights vs. 13,824 Units: What Is 1 Taxon?",
        "1 Taxon = MLP Hidden Unit #i (Row i of W_gate + Row i of W_up + Col i of W_down)",
        badge_color="#4F46E5"
    )
    draw_panel_1(ax1)

    ax2 = add_stage_card(
        fig, [0.507, 0.475, 0.478, 0.435],
        "STEP 2 • ONE CLADE",
        "What Is the Tree & What Is a Clade in the Matrices?",
        "13,824 leaves per layer tree; 1 Clade (~108 leaves) = 108 rows of W_gate/up & 108 cols of W_down",
        badge_color="#0284C7"
    )
    draw_panel_2(ax2)

    ax3 = add_stage_card(
        fig, [0.015, 0.02, 0.478, 0.435],
        "STEP 3 • TWO MODELS",
        "Comparing Base Tree vs. Specialist Tree (Same Leaf IDs)",
        "Why Quartet Violations catch units swapping partners while ignoring rigid block rotation",
        badge_color="#DC2626"
    )
    draw_panel_3(ax3)

    ax4 = add_stage_card(
        fig, [0.507, 0.02, 0.478, 0.435],
        "STEP 4 • SURGERY",
        "Exact Row & Column Slicing on W_gate, W_up, W_down",
        "Swapping the 108 rows & columns of a rewired Clade C* between models (Zero Training)",
        badge_color="#059669"
    )
    draw_panel_4(ax4)

    png_path = os.path.join(out_dir, "qwen_tree_surgery_method.png")
    pdf_path = os.path.join(out_dir, "qwen_tree_surgery_method.pdf")
    svg_path = os.path.join(out_dir, "qwen_tree_surgery_method.svg")

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved:\n  {png_path}\n  {pdf_path}\n  {svg_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
generate_qwen_method_diagram.py

Generates a pure Method & Intuition architectural schematic (no results charts)
explaining Phylogenetic Tree Surgery on Transformer LLMs in 4 visual panels:
  1. PANEL 1: Two Transformer Models (Base vs. Fine-Tuned) & Inside the SwiGLU MLP Block
              (How each neuron i is a rank-1 slice of W_gate, W_up, W_down).
  2. PANEL 2: What the Phylogenetic Tree Actually Is
              (Leaves = individual MLP neurons; Clades = functional sub-circuits / sub-matrices).
  3. PANEL 3: What Tree Changes Mean Between Two Models
              (Uniform SGD Subspace Drift vs. True Relational Rewiring / Quartet Partner Swaps).
  4. PANEL 4: How "Tree Surgery" Happens on Transformer Weight Matrices
              (Using tree clades as an index mask for Surgical Knock-Out & Chimera Grafting).

Outputs:
  - qwen_tree_surgery_method.png (300 DPI)
  - qwen_tree_surgery_method.pdf (Vector PDF)
  - qwen_tree_surgery_method.svg (Vector SVG)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


def add_stage_card(fig, bbox, step_badge, title_text, subtitle_text, badge_color="#4F46E5"):
    """Create a clean card with a colored badge, title, and subtitle."""
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
        (0.022, 0.892), 0.21, 0.072,
        boxstyle="round,pad=0.004,rounding_size=0.014",
        facecolor=badge_color, edgecolor="none", zorder=2
    )
    ax.add_patch(pill)
    ax.text(
        0.127, 0.928, step_badge,
        color="#FFFFFF", fontsize=9.0, fontweight="bold",
        ha="center", va="center", zorder=3
    )
    ax.text(
        0.248, 0.938, title_text,
        color="#0F172A", fontsize=11.5, fontweight="bold",
        ha="left", va="center", zorder=3
    )
    ax.text(
        0.248, 0.898, subtitle_text,
        color="#475569", fontsize=8.4,
        ha="left", va="center", zorder=3
    )
    ax.plot([0.022, 0.978], [0.865, 0.865], color="#E2E8F0", lw=1.1, zorder=2)
    return ax


def draw_neuron_matrix_strips(ax, x0, y0, w, h, stripe_colors, title_lbl, sub_lbl, vertical=True):
    """Draw a stylized weight matrix partitioned into colored neuron stripes."""
    outer = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.002,rounding_size=0.008",
        facecolor="#F8FAFC", edgecolor="#334155", lw=1.3, zorder=3
    )
    ax.add_patch(outer)
    n_stripes = len(stripe_colors)
    pad = 0.004
    if vertical:
        sw = (w - 2 * pad) / n_stripes
        for idx, col in enumerate(stripe_colors):
            r = Rectangle(
                (x0 + pad + idx * sw, y0 + pad), sw * 0.86, h - 2 * pad,
                facecolor=col, edgecolor="none", alpha=0.85, zorder=4
            )
            ax.add_patch(r)
    else:
        sh = (h - 2 * pad) / n_stripes
        for idx, col in enumerate(stripe_colors):
            r = Rectangle(
                (x0 + pad, y0 + pad + idx * sh), w - 2 * pad, sh * 0.84,
                facecolor=col, edgecolor="none", alpha=0.85, zorder=4
            )
            ax.add_patch(r)

    ax.text(x0 + w / 2.0, y0 + h + 0.035, title_lbl,
            ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#0F172A", zorder=5)
    if sub_lbl:
        ax.text(x0 + w / 2.0, y0 + h + 0.008, sub_lbl,
                ha="center", va="bottom", fontsize=6.7, color="#475569", zorder=5)


def draw_panel_1(ax):
    """
    PANEL 1: Two Sibling Transformer Models & Inside the SwiGLU MLP Block.
    Shows Base Model vs. Specialist Model -> Zoom into Layer l -> MLP Matrices -> Neuron i.
    """
    # Left side: Two Transformer Stacks (Base vs Fine-Tuned Specialist)
    ax.text(0.195, 0.825, "Two Sibling 14B Transformer Models\n(Shared Ancestor Initialization)",
            ha="center", va="top", fontsize=8.5, fontweight="bold", color="#1E293B")

    # Model A: Base
    mod_a = FancyBboxPatch((0.025, 0.25), 0.155, 0.49,
                           boxstyle="round,pad=0.006,rounding_size=0.016",
                           facecolor="#F8FAFC", edgecolor="#64748B", lw=1.4, zorder=2)
    ax.add_patch(mod_a)
    ax.text(0.1025, 0.705, "Model A (Base)", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="#334155")
    ax.text(0.1025, 0.668, "Qwen2.5-14B", ha="center", va="center",
            fontsize=7.1, color="#64748B")

    # Draw layers inside Model A
    for ly, lbl, hi in [(0.30, "Layer 1", False), (0.42, "Layer l (MLP)", True), (0.55, "Layer 48", False)]:
        fc = "#E2E8F0" if not hi else "#DBEAFE"
        ec = "#94A3B8" if not hi else "#2563EB"
        box = FancyBboxPatch((0.038, ly), 0.129, 0.075,
                             boxstyle="round,pad=0.003,rounding_size=0.01",
                             facecolor=fc, edgecolor=ec, lw=1.5 if hi else 1.0, zorder=3)
        ax.add_patch(box)
        ax.text(0.1025, ly + 0.0375, lbl, ha="center", va="center",
                fontsize=7.1, fontweight="bold" if hi else "normal",
                color="#1E3A8A" if hi else "#475569", zorder=4)

    # Model B: Specialist (R1 / Coder)
    mod_b = FancyBboxPatch((0.205, 0.25), 0.155, 0.49,
                           boxstyle="round,pad=0.006,rounding_size=0.016",
                           facecolor="#FAF5FF", edgecolor="#7C3AED", lw=1.4, zorder=2)
    ax.add_patch(mod_b)
    ax.text(0.2825, 0.705, "Model B (Specialist)", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="#6D28D9")
    ax.text(0.2825, 0.668, "DeepSeek-R1 / Coder", ha="center", va="center",
            fontsize=7.0, color="#7C3AED")

    for ly, lbl, hi in [(0.30, "Layer 1", False), (0.42, "Layer l (MLP)", True), (0.55, "Layer 48", False)]:
        fc = "#F3E8FF" if not hi else "#EDE9FE"
        ec = "#C084FC" if not hi else "#7C3AED"
        box = FancyBboxPatch((0.218, ly), 0.129, 0.075,
                             boxstyle="round,pad=0.003,rounding_size=0.01",
                             facecolor=fc, edgecolor=ec, lw=1.5 if hi else 1.0, zorder=3)
        ax.add_patch(box)
        ax.text(0.2825, ly + 0.0375, lbl, ha="center", va="center",
                fontsize=7.1, fontweight="bold" if hi else "normal",
                color="#5B21B6" if hi else "#6B21A8", zorder=4)

    # Zoom arrow from Layer l to the SwiGLU MLP Block diagram on the right
    ax.annotate("", xy=(0.455, 0.458), xytext=(0.365, 0.458),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.45",
                                color="#4F46E5", lw=2.0))
    ax.text(0.41, 0.505, "Zoom into\nMLP Layer l", ha="center", va="bottom",
            fontsize=6.9, fontweight="bold", color="#4F46E5")

    # Right side: Inside the Transformer SwiGLU MLP Block
    mlp_card = FancyBboxPatch((0.46, 0.19), 0.51, 0.63,
                              boxstyle="round,pad=0.008,rounding_size=0.018",
                              facecolor="#F8FAFC", edgecolor="#94A3B8", lw=1.3, ls="--", zorder=2)
    ax.add_patch(mlp_card)
    ax.text(0.715, 0.785, "Inside One Transformer SwiGLU MLP Layer (13,824 Neurons)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0F172A")
    ax.text(0.715, 0.735, "MLP(x) = Σ_{i=1..13,824}  [ SiLU(w_gate,i · x) × (w_up,i · x) ] · w_down,i",
            ha="center", va="center", fontsize=7.3, fontweight="bold", color="#4338CA")

    # Draw W_gate, W_up, and W_down with Neuron i highlighted in bright red/orange!
    stripes = ["#CBD5E1", "#CBD5E1", "#EF4444", "#CBD5E1", "#CBD5E1", "#CBD5E1", "#CBD5E1", "#CBD5E1"]
    draw_neuron_matrix_strips(
        ax, 0.49, 0.36, 0.125, 0.25, stripes,
        "W_gate", "[5,120 × 13,824]", vertical=True
    )
    draw_neuron_matrix_strips(
        ax, 0.65, 0.36, 0.125, 0.25, stripes,
        "W_up", "[5,120 × 13,824]", vertical=True
    )
    draw_neuron_matrix_strips(
        ax, 0.81, 0.36, 0.13, 0.25, stripes,
        "W_down", "[13,824 × 5,120]", vertical=False
    )

    # Callout box and 3 arrows pointing to the red slice in W_gate, W_up, W_down
    callout = FancyBboxPatch((0.555, 0.21), 0.32, 0.088,
                             boxstyle="round,pad=0.004,rounding_size=0.012",
                             facecolor="#FEF2F2", edgecolor="#EF4444", lw=1.2, zorder=5)
    ax.add_patch(callout)
    ax.text(0.715, 0.254, "Neuron i's Weight Slice (1 of 13,824)\nCol i of W_gate, W_up  +  Row i of W_down",
            ha="center", va="center", fontsize=7.1, fontweight="bold", color="#B91C1C", zorder=6)
    for tx, ty in [(0.53, 0.358), (0.69, 0.358), (0.875, 0.435)]:
        ax.annotate("", xy=(tx, ty), xytext=(0.715, 0.302),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.32",
                                    color="#DC2626", lw=1.3), zorder=5)

    # Bottom explanation bar
    foot = FancyBboxPatch((0.025, 0.035), 0.95, 0.115,
                          boxstyle="round,pad=0.005,rounding_size=0.014",
                          facecolor="#EEF2FF", edgecolor="#6366F1", lw=1.1, zorder=2)
    ax.add_patch(foot)
    ax.text(0.50, 0.092,
            "What is a 'Species' here?  Each MLP neuron i (out of 13,824 in layer l) is defined by its input/output weight vectors.\n"
            "Because Model A and Model B started from the same checkpoint, Neuron #412 in Model A is the SAME ancestral neuron as Neuron #412 in Model B!",
            ha="center", va="center", fontsize=6.9, fontweight="bold", color="#1E1B4B")


def draw_panel_2(ax):
    """
    PANEL 2: What the Phylogenetic Tree Is (Leaves = Neurons, Clades = Functional Circuits).
    Shows how the 13,824 neurons form a binary tree where subtrees (clades) map to blocks of columns/rows.
    """
    ax.text(0.50, 0.825, "We Group the 13,824 Neurons in Layer l into a Binary Phylogenetic Tree by Weight Similarity",
            ha="center", va="top", fontsize=8.6, fontweight="bold", color="#0F172A")

    # Top-left to bottom-left: Tree structure from Root -> Internal Branches -> 4 Clades -> Individual Neuron Leaves
    # Root at (0.30, 0.72)
    edges = [
        ((0.30, 0.72), (0.16, 0.62)),
        ((0.30, 0.72), (0.44, 0.62)),
        ((0.16, 0.62), (0.09, 0.50)),
        ((0.16, 0.62), (0.23, 0.50)),
        ((0.44, 0.62), (0.37, 0.50)),
        ((0.44, 0.62), (0.51, 0.50)),
    ]
    for (x1, y1), (x2, y2) in edges:
        ax.plot([x1, x2], [y1, y1], color="#475569", lw=2.0, zorder=2)
        ax.plot([x2, x2], [y1, y2], color="#475569", lw=2.0, zorder=2)

    ax.plot(0.30, 0.72, "o", color="#0F172A", markersize=6.0, zorder=4)
    ax.text(0.30, 0.745, "Layer l Tree Root (All 13,824 Neurons)",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#0F172A")

    # 4 Clade triangles (Clade A: Purple/Math, Clade B: Green/Shared, Clade C: Blue/Code, Clade D: Slate/Shared)
    clades = [
        (0.09, "#7C3AED", "#F3E8FF", "Clade A\n(~108)", ["n_12", "n_89", "n_401"]),
        (0.23, "#10B981", "#D1FAE5", "Clade B\n(~108)", ["n_5", "n_77", "n_210"]),
        (0.37, "#0284C7", "#E0F2FE", "Clade C\n(~108)", ["n_33", "n_94", "n_512"]),
        (0.51, "#64748B", "#F1F5F9", "Clade D\n(~108)", ["n_8", "n_61", "n_809"]),
    ]
    for cx, col, fill, cl_lbl, leaves in clades:
        ax.plot(cx, 0.50, "o", color=col, markersize=6.5, zorder=5)
        tri = Polygon([[cx, 0.50], [cx - 0.058, 0.30], [cx + 0.058, 0.30]],
                      closed=True, facecolor=fill, edgecolor=col, lw=1.8, zorder=3)
        ax.add_patch(tri)
        ax.text(cx, 0.355, cl_lbl, ha="center", va="center",
                fontsize=6.5, fontweight="bold", color=col, zorder=4)
        for l_idx, (dx, l_nm) in enumerate(zip([-0.042, 0.0, 0.042], leaves)):
            lx = cx + dx
            ax.plot([lx, lx], [0.30, 0.265], color=col, lw=1.5, zorder=3)
            ax.plot(lx, 0.265, "s", color=col, markersize=4.2, zorder=4)
            ax.text(lx, 0.248, l_nm, ha="center", va="top", fontsize=6.1,
                    fontweight="bold", color="#334155")

    ax.text(0.30, 0.19, "Leaves = Individual MLP Neurons (1..13,824)   |   Clades = Subtrees of Co-Adapted Neurons",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color="#334155")

    # Arrow from Tree Clades to Functional Sub-Matrices on the right
    ax.annotate("", xy=(0.665, 0.43), xytext=(0.585, 0.43),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.45",
                                color="#0284C7", lw=2.0))
    ax.text(0.625, 0.475, "Maps to\nMLP Slices", ha="center", va="bottom",
            fontsize=6.9, fontweight="bold", color="#0284C7")

    # Right box: What a Clade Means Inside the MLP Weight Matrix
    sub_card = FancyBboxPatch((0.67, 0.19), 0.305, 0.58,
                              boxstyle="round,pad=0.008,rounding_size=0.018",
                              facecolor="#F8FAFC", edgecolor="#0284C7", lw=1.4, zorder=2)
    ax.add_patch(sub_card)
    ax.text(0.8225, 0.725, "What a Clade Means in ML Terms",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="#0369A1")
    ax.text(0.8225, 0.675, "Each Clade = A Multi-Neuron Sub-Circuit!",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="#0F172A")

    # Draw MLP weight matrix grouped into Clade A (purple), Clade B (green), Clade C (blue), Clade D (slate)
    clade_stripes = [
        "#7C3AED", "#7C3AED", "#7C3AED",  # Clade A
        "#10B981", "#10B981", "#10B981",  # Clade B
        "#0284C7", "#0284C7", "#0284C7",  # Clade C
        "#94A3B8", "#94A3B8", "#94A3B8",  # Clade D
    ]
    draw_neuron_matrix_strips(
        ax, 0.70, 0.35, 0.245, 0.25, clade_stripes,
        "MLP Weight Matrix W_layer", "Grouped by Tree Clades (A, B, C, D)", vertical=True
    )

    ax.text(0.8225, 0.245,
            "• Single neurons are too small / polysemantic\n"
            "• Whole layers (13,824 neurons) are too coarse\n"
            "• A Tree Clade (~100 neurons) is the natural\n"
            "  functional building block (organelle) of an MLP!",
            ha="center", va="center", fontsize=6.9, color="#1E293B")

    # Bottom explanation bar
    foot = FancyBboxPatch((0.025, 0.035), 0.95, 0.115,
                          boxstyle="round,pad=0.005,rounding_size=0.014",
                          facecolor="#F0F9FF", edgecolor="#0284C7", lw=1.1, zorder=2)
    ax.add_patch(foot)
    ax.text(0.50, 0.092,
            "What is the Tree?  Just as a phylogenetic tree in biology groups species into families & genera, our GPU BME tree\n"
            "groups the 13,824 neurons in each Transformer layer into ~128 evolutionary clades (6,171 clades across all 48 layers).",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="#0C4A6E")


def draw_panel_3(ax):
    """
    PANEL 3: What Tree Changes Mean Between Two Models (Uniform Drift vs. Quartet Rewiring).
    Visualizes 4 neurons in Model A vs Model B in both weight space and tree topology.
    """
    # Left half: Case 1 — Uniform Subspace Drift / Rotation (No Tree Change)
    c1_box = FancyBboxPatch((0.025, 0.185), 0.455, 0.64,
                            boxstyle="round,pad=0.008,rounding_size=0.018",
                            facecolor="#FFFBEB", edgecolor="#F59E0B", lw=1.4, zorder=2)
    ax.add_patch(c1_box)
    ax.text(0.2525, 0.785, "Case A: Uniform Weight Drift / Rotation",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color="#B45309")
    ax.text(0.2525, 0.738, "Why Standard Weight Diffs (||W_B − W_A||) Get Fooled",
            ha="center", va="center", fontsize=7.4, color="#92400E")

    # Base tree of 4 neurons: (n1, n2) | (n3, n4)
    ax.text(0.135, 0.67, "Model A (Base Tree)", ha="center", va="center",
            fontsize=7.4, fontweight="bold", color="#475569")
    ax.plot([0.065, 0.105], [0.60, 0.53], color="#475569", lw=2.0)
    ax.plot([0.065, 0.105], [0.46, 0.53], color="#475569", lw=2.0)
    ax.plot([0.105, 0.165], [0.53, 0.53], color="#475569", lw=2.5)
    ax.plot([0.165, 0.205], [0.53, 0.60], color="#475569", lw=2.0)
    ax.plot([0.165, 0.205], [0.53, 0.46], color="#475569", lw=2.0)
    for px, py, lbl in [(0.065, 0.60, "n1"), (0.065, 0.46, "n2"), (0.205, 0.60, "n3"), (0.205, 0.46, "n4")]:
        ax.plot(px, py, "o", color="#334155", markersize=5.5)
        ax.text(px, py + 0.028, lbl, ha="center", fontsize=7.4, fontweight="bold", color="#0F172A")

    ax.annotate("", xy=(0.285, 0.53), xytext=(0.23, 0.53),
                arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.35", color="#D97706", lw=1.8))
    ax.text(0.257, 0.565, "SGD\nDrift", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color="#B45309")

    # Model B tree: still (n1, n2) | (n3, n4)!
    ax.text(0.37, 0.67, "Model B (Rotated, Same Tree!)", ha="center", va="center",
            fontsize=7.4, fontweight="bold", color="#B45309")
    ax.plot([0.305, 0.345], [0.60, 0.53], color="#D97706", lw=2.0)
    ax.plot([0.305, 0.345], [0.46, 0.53], color="#D97706", lw=2.0)
    ax.plot([0.345, 0.405], [0.53, 0.53], color="#D97706", lw=2.5)
    ax.plot([0.405, 0.445], [0.53, 0.60], color="#D97706", lw=2.0)
    ax.plot([0.405, 0.445], [0.53, 0.46], color="#D97706", lw=2.0)
    for px, py, lbl in [(0.305, 0.60, "n1'"), (0.305, 0.46, "n2'"), (0.445, 0.60, "n3'"), (0.445, 0.46, "n4'")]:
        ax.plot(px, py, "o", color="#B45309", markersize=5.5)
        ax.text(px, py + 0.028, lbl, ha="center", fontsize=7.4, fontweight="bold", color="#92400E")

    ax.text(0.2525, 0.30,
            "• Every weight vector moved (||W_B − W_A|| is LARGE)\n"
            "• Magnitude / TIES / Cosine think this clade changed!\n"
            "• But tree partners {n1, n2} | {n3, n4} are IDENTICAL\n"
            "  → Our Tree Quartet Score = 0 (Ignored as harmless drift)",
            ha="center", va="center", fontsize=7.0, color="#78350F")

    # Right half: Case 2 — True Relational Rewiring (Quartet Topology Violation)
    c2_box = FancyBboxPatch((0.52, 0.185), 0.455, 0.64,
                            boxstyle="round,pad=0.008,rounding_size=0.018",
                            facecolor="#FEF2F2", edgecolor="#EF4444", lw=1.5, zorder=2)
    ax.add_patch(c2_box)
    ax.text(0.7475, 0.785, "Case B: Tree Quartet Violation (Partner Swap)",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color="#B91C1C")
    ax.text(0.7475, 0.738, "True Functional Circuit Rewiring Between Models",
            ha="center", va="center", fontsize=7.4, color="#991B1B")

    # Base tree: (n1, n2) | (n3, n4)
    ax.text(0.63, 0.67, "Model A (Base Tree)", ha="center", va="center",
            fontsize=7.4, fontweight="bold", color="#475569")
    ax.plot([0.56, 0.60], [0.60, 0.53], color="#475569", lw=2.0)
    ax.plot([0.56, 0.60], [0.46, 0.53], color="#475569", lw=2.0)
    ax.plot([0.60, 0.66], [0.53, 0.53], color="#475569", lw=2.5)
    ax.plot([0.66, 0.70], [0.53, 0.60], color="#475569", lw=2.0)
    ax.plot([0.66, 0.70], [0.53, 0.46], color="#475569", lw=2.0)
    for px, py, lbl in [(0.56, 0.60, "n1"), (0.56, 0.46, "n2"), (0.70, 0.60, "n3"), (0.70, 0.46, "n4")]:
        ax.plot(px, py, "o", color="#334155", markersize=5.5)
        ax.text(px, py + 0.028, lbl, ha="center", fontsize=7.4, fontweight="bold", color="#0F172A")

    ax.annotate("", xy=(0.78, 0.53), xytext=(0.725, 0.53),
                arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.35", color="#DC2626", lw=1.8))
    ax.text(0.752, 0.565, "SPR\nSwap!", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color="#DC2626")

    # Rewired tree: (n1, n3) | (n2, n4) — n2 and n3 swapped partners!
    ax.text(0.865, 0.67, "Model B (Partners Swapped!)", ha="center", va="center",
            fontsize=7.4, fontweight="bold", color="#B91C1C")
    ax.plot([0.80, 0.84], [0.60, 0.53], color="#DC2626", lw=2.0)
    ax.plot([0.80, 0.84], [0.46, 0.53], color="#DC2626", lw=2.0)
    ax.plot([0.84, 0.90], [0.53, 0.53], color="#DC2626", lw=2.5)
    ax.plot([0.90, 0.94], [0.53, 0.60], color="#DC2626", lw=2.0)
    ax.plot([0.90, 0.94], [0.53, 0.46], color="#DC2626", lw=2.0)
    for px, py, lbl in [(0.80, 0.60, "n1"), (0.80, 0.46, "n3"), (0.94, 0.60, "n2"), (0.94, 0.46, "n4")]:
        is_swapped = lbl in ("n2", "n3")
        c_pt = "#DC2626" if is_swapped else "#334155"
        ax.plot(px, py, "o", color=c_pt, markersize=6.0 if is_swapped else 5.5)
        ax.text(px, py + 0.028, lbl, ha="center", fontsize=7.6 if is_swapped else 7.4,
                fontweight="bold", color=c_pt)

    ax.text(0.7475, 0.30,
            "• Neurons n2 and n3 SWAPPED evolutionary partners:\n"
            "  Base: {n1, n2} | {n3, n4}   →   Specialist: {n1, n3} | {n2, n4}\n"
            "• Who fires with whom inside the MLP block changed!\n"
            "  → High Quartet Violation Score (Flagged for Tree Surgery)",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color="#991B1B")

    # Bottom explanation bar
    foot = FancyBboxPatch((0.025, 0.035), 0.95, 0.115,
                          boxstyle="round,pad=0.005,rounding_size=0.014",
                          facecolor="#FEF2F2", edgecolor="#EF4444", lw=1.1, zorder=2)
    ax.add_patch(foot)
    ax.text(0.50, 0.092,
            "The Core Idea:  Instead of asking 'Which individual weights moved the most?' (which picks up noisy SGD drift),\n"
            "we ask a relational question: 'Which tree clades broke their neighborhood topology (swapped quartet partners) during fine-tuning?'",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="#7F1D1D")


def draw_panel_4(ax):
    """
    PANEL 4: How 'Tree Surgery' Happens on Transformer Weight Matrices & What It Means.
    Shows Mode 1 (Clade Knock-Out / Ablation) and Mode 2 (Clade Grafting / Hybrid Chimera).
    """
    # Left half: Surgery Mode 1 — Clade Knock-Out (Causal Ablation)
    s1_box = FancyBboxPatch((0.025, 0.185), 0.455, 0.64,
                            boxstyle="round,pad=0.008,rounding_size=0.018",
                            facecolor="#F8FAFC", edgecolor="#6366F1", lw=1.4, zorder=2)
    ax.add_patch(s1_box)
    ax.text(0.2525, 0.785, "Surgery Mode 1: Clade Knock-Out (Ablation)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#312E81")
    ax.text(0.2525, 0.738, "Test If a Rewired Clade Is Causally Necessary for a Skill",
            ha="center", va="center", fontsize=7.3, color="#4338CA")

    # Specialist MLP Matrix + Base Clade Slice -> Surgically Knocked-Out Matrix
    s_r1 = ["#7C3AED", "#7C3AED", "#EF4444", "#EF4444", "#7C3AED", "#7C3AED"]
    draw_neuron_matrix_strips(
        ax, 0.045, 0.43, 0.115, 0.22, s_r1,
        "Specialist MLP", "(Rewired Clade C* in Red)", vertical=True
    )
    ax.text(0.18, 0.54, "←", ha="center", va="center", fontsize=13.0, fontweight="bold", color="#DC2626")

    s_base = ["#CBD5E1", "#CBD5E1", "#64748B", "#64748B", "#CBD5E1", "#CBD5E1"]
    draw_neuron_matrix_strips(
        ax, 0.20, 0.43, 0.115, 0.22, s_base,
        "Base Clade C*", "(Original Ancestor Slice)", vertical=True
    )
    ax.text(0.335, 0.54, "=", ha="center", va="center", fontsize=13.0, fontweight="bold", color="#334155")

    s_ko = ["#7C3AED", "#7C3AED", "#64748B", "#64748B", "#7C3AED", "#7C3AED"]
    draw_neuron_matrix_strips(
        ax, 0.35, 0.43, 0.115, 0.22, s_ko,
        "Knocked-Out MLP", "(Only Clade C* Reverted)", vertical=True
    )

    ax.text(0.2525, 0.285,
            "• How it works: Keep 99% of the Reasoning Model intact,\n"
            "  but overwrite ONLY the columns/rows of Clade C* in\n"
            "  W_gate, W_up, W_down with the Base Model's weights.\n"
            "• What it means: If reasoning collapses while general text stays\n"
            "  intact, Clade C* is the actual causal reasoning circuit!",
            ha="center", va="center", fontsize=6.9, color="#1E293B")

    # Right half: Surgery Mode 2 — Clade Grafting (Zero-Training Chimera)
    s2_box = FancyBboxPatch((0.52, 0.185), 0.455, 0.64,
                            boxstyle="round,pad=0.008,rounding_size=0.018",
                            facecolor="#ECFDF5", edgecolor="#059669", lw=1.5, zorder=2)
    ax.add_patch(s2_box)
    ax.text(0.7475, 0.785, "Surgery Mode 2: Clade Grafting (Chimera)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#065F46")
    ax.text(0.7475, 0.738, "Combine Specialist Circuits into One Model (Zero Training)",
            ha="center", va="center", fontsize=7.3, color="#047857")

    # Math Clade (from R1) + Code Clade (from Coder) grafted onto Base -> Chimera MLP
    g_r1 = ["#7C3AED", "#7C3AED", "#E2E8F0", "#E2E8F0", "#E2E8F0", "#E2E8F0"]
    draw_neuron_matrix_strips(
        ax, 0.54, 0.43, 0.115, 0.22, g_r1,
        "Math Clades", "(From DeepSeek-R1)", vertical=True
    )
    ax.text(0.675, 0.54, "+", ha="center", va="center", fontsize=13.0, fontweight="bold", color="#059669")

    g_code = ["#E2E8F0", "#E2E8F0", "#E2E8F0", "#E2E8F0", "#0284C7", "#0284C7"]
    draw_neuron_matrix_strips(
        ax, 0.695, 0.43, 0.115, 0.22, g_code,
        "Code Clades", "(From Coder-14B)", vertical=True
    )
    ax.text(0.83, 0.54, "→", ha="center", va="center", fontsize=13.0, fontweight="bold", color="#059669")

    g_chim = ["#7C3AED", "#7C3AED", "#94A3B8", "#94A3B8", "#0284C7", "#0284C7"]
    draw_neuron_matrix_strips(
        ax, 0.845, 0.43, 0.115, 0.22, g_chim,
        "Chimera MLP", "(Math + Base + Code)", vertical=True
    )

    ax.text(0.7475, 0.285,
            "• Why 50/50 weight averaging fails: Averaging two models\n"
            "  blends every neuron half-and-half, scrambling non-linear circuits.\n"
            "• How Tree Grafting works: Transplant whole, intact Math clades\n"
            "  (purple slices) and non-overlapping Code clades (blue slices)\n"
            "  into the Base MLP block — zero training or fine-tuning!",
            ha="center", va="center", fontsize=6.9, color="#064E3B")

    # Bottom explanation bar
    foot = FancyBboxPatch((0.025, 0.035), 0.95, 0.115,
                          boxstyle="round,pad=0.005,rounding_size=0.014",
                          facecolor="#ECFDF5", edgecolor="#10B981", lw=1.1, zorder=2)
    ax.add_patch(foot)
    ax.text(0.50, 0.092,
            "Why 'Tree Surgery' Works:  Because SwiGLU MLPs are a sum over 13,824 neurons, replacing or grafting an entire tree clade C\n"
            "(cols C of W_gate, W_up and rows C of W_down) swaps a complete, self-contained functional sub-circuit without breaking the layer!",
            ha="center", va="center", fontsize=6.9, fontweight="bold", color="#065F46")


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
        "How Phylogenetic Tree Surgery Works on Transformer LLMs (Concept & Method)",
        color="#FFFFFF", fontsize=15.0, fontweight="bold", ha="left", va="center"
    )
    title_ax.text(
        0.02, 0.23,
        "1. Treat MLP Neurons as Homologous Species  →  2. Build Layer Trees of Clades (Sub-Circuits)  →  3. Detect Partner Swaps (Quartet Violations)  →  4. Perform Matrix Surgery",
        color="#94A3B8", fontsize=9.3, ha="left", va="center"
    )

    # 2x2 Grid of Detailed Conceptual Panels
    ax1 = add_stage_card(
        fig, [0.015, 0.475, 0.478, 0.435],
        "STEP 1 • THE MODELS",
        "Two Transformer Models & Inside the MLP Block",
        "Each MLP layer has 13,824 neurons — same index i across Base and Specialist models",
        badge_color="#4F46E5"
    )
    draw_panel_1(ax1)

    ax2 = add_stage_card(
        fig, [0.507, 0.475, 0.478, 0.435],
        "STEP 2 • THE TREE",
        "What the Tree Is: Leaves = Neurons, Clades = Circuits",
        "Clustering neurons by weight similarity partitions the MLP matrix into functional clades",
        badge_color="#0284C7"
    )
    draw_panel_2(ax2)

    ax3 = add_stage_card(
        fig, [0.015, 0.02, 0.478, 0.435],
        "STEP 3 • REWIRING",
        "What Tree Changes Mean: Drift vs. Partner Swaps",
        "Why Tree Quartet Violations isolate real circuit rewiring while ignoring uniform SGD drift",
        badge_color="#DC2626"
    )
    draw_panel_3(ax3)

    ax4 = add_stage_card(
        fig, [0.507, 0.02, 0.478, 0.435],
        "STEP 4 • SURGERY",
        "How Tree Surgery Happens on Transformer Matrices",
        "Using tree clades as a surgical index mask for Knock-Out (ablation) and Chimera (grafting)",
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

#!/usr/bin/env python3
"""FIGURE FOR THE TOWER REVISION.  (figures/fig_tower.pdf, four panels.)

(a) ONE INEQUALITY, NOT SIX.  The cumulant ratio c_{k+1}/c_k must be
    non-decreasing in k; its first link is the channel-budget inequality and its
    second is the channel-slope wall.  Drawn for three clocks: a single-atom
    (cure) clock, where the chain is flat and every rung is tight, and two
    multi-jump clocks, where it rises.  A two-atom (indivisible) chart is shown
    for contrast: the chain falls, which is the refutation.

(b) THE ORDER THEOREM.  The order at which each hypothesis first fails under an
    inadmissible clock m = e^{ax} - 1.  The composition law breaks at order 3,
    inside the measurable range; the axiom survives to order 5.  The shaded band
    is what a published window actually resolves.

(c) THE FOUR-POINT TEST.  T = ln L1 + ln L3 - 2 ln L2 on truths whose verdict is
    known, analytic against the median of 4000 draws at 10^4 items, with the
    panel's two families overlaid with their bootstrap intervals.  The statistic
    is unbiased (points on the diagonal) and exactly signed (divisible right of
    zero, floor+1-channel left).

(d) THE ARC IS A SQUARE ROOT OF THE CHANNEL COUNT.  Per-frontier R^2 of the
    cumulative arc fitted as affine in sqrt(N_eff), against the same fit in
    N_eff and in S, on the 90 archive frontiers with a readable arc.

Reads  results/table_fourpoint_calibration.csv, table_fourpoint_summary.csv,
       table_arc_depth.csv   (scripts 103, 104)
Writes figures/fig_tower.pdf
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
    
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 220, "lines.linewidth": 1.2,
})
INK = "#1b1b1b"
C = ["#2f6f9f", "#8c3f6b", "#3f8c62", "#b07d2b"]


def moments(nu, x, K=7):
    return np.array([sum(w * c ** k * np.exp(-c * x) for w, c in nu)
                     for k in range(K)])


def panel_a(ax):
    """the ratio chain c_{k+1}/c_k, normalised to its first link."""
    clocks = [
        ("cure: one jump type", [(1.7, 0.8)], C[0], "-", "o"),
        ("three jump types", [(0.9, 0.3), (0.5, 1.1), (0.2, 4.0)], C[2], "-", "D"),
        ("Gamma clock (8 jumps)", [(0.6, 0.25 * j) for j in range(1, 9)],
         C[3], "-", "v"),
    ]
    ks = np.arange(0, 5)
    for lab, nu, col, ls, mk in clocks:
        m = moments(nu, 1.0)
        r = np.array([m[k + 1] / m[k] for k in ks])
        ax.plot(ks, np.clip(r / r[0], None, 3.05), ls, color=col, marker=mk,
                ms=3.6, label=lab, zorder=3)
    # indivisible contrast: the incumbent chart eps = f + c e^{-a x}
    f, cc, a_, x, h = 0.12, 0.88, 1.0, 1.0, 1e-3

    def Sf(t):
        return -np.log(f + cc * np.exp(-a_ * t))
    stencils = {1: ([-0.5, 0, 0.5], 1), 2: ([1, -2, 1], 2),
                3: ([-0.5, 1, 0, -1, 0.5], 3), 4: ([1, -4, 6, -4, 1], 4),
                5: ([-0.5, 2, -2.5, 0, 2.5, -2, 0.5], 5)}
    der = []
    for k in range(1, 6):
        w, kk = stencils[k]
        off = np.arange(len(w)) - (len(w) - 1) // 2
        der.append(sum(wi * Sf(x + o * h) for wi, o in zip(w, off)) / h ** kk)
    ck = np.array([(-1) ** (k + 1) * der[k - 1] for k in range(1, 6)])
    dS = np.log((f + cc * np.exp(-a_ * x)) / f)
    ck = np.concatenate([[dS], ck])
    r = np.array([ck[k + 1] / ck[k] for k in ks])
    ax.plot(ks, np.clip(r / r[0], -1.0, 3.2), "--", color="#a33", marker="s",
            ms=3.6, label="floor $+1$ channel (indivisible)", zorder=3)
    ax.axhline(1.0, color=INK, lw=.6, ls=":", zorder=1)
    ax.axhspan(-1.05, 1.0, color="#fdeeee", zorder=0)
    ax.text(3.45, -0.55, "forbidden:\nthe chain falls", fontsize=6.2,
            color="#a33", ha="center")
    ax.annotate("$k{=}0$: budget\n$N_{eff}\\leq\\Delta S$", xy=(0.45, 1.0),
                xytext=(2.25, 0.30), fontsize=6.3, color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", lw=.5, color=INK))
    ax.annotate("$k{=}1$: slope\n$\\sigma\\geq-1$", xy=(1.45, 1.0),
                xytext=(2.25, 1.30), fontsize=6.3, color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", lw=.5, color=INK))
    ax.set_ylim(-1.15, 3.1)
    ax.set_xlabel("rung $k$")
    ax.set_ylabel("$c_{k+1}/c_k$, normalised")
    ax.set_title("(a) one inequality at consecutive $k$")
    ax.set_xticks(ks)
    ax.legend(frameon=False, loc="upper left", ncol=1,
              borderpad=0.1, labelspacing=0.25)


def panel_b(ax):
    labels = ["$m=e^{x}-1$", "$m=e^{2x}-1$", "drift $+$ jumps"]
    comp = [3, 3, 2]
    ax_ = [5, 5, 2]
    y = np.arange(len(labels))
    ax.axhspan(-0.6, len(labels) - 0.4, xmin=0, xmax=(2.5 - 0.5) / (7 - 0.5),
               color="#eceff3", zorder=0)
    ax.barh(y - 0.17, comp, height=0.3, color=C[1], label="composition law fails")
    ax.barh(y + 0.17, ax_, height=0.3, color=C[0], label="axiom fails")
    for i, (a, b) in enumerate(zip(comp, ax_)):
        ax.text(a + .08, i - .17, str(a), va="center", fontsize=6.5, color=C[1])
        ax.text(b + .08, i + .17, str(b), va="center", fontsize=6.5, color=C[0])
    ax.text(1.45, len(labels) - 0.55, "resolved by\na published window",
            fontsize=6.2, color="#555", ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 7)
    ax.set_xlabel("first violating derivative order")
    ax.set_title("(b) which hypothesis breaks first")
    ax.legend(frameon=False, loc="lower right")


def panel_c(ax, cal, summ):
    for _, r in cal.iterrows():
        div = bool(r.divisible)
        onesided = "atoms in g" in r.truth
        col = C[2] if div else ("#999" if onesided else "#a33")
        ax.scatter(r.T_exact, r.T_sim_median, s=26, color=col, zorder=3,
                   marker=("o" if div else ("^" if onesided else "s")),
                   edgecolor="white", linewidth=.5)
    lim = (-0.19, 0.58)
    ax.plot(lim, lim, color=INK, lw=.7, ls="--", zorder=1)
    ax.axvline(0, color=INK, lw=.6, ls=":")
    ax.axhline(0, color=INK, lw=.6, ls=":")
    for _, r in summ.iterrows():
        ax.errorbar(r.T_obs, r.T_obs, xerr=r.T_se, yerr=r.T_se, fmt="*",
                    ms=11, color=C[3], ecolor=C[3], elinewidth=.8, capsize=1.5,
                    zorder=4)
        ax.annotate(r.family, (r.T_obs, r.T_obs), textcoords="offset points",
                    xytext=(6, -8), fontsize=6.2, color=C[3])
    ax.text(-0.17, 0.50, "refuted", fontsize=6.4, color="#a33")
    ax.text(0.26, -0.14, "not refuted", fontsize=6.4, color=C[2])
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("$T$ analytic")
    ax.set_ylabel("$T$ median, $4000$ draws at $10^4$ items")
    ax.set_title("(c) the four-point test is unbiased and signed")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=C[2], ms=5, label="divisible"),
        Line2D([], [], marker="s", ls="", color="#a33", ms=5,
               label="floor $+1$ channel"),
        Line2D([], [], marker="^", ls="", color="#999", ms=5,
               label="indivisible, invisible"),
        Line2D([], [], marker="*", ls="", color=C[3], ms=8, label="panel")],
        frameon=False, loc="upper left")


def panel_d(ax, arc):
    a = arc.dropna(subset=["r2_sqrtN", "r2_N", "r2_S"])
    b = np.linspace(0.5, 1.0, 26)
    ax.hist(np.clip(a.r2_sqrtN, 0.5, 1), bins=b, color=C[0], alpha=.85,
            label="affine in $\\sqrt{N_{eff}}$")
    ax.hist(np.clip(a.r2_N, 0.5, 1), bins=b, histtype="step", color=C[1],
            lw=1.1, label="affine in $N_{eff}$")
    ax.hist(np.clip(a.r2_S, 0.5, 1), bins=b, histtype="step", color=C[3],
            lw=1.1, ls="--", label="affine in $S$")
    ax.axvline(float(a.r2_sqrtN.median()), color=C[0], lw=.8, ls=":")
    ax.set_xlabel("$R^2$ of the cumulative arc")
    ax.set_ylabel("frontiers")
    ax.set_title(f"(d) arc $\\propto \\sqrt{{N_{{eff}}}}$: "
                 f"{int(a.sqrtN_wins.sum())}/{len(a)}, median "
                 f"$R^2={a.r2_sqrtN.median():.4f}$")
    ax.legend(frameon=False, loc="upper left")


def main():
    cal = pd.read_csv(os.path.join(RES, "table_fourpoint_calibration.csv"))
    summ = pd.read_csv(os.path.join(RES, "table_fourpoint_summary.csv"))
    arc = pd.read_csv(os.path.join(RES, "table_arc_depth.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.4))
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0], cal, summ)
    panel_d(axes[1, 1], arc)
    fig.tight_layout(pad=1.1)
    out = os.path.join(FIG, "fig_tower.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

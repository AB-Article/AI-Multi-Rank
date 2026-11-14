#!/usr/bin/env python3
"""The channel-depth plane (the 'plane' figure of paper.tex).

(a) THE PLANE.  N_eff = s^2/kappa against the depth dS = ln(eps/eps_inf).  The
    axiom confines every completely monotone frontier below e^{dS} - 1; the
    hypothesis confines every divisible one below that curve's own TANGENT AT THE
    ORIGIN, dS.  The two bounding solutions are charts: the two-atom chart IS the
    axiom wall and the cure chart IS the hypothesis wall (Prop. 'tangent'), and
    by Prop. 'plane' each is the unique frontier saturating its wall.  Every
    constant-sigma chart is a straight RAY of slope |sigma| <= 1 (Prop.
    'pencil'), so the chart zoo is a pencil and the floor is the ray's own
    intercept.  The operating height N_eff = 1/t is a horizontal line; where it
    meets a wall is the recommended depth, ln(1+1/t) against 1/t -- one bit
    against one nat at grain one.

(b) WHY THE SECOND-ORDER TEST DIES AT THE FLOOR.  The vertical gap between the
    walls, (e^{dS}-1) - dS, closes quadratically at the origin.  The floor is the
    origin.  So a test built on that gap is powerless at the deep end BY THEOREM,
    which is what the archive measured as a 0.0012-nat band and filed as a
    disappointment.

(c) THE ARC IS THE SAME CURVE.  L = int dS / sqrt(N_eff), so the walls give two
    sharp lower bounds: 2 arccos sqrt(eps_inf) for every floored frontier,
    rising to pi, and 2 sqrt(-ln eps_inf) for a divisible one.  They cross at
    lambda = pi^2/4.  With the covering law this is a lower bound on the number
    of operating points a benchmark needs -- 4.65 at a 1 % floor, 6.79 if
    divisible -- so benchmark design is itself a divisibility discriminator.

(d) THE CORPUS ON THE PLANE.  The 117 frontiers' own measured curves, from three
    accuracies per point with the spacing cancelling (eq. 'nhat').  They are
    noise-dominated: median straightness R^2 = 0.22, and against a design-matched
    null no statistic read from them has power (scripts 129-130).  The panel is
    the honest picture of what published single-run accuracies support.

Writes figures/fig_plane.pdf.
"""
from __future__ import annotations

import os
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 6.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "figure.dpi": 160,
})
GREY, INK = "0.55", "0.15"


def panel_a(ax):
    v = np.linspace(1e-4, 3.0, 400)
    ax.plot(v, np.exp(v) - 1, color=INK, lw=1.6,
            label=r"axiom wall $e^{\Delta S}-1$ $=$ two atoms")
    ax.plot(v, v, color="C3", lw=1.6, ls="--",
            label=r"hypothesis wall $\Delta S$ $=$ cure chart")
    for sig, c in ((0.6, "C0"), (0.3, "C2"), (0.1, "C4")):
        ax.plot(v, sig * v, color=c, lw=0.9, alpha=0.9,
                label=rf"ray $|\sigma|={sig}$")
    for t, lbl in ((1.0, r"$t=1$"), (3.0, r"$t=3$")):
        ax.axhline(1.0 / t, color=GREY, lw=0.7, ls=":")
        ax.annotate(rf"operate at $N_{{\rm eff}}=1/t$ ({lbl})", (2.95, 1.0 / t),
                    ha="right", va="bottom", fontsize=6, color="0.35")
    ax.plot([np.log(2)], [1.0], "o", ms=4, color=INK, zorder=5)
    ax.plot([1.0], [1.0], "o", ms=4, color="C3", zorder=5)
    ax.annotate("one bit", (np.log(2), 1.0), xytext=(-14, 10),
                textcoords="offset points", fontsize=6.4, ha="right", color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    ax.annotate("one nat", (1.0, 1.0), xytext=(16, -12),
                textcoords="offset points", fontsize=6.4, color="C3",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="C3"))
    ax.set_xlim(0, 3); ax.set_ylim(0, 3)
    ax.set_xlabel(r"depth remaining  $\Delta S=\ln(\varepsilon/\varepsilon_\infty)$  (nats)")
    ax.set_ylabel(r"channels remaining  $N_{\rm eff}=s^2/\kappa$")
    ax.set_title("(a) the plane: the two walls, and the pencil between them")
    ax.legend(frameon=False, loc="upper left", handlelength=1.5)


def panel_b(ax):
    v = np.linspace(0, 2.0, 400)
    gap = (np.exp(v) - 1) - v
    ax.plot(v, gap, color=INK, lw=1.5)
    ax.plot(v, v ** 2 / 2, color="C3", lw=0.9, ls="--",
            label=r"$\Delta S^2/2$: quadratic at the floor")
    ax.axhline(0.20, color=GREY, lw=0.7, ls=":")
    ax.axhline(0.094, color=GREY, lw=0.7, ls=":")
    ax.annotate("count-axis floor bracket\n0.20 / 0.094 nats", (1.95, 0.21),
                ha="right", va="bottom", fontsize=6.2, color="0.35")
    ax.set_xlim(0, 2); ax.set_ylim(0, 1.2)
    ax.set_xlabel(r"depth remaining $\Delta S$ (nats)")
    ax.set_ylabel("gap between the walls (nats)")
    ax.set_title("(b) the gap closes quadratically at the floor")
    ax.legend(frameon=False, loc="upper left")


def panel_c(ax):
    lam = np.linspace(0.05, 9.0, 400)
    ax.plot(lam, 2 * np.arccos(np.exp(-lam / 2)), color=INK, lw=1.5,
            label=r"$2\arccos\sqrt{\varepsilon_\infty}$ (any floored frontier)")
    ax.plot(lam, 2 * np.sqrt(lam), color="C3", lw=1.5, ls="--",
            label=r"$2\sqrt{\lambda}$ (divisible)")
    ax.axhline(np.pi, color=GREY, lw=0.7, ls=":")
    ax.annotate(r"$\pi$", (8.9, np.pi), xytext=(0, 3), textcoords="offset points",
                ha="right", fontsize=7, color="0.35")
    ax.axvline(np.pi ** 2 / 4, color=GREY, lw=0.7, ls=":")
    ax.annotate(r"$\lambda=\pi^2/4$", (np.pi ** 2 / 4, 5.6), xytext=(3, 0),
                textcoords="offset points", fontsize=6.4, color="0.35")
    sec = ax.secondary_yaxis("right", functions=(lambda L: L / (2 * np.sqrt(2 * 0.05)),
                                                 lambda K: K * 2 * np.sqrt(2 * 0.05)))
    sec.set_ylabel(r"operating points needed at $\varrho=0.05$", fontsize=7.4)
    sec.tick_params(labelsize=7)
    ax.set_xlim(0, 9); ax.set_ylim(0, 6)
    ax.set_xlabel(r"depth of the frontier  $\lambda=-\ln\varepsilon_\infty$  (nats)")
    ax.set_ylabel(r"minimum arc $L=\int \mathrm{d}S/\sqrt{N_{\rm eff}}$")
    ax.set_title(r"(c) the arc is the same curve, and $\pi$ its limit")
    ax.legend(frameon=False, loc="lower right")


def panel_d(ax):
    p = os.path.join(RESULTS, "table_plane_frontiers.csv")
    if not os.path.exists(p):
        ax.text(.5, .5, "run 129_plane_corpus.py", ha="center", transform=ax.transAxes)
        return
    df = pd.read_csv(p)
    d = df.dropna(subset=["sigma_plane", "line_r2"])
    sc = ax.scatter(d.line_r2, d.sigma_plane.clip(-6, 2), s=9,
                    c=np.log10(d.sigma_noise.clip(1e-4, 1)), cmap="viridis",
                    alpha=0.85, linewidths=0)
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}$ residual scale (nats)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.axhline(-1, color="C3", lw=1.0, ls="--")
    ax.annotate(r"$\sigma=-1$: the hypothesis' own slope", (0.02, -1),
                xytext=(0, 4), textcoords="offset points", fontsize=6.4, color="C3")
    ax.axvline(0.95, color=GREY, lw=0.7, ls=":")
    ax.annotate(f"straight at 0.95: {int(d.line_r2.gt(0.95).sum())} of {len(d)}",
                (0.02, -5.5), fontsize=6.4, color="0.35")
    ax.set_xlim(0, 1.02); ax.set_ylim(-6, 2)
    ax.set_xlabel(r"straightness of the measured curve, $R^2$")
    ax.set_ylabel(r"measured slope $\sigma$")
    ax.set_title(f"(d) the corpus: median $R^2={d.line_r2.median():.2f}$, "
                 "no power against a null")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.5))
    panel_a(axes[0, 0]); panel_b(axes[0, 1])
    panel_c(axes[1, 0]); panel_d(axes[1, 1])
    for ax in axes.ravel():
        ax.tick_params(length=2.5, width=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.6)
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "fig_plane.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"  -> figures/fig_plane.pdf")


if __name__ == "__main__":
    main()

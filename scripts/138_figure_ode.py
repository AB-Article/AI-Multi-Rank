#!/usr/bin/env python3
"""138_figure_ode.py -- the ODE-first figure (the 'odefirst' figure of paper.tex).

(a) TWO RUNGS, TWO SOLUTIONS.  The theory's primitive is one function N of the
    depth still available, with N(0) = 0 at the floor.  The axiom says the
    channel count may grow with depth no faster than Ndot = N + 1; the
    hypothesis says no faster than Ndot = 1.  Integrated under the same initial
    condition those two inequalities ARE the two charts -- e^{dS} - 1 and dS --
    so the tangency at the origin is a property of one initial-value problem,
    not a coincidence of two propositions.  The shaded wedge between them is
    the whole divisibility question.

(b) THE FACTOR TWO IN THE TEST.  The statistic the archive uses, c_3 >= 0, is
    exactly Ndot <= 2; the hypothesis's own first rung is Ndot <= 1.  The band
    between the two slopes is what the test in use cannot see, and D_1 >= 0 --
    the same four accuracies, one more ratio -- closes it.

(c) THE GAUGE-FREE READING.  With phi = 2 arctan sqrt(N) and dL = d(dS)/sqrt(N),
    the regret is -ln sin phi identically and the axiom is phi' <= 1: the angle
    advances no faster than the arc.  A complete floored frontier sweeps phi
    from pi to 0, so L >= pi -- Law III's constant as a Lipschitz bound -- and
    the rule is the right angle, where the regret is zero.

(d) WHAT THE STRONGER RUNG IS WORTH.  Clean-chart power of the two statistics
    against the two-atom truth at matched level (script 136), with the archive's
    own per-model noise marked.  The gain is real and modest; on the corpus both
    statistics remain at their nominal level, which is script 131's conclusion
    and not changed by a better test.

Writes figures/fig_odefirst.pdf.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
sys.path.insert(0, HERE)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "axes.titlesize": 8.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "figure.dpi": 160,
                     "axes.spines.top": False, "axes.spines.right": False})
C_AX, C_HY, C_MID, C_PT = "#1f3b73", "#b3452a", "#6b8f3a", "#444444"


def panel_a(ax):
    d = np.linspace(0, 2.4, 400)
    ax.fill_between(d, d, np.e ** d - 1, color=C_MID, alpha=0.14, lw=0,
                    label="the divisibility wedge")
    ax.plot(d, np.e ** d - 1, color=C_AX, lw=1.6,
            label=r"$\dot N=N+1\;\Rightarrow\;N=e^{\Delta S}-1$  (two atoms)")
    ax.plot(d, d, color=C_HY, lw=1.6,
            label=r"$\dot N=1\;\Rightarrow\;N=\Delta S$  (cure chart)")
    for sig in (0.3, 0.6, 0.9):
        ax.plot(d, sig * d, color=C_HY, lw=0.7, ls=":", alpha=0.8)
    ax.annotate(r"rays $N=|\sigma|\Delta S$", xy=(2.05, 0.6 * 2.05),
                xytext=(1.30, 0.30), fontsize=6.5, color=C_HY,
                arrowprops=dict(arrowstyle="-", color=C_HY, lw=0.5))
    ax.plot([0], [0], marker="o", ms=4, color="k", zorder=5)
    ax.annotate("the floor:\n" + r"$N(0)=0$", xy=(0, 0), xytext=(0.12, 1.55),
                fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.6, color="k"))
    ax.set_xlim(0, 2.4); ax.set_ylim(0, 4.2)
    ax.set_xlabel(r"depth still available  $\Delta S=\ln(\varepsilon/\varepsilon_\infty)$")
    ax.set_ylabel(r"channels still open  $N=s^2/\kappa$")
    ax.set_title("(a) two rungs, two initial-value problems")
    ax.legend(loc="upper left", frameon=False)


def panel_b(ax):
    d = np.linspace(0, 2.4, 400)
    ax.fill_between(d, d, 2 * d, color="#999999", alpha=0.18, lw=0,
                    label=r"what $c_3\geq0$ cannot see")
    ax.plot(d, 2 * d, color=C_PT, lw=1.4, ls="--",
            label=r"$\dot N\leq2\Leftrightarrow c_3\geq0$  (the test in use)")
    ax.plot(d, d, color=C_HY, lw=1.6,
            label=r"$\dot N\leq1\Leftrightarrow D_1\geq0$  (the hypothesis's own rung)")
    ax.plot(d, np.e ** d - 1, color=C_AX, lw=1.0, alpha=0.6,
            label=r"$\dot N\leq N+1$  (the axiom)")
    ax.set_xlim(0, 2.4); ax.set_ylim(0, 4.2)
    ax.set_xlabel(r"$\Delta S$")
    ax.set_ylabel(r"$N$")
    ax.set_title("(b) the statistic in use is twice as loose")
    ax.legend(loc="upper left", frameon=False)


def panel_c(ax):
    phi = np.linspace(0.02, np.pi - 0.02, 400)
    ax.plot(phi, -np.log(np.sin(phi)), color=C_MID, lw=1.6,
            label=r"$R=-\ln\sin\varphi$  (an identity)")
    ax.axvline(np.pi / 2, color="k", lw=0.7, ls=":")
    ax.annotate("the rule:\n" + r"$\varphi=\pi/2$, $R=0$", xy=(np.pi / 2, 0),
                xytext=(np.pi / 2 + 0.18, 0.95), fontsize=6.5,
                arrowprops=dict(arrowstyle="->", lw=0.6, color="k"))
    for t, lab in ((2.0, r"$t=2$"), (7.0, r"$t=7$")):
        p = 2 * np.arctan(1 / np.sqrt(t))
        ax.plot([p], [-np.log(np.sin(p))], marker="o", ms=3.5, color=C_AX)
        ax.annotate(lab, xy=(p, -np.log(np.sin(p))), xytext=(p - 0.33, -np.log(np.sin(p)) + 0.3),
                    fontsize=6.5, color=C_AX)
    ax.set_xlim(0, np.pi); ax.set_ylim(0, 2.6)
    ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax.set_xlabel(r"angle  $\varphi=2\arctan\sqrt{N}$   (1-Lipschitz in the arc $L$)")
    ax.set_ylabel("regret  $R$  (nats)")
    ax.set_title(r"(c) gauge-free: $\varphi'\leq1$, and $L\geq\pi$")
    ax.legend(loc="upper center", frameon=False)


def panel_d(ax):
    p = os.path.join(RES, "table_ode_power_charts.csv")
    if not os.path.exists(p):
        ax.text(.5, .5, "run script 136 first", ha="center", va="center")
        ax.set_axis_off(); return
    df = pd.read_csv(p)
    for k, lab, col, mk in (("old", r"$c_3\geq0$  ($\dot N\leq2$)", C_PT, "s"),
                            ("new", r"$D_1\geq0$  ($\dot N\leq1$)", C_HY, "o")):
        s = df[df.statistic == k].sort_values("sigma")
        ax.plot(s.sigma * 1000, s.power, marker=mk, ms=3.5, lw=1.3, color=col,
                label=lab)
    ax.axhline(0.8, color="k", lw=0.6, ls=":")
    ax.axvline(20.0, color=C_AX, lw=0.7, ls="--")
    ax.annotate("the archive's\nown noise", xy=(20, 0.35), xytext=(23, 0.22),
                fontsize=6.5, color=C_AX)
    ax.set_xscale("log")
    ax.set_xticks([2.5, 5, 10, 20, 40])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("per-model noise (milli-nats)")
    ax.set_ylabel("power against the two-atom truth")
    ax.set_ylim(0.4, 1.03)
    ax.set_title("(d) what the stronger rung is worth")
    ax.legend(loc="lower left", frameon=False)


def run():
    os.makedirs(FIG, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.5))
    panel_a(axes[0, 0]); panel_b(axes[0, 1])
    panel_c(axes[1, 0]); panel_d(axes[1, 1])
    fig.tight_layout(pad=0.9)
    out = os.path.join(FIG, "fig_odefirst.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"  -> figures/fig_odefirst.pdf")


if __name__ == "__main__":
    run()

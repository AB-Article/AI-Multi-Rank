#!/usr/bin/env python3
"""The composition law (the 'composition' figure of paper.tex).

(a) The two walls, in the only coordinate that matters.  The axiom bounds the
    channel slope by sigma >= -(1+N_eff) and is tight on the two-atom chart;
    the composition law bounds it by the universal constant sigma >= -1 and is
    tight on the cure chart.  The gap between the walls is exactly N_eff, which
    is why the test of the law has power only where channels are many -- and the
    archive's median N_eff is 0.47.
(b) Channels against nats remaining.  N_eff <= DeltaS under the composition law
    (equality: the cure chart) against N_eff <= e^DeltaS - 1 under the axiom
    alone (equality: the two-atom chart).  At the archive's median span of 5 nats
    the two ceilings differ by a factor of 30.
(c) The floor bound.  eps_inf <= eps e^{-N_eff} is tighter than
    eps_inf <= eps/(1+N_eff) at every point by e^{-N}(1+N); measured tightening
    over the archive, by position in the window.
(d) The cost of ignorance.  On the gapped geometry it is exactly one bit for any
    frontier; on the compositional geometry it is depth-dependent, minimised at
    lam* = 5.434 nats (the root of A = 2 arctan A) at 0.465 bits.

Writes figures/fig_composition.pdf.
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                            "axes.titlesize": 8, "axes.labelsize": 8,
                            "legend.fontsize": 6.0,
                            "xtick.labelsize": 7, "ytick.labelsize": 7})
BLUE, RED, GREY, GREEN = "#3b6ea5", "#c0504d", "#888888", "#4f8a5b"


def rd(name):
    p = os.path.join(RESULTS, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(7.3, 2.0))

    # ------------------------------------------------------------ panel a ---
    a0 = ax[0]
    N = np.linspace(0.02, 5.0, 400)
    a0.fill_between(N, -(1 + N), -1, color=RED, alpha=.12, lw=0)
    a0.plot(N, -(1 + N), color=RED, lw=1.5,
            label=r"axiom: $\sigma\geq-(1{+}N_{\rm eff})$")
    a0.axhline(-1, color=BLUE, lw=1.5, label=r"composition: $\sigma\geq-1$")
    a0.axhline(0, color=GREEN, lw=1.0, ls="--", label=r"$\Gamma(k)$ clock")
    for b, ls in ((0.25, ":"), (0.5, "-."),):
        a0.axhline(b / (1 - b), color=GREY, lw=.9, ls=ls,
                   label=rf"stretched, $\beta={b}$")
    a0.plot([1], [-1], "o", color=BLUE, ms=3.5, zorder=5)
    a0.annotate("cure chart", (1.0, -1.0), textcoords="offset points",
                xytext=(6, -11), fontsize=5.8, color=BLUE)
    a0.annotate("two-atom chart", (3.0, -4.0), textcoords="offset points",
                xytext=(5, 1), fontsize=5.8, color=RED)
    B = rd("table_budget_bound.csv")
    if B is not None:
        a0.axvline(float(B.N_eff_med.median()), color="k", lw=.7, ls=(0, (1, 2)))
        a0.annotate("archive\nmedian", (float(B.N_eff_med.median()), -4.9),
                    textcoords="offset points", xytext=(3, -2), fontsize=5.5)
    a0.set_xlabel(r"$N_{\rm eff}=s^2/\kappa$")
    a0.set_ylabel(r"channel slope $\sigma=\mathrm{d}N_{\rm eff}/\mathrm{d}S$")
    a0.set_ylim(-6.2, 5.6); a0.set_xlim(0, 5)
    a0.set_title("(a) one wall is universal")
    a0.legend(loc="upper center", frameon=False, handlelength=1.4, ncol=1,
              borderaxespad=0.1, labelspacing=0.25)

    # ------------------------------------------------------------ panel b ---
    a1 = ax[1]
    D = np.linspace(0.05, 7.0, 300)
    a1.semilogy(D, np.exp(D) - 1, color=RED, lw=1.5,
                label=r"axiom: $N_{\rm eff}\leq e^{\Delta S}-1$")
    a1.semilogy(D, D, color=BLUE, lw=1.5,
                label=r"composition: $N_{\rm eff}\leq\Delta S$")
    a1.fill_between(D, D, np.exp(D) - 1, color=RED, alpha=.10, lw=0)
    for d0 in (5.0,):
        a1.plot([d0, d0], [d0, np.exp(d0) - 1], color="k", lw=.8)
        a1.annotate(rf"$\times{(np.exp(d0)-1)/d0:.0f}$", (d0, np.sqrt(d0 * (np.exp(d0) - 1))),
                    textcoords="offset points", xytext=(4, -3), fontsize=6)
    a1.set_xlabel(r"nats remaining $\Delta S=\ln(\varepsilon/\varepsilon_\infty)$")
    a1.set_ylabel(r"admissible $N_{\rm eff}$")
    a1.set_title("(b) a ceiling on channels")
    a1.legend(loc="upper left", frameon=False, handlelength=1.5)

    # ------------------------------------------------------------ panel c ---
    a2 = ax[2]
    Ng = np.linspace(0.02, 8, 300)
    a2.semilogy(Ng, np.exp(-Ng) * (1 + Ng), color=BLUE, lw=1.5)
    a2.set_xlabel(r"$N_{\rm eff}$ at the reading point")
    a2.set_ylabel(r"$\varepsilon_\infty$ bound ratio, new$/$old")
    a2.set_title("(c) the floor bound tightens")
    a2.axhline(1, color=GREY, lw=.8, ls=":")
    if B is not None:
        g = B[np.isfinite(B.tighten_B1_over_B2) & (B.tighten_B1_over_B2 > 0)]
        if len(g):
            a2.plot(g.N_eff_max.clip(0, 8), 1.0 / g.tighten_B1_over_B2.clip(1, 1e12),
                    ".", color=RED, ms=2.5, alpha=.55, label="archive frontiers")
            a2.legend(loc="lower left", frameon=False)
    a2.set_ylim(1e-4, 2)

    # ------------------------------------------------------------ panel d ---
    a3 = ax[3]
    lam = np.linspace(0.35, 60, 600)
    A = np.sqrt(lam)
    ER = np.log(np.cosh(np.log(A))) - 1 + 2 * np.arctan(A) / A
    a3.plot(lam, ER / np.log(2), color=BLUE, lw=1.5, label="compositional, floored")
    a3.axhline(1.0, color=RED, lw=1.4, label="gapped (two-atom): 1 bit")
    a3.axhline(np.euler_gamma / np.log(2) + 1, color=RED, lw=.8, ls=":",
               label=r"gapped ceiling $\gamma+\ln2$")
    Ast = 2.3311223704144226
    a3.plot([Ast ** 2], [np.log(np.cosh(np.log(Ast))) / np.log(2)], "o", color=BLUE, ms=3.5)
    a3.annotate(r"$\lambda^\star=5.434$" "\n" r"$0.465$ bits",
                (Ast ** 2, np.log(np.cosh(np.log(Ast))) / np.log(2)),
                textcoords="offset points", xytext=(6, -16), fontsize=5.8, color=BLUE)
    a3.set_xscale("log")
    a3.set_xlabel(r"depth $\lambda=-\ln\varepsilon_\infty$ (nats)")
    a3.set_ylabel(r"$\mathbb{E}[R]$ (bits)")
    a3.set_title("(d) the cost of ignorance")
    a3.set_ylim(0, 2.1)
    a3.legend(loc="upper left", frameon=False, handlelength=1.4)

    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.45)
    out = os.path.join(FIG, "fig_composition.pdf")
    fig.savefig(out); print("  ->", out)


if __name__ == "__main__":
    main()

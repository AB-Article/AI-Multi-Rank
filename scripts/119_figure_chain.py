#!/usr/bin/env python3
"""FIGURE: THE ATTEMPT CHAIN.  -> figures/fig_chain.pdf
   (a) the hypothesis read on the DEPTH frontier: Delta S is a power (power-floor),
       an exponential (cure) or a LOGARITHM (incumbent) -- and only the first two
       are completely monotone
   (b) the incumbent chart in closed form: c_3 = alpha^3 psi(1-psi)/(1+psi)^3,
       negative exactly where eps > 2 eps_inf, extremal at psi = 2+sqrt3
   (c) pi is the compactified gauge orbit: d phi = (1/2)sech(u/2) du, total area pi
   (d) the rule with its input: the implied attempt budget 1/N_eff per frontier
Reads results/table_retry_budget.csv.
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
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})
C = {"cure": "#1b4b8f", "pf": "#c2521d", "lumpy": "#b0122e", "grey": "#8a8a8a",
     "arc": "#2c7a52"}

fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0))

# ---------------------------------------------------------------- panel (a) --
a = ax[0, 0]
x = np.linspace(0.35, 6.0, 500)
dS_cure = 4.0 * np.exp(-0.8 * x)                      # pure exponential
dS_pf = 2.2 * x ** (-0.45)                            # pure power law
dS_inc = np.log1p(30.0 * np.exp(-0.8 * x))            # a LOGARITHM
a.plot(x, dS_cure, color=C["cure"], lw=1.6, label=r"cure: $\Lambda e^{-cx}$")
a.plot(x, dS_pf, color=C["pf"], lw=1.6, label=r"power-floor: $Cx^{\beta}$, $\beta<0$")
a.plot(x, dS_inc, color=C["lumpy"], lw=1.6, ls="--",
       label=r"incumbent: $\ln(1+\psi)$")
a.set_yscale("log")
a.set_xlabel(r"log-resource $x$")
a.set_ylabel(r"remaining depth $\Delta S=\ln(\varepsilon/\varepsilon_\infty)$")
a.set_title(r"(a) the hypothesis is ``$\Delta S$ is completely monotone''")
a.legend(frameon=False, loc="lower left")
a.text(0.40, 0.93, "completely monotone", transform=a.transAxes,
       color=C["cure"], fontsize=7)
a.text(0.40, 0.85, "not, at any point", transform=a.transAxes,
       color=C["lumpy"], fontsize=7)

# ---------------------------------------------------------------- panel (b) --
b = ax[0, 1]
psi = np.logspace(-2, 2.3, 600)
c3 = psi * (1 - psi) / (1 + psi) ** 3
b.axhline(0.0, color="k", lw=0.6)
b.plot(psi, c3, color=C["lumpy"], lw=1.6)
b.axvline(1.0, color=C["grey"], lw=0.9, ls=":")
b.set_xscale("log")
pstar = 2 + np.sqrt(3)
b.plot([pstar], [pstar * (1 - pstar) / (1 + pstar) ** 3], "o", ms=4,
       color=C["lumpy"])
b.annotate(r"$\psi=2+\sqrt{3}$" "\n" r"share $0.211$",
           xy=(pstar, pstar * (1 - pstar) / (1 + pstar) ** 3),
           xytext=(9.0, -0.055), fontsize=7, color=C["lumpy"],
           arrowprops=dict(arrowstyle="-", lw=0.7, color=C["lumpy"]))
b.text(1.15, 0.055, r"$\varepsilon=2\varepsilon_\infty$", fontsize=7,
       color=C["grey"])
b.text(0.013, 0.055, "passes", fontsize=7, color=C["grey"])
b.text(20, 0.055, "refuted", fontsize=7, color=C["lumpy"])
b.set_xlabel(r"$\psi=(\varepsilon-\varepsilon_\infty)/\varepsilon_\infty$")
b.set_ylabel(r"$c_3/\alpha^3$")
b.set_title(r"(b) the incumbent chart, in closed form")

# ---------------------------------------------------------------- panel (c) --
c = ax[1, 0]
u = np.linspace(-12, 12, 800)
dphi = 0.5 / np.cosh(u / 2)
c.fill_between(u, 0, dphi, color=C["arc"], alpha=0.18)
c.plot(u, dphi, color=C["arc"], lw=1.6)
c.plot(u, 1.0 / np.cosh(u / 2), color=C["grey"], lw=1.0, ls="--",
       label=r"$\sin\varphi=\mathrm{sech}(u/2)$")
c.set_xlabel(r"gauge coordinate $u=\ln N_{\mathrm{eff}}$  (the error gauge translates $u$)")
c.set_ylabel(r"$\mathrm{d}\varphi/\mathrm{d}u$")
c.set_title(r"(c) $\pi$ is the area under a secant")
c.text(0.05, 0.62, r"$\int\frac{1}{2}\mathrm{sech}\frac{u}{2}\,\mathrm{d}u=\pi$", transform=c.transAxes, fontsize=8, color=C["arc"])
c.axvline(0.0, color="k", lw=0.6, ls=":")
c.annotate("", xy=(np.log(8), 0.085), xytext=(0.0, 0.085),
           arrowprops=dict(arrowstyle="->", lw=0.8, color="k"))
c.text(0.9, 0.095, r"budget $\times t$", fontsize=7)
c.legend(frameon=False, loc="upper right")

# ---------------------------------------------------------------- panel (d) --
d = ax[1, 1]
f = os.path.join(RES, "table_retry_budget.csv")
if os.path.exists(f):
    T = pd.read_csv(f)
    bud = T["budget_median_window"].dropna()
    bud = bud[(bud > 0) & np.isfinite(bud)]
    bins = np.logspace(np.log10(max(bud.min(), 1e-2)), np.log10(bud.max()), 26)
    d.hist(bud, bins=bins, color=C["cure"], alpha=0.75, edgecolor="white", lw=0.4)
    d.axvline(bud.median(), color=C["lumpy"], lw=1.4)
    d.text(bud.median() * 1.15, d.get_ylim()[1] * 0.82,
           f"median $t={bud.median():.2f}$", fontsize=7, color=C["lumpy"])
    d.axvline(1.0, color=C["grey"], lw=0.9, ls=":")
    d.text(1.05, d.get_ylim()[1] * 0.55, r"$t=1$ (the paper's rule)", fontsize=7,
           color=C["grey"], rotation=90, va="top")
    d.set_xscale("log")
    d.set_xlabel(r"implied attempt budget $t=1/N_{\mathrm{eff}}$")
    d.set_ylabel("frontiers")
else:
    d.text(0.5, 0.5, "run 117_retry_budget.py", ha="center", transform=d.transAxes)
d.set_title(r"(d) the rule with its input: $N_{\mathrm{eff}}=1/t$")

fig.tight_layout(pad=0.6)
out = os.path.join(FIG, "fig_chain.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"  -> figures/fig_chain.pdf")

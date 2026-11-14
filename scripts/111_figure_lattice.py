#!/usr/bin/env python3
"""FIGURE: THE HORIZON LATTICE.  -> figures/fig_lattice.pdf
   (a) the horizon array: monotone in resource and in level, and the forbidden
       region a lumpy chart falls into
   (b) the chart line: one coordinate sigma >= -1, with the empty cell filled
   (c) the compute-axis four-point test: exactly unbiased on the real designs
   (d) the power-floor chart held out against every other chart in the paper
Reads results/table_dd3_calibration.csv and table_powerfloor_summary.csv.
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
sys.path.insert(0, HERE)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})
C = {"cure": "#1b4b8f", "pf": "#c2521d", "stretch": "#2c7a52",
     "lomax": "#7a5bbd", "lumpy": "#b0122e", "grey": "#8a8a8a"}


# ----------------------------------------------------------------- charts ----
def jets_of(S, x, k):
    """c_k = (-1)^{k+1} S^{(k)} by central differences on a fine grid."""
    h = 1e-4
    out = []
    for j in range(k + 2):
        pass
    return out


def horizons(name, x):
    """analytic h_1, h_2 for the named charts."""
    if name == "cure":
        lam, c = 2.0, 0.8
        s = lam * c * np.exp(-c * x)
        kap = lam * c ** 2 * np.exp(-c * x)
        k3 = lam * c ** 3 * np.exp(-c * x)
    elif name == "pf":
        b, C_, x0 = -0.6, 1.1, -0.7
        u = x - x0
        s = -C_ * b * u ** (b - 1)
        kap = C_ * b * (b - 1) * u ** (b - 2)
        k3 = -C_ * b * (b - 1) * (b - 2) * u ** (b - 3)
    elif name == "stretch":
        b, a = 0.4, 1.3
        u = np.clip(x, 1e-6, None)
        s = b * a ** b * u ** (b - 1)
        kap = -b * (b - 1) * a ** b * u ** (b - 2)
        k3 = b * (b - 1) * (b - 2) * a ** b * u ** (b - 3)
    elif name == "lumpy":
        f, a = 0.15, 0.9
        E = f + (1 - f) * np.exp(-a * x)
        r = (1 - f) * np.exp(-a * x) / E
        s = a * r
        kap = a ** 2 * r * (1 - r)
        k3 = a ** 3 * r * (1 - r) * (1 - 2 * r)
    else:
        raise ValueError(name)
    return s / kap, kap / np.abs(k3) * np.sign(k3)


def panel_a(ax):
    x = np.linspace(0.35, 4.0, 400)
    for nm, lab in [("cure", "cure  $\\sigma=-1$"),
                    ("pf", "power-floor  $\\sigma\\in(-1,0)$"),
                    ("stretch", "stretched  $\\sigma>0$")]:
        h1, h2 = horizons(nm, x)
        ax.plot(x, h1, color=C[nm], lw=1.6, label=lab)
        ax.plot(x, h2, color=C[nm], lw=0.9, ls=(0, (3, 2)), alpha=.75)
    h1, h2 = horizons("lumpy", x)
    ax.plot(x, h1, color=C["lumpy"], lw=1.6, label="two-atom  $\\sigma<-1$")
    ax.plot(x, h2, color=C["lumpy"], lw=0.9, ls=(0, (3, 2)), alpha=.75)
    ax.set_xlabel("log-resource $x$")
    ax.set_ylabel("horizon  $h_k=c_k/c_{k+1}$   (e-folds)")
    ax.set_title("(a) the horizons recede, and shorten with level", loc="left")
    ax.set_ylim(0, 4.2)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)
    ax.annotate("solid $h_1$,  dashed $h_2$", xy=(.97, .05),
                xycoords="axes fraction", ha="right", color=C["grey"], fontsize=7)
    ax.annotate("$h_1$ falling:\nthe lattice's\nforbidden direction",
                xy=(3.4, 1.65), xytext=(3.25, 3.6), color=C["lumpy"], fontsize=6.8,
                ha="center", arrowprops=dict(arrowstyle="->", color=C["lumpy"], lw=.8))


def panel_b(ax):
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlim(-1.45, 3.2)
    ax.set_ylim(-1.05, 1.15)
    ax.axvspan(-1.0, 0.0, color=C["pf"], alpha=.10, lw=0)
    ax.axvspan(-1.45, -1.0, color=C["lumpy"], alpha=.10, lw=0)
    sg = np.linspace(-1.0, 0.0, 200)
    ax.plot(sg, -sg, color=C["pf"], lw=2.0)
    ax.plot([0, 3.2], [0, 0], color=C["stretch"], lw=2.0)
    for s, y, nm, lab, dy in [(-1.0, 1.0, "cure", "cure", .09),
                              (-0.5, 0.5, "pf", "power-floor\n(new cell)", .10),
                              (0.0, 0.0, "lomax", "Lomax", .10),
                              (1.0, 0.0, "stretch", "stretched", .10)]:
        ax.plot([s], [y], "o", ms=5, color=C[nm], zorder=5)
        ax.annotate(lab, xy=(s, y), xytext=(s, y + dy), ha="center",
                    fontsize=7, color=C[nm])
    ax.annotate("lumpy\n(refuted)", xy=(-1.25, -0.6), ha="center", fontsize=7,
                color=C["lumpy"])
    ax.annotate("floor", xy=(-0.5, -0.85), ha="center", fontsize=7, color=C["pf"])
    ax.annotate("no floor", xy=(1.6, -0.85), ha="center", fontsize=7,
                color=C["stretch"])
    ax.set_xlabel("$\\sigma=\\mathrm{d}N_{\\mathrm{eff}}/\\mathrm{d}S"
                  "=D_1-1=\\beta/(1-\\beta)$")
    ax.set_ylabel("$N_{\\mathrm{eff}}/\\Delta S=|\\sigma|$")
    ax.set_title("(b) one coordinate for every chart", loc="left")


def panel_c(ax):
    f = os.path.join(RES, "table_dd3_calibration.csv")
    if not os.path.exists(f):
        ax.text(.5, .5, "run script 108", ha="center", transform=ax.transAxes)
        return
    D = pd.read_csv(f)
    g = D.groupby(["truth", "verdict"]).agg(
        a=("z_analytic", "median"), s=("z_sim_mean", "median"),
        e=("se", "median")).reset_index()
    for _, r in g.iterrows():
        col = C["stretch"] if r["verdict"] == "divisible" else C["lumpy"]
        mk = "o" if r["verdict"] == "divisible" else "s"
        ax.errorbar(r["a"], r["s"], yerr=r["e"] / np.sqrt(len(D) / 6), fmt=mk,
                    ms=5, color=col, capsize=2, lw=.9)
        dodge = {"law_f0.02": (5, 5), "law_f0.15": (5, -11), "law_f0.40": (5, 4),
                 "cure": (5, -10), "stretched": (5, 4), "powerfloor": (-6, 7)}
        ha = "right" if r["truth"] == "powerfloor" else "left"
        ax.annotate(r["truth"].replace("law_f", "two-atom $f=$"),
                    xy=(r["a"], r["s"]), xytext=dodge.get(r["truth"], (5, 4)),
                    textcoords="offset points", fontsize=6.3, color=col, ha=ha)
    lim = [min(g.a.min(), g.s.min()) - .09, max(g.a.max(), g.s.max()) + .10]
    ax.plot(lim, lim, color=C["grey"], lw=.8, ls=(0, (4, 3)))
    ax.axhline(0, color="k", lw=.6)
    ax.axvline(0, color="k", lw=.6)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("analytic $z$ (noiseless)")
    ax.set_ylabel("simulated median $z$")
    ax.set_title("(c) the four-point test is exactly unbiased", loc="left")
    bias = (g.s - g.a).abs().max()
    ax.annotate(f"max $|$bias$|={bias:.3f}$\nagainst $\\approx1$ for the\n"
                "plug-in trend of $G$", xy=(.04, .74), xycoords="axes fraction",
                fontsize=6.8, color=C["grey"])


def panel_d(ax):
    f = os.path.join(RES, "table_powerfloor_summary.csv")
    if not os.path.exists(f):
        ax.text(.5, .5, "run script 109", ha="center", transform=ax.transAxes)
        return
    S = pd.read_csv(f)
    order = ["power", "cure", "law", "stable", "powerfloor", "law_beta", "law2"]
    S = S.set_index("chart").reindex([o for o in order if o in set(S.chart)])
    lbl = {"power": "power [2]", "cure": "cure [3]", "law": "two-atom [3]",
           "stable": "stretched [3]", "powerfloor": "power-floor [4]",
           "law_beta": "two-atom+$\\beta$ [4]", "law2": "two atoms+floor [5]"}
    pos = np.arange(len(S))
    wd = 0.27
    for j, (reg, col) in enumerate([("INTERIOR", C["cure"]),
                                    ("FORWARD", C["pf"]),
                                    ("BACKWARD", C["stretch"])]):
        v = (np.exp(S[f"median_{reg}"].values) - 1) * 100
        bars = ax.bar(pos + (j - 1) * wd, v, wd, color=col, lw=0,
                      label=reg.capitalize())
        if reg == "INTERIOR":
            for b, nm in zip(bars, S.index):
                if nm == "powerfloor":
                    b.set_edgecolor("k")
                    b.set_linewidth(1.0)
    ax.set_xticks(pos)
    ax.set_xticklabels([lbl.get(i, i) for i in S.index], rotation=28,
                       ha="right", fontsize=6.5)
    ax.set_ylabel("held-out relative error, \\%")
    ax.set_title("(d) the new cell, held out on the same splits", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper center", columnspacing=1.1)
    ax.set_ylim(0, None)


def main():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.5))
    panel_a(axs[0, 0])
    panel_b(axs[0, 1])
    panel_c(axs[1, 0])
    panel_d(axs[1, 1])
    fig.tight_layout(pad=1.0, w_pad=1.8, h_pad=1.8)
    out = os.path.join(FIG, "fig_lattice.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"  -> figures/fig_lattice.pdf")


if __name__ == "__main__":
    main()

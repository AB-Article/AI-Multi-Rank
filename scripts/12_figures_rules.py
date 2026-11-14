#!/usr/bin/env python3
"""Figures for Section 10 (fit-free consequences of the single-channel law).

Reads the tables written by 11_selection_rules.py and writes
    figures/fig_rules_theory.pdf     (+ .png preview)
    figures/fig_rules_empirical.pdf  (+ .png preview)
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
    # pin the family so the figure is byte-stable across environments, and
    # keep every glyph a TrueType subset (fonttype 42) rather than a Type 3
    # bitmap.
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.2, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, GREY = "#1f5fa8", "#c0392b", "#2e8b57", "#8a8a8a"
R_BAND = 0.05
Z_BAND = 2 * np.arccosh(np.exp(R_BAND))          # |zeta| <= 0.638


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print("  wrote", os.path.join(FIG, name + ".pdf"))


# ------------------------------------------------------------------ theory ---
z = np.linspace(-4, 4, 801)
sig = 1 / (1 + np.exp(z))                           # reducible share p1
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.25))

a = ax[0]
a.axvspan(-Z_BAND, Z_BAND, color=C3, alpha=0.15, lw=0)
a.semilogy(z, 1 + np.exp(-z), color=C1)
a.axhline(2, color=GREY, lw=0.6, ls=":")
a.plot([0], [2], "o", color=C2, ms=4)
a.set_xlabel(r"normalised resource $\zeta$")
a.set_ylabel(r"$\varepsilon/\varepsilon_\infty$")
a.set_title("(a) universal curve, 0.05-nat band")
a.text(0.12, 2.25, "canonical", color=C2, fontsize=7)

a = ax[1]
a.plot(z, sig, color=C1, label=r"slope $\eta/\alpha=p_1$")
a.plot(z, 4 * sig * (1 - sig), color=C2, label=r"curvature $4|S''|/\alpha^2$")
a.axvline(0, color=GREY, lw=0.6, ls=":")
a.axhline(0.5, color=GREY, lw=0.6, ls=":")
a.set_xlabel(r"$\zeta$"); a.set_ylim(0, 1.5)
a.set_title("(b) half-slope = knee")
a.legend(frameon=False, loc="upper center", ncol=1)

a = ax[2]
a.plot(z, np.log2(1 + np.exp(-z)), color=C1, label="bits left")
a.plot(z, np.log(np.cosh(z / 2)), color=C2, label="regret (nats)")
a.axvspan(-Z_BAND, Z_BAND, color=C3, alpha=0.15, lw=0)
a.axhline(1, color=GREY, lw=0.6, ls=":")
a.axvline(0, color=GREY, lw=0.6, ls=":")
a.set_xlabel(r"$\zeta$"); a.set_ylim(0, 3.8)
a.set_title("(c) one bit left; flat optimum")
a.legend(frameon=False, loc="upper right")
fig.tight_layout(w_pad=1.2)
save(fig, "fig_rules_theory")

# --------------------------------------------------------------- empirical ---
D = pd.read_csv(os.path.join(RES, "table_rules_per_frontier.csv"))
A = D[D.attainable == True].copy()                 # noqa: E712
rules = [("canonical", "canonical\n(full fit)"), ("two_x_best", "2x best"),
         ("knee", "knee"), ("three_point_cheap60", "3-point\ncheap 60%"),
         ("half_slope", "half-slope"), ("three_point", "3-point\nfull"),
         ("cheapest", "cheapest"), ("accuracy_max", "accuracy\nmax")]
fig = plt.figure(figsize=(7.0, 4.7))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
ax = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
a = ax[0]
rng = np.random.default_rng(0)
for k, (r, lab) in enumerate(rules):
    v = A[f"{r}_regret"].dropna().clip(upper=3).to_numpy()
    col = C3 if r in ("two_x_best", "knee", "three_point_cheap60") else (
        C1 if r == "canonical" else GREY)
    a.scatter(k + rng.uniform(-0.18, 0.18, len(v)), np.maximum(v, 1e-4),
              s=6, color=col, alpha=0.75, lw=0)
    a.plot([k - 0.3, k + 0.3], [max(np.median(v), 1e-4)] * 2, color="k", lw=1.2)
    a.text(k, 4.2, f"{np.mean(v <= R_BAND):.0%} in band", ha="center", fontsize=7)
a.axhline(R_BAND, color=C2, lw=0.8, ls="--")
a.set_yscale("log"); a.set_ylim(1e-4, 6)
a.set_xticks(range(len(rules))); a.set_xticklabels([l for _, l in rules], fontsize=7.5)
a.set_ylabel("regret (nats)")
a.set_title(f"(a) regret of each selection rule on {len(A)} frontiers (dashed: 0.05 nats)", fontsize=8.5, pad=14)

a = ax[1]
zz = np.linspace(0, 1.05 * (A.zeta_best - A.zeta_canonical).max(), 300)
a.plot(zz, np.log2(2 / (1 + np.exp(-zz))), color=C1, label="theory")
a.scatter(A.zeta_best - A.zeta_canonical, A.bits_past_canonical, s=9, color=C2,
          lw=0, alpha=0.8, label="frontiers")
a.axhline(1, color=GREY, lw=0.6, ls=":")
a.set_xlabel(r"extra resource past canonical, $\Delta\zeta$")
a.set_ylabel("bits gained")
a.set_title(f"(b) median {A.bits_past_canonical.median():.2f} bit "
            f"at {A.cost_mult_past_canonical.median():.0f}x cost", fontsize=8.5)
a.legend(frameon=False, loc="lower right")

a = ax[2]
lim = [0.01, 0.6]
a.plot(lim, lim, color=GREY, lw=0.6)
a.scatter(A.eps_inf, A.eps_inf_3pt_cheap60, s=10, color=C3, lw=0,
          label="3 cheap models")
a.scatter(A.eps_inf, A.eps_inf_3pt, s=10, facecolors="none", edgecolors=C1,
          lw=0.6, label="3 models, full span")
a.legend(frameon=False, loc="lower right")
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lim); a.set_ylim(lim)
# newer matplotlib labels log minor ticks when the range spans few decades,
# which collides on this panel; decades only.
a.xaxis.set_minor_formatter(NullFormatter())
a.yaxis.set_minor_formatter(NullFormatter())
a.set_xlabel(r"$\hat\varepsilon_\infty$ (full fit)")
a.set_ylabel(r"$\hat\varepsilon_\infty$ (three points)")
a.set_title("(c) floor from three models", fontsize=8.5)
fig.tight_layout(h_pad=1.2, w_pad=2.0)
save(fig, "fig_rules_empirical")

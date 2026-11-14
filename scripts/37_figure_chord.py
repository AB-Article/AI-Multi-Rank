#!/usr/bin/env python3
"""Figure for the chord form of the path (main text, Section 'The Frontier as
a Path').

(a) the three-point chord distance d = 2 arccos exp(-J) against the smoothed
    path length L = int sqrt(kappa) dx on the measured frontiers: the two
    agree, so the length needs three error values and no estimator;
(b) power: the distribution of d/L for exactly two atoms and for three atoms,
    at a realistic design (30 Pareto points over 2.5 nats) and several noise
    levels.  The two are indistinguishable -- the smoother's own bias exceeds
    the deficit that mode counting would have to detect;
(c) inscribed chord sums D_k on raw Pareto data: refining past k = 2 adds
    noise, not length, and D_16 runs past pi, so arc length is defined only
    at coarse scale.

Reads results/table_chord_per_frontier.csv, table_chord_power.csv,
table_chord_refine.csv; writes figures/fig_chord.pdf.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _loaders import RES, ROOT

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.size": 8, "axes.titlesize": 8,
                            "axes.labelsize": 8, "legend.fontsize": 7,
                            "xtick.labelsize": 7, "ytick.labelsize": 7})
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
BLUE, RED, GREY, GREEN = "#3b6ea5", "#c0504d", "#7f7f7f", "#4e8a55"

C = pd.read_csv(os.path.join(RES, "table_chord_per_frontier.csv"))
P = pd.read_csv(os.path.join(RES, "table_chord_power.csv"))
R = pd.read_csv(os.path.join(RES, "table_chord_refine.csv"))

fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))

# ---------------------------------------------------------------- panel (a)
a0 = ax[0]
clean = C.frac_neg_kappa <= 0
a0.errorbar(C.L[clean], C.d[clean], xerr=C.L_se[clean], yerr=C.d_se[clean],
            fmt="o", ms=3.4, lw=.7, color=BLUE, ecolor=GREY, alpha=.85,
            label=r"$\kappa\geq0$ on the window")
a0.errorbar(C.L[~clean], C.d[~clean], xerr=C.L_se[~clean], yerr=C.d_se[~clean],
            fmt="s", ms=3.4, lw=.7, mfc="none", color=RED, ecolor=GREY,
            alpha=.85, label=r"$\kappa<0$ somewhere")
lim = [0, np.pi + .15]
a0.plot(lim, lim, color=GREY, lw=.9, ls=":")
a0.axhline(np.pi, color=RED, lw=.9, ls="--")
a0.axvline(np.pi, color=RED, lw=.9, ls="--")
a0.text(np.pi - .06, .28, r"$\pi$", color=RED, ha="right", fontsize=7)
r = np.corrcoef(C.d, C.L)[0, 1]
a0.text(.05, .95, f"$r={r:.3f}$\nmedian $d/L={C.ratio.median():.3f}$",
        transform=a0.transAxes, va="top", fontsize=7)
a0.set_xlim(lim); a0.set_ylim(lim)
a0.set_xlabel(r"path length $L=\int\sqrt{\kappa}\,dx$  (smoothed)")
a0.set_ylabel(r"chord distance $d=2\arccos e^{-J}$  (3 points)")
a0.set_title("(a) three numbers reproduce the length")
a0.legend(loc="lower right", frameon=False)

# ---------------------------------------------------------------- panel (b)
a1 = ax[1]
sig = sorted(P.sigma.unique())
xpos = np.arange(len(sig))
for k, (atoms, col, off, lab) in enumerate(
        [(2, BLUE, -.13, "two atoms (one channel + floor)"),
         (3, GREEN, .13, "three atoms")]):
    Q = P[P.atoms == atoms].sort_values("sigma")
    a1.errorbar(xpos + off, Q.med,
                yerr=[Q.med - Q.q05, Q.q95 - Q.med],
                fmt="o", ms=3.6, lw=1.1, capsize=2.5, color=col, label=lab)
a1.axhline(1.0, color=GREY, lw=.9, ls=":")
a1.set_xticks(xpos); a1.set_xticklabels([f"{s:g}" for s in sig])
a1.set_xlabel(r"multiplicative noise $\sigma$ on the error")
a1.set_ylabel(r"$d/L$   (median, $5$--$95\%$)")
a1.set_title("(b) mode counting has no power")
a1.legend(loc="lower left", frameon=False)

# ---------------------------------------------------------------- panel (c)
a2 = ax[2]
KS = [1, 2, 4, 8, 16]
M = R[[f"D{k}" for k in KS]].values
for row in M:
    a2.plot(KS, row, color=BLUE, lw=.6, alpha=.32)
a2.plot(KS, np.median(M, axis=0), color=RED, lw=1.8, marker="o", ms=3.6,
        label="median")
a2.axhline(np.pi, color=GREY, lw=.9, ls="--")
a2.text(1.05, np.pi + .06, r"$\pi$ (complete transition)", color=GREY, fontsize=7)
a2.set_xscale("log", base=2); a2.set_xticks(KS)
a2.set_xticklabels([str(k) for k in KS])
a2.set_xlabel(r"subintervals $k$")
a2.set_ylabel(r"inscribed chord sum $D_k$")
a2.set_title("(c) fine-scale length is noise")
a2.legend(loc="upper left", frameon=False)

fig.tight_layout()
out = os.path.join(FIG, "fig_chord.pdf")
fig.savefig(out, bbox_inches="tight")
print("  ->", out)

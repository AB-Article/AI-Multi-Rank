#!/usr/bin/env python3
"""The gauge-invariant calculus (the 'invariant' figure of paper.tex).

(a) The admissible region.  In the gauge-invariant coordinates w = kappa/s^2
    and w_2 = dln kappa/dS every completely monotone frontier obeys
    w_2 <= 1 - w (Thm. 'monofloor'); the boundary IS the floor + one channel
    chart, and the floorless stable chart is an interior curve.  The archive's
    measured points are plotted against both.
(b) The monotone local floor.  F = kappa eps/(s^2+kappa) is a non-increasing
    upper bound on eps_inf, so a frontier with a floor is a flat line and a
    floorless one declines.  Measured on the deepest windows of the archive,
    against the old bound eps(x_0+X).
(c) Which chart.  The observed second-order invariant T = dzeta/dS per
    frontier against design-matched nulls for both charts: the floor chart at
    its own fitted floor share, and the stable chart at three shape indices.

Writes figures/fig_invariant.pdf.
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
                            "legend.fontsize": 6.3,
                            "xtick.labelsize": 7, "ytick.labelsize": 7})
BLUE, RED, GREY, GREEN = "#3b6ea5", "#c0504d", "#888888", "#4f8a5b"

sys.path.insert(0, HERE)
from _loaders import all_frontiers

P = pd.read_csv(os.path.join(RESULTS, "table_chart_test_per_frontier.csv"))
N = pd.read_csv(os.path.join(RESULTS, "table_chart_test_nulls.csv"))
M = pd.read_csv(os.path.join(RESULTS, "table_monotone_floor_bound.csv"))

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.45))


# ----------------------------------------------------------------- jets -----
def jets(x, S, hfrac=0.40, ng=9):
    Z = x[-1] - x[0]; h = Z * hfrac
    grid = np.linspace(x[0] + .15 * Z, x[-1] - .15 * Z, ng)
    out = []
    for g in grid:
        d = x - g; w = np.exp(-.5 * (d / h) ** 2)
        A = np.column_stack([np.ones_like(d), d, .5 * d ** 2])
        Wr = np.sqrt(w)[:, None]
        b, *_ = np.linalg.lstsq(A * Wr, S * Wr[:, 0], rcond=None)
        if b[1] > 0 and -b[2] > 0:
            out.append((b[0], b[1], -b[2]))
    return np.array(out) if len(out) >= 5 else None


# --------------------------------------------------------------------- (a) --
a = ax[0]
ws = np.logspace(-1.3, 1.3, 200)
a.fill_between(ws, 1 - ws, 6, color=RED, alpha=.11, lw=0)
a.plot(ws, 1 - ws, "-", color=RED, lw=1.5,
       label=r"boundary $w_2=1-w$: floor $+1$ channel")
for S0, st in ((1.0, "--"), (2.5, ":")):
    a.plot(ws, -2 * ws - 1 / S0, st, color=BLUE, lw=1.2,
           label=rf"floorless stable, $S={S0:g}$")
pts = []
for fr in all_frontiers():
    x, e = fr["x"], fr["e"]
    o = np.argsort(x); x, e = x[o], e[o]
    J = jets(x, -np.log(e))
    if J is None:
        continue
    Sg, s, k = J[:, 0], J[:, 1], J[:, 2]
    w = k / s ** 2
    lnk = np.log(k)                            # w_2 = dln kappa / dS
    sl = np.polyfit(Sg, lnk, 1)[0]
    pts.append((float(np.median(w)), float(sl)))
pts = np.array(pts)
a.plot(pts[:, 0], pts[:, 1], "o", color=GREY, ms=2.6, mew=0, alpha=.75,
       label=f"archive, {len(pts)} frontiers (plug-in)")
a.set_xscale("log"); a.set_xlim(ws[0], ws[-1]); a.set_ylim(-6, 4)
a.set_xlabel(r"$w=\kappa/s^2$  (squared CV of the rate posterior)")
a.set_ylabel(r"$w_2=\mathrm{d}\ln\kappa/\mathrm{d}S$")
a.axvline(1.0, color="k", lw=.6, ls=(0, (1, 2)))
a.text(1.05, 3.3, "canonical\npoint", fontsize=5.6, va="top")
a.set_title("(a) one inequality bounds every frontier")
a.legend(loc="lower left", frameon=False, borderpad=0.1, labelspacing=0.25)

# --------------------------------------------------------------------- (b) --
b = ax[1]
U = M[M.usable == 1].sort_values("tighten", ascending=False)
pick = set(U.frontier.head(5)) | set(U.frontier.tail(2))
for fr in all_frontiers():
    if fr["name"] not in pick:
        continue
    x, e = fr["x"], fr["e"]
    o = np.argsort(x); x, e = x[o], e[o]
    J = jets(x, -np.log(e))
    if J is None:
        continue
    Sg, s_, k = J[:, 0], J[:, 1], J[:, 2]
    F = k * np.exp(-Sg) / (s_ ** 2 + k)
    b.plot(Sg - Sg[0], F / e.min(), "-", color=BLUE, lw=1.1, alpha=.85)
    b.plot([Sg[-1] - Sg[0]], [F[-1] / e.min()], "o", color=BLUE, ms=2.8, mew=0)
b.axhline(1.0, color=RED, lw=1.4, label=r"old bound $\varepsilon(x_0+X)$")
b.set_xlabel(r"nats of log-error gained across the window")
b.set_ylabel(r"bound on $\varepsilon_\infty$, in units of $\varepsilon_{\rm best}$")
b.set_ylim(0, 1.25)
b.set_title("(b) the local floor can only fall")
b.legend(loc="lower right", frameon=False)
b.text(.03, .95, r"$F=\kappa\varepsilon/(s^2+\kappa)$, monotone by Thm. 4",
       transform=b.transAxes, fontsize=6.4, va="top")

# --------------------------------------------------------------------- (c) --
c = ax[2]
lim = (.4, 60)
c.plot(lim, lim, "-", color="k", lw=.7)
for t, (nm, col, mk) in {"floor_p0_0.50": (r"floor chart at its own $p_0$", RED, "o"),
                         "stable_beta_0.25": (r"stable, $\beta=0.25$", BLUE, "s")}.items():
    g = N[N.truth == t][["frontier", "T_med"]]
    m = P.merge(g, on="frontier", how="inner").dropna(subset=["T_obs", "T_med"])
    c.plot(np.clip(m.T_med, *lim), np.clip(m.T_obs, *lim), mk, color=col, ms=2.7,
           mew=0, alpha=.8, label=f"{nm}  (n={len(m)})")
c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(*lim); c.set_ylim(*lim)
c.set_xlabel(r"design-matched null median of $T$")
c.set_ylabel(r"observed $T=\mathrm{d}\zeta/\mathrm{d}S$")
c.set_title("(c) the archive is not in the floor chart")
c.legend(loc="upper left", frameon=False)

for q in ax:
    q.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=.45)
os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, "fig_invariant.pdf")
fig.savefig(out, bbox_inches="tight")
print("  -> figures/fig_invariant.pdf")

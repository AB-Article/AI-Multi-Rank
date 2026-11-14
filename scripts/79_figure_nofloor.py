#!/usr/bin/env python3
"""The floor is not a parameter (the 'floor is not a parameter' figure of paper.tex).

(a) A two-atom fit to a frontier whose floor is EXACTLY zero returns a floor
    that is a large fraction of the best observed error, monotone in the shape
    index.  The archive's measured median floor share sits on that curve.
(b) The shape index from three independent routes: a direct floorless fit, the
    bandwidth-free three-point chord arc, and the collapse coordinate zeta.
(c) The arc of Law III predicted from the shape index against the measured
    chord arc, and the increasing-hazard frontiers that have no arc at all.

Writes figures/fig_nofloor.pdf.
"""
from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                            "axes.titlesize": 8, "axes.labelsize": 8,
                            "legend.fontsize": 6.3,
                            "xtick.labelsize": 7, "ytick.labelsize": 7})
BLUE, RED, GREY, GREEN = "#3b6ea5", "#c0504d", "#888888", "#4f8a5b"

N = pd.read_csv(os.path.join(RESULTS, "table_no_floor_null.csv"))
A = pd.read_csv(os.path.join(RESULTS, "table_shape_from_arc.csv"))
P = pd.read_csv(os.path.join(RESULTS, "table_canonical_routes.csv"))
MEASURED_SHARE = 0.51

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))

# ------------------------------------------------------------------- (a)
a = ax[0]
a.fill_between(N.beta, N.share_q10, N.share_q90, color=BLUE, alpha=.18, lw=0)
a.plot(N.beta, N.share_median, "o-", color=BLUE, ms=3.2, lw=1.3,
       label="fitted floor, true floor $=0$")
a.axhline(MEASURED_SHARE, color=RED, lw=1.2, ls="--",
          label=f"archive median, ${MEASURED_SHARE}$")
b_hit = float(np.interp(MEASURED_SHARE, N.share_median[::-1], N.beta[::-1]))
a.plot([b_hit], [MEASURED_SHARE], "s", color=RED, ms=4.2, zorder=5)
a.annotate(rf"$\beta={b_hit:.2f}$", (b_hit, MEASURED_SHARE),
           textcoords="offset points", xytext=(6, 7), fontsize=6.4, color=RED)
a.set_xlabel(r"true shape index $\beta$  (floor $=0$)")
a.set_ylabel(r"fitted $\hat\varepsilon_\infty\,/\,\varepsilon_{\rm best}$")
a.set_title("(a) a floor where there is none")
a.set_ylim(0, 1); a.legend(loc="lower left", frameon=False)

# ------------------------------------------------------------------- (b)
b = ax[1]
ok = A[(A.beta_fit > 0) & (A.beta_fit < 1) & np.isfinite(A.beta_chord)]
b.plot([0, 1], [0, 1], color=GREY, lw=.9, ls=":")
b.plot(ok.beta_fit, ok.beta_chord, "o", color=BLUE, ms=4.0,
       label="chord arc (no bandwidth, no floor)")
pp = P[(P.beta_zeta > 0) & (P.beta_zeta < 1) & (P.beta_arc > 0) & (P.beta_arc < 1)]
b.plot(pp.beta_zeta, pp.beta_arc, "^", color=GREEN, ms=3.2, alpha=.75,
       label=r"integrated arc vs $\zeta$ (55 frontiers)")
b.set_xlabel(r"$\beta$ from a shape fit  /  from $\zeta=\ln(\kappa/s^2)$")
b.set_ylabel(r"$\beta$ from the arc")
b.set_title(r"(b) three routes to $\beta$")
b.set_xlim(0, 1); b.set_ylim(0, 1)
b.legend(loc="upper left", frameon=False)

# ------------------------------------------------------------------- (c)
c = ax[2]
dec = A[A.hazard == "decreasing"]; inc = A[A.hazard == "increasing"]
lim = max(0.1, float(np.nanmax([dec.arc.max(), dec.arc_pred.max()])) * 1.1)
c.plot([0, lim], [0, lim], color=GREY, lw=.9, ls=":")
c.plot(dec.arc, dec.arc_pred, "o", color=BLUE, ms=4.2,
       label="decreasing hazard ($\\beta<1$)")
c.plot(inc.arc, np.full(len(inc), 0.02 * lim), "x", color=RED, ms=5,
       label="increasing hazard: no arc")
c.set_xlabel("measured chord arc  (rad)")
c.set_ylabel(r"arc predicted from $\beta$  (rad)")
c.set_title("(c) the arc is the shape index")
c.set_xlim(-0.05 * lim, lim); c.set_ylim(-0.05 * lim, lim)
c.legend(loc="lower right", frameon=False)

for q in ax:
    q.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.5)
os.makedirs(FIG, exist_ok=True)
fig.savefig(os.path.join(FIG, "fig_nofloor.pdf"))
print("  -> figures/fig_nofloor.pdf")

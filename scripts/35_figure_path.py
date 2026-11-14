#!/usr/bin/env python3
"""Figure for the path-length law (main text, Section 'The Frontier as a Path').

(a) the universal arc: every completely monotone frontier with one channel and
    a floor traverses the same path of Fisher-Rao length pi, parameterised by
    the error share p; the measured windows are drawn as segments on it;
(b) measured length L = int sqrt(kappa) dx against the floor-free prediction
    2[arcsin sqrt(s1/alpha_hat) - arcsin sqrt(s2/alpha_hat)], with pi marked;
(c) the Gamma dichotomy: g = kappa/s^2 in the lower against the upper third of
    each window.  A Gamma (gapless) spectrum sits on the diagonal; an atom at
    the origin pushes points above it.

Reads results/table_path_per_frontier.csv; writes figures/fig_path.pdf.
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
BLUE, RED, GREY, GREEN = "#3b6ea5", "#c0504d", "#7f7f7f", "#4e8a55"

D = pd.read_csv(os.path.join(RES, "table_path_per_frontier.csv"))
W = D[(D.span >= 1.0) & (D.n >= 15)].copy().reset_index(drop=True)

fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))

# ---------------------------------------------------------------- panel (a)
a0 = ax[0]
W["arc_lo"] = 2 * np.arcsin(np.sqrt(np.clip(W.p_top, 0, 1)))
W["arc_hi"] = 2 * np.arcsin(np.sqrt(np.clip(W.p_bot, 0, 1)))
O = W.sort_values("arc_hi").reset_index(drop=True)
for i, r in O.iterrows():
    a0.plot([r.arc_lo, r.arc_hi], [i, i], color=BLUE, lw=1.8,
            solid_capstyle="butt", alpha=.85)
a0.axvline(np.pi, color=RED, lw=1.0, ls="--")
a0.axvline(0, color=GREEN, lw=1.0, ls="--")
a0.text(np.pi - .05, len(O) * .5, "pure power law\n$p=1$", color=RED,
        rotation=90, ha="right", va="center", fontsize=7)
a0.text(.06, len(O) * .5, "floor\n$p=0$", color=GREEN,
        rotation=90, ha="left", va="center", fontsize=7)
a0.set_xlim(-0.05, np.pi + .05); a0.set_ylim(-1, len(O))
a0.set_yticks([])
a0.set_xlabel(r"position on the arc, $2\arcsin\sqrt{p}$   (total length $\pi$)")
a0.set_ylabel("frontiers, ordered by window start")
a0.set_title("(a) where the measured windows sit")

# ---------------------------------------------------------------- panel (b)
a1 = ax[1]
m = np.isfinite(W.L) & np.isfinite(W.L_pred)
a1.errorbar(W.L_pred[m], W.L[m], yerr=W.L_se[m], fmt="o", ms=3.2, lw=.7,
            color=BLUE, ecolor=GREY, alpha=.85)
lim = [-0.1, np.pi + .1]
a1.plot(lim, lim, color=GREY, lw=.9, ls=":")
a1.axhline(np.pi, color=RED, lw=.9, ls="--")
a1.text(lim[0] + .05, np.pi - .12, r"$L=\pi$ (complete transition)",
        color=RED, fontsize=7, va="top")
r = np.corrcoef(W.L[m], W.L_pred[m])[0, 1]
a1.text(.05, .92, f"$r={r:.3f}$, $n={int(m.sum())}$", transform=a1.transAxes, fontsize=7)
a1.set_xlim(lim); a1.set_ylim(lim)
a1.set_xlabel(r"floor-free prediction $2[\arcsin\sqrt{s_1/\hat\alpha}-\arcsin\sqrt{s_2/\hat\alpha}]$")
a1.set_ylabel(r"measured $L=\int\sqrt{\kappa}\,dx$")
a1.set_title("(b) the length identity on measured frontiers")

# ---------------------------------------------------------------- panel (c)
a2 = ax[2]
G = W[(W.g_lo > 0) & (W.g_hi > 0)]
a2.loglog(G.g_lo, G.g_hi, "o", ms=3.4, color=GREEN, alpha=.85)
lo = min(G.g_lo.min(), G.g_hi.min()) * .6
hi = max(G.g_lo.max(), G.g_hi.max()) * 1.6
a2.plot([lo, hi], [lo, hi], color=GREY, lw=.9, ls=":")
a2.text(lo * 1.3, lo * 1.05, "Gamma spectrum\n(no gap): $g$ constant",
        color=GREY, fontsize=7, va="bottom")
frac = float((W.dlng_du > 0).mean())
a2.text(.05, .92, f"$d\\ln g/dx>0$ on {frac:.0%} of {len(W)}", transform=a2.transAxes, fontsize=7)
a2.set_xlim(lo, hi); a2.set_ylim(lo, hi)
a2.set_xlabel(r"$g=\kappa/s^2$, lower third of window")
a2.set_ylabel(r"$g$, upper third of window")
a2.set_title("(c) an atom at the origin, not a Gamma tail")

for a in ax:
    a.grid(alpha=.18, lw=.5)
fig.tight_layout()
os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, "fig_path.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"  -> figures/fig_path.pdf")

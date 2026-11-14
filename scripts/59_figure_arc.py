#!/usr/bin/env python3
"""The arc figure (Law III).

(a) The complete sweep of a single-channel frontier as an arc of length pi:
    the 0.05-nat indifference band, the five checkpoints that cover any
    single-channel frontier, and the measured windows placed on the same
    absolute scale.  phi = 2 arccos sqrt(p0) = 2 arctan sqrt(N_eff) is the
    channel angle of eq:angle, measured FROM THE FLOOR (phi = 0) towards the
    pure power law (phi = pi); the mirror 2 arcsin sqrt(p0) runs the other way
    and gives the same sin phi, hence the same regret and the same arc length.
    Revision 2026-10-08m fixed the two end labels, which had been drawn in the
    mirror orientation while the text used this one.
(b) Regret as minus the log sine of arc position, R = -ln sin phi, with the
    covering cells of the five checkpoints.
(c) What a published frontier carries: checkpoints needed against Pareto
    points measured, one point per frontier.

Writes figures/fig_arc.pdf.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.size": 8, "axes.titlesize": 8,
                            "axes.labelsize": 8, "legend.fontsize": 7,
                            "xtick.labelsize": 7, "ytick.labelsize": 7})
BLUE, RED, GREY = "#3b6ea5", "#c0504d", "#888888"

place = pd.read_csv(os.path.join(RESULTS, "table_arc_placement.csv"))
cov = pd.read_csv(os.path.join(RESULTS, "table_arc_covering.csv"))
per = pd.read_csv(os.path.join(RESULTS, "table_arc_per_frontier.csv"))
summ = pd.read_csv(os.path.join(RESULTS, "table_arc_summary.csv")).iloc[0]
BAND = float(summ.band_halfwidth_rad)
COVER = float(summ.cover_halfwidth_rad)

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))

# ---------------------------------------------------------------- (a) the arc
a0 = ax[0]
th = np.linspace(0, np.pi, 400)
a0.plot(np.cos(th), np.sin(th), color="black", lw=1.4, zorder=2)
tb = np.linspace(np.pi / 2 - BAND, np.pi / 2 + BAND, 120)
a0.plot(np.cos(tb), np.sin(tb), color=RED, lw=4.0, alpha=.22,
        solid_capstyle="butt", zorder=1)
# Measured windows, drawn just inside the arc and CENTRED ON THE RIGHT ANGLE so
# that their lengths can be compared.  This is an illustrative placement, not a
# measured one: the three-accuracy chord returns an arc LENGTH and no position.
# (The positions the jet does return sit deep -- median N_eff = 0.465, i.e.
# phi = 1.20 < pi/2 -- so nothing here should be read as "the windows lie in the
# upper half".)
per_s = per.sort_values("arc", ascending=False).reset_index(drop=True)
for i, r in per_s.iterrows():
    lo = np.pi / 2 - 0.25 * r.arc       # centred on the canonical point
    tw = np.linspace(lo, min(lo + r.arc, np.pi), 60)
    if r.arc <= 1e-9:
        continue
    rr = 0.94 - 0.016 * i
    a0.plot(rr * np.cos(tw), rr * np.sin(tw), color=BLUE, lw=1.0, alpha=.5)
a0.scatter(np.cos(place.phi), np.sin(place.phi), s=22, zorder=4,
           facecolor="white", edgecolor=RED, lw=1.1)
a0.plot([0], [1], marker="|", ms=7, color=RED, mew=1.4, zorder=5)
a0.annotate(r"canonical point  $\varphi=\pi/2$", xy=(0, 1), xytext=(0.05, 1.17),
            ha="center", color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.7))
a0.text(1.16, 0.10, r"$\varphi=\pi$" "\n" "power law", ha="center", va="bottom",
        fontsize=6.5, color="#444444")
a0.text(-1.16, 0.10, r"$\varphi=0$" "\n" "floor", ha="center", va="bottom",
        fontsize=6.5, color="#444444")
a0.text(0.0, -0.10, f"{int(summ.frontiers_with_positive_arc)} measured windows, median "
        f"{summ.pos_median_arc_frac:.2f}" r"$\pi$", ha="center", fontsize=6.5, color=BLUE)
a0.set_xlim(-1.45, 1.45); a0.set_ylim(-0.28, 1.42)
a0.set_aspect("equal"); a0.axis("off")
a0.set_title("(a) one channel is an arc of length $\\pi$", loc="left")

# ------------------------------------------------------- (b) R = -ln sin phi
a1 = ax[1]
ph = np.linspace(0.06, np.pi - 0.06, 600)
a1.plot(ph, -np.log(np.sin(ph)), color="black", lw=1.4, zorder=3,
        label=r"$R=-\ln\sin\varphi$")
a1.axvspan(np.pi / 2 - BAND, np.pi / 2 + BAND, color=RED, alpha=.10, lw=0)
a1.axvline(np.pi / 2, color=RED, ls="--", lw=1.0)
a1.axhline(0.05, color=GREY, ls=":", lw=0.9)
# Panel (b)'s abscissa IS the paper's phi, so the positions must be read in
# the paper's convention, not script 57's internal mirror (round 13).
for _, r in place.iterrows():
    php = r.phi_paper if hasattr(r, 'phi_paper') else np.pi - r.phi
    a1.axvline(php, color=BLUE, lw=0.8, alpha=.75)
    a1.axvspan(php - COVER, php + COVER, color=BLUE, alpha=.07, lw=0)
a1.set_xlim(0, np.pi); a1.set_ylim(0, 1.45)
a1.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
a1.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
a1.set_xlabel(r"arc position $\varphi=2\arccos\sqrt{p_0}=2\arctan\sqrt{N_{\rm eff}}$")
a1.set_ylabel("regret from the reference (nats)")
a1.text(np.pi/2, 1.42, f"{int(cov.loc[cov.R_nats==0.05,'K_full_arc'].iloc[0])}"
        " checkpoints cover the whole arc\nto 0.05 nats (shaded cells)",
        fontsize=6.5, color=BLUE, ha="center", va="top")
a1.set_title("(b) regret is a log sine", loc="left")

# --------------------------------------------- (c) what a frontier carries
a2 = ax[2]
pos = per[per.arc > 1e-9]; nul = per[per.arc <= 1e-9]
a2.scatter(pos.n_pareto, pos.K_needed, s=20, color=BLUE, alpha=.85, zorder=3,
           label="measurable arc")
a2.scatter(nul.n_pareto, np.full(len(nul), 0.72), s=20, marker="x", color=GREY,
           lw=0.9, zorder=3, label="no measurable arc")
a2.legend(loc="lower right", frameon=False, fontsize=6, handletextpad=.3,
          borderpad=.1)
nn = np.linspace(per.n_pareto.min() * .9, per.n_pareto.max() * 1.05, 50)
a2.plot(nn, nn, color=GREY, ls=":", lw=0.9)
a2.axhline(float(summ.pos_median_K_needed), color=RED, ls="--", lw=1.0)
a2.text(per.n_pareto.min() * .95, float(summ.pos_median_K_needed) + 0.28,
        f"median {summ.pos_median_K_needed:.2f}", color=RED, fontsize=6.5)
a2.text(per.n_pareto.min() * .95, 7.4,
        f"median redundancy {summ.median_redundancy:.0f}$\\times$; "
        f"{summ.median_models_per_cell:.0f} models\nfit inside one 0.05-nat band",
        fontsize=6.5, color="#444444", va="top")
a2.set_yscale("log"); a2.set_xscale("log")
a2.set_ylim(0.6, 9); a2.set_xlim(per.n_pareto.min() * .85, per.n_pareto.max() * 1.15)
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
a2.yaxis.set_major_locator(FixedLocator([1, 2, 3, 5]))
a2.yaxis.set_major_formatter(FixedFormatter(["1", "2", "3", "5"]))
a2.yaxis.set_minor_locator(FixedLocator([])); a2.yaxis.set_minor_formatter(NullFormatter())
a2.xaxis.set_major_locator(FixedLocator([10, 15, 20, 30]))
a2.xaxis.set_major_formatter(FixedFormatter(["10", "15", "20", "30"]))
a2.xaxis.set_minor_locator(FixedLocator([])); a2.xaxis.set_minor_formatter(NullFormatter())
a2.set_xlabel("Pareto points measured")
a2.set_ylabel("checkpoints the arc needs")
a2.set_title("(c) a frontier carries two checkpoints", loc="left")

for a in ax[1:]:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.5)
out = os.path.join(FIG, "fig_arc.pdf")
fig.savefig(out)
print("  ->", out)

#!/usr/bin/env python3
"""The count axis.

(a) P(m) = Pr[all m attempts fail] against the Hausdorff inversion
    int g^m dnu, on both families of the co-failure panel.  Law I on a count
    axis, where it is a theorem rather than an axiom.
(b) The inverted spectrum nu: two atoms, one of them ABOVE 1 - q0, and no mass
    at g = 1.  The upper atom is the systematically-wrong items.
(c) Held-out higher-order co-failure: the paper's nesting law extended
    (kappa^{m-1} h_b, kappa fitted on pairs) against the count axis
    (nu fitted on orders 1..4 only).

Reads results/table_count_axis_{chain,spectrum,holdout}.csv  (71, 72)
Writes figures/fig_count_axis.pdf.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.size": 8, "axes.titlesize": 8,
                            "axes.labelsize": 8, "legend.fontsize": 6.5,
                            "xtick.labelsize": 7, "ytick.labelsize": 7})
BLUE, RED, GREY, GREEN = "#3b6ea5", "#c0504d", "#888888", "#4f8a5b"
Q0 = 0.10

ch = pd.read_csv(os.path.join(RESULTS, "table_count_axis_chain.csv"))
sp = pd.read_csv(os.path.join(RESULTS, "table_count_axis_spectrum.csv"))
ho = pd.read_csv(os.path.join(RESULTS, "table_count_axis_holdout.csv"))

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.45))

# ---- (a) the count-axis frontier -------------------------------------------
a = ax[0]
for fam, col, mk in (("indep", BLUE, "o"), ("nested", RED, "s")):
    d = ch[ch.family == fam]
    s = sp[sp.family == fam]
    m = d.m.to_numpy()
    a.errorbar(m, d.P, yerr=2 * d.se, fmt=mk, ms=2.6, lw=0.7, color=col,
               label=f"{fam}: observed", zorder=3)
    pred = np.array([(s.mass.to_numpy() * s.g.to_numpy() ** k).sum() for k in m])
    a.plot(m, pred, "-", lw=1.0, color=col, alpha=0.8,
           label=f"{fam}: " + r"$\int g^m d\nu$")
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("attempts $m$ (log scale)")
a.set_ylabel(r"$P(m)=\Pr[\mathrm{all}\ m\ \mathrm{fail}]$")
a.set_title("(a) Law I on the count axis")
a.legend(frameon=False, loc="lower left")

# ---- (b) the spectrum ------------------------------------------------------
b = ax[1]
for fam, col, off in (("indep", BLUE, -0.012), ("nested", RED, 0.012)):
    s = sp[sp.family == fam]
    b.bar(s.g + off, s.mass, width=0.022, color=col, label=fam, alpha=0.85)
b.axvline(1 - Q0, color=GREEN, lw=1.0, ls="--")
b.text(1 - Q0 - 0.02, b.get_ylim()[1] * 0.92, r"$1-q_0$", color=GREEN,
       ha="right", fontsize=6.5)
b.axvline(1.0, color=GREY, lw=1.0, ls=":")
b.text(0.995, b.get_ylim()[1] * 0.45, r"$\nu(\{1\})=0$", color=GREY,
       ha="right", va="top", rotation=90, fontsize=6.5)
b.set_xlim(0, 1.06)
b.set_xlabel(r"per-item attempt-failure rate $g$")
b.set_ylabel(r"mass $d\nu$")
b.set_title("(b) the inverted spectrum")
b.legend(frameon=False, loc="upper left")

# ---- (c) held-out higher orders --------------------------------------------
c = ax[2]
for fam, mk in (("indep", "o"), ("nested", "s")):
    d = ho[ho.family == fam].sort_values("order")
    c.plot(d.order, d.errB, mk + "-", ms=2.6, lw=1.0, color=RED, alpha=.9 if fam=="indep" else .5,
           label=f"nesting law, $\\kappa^{{m-1}}h_b$ ({fam})")
    c.plot(d.order, d.errD, mk + "-", ms=2.6, lw=1.0, color=BLUE, alpha=.9 if fam=="indep" else .5,
           label=f"count axis ({fam})")
c.axhline(0, color=GREY, lw=0.6)
c.set_xlabel(r"order $m$ (held out; $\nu$ fitted on $m \leq 4$)")
c.set_ylabel(r"median $|\mathrm{pred}/\mathrm{obs}-1|$")
c.set_title("(c) held-out co-failure")
c.legend(frameon=False, loc="upper left", ncol=1)

for z in ax:
    z.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(FIG, "fig_count_axis.pdf"))
print("wrote figures/fig_count_axis.pdf")

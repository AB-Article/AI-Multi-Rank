#!/usr/bin/env python3
"""What the law predicts (Figure 3 of paper.tex).

(a) Learning curve: train on m models spread over the window, predict the rest.
    The three-parameter law plateaus well above the measurement noise; one
    shape parameter removes most of the gap.
(b) Held-out agreement, pooled over every frontier and every interior
    leave-one-out point.
(c) The shape parameter is weakly identified: profile intervals by corpus.

Writes figures/fig_prediction.pdf.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                            "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.3,
                            "xtick.labelsize": 7, "ytick.labelsize": 7})
BLUE, RED, GREY, GREEN = "#3b6ea5", "#c0504d", "#888888", "#4f8a5b"

S = pd.read_csv(os.path.join(RESULTS, "table_breadth_summary.csv"))
C = pd.read_csv(os.path.join(RESULTS, "table_prediction_correlations.csv"))
F = pd.read_csv(os.path.join(RESULTS, "table_prediction_frontiers.csv"))
B = pd.read_csv(os.path.join(RESULTS, "table_breadth_beta.csv"))

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.45))

# ------------------------------------------------------------------ (a)
a = ax[0]
a.plot(S.m, S.law, "o-", color=BLUE, ms=3.4, lw=1.3, label="law (3 par.)")
a.plot(S.m, S.sb, "s-", color=RED, ms=3.4, lw=1.3, label="+ shape par. $\\beta$")
a.plot(S.m, S.cm, "^--", color=GREEN, ms=3, lw=1.0, label="full monotone class")
a.plot(S.m, S.power, "v:", color=GREY, ms=3, lw=1.0, label="power law, no floor")
a.fill_between(S.m, 0, S.noise, color="k", alpha=.10, lw=0)
a.text(13.2, S.noise.iloc[-1] * 0.45, "measurement noise", fontsize=5.8, color="k", ha="right")
a.set_xscale("log"); a.set_xticks([3, 4, 6, 8, 12, 16]); a.set_xticklabels([3, 4, 6, 8, 12, 16])
a.set_xlabel("models used to fit, $m$"); a.set_ylabel("held-out $|\\Delta\\ln\\varepsilon|$ (nats)")
a.set_ylim(0, 0.118); a.legend(loc="upper right", frameon=False)
a.set_title("(a) the law plateaus; $\\beta$ removes it", loc="left")

# ------------------------------------------------------------------ (b)
b = ax[1]
r = C.set_index("method")
meth = ["power", "law", "interp"]
lab = {"power": "power law", "law": "saturating law", "interp": "local interpolation"}
col = {"power": GREY, "law": BLUE, "interp": GREEN}
yy = np.arange(len(meth))
b.barh(yy, [r.loc[m, "R2_log"] for m in meth], color=[col[m] for m in meth], height=.55)
for i, m in enumerate(meth):
    v = r.loc[m, "R2_log"]
    if v > 0.9755:
        b.text(v - .0004, i, f"{v:.4f} ", va="center", ha="right", fontsize=6.5, color="w")
    else:
        b.text(v + .0004, i, f" {v:.4f}", va="center", ha="left", fontsize=6.5, color="k")
    b.text(0.9703, i + 0.30, f"{r.loc[m,'median_rel_pct']:.1f}% median rel. err.",
           va="center", fontsize=6.0, color=GREY)
b.set_yticks(yy); b.set_yticklabels([lab[m] for m in meth])
b.set_xlim(0.9700, 0.9930); b.set_xlabel("$R^2$ on held-out log-error")
b.set_title(f"(b) agreement, {int(r.loc['law','n_points'])} held-out models", loc="left")
b.set_xticks([0.975, 0.985, 0.990])

# ------------------------------------------------------------------ (c)
c = ax[2]
g = B.groupby("corpus")["beta_free"]
order = g.median().sort_values().index.tolist()
for i, k in enumerate(order):
    v = B[B.corpus == k]["beta_free"].values
    c.plot(v, np.full_like(v, i, dtype=float), "o", ms=3, mfc="none", mew=.8, color=BLUE)
    c.plot([np.median(v)], [i], "|", ms=12, mew=2, color=RED)
c.axvline(1.0, color="k", lw=.9)
c.text(1.08, 0.30, "$\\beta=1$:\nthe two-atom law", fontsize=5.8, color="k", va="center")
c.set_yticks(range(len(order))); c.set_yticklabels(order)
c.set_xscale("log"); c.set_xlim(0.25, 6)
c.set_xticks([0.3, 0.5, 1, 2, 4]); c.set_xticklabels(["0.3", "0.5", "1", "2", "4"])
c.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
c.set_ylim(-0.6, len(order) - 0.3)
c.set_xlabel("fitted shape parameter $\\beta$")
c.set_title("(c) $\\beta$ varies by corpus", loc="left")

fig.tight_layout(pad=0.4)
os.makedirs(FIG, exist_ok=True)
fig.savefig(os.path.join(FIG, "fig_prediction.pdf"))
print("wrote", os.path.join(FIG, "fig_prediction.pdf"))

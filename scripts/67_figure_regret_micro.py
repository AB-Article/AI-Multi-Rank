#!/usr/bin/env python3
"""The regret law and the microfoundation test.

(a) The regret law: P(R <= rho) = (2/pi) arccos e^{-rho}, the sqrt(rho)
    approximation, and the panel's observed band fractions against the
    prediction conditional on each cell's own arc.
(b) Mean regret from the canonical point for an operating point placed
    uniformly in arc length: ln 2 for one channel, rising to gamma + ln 2.
(c) Which microfoundation: observed pair co-failure rates against the three
    mechanisms, on the Fashion-MNIST co-failure panel.

Writes figures/fig_regret_micro.pdf.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import digamma

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.size": 8, "axes.titlesize": 8,
                            "axes.labelsize": 8, "legend.fontsize": 6.5,
                            "xtick.labelsize": 7, "ytick.labelsize": 7})
BLUE, RED, GREY, GREEN = "#3b6ea5", "#c0504d", "#888888", "#4f8a5b"

MP = pd.read_csv(os.path.join(RESULTS, "table_micro_pairs.csv"))

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.45))

# ------------------------------------------------------------------ (a)
a = ax[0]
r = np.linspace(1e-4, 0.5, 600)
a.plot(r, (2 / np.pi) * np.arccos(np.exp(-r)), color=BLUE, lw=1.6,
       label=r"$(2/\pi)\arccos e^{-\varrho}$")
a.plot(r, (2 * np.sqrt(2) / np.pi) * np.sqrt(r), color=RED, lw=1.0, ls="--",
       label=r"$0.900\,\sqrt{\varrho}$")
for rr, lab in [(0.05, "0.05")]:
    a.plot([rr], [(2 / np.pi) * np.arccos(np.exp(-rr))], "o", ms=4, color=BLUE)
a.set_xlabel(r"tolerance $\varrho$ (nats)")
a.set_ylabel("fraction of the arc within $\\varrho$")
a.set_xlim(0, 0.5); a.set_ylim(0, 0.72)
a.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.02,0.99))
a.set_title("(a) the regret law", loc="left")
a.text(0.50, 0.17,
       "placement of a frontier's own\nmodels is not predicted by\nthe theory: see Fig. 4",
       transform=a.transAxes, fontsize=5.8, color=GREY, ha="left", va="center")

# ------------------------------------------------------------------ (b)
b = ax[1]
M = np.arange(2, 60)
EM = digamma(M / 2) - digamma(0.5) - np.log(M)
b.plot(M - 1, EM, "o-", color=BLUE, ms=3, lw=1.1, label=r"$\mathbb{E}[R]$, $N+1$ modes")
b.axhline(np.euler_gamma + np.log(2), color=RED, lw=1.0, ls="--",
          label=r"$\gamma+\ln 2=1.2704$")
b.axhline(np.log(2), color=GREEN, lw=1.0, ls=":", label=r"$\ln 2=0.6931$  ($N=1$)")
b.set_xscale("log")
b.set_xlabel("$N$ (reducible modes)")
b.set_ylabel("mean regret (nats)")
b.set_ylim(0.6, 1.35)
b.legend(loc="lower right", frameon=False)
b.set_title("(b) the cost of not knowing", loc="left")

# ------------------------------------------------------------------ (c)
c = ax[2]
d = MP[MP.family == "indep"]
di = d.dropna(subset=["A_spectral"])
c.plot(di.obs, di.A_spectral, "s", ms=3, mfc="none", mew=.8, color=RED,
       label="(A) modes, cond. indep.")
c.plot(d.obs, d.B_nested, "o", ms=3, mfc="none", mew=.8, color=BLUE,
       label="(B) nested difficulty")
c.plot(d.obs, d.C_indep, "^", ms=3, mfc="none", mew=.8, color=GREY,
       label="(C) independent")
lim = [4e-3, 1.0]
c.plot(lim, lim, color="k", lw=.8)
c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(*lim); c.set_ylim(*lim)
c.set_xlabel("measured $\\Pr[\\mathrm{both\\ fail}]$")
c.set_ylabel("predicted")
c.legend(loc="lower right", frameon=False)
c.set_title("(c) which microfoundation", loc="left")

fig.tight_layout(pad=0.4)
os.makedirs(FIG, exist_ok=True)
fig.savefig(os.path.join(FIG, "fig_regret_micro.pdf"))
print("wrote", os.path.join(FIG, "fig_regret_micro.pdf"))

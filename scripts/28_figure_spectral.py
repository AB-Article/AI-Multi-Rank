#!/usr/bin/env python3
"""Figure for the spectral extension (Sections 8.9 and 8.13 of the main text).

(a) resolution number against the recovered channel count, by corpus;
(b) calibrated detection probability against the resolution number;
(c) reciprocity of the fitted interaction and the fitted exponent on the
    factorial grids, with the predicted slope of -1.

Reads results/table_resolution_frontiers.csv, table_resolution_sim.csv and
table_resolution_model.csv; writes figures/fig_spectral.pdf.
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
RNG = np.random.default_rng(3)


def main():
    D = pd.read_csv(os.path.join(RES, "table_resolution_frontiers.csv"))
    S = pd.read_csv(os.path.join(RES, "table_resolution_sim.csv"))
    M = pd.read_csv(os.path.join(RES, "table_resolution_model.csv"))
    L = pd.read_csv(os.path.join(RES, "table_factorial_N_test_lm.csv"))
    a = np.array([[float(z) for z in s.split(";")] for s in L.alpha_theta1])
    L["A_hat"] = np.exp(np.log(a).mean(1))
    L["grid"] = [f.split("|")[0] for f in L.frontier]

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35))

    # (a) resolution number vs recovered channel count
    corpora = ["82 frontiers", "LM factorial"]
    marks = {"82 frontiers": ("o", "#3b6ea5"), "LM factorial": ("s", "#c0504d")}
    for ci, c in enumerate(corpora):
        g = D[D.corpus == c]
        for k in sorted(g.N_hat.unique()):
            h = g[g.N_hat == k]
            xpos = k + (ci - 0.5) * 0.26
            ax[0].scatter(xpos + RNG.uniform(-.06, .06, len(h)), h.V,
                          s=9, alpha=.55, marker=marks[c][0], color=marks[c][1],
                          label=c if k == 1 else None, linewidths=0)
            ax[0].plot([xpos - .11, xpos + .11], [h.V.median()] * 2,
                       color=marks[c][1], lw=1.6)
    ax[0].set_yscale("log"); ax[0].set_xticks([1, 2, 3])
    ax[0].set_xlabel(r"channels selected by BIC, $\hat{N}$")
    ax[0].set_ylabel(r"resolution number $\mathcal{V}$")
    ax[0].set_title("(a) what the window can resolve", loc="left")
    ax[0].legend(frameon=False, loc="lower right")

    # (b) calibration
    S["logV"] = np.log10(S.V)
    bins = np.array([1.0, 2.0, 2.9, 4.6])
    ctr = 0.5 * (bins[:-1] + bins[1:])
    styles = {"two channels (ratio 3)": ("-", "o", "#3b6ea5"),
              "Gamma spectrum (shape 4)": ("--", "^", "#7b3294"),
              "one channel": (":", "s", "#888888")}
    for t, (ls, mk, col) in styles.items():
        g = S[S.truth == t]
        p = [(g.N_hat[(g.logV >= lo) & (g.logV < hi)] >= 2).mean()
             for lo, hi in zip(bins[:-1], bins[1:])]
        ax[1].plot(10 ** ctr, p, ls, marker=mk, ms=3.5, color=col, lw=1.2,
                   label=t.split(" (")[0])
    ax[1].set_xscale("log"); ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_xlabel(r"resolution number $\mathcal{V}$")
    ax[1].set_ylabel(r"$P(\hat{N} \geq 2)$")
    ax[1].set_title("(b) calibration, simulated", loc="left")
    ax[1].legend(frameon=False, loc="upper left")

    # (c) reciprocity
    cols = {"pythia": "#3b6ea5", "pythia-deduped": "#4f9153", "gadre": "#c0504d"}
    for grid, col in cols.items():
        g = L[L.grid == grid]
        if not len(g):
            continue
        ax[2].scatter(g.A_hat, g.theta_hat, s=16, color=col, label=grid, linewidths=0)
        xs = np.linspace(g.A_hat.min(), g.A_hat.max(), 20)
        med = float(M.loc[M.grid == grid, "median_product"].iloc[0])
        ax[2].plot(xs, med / xs, color=col, lw=1.0, alpha=.7)
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel(r"exponent scale $\hat A_t$")
    ax[2].set_ylabel(r"interaction $\hat\theta_t$")
    ax[2].set_title(r"(c) $\hat\theta_t\hat A_t$ fixed per grid", loc="left")
    ax[2].legend(frameon=False, loc="lower left")

    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig_spectral.pdf")
    fig.savefig(out); print("  ->", out)


if __name__ == "__main__":
    main()

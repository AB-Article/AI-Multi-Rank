#!/usr/bin/env python3
"""Figure for the surprisal identity and the running floor (main text,
Sections 'The Surprisal Identity and the Running Floor' and 'The Floor Runs').

(a) the single coordinate: excess log-error against the floor share, with the
    measured operating points of the rule-selected candidates;
(b) the running floor on nested windows, against a gap null and a no-gap null;
(c) the measured spectral width g = k/s^2 against the window statistic V,
    coloured by the recovered channel count, with the g*V = 1 separatrix.

Reads results/table_p0_selection.csv, table_running_floor.csv and
table_spectral_width.csv; writes figures/fig_surprisal.pdf.
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
RNG = np.random.default_rng(11)
BLUE, RED, GREY = "#3b6ea5", "#c0504d", "#7f7f7f"


def _auc(score, label):
    s, y = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    P = pd.read_csv(os.path.join(RES, "table_p0_selection.csv"))
    R = pd.read_csv(os.path.join(RES, "table_running_floor.csv"))
    W = pd.read_csv(os.path.join(RES, "table_spectral_width.csv"))
    os.makedirs(FIG, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))

    # ---- (a) one coordinate ------------------------------------------------
    p = np.linspace(0.02, 0.98, 400)
    ax[0].plot(p, -np.log(p), color="k", lw=1.3, zorder=3)
    band = (p >= 0.346) & (p <= 0.654)
    ax[0].fill_between(p[band], 0, -np.log(p[band]), color=BLUE, alpha=.16, lw=0)
    for q, lab in ((0.5, r"$p_0=\frac{1}{2}$"), (1 / 3, r"$\frac{1}{3}$"), (0.25, r"$\frac{1}{4}$")):
        ax[0].plot([q, q], [0, -np.log(q)], color=GREY, lw=.7, ls=":")
        ax[0].annotate(lab, (q, -np.log(q)), textcoords="offset points",
                       xytext=(2, 3), fontsize=6.5, color=GREY)
    y = -np.log(P.p0_selected.to_numpy())
    ax[0].scatter(P.p0_selected, y + RNG.uniform(-.02, .02, len(P)), s=13,
                  facecolor="none", edgecolor=RED, lw=.8, zorder=4,
                  label=f"selected points ($n={len(P)}$)")
    ax[0].axvline(P.p0_selected.median(), color=RED, lw=.9, ls="--")
    ax[0].set_xlabel(r"floor share $p_0=\varepsilon_\infty/\varepsilon$")
    ax[0].set_ylabel(r"excess log-error $\ln(\varepsilon/\varepsilon_\infty)$  [nats]")
    ax[0].set_title(r"(a) $\ln(\varepsilon/\varepsilon_\infty)=-\ln p_0$")
    ax[0].set_xlim(0.05, 1); ax[0].set_ylim(0, 2.4)
    ax[0].legend(loc="upper right", frameon=False, handletextpad=.3,
                 borderaxespad=.2)
    ax[0].text(.52, .18, f"median $\\hat p_0={P.p0_selected.median():.2f}$",
               color=RED, fontsize=6.5)

    # ---- (b) the running floor --------------------------------------------
    I = R[R.ratio > 1e-6]
    g = I.groupby("window")
    med = g.ratio.median(); q1 = g.ratio.quantile(.25); q3 = g.ratio.quantile(.75)
    f = med.index.to_numpy()
    ax[1].fill_between(f, q1, q3, color=RED, alpha=.15, lw=0)
    ax[1].plot(f, med, "o-", color=RED, ms=3.2, lw=1.3, label="measured (IQR)")
    ax[1].plot(f, g.null_nogap.median(), "^:", color="k", ms=3, lw=1.1,
               label="null: no spectral gap")
    ax[1].plot(f, g.null_gap.median(), "s--", color=BLUE, ms=3, lw=1.1,
               label="null: spectral gap")
    ax[1].axhline(1.0, color=GREY, lw=.6)
    ax[1].set_ylim(0.85, 1.42)
    ax[1].set_xlabel("window, fraction of the log-resource span")
    ax[1].set_ylabel(r"$\hat{\varepsilon}_\infty(X)/\hat{\varepsilon}_\infty(\mathrm{full})$")
    ax[1].set_title("(b) the fitted floor runs")
    ax[1].legend(loc="upper right", frameon=False)
    tw = ax[1].twinx()
    fi = R.groupby("window").apply(lambda d: float((d.ratio > 1e-6).mean()),
                                   include_groups=False)
    tw.plot(fi.index, 100 * fi.to_numpy(), color=GREY, lw=.9, ls="-.")
    tw.set_ylim(0, 165); tw.set_yticks([0, 50, 100])
    tw.set_ylabel("% with an identified floor", fontsize=6.3, color=GREY)
    tw.tick_params(labelsize=6.5, colors=GREY)
    tw.spines[["top"]].set_visible(False)

    # ---- (c) measured width against window statistic ----------------------
    for k, col, mk, lab in ((1, BLUE, "o", r"$\hat{N}=1$"), (2, RED, "s", r"$\hat{N}\geq 2$")):
        h = W[W.N_hat == 1] if k == 1 else W[W.N_hat >= 2]
        ax[2].scatter(h.V, h.g, s=15, facecolor="none", edgecolor=col, lw=.8,
                      marker=mk, label=lab)
    vv = np.logspace(np.log10(max(W.V.min(), 1e-1)), np.log10(W.V.max()), 50)
    ax[2].plot(vv, 1e2 / vv, color="k", lw=1.0, ls="--")
    ax[2].annotate(r"$g\mathcal{V}=10^2$", (vv[26], 1e2 / vv[26]), fontsize=6.5,
                   textcoords="offset points", xytext=(3, 4))
    lab = (W.N_hat >= 2).to_numpy()
    ax[2].text(.02, .93, "AUC   $g$ %.2f,  $\\mathcal{V}$ %.2f,  $g\\mathcal{V}$ %.2f"
               % (_auc(W.g, lab), _auc(W.V, lab), _auc(W.g * W.V, lab)),
               transform=ax[2].transAxes, fontsize=6.5)
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel(r"window statistic $\mathcal{V}$")
    ax[2].set_ylabel(r"measured width $g=\kappa/s^2$")
    ax[2].set_title("(c) measurable resolution")
    ax[2].legend(loc="lower left", frameon=False, handletextpad=.3,
                 borderaxespad=.2)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.55, w_pad=1.4)
    out = os.path.join(FIG, "fig_surprisal.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print("  ->", out)


if __name__ == "__main__":
    main()

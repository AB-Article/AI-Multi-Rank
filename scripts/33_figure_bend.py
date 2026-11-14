#!/usr/bin/env python3
"""Figure for the selection law and the bend divergence (main text, Section
'Selection on the Spectrum').

(a) the master equation: as resources grow the tilted measure is driven toward
    its slowest modes, so the slope s = E_p[a] falls at rate Var_p(a); shown
    for a three-atom spectrum, with -ds/dx and the variance overlaid;
(b) the geometry of B: chord minus terminal tangent on one measured frontier,
    equal to the shaded area under (u - x1) * kappa(u);
(c) the integrated bend against the pointwise criterion: bootstrap z-score
    Z_B and the width-window product g*V, coloured by the recovered channel
    count, with the two AUCs.

Reads results/table_bend_per_frontier.csv; writes figures/fig_bend.pdf.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _loaders import RES, ROOT, selection_rules

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.size": 8, "axes.titlesize": 8,
                            "axes.labelsize": 8, "legend.fontsize": 7,
                            "xtick.labelsize": 7, "ytick.labelsize": 7})
FIG = os.path.join(ROOT, "figures")
BLUE, RED, GREY, GREEN = "#3b6ea5", "#c0504d", "#7f7f7f", "#4e8a55"
sr = selection_rules()


def _auc(score, label):
    s, y = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def local(x, S, xc, h):
    w = np.exp(-0.5 * ((x - xc) / h) ** 2)
    M = np.column_stack([np.ones_like(x), x - xc, 0.5 * (x - xc) ** 2])
    W = np.sqrt(w)[:, None]
    b, *_ = np.linalg.lstsq(M * W, S * W[:, 0], rcond=None)
    return b[0], b[1], -b[2]


def main():
    P = pd.read_csv(os.path.join(RES, "table_bend_per_frontier.csv"))
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))

    # ---- (a) the master equation ------------------------------------------
    a = np.array([0.0, 0.45, 1.6])
    c = np.array([0.05, 0.55, 0.40])
    x = np.linspace(-1.0, 8.0, 600)
    W = c[None, :] * np.exp(-np.outer(x, a))
    eps = W.sum(1)
    p = W / eps[:, None]
    s = (p * a[None, :]).sum(1)
    var = (p * a[None, :] ** 2).sum(1) - s ** 2
    dsdx = np.gradient(s, x)

    ax[0].plot(x, s, color=BLUE, lw=1.6, label=r"$s=\mathbb{E}_{p_x}[a]$")
    ax[0].plot(x, var, color=RED, lw=1.2, label=r"$\mathrm{Var}_{p_x}(a)$")
    ax[0].plot(x, -dsdx, color="k", lw=0.0, marker="o", ms=1.6,
               markevery=18, label=r"$-\,\mathrm{d}s/\mathrm{d}x$")
    for aj, lab in zip(a[1:], ["slow mode", "fast mode"]):
        ax[0].axhline(aj, color=GREY, lw=0.5, ls=":")
    ax[0].set_xlabel(r"log-resource $x$")
    ax[0].set_ylabel("nats per e-fold")
    ax[0].set_title(r"(a) $\mathrm{d}s/\mathrm{d}x=-\mathrm{Var}_{p_x}(a)$")
    ax[0].legend(frameon=False, loc="upper right")
    ax[0].set_xlim(x.min(), x.max())

    # ---- (b) the geometry of B --------------------------------------------
    target = None
    for field, name, cost, xx, ee in sr.single_resource_frontiers():
        if len(ee) >= 18 and np.ptp(xx) > 2.0:
            target = (name, cost, xx, ee)
            break
    name, cost, xx, ee = target
    S = -np.log(ee)
    h = np.ptp(xx) / 4
    x1, x2 = xx.min(), xx.max()
    S1, _, _ = local(xx, S, x1, h)
    S2, s2, _ = local(xx, S, x2, h)
    grid = np.linspace(x1, x2, 200)
    Sg = np.array([local(xx, S, u, h)[0] for u in grid])
    tang = S2 + s2 * (grid - x2)
    ax[1].plot(xx, S, "o", ms=2.6, color=GREY, alpha=.7, label="Pareto points")
    ax[1].plot(grid, Sg, color=BLUE, lw=1.4, label=r"$S(x)$")
    ax[1].plot(grid, tang, color=RED, lw=1.0, ls="--",
               label=r"tangent at $x_2$")
    ax[1].fill_between(grid, Sg, tang, where=tang >= Sg, color=RED, alpha=.10,
                       lw=0)
    ax[1].annotate("", xy=(x1, S1), xytext=(x1, S2 - s2 * (x2 - x1)),
                   arrowprops=dict(arrowstyle="<->", color=RED, lw=1.0))
    ax[1].annotate(r"$B=\mathrm{KL}(p_{x_2}\Vert p_{x_1})$",
                   xy=(x1 + .06 * (x2 - x1), S1 + .55 * (S2 - s2 * (x2 - x1) - S1)),
                   fontsize=7, color=RED)
    ax[1].set_xlabel(r"log-resource $x$")
    ax[1].set_ylabel(r"$S=-\ln\varepsilon$")
    ax[1].set_title("(b) chord minus terminal tangent")
    ax[1].legend(frameon=False, loc="lower right")

    # ---- (c) integrated versus pointwise ----------------------------------
    Q = P.dropna(subset=["Z_B", "gV"]).copy()
    two = Q.N_hat >= 2
    ax[2].scatter(Q.gV[~two], Q.Z_B[~two], s=12, facecolors="none",
                  edgecolors=BLUE, lw=.8, label=r"$\hat N=1$")
    ax[2].scatter(Q.gV[two], Q.Z_B[two], s=14, color=RED, alpha=.75,
                  label=r"$\hat N \geq 2$")
    ax[2].axhline(1, color=GREY, lw=.6, ls=":")
    ax[2].axvline(1, color=GREY, lw=.6, ls=":")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel(r"pointwise  $g\,\mathcal{V}$")
    ax[2].set_ylabel(r"integrated  $Z_B$")
    ax[2].set_title("(c) two visibility criteria")
    ax[2].legend(frameon=False, loc="lower right")
    ax[2].text(.04, .94,
               f"AUC $Z_B$ = {_auc(Q.Z_B, two):.2f}\n"
               f"AUC $g\\mathcal{{V}}$ = {_auc(Q.gV, two):.2f}",
               transform=ax[2].transAxes, va="top", fontsize=7)

    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_bend.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print("  ->", out)


if __name__ == "__main__":
    main()

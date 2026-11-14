#!/usr/bin/env python3
"""Figure for the accuracy-only collapse.

(a) The canonical value J(eps) = 1/2 ln(eps - eps_inf) - ln eps on one measured
    frontier: concave in ln eps, peak at eps = 2 eps_inf, with the 0.05-nat
    band.  The measured candidates are the dots; the rule picks the highest.
(b) The two-candidate rule against the truth.  x is the rule's own signed
    statistic ln(harmonic mean / 2 eps_inf), positive when the rule says take
    the more accurate candidate; y is the measured-resource margin, positive
    when that is in fact right.  The rule is correct in quadrants I and III.
    Disagreements are confined to a band around the origin: they are the pairs
    that are nearly tied, where being wrong is nearly free.
(c) The blind spot.  The accuracy-only choice has R = 0 by construction, so
    all of its cost is invisible in error space.  Its measured-resource regret
    R_sigma is plotted against the fitted law's residual scale, one dot per
    cell: the cost is law misfit and nothing else.

Writes figures/fig_acconly.pdf.
"""
import os, itertools, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _loaders import selection_rules, ROOT

warnings.filterwarnings("ignore")
sr = selection_rules()
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.size": 8, "axes.titlesize": 8,
                            "axes.labelsize": 8, "legend.fontsize": 7,
                            "xtick.labelsize": 7, "ytick.labelsize": 7})
FIG = os.path.join(ROOT, "figures")
BAND = 0.05


def J(e, ei):
    e = np.asarray(e, float); r = e - ei
    o = np.full(e.shape, -np.inf); m = r > 0
    o[m] = 0.5 * np.log(r[m]) - np.log(e[m]); return o


def pick_J(e, ei):
    jv = J(e, ei)
    return int(np.argmax(e)) if not np.isfinite(jv).any() else int(np.argmax(jv))


def panel():
    out = []
    for field, name, cost, x, e in sr.single_resource_frontiers():
        ei, c, a, rms = sr.fit_law(x, e)
        if not (1e-3 < ei < 0.5) or not (1e-3 < a < 5.99):
            continue
        o = np.argsort(x)
        out.append((field, name, cost, x[o], e[o], ei, c, a, rms))
    return out


def main():
    P = panel()

    # ---- choose a legible frontier for (a): attainable, long, well fitted ---
    best = None
    for field, name, cost, x, e, ei, c, a, rms in P:
        if not (e.min() <= 2 * ei <= e.max()) or len(e) < 12:
            continue
        score = np.ptp(np.log(e)) / max(rms, 1e-3)
        if best is None or score > best[0]:
            best = (score, field, name, cost, x, e, ei, c, a, rms)
    _, field, name, cost, x, e, ei, c, a, rms = best

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.45))

    # ------------------------------------------------------------------ (a)
    grid = np.exp(np.linspace(np.log(ei * 1.02), np.log(e.max() * 1.15), 600))
    ax[0].plot(grid / ei, J(grid, ei) - J(2 * ei, ei), "-", lw=1.2,
               color="0.35", zorder=2)
    lo, hi = 1.53, 2.89                      # the 0.05-nat band in eps/eps_inf
    ax[0].axvspan(lo, hi, color="#cfe3f3", alpha=.8, lw=0, zorder=0)
    ax[0].axvline(2, color="#1f6fb4", lw=1.0, ls="--", zorder=1)
    jv = J(e, ei) - J(2 * ei, ei)
    ax[0].plot(e / ei, jv, "o", ms=3.4, mfc="white", mec="0.25", mew=.7, zorder=3)
    k = pick_J(e, ei)
    ax[0].plot(e[k] / ei, jv[k], "o", ms=6.5, mfc="#d9534f", mec="k", mew=.6, zorder=4)
    ax[0].set_xscale("log")
    ax[0].set_xticks([1, 2, 5, 10, 20])
    ax[0].set_xticklabels(["1", "2", "5", "10", "20"])
    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_xlabel(r"$\varepsilon/\varepsilon_\infty$")
    ax[0].set_ylabel(r"canonical value $J$ (nats, rel. max)")
    ax[0].set_ylim(-0.75, 0.06)
    ax[0].set_title("(a) the rule, on one frontier", loc="left")
    ax[0].text(2.05, -0.70, r"$\varepsilon^\star=2\varepsilon_\infty$",
               color="#1f6fb4", fontsize=7)

    # ------------------------------------------------------------------ (b)
    stat, marg = [], []
    for field, name, cost, x, e, ei, c, a, rms in P:
        sig = -np.log(e) - 0.5 * a * (x - x.min())
        for i, k2 in itertools.combinations(range(len(e)), 2):
            hi_, lo_ = (i, k2) if e[i] > e[k2] else (k2, i)
            if e[hi_] == e[lo_]:
                continue
            hm = 2 * e[hi_] * e[lo_] / (e[hi_] + e[lo_])
            stat.append(np.log(hm / (2 * ei)))
            marg.append(float(sig[lo_] - sig[hi_]))
    stat, marg = np.array(stat), np.array(marg)
    good = (stat > 0) == (marg > 0)
    ax[1].axhline(0, color="0.6", lw=.6, zorder=1)
    ax[1].axvline(0, color="0.6", lw=.6, zorder=1)
    ax[1].plot(stat[good], marg[good], ".", ms=1.0, color="#9ec6e0", alpha=.30,
               rasterized=True, zorder=2)
    ax[1].plot(stat[~good], marg[~good], ".", ms=1.0, color="#d9534f", alpha=.30,
               rasterized=True, zorder=3)
    ax[1].set_xlim(-4, 4); ax[1].set_ylim(-1.6, 1.6)
    ax[1].set_xlabel(r"rule: $\ln\,(\mathrm{harmonic\ mean}/2\varepsilon_\infty)$")
    ax[1].set_ylabel(r"truth: margin $\sigma_{\lambda^\star}$ (nats)")
    ax[1].set_title("(b) rule vs truth, all Pareto pairs", loc="left")
    ax[1].text(.03, .96, f"agree {good.mean():.1%}", transform=ax[1].transAxes,
               va="top", fontsize=7, color="#1f6fb4")
    ax[1].text(.97, .96, "both say\nthe accurate one", transform=ax[1].transAxes,
               ha="right", va="top", fontsize=6.2, color="0.35")
    ax[1].text(.03, .04, "both say\nthe cheaper one", transform=ax[1].transAxes,
               va="bottom", fontsize=6.2, color="0.35")

    # ------------------------------------------------------------------ (c)
    Rs, rm = [], []
    for field, name, cost, x, e, ei, c, a, rms in P:
        sig = -np.log(e) - 0.5 * a * (x - x.min())
        iS = int(np.argmax(sig)); k = pick_J(e, ei)
        Rs.append(float(sig[iS] - sig[k])); rm.append(rms)
    Rs, rm = np.array(Rs), np.array(rm)
    FLOORV = 3e-4                                  # draw exact zeros at the axis
    exact = Rs <= 0
    ax[2].scatter(rm[~exact], Rs[~exact], s=20, color="#4878a8",
                  edgecolor="k", linewidth=.35, zorder=3)
    ax[2].scatter(rm[exact], np.full(exact.sum(), FLOORV), s=20, marker="v",
                  color="#9ec6e0", edgecolor="k", linewidth=.35, zorder=3)
    ax[2].axhline(BAND, color="#1f6fb4", lw=1.0, ls="--", zorder=2)
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_ylim(FLOORV / 1.8, max(Rs.max() * 2.2, 1))
    ax[2].set_xlabel("law residual scale (nats)")
    ax[2].set_ylabel(r"$R_\sigma$ of the accuracy-only choice (nats)")
    ax[2].set_title("(c) the blind spot", loc="left")
    rho = pd.Series(Rs).corr(pd.Series(rm), method="spearman")
    ax[2].text(.04, .96, rf"Spearman $\rho={rho:.2f}$", transform=ax[2].transAxes,
               fontsize=7, va="top")
    ax[2].text(.96, .30, "0.05-nat band", transform=ax[2].transAxes, ha="right",
               fontsize=6.5, color="#1f6fb4")
    ax[2].text(.5, .125, r"$\blacktriangledown$  exact, $R_\sigma=0$",
               transform=ax[2].transAxes, ha="center", fontsize=6.4, color="0.35")

    for b in ax:
        b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.5)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig_acconly.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"  -> figures/fig_acconly.pdf   (panel a: {name} | {cost}, "
          f"{len(e)} points; b: {len(stat)} pairs, agree {good.mean():.3f}; "
          f"c: {len(P)} cells, rho={rho:.3f})")


if __name__ == "__main__":
    main()

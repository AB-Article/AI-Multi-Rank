#!/usr/bin/env python3
"""Overview figure for the introduction.

(a) One measured frontier with its fitted floor, the canonical reference
    eps = 2 einf, the 0.05-nat tolerance band, and the cost of the last bit.
(b) Every saturating single-resource frontier in normalized coordinates,
    collapsed onto the shape-free curve eps/einf = 1 + exp(-zeta).

Writes figures/fig_overview.pdf.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _loaders import selection_rules, ROOT

sr = selection_rules()
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.size": 8, "axes.titlesize": 8,
                            "axes.labelsize": 8, "legend.fontsize": 7,
                            "xtick.labelsize": 7, "ytick.labelsize": 7})
FIG = os.path.join(ROOT, "figures")
BAND = 0.638                      # |zeta| <= 0.638 is the 0.05-nat band


def ok(einf, a):
    return (1e-3 < einf < 0.5) and (a < 5.99)


def main():
    frs = sr.single_resource_frontiers()
    curves, best = [], None
    for field, name, cost, x, e in frs:
        einf, c, a, rms = sr.fit_law(x, e)
        if not (ok(einf, a) and e.min() <= 2 * einf <= e.max()):
            continue
        zeta = a * (x - x.min()) - np.log(c / einf)
        curves.append((field, zeta, e / einf))
        istar = sr.nearest(np.log(e), np.log(2 * einf))
        frac = (x[istar] - x.min()) / np.ptp(x)
        good = len(e) >= 12 and 0.2 <= frac <= 0.7 and np.ptp(np.log(e)) > 0.9
        score = (np.ptp(x) / max(rms, 1e-3),) if good else (-np.inf,)
        if best is None or score > best[0]:
            best = (score, field, name, cost, x, e, einf, c, a)

    _, field, name, cost, x, e, einf, c, a = best
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))

    # ---- (a) one measured frontier ------------------------------------
    t = np.exp(x)
    ax[0].scatter(t, e, s=13, color="#3b6ea5", zorder=3, label="measured models")
    tg = np.exp(np.linspace(x.min(), x.max(), 200))
    ax[0].plot(tg, einf + c * np.exp(-a * (np.log(tg) - x.min())), color="#3b6ea5",
               lw=1.1, alpha=.8, label=r"fit $\epsilon_\infty + c\,t^{-\alpha}$")
    ax[0].axhline(einf, color="#888888", ls=":", lw=1.0)
    ax[0].axhline(2 * einf, color="#c0504d", ls="--", lw=1.0)
    tstar = np.exp(x.min() + np.log(c / einf) / a)
    ax[0].axvspan(tstar * np.exp(-BAND / a), tstar * np.exp(BAND / a),
                  color="#c0504d", alpha=.10, lw=0)
    i = sr.nearest(np.log(e), np.log(2 * einf))
    ax[0].scatter([t[i]], [e[i]], s=45, facecolor="none", edgecolor="#c0504d",
                  lw=1.4, zorder=4)
    ax[0].annotate(r"reference $\epsilon=2\epsilon_\infty$",
                   xy=(t[i], e[i]), xytext=(0.03, 0.10), textcoords="axes fraction",
                   color="#c0504d",
                   arrowprops=dict(arrowstyle="->", color="#c0504d", lw=0.8))
    ax[0].annotate("", xy=(t[-1], e[-1]), xytext=(t[i], e[i]),
                   arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8, ls="-"))
    bits = np.log2(e[i] / e[-1]); mult = t[-1] / t[i]
    ax[0].text(0.50, 0.34, f"last {bits:.2f} bits cost {mult:.0f}$\\times$",
               transform=ax[0].transAxes, color="#444444")
    ax[0].text(1.02 * t[0], einf, r"$\epsilon_\infty$", color="#666666",
               va="bottom", ha="left")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(f"resource ({cost})")
    ax[0].set_ylabel(r"chance-corrected error $\epsilon$")
    ax[0].set_title(f"(a) {name}", loc="left")
    ax[0].legend(frameon=False, loc="upper right")

    # ---- (b) collapse --------------------------------------------------
    fields = sorted({f for f, _, _ in curves})
    cmap = plt.get_cmap("tab10")
    col = {f: cmap(i % 10) for i, f in enumerate(fields)}
    seen = set()
    for f, z, r in curves:
        ax[1].plot(z, r, "o", ms=2.2, alpha=.45, color=col[f],
                   label=f if f not in seen else None)
        seen.add(f)
    zg = np.linspace(-3.5, 6, 200)
    ax[1].plot(zg, 1 + np.exp(-zg), color="black", lw=1.3,
               label=r"$1+e^{-\zeta}$")
    ax[1].axvspan(-BAND, BAND, color="#c0504d", alpha=.10, lw=0)
    ax[1].axvline(0, color="#c0504d", ls="--", lw=1.0)
    ax[1].set_yscale("log"); ax[1].set_xlim(-3.5, 6); ax[1].set_ylim(0.8, 60)
    ax[1].set_xlabel(r"normalized resource $\zeta$")
    ax[1].set_ylabel(r"$\epsilon/\epsilon_\infty$")
    ax[1].set_title(f"(b) {len(curves)} frontiers, one curve", loc="left")
    ax[1].legend(frameon=False, loc="upper right", ncol=2, handletextpad=.4,
                 columnspacing=.8)

    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig_overview.pdf")
    fig.savefig(out); print("  ->", out, f"| panel (a): {name} [{cost}]")


if __name__ == "__main__":
    main()

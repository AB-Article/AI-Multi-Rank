#!/usr/bin/env python3
"""The ladder, the channel count and the knee (the 'ladder' figure of paper.tex).

(a) The admissible STRIP.  Cauchy-Schwarz gives the upper wall w_2 = 1 - w
    (floor + one channel); the upper principal representation gives the mirror
    wall w_2 = -rho + w/rho, which depends on the fastest mode through
    rho = (a_max - s)/s and is tight on any two-atom spectrum.  Every completely
    monotone frontier lives between them, and the strip closes exactly when
    rho = w.
(b) Nats per effective channel.  S* = S w is the floorless chart's conserved
    quantity and the reportable form of the shape index: one new effective
    channel per S* nats.  Distribution over the archive, invariant route against
    the fitted (1-beta)/beta.
(c) The knee.  W = (r-1)^2/(4r) reaches 1 only at r = 3 + 2 sqrt 2, so channels
    closer than that factor produce no canonical point -- and the plug-in W is
    unusable at these window lengths, which the right-hand bars show: its null
    median is 13 for a truth whose W is 3.4 and 1.0 for a truth whose W is
    infinite.

Writes figures/fig_ladder.pdf.
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                            "axes.titlesize": 8, "axes.labelsize": 8,
                            "legend.fontsize": 6.3,
                            "xtick.labelsize": 7, "ytick.labelsize": 7})
BLUE, RED, GREY, GREEN = "#3b6ea5", "#c0504d", "#888888", "#4f8a5b"


def main():
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(7.3, 2.35))

    # ----------------------------------------------------------- panel a ---
    a0 = ax[0]
    w = np.linspace(0.02, 4.0, 400)
    a0.plot(w, 1 - w, color=RED, lw=1.4, label=r"$w_2=1-w$  (floor $+$ 1 channel)")
    for rho, ls in ((0.5, ":"), (1.5, "--"), (4.0, "-.")):
        a0.plot(w, -rho + w / rho, color=BLUE, lw=1.0, ls=ls,
                label=rf"mirror wall, $\rho={rho}$")
    a0.fill_between(w, -rho + w / rho, 1 - w, where=(1 - w) > (-rho + w / rho),
                    color=GREY, alpha=.16, lw=0)
    a0.plot(w, -2 * w - 1 / 3.0, color=GREEN, lw=1.2,
            label=r"floorless chart, $S=3$")
    a0.axvline(1, color="k", lw=.6, alpha=.5)
    a0.set_xlim(0, 4); a0.set_ylim(-6.4, 1.8)
    a0.set_xlabel(r"$w=\kappa/s^2=1/N_{\rm eff}$")
    a0.set_ylabel(r"$w_2=\mathrm{d}\ln\kappa/\mathrm{d}S$")
    a0.set_title("(a) the admissible strip")
    a0.legend(frameon=False, loc="lower left", fontsize=5.6)

    # ----------------------------------------------------------- panel b ---
    a1 = ax[1]
    K = pd.read_csv(os.path.join(RESULTS, "table_knee_archive.csv"))
    inv = K.S_star_invariant.dropna(); fit = K.S_route_beta.dropna()
    bins = np.logspace(np.log10(0.2), np.log10(60), 22)
    a1.hist(inv, bins=bins, color=BLUE, alpha=.7, label=r"invariant, $S^\star=Sw$")
    a1.hist(fit, bins=bins, histtype="step", color=RED, lw=1.3,
            label=r"fitted, $(1-\beta)/\beta$")
    a1.set_xscale("log")
    a1.axvline(float(inv.median()), color=BLUE, lw=.9, ls="--")
    a1.axvline(float(fit.median()), color=RED, lw=.9, ls="--")
    a1.set_xlabel(r"$S^\star$: nats of log-error per effective channel")
    a1.set_ylabel("frontiers")
    a1.set_title("(b) one channel per $S^\\star$ nats")
    a1.legend(frameon=False, loc="upper right")

    # ----------------------------------------------------------- panel c ---
    a2 = ax[2]
    r = np.logspace(np.log10(1.2), np.log10(60), 400)
    rs = 3 + 2 * np.sqrt(2)
    a2.fill_between(r, 1e-2, 1, where=r < rs, color=RED, alpha=.08, lw=0)
    a2.plot(r, (r - 1) ** 2 / (4 * r), color=BLUE, lw=1.5, label=r"$W=(r-1)^2/4r$")
    a2.axhline(1, color="k", lw=.6)
    a2.axvline(rs, color=RED, lw=1.0, ls="--")
    a2.text(rs*1.10, 1.5e-2, r"$r=3+2\sqrt{2}$", color=RED, fontsize=6.3)
    a2.text(1.26, 1.6e-1, "no knee", color=RED, fontsize=6.5)
    a2.text(1.26, 9.0e-2, "exists", color=RED, fontsize=6.5)
    a2.set_xscale("log"); a2.set_yscale("log"); a2.set_ylim(1e-2, 40)
    a2.set_xlabel(r"exponent ratio $r=a_2/a_1$")
    a2.set_ylabel(r"$W=\sup_x w$")
    a2.set_title("(c) when a knee exists")
    a2.legend(frameon=False, loc="upper left", fontsize=6.0)

    # inset: the plug-in W is not usable at these window lengths
    N = pd.read_csv(os.path.join(RESULTS, "table_knee_null_summary.csv"))
    g = dict(zip(N.quantity, N.value))
    ins = a2.inset_axes([0.55, 0.13, 0.42, 0.33])
    val = [g.get("C1 median TRUE W of that truth over the same window", np.nan),
           g.get("C1 median plug-in W under the FLOORLESS truth (beta=0.25)", np.nan),
           g.get("C2 median plug-in W under the FLOOR truth (W true = infinity)", np.nan)]
    ins.bar([0, 1, 2], val, color=[GREEN, GREY, RED], alpha=.65, width=.7)
    ins.set_yscale("log"); ins.set_ylim(0.4, 40)
    ins.set_xticks([])

    ins.tick_params(axis="y", labelsize=5.2, length=2); ins.tick_params(axis="x", length=0)
    ins.set_title(r"true / plug-in / plug-in (floor)", fontsize=5.0, pad=2)

    fig.tight_layout(w_pad=1.6)
    out = os.path.join(FIG, "fig_ladder.pdf")
    fig.savefig(out, bbox_inches="tight"); print(f"  -> figures/fig_ladder.pdf")


if __name__ == "__main__":
    main()

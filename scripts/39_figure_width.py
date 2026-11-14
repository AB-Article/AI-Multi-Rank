#!/usr/bin/env python3
"""Figure for the width coordinate (main text, Section 'The Width Is the
Normalized Resource').

(a) zeta = ln g = ln(kappa/s^2) against log-resource on every measured
    frontier, each shifted to its own zero crossing.  The theory says zeta is
    affine with slope the exponent and that eps = 2 eps_inf where it crosses
    zero; the grey band is the interquartile range of the measured windows.
(b) the slope of that line against the exponent of a three-parameter fit: the
    prediction is the diagonal.
(c) the membership statistic R = (d zeta/dx)/alpha_hat.  It is exactly 1 for
    one channel plus a floor and exactly 0 for a gapless Gamma spectrum.
    Synthetic two- and three-atom spectra at a realistic design are separated
    by it; the measured corpus sits above both, which is a membership failure
    rather than a channel count.

Recomputes the zeta profiles from the corpus (cheap, no bootstrap) and reads
results/table_width_per_frontier.csv and table_width_power.csv.
Writes figures/fig_width.pdf.
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _loaders import RES, ROOT, selection_rules

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.size": 8, "axes.titlesize": 8,
                            "axes.labelsize": 8, "legend.fontsize": 7,
                            "xtick.labelsize": 7, "ytick.labelsize": 7})
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
BLUE, RED, GREY, GREEN = "#3b6ea5", "#c0504d", "#7f7f7f", "#4e8a55"

sr = selection_rules()
W = pd.read_csv(os.path.join(RES, "table_width_per_frontier.csv"))
P = pd.read_csv(os.path.join(RES, "table_width_power.csv"))

RNG = np.random.default_rng(11)


def local(x, S, xc, h):
    w = np.exp(-0.5 * ((x - xc) / h) ** 2)
    M = np.column_stack([np.ones_like(x), x - xc, 0.5 * (x - xc) ** 2])
    V = np.sqrt(w)[:, None]
    b, *_ = np.linalg.lstsq(M * V, S * V[:, 0], rcond=None)
    return b[0], b[1], -b[2]


def zeta_curve(x, e, ng=41, trim=0.10):
    S = -np.log(e); span = np.ptp(x); h = span / 4
    gr = np.linspace(x.min() + trim * span, x.max() - trim * span, ng)
    o = np.array([local(x, S, xc, h) for xc in gr])
    s, kap = o[:, 1], o[:, 2]
    ok = (s > 1e-6) & (kap > 1e-12)
    if ok.sum() < 10:
        return None
    return gr[ok], np.log(kap[ok] / s[ok] ** 2)


def width_fit_light(x, e):
    c = zeta_curve(x, e)
    if c is None:
        return None
    u, z = c
    b1, b0 = np.polyfit(u, z, 1)
    return u, z, b1, b0


fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))

# ---------------------------------------------------------------- panel (a)
a0 = ax[0]
lo, hi = [], []
for field, name, cost, x, e in sr.single_resource_frontiers():
    x = np.asarray(x, float); e = np.asarray(e, float)
    if len(e) < 15 or np.ptp(x) < 1.0:
        continue
    r = width_fit_light(x, e)
    if r is None:
        continue
    u, z, b1, b0 = r
    if abs(b1) < 1e-9:
        continue
    xs = -b0 / b1
    a0.plot(u - xs, z, lw=.8, color=BLUE, alpha=.45)
    lo.append(u[0] - xs); hi.append(u[-1] - xs)
a0.axhline(0, color=RED, lw=1.0, zorder=5)
a0.axvline(0, color=RED, lw=1.0, ls=":", zorder=5)
if lo:
    a0.axvspan(np.percentile(lo, 25), np.percentile(hi, 75),
               color=GREY, alpha=.14, zorder=0)
a0.set_xlim(-4.5, 4.5); a0.set_ylim(-4.5, 8.0)
a0.set_xlabel(r"$x-x^{\star}$  (nats of log-resource)")
a0.set_ylabel(r"$\zeta=\ln g=\ln(\kappa/s^{2})$")
a0.set_title(r"(a) the width coordinate is affine")
a0.text(.03, .05,
        r"$\zeta=0\ \Leftrightarrow\ p_0=1/2\ \Leftrightarrow\ "
        r"\varepsilon=2\varepsilon_\infty$",
        transform=a0.transAxes, va="bottom", fontsize=7, color=RED)

# ---------------------------------------------------------------- panel (b)
a1 = ax[1]
m = np.isfinite(W.alpha_fit) & (W.alpha_fit > 0) & np.isfinite(W.slope)
a1.scatter(W.alpha_fit[m], W.slope[m], s=16, color=BLUE, alpha=.85, zorder=3)
lim = [0.05, max(W.slope[m].max(), W.alpha_fit[m].max()) * 1.35]
a1.plot(lim, lim, color=RED, lw=1.0, zorder=2, label=r"$d\zeta/dx=\alpha$")
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(lim); a1.set_ylim(lim)
a1.set_xlabel(r"fitted exponent $\alpha$ (three-parameter)")
a1.set_ylabel(r"slope $d\zeta/dx$ (per nat)")
a1.set_title("(b) the slope is the exponent")
a1.legend(loc="lower right", frameon=False)
rho = np.corrcoef(np.log(W.alpha_fit[m]), np.log(W.slope[m]))[0, 1]
a1.text(.03, .95, f"$n={int(m.sum())}$\n$r={rho:.2f}$ on logs\n"
                  f"median ratio ${np.median(W.slope[m]/W.alpha_fit[m]):.2f}$",
        transform=a1.transAxes, va="top", fontsize=7)

# ---------------------------------------------------------------- panel (c)
a2 = ax[2]
row = P[(P.span_nats == 4.7) & (P.sigma == 0.01)]
row = row.iloc[0] if len(row) else P.iloc[-1]
box = [
    ("two atoms\n(synthetic)", row.R_two_p05, row.R_two_median, row.R_two_p95, GREEN),
    ("three atoms\n(synthetic)", row.R_three_p05, row.R_three_median, row.R_three_p95, BLUE),
    ("measured\ncorpus", float(W.R.quantile(.05)), float(W.R.median()),
     float(W.R.quantile(.95)), RED)]
for i, (lab, p5, med, p95, col) in enumerate(box):
    a2.plot([p5, p95], [i, i], lw=2.4, color=col, solid_capstyle="round", alpha=.75)
    a2.plot([med], [i], "o", ms=6, color=col, zorder=4)
# The synthetic rows above are centred on the crossover, where the estimator
# is unbiased.  The measured windows are not, so the reference for the measured
# row is the per-frontier calibrated null of script 40, not the analytic 1.
_cal = os.path.join(RES, "table_width_null_summary.csv")
_null4 = None
if os.path.exists(_cal):
    _c = pd.read_csv(_cal)
    _r = _c[_c.bandwidth == "span/4"]
    if len(_r):
        _null4 = float(_r.null_median.iloc[0])
a2.axvline(1.0, color="k", lw=1.0, ls=":", zorder=2)
if _null4 is not None:
    a2.axvline(_null4, color=RED, lw=1.4, ls="-", zorder=3)
    a2.annotate(f"calibrated null for the\nmeasured windows ({_null4:.2f})",
                xy=(_null4, -0.55), xytext=(_null4 + 0.22, -0.52),
                color=RED, fontsize=6.5, va="top", ha="left")
a2.axvline(0.0, color=GREY, lw=1.0, ls=":", zorder=2)
a2.set_yticks(range(len(box))); a2.set_yticklabels([b[0] for b in box])
a2.set_xlabel(r"$R=(d\zeta/dx)\,/\,\hat\alpha$")
a2.set_title(r"(c) $R=1$ only on a centred window")
a2.set_xlim(-0.6, 3.2)
a2.set_ylim(-.6, len(box) - .4)
a2.invert_yaxis()

for a, t in zip(ax, "abc"):
    a.tick_params(direction="out", length=2.5)
fig.tight_layout()
out = os.path.join(FIG, "fig_width.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"  -> figures/fig_width.pdf")

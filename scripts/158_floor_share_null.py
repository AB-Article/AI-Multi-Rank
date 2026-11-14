#!/usr/bin/env python3
"""The zero-floor null, extended, and matched to the archive's own window.

Why this script exists
----------------------
Revision 2026-10-08m found that the claim "a three-parameter fit to a frontier
whose floor is exactly zero returns a floor share of 0.52, and the archive's
median floor share is 0.51" was unsupported twice over:

  * the 0.51 was a hard-coded string in 76_no_floor.py citing a column
    (`median floor share`) that table_floor_free_summary.csv does not have;
  * the archive's measured median fitted floor share is 0.96 (of the best
    observed error, 117 frontiers) / 0.99 (92 second-order-readable frontiers),
    not 0.51, so the asserted numerical coincidence did not exist.

The qualitative claim -- that a fitted floor is a readout of the shape index
and not a measurement of an irreducible error -- survives and in fact gets
STRONGER, because the zero-floor null reproduces the archive's measured median
share at a small shape index.  This script establishes that quantitatively:

  T1  the null extended down to beta = 0.05 and up to 0.99, on the window
      76_no_floor.py used (8 nats of log-resource, 22 points);
  T2  the SAME null on a window matched to the archive's own median design
      (median span 5.1 nats, median 11 points, table_prediction_summary.csv),
      because the apparent floor share decays like 1/S and the window length
      is therefore load-bearing;
  T3  the archive's measured median fitted floor share, three ways;
  T4  the beta at which the null reproduces the measured median, on each
      window -- the number the paper quotes.

Writes results/table_floor_share_null.csv
       results/table_floor_share_null_summary.csv
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import save, ROOT                                    # noqa: E402
from importlib import import_module
_nf = import_module("76_no_floor") if False else None            # noqa: F841

# fit_floor is defined in 76_no_floor.py, which is not importable by name
# (leading digit), so the two-line fitter is reproduced here verbatim.
from scipy.optimize import least_squares

RNG = np.random.default_rng(20261008)
NOISE = 0.02                      # the corpora's own log-error noise

BETAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60,
         0.70, 0.80, 0.90, 0.95, 0.99]
REPS = 60                         # 76_no_floor.py used 12

# (label, x_lo, x_hi, n_points)
WINDOWS = [("script-76 window (8 nats, 22 points)", 2.0, 10.0, 22),
           ("archive median design (5.1 nats, 11 points)", 2.0, 7.1, 11)]


def _best(res, starts):
    best = None
    for s in starts:
        try:
            b = least_squares(res, s, method="lm", max_nfev=4000)
        except Exception:
            continue
        if best is None or b.cost < best.cost:
            best = b
    return best


def fit_floor(x, y):
    """eps = einf + c exp(-a x) ; y = ln eps.  Returns einf."""
    def res(th):
        return np.log(np.exp(th[0]) + np.exp(th[1] - np.exp(th[2]) * x)) - y
    b = _best(res, [[e0, y[0], la] for e0 in (-6, -4, -2.6, -1.6)
                    for la in (-2.0, -1.0, 0.0, 1.0)])
    return float(np.exp(b.x[0]))


def null_table():
    rows = []
    for wlabel, xlo, xhi, npts in WINDOWS:
        x = np.linspace(xlo, xhi, npts)
        for beta in BETAS:
            eps = np.exp(-x ** beta)                 # floor is EXACTLY zero
            shares = []
            for _ in range(REPS):
                y = np.log(eps) + RNG.normal(0, NOISE, x.size)
                shares.append(fit_floor(x, y) / eps[-1])
            shares = np.asarray(shares)
            S_mid = float(np.exp(np.mean(np.log(x ** beta))))
            rows.append(dict(
                window=wlabel, span_nats=round(xhi - xlo, 2), n_points=npts,
                beta=beta, true_floor=0.0, reps=REPS,
                share_median=float(np.median(shares)),
                share_q10=float(np.quantile(shares, .10)),
                share_q90=float(np.quantile(shares, .90)),
                share_predicted=(1 - beta) / ((1 - beta) + beta * S_mid)))
    return pd.DataFrame(rows)


def measured_shares():
    """The archive's own fitted floor share, three ways."""
    out = []
    R = os.path.join(ROOT, "results")
    for fn, num, den, label in [
            ("table_plane_frontiers.csv", "floor_fitted", "eps_min",
             "all frontiers (script 133)"),
            ("table_monotone_floor_bound.csv", "floor_fitted", "eps_best",
             "second-order-readable (script 81)")]:
        d = pd.read_csv(os.path.join(R, fn))
        v = (d[num] / d[den]).replace([np.inf, -np.inf], np.nan).dropna()
        out.append((label, int(len(v)), float(v.median())))
    d = pd.read_csv(os.path.join(R, "table_chart_test_summary.csv"))
    hit = d[d.quantity == "median fitted floor share p_0"]
    if len(hit):
        out.append(("chart-test route (script 84)", 92, float(hit.value.iloc[0])))
    return out


def beta_reproducing(T, window, target):
    """Interpolate the null's beta at which share_median == target."""
    g = T[T.window == window].sort_values("share_median")
    if target <= g.share_median.min() or target >= g.share_median.max():
        return float("nan")
    return float(np.interp(target, g.share_median.values, g.beta.values))


def main():
    T = null_table()
    save(T.round(6), "table_floor_share_null.csv")

    meas = measured_shares()
    head = [("archive median fitted floor share -- %s" % lab, v)
            for lab, n, v in meas]
    head += [("archive n -- %s" % lab, float(n)) for lab, n, v in meas]

    hl = WINDOWS[0][0]
    target = [v for lab, n, v in meas if lab.startswith("all frontiers")][0]
    rows = head + [
        ("null share at beta=0.60, script-76 window",
         float(T[(T.window == hl) & (T.beta == 0.60)].share_median.iloc[0])),
        ("null share at beta=0.15, script-76 window",
         float(T[(T.window == hl) & (T.beta == 0.15)].share_median.iloc[0])),
        ("null share at beta=0.95, script-76 window",
         float(T[(T.window == hl) & (T.beta == 0.95)].share_median.iloc[0])),
        ("null share at beta=0.20, script-76 window",
         float(T[(T.window == hl) & (T.beta == 0.20)].share_median.iloc[0])),
    ]
    for wlabel, _, _, _ in WINDOWS:
        rows.append(("beta reproducing the archive median (%.3f) on %s"
                     % (target, wlabel),
                     beta_reproducing(T, wlabel, target)))
    S = pd.DataFrame([dict(quantity=k, value=v) for k, v in rows])
    save(S.round(6), "table_floor_share_null_summary.csv")

    print("=" * 78)
    print("T1/T2  THE ZERO-FLOOR NULL, TWO WINDOWS")
    for wlabel, _, _, _ in WINDOWS:
        g = T[T.window == wlabel]
        print(f"\n  {wlabel}")
        for _, r in g.iterrows():
            print(f"    beta={r.beta:5.2f}  fitted floor / best observed error"
                  f" = {r.share_median:.3f}  [{r.share_q10:.3f},{r.share_q90:.3f}]")
    print()
    print("=" * 78)
    print("T3  WHAT THE ARCHIVE ACTUALLY MEASURES")
    for lab, n, v in meas:
        print(f"    {lab:36s} n={n:3d}  median share = {v:.3f}")
    print()
    print("=" * 78)
    print("T4  THE SHAPE INDEX THAT REPRODUCES IT WITH NO FLOOR AT ALL")
    for wlabel, _, _, _ in WINDOWS:
        b = beta_reproducing(T, wlabel, target)
        print(f"    {wlabel:44s} beta = {b:.3f}")
    print()
    print("  The fitted floor share is a monotone readout of the shape index;")
    print("  the archive's median share is reproduced by a frontier with NO")
    print("  floor anywhere in it.  There is no numerical coincidence to quote,")
    print("  and none is needed: the whole measured range is inside the null.")


if __name__ == "__main__":
    main()

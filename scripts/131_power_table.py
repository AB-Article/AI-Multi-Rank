#!/usr/bin/env python3
"""131_power_table.py -- the power table the programme has been missing.

Scripts 129 and 130 establish that no bandwidth-free divisibility statistic has
power on this corpus at its published precision.  That is a statement about the
corpus, not about the test, and the useful form of it is a NUMBER: the per-model
precision, and the admissible span, at which the test would decide.

The computation is analytic, so it covers every frontier.  Let the quadruple have
divided-difference weights w, let dd3_div be the third divided difference of
S = -ln eps under the divisible chart fitted to this frontier's design and
dd3_alt the same under the two-atom chart fitted to it.  With i.i.d. Gaussian
per-model noise of scale sigma,

    z = dd3/(sigma ||w||_2) ~ N( dd3_true/(sigma ||w||_2), 1 ),

so the one-sided 95 % cut taken from the divisible null is
dd3_div/(sigma||w||_2) - z95 and the power against the two-atom alternative is

    Phi( (dd3_div - dd3_alt)/(sigma ||w||_2) - z95 ).                      (*)

Hence the precision at which the test reaches power 1-beta is, in closed form,

    sigma_* = (dd3_div - dd3_alt) / ( ||w||_2 (z95 + z_{1-beta}) ).        (**)

(*) is checked against the simulated power of script 130 before (**) is used.

Reported per frontier, for the widest quadruple placed at or left of the measured
curvature maximum (Prop. 'tangent': the test has no power to its right):
sigma_80, the ratio to the frontier's own residual scale, and the admissible span.

Outputs: results/table_power_requirement.csv
         results/table_power_summary.csv
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save, all_frontiers                        # noqa: E402

Z95, Z80 = norm.ppf(0.95), norm.ppf(0.80)
NOMINAL = 0.020            # the archive's quoted per-model noise, in nats
rng = np.random.default_rng(7)


def dd(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return float(sum(ys[i] / np.prod([xs[i] - xs[j] for j in range(len(xs)) if j != i])
                     for i in range(len(xs))))


def dd_w(xs):
    xs = np.asarray(xs, float)
    return np.array([1.0 / np.prod([xs[i] - xs[j] for j in range(len(xs)) if j != i])
                     for i in range(len(xs))])


def fit(x, L, kind):
    if kind == "cure":
        def f(th, xx):
            return -np.exp(th[0]) * (1 - np.exp(-np.exp(th[1]) * (xx - th[2])))
        th0 = [np.log(max(-L.min(), 1e-3)), np.log(0.5), x[0] - 0.5 * (x[-1] - x[0])]
    else:
        def f(th, xx):
            return np.log(np.exp(th[0]) + np.exp(th[1]) * np.exp(-np.exp(th[2]) * xx))
        th0 = [np.log(max(np.exp(L.min()) * .5, 1e-9)), L.max(), np.log(.4)]
    try:
        r = least_squares(lambda th: f(th, x) - L, th0, max_nfev=20000)
        p = f(r.x, x)
        if not np.all(np.isfinite(p)):
            return None
        return p, float(np.sqrt(np.sum(r.fun ** 2) / max(len(x) - len(th0), 1)))
    except Exception:
        return None


def quad(x, right=None):
    ok = np.where(x <= right + 1e-12)[0] if right is not None else np.arange(len(x))
    if len(ok) < 4:
        return None
    idx = sorted({ok[0], ok[max(1, len(ok) // 3)],
                  ok[min(len(ok) - 2, (2 * len(ok)) // 3)], ok[-1]})
    return np.array(idx) if len(idx) == 4 else None


def run():
    rows, checks = [], []
    for fr in all_frontiers():
        x, e = np.asarray(fr["x"], float), np.asarray(fr["e"], float)
        o = np.argsort(x); x, e = x[o], e[o]
        m = (e > 0) & np.isfinite(e); x, e = x[m], e[m]
        if len(x) < 6:
            continue
        L = np.log(e)
        fd, fa = fit(x, L, "cure"), fit(x, L, "two_atom")
        if fd is None or fa is None:
            continue
        sigma = float(np.median([fd[1], fa[1]]))
        kx = [(x[j + 1], 2.0 * dd(x[j:j + 3], L[j:j + 3])) for j in range(len(x) - 2)]
        kx = [(a, b) for a, b in kx if np.isfinite(b)]
        if not kx:
            continue
        x_el = max(kx, key=lambda p: p[1])[0]
        for tag, idx in (("placed", quad(x, x_el)), ("whole", quad(x))):
            if idx is None:
                continue
            w = dd_w(x[idx])
            nw = float(np.linalg.norm(w))
            d_div = dd(x[idx], -fd[0][idx])
            d_alt = dd(x[idx], -fa[0][idx])
            gap = d_div - d_alt                       # positive => the test can see it
            sig80 = gap / (nw * (Z95 + Z80)) if gap > 0 else np.nan
            rows.append(dict(
                corpus=fr["corpus"], name=fr["name"], placement=tag, n=len(x),
                span=float(x[-1] - x[0]),
                admissible_span=float(x_el - x[0]), n_admissible=int(np.sum(x <= x_el + 1e-12)),
                quad_span=float(x[idx[-1]] - x[idx[0]]), wnorm=nw,
                dd3_div=d_div, dd3_alt=d_alt, gap=gap, sigma_resid=sigma,
                sigma_80=sig80,
                ratio_to_resid=(sigma / sig80 if sig80 and sig80 > 0 else np.nan),
                ratio_to_nominal=(NOMINAL / sig80 if sig80 and sig80 > 0 else np.nan),
                power_at_resid=(norm.cdf(gap / (sigma * nw) - Z95) if gap > 0 else np.nan),
                power_at_nominal=(norm.cdf(gap / (NOMINAL * nw) - Z95) if gap > 0 else np.nan),
            ))
            if tag == "placed" and gap > 0 and len(checks) < 12:
                # verify (*) by simulation on this design
                B = 4000
                zc = (d_div / (sigma * nw)) - Z95
                za = d_alt / (sigma * nw) + rng.standard_normal(B)
                checks.append((float(np.mean(za < zc)),
                               float(norm.cdf(gap / (sigma * nw) - Z95))))
    df = pd.DataFrame(rows)

    print("\n  CHECK of the analytic power (*) against simulation on 12 designs:")
    err = max(abs(a - b) for a, b in checks) if checks else float("nan")
    print(f"    max |simulated - analytic| = {err:.4f}   (4000 replicates each)")

    for tag in ("placed", "whole"):
        d = df[df.placement == tag].dropna(subset=["sigma_80"])
        print(f"\n  {tag.upper()} quadruple ({len(d)} frontiers of {df.name.nunique()})")
        print(f"    median quadruple span ...................... {d.quad_span.median():.2f} e-folds")
        print(f"    median sigma_80 ............................ "
              f"{d.sigma_80.median()*1000:.2f} milli-nats"
              f"   IQR [{d.sigma_80.quantile(.25)*1000:.2f}, {d.sigma_80.quantile(.75)*1000:.2f}]")
        print(f"    the archive's residual scale is ............ "
              f"{d.sigma_resid.median()*1000:.1f} milli-nats  -> needs x"
              f"{(d.sigma_resid/d.sigma_80).median():.1f}")
        print(f"    against the nominal 20 milli-nats .......... needs x"
              f"{(NOMINAL/d.sigma_80).median():.1f}")
        print(f"    frontiers already decidable at 20 milli-nats "
              f"{int((d.power_at_nominal >= 0.8).sum())} of {len(d)}"
              f"   (at the residual scale: {int((d.power_at_resid >= 0.8).sum())})")

    pl = df[df.placement == "placed"].dropna(subset=["sigma_80"])
    wh = df[df.placement == "whole"].dropna(subset=["sigma_80"])
    j = pl.merge(wh, on=["corpus", "name"], suffixes=("_pl", "_wh"))
    if len(j):
        print(f"\n  PLACEMENT GAIN, like for like on {len(j)} frontiers:")
        print(f"    median sigma_80 placed / whole-window ...... "
              f"{(j.sigma_80_pl / j.sigma_80_wh).median():.2f}"
              f"   (below 1 means placement COSTS precision here)")
        print(f"    median admissible span ..................... "
              f"{j.admissible_span_pl.median():.2f} of {j.span_pl.median():.2f} e-folds")
    save(df, "table_power_requirement.csv")
    # every field below is named for the subset it describes, so that a number
    # quoted anywhere can be traced to exactly one column.
    row = dict(analytic_check_max_err=err,
               n_placed=len(pl), n_whole=len(wh),
               median_sigma80_placed=float(pl.sigma_80.median()),
               median_sigma80_whole=float(wh.sigma_80.median()),
               median_resid_placed=float(pl.sigma_resid.median()),
               median_resid_whole=float(wh.sigma_resid.median()),
               factor_vs_nominal_placed=float((NOMINAL / pl.sigma_80).median()),
               factor_vs_nominal_whole=float((NOMINAL / wh.sigma_80).median()),
               factor_vs_resid_whole=float((wh.sigma_resid / wh.sigma_80).median()),
               decidable_at_nominal_placed=int((pl.power_at_nominal >= 0.8).sum()),
               decidable_at_nominal_whole=int((wh.power_at_nominal >= 0.8).sum()),
               decidable_at_resid_whole=int((wh.power_at_resid >= 0.8).sum()))
    if len(j):
        # the placement cost is a MEDIAN OF RATIOS on the frontiers where both
        # placements exist -- not a ratio of medians, which mixes two subsets.
        row.update(n_both=len(j),
                   placement_cost=float((j.sigma_80_wh / j.sigma_80_pl).median()),
                   median_admissible_span_both=float(j.admissible_span_pl.median()),
                   median_span_both=float(j.span_pl.median()))
    save(pd.DataFrame([row]), "table_power_summary.csv")


if __name__ == "__main__":
    run()

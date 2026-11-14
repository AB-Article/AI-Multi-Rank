#!/usr/bin/env python3
"""130_plane_verdict.py -- the divisibility verdict, with a design-matched null.

The method law of this archive is that a plug-in estimator of an exact identity
must be compared against a null simulated on the SAME design, never against its
analytic value.  Six casualties so far.  This script applies it to the plane
readings of script 129 before any of them is allowed to carry a verdict.

For every frontier, three charts are fitted to the frontier's own design and its
own residual scale:

    NULL  1  the cure chart   eps = exp(-lam(1-e^{-c(x-x0)}))   -- divisible, the
             extremal of the hypothesis wall (Prop. 'tangent');
    NULL  2  the power-floor chart eps = exp(-Lam + C(x-x0)^beta), beta<0 --
             divisible, interior;
    ALT      the two-atom chart eps = eps_inf + c e^{-alpha x} -- the extremal of
             the AXIOM wall, divisible nowhere (Prop. 'lumpy').

Each statistic is then scored three ways: its observed value, the 95th
percentile of its distribution under the nulls, and -- this is the number the
programme actually needs -- its POWER, the probability that it clears that null
cut when the truth is the alternative, on this frontier's design and noise.

Statistics (all bandwidth-free, all from measured accuracies):
    z_placed    widest quadruple whose right edge is at or left of the measured
                curvature maximum: dd3(S)/(sigma ||w||_2), refute if < -z95
    z_unplaced  the widest quadruple of the whole window (the archive's reading)
    rises       number of steps on which the second divided difference of ln eps
                increases (a divisible frontier's curvature never rises)
    line_r2     straightness of N_eff against S (Prop. 'pencil')

Outputs: results/table_plane_verdict.csv
         results/table_plane_power.csv
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save, all_frontiers                        # noqa: E402

B = 300
Z95 = 1.6448536269514722
rng = np.random.default_rng(20261006)


def dd(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return float(sum(ys[i] / np.prod([xs[i] - xs[j] for j in range(len(xs)) if j != i])
                     for i in range(len(xs))))


def dd_w(xs):
    xs = np.asarray(xs, float)
    return np.array([1.0 / np.prod([xs[i] - xs[j] for j in range(len(xs)) if j != i])
                     for i in range(len(xs))])


# --------------------------------------------------------------- the three charts
def fit_chart(x, L, kind):
    """Least squares in log-error. Returns (predict(x), sigma_resid) or None."""
    if kind == "cure":
        def f(th, xx):
            return -np.exp(th[0]) * (1 - np.exp(-np.exp(th[1]) * (xx - th[2])))
        th0 = [np.log(max(-L.min(), 1e-3)), np.log(0.5), x[0] - 0.5 * (x[-1] - x[0])]
    elif kind == "pfloor":
        def f(th, xx):
            return -np.exp(th[0]) + np.exp(th[1]) * np.maximum(xx - th[2], 1e-9) ** (-np.exp(th[3]))
        th0 = [np.log(max(-L.min(), 1e-3)), 0.0, x[0] - 0.3 * (x[-1] - x[0]), np.log(0.5)]
    else:                                                        # two atoms
        def f(th, xx):
            return np.log(np.exp(th[0]) + np.exp(th[1]) * np.exp(-np.exp(th[2]) * xx))
        th0 = [np.log(max(np.exp(L.min()) * .5, 1e-9)), L.max(), np.log(.4)]
    try:
        r = least_squares(lambda th: f(th, x) - L, th0, max_nfev=20000)
        pred = f(r.x, x)
        if not np.all(np.isfinite(pred)):
            return None
        sig = float(np.sqrt(np.sum(r.fun ** 2) / max(len(x) - len(th0), 1)))
        return pred, sig
    except Exception:
        return None


# ------------------------------------------------------------------ statistics
def stats(x, L, sigma, idx_un, idx_pl):
    S = -L
    kap = [2.0 * dd(x[j:j + 3], L[j:j + 3]) for j in range(len(x) - 2)]
    kap = [k for k in kap if np.isfinite(k)]
    rises = sum(1 for a, b in zip(kap, kap[1:]) if b > a)
    out = dict(rises=rises)
    for nm, idx in (("z_unplaced", idx_un), ("z_placed", idx_pl)):
        if idx is None or sigma <= 0:
            out[nm] = np.nan
        else:
            w = dd_w(x[idx])
            out[nm] = dd(x[idx], S[idx]) / (sigma * np.linalg.norm(w))
    # straightness of N_eff against S
    pts = []
    for j in range(len(x) - 2):
        s1 = -(L[j + 2] - L[j]) / (x[j + 2] - x[j])
        k = 2.0 * dd(x[j:j + 3], L[j:j + 3])
        if np.isfinite(s1) and np.isfinite(k) and k > 0 and s1 > 0:
            pts.append((S[j + 1], s1 * s1 / k))
    if len(pts) >= 3:
        A = np.column_stack([[p[0] for p in pts], np.ones(len(pts))])
        yv = np.array([p[1] for p in pts])
        coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
        ss = float(np.sum((yv - A @ coef) ** 2)); tot = float(np.sum((yv - yv.mean()) ** 2))
        out["line_r2"] = 1 - ss / tot if tot > 0 else np.nan
        out["slope"] = float(coef[0])
    else:
        out["line_r2"] = out["slope"] = np.nan
    return out


def quad_indices(x, right_limit=None):
    """Widest 4-point subset, optionally restricted to x <= right_limit."""
    ok = np.where(x <= right_limit + 1e-12)[0] if right_limit is not None else np.arange(len(x))
    if len(ok) < 4:
        return None
    a, d = ok[0], ok[-1]
    i1 = ok[max(1, len(ok) // 3)]
    i2 = ok[min(len(ok) - 2, (2 * len(ok)) // 3)]
    idx = sorted({a, i1, i2, d})
    return np.array(idx) if len(idx) == 4 else None


def run():
    rows = []
    for fr in all_frontiers():
        x, e = np.asarray(fr["x"], float), np.asarray(fr["e"], float)
        o = np.argsort(x); x, e = x[o], e[o]
        k = (e > 0) & np.isfinite(e); x, e = x[k], e[k]
        if len(x) < 6:
            continue
        L = np.log(e)
        fits = {kd: fit_chart(x, L, kd) for kd in ("cure", "pfloor", "two_atom")}
        if fits["cure"] is None or fits["two_atom"] is None:
            continue
        sigma = float(np.median([f[1] for f in fits.values() if f is not None]))
        if not np.isfinite(sigma) or sigma <= 0:
            continue

        # the measured curvature maximum, no kernel
        kx = [(x[j + 1], 2.0 * dd(x[j:j + 3], L[j:j + 3])) for j in range(len(x) - 2)]
        kx = [(a, b) for a, b in kx if np.isfinite(b)]
        x_elbow = max(kx, key=lambda p: p[1])[0] if kx else np.nan
        idx_un = quad_indices(x)
        idx_pl = quad_indices(x, x_elbow) if np.isfinite(x_elbow) else None

        obs = stats(x, L, sigma, idx_un, idx_pl)

        # ---- null and alternative distributions on this design ---------------
        sims = {}
        for kd, fit in fits.items():
            if fit is None:
                continue
            pred = fit[0]
            acc = {q: [] for q in ("z_placed", "z_unplaced", "rises", "line_r2")}
            for _ in range(B):
                Lb = pred + sigma * rng.standard_normal(len(x))
                st = stats(x, Lb, sigma, idx_un, idx_pl)
                for q in acc:
                    acc[q].append(st[q])
            sims[kd] = {q: np.array(v, float) for q, v in acc.items()}

        row = dict(corpus=fr["corpus"], name=fr["name"], n=len(x),
                   span=float(x[-1] - x[0]), sigma=sigma, x_elbow=x_elbow,
                   elbow_frac=(x_elbow - x[0]) / (x[-1] - x[0]) if np.isfinite(x_elbow) else np.nan,
                   n_admissible=int(np.sum(x <= x_elbow + 1e-12)) if np.isfinite(x_elbow) else 0,
                   **{f"obs_{q}": obs[q] for q in ("z_placed", "z_unplaced", "rises", "line_r2", "slope")})
        for q, side in (("z_placed", "lo"), ("z_unplaced", "lo"), ("rises", "hi"), ("line_r2", "lo")):
            nulls = np.concatenate([sims[kd][q][np.isfinite(sims[kd][q])]
                                    for kd in ("cure", "pfloor") if kd in sims])
            if len(nulls) < 20:
                continue
            cut = np.percentile(nulls, 5 if side == "lo" else 95)
            row[f"cut_{q}"] = float(cut)
            row[f"refuted_{q}"] = int(obs[q] < cut) if side == "lo" else int(obs[q] > cut)
            alt = sims["two_atom"][q]
            alt = alt[np.isfinite(alt)]
            row[f"power_{q}"] = float(np.mean(alt < cut) if side == "lo" else np.mean(alt > cut))
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n  {len(df)} frontiers, {B} replicates per chart, three charts each")
    print(f"  median per-model residual scale ............... {df.sigma.median():.4f} nats")
    print(f"  curvature maximum at ......................... "
          f"{df.elbow_frac.median():.2f} of the window (median), "
          f"{int(df.n_admissible.median())} models to its left")
    print("\n  statistic      refuted (vs its own null)   median power against the")
    print("                                             two-atom alternative")
    for q, lab in (("z_placed", "placed quadruple"), ("z_unplaced", "unplaced quadruple"),
                   ("rises", "curvature rises"), ("line_r2", "straightness")):
        if f"refuted_{q}" not in df:
            continue
        r = df[f"refuted_{q}"].dropna(); p = df[f"power_{q}"].dropna()
        print(f"  {lab:<22} {int(r.sum()):>4} of {len(r):<4}            "
              f"{p.median():.3f}   (>=0.5 on {int((p >= 0.5).sum())} frontiers)")
    print("\n  the nominal level is 0.05 by construction, so a statistic is")
    print("  informative only where its refutation count exceeds 0.05 x n and its")
    print("  power is appreciably above 0.05.")
    save(df, "table_plane_verdict.csv")

    summ = pd.DataFrame([dict(
        n_frontiers=len(df), B=B, median_sigma=float(df.sigma.median()),
        median_elbow_frac=float(df.elbow_frac.median()),
        median_n_admissible=float(df.n_admissible.median()),
        **{f"refuted_{q}": int(df[f"refuted_{q}"].sum()) for q in
           ("z_placed", "z_unplaced", "rises", "line_r2") if f"refuted_{q}" in df},
        **{f"n_{q}": int(df[f"refuted_{q}"].notna().sum()) for q in
           ("z_placed", "z_unplaced", "rises", "line_r2") if f"refuted_{q}" in df},
        **{f"medpower_{q}": float(df[f"power_{q}"].median()) for q in
           ("z_placed", "z_unplaced", "rises", "line_r2") if f"power_{q}" in df},
    )])
    save(summ, "table_plane_power.csv")


if __name__ == "__main__":
    run()

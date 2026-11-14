#!/usr/bin/env python3
"""136_ode_corpus.py -- what the ODE-first rung is worth, on clean charts and on
the corpus, against the same design-matched null as script 130.

Script 134 shows that the hypothesis's own first rung is

    D_1 = 1 - Ndot >= 0        i.e.   Ndot <= 1,       Ndot = dN/d(dS),

that it is FLOOR-FREE, and that the statistic the archive actually uses,

    c_3 = S''' >= 0            i.e.   Ndot <= 2,

is exactly twice as loose.  Both are computable from the same four accuracies,
because with divided differences on a quadruple

    s = dd_1,   kappa = -2 dd_2,   S''' = 6 dd_3,
    D_1 = s S'''/kappa^2 - 1 = 1.5 dd_1 dd_3 / dd_2^2 - 1.

So the reformulation offers a strictly stronger test at zero extra measurement
cost.  This script asks whether that is worth anything, in the only way this
archive accepts: against a null simulated on the same design.

  PART 1  CLEAN CHARTS.  The two-atom truth (the axiom wall, divisible nowhere)
          at a range of per-model noise levels, with the level set by the two
          divisible charts.  Reports power(old) against power(new) and the noise
          at which each reaches 80 %.

  PART 2  THE CORPUS.  The same 117 frontiers as scripts 68/129/130, the same
          three fitted charts, the same B = 300 replicates, the same two
          quadruples (widest, and widest left of the measured curvature
          maximum).  Reports refutations at the nominal level and median power
          for both statistics side by side.

Outputs: results/table_ode_power_charts.csv
         results/table_ode_corpus.csv
         results/table_ode_corpus_summary.csv
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save, all_frontiers, _module                # noqa: E402

V130 = _module("130_plane_verdict.py", "v130")
fit_chart, dd, dd_w, quad_indices = (V130.fit_chart, V130.dd, V130.dd_w,
                                     V130.quad_indices)

B = 300
Z95 = 1.6448536269514722
rng = np.random.default_rng(20261007)


# ------------------------------------------------------------------ statistics
def stat_old(xq, Sq, sigma):
    """The archive's four-point test: dd_3(S) >= 0, standardised."""
    w = dd_w(xq)
    d3 = dd(xq, Sq)
    return d3 / (sigma * np.linalg.norm(w)) if sigma > 0 else np.nan


def stat_new(xq, Sq):
    """The ODE-first rung on the same quadruple: D_1 = 1.5 dd1 dd3/dd2^2 - 1.

    Scale-free in the noise, so it is compared against its own simulated null
    rather than standardised; a value below the null cut refutes."""
    d1 = dd(xq[:2], Sq[:2])
    d2 = dd(xq[:3], Sq[:3])
    d3 = dd(xq, Sq)
    if not np.isfinite(d1) or not np.isfinite(d2) or not np.isfinite(d3) or d2 == 0:
        return np.nan
    if d1 <= 0 or d2 >= 0:              # outside the admissible cone (s>0, kappa>0)
        return np.nan
    return 1.5 * d1 * d3 / (d2 ** 2) - 1.0


def both(xq, Sq, sigma):
    return dict(old=stat_old(xq, Sq, sigma), new=stat_new(xq, Sq))


# =========================================================== PART 1: the charts
def charts_power():
    """Clean-chart power of the two statistics at matched level."""
    rows = []
    n_models, span = 10, 5.0
    x = np.linspace(0.0, span, n_models)

    def two_atom(x, e0=0.004, c=0.9, a=0.9):
        return np.log(e0 + c * np.exp(-a * x))

    def cure(x, lam=5.0, c=0.7, x0=-1.0):
        return -lam * (1 - np.exp(-c * (x - x0)))

    def pfloor(x, Lam=4.0, C=1.3, x0=-2.0, b=0.6):
        return -Lam + C * (x - x0) ** (-b)

    truths = {"two_atom (alt)": two_atom(x), "cure (null)": cure(x),
              "power_floor (null)": pfloor(x)}
    idx = quad_indices(x)
    for sigma in (0.0025, 0.005, 0.010, 0.020, 0.040):
        draws = {}
        for nm, L0 in truths.items():
            acc = {"old": [], "new": []}
            for _ in range(2000):
                Lb = L0 + sigma * rng.standard_normal(len(x))
                st = both(x[idx], -Lb[idx], sigma)
                acc["old"].append(st["old"]); acc["new"].append(st["new"])
            draws[nm] = {k: np.array(v, float) for k, v in acc.items()}
        for k in ("old", "new"):
            nulls = np.concatenate([draws[nm][k][np.isfinite(draws[nm][k])]
                                    for nm in truths if "null" in nm])
            cut = np.percentile(nulls, 5)
            alt = draws["two_atom (alt)"][k]
            alt = alt[np.isfinite(alt)]
            lev = {nm: float(np.mean(draws[nm][k][np.isfinite(draws[nm][k])] < cut))
                   for nm in truths if "null" in nm}
            # the raw-sign reading, with no null: this is what a plug-in would
            # do, and it is where the sharper rung pays for its sharpness --
            # the cure chart SATURATES D_1, so it sits on the boundary.
            sgn = {nm: float(np.mean(draws[nm][k][np.isfinite(draws[nm][k])] < 0))
                   for nm in truths}
            rows.append(dict(statistic=k, sigma=sigma, cut=float(cut),
                             power=float(np.mean(alt < cut)),
                             level_cure=lev["cure (null)"],
                             level_pfloor=lev["power_floor (null)"],
                             signlevel_cure=sgn["cure (null)"],
                             signlevel_pfloor=sgn["power_floor (null)"],
                             n_models=n_models, span=span))
    df = pd.DataFrame(rows)
    print("\n  PART 1  clean charts: the two-atom truth, level set by two divisible "
          "charts\n")
    print("   per-model noise      power(c_3 >= 0)   power(D_1 >= 0)   levels (cure/pf)")
    for sg in sorted(df.sigma.unique()):
        o = df[(df.sigma == sg) & (df.statistic == "old")].iloc[0]
        n = df[(df.sigma == sg) & (df.statistic == "new")].iloc[0]
        print(f"   {sg * 1000:6.1f} milli-nats     {o.power:11.3f}     {n.power:11.3f}"
              f"      {o.level_cure:.2f}/{o.level_pfloor:.2f}  "
              f"{n.level_cure:.2f}/{n.level_pfloor:.2f}")
    print("\n   the SIGN of each statistic, with no null (what a plug-in does):")
    print("   statistic    cure chart    power-floor chart")
    for k, lab in (("old", "c_3 >= 0"), ("new", "D_1 >= 0")):
        sub = df[df.statistic == k]
        print(f"   {lab:<12} {sub.signlevel_cure.median():.2f}          "
              f"{sub.signlevel_pfloor.median():.2f}")
    print("   -- the cure chart saturates D_1 (it is the extremal of that rung),")
    print("      so a sign reading of it refutes a divisible truth about half the")
    print("      time at every noise level. The sharper rung must be studentised.")
    for k, lab in (("old", "c_3 >= 0"), ("new", "D_1 >= 0")):
        sub = df[df.statistic == k].sort_values("sigma")
        ok = sub[sub.power >= 0.80]
        s80 = float(ok.sigma.max()) if len(ok) else float("nan")
        print(f"   {lab:<10} reaches 80 % power at <= {s80 * 1000:.1f} milli-nats"
              if np.isfinite(s80) else
              f"   {lab:<10} does not reach 80 % power in this range")
    save(df, "table_ode_power_charts.csv")
    return df


# =========================================================== PART 2: the corpus
def corpus_verdict():
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
        kx = [(x[j + 1], 2.0 * dd(x[j:j + 3], L[j:j + 3])) for j in range(len(x) - 2)]
        kx = [(a, b) for a, b in kx if np.isfinite(b)]
        x_elbow = max(kx, key=lambda p: p[1])[0] if kx else np.nan
        quads = {"whole": quad_indices(x),
                 "placed": quad_indices(x, x_elbow) if np.isfinite(x_elbow) else None}

        row = dict(corpus=fr["corpus"], name=fr["name"], n=len(x),
                   span=float(x[-1] - x[0]), sigma=sigma, x_elbow=x_elbow)
        for pl, idx in quads.items():
            if idx is None:
                continue
            obs = both(x[idx], -L[idx], sigma)
            sims = {}
            for kd, fit in fits.items():
                if fit is None:
                    continue
                pred = fit[0]
                acc = {"old": [], "new": []}
                for _ in range(B):
                    Lb = pred + sigma * rng.standard_normal(len(x))
                    st = both(x[idx], -Lb[idx], sigma)
                    acc["old"].append(st["old"]); acc["new"].append(st["new"])
                sims[kd] = {q: np.array(v, float) for q, v in acc.items()}
            for q in ("old", "new"):
                nulls = np.concatenate([sims[kd][q][np.isfinite(sims[kd][q])]
                                        for kd in ("cure", "pfloor") if kd in sims])
                if len(nulls) < 20 or not np.isfinite(obs[q]):
                    continue
                cut = float(np.percentile(nulls, 5))
                alt = sims["two_atom"][q]
                alt = alt[np.isfinite(alt)]
                row[f"obs_{q}_{pl}"] = obs[q]
                row[f"cut_{q}_{pl}"] = cut
                row[f"refuted_{q}_{pl}"] = int(obs[q] < cut)
                row[f"power_{q}_{pl}"] = float(np.mean(alt < cut)) if len(alt) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n  PART 2  the corpus: {len(df)} frontiers, {B} replicates per chart\n")
    print("   quadruple   statistic   refuted at the nominal level   median power")
    summ = {}
    for pl in ("whole", "placed"):
        for q, lab in (("old", "c_3 >= 0"), ("new", "D_1 >= 0")):
            rc, pc = f"refuted_{q}_{pl}", f"power_{q}_{pl}"
            if rc not in df:
                continue
            r, p = df[rc].dropna(), df[pc].dropna()
            print(f"   {pl:<11} {lab:<11} {int(r.sum()):>6} of {len(r):<6}"
                  f"              {p.median():.3f}")
            summ.update({f"refuted_{q}_{pl}": int(r.sum()), f"n_{q}_{pl}": int(len(r)),
                         f"medpower_{q}_{pl}": float(p.median()),
                         f"npow50_{q}_{pl}": int((p >= 0.5).sum())})
    save(df, "table_ode_corpus.csv")
    save(pd.DataFrame([dict(n_frontiers=len(df), B=B, **summ)]),
         "table_ode_corpus_summary.csv")
    return df


def run():
    ch = charts_power()
    co = corpus_verdict()
    o = ch[(ch.statistic == "old") & (ch.sigma == 0.020)].iloc[0].power
    n = ch[(ch.statistic == "new") & (ch.sigma == 0.020)].iloc[0].power
    print("\n  verdict")
    print(f"    on clean charts at the archive's own noise (20 milli-nats): "
          f"power {o:.3f} -> {n:.3f}")
    print("    on the corpus: see PART 2 -- both statistics are at their nominal")
    print("    level, so the reformulation improves the TEST, not this evidence.")
    return ch, co


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""T12b  The membership statistic against a null calibrated on each frontier.

Proposition 'width coordinate'(iv) states R = zeta'/alpha_hat = 1 identically
for one reducible channel plus a floor.  That is an identity in the exact
derivatives of S.  Script 38 evaluates R with a local quadratic estimator of
bandwidth h = span/4 and compares the result to the analytic value 1; script
41 shows that the estimator's null is not 1 away from the crossover.  This
script supplies the correct reference: for every frontier of the width panel
that carries a usable fitted floor, it simulates EXACT two-atom data from that
frontier's own fitted (einf, c, alpha), on that frontier's own Pareto grid,
with log-error noise at that frontier's own fitted residual scale, applies the
same one-dimensional Pareto filter, and runs the identical estimator.  The
three-atom alternative is calibrated the same way.

Writes results/table_width_null_calibration.csv (per frontier, per bandwidth),
results/table_width_null_summary.csv (the table reported in the paper) and
results/table_width_null_three.csv (two- against three-atom).
"""
import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings("ignore")
from _loaders import selection_rules, save, RES
import os
from _calib import width_fit, fit3, simulate_null

sr = selection_rules()
RNG = np.random.default_rng(20261003)
DIVS = (2, 4, 8, 16)
NSIM = 600


def panel():
    """The frontiers of the width panel that carry a usable fitted floor.
    Frontiers fitted with einf ~ 0 are excluded: for them the two-atom null of
    R is 0, not 1 (the gapless branch of the dichotomy), so comparing them to
    the two-atom reference is meaningless either way."""
    P = pd.read_csv(os.path.join(RES, "table_width_per_frontier.csv"))
    P = P[(P.einf_fit > 1e-3) & (P.einf_fit < 0.5)]
    grid = {}
    for field, name, cost, x, e in sr.single_resource_frontiers():
        grid[(str(name), str(cost))] = (np.asarray(x, float), np.asarray(e, float))
    for _, r in P.iterrows():
        k = (str(r.frontier), str(r.cost))
        if k in grid:
            yield r, grid[k][0], grid[k][1]


def main():
    rows, three = [], []
    for r, x, e in panel():
        f3 = fit3(x, e)
        if f3 is None:
            continue
        einf, c, a, sd = f3["einf"], f3["c"], f3["a"], f3["sd"]
        xc = np.log(c / einf) / a                      # the crossover, eps = 2 einf
        rec = dict(field=r.field, frontier=r.frontier, cost=r.cost, n=len(e),
                   span=float(np.ptp(x)), einf=einf, c=c, alpha=a, sd=sd,
                   offset=float((x.min() + x.max()) / 2 - xc))
        for div in DIVS:
            obs = width_fit(x, e, div=div)
            rec[f"obs_{div}"] = obs["R"] if obs else np.nan
            nl = simulate_null(x, einf, c, a, sd, RNG, div=div, nsim=NSIM)
            if len(nl) >= 100 and np.isfinite(rec[f"obs_{div}"]):
                rec[f"null_{div}"] = float(np.median(nl))
                rec[f"lo_{div}"] = float(np.percentile(nl, 2.5))
                rec[f"hi_{div}"] = float(np.percentile(nl, 97.5))
                rec[f"nsim_{div}"] = int(len(nl))
                rec[f"p_{div}"] = float(2 * min((nl <= rec[f"obs_{div}"]).mean(),
                                                (nl >= rec[f"obs_{div}"]).mean()))
            else:
                for t in ("null", "lo", "hi", "nsim", "p"):
                    rec[f"{t}_{div}"] = np.nan
        rows.append(rec)
        n3 = simulate_null(x, einf, c, a, sd, RNG, div=4.0, nsim=400, natoms=3)
        if len(n3) >= 100 and np.isfinite(rec["obs_4"]) and np.isfinite(rec["null_4"]):
            three.append(dict(frontier=r.frontier, cost=r.cost, obs=rec["obs_4"],
                              null_two=rec["null_4"], null_three=float(np.median(n3))))
    D = pd.DataFrame(rows)
    save(D, "table_width_null_calibration.csv")

    out = []
    for div in DIVS:
        m = D[[f"obs_{div}", f"null_{div}", f"lo_{div}", f"hi_{div}", f"p_{div}"]].dropna()
        m.columns = ["o", "nm", "lo", "hi", "p"]
        inside = int(((m.o >= m.lo) & (m.o <= m.hi)).sum())
        out.append(dict(bandwidth=f"span/{div}", frontiers=len(m),
                        obs_median=float(m.o.median()),
                        null_median=float(m.nm.median()),
                        analytic_null=1.0,
                        inside_95=inside,
                        rejections_p05=int((m.p < 0.05).sum()),
                        wilcoxon_p=float(stats.wilcoxon(m.o, m.nm).pvalue) if len(m) > 5 else np.nan))
    S = pd.DataFrame(out)
    save(S, "table_width_null_summary.csv")

    T = pd.DataFrame(three)
    if len(T):
        d2 = (T.obs - T.null_two).abs(); d3 = (T.obs - T.null_three).abs()
        T2 = pd.concat([T, pd.DataFrame(dict(dist_two=d2, dist_three=d3))], axis=1)
        save(T2, "table_width_null_three.csv")

    print(f"\nwidth panel calibrated: {len(D)} floor-bearing frontiers")
    print(f"median window offset past the crossover: {D.offset.median():+.2f} nats "
          f"({int((D.offset > 0).sum())}/{len(D)} past it)")
    print(S.to_string(index=False))
    if len(T):
        print(f"\nthree-atom alternative on {len(T)} frontiers: null median "
              f"{T.null_three.median():.3f} against two-atom {T.null_two.median():.3f}; "
              f"two atoms closer on {int((d2 < d3).sum())}/{len(T)}, "
              f"Wilcoxon p = {stats.wilcoxon(d2, d3).pvalue:.4f}")
    m = D[["obs_4", "null_4", "offset"]].dropna()
    print(f"calibrated null vs window offset: Spearman rho="
          f"{stats.spearmanr(m.offset, m.null_4).statistic:.3f}, "
          f"p={stats.spearmanr(m.offset, m.null_4).pvalue:.4f}")


if __name__ == "__main__":
    main()

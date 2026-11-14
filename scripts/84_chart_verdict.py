#!/usr/bin/env python3
"""WHICH CHART IS THE ARCHIVE IN? THE VERDICT TABLE.
   (Section "The floor is not a parameter", paragraph "Which chart the archive
   is in".)

Reads the per-frontier invariants and the design-matched nulls of script 82 and
reports, for each candidate truth, the one-sided exceedance rate of the
observed second-order invariant T = dzeta/dS against that truth's own p90, the
fraction inside its central 80%, and the median ratio T_obs/T_null.  A truth
that describes the archive should exceed its p90 on about 10% of frontiers.

"Deep windows" are those spanning at least 4 nats of log-resource, where a
second derivative is worth estimating at all.

Also reports the floor share the floor chart would need in order to reproduce
the measured invariant, against the floor share it actually fits -- the
internal-consistency statement.

Outputs: results/table_chart_verdict.csv
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _loaders import save

RES = os.path.join(os.path.dirname(HERE), "results")
DEEP = 4.0


def run():
    P = pd.read_csv(os.path.join(RES, "table_chart_test_per_frontier.csv"))
    N = pd.read_csv(os.path.join(RES, "table_chart_test_nulls.csv"))
    rows = []
    for t in sorted(N.truth.unique()):
        g = N[N.truth == t][["frontier", "T_med", "T_p10", "T_p90"]]
        m = P.merge(g, on="frontier").dropna(subset=["T_obs"])
        if len(m) < 5:
            continue
        d = m[m.span >= DEEP]
        rows.append(dict(
            truth=t, n=len(m),
            above_p90=float((m.T_obs > m.T_p90).mean()),
            inside_p10_p90=float(((m.T_obs >= m.T_p10) & (m.T_obs <= m.T_p90)).mean()),
            median_ratio=float(np.median(m.T_obs / m.T_med)),
            n_deep=len(d),
            above_p90_deep=float((d.T_obs > d.T_p90).mean()) if len(d) else np.nan,
            median_ratio_deep=float(np.median(d.T_obs / d.T_med)) if len(d) else np.nan))
    V = pd.DataFrame(rows).sort_values("above_p90")
    extra = pd.DataFrame([
        dict(quantity="frontiers with a usable second-order reading", value=float(len(P))),
        dict(quantity="observed T, median", value=float(P.T_obs.median())),
        dict(quantity="floor share the floor chart must assume to match "
                      "(free fit, median)", value=float(P.p0_fitted.median())),
        dict(quantity="frontiers whose free floor fit is pinned at the ceiling p0=0.98",
             value=float((P.p0_fitted >= 0.979).mean())),
        dict(quantity="floor share the paper's own fits report (corpus median)",
             value=0.51),
        dict(quantity="nominal one-sided exceedance rate of a correct truth", value=0.10)])
    return V, extra


if __name__ == "__main__":
    print(__doc__.split("Outputs")[0])
    V, E = run()
    pd.set_option("display.width", 200)
    print(V.to_string(index=False)); print(); print(E.to_string(index=False))
    save(V, "table_chart_verdict.csv")
    save(E, "table_chart_verdict_context.csv")

#!/usr/bin/env python3
"""FRONTIER-CLUSTERED UNCERTAINTY FOR THE HELD-OUT PREDICTION CLAIM.

Script 68 produces 2025 interior leave-one-out prediction events over 2259
measurements on 117 frontiers.  Those events are *not* independent: each one
trains on n-1 points of its own frontier, so every measurement is reused as
training data by its neighbours.  Pooling them and reporting one R^2 overstates
the evidence.  This script re-reads script 68's per-frontier output and

    (a) bootstraps the per-frontier median error over the 117 frontiers,
        which is the independent unit, for every predictor and every regime;
    (b) reports the PAIRED comparison of the law against each baseline,
        per frontier, with the fraction of frontiers on which the law wins.

Baselines are script 68's own, unchanged:
    law     eps = eps_inf + c exp(-a x)      the three-parameter form
    law2    two reducible atoms
    power   eps = c t^-alpha                 the law with the floor removed
    interp  log-linear in log-resource through the neighbours
    last    the nearest training point's error

Writes (results/):
    table_prediction_clustered.csv   per-frontier median + 95% CI, by method and regime
    table_prediction_paired.csv      paired law - baseline, median, CI, win rate
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

METHODS = ["law", "law2", "power", "interp", "last"]
REGIMES = ["INTERIOR", "FORWARD", "BACKWARD"]
B = 10000
SEED = 20261114


def boot_median(v, rng, B=B):
    n = len(v)
    idx = rng.integers(0, n, size=(B, n))
    return np.median(v[idx], axis=1)


if __name__ == "__main__":
    T = pd.read_csv(os.path.join(RES, "table_prediction_frontiers.csv"))
    print(f"{len(T)} frontiers, {int(T.n.sum())} measurements, "
          f"{int((T.n - 2).sum())} interior leave-one-out prediction events")

    rng = np.random.default_rng(SEED)
    rows = []
    for m in METHODS:
        for reg in REGIMES:
            v = T[f"{m}_{reg}"].dropna().values
            meds = boot_median(v, rng)
            rows.append(dict(method=m, regime=reg, n_frontiers=len(v),
                             median_nats=round(float(np.median(v)), 4),
                             ci_lo=round(float(np.percentile(meds, 2.5)), 4),
                             ci_hi=round(float(np.percentile(meds, 97.5)), 4)))
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(RES, "table_prediction_clustered.csv"), index=False)

    rng = np.random.default_rng(SEED)
    rows = []
    for b in ["power", "interp", "last", "law2"]:
        for reg in REGIMES:
            d = (T[f"law_{reg}"] - T[f"{b}_{reg}"]).dropna().values
            meds = boot_median(d, rng)
            rows.append(dict(baseline=b, regime=reg, n_frontiers=len(d),
                             paired_median_nats=round(float(np.median(d)), 4),
                             ci_lo=round(float(np.percentile(meds, 2.5)), 4),
                             ci_hi=round(float(np.percentile(meds, 97.5)), 4),
                             law_better_pct=round(float((d < 0).mean() * 100), 1)))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(RES, "table_prediction_paired.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\nPER-FRONTIER median |ln eps_pred - ln eps_obs| (nats), "
          "95% CI bootstrapped over the 117 frontiers\n")
    print(C.to_string(index=False))
    print("\nPAIRED law - baseline, per frontier (negative = the law is better)\n")
    print(P.to_string(index=False))

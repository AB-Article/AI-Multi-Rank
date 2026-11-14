#!/usr/bin/env python3
"""T13  Should the fitted floor be capped at the best observed error?

Proposition 'running floor' (ii) bounds eps_inf by the CURVE, eps(x_0+X), not
by the measurement.  Under log-error noise the smallest observed error is not
an upper bound for eps_inf, so capping the fit at it is not implied by the
model class.  This script shows that the cap is actively harmful where the
question arises: on a saturated window the capped estimator is biased low,
while the unconstrained one is essentially unbiased.

Design: 400 replicates of eps = eps_inf + c e^{-alpha x} with eps_inf = 0.10,
alpha = 0.6, m = 25 points equally spaced in the normalized coordinate zeta,
multiplicative log-normal noise of scale sigma, at three window positions.
The fitter is the one of script 11 (absolute residuals, bounds as there); the
capped variant differs only in the upper bound on eps_inf.

No data, no network.  Writes results/table_floor_cap_bias.csv.
"""
import os
import numpy as np, pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
RNG = np.random.default_rng(20261004)
EINF, ALPHA, M, NREP = 0.10, 0.6, 25, 400
WINDOWS = {"knee-centred": (-2.0, 2.0), "saturated": (1.0, 5.0), "steep": (-5.0, -1.0)}
SIGMAS = (0.02, 0.05, 0.10)


def fit_law(x, e, cap=None):
    """Script 11's fitter; cap bounds eps_inf above when given."""
    u = x - x.min()
    hi = [0.9 if cap is None else cap, 1e3, 6.0]
    r = least_squares(lambda t: t[0] + t[1] * np.exp(-t[2] * u) - e,
                      [0.8 * e.min(), max(e.max() - e.min(), 1e-3), 0.5],
                      bounds=([0, 0, 1e-4], hi), max_nfev=100000)
    return float(r.x[0])


def run():
    rows = []
    for wname, (z0, z1) in WINDOWS.items():
        for sig in SIGMAS:
            free, capped, above = [], [], 0
            for _ in range(NREP):
                z = np.linspace(z0, z1, M)
                x = z / ALPHA
                e = EINF * (1.0 + np.exp(-z)) * np.exp(RNG.normal(0.0, sig, M))
                try:
                    f = fit_law(x, e)
                    c = fit_law(x, e, cap=max(e.min(), 1e-9))
                except Exception:
                    continue
                free.append(f - EINF); capped.append(c - EINF)
                above += int(f >= e.min())
            free, capped = np.array(free), np.array(capped)
            rows.append(dict(window=wname, sigma=sig, n=len(free),
                             frac_floor_above_best=above / max(len(free), 1),
                             bias_free=float(np.median(free)),
                             bias_capped=float(np.median(capped)),
                             rel_bias_free=float(np.median(free) / EINF),
                             rel_bias_capped=float(np.median(capped) / EINF)))
    T = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    T.to_csv(os.path.join(RES, "table_floor_cap_bias.csv"), index=False)
    print(T.round(4).to_string(index=False))
    print("\nReading: the cap never helps and costs most on saturated windows,")
    print("which is exactly where a fitted floor above the best observed error")
    print("occurs.  The unconstrained fit is therefore the one to report.")


if __name__ == "__main__":
    run()

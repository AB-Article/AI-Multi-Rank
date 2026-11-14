#!/usr/bin/env python3
"""T12c  Why the membership statistic needs a calibrated null: the estimator's
null depends on the bandwidth and on where the window sits.

On EXACT two-atom data, with no noise and no Pareto filtering, R = 1 is an
identity.  The plug-in estimator of script 38 does not return 1 unless the
observation window happens to be centred on the crossover eps = 2 einf.  Part
one tabulates R against the window offset for three spans; part two shows the
bandwidth dependence at a centred window, where the estimator is nearly
unbiased; part three confirms that R computed from exact moments is 1.

The synthetic study of Online Appendix S5.19 centred its windows on the
crossover, which is why it found the estimator unbiased; the measured corpus
is not centred, which is why the two disagree.

Writes results/table_width_window_profile.csv,
results/table_width_bandwidth_centred.csv and
results/table_width_exact_moments.csv.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from _loaders import save
from _calib import width_fit

AL, EI, C = 0.8, 0.1, 1.0                   # exponent, floor, reducible scale
XC = np.log(C / EI) / AL                    # the crossover


def two_atom(x):
    return EI + C * np.exp(-AL * x)


def main():
    rows = []
    for off in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
        rec = dict(offset_nats=off)
        for span in (2.5, 4.7, 8.0):
            x = np.linspace(XC + off - span / 2, XC + off + span / 2, 30)
            r = width_fit(x, two_atom(x), div=4.0)
            rec[f"R_span_{span}"] = r["R"] if r else np.nan
        rows.append(rec)
    P = pd.DataFrame(rows)
    save(P, "table_width_window_profile.csv")

    rows = []
    for npts in (30, 60, 120, 400):
        rec = dict(n_points=npts)
        x = np.linspace(XC - 4.7 / 2, XC + 4.7 / 2, npts)
        for div in (2, 4, 8, 16, 32, 64):
            r = width_fit(x, two_atom(x), div=float(div))
            rec[f"R_h_span_over_{div}"] = r["R"] if r else np.nan
        rows.append(rec)
    B = pd.DataFrame(rows)
    save(B, "table_width_bandwidth_centred.csv")

    # exact moments: s = alpha p1, kappa = alpha^2 p0 p1, no smoothing at all
    x = np.linspace(XC - 2.35, XC + 2.35, 41)
    p0 = EI / two_atom(x); p1 = 1 - p0
    s, kap = AL * p1, AL ** 2 * p0 * p1
    z = np.log(kap / s ** 2)
    slope = float(np.polyfit(x, z, 1)[0])
    ah = float(np.median(s + kap / s))
    E = pd.DataFrame([dict(statistic="R from exact moments", value=slope / ah,
                           analytic_value=1.0, deviation=abs(slope / ah - 1.0))])
    save(E, "table_width_exact_moments.csv")

    print("\nEXACT two atoms, no noise, no Pareto filter, h = span/4, 30 points.")
    print("The analytic null is R = 1 at every offset and every span.\n")
    print(P.round(3).to_string(index=False))
    print("\nCentred window, varying bandwidth (the estimator is nearly unbiased here):\n")
    print(B.round(3).to_string(index=False))
    print(f"\nR from exact moments: {slope / ah:.6f}")


if __name__ == "__main__":
    main()

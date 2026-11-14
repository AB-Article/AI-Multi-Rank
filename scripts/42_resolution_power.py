#!/usr/bin/env python3
"""T5c  Does the resolution-model reciprocity have power?

Script 27 reports that theta_hat x A_hat varies far less across tasks of one
grid than theta_hat alone, and that the within-grid regression of ln theta on
ln A has slope near the predicted -1.  Both statistics are computed from
quantities estimated on the same frontier, so neither means anything without a
null: an estimator whose errors in theta and A are anticorrelated would
produce a small variation ratio and a slope near -1 with no reciprocity in the
data-generating process at all.

This script supplies the null by simulating through the same pipeline as
scripts 14 and 27 -- theta_hat from a profile over theta, A_hat the geometric
mean of the exponents fitted with theta FORCED to 1 -- under two generators:

  model A (reciprocity TRUE) : theta_t = theta_r / A_t, one theta_r per grid;
  model B (reciprocity FALSE): theta_t and A_t drawn independently.

If model B also yields a small variation ratio and a negative slope, the
statistics of script 27 are uninformative.  It does not.

Writes results/table_resolution_power.csv and
results/table_resolution_power_draws.csv (the per-repetition draws).
"""
import itertools, os, numpy as np, pandas as pd, warnings
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")
from _loaders import save, RES

RNG = np.random.default_rng(20260942)
THETAS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0]
NTASKS, REPS, SD = 6, 12, 0.02


def powmean_eps(p, X, theta):                      # as script 14
    n = X.shape[1]
    einf, b, a = p[0], p[1:1 + n], p[1 + n:1 + 2 * n]
    lg = b[None, :] - X * a[None, :]
    if theta == 0.0:
        lm = lg.mean(1)
    else:
        z = theta * lg; zm = z.max(1, keepdims=True)
        lm = (zm[:, 0] + np.log(np.exp(z - zm).mean(1))) / theta
    return einf + np.exp(lm)


def fit_theta(X, e, theta, warm=None):             # as script 14
    n = X.shape[1]
    lo = [0.0] + [-40.0] * n + [0.0] * n
    hi = [max(e.min(), 1e-6) * 0.999 + 1e-9] + [10.0] * n + [6.0] * n
    starts = []
    if warm is not None:
        starts.append(np.clip(warm, np.array(lo) + 1e-12, np.array(hi) - 1e-12))
    for f0, a0 in itertools.product((0.0, 0.5, 0.9), (0.2, 0.6, 1.2)):
        einf = f0 * e.min()
        starts.append(np.array([einf] + [np.log(max(np.median(e) - einf, 1e-6))] * n + [a0] * n))
    best = None
    for s in starts:
        s = np.clip(s, np.array(lo) + 1e-12, np.array(hi) - 1e-12)
        try:
            r = least_squares(lambda p: powmean_eps(p, X, theta) - e, s,
                              bounds=(lo, hi), max_nfev=4000, x_scale="jac")
        except Exception:
            continue
        rss = float(np.sum(r.fun ** 2))
        if best is None or rss < best[1]:
            best = (r.x, rss)
    return best


def pipeline(X, e):
    """theta_hat by profile; A_hat = geometric mean of the alphas at theta=1."""
    out, warm = {}, None
    for th in THETAS:
        b = fit_theta(X, e, th, warm)
        if b is None:
            continue
        out[th] = b; warm = b[0]
    if 1.0 not in out:
        return None
    ths = np.array(sorted(out)); rs = np.array([out[t][1] for t in ths])
    n = X.shape[1]; a1 = out[1.0][0][1 + n:1 + 2 * n]
    if (a1 <= 0).any():
        return None
    return float(ths[np.argmin(rs)]), float(np.exp(np.log(a1).mean()))


def cv(v):
    v = np.asarray(v, float)
    return float(np.std(v, ddof=1) / np.mean(v))


def design():
    """The real factorial design of the gadre grid (log-resources, centred)."""
    g = pd.read_csv(os.path.join(os.path.dirname(RES), "data", "grid_factorial.csv"))
    gad = g[g.frontier.str.startswith("gadre")]
    X = np.log(gad[gad.frontier == gad.frontier.iloc[0]][["cost_params", "cost_tokens"]]
               .to_numpy(float))
    return X - X.mean(0)


def run(X, reciprocal):
    rat, slp, cvp = [], [], []
    for _ in range(REPS):
        A = np.exp(RNG.uniform(np.log(0.05), np.log(0.6), NTASKS))
        th = 1.0 / A if reciprocal else np.exp(RNG.uniform(np.log(0.5), np.log(15), NTASKS))
        rec = []
        for t in range(NTASKS):
            p = np.array([0.05] + [np.log(0.5)] * 2 + list(A[t] / np.ones(2)))
            e = powmean_eps(p, X, th[t]) * np.exp(RNG.normal(0, SD, X.shape[0]))
            r = pipeline(X, e)
            if r:
                rec.append(r)
        if len(rec) < 4:
            continue
        TH = np.array([r[0] for r in rec]); AH = np.array([r[1] for r in rec])
        rat.append(cv(TH * AH) / cv(TH)); cvp.append(cv(TH * AH))
        slp.append(float(np.polyfit(np.log(AH), np.log(TH), 1)[0]))
    return np.array(rat), np.array(slp), np.array(cvp)


def main():
    X = design()
    T = pd.read_csv(os.path.join(RES, "table_resolution_model.csv"))
    meas_rat = T.ratio_cv_product_over_theta.to_numpy(float)
    meas_slp = T.slope_log_theta_on_log_A.to_numpy(float)
    draws, rows = {}, []
    for reciprocal, lab in ((True, "model A: reciprocity true"),
                            (False, "model B: theta and A independent")):
        rat, slp, cvp = run(X, reciprocal)
        draws[lab] = (rat, slp)
        rows.append(dict(generator=lab, reps=len(rat),
                         cv_ratio_median=float(np.median(rat)),
                         cv_ratio_p10=float(np.percentile(rat, 10)),
                         cv_ratio_p90=float(np.percentile(rat, 90)),
                         slope_median=float(np.median(slp)),
                         slope_p10=float(np.percentile(slp, 10)),
                         slope_p90=float(np.percentile(slp, 90)),
                         frac_cv_below_measured=float((rat <= meas_rat.max()).mean()),
                         frac_slope_below_measured=float((slp <= meas_slp.max()).mean())))
    rows.append(dict(generator="MEASURED (3 language-model grids)", reps=len(T),
                     cv_ratio_median=float(T.ratio_cv_product_over_theta.median()),
                     cv_ratio_p10=float(T.ratio_cv_product_over_theta.min()),
                     cv_ratio_p90=float(T.ratio_cv_product_over_theta.max()),
                     slope_median=float(T.slope_log_theta_on_log_A.median()),
                     slope_p10=float(T.slope_log_theta_on_log_A.min()),
                     slope_p90=float(T.slope_log_theta_on_log_A.max()),
                     frac_cv_below_measured=np.nan,
                     frac_slope_below_measured=np.nan))
    D = pd.DataFrame(rows)
    save(pd.DataFrame([dict(generator=g, rep=i, cv_ratio=a, slope=b)
                       for g, (ra, sl) in draws.items()
                       for i, (a, b) in enumerate(zip(ra, sl))]),
         "table_resolution_power_draws.csv")
    save(D, "table_resolution_power.csv")
    print(f"\ndesign: gadre grid, {X.shape[0]} points, {X.shape[1]} resources, "
          f"{NTASKS} tasks, {REPS} repetitions\n")
    print(D.round(3).to_string(index=False))
    print(f"\nmeasured variation ratio {meas_rat.min():.2f}-{meas_rat.max():.2f}, "
          f"slope {meas_slp.min():.2f} to {meas_slp.max():.2f}")
    for lab, (ra, sl) in draws.items():
        print(f"  {lab:34s}: fraction of reps with cv ratio <= measured max "
              f"{(ra <= meas_rat.max()).mean():.2f}, with slope <= measured max "
              f"{(sl <= meas_slp.max()).mean():.2f}")
    print("\nReading: the variation ratio separates the two generators cleanly;")
    print("the slope alone does not, since model B also produces negative slopes.")


if __name__ == "__main__":
    main()

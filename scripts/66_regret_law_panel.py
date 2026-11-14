#!/usr/bin/env python3
"""THE REGRET LAW ON THE PANEL  (Section "Law III" of paper.tex).

Law III predicts, with no free parameter, where the candidates of a frontier
sit in regret.  For an operating point placed uniformly in ARC LENGTH -- the
Fisher-Rao measure, which is also the measure in which the covering of the
covering law is uniform -- the regret from the canonical point obeys

    P(R <= rho) = (2/pi) arccos e^{-rho}   ~   (2 sqrt 2/pi) sqrt(rho),
    E[R] = ln 2  (one bit),   median R = (ln 2)/2 .

A published frontier covers only part of the arc, so the test is conditional:
given the arc interval [phi_lo, phi_hi] a frontier actually spans, the
predicted fraction of its candidates within rho nats of optimal is the
rho-band's share of THAT interval, and the predicted mean regret is the
interval average of -ln sin phi.  Both are compared with what the candidates do.

Inputs: results/table_acconly_pairs.csv (per-cell candidate error lists and
fitted floors, 54 cells).  Uses route (i) of the three-estimator proposition,
phi = 2 arcsin sqrt(eps_inf/eps), so only cells with an attainable floor enter.

Writes (results/):
    table_regret_panel.csv          per cell: observed vs predicted
    table_regret_panel_summary.csv  pooled, and the unconditional constants
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
RHO = 0.05
BAND_HALF = np.arccos(np.exp(-RHO))        # = arcsin(e^-rho) complement; 0.3136 rad


def band_share(lo, hi, half=BAND_HALF):
    """fraction of [lo,hi] inside [pi/2-half, pi/2+half]"""
    a, b = max(lo, np.pi / 2 - half), min(hi, np.pi / 2 + half)
    return max(b - a, 0.0) / (hi - lo)


def mean_R(lo, hi, n=20001):
    p = np.linspace(lo, hi, n)
    return float(np.trapezoid(-np.log(np.sin(p)), p) / (hi - lo))


def median_R(lo, hi, n=20001):
    p = np.linspace(lo, hi, n)
    return float(np.median(-np.log(np.sin(p))))


P = pd.read_csv(os.path.join(RES, "table_acconly_pairs.csv"))
rows = []
for (fr, cost), d in P.groupby(["frontier", "cost"]):
    eps = np.unique(np.r_[d.eps_hi.values, d.eps_lo.values])
    f = float(d.eps_inf.iloc[0])
    u = f / eps
    if not np.all(np.isfinite(u)) or u.max() >= 1 or f <= 0:
        continue                                   # floor at/above best observed error
    phi = 2 * np.arcsin(np.sqrt(u))
    R = -np.log(np.sin(phi))
    lo, hi = phi.min(), phi.max()
    if hi - lo < 1e-6:
        continue
    rows.append(dict(
        field=d.field.iloc[0], frontier=fr, cost=cost, n=len(eps),
        phi_lo=lo, phi_hi=hi, arc=hi - lo, arc_frac=(hi - lo) / np.pi,
        obs_frac_in_band=float((R <= RHO).mean()), pred_frac_in_band=band_share(lo, hi),
        obs_mean_R=float(R.mean()), pred_mean_R=mean_R(lo, hi),
        obs_median_R=float(np.median(R)), pred_median_R=median_R(lo, hi),
        law_rms=float(d.law_rms_nats.iloc[0])))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(RES, "table_regret_panel.csv"), index=False)

ob, pr = T.obs_frac_in_band, T.pred_frac_in_band
from scipy.stats import spearmanr, wilcoxon
s = dict(
    cells_used=len(T), cells_total=P.groupby(["frontier", "cost"]).ngroups,
    median_arc_frac=float(T.arc_frac.median()),
    # conditional prediction vs observation
    median_obs_frac_in_band=float(ob.median()), median_pred_frac_in_band=float(pr.median()),
    mean_abs_err_frac=float(np.mean(np.abs(ob - pr))),
    spearman_frac=float(spearmanr(ob, pr).statistic),
    p_wilcoxon_frac=float(wilcoxon(ob - pr).pvalue),
    median_obs_mean_R=float(T.obs_mean_R.median()), median_pred_mean_R=float(T.pred_mean_R.median()),
    spearman_meanR=float(spearmanr(T.obs_mean_R, T.pred_mean_R).statistic),
    median_ratio_meanR=float((T.obs_mean_R / T.pred_mean_R).median()),
    p_wilcoxon_meanR=float(wilcoxon(T.obs_mean_R - T.pred_mean_R).pvalue),
    # the unconditional constants, for reference
    full_arc_mean_R=float(np.log(2)), full_arc_median_R=float(np.log(2) / 2),
    full_arc_band_frac=float(2 * BAND_HALF / np.pi),
    sqrt_law_constant=float(2 * np.sqrt(2) / np.pi),
)
pd.DataFrame([s]).to_csv(os.path.join(RES, "table_regret_panel_summary.csv"), index=False)
for k, v in s.items():
    print(f"  {k:<28} {v}")
print("\nwrote table_regret_panel.csv, table_regret_panel_summary.csv")

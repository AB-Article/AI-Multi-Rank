#!/usr/bin/env python3
"""Summary of the factorial test by corpus (Table factorial)."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
R = pd.concat([pd.read_csv(os.path.join(RES, f)) for f in
               ("table_factorial_N_test_lm.csv", "table_factorial_N_test_nas.csv")])
R.to_csv(os.path.join(RES, "table_factorial_N_test.csv"), index=False)
def corpus(r):
    f = r.frontier
    if f.startswith("chinchilla"): return "1 Chinchilla (loss)"
    if f.startswith("porian"): return "2 Porian (loss)"
    if f.startswith("gadre") and "loss" in f: return "3 Gadre (loss)"
    if f.startswith("gadre"): return "4 Gadre (downstream)"
    if f.startswith("pythia"): return "5 Pythia (downstream)"
    return "6 NATS-SSS widths"
R["corpus"] = R.apply(corpus, axis=1)
R["theta0_out"] = R.theta_lo > 0
R["theta1_in"] = (R.theta_lo <= 1) & (R.theta_hi >= 1)
R["informative"] = ~((R.theta_lo == 0) & (R.theta_hi >= 15))
R["nn_extrap_wins"] = R.extrap_mape_theta1 < R.extrap_mape_theta0
S = R.groupby("corpus").agg(frontiers=("frontier", "size"), points=("m", "sum"),
        theta_min=("theta_hat", "min"), theta_med=("theta_hat", "median"), theta_max=("theta_hat", "max"),
        theta0_excluded=("theta0_out", "sum"), theta1_inside=("theta1_in", "sum"),
        informative=("informative", "sum"),
        dBIC_med=("dBIC_1_minus_0", "median"), gamma_neg=("tukey_gamma", lambda s: int((s < 0).sum())),
        mape_N1=("extrap_mape_theta0", "median"), mape_Nn=("extrap_mape_theta1", "median"),
        Nn_extrap_wins=("nn_extrap_wins", "sum"))
S.to_csv(os.path.join(RES, "table_factorial_summary.csv"))
pd.set_option("display.width", 250); print(S.round(2).to_string())
lm = R[~R.corpus.str.contains("NATS")]
print("LM:", len(lm), "frontiers", lm.m.sum(), "points; theta0 excluded", int(lm.theta0_out.sum()),
      "; informative", int(lm.informative.sum()), "; theta1 inside", int(lm.theta1_in.sum()),
      "; gamma<0", int((lm.tukey_gamma < 0).sum()))
p = R[R.corpus.str.contains("Pythia")]
print("Pythia contrasts: obs median", p.contrast_med_obs.median(), "range", p.contrast_med_obs.min(), p.contrast_med_obs.max(),
      "; N1 pred", p.contrast_med_N1.min(), p.contrast_med_N1.max(), "; frac>0", p.contrast_pos_obs.min(), p.contrast_pos_obs.max())
print("Pythia dBIC median", p.dBIC_1_minus_0.median(), "theta_lo min", p.theta_lo.min())
print(R[["frontier","theta_hat","theta_lo","theta_hi","dBIC_1_minus_0","tukey_gamma","N_hat","extrap_mape_theta0","extrap_mape_theta1"]].round(2).to_string())

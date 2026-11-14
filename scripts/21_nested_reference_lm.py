#!/usr/bin/env python3
"""THE NESTED CANONICAL REFERENCE, READ OFF THE LANGUAGE-MODEL GRIDS.
   (Technical report, Thm. 'nested', "Interacting resources: the nested dual";
    paper, Thm. 'canonical', which is the case theta = 1.)

Thm. 'nested' says that when n resources interact with nesting exponent theta
the minimax declaration is no longer uniform on n+1 cells.  Writing

    m = n^(1/theta),

the rule declares reducible share m/(1+m), its worst case is ln(1+m) nats, and
it selects

    eps* = (1 + m) eps_inf,     Delta S = ln(1 + m),

feasible exactly when eps_inf <= 1/(1+m).  At theta = 1 this is Thm.
'canonical' with m = n: uniform shares and ln(n+1).

theta is not a free parameter here.  It is the power-mean exponent of script
14's factorial law, eps = einf + (mean_i g_i^theta)^(1/theta), already fitted
per frontier with a 95% profile-likelihood interval; theta = 1 is the additive
("N = n") law and theta = 0 the single channel.  This script reads those fits
for the language-model grids and reports what the nested reference does to the
operating point that assuming additivity would have named.

Columns: the fitted theta and its interval, the implied m, the declared
reducible share and worst-case regret, the selected depth Delta S, the
feasibility edge on eps_inf, and the shift from the theta = 1 reference --
d_depth = ln(1+m) - ln(1+n), the number of nats by which assuming additivity
misplaces the canonical depth.

Reads  results/table_factorial_N_test_lm.csv   (script 14)
Writes results/table_nested_reference_lm.csv
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
SRC = os.path.join(RES, "table_factorial_N_test_lm.csv")

THETA_MIN = 1e-3            # below this the power mean is the geometric one and
                            # m = n^(1/theta) overflows: reported, not computed.


def m_of(n, theta):
    if theta is None or not np.isfinite(theta) or theta <= THETA_MIN:
        return np.inf
    return float(n) ** (1.0 / float(theta))


def row_for(r):
    n = int(r["n_res"])
    out = dict(field=r["field"], frontier=r["frontier"], n_res=n,
               theta_hat=r["theta_hat"], theta_lo=r["theta_lo"],
               theta_hi=r["theta_hi"], eps_inf=r.get("einf_theta1", np.nan))
    m = m_of(n, r["theta_hat"])
    out["m"] = m
    # the theta = 1 (additive) reference, which is Thm. 'canonical'
    out["m_theta1"] = float(n)
    out["depth_theta1"] = float(np.log1p(n))
    out["share_theta1"] = n / (n + 1.0)
    if np.isfinite(m):
        out["reducible_share"] = m / (1.0 + m)
        out["worst_case_regret"] = float(np.log1p(m))
        out["depth_nats"] = float(np.log1p(m))
        out["eps_star_over_eps_inf"] = 1.0 + m
        out["feasible_eps_inf_max"] = 1.0 / (1.0 + m)
        out["d_depth_vs_theta1"] = float(np.log1p(m) - np.log1p(n))
        ei = out["eps_inf"]
        out["feasible"] = bool(np.isfinite(ei) and ei <= 1.0 / (1.0 + m))
    else:
        for k in ("reducible_share", "worst_case_regret", "depth_nats",
                  "eps_star_over_eps_inf", "feasible_eps_inf_max",
                  "d_depth_vs_theta1"):
            out[k] = np.inf
        out["feasible"] = False
    # interval on the depth, from the profile interval on theta
    ms = [m_of(n, t) for t in (r["theta_lo"], r["theta_hi"])]
    ds = [np.log1p(x) if np.isfinite(x) else np.inf for x in ms]
    out["depth_lo"], out["depth_hi"] = float(min(ds)), float(max(ds))
    # does the additive reference sit inside the interval the data allow?
    out["theta1_inside_ci"] = bool(r["theta_lo"] <= 1.0 <= r["theta_hi"])
    out["depth_theta1_inside"] = bool(out["depth_lo"] <= np.log1p(n) <= out["depth_hi"])
    return out


def main():
    src = pd.read_csv(SRC)
    rows = [row_for(r) for _, r in src.iterrows()]
    df = pd.DataFrame(rows)
    out = os.path.join(RES, "table_nested_reference_lm.csv")
    df.to_csv(out, index=False)
    print(f"  -> results/table_nested_reference_lm.csv  ({len(df)} frontiers)")

    fin = df[np.isfinite(df.depth_nats)]
    print(f"     theta = 1 inside the 95% interval on "
          f"{int(df.theta1_inside_ci.sum())} of {len(df)} grids")
    if len(fin):
        print(f"     median nested depth {fin.depth_nats.median():.3f} nats "
              f"against ln(1+n) = {np.log1p(fin.n_res).median():.3f}")
        print(f"     median |shift| from assuming additivity "
              f"{fin.d_depth_vs_theta1.abs().median():.3f} nats; "
              f"max {fin.d_depth_vs_theta1.abs().max():.3f}")
        print(f"     the additive depth sits inside the interval the data allow "
              f"on {int(fin.depth_theta1_inside.sum())} of {len(fin)}")


if __name__ == "__main__":
    main()

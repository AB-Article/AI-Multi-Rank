#!/usr/bin/env python3
"""THE ARC IS A LOWER BOUND ON THE NUMBER OF MODES, NOT THE COUNT.
   (Section "Law III", Prop. 'arcweights'; Section "Reading a frontier without
   fitting it", the channel-count paragraph.)

Proposition 'path'(iii) already bounds the Fisher-Rao length of a frontier with
N+1 atoms by pi <= L <= N pi, and already notes that the upper bound is reached
only when every mode dominates the error over a long stretch.  Script 89 makes
that exact: ordering the atoms by increasing rate, with P_k = p_1+...+p_k,

    L  =  pi + sum_{k=2}^{K-1} 2 arcsin sqrt( p_k / P_k )   <=  (K-1) pi .

Two things follow, and this script records both against the archive's own
tables.

  A1  L/pi is a LOWER bound on the number of reducible channels, not an
      estimate of it: N >= L/pi, i.e. the number of atoms is at least 1 + L/pi.
      The archive's own verification table already shows this -- three
      coincident atoms (N=2) give L = 1.297 pi, three separated atoms give
      1.500 pi -- so the quantity to report is a bound and its direction is
      known.

  A2  L < pi CERTIFIES that exactly two modes are resolved, because the K=3
      minimum of the formula above exceeds pi for every weight vector.  On the
      archive this converts "no frontier traverses even one complete
      transition" into a positive statement: on every frontier of the corpus the
      measured curvature is that of a two-atom spectrum, which is what licenses
      the single-channel arithmetic used throughout the paper.

Outputs: results/table_mode_lower_bound.csv
         results/table_mode_lower_bound_summary.csv
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _loaders import save

RES = os.path.join(os.path.dirname(HERE), "results")


def run():
    A = pd.read_csv(os.path.join(RES, "table_arc_per_frontier.csv"))
    A = A.dropna(subset=["arc"])
    A["L_over_pi"] = A.arc / np.pi
    A["modes_lower_bound"] = 1 + A.L_over_pi
    A["channels_lower_bound"] = A.L_over_pi
    A["two_modes_certified"] = A.arc < np.pi
    keep = [c for c in ["eval_set", "block", "n_pareto", "span_nats", "arc",
                        "L_over_pi", "channels_lower_bound", "modes_lower_bound",
                        "two_modes_certified"] if c in A.columns]
    save(A[keep], "table_mode_lower_bound.csv")

    V = pd.read_csv(os.path.join(RES, "table_sum_rules.csv"))
    rows = [
        dict(quantity="frontiers with a measured arc", value=float(len(A))),
        dict(quantity="median L/pi", value=float(A.L_over_pi.median())),
        dict(quantity="max L/pi", value=float(A.L_over_pi.max())),
        dict(quantity="frontiers with L < pi (exactly two modes certified)",
             value=float(A.two_modes_certified.sum())),
        dict(quantity="fraction with L < pi", value=float(A.two_modes_certified.mean())),
        dict(quantity="median lower bound on the number of modes, 1 + L/pi",
             value=float(A.modes_lower_bound.median())),
        dict(quantity="frontiers whose arc would admit a third mode (L > pi)",
             value=float((~A.two_modes_certified).sum())),
    ]
    # the archive's own synthetic check: L/pi against the true channel count
    for _, r in V.iterrows():
        if "N_channels" in V.columns and np.isfinite(r.get("L_over_pi", np.nan)):
            rows.append(dict(quantity=f"verification: {r['spectrum']} -> true N={int(r['N_channels'])}, "
                                      f"L/pi", value=float(r["L_over_pi"])))
    S = pd.DataFrame(rows)
    save(S, "table_mode_lower_bound_summary.csv")
    print(S.to_string(index=False))


if __name__ == "__main__":
    run()

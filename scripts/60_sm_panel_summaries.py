#!/usr/bin/env python3
"""Panel summaries quoted in the Online Appendix (Tables S-channels and S-runfloor).

Both tables are derived from results already written by scripts 23 and 30;
this script only aggregates them into the form printed in the appendix, so
that every number there has a single reproducible source.

Reads  results/table_resolution_frontiers.csv, table_resolution_summary.csv,
       table_running_floor_summary.csv
Writes results/table_sm_channel_count.csv   (per-panel channel-count rows and
                                             the N_hat-vs-n_res counts)
       results/table_sm_running_floor.csv   (the running-floor table, rounded)
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")

# ---------------------------------------------------------------- channels
fr = pd.read_csv(os.path.join(RESULTS, "table_resolution_frontiers.csv"))
summ = pd.read_csv(os.path.join(RESULTS, "table_resolution_summary.csv"))

rows = []
for corpus in ("82 frontiers", "LM factorial"):
    sub = fr[fr.corpus == corpus]
    for nhat, g in sub.groupby("N_hat"):
        rows.append(dict(panel=corpus, N_hat=int(nhat), frontiers=len(g),
                         median_Z=g.Z.median(), median_m=g.m.median(),
                         median_sigma=g.sigma.median(), median_V=g.V.median(),
                         n_below_resources=int((g.N_hat < g.n_res).sum()),
                         n_above_resources=int((g.N_hat > g.n_res).sum())))
    s = summ[(summ.corpus == corpus) & (summ.N_hat == -1)].iloc[0]
    rows.append(dict(panel=corpus, N_hat=-1, frontiers=len(sub),
                     n_below_resources=int((sub.N_hat < sub.n_res).sum()),
                     n_above_resources=int((sub.N_hat > sub.n_res).sum()),
                     frac_N1=float((sub.N_hat == 1).mean()),
                     auc=float(s.auc), mw_p_greater=float(s.mw_p_greater),
                     log10V_at_half=float(s.log10V_at_half),
                     sim_log10V_at_half=float(s.sim_log10V_at_half)))
ch = pd.DataFrame(rows)
ch.to_csv(os.path.join(RESULTS, "table_sm_channel_count.csv"), index=False)
print(ch.to_string(index=False))

# ------------------------------------------------------------ running floor
rf = pd.read_csv(os.path.join(RESULTS, "table_running_floor_summary.csv"))
out = pd.DataFrame(dict(
    window=rf.window,
    frontiers=rf.frontiers,
    frac_identified=rf.frac_floor_identified.round(2),
    median_ratio=rf.median_ratio.round(2),
    q25_ratio=rf.q25_ratio.round(2),
    q75_ratio=rf.q75_ratio.round(2),
    frac_above_gapped_null_q95=rf.frac_above_gap_null_q95.round(2),
    median_gapped_null=rf.median_null_gap.round(2),
    median_gapless_null=rf.median_null_nogap.round(2),
))
out.to_csv(os.path.join(RESULTS, "table_sm_running_floor.csv"), index=False)
print(out.to_string(index=False))

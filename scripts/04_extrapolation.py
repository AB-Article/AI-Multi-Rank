#!/usr/bin/env python3
"""Held-out test of the normal form: fit gauge on the cheap 60% of a frontier,
predict the expensive 40%.  The normal form has no shape parameter."""
import numpy as np, pandas as pd
from common import frontier, load_resources, fit_law, LAWS, EVAL_FILES, save
B = load_resources()

BLOCKS = {"latency":["lat_rtx3090"], "params":["param_count"],
          "latency x params":["lat_rtx3090","param_count"],
          "GPU x CPU latency":["lat_rtx3090","lat_cpu_i9"]}
FLOOR_PRESENT = {"ImageNet-1k","ImageNet-ReaL","ImageNet-V2"}

def split(X,e,frac=0.6):
    o=np.argsort(X.sum(1)); X,e=X[o],e[o]; m=int(round(frac*len(e)))
    return X[:m],e[:m],X[m:],e[m:]

rows=[]
for ev in EVAL_FILES:
    for bn,cols in BLOCKS.items():
        try: d,X,e = frontier(ev,cols,resources=B)
        except Exception: continue
        if len(e) < 30: continue
        Xtr,etr,Xte,ete = split(X,e)
        rec = dict(eval_set=ev, resources=bn, n_train=len(etr), n_test=len(ete),
                   floor_present=(ev in FLOOR_PRESENT))
        for nm in LAWS:
            th = fit_law(nm, Xtr, etr)
            pred = LAWS[nm][0](th, Xte)
            rec[f"mape_{nm.replace(' ','_')}"] = float(np.mean(np.abs(pred-ete)/ete))
            if nm == "normal form":
                rec["einf_hat"] = th[0]
        rows.append(rec)
ex = pd.DataFrame(rows)
save(ex, "table_extrapolation.csv")

cols = [c for c in ex.columns if c.startswith("mape_")]
print((ex[["eval_set","resources","n_test","floor_present"]+cols]
       .assign(**{c:(ex[c]*100).round(1) for c in cols}).to_string(index=False)))
summ = ex.groupby("floor_present")[cols].median().mul(100).round(2)
wins = ex.assign(nf_wins=ex["mape_normal_form"] < ex["mape_floorless"]) \
         .groupby("floor_present").agg(n=("nf_wins","size"), nf_beats_floorless=("nf_wins","sum"))
print("\nmedian MAPE (%) by floor status:\n", summ.to_string())
print("\nnormal form vs floorless power law:\n", wins.to_string())
save(summ.reset_index(), "table_extrapolation_summary.csv")
save(wins.reset_index(), "table_extrapolation_wins.csv")

#!/usr/bin/env python3
"""Extract Pareto frontiers and write them to ../results/frontiers/."""
import os, itertools, numpy as np, pandas as pd
from common import (load_resources, frontier, EVAL_FILES, RESULTS, save)

os.makedirs(os.path.join(RESULTS, "frontiers"), exist_ok=True)
B = load_resources()
print(f"resource table: {len(B)} (architecture, resolution) rows")

BLOCKS = {
    "lat3090":            ["lat_rtx3090"],
    "params":             ["param_count"],
    "gmacs":              ["infer_gmacs"],
    "lat3090_params":     ["lat_rtx3090", "param_count"],
    "lat3090_cpui9":      ["lat_rtx3090", "lat_cpu_i9"],
    "lat3090_gmacs":      ["lat_rtx3090", "infer_gmacs"],
    "gmacs_params":       ["infer_gmacs", "param_count"],
    "cpui9_params":       ["lat_cpu_i9", "param_count"],
    "train_lat3090":      ["train_us", "lat_rtx3090"],
    "macts_params":       ["infer_macts", "param_count"],
    "five_resources":     ["lat_rtx3090", "lat_cpu_i9", "param_count",
                           "infer_gmacs", "infer_macts"],
}

index = []
for ev in EVAL_FILES:
    for bname, cols in BLOCKS.items():
        try:
            d, X, e = frontier(ev, cols, resources=B)
        except Exception:
            continue
        if len(e) < 8:
            continue
        out = d[["base", "img_size", "eps"] + cols].copy()
        fn = f"frontiers/{ev}__{bname}.csv"
        out.to_csv(os.path.join(RESULTS, fn), index=False)
        rho = (np.corrcoef(X[:, 0], X[:, 1])[0, 1] if X.shape[1] == 2 else np.nan)
        index.append(dict(eval_set=ev, block=bname, n_resources=len(cols),
                          n_pareto=len(e), corr_logres=rho,
                          eps_min=e.min(), eps_max=e.max(), file=fn))
save(pd.DataFrame(index), "frontier_index.csv")

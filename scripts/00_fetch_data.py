#!/usr/bin/env python3
"""Download the two public corpora into ../data/.  Re-run only if data/ is empty."""
import os, subprocess, sys, glob, yaml, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

def sh(*c):
    print("$", " ".join(c)); subprocess.run(c, check=True)

# ---- timm: sparse clone of the results directory only ----------------------
timm = os.path.join(DATA, "timm_raw")
if not os.path.isdir(timm) or not os.listdir(timm):
    tmp = os.path.join(DATA, "_timm")
    sh("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
       "https://github.com/huggingface/pytorch-image-models.git", tmp)
    sh("git", "-C", tmp, "sparse-checkout", "set", "results")
    os.rename(os.path.join(tmp, "results"), timm)
    subprocess.run(["rm", "-rf", tmp])
print(f"timm result files: {len(os.listdir(timm))}")

# ---- Matbench Discovery: model metadata -> flat table ----------------------
out = os.path.join(DATA, "matbench_models.csv")
if not os.path.exists(out):
    tmp = os.path.join(DATA, "_mbd")
    sh("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
       "https://github.com/janosh/matbench-discovery.git", tmp)
    sh("git", "-C", tmp, "sparse-checkout", "set", "models", "data")
    reg = {k: (v.get("n_structures") or v.get("n_materials"))
           for k, v in yaml.safe_load(open(os.path.join(tmp, "data/datasets.yml"))).items()
           if isinstance(v, dict)}
    rows = []
    for f in glob.glob(os.path.join(tmp, "models/*/*.yml")):
        try: y = yaml.safe_load(open(f))
        except Exception: continue
        if not isinstance(y, dict): continue
        disc = (y.get("metrics") or {}).get("discovery") or {}
        full = (disc.get("full_test_set") or {}) if isinstance(disc, dict) else {}
        ts = y.get("training_sets")
        n = sum(reg.get(t) or 0 for t in ts if isinstance(t, str)) if isinstance(ts, list) else None
        rows.append(dict(model=y.get("model_name"), params=y.get("model_params"),
                         ntrain=n or None, F1=full.get("F1"), MAE=full.get("MAE")))
    pd.DataFrame(rows).dropna(subset=["F1", "params", "ntrain"]).to_csv(out, index=False)
    subprocess.run(["rm", "-rf", tmp])
print(f"matbench models: {len(pd.read_csv(out))}")

#!/usr/bin/env python3
"""Build the FACTORIAL corpora used to test N = n (one channel per resource).

Every earlier corpus is a leaderboard: resources co-vary along the frontier, so
"one channel per resource" and "one shared channel" are nearly unidentifiable
(Corollary rank).  Testing N = n needs designs in which the resources are varied
INDEPENDENTLY.  Four public sources provide such designs for the pair
(parameters, training tokens), and one supplies five independently varied
per-stage widths.  All are fetched from GitHub (pinned commits below) with

    git clone --depth 1 https://github.com/<repo>.git  ext/<name>

and flattened here to  data/grid_factorial.csv  with columns
    field, frontier, item, eps, cost_params, cost_tokens, [cost_w1..cost_w5]

Sources (commit hashes are those used for the paper):
  chinchilla  epoch-research/analyzing-chinchilla      9225883  (Besiroglu et al. 2024)
  porian      formll/resolving-scaling-law-discrepancies 0dc60d9 (Porian et al. 2024)
  gadre       mlfoundations/scaling                     a003c49  (Gadre et al. 2024)
  pythia      EleutherAI/pythia                         a19eecb  (Biderman et al. 2023)
  nats-sss    already bundled: data/field_vision_size_natssss.csv

Error conventions.  Losses are converted to eps = L / ln V (V = vocabulary),
i.e. loss relative to the uniform predictor, which is the chance level of
next-token prediction; the functional-form tests below are invariant to this
rescaling (gauge G1).  Downstream accuracies are chance-corrected exactly as in
Eq. (defs).
"""
import os, glob, json, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.environ.get("EXT", os.path.join(os.path.dirname(ROOT), "ext"))
OUT = os.path.join(ROOT, "data", "grid_factorial.csv")
rows = []

# ---------------------------------------------------------------- Chinchilla
f = os.path.join(EXT, "analyzing-chinchilla", "data", "svg_extracted_data.csv")
d = pd.read_csv(f)
for i, r in d.iterrows():
    N, C, L = r["Model Size"], r["Training FLOP"], r["loss"]
    rows.append(dict(field="lm-loss", frontier="chinchilla|MassiveText",
                     item=f"pt{i}", eps=L / np.log(32000),
                     cost_params=N, cost_tokens=C / (6 * N)))

# ---------------------------------------------------------------- Porian
sys.path.insert(0, os.path.join(EXT, "resolving-scaling-law-discrepancies"))
from data import precise_param_count_open_lm  # noqa: E402
d = pd.read_pickle(os.path.join(EXT, "resolving-scaling-law-discrepancies",
                                "data", "experiment_results.pickle.xz"))
d = d[(d.hparams == "base") & (d.decay == "chinchilla")]
for _, r in d.iterrows():
    v = r["val/loss"].dropna()
    if len(v) == 0:
        continue
    step = int(v.index[-1])
    if step < 0.9 * r.max_step:          # final (annealed) value only
        continue
    P = float(precise_param_count_open_lm(np.array([r.width]), np.array([r.depth]))[0])
    rows.append(dict(field="lm-loss", frontier=f"porian|{r.dataset}",
                     item=f"w{int(r.width)}d{int(r.depth)}s{int(r.max_step)}",
                     eps=float(v.iloc[-1]) / np.log(r.vocab_size),
                     cost_params=P, cost_tokens=float(r.max_step * r.bs * r.seq_len)))

# ---------------------------------------------------------------- Gadre
G = os.path.join(EXT, "scaling", "exp_data")
meta = pd.read_csv(os.path.join(G, "eval_metadata.csv"))
chance = dict(zip(meta["Eval Task"], meta["Random baseline"] / 100.0))
models = {}
for f in glob.glob(os.path.join(G, "models", "*.json")):
    m = json.load(open(f))
    h = m["hyperparameters"]
    c4 = [r["loss"] for r in m["results"] if "c4_val" in r["val_data"][0]][0]
    models[m["name"]] = dict(ds=m["dataset_name"], P=h["params"], T=h["tokens"], c4=c4)
acc = {}
for f in glob.glob(os.path.join(G, "evals", "*.json")):
    name = os.path.basename(f)[len("evaluation_"):-len("_heavy.json")]
    acc[name] = json.load(open(f))["eval_metrics"]["icl"]
A = pd.DataFrame(acc).T
# downstream task set fixed BEFORE fitting: multiple-choice tasks with a
# positive chance level on which the best model clears chance by >= 10 points
mc = [t for t in A.columns if chance.get(t, 0) > 0]
keep = [t for t in mc if A[t].max() - chance[t] >= 0.10]
for name, m in models.items():
    rows.append(dict(field="lm-loss", frontier=f"gadre|{m['ds']}|c4-val-loss",
                     item=name, eps=m["c4"] / np.log(50432),
                     cost_params=m["P"], cost_tokens=m["T"]))
    if name in A.index:
        e = np.mean([(1 - A.loc[name, t]) / (1 - chance[t]) for t in keep])
        rows.append(dict(field="lm-downstream", frontier=f"gadre|{m['ds']}|{len(keep)}-task",
                         item=name, eps=e, cost_params=m["P"], cost_tokens=m["T"]))
print(f"gadre downstream task set ({len(keep)}): {', '.join(sorted(keep))}")

# ---------------------------------------------------------------- Pythia
PY = os.path.join(EXT, "pythia", "evals", "pythia-v1")
PARAMS = {"70m": 70426624, "160m": 162322944, "410m": 405334016,
          "1b": 1011781632, "1.4b": 1414647808, "2.8b": 2775208960,
          "6.9b": 6857302016, "12b": 11846072320}
TOK_PER_STEP = 1024 * 2048
TASKS = {"lambada_openai": 0.0, "piqa": 0.5, "arc_easy": 0.25,
         "sciq": 0.25, "winogrande": 0.5}
for dedup in ["", "-deduped"]:
    for size, P in PARAMS.items():
        for f in glob.glob(os.path.join(PY, f"pythia-{size}{dedup}", "zero-shot", "*.json")):
            step = int(os.path.basename(f).split("step")[1][:-5])
            if step < 3000:          # past the 1% warm-up; see text
                continue
            res = json.load(open(f))["results"]
            eps = {t: (1 - res[t]["acc"]) / (1 - q) for t, q in TASKS.items()}
            corp = "pythia" + dedup
            for t, e in eps.items():
                rows.append(dict(field="lm-downstream", frontier=f"{corp}|{t}",
                                 item=f"{size}@{step}", eps=e,
                                 cost_params=P, cost_tokens=step * TOK_PER_STEP))
            rows.append(dict(field="lm-downstream", frontier=f"{corp}|5-task",
                             item=f"{size}@{step}", eps=float(np.mean(list(eps.values()))),
                             cost_params=P, cost_tokens=step * TOK_PER_STEP))

# ---------------------------------------------------------------- NATS-SSS
d = pd.read_csv(os.path.join(ROOT, "data", "field_vision_size_natssss.csv"))
W = d["item"].str.split(":", expand=True).astype(float)
for k in range(5):
    d[f"cost_w{k+1}"] = W[k]
d["eps"] = (1 - d.quality) / (1 - d.chance)
for _, r in d.iterrows():
    rows.append(dict(field="nas-widths", frontier=r.frontier, item=r["item"], eps=r.eps,
                     **{f"cost_w{k+1}": r[f"cost_w{k+1}"] for k in range(5)}))

D = pd.DataFrame(rows)
D.to_csv(OUT, index=False)
print(D.groupby(["field", "frontier"]).size().to_string())
print(f"-> {OUT}  ({len(D)} rows)")

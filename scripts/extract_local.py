#!/usr/bin/env python3
"""
extract_local.py — run this on a machine with unrestricted internet, then upload
ONLY the small CSVs it writes to ./upload_me/.

Each block is independent. Run either; neither is needed to reproduce a
published number, because the extracts they write are already in data/.

    python extract_local.py nb201     --nb201-file /path/NAS-Bench-201-v1_1-096897.pth
    python extract_local.py tabrepo

Blocks for HELM, precisionFDA and WeatherBench were removed: no corpus in the article draws on them, so the code was dead and
named dependencies the archive does not declare.

Output CSVs are small (2-25 MB). Upload the whole upload_me/ folder.
"""
import argparse, os, sys, json, glob
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload_me")
os.makedirs(OUT, exist_ok=True)


def write(df, name):
    p = os.path.join(OUT, name)
    df.to_csv(p, index=False)
    print(f"  wrote {p}  ({len(df)} rows, {os.path.getsize(p)/1e6:.1f} MB)")


# =====================================================================
# 1. NAS-BENCH-201  —  HIGHEST PRIORITY
# =====================================================================
# Download (pick ONE):
#   (a) NAS-Bench-201-v1_1-096897.pth  (~4.7 GB, full, has train times)
#       Drive link in https://github.com/D-X-Y/NAS-Bench-201  (README, "v1.1")
#   (b) NATS-tss-v1_0-3ffb9-simple.tar (~500 MB, smaller, same 15625 archs)
#       Drive link in https://github.com/D-X-Y/NATS-Bench
#
#   pip install nas-bench-201        # for (a)
#   pip install nats_bench           # for (b)
#
# WHY THIS ONE MATTERS: joined with HW-NAS-Bench (which I already have — it
# ships its data in-repo) this gives 15,625 architectures x 3 datasets x
# 5 measured device latencies + 3 energies, with the Pareto set EXACT rather
# than sampled. It settles O2 (is N an observable?) on its own.
# =====================================================================
def do_nb201(args):
    rows = []
    if args.nb201_file and args.nb201_file.endswith(".pth"):
        from nas_201_api import NASBench201API as API
        api = API(args.nb201_file, verbose=False)
        for i in range(len(api)):
            arch = api.arch(i)
            rec = dict(index=i, arch_str=arch)
            for dset, acckey in [("cifar10", "ori-test"),
                                 ("cifar100", "x-test"),
                                 ("ImageNet16-120", "x-test")]:
                try:
                    info = api.get_more_info(i, dset, hp="200", is_random=False)
                    rec[f"{dset}_test_acc"] = info.get("test-accuracy")
                    rec[f"{dset}_valid_acc"] = info.get("valid-accuracy")
                    rec[f"{dset}_train_time_s"] = info.get("train-all-time")
                except Exception:
                    pass
            rows.append(rec)
            if i % 2000 == 0:
                print(f"    {i}/{len(api)}")
    else:
        from nats_bench import create
        api = create(args.nats_dir, "tss", fast_mode=True, verbose=False)
        for i in range(len(api)):
            rec = dict(index=i, arch_str=api.arch(i))
            for dset in ("cifar10", "cifar100", "ImageNet16-120"):
                try:
                    info = api.get_more_info(i, dset, hp="200", is_random=False)
                    rec[f"{dset}_test_acc"] = info.get("test-accuracy")
                    rec[f"{dset}_valid_acc"] = info.get("valid-accuracy")
                    rec[f"{dset}_train_time_s"] = info.get("train-all-time")
                except Exception:
                    pass
            cost = api.get_cost_info(i, "cifar10")
            rec["params_M"] = cost.get("params")
            rec["flops_M"] = cost.get("flops")
            rows.append(rec)
            if i % 2000 == 0:
                print(f"    {i}/{len(api)}")
    df = pd.DataFrame(rows)
    # arch_str is ESSENTIAL: I use it to verify index alignment with HW-NAS-Bench
    assert "arch_str" in df and df.arch_str.notna().all(), "arch_str missing"
    write(df, "nb201_accuracy.csv")


# =====================================================================
# 2. TABREPO  —  second field, train time AND inference time, low collinearity
# =====================================================================
#   pip install tabrepo
# Downloads ~10-30 GB from S3 on first call and caches it; the CSV out is small.
# =====================================================================
def do_tabrepo(args):
    from tabrepo import load_repository
    repo = load_repository(args.tabrepo_context, cache=True)
    recs = []
    for dataset in repo.datasets():
        for cfg in repo.configs():
            try:
                m = repo.metrics(datasets=[dataset], configs=[cfg])
            except Exception:
                continue
            if m is None or len(m) == 0:
                continue
            m = m.reset_index()
            recs.append(dict(
                dataset=dataset, config=cfg,
                metric_error=float(m["metric_error"].median()),
                time_train_s=float(m["time_train_s"].median()),
                time_infer_s=float(m["time_infer_s"].median()),
                n_folds=len(m)))
    write(pd.DataFrame(recs), "tabrepo_metrics.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["nb201", "tabrepo"])
    ap.add_argument("--nb201-file", default=None)
    ap.add_argument("--nats-dir", default=None)
    ap.add_argument("--tabrepo-context", default="D244_F3_C1530_200")
    a = ap.parse_args()
    {"nb201": do_nb201, "tabrepo": do_tabrepo}[a.which](a)

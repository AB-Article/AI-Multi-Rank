"""Shared loader for the practical-pipeline scripts 139-156.

Imports the stdlib reading layer of ``examples/model_selection/pipeline_core.py``
-- the same code the worked example runs, so the archive's numbers and the
example's numbers cannot drift apart -- and adds the evaluation-set sizes the
precision gate needs.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
EX = os.path.join(ROOT, "examples", "model_selection")
sys.path.insert(0, EX)
sys.path.insert(0, HERE)

import pipeline_core as pc                                      # noqa: E402,F401

#: Items in the evaluation set, where it is public and unambiguous.  Used only
#: by the precision gate; a frontier with no entry is reported as
#: "n_test unknown" and gets the REQUIRED n rather than a verdict.
EVAL_ITEMS = {
    "ImageNet-1k": 50_000,
    "ImageNet-ReaL": 50_000,
    "ImageNet-V2": 10_000,
    "ImageNet-Sketch": 50_889,
    "ImageNet-A": 7_500,
    "ImageNet-R": 30_000,
    "cifar10": 10_000,
    "cifar100": 10_000,
    "ImageNet16-120": 6_000,
}


def eval_items(name: str):
    """Items in the evaluation set of a frontier named ``<eval set>__<block>``."""
    head = name.split("__")[0]
    if head in EVAL_ITEMS:
        return EVAL_ITEMS[head]
    for k, v in EVAL_ITEMS.items():
        if k.lower() in name.lower():
            return v
    return None


def frontiers():
    """The 117 single-resource frontiers, as script 68 builds them."""
    import _loaders
    return _loaders.all_frontiers()


def save(rows, name, cols=None):
    import pandas as pd
    df = pd.DataFrame(rows)
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, name), index=False)
    print(f"  -> results/{name}  ({len(df)} rows)")
    return df

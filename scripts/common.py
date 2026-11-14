"""Shared utilities for the frontier analysis.

Data model
----------
A *frontier* is a set of Pareto-optimal points (eps, t_1..t_n) where eps is the
chance-corrected error and t_i are resources.  Log-resources are x_i = ln t_i.

Laws fitted (all with eps_inf >= 0):
    additive        eps = einf + sum_i c_i exp(-a_i x_i)
    multiplicative  eps = einf + c exp(-sum_i a_i x_i)
    soft-min(beta)  eps = einf + ( sum_i (c_i exp(-a_i x_i))^(1/beta) )^beta
    normal form     eps = einf * ( 1 + sum_i exp(-(x_i - x0_i)/w_i) )   [no shape param]
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TIMM = os.path.join(ROOT, "data", "timm_raw")
RESULTS = os.path.join(ROOT, "results")
CHANCE_IN1K = 1e-3  # 1000-way classification

BENCH_FILES = {
    "rtx3090": "benchmark-infer-amp-nchw-pt210-cu121-rtx3090.csv",
    "rtx4090": "benchmark-infer-amp-nchw-pt240-cu124-rtx4090.csv",
    "rtx5090": "benchmark-infer-amp-nchw-pt291-cu130-5090.csv",
    "pro6000": "benchmark-infer-amp-nchw-pt291-cu130-pro6000maxq.csv",
    "cpu_i9":  "benchmark-infer-fp32-nchw-pt240-cpu-i9_10940x-dynamo.csv",
    "cpu_i7":  "benchmark-infer-fp32-nchw-pt240-cpu-i7_12700h-dynamo.csv",
}
TRAIN_FILE = "benchmark-train-amp-nchw-pt112-cu113-rtx3090.csv"

EVAL_FILES = {
    "ImageNet-1k":     "results-imagenet.csv",
    "ImageNet-ReaL":   "results-imagenet-real.csv",
    "ImageNet-V2":     "results-imagenetv2-matched-frequency.csv",
    "ImageNet-Sketch": "results-sketch.csv",
    "ImageNet-R":      "results-imagenet-r.csv",
    "ImageNet-A":      "results-imagenet-a.csv",
}


def _num(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_accuracy(eval_name, chance=CHANCE_IN1K):
    """Best checkpoint per architecture, with chance-corrected error."""
    d = _num(pd.read_csv(os.path.join(TIMM, EVAL_FILES[eval_name])),
             ["top1", "img_size"])
    d["base"] = d["model"].str.split(".").str[0]
    d = d.sort_values("top1", ascending=False).drop_duplicates("base")
    d["eps"] = (1 - d.top1 / 100) / (1 - chance)
    return d[["base", "img_size", "eps"]].dropna()


def load_resources():
    """Merged resource table: per-image latency on six processors (microseconds),
    training step time, parameter count, MACs, activations."""
    out = None
    for tag, fn in BENCH_FILES.items():
        d = _num(pd.read_csv(os.path.join(TIMM, fn)),
                 ["infer_step_time", "infer_batch_size", "infer_img_size",
                  "param_count", "infer_gmacs", "infer_macts"])
        d["lat_" + tag] = d.infer_step_time / d.infer_batch_size * 1000.0
        keep = ["model", "infer_img_size", "lat_" + tag]
        if tag == "rtx3090":
            keep += ["param_count", "infer_gmacs", "infer_macts"]
        d = d[keep].rename(columns={"model": "base", "infer_img_size": "img_size"})
        out = d if out is None else out.merge(d, on=["base", "img_size"], how="outer")
    tr = _num(pd.read_csv(os.path.join(TIMM, TRAIN_FILE)),
              ["train_step_time", "train_batch_size", "train_img_size"])
    tr["train_us"] = tr.train_step_time / tr.train_batch_size * 1000.0
    tr = tr[["model", "train_img_size", "train_us"]].rename(
        columns={"model": "base", "train_img_size": "img_size"})
    return out.merge(tr, on=["base", "img_size"], how="left")


def pareto_mask(M):
    """Rows of M (all columns minimised) that are not dominated."""
    keep = np.ones(len(M), bool)
    for i in range(len(M)):
        if keep[i]:
            dom = np.all(M <= M[i], axis=1) & np.any(M < M[i], axis=1)
            if dom.any():
                keep[i] = False
    return keep


def frontier(eval_name, cols, resources=None):
    """Return (dataframe, X = log-resources, e = error) on the Pareto set."""
    B = load_resources() if resources is None else resources
    d = load_accuracy(eval_name).merge(B, on=["base", "img_size"])
    d = d.dropna(subset=["eps"] + list(cols))
    M = d[["eps"] + list(cols)].to_numpy(float)
    d = d[pareto_mask(M)].reset_index(drop=True)
    return d, np.log(d[list(cols)].to_numpy(float)), d["eps"].to_numpy(float)


# ---------------------------------------------------------------- laws ------
def _ic(rss, m, k):
    return m * np.log(rss / m) + 2 * k, m * np.log(rss / m) + k * np.log(m)


def fit_additive(X, e, beta_free=False):
    n = X.shape[1]

    def resid(th):
        einf, c, a = th[0], th[1:1 + n], th[1 + n:1 + 2 * n]
        b = th[-1] if beta_free else 1.0
        terms = c[None, :] * np.exp(-X * a[None, :])
        return einf + np.sum(terms ** (1.0 / b), axis=1) ** b - e

    p0 = [0.05] + [0.5] * n + [0.3] * n + ([1.0] if beta_free else [])
    lo = [0.0] + [0.0] * n + [0.0] * n + ([0.15] if beta_free else [])
    hi = [0.6] + [1e3] * n + [5.0] * n + ([6.0] if beta_free else [])
    r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=400000)
    rss = float(np.sum(r.fun ** 2))
    aic, bic = _ic(rss, len(e), len(p0))
    return dict(theta=r.x, rss=rss, aic=aic, bic=bic,
                einf=r.x[0], c=r.x[1:1 + n], alpha=r.x[1 + n:1 + 2 * n],
                beta=(r.x[-1] if beta_free else 1.0))


def fit_multiplicative(X, e):
    n = X.shape[1]

    def resid(th):
        return th[0] + th[1] * np.exp(-X @ th[2:2 + n]) - e

    p0 = [0.05, 0.5] + [0.3] * n
    r = least_squares(resid, p0, bounds=([0, 0] + [0] * n, [0.6, 1e3] + [5.0] * n),
                      max_nfev=400000)
    rss = float(np.sum(r.fun ** 2))
    aic, bic = _ic(rss, len(e), len(p0))
    return dict(theta=r.x, rss=rss, aic=aic, bic=bic,
                einf=r.x[0], alpha=r.x[2:2 + n])


# ------------------------------------------------- normal form and rivals ---
def normal_form(th, X):
    """eps = einf * (1 + sum_i exp(-xi_i)),  xi_i = a_i (x_i - x0_i).  No shape param."""
    n = X.shape[1]
    einf, a, x0 = th[0], th[1:1 + n], th[1 + n:1 + 2 * n]
    return einf * (1.0 + np.sum(np.exp(-a[None, :] * (X - x0[None, :])), axis=1))


def normal_form_stretched(th, X):
    """Same, with one extra shape parameter gamma on the sum."""
    n = X.shape[1]
    einf, a, x0, g = th[0], th[1:1 + n], th[1 + n:1 + 2 * n], th[-1]
    return einf * (1.0 + np.sum(np.exp(-a[None, :] * (X - x0[None, :])), axis=1)) ** g


def logistic_form(th, X):
    """Same parameter count as the normal form, different saturation shape."""
    n = X.shape[1]
    einf, a, x0 = th[0], th[1:1 + n], th[1 + n:1 + 2 * n]
    z = np.sum(a[None, :] * (X - x0[None, :]), axis=1)
    return einf * (1.0 + 1.0 / (1.0 + np.exp(z))) * 2.0


def floorless_power(th, X):
    """eps = c * prod_i t_i^{-a_i}; the correct reduced form when einf = 0."""
    n = X.shape[1]
    return th[0] * np.exp(-X @ th[1:1 + n])


LAWS = {
    "normal form": (
        normal_form,
        lambda n: ([0.02] + [0.5] * n + [0.0] * n,
                   [1e-4] + [1e-3] * n + [-20.0] * n,
                   [0.6] + [6.0] * n + [20.0] * n)),
    "stretched": (
        normal_form_stretched,
        lambda n: ([0.02] + [0.5] * n + [0.0] * n + [1.0],
                   [1e-4] + [1e-3] * n + [-20.0] * n + [0.2],
                   [0.6] + [6.0] * n + [20.0] * n + [5.0])),
    "logistic": (
        logistic_form,
        lambda n: ([0.02] + [0.5] * n + [0.0] * n,
                   [1e-4] + [1e-3] * n + [-20.0] * n,
                   [0.6] + [6.0] * n + [20.0] * n)),
    "floorless": (
        floorless_power,
        lambda n: ([0.5] + [0.3] * n, [1e-6] + [0.0] * n, [1e4] + [5.0] * n)),
}


def fit_law(name, X, e):
    f, init = LAWS[name]
    p0, lo, hi = init(X.shape[1])
    r = least_squares(lambda t: f(t, X) - e, p0, bounds=(lo, hi), max_nfev=200000)
    return r.x


def save(df, name):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    df.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows)")
    return path

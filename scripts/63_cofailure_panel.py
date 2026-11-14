#!/usr/bin/env python3
"""BUILD THE CO-FAILURE PANEL  (Section "The co-failure law" of paper.tex).

The frontiers archived in this repository record *aggregate* accuracies only.
The co-failure law

    Pr[ every model in S fails an item ]  =  eps( sum_{i in S} x_i )        (CF)

is a statement about per-item correctness, so testing it needs a frontier whose
item-level outcomes are available.  This script builds one from scratch, with
real data and real trained models, so that the test is reproducible offline.

Design
------
Data      Fashion-MNIST (60k train / 10k test, 10 classes), fetched from the
          official repository.  Chosen over MNIST because it saturates at a
          clearly non-zero error, so the frontier has a visible floor.
Models    Random-feature ridge classifiers: x -> cos(W x + b) with W Gaussian,
          then one-hot ridge.  The resource is the feature count D, so the
          log-resource is x = ln D; the grid is 8 .. 8192 in sqrt(2) steps
          (21 models, a span of 6.93 nats, matching the panel median of 6.74).
          The ridge penalty is chosen per D on a 5k validation split, so each
          model is the best one at its resource -- a Pareto point by
          construction.
Families  The law assumes failures are independent ACROSS MODELS GIVEN THE ITEM.
          For deterministic predictors that has to be read as independence over
          the draw of the model at a given resource, so two families are built:
            indep  -- every width draws its own (W, b).  Independent draws.
            nested -- one (W, b) is drawn at D = 8192 and every smaller model
                      uses its leading D columns.  Maximally shared randomness.
          (CF) is predicted to hold for `indep` and to fail for `nested`; the
          pair is the control that makes the test interpretable rather than a
          bare goodness-of-fit.

Writes (data/cofailure/):
    fmnist_<family>_correct.npy     bool, 10000 x 21, per-item correctness
    fmnist_<family>_frontier.csv    D, x=ln D, acc, err, eps=(1-acc)/(1-q0)
"""
import os, sys, gzip, time, argparse
import numpy as np, pandas as pd
from scipy.linalg import solve as _solve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW  = os.path.join(ROOT, "data", "cofailure", "fashion_mnist_raw")
OUT  = os.path.join(ROOT, "data", "cofailure")
BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz",  "t10k-labels-idx1-ubyte.gz"]

Q0    = 0.10                     # chance level, 10 classes
GAMMA = 0.02                     # RBF bandwidth of the random features
LAMS  = [3e-4, 1e-3, 3e-3]       # ridge grid, chosen per D on a validation split
NVAL  = 5000
WIDTHS = [int(round(8 * 2 ** (k / 2))) for k in range(21)]   # 8 .. 8192


def fetch():
    os.makedirs(RAW, exist_ok=True)
    import urllib.request
    for f in FILES:
        p = os.path.join(RAW, f)
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            continue
        print(f"  downloading {f}")
        urllib.request.urlretrieve(f"{BASE}/{f}", p)


def load():
    def rd(img, lab):
        with gzip.open(os.path.join(RAW, img), "rb") as fh:
            X = np.frombuffer(fh.read(), np.uint8, offset=16).reshape(-1, 784)
        with gzip.open(os.path.join(RAW, lab), "rb") as fh:
            y = np.frombuffer(fh.read(), np.uint8, offset=8)
        return X.astype(np.float32) / 255.0, y.astype(np.int64)
    Xtr, ytr = rd(FILES[0], FILES[1])
    Xte, yte = rd(FILES[2], FILES[3])
    return Xtr, ytr, Xte, yte


def gram_chunked(X, Y, W, b, chunk=6000):
    """Accumulate F^T F and F^T Y without materialising F."""
    D = W.shape[1]
    G = np.zeros((D, D), np.float64)
    H = np.zeros((D, Y.shape[1]), np.float64)
    for s in range(0, len(X), chunk):
        F = np.cos(X[s:s + chunk] @ W + b)
        G += F.T @ F
        H += F.T @ Y[s:s + chunk]
    return G, H


def predict_chunked(X, W, b, C, chunk=6000):
    out = np.empty((len(X), C.shape[1]), np.float32)
    for s in range(0, len(X), chunk):
        out[s:s + chunk] = np.cos(X[s:s + chunk] @ W + b) @ C
    return out


def build(family, Xtr, ytr, Xte, yte, seed0=20261004):
    Y = np.eye(10, dtype=np.float32)[ytr]
    ntr = len(Xtr) - NVAL
    Xf, Yf = Xtr[:ntr], Y[:ntr]
    Xv, yv = Xtr[ntr:], ytr[ntr:]
    Dmax = max(WIDTHS)
    rng = np.random.default_rng(seed0)
    if family == "nested":
        Wfull = rng.normal(0, np.sqrt(2 * GAMMA), (784, Dmax)).astype(np.float32)
        bfull = rng.uniform(0, 2 * np.pi, Dmax).astype(np.float32)

    correct, rows = [], []
    for D in WIDTHS:
        t0 = time.time()
        if family == "nested":
            W, b = Wfull[:, :D], bfull[:D]
        else:
            r = np.random.default_rng(seed0 + 7919 * D)
            W = r.normal(0, np.sqrt(2 * GAMMA), (784, D)).astype(np.float32)
            b = r.uniform(0, 2 * np.pi, D).astype(np.float32)

        Gf, Hf = gram_chunked(Xf, Yf, W, b)
        Gv, Hv = gram_chunked(Xv, np.eye(10, dtype=np.float32)[yv], W, b)
        I = np.eye(D)
        best = None
        for lam in LAMS:
            C = _solve(Gf + lam * ntr * I, Hf, assume_a="pos")
            e = (predict_chunked(Xv, W, b, C.astype(np.float32)).argmax(1) != yv).mean()
            if best is None or e < best[0]:
                best = (e, lam)
        lam = best[1]
        C = _solve(Gf + Gv + lam * len(Xtr) * I, Hf + Hv, assume_a="pos").astype(np.float32)
        ok = predict_chunked(Xte, W, b, C).argmax(1) == yte
        acc = ok.mean()
        correct.append(ok)
        rows.append(dict(D=D, x=np.log(D), lam=lam, acc=acc, err=1 - acc,
                         eps=(1 - acc) / (1 - Q0)))
        print(f"  [{family}] D={D:>5} lam={lam:<7.0e} err={1-acc:.4f} "
              f"eps={(1-acc)/(1-Q0):.4f}  ({time.time()-t0:.1f}s)")
    Cm = np.asarray(correct).T                      # items x models
    np.save(os.path.join(OUT, f"fmnist_{family}_correct.npy"), Cm)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"fmnist_{family}_frontier.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="indep,nested")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("fetching Fashion-MNIST ...")
    fetch()
    Xtr, ytr, Xte, yte = load()
    print(f"  train {Xtr.shape}  test {Xte.shape}")
    for fam in a.families.split(","):
        print(f"building family '{fam}' ({len(WIDTHS)} models, "
              f"span {np.log(max(WIDTHS)/min(WIDTHS)):.2f} nats)")
        build(fam, Xtr, ytr, Xte, yte)
    print("done ->", OUT)

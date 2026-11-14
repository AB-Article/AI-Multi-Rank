#!/usr/bin/env python3
"""DOES THE COUNT AXIS BEAT THE OLD READING?  A held-out, higher-order test.
   (Section "The count axis" of paper.tex, Table "which reading".)

Script 64 scored three mechanisms on PAIRS.  Pairs are where the paper's own
nesting law was fitted, so pairs cannot separate a derivation from a curve fit.
This script moves the test to orders the fit never saw.

Setup.  For each model b and each set S of models with b the best member,
Pr[all of S fail] is observable.  The three readings predict:

  (A) old mechanism, modes failing independently given the item:
          Pr[all of S fail] = h( sum_{i in S} x_i )
      -- refuted on pairs by 64; carried here for completeness.
  (B) the paper's nesting law, extended the only way it can be:
          Pr[all of S fail] = kappa^{|S|-1} h_b,  kappa fitted on PAIRS.
  (D) the count axis: Pr[all of S fail] = int g^{|S|} dnu_b(g), with nu_b
      obtained by Hausdorff inversion of the chain P_b(1..M_TRAIN) ONLY.

(D) is fitted on orders 1..4 and tested on orders 5..12; (B) is fitted on order
2.  Both are then asked for the same held-out quantity.  (D) also makes a
structural prediction neither of the others can: Pr[all of S fail] depends on S
only through its best member and its size, because under nesting the hardest
member of S is the best model in it.  That is tested directly, as a
within-group vs between-group variance ratio over random subsets.

Reads  data/cofailure/fmnist_{indep,nested}_correct.npy  (from 63)
Writes (results/):
    table_count_axis_holdout.csv        per (family, reading, order) scores
    table_count_axis_holdout_summary.csv  the headline comparison
    table_count_axis_sufficiency.csv    the (best, size) sufficiency test
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "cofailure")
RESULTS = os.path.join(ROOT, "results")

M_TRAIN = 4          # orders available to the count-axis inversion
ORDERS = range(5, 13)  # held-out orders
N_SUBSETS = 60       # random subsets per (best model, order)
RNG = np.random.default_rng(20261005)
Q0 = 0.10
FAMILIES = ("indep", "nested")


def save(df, name):
    df.to_csv(os.path.join(RESULTS, name), index=False)
    print(f"  wrote {name}  ({len(df)} rows)")


def invert(P, ngrid=401):
    g = np.linspace(0.0, 1.0, ngrid)
    A = np.stack([g ** m for m in range(1, len(P) + 1)])
    nu, _ = nnls(A, P)
    return g, nu


def main():
    print("HELD-OUT COMPARISON OF THE THREE READINGS\n" + "=" * 68)
    rows, suff = [], []

    for fam in FAMILIES:
        C = np.load(os.path.join(DATA, f"fmnist_{fam}_correct.npy"))
        F = ~C
        n, M = F.shape
        h = F.mean(0)
        order = np.argsort(h)                       # best first
        fr = pd.read_csv(os.path.join(DATA, f"fmnist_{fam}_frontier.csv"))
        x = fr["x"].to_numpy()
        # the measured marginal curve h(x), for reading (A); log-linear interp
        xs, hs = x[np.argsort(x)], h[np.argsort(x)]

        def h_at(z):
            return np.interp(z, xs, np.log(hs), left=np.nan, right=np.nan)

        # --- kappa for reading (B): fitted on pairs only -------------------
        kaps = []
        for ii in range(M):
            for jj in range(ii + 1, M):
                b = ii if h[ii] < h[jj] else jj
                kaps.append((F[:, ii] & F[:, jj]).mean() / h[b])
        kappa = float(np.median(kaps))

        # --- nu_b for reading (D): fitted on the first M_TRAIN orders only -
        nus = {}
        for pos, b in enumerate(order):
            chain = order[pos:pos + M_TRAIN]
            if len(chain) < M_TRAIN:
                continue
            Pb = np.array([F[:, chain[:m]].all(1).mean() for m in range(1, M_TRAIN + 1)])
            nus[b] = invert(Pb)

        print(f"\n[{fam}]  kappa fitted on pairs = {kappa:.4f}; "
              f"nu fitted on orders 1..{M_TRAIN} for {len(nus)} anchors")

        for k in ORDERS:
            errA, errB, errD, obs_all = [], [], [], []
            for b in nus:
                pool = [i for i in range(M) if h[i] > h[b]]
                if len(pool) < k - 1:
                    continue
                g, nu = nus[b]
                for _ in range(N_SUBSETS):
                    S = np.r_[b, RNG.choice(pool, size=k - 1, replace=False)]
                    obs = F[:, S].all(1).mean()
                    if obs <= 0:
                        continue
                    obs_all.append(obs)
                    # (A)
                    la = h_at(x[S].sum())
                    if np.isfinite(la):
                        errA.append(abs(np.exp(la) / obs - 1))
                    # (B)
                    errB.append(abs(kappa ** (k - 1) * h[b] / obs - 1))
                    # (D)
                    predD = float((nu * g ** k).sum())
                    errD.append(abs(predD / obs - 1))
            rows.append(dict(family=fam, order=k, n_subsets=len(obs_all),
                             median_obs=np.median(obs_all),
                             errA=np.median(errA) if errA else np.nan,
                             errB=np.median(errB), errD=np.median(errD)))
            print(f"  order {k:2d}: n={len(obs_all):4d}  "
                  f"(A) {rows[-1]['errA'] if errA else float('nan'):>6.3f}   "
                  f"(B) {rows[-1]['errB']:.3f}   (D) {rows[-1]['errD']:.3f}")

        # --- sufficiency of (best member, size) ----------------------------
        for k in (3, 5, 8):
            recs = []
            for b in list(nus)[:14]:
                pool = [i for i in range(M) if h[i] > h[b]]
                if len(pool) < k - 1:
                    continue
                v = [F[:, np.r_[b, RNG.choice(pool, size=k - 1, replace=False)]].all(1).mean()
                     for _ in range(N_SUBSETS)]
                recs.append((b, np.mean(v), np.var(v)))
            if len(recs) < 3:
                continue
            _, mu, within = map(np.array, zip(*recs))
            ratio = within.mean() / mu.var()
            suff.append(dict(family=fam, order=k, n_groups=len(recs),
                             within_group_var=within.mean(),
                             between_group_var=mu.var(), ratio=ratio))
            print(f"  sufficiency at order {k}: within/between variance = {ratio:.4f} "
                  f"(predicted ~0: S matters only through its best member and size)")

    df = pd.DataFrame(rows)
    df["A_evaluable"] = np.isfinite(df.errA)
    save(df, "table_count_axis_holdout.csv")
    s = df.groupby("family")[["errA", "errB", "errD"]].median().reset_index()
    s["improvement_D_over_B"] = 1 - s.errD / s.errB
    # (A) cannot be evaluated above order 4 at all: sum_{i in S} x_i leaves the
    # measured window, so the old mechanism has no prediction to score.  That
    # is a limitation of the reading, not of the data.
    s["A_evaluable_at_these_orders"] = df.groupby("family")["A_evaluable"].any().to_numpy()
    print("\n" + "=" * 68)
    print(s.to_string(index=False))
    save(s, "table_count_axis_holdout_summary.csv")
    save(pd.DataFrame(suff), "table_count_axis_sufficiency.csv")


if __name__ == "__main__":
    main()

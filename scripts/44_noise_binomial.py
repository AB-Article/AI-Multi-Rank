#!/usr/bin/env python3
"""T7b  Which noise scale belongs in the resolution criterion.

The resolution number V = Z^2 sqrt(m) / (16 sigma) of Proposition
'resolution' needs a log-error noise scale.  A benchmark score is an average
over M evaluation items, so the binomial sampling error of the error count
contributes, with no free parameter,

    sigma_bin^2 = Var(ln eps_hat) ~ Q / (M (1 - Q)),

since Var(eps_hat) = e(1-e)/M with e = 1-Q and d ln e / d e = 1/e.  This
script asks whether sigma_bin can be used in place of sigma.  It cannot: it is
a lower bound, exceeded by the residual misfit of the fitted law by a factor
that is itself structured, being near one within a single architecture search
space and several-fold across a heterogeneous model zoo.

Writes results/table_noise_binomial.csv.
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from _loaders import selection_rules, save, RES

sr = selection_rules()

ITEMS = {"ImageNet-1k": 50000, "ImageNet-ReaL": 50000, "ImageNet-V2": 10000,
         "ImageNet-A": 7500, "ImageNet-R": 30000, "ImageNet-Sketch": 50889,
         "cifar10": 10000, "cifar100": 10000, "ImageNet16-120": 6000}


def items(name):
    for k, v in ITEMS.items():
        if k.lower() in str(name).lower():
            return v
    return np.nan


def chance(name):
    s = str(name).lower()
    if "imagenet16" in s:
        return 1.0 / 120
    if "cifar100" in s:
        return 0.01
    if "cifar10" in s:
        return 0.10
    return 0.001


def main():
    L = pd.read_csv(os.path.join(RES, "table_holdout_floor_per_frontier.csv"))
    emin = {(str(n), str(c)): float(np.min(e))
            for f, n, c, x, e in sr.single_resource_frontiers()}
    rows = []
    for _, r in L.iterrows():
        M = items(r.frontier)
        em = emin.get((str(r.frontier), str(r.cost)), np.nan)
        if not np.isfinite(M) or not np.isfinite(em) or not (r.law_rms > 0):
            continue
        Q = 1 - em * (1 - chance(r.frontier))
        if not (0.3 < Q < 0.999):
            continue
        sb = float(np.sqrt(Q / (M * (1 - Q))))
        rows.append(dict(field=r.field, frontier=r.frontier, cost=r.cost,
                         items=int(M), accuracy=Q, sigma_binomial=sb,
                         law_rms=float(r.law_rms), ratio=float(r.law_rms / sb)))
    D = pd.DataFrame(rows)
    save(D, "table_noise_binomial.csv")
    homog = D[D.frontier.str.contains("nats-sss|nb201", case=False, regex=True)]
    heter = D[D.frontier.str.contains("timm", case=False)]
    print(f"\nfrontiers with known evaluation-set sizes: {len(D)}")
    print(f"median sigma_binomial (parameter-free): {D.sigma_binomial.median():.4f} nats")
    print(f"median law_rms (fitted-law residual) : {D.law_rms.median():.4f} nats")
    print(f"median ratio law_rms / sigma_binomial: {D.ratio.median():.2f}x "
          f"(range {D.ratio.min():.1f}x to {D.ratio.max():.1f}x)")
    if len(homog):
        print(f"  within one search space (NATS/NB201): {homog.ratio.min():.1f}x to {homog.ratio.max():.1f}x")
    if len(heter):
        print(f"  across the timm zoo                 : {heter.ratio.min():.1f}x to {heter.ratio.max():.1f}x")
    print("\n=> sigma_binomial is a floor on sigma, not sigma; using it in V would")
    print("   overstate the resolvable channel count by roughly the median ratio.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""THE AXIOM IS A SYMMETRY: EXCHANGEABILITY, NOT INDEPENDENCE.
   (Section "The count axis", Prop. 'countaxis' and the remark 'definetti'.)

Proposition 'countaxis' was stated from conditional independence of attempts
given the item.  That hypothesis can be weakened to its symmetry core, because
P(m) = int g^m dnu is exactly de Finetti's representation of a binary
EXCHANGEABLE sequence:

    if the attempt sequence is exchangeable -- the order in which attempts are
    made is irrelevant, which is what "repeated attempt" means -- and
    infinitely extendible, then P(m) = int_0^1 g^m dnu(g), and conditional
    independence given a latent is a CONCLUSION rather than an assumption.

Three consequences are measured here.

  E1  Exchangeability is testable without any model: for an exchangeable
      sequence Pr[all of a subset fail] depends on the subset only through its
      SIZE.  We measure the between-subset variance of the all-fail rate at
      fixed size, as a fraction of the total variance over sizes.  (The
      panel's models are not statistically identical -- they differ in
      resource -- so the test is run both on the raw ladder and after the
      ladder is made homogeneous by conditioning on the best member, which is
      the paper's own structural prediction.)

  E2  Finite exchangeability is enough, with a known correction: by
      Diaconis-Freedman the total-variation distance between a finitely
      exchangeable sequence of length m and the closest mixture of i.i.d.
      sequences is at most 2k(k-1)/m at order k.  We compare the measured
      deviation of P(1..k) from its best Hausdorff mixture with that bound,
      which is where the running-floor bias and the resolution constant should
      eventually come from.

  E3  Exchangeability must be read on the POPULATION, not per item: a per-item
      failure curve is near-indicator, and an indicator sequence is not
      completely monotone.  We verify that the population curve passes the
      complete-monotonicity test that the per-item curves fail, which is the
      same statement as the refutation in Section "What the item level says",
      now with the right axis attached to it.

Outputs: results/table_exchangeability.csv
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _loaders import save

ROOT = os.path.dirname(HERE)
PANEL = os.path.join(ROOT, "data", "cofailure")
RNG = np.random.default_rng(20261005)
FAM = ("indep", "nested")


def load(fam):
    C = np.load(os.path.join(PANEL, f"fmnist_{fam}_correct.npy"))
    F = ~C                                   # failure indicators, items x models
    order = np.argsort(F.mean(0))            # best model first
    return F[:, order]


def allfail_curve(F, cols):
    """P(m) over the first m of the given column order."""
    return np.array([F[:, cols[:m]].all(1).mean() for m in range(1, len(cols) + 1)])


def hausdorff_residual(P, ngrid=200):
    """best non-negative mixture int g^m dnu matching P(1..k); rms residual."""
    k = len(P)
    g = np.linspace(0.0, 1.0, ngrid)
    A = np.vstack([g ** m for m in range(1, k + 1)])
    wt, _ = nnls(A, P)
    return float(np.sqrt(np.mean((A @ wt - P) ** 2))), wt, g


def e1_exchangeability(F, nsub=400):
    """between-subset variance of the all-fail rate at fixed subset size."""
    nm = F.shape[1]
    rows = []
    for size in (2, 3, 4, 6):
        vals = []
        for _ in range(nsub):
            cols = RNG.choice(nm, size=size, replace=False)
            vals.append(F[:, cols].all(1).mean())
        vals = np.asarray(vals)
        # homogeneous version: condition on the best member's own failure rate
        h = []
        for _ in range(nsub):
            cols = RNG.choice(nm, size=size, replace=False)
            b = cols[np.argmin(F[:, cols].mean(0))]
            denom = F[:, b].mean()
            h.append(F[:, cols].all(1).mean() / max(denom, 1e-9))
        h = np.asarray(h)
        rows.append(dict(size=size, raw_cv=float(vals.std() / max(vals.mean(), 1e-12)),
                         conditioned_cv=float(h.std() / max(h.mean(), 1e-12))))
    return pd.DataFrame(rows)


def run():
    rows = []
    for fam in FAM:
        F = load(fam)
        nm = F.shape[1]
        P = allfail_curve(F, np.arange(nm))
        se = np.sqrt(np.maximum(P * (1 - P), 1e-12) / F.shape[0])
        # E1
        X = e1_exchangeability(F)
        for _, r in X.iterrows():
            rows.append(dict(family=fam, test=f"E1 all-fail rate, size {int(r['size'])}: "
                             f"coeff. of variation across subsets (raw)", value=r.raw_cv))
            rows.append(dict(family=fam, test=f"E1 same, conditioned on the best member",
                             value=r.conditioned_cv))
        # E2  Hausdorff residual vs binomial s.e. and vs Diaconis-Freedman
        for k in (4, 8, 12, 21):
            res, wt, g = hausdorff_residual(P[:k])
            rows.append(dict(family=fam, test=f"E2 Hausdorff rms residual of P(1..{k})",
                             value=res))
            rows.append(dict(family=fam, test=f"E2 binomial s.e. at order {k}",
                             value=float(se[:k].mean())))
            rows.append(dict(family=fam,
                             test=f"E2 Diaconis-Freedman bound 2k(k-1)/m at order {k}, m={nm}",
                             value=float(2 * k * (k - 1) / nm)))
        # the nu returned by the full inversion
        res, wt, g = hausdorff_residual(P)
        sup = g[wt > 1e-6]
        rows += [dict(family=fam, test="E2 nu({g=1}) from the Hausdorff inversion",
                      value=float(wt[-1] / max(wt.sum(), 1e-12))),
                 dict(family=fam, test="E2 number of atoms with weight > 1e-6",
                      value=float(len(sup))),
                 dict(family=fam, test="E2 largest atom location g",
                      value=float(sup.max()) if len(sup) else np.nan)]
        # E3 population vs per item
        d3 = np.diff(P, 3)
        rows.append(dict(family=fam,
                         test="E3 population: min (-1)^3 Delta^3 P, in binomial s.e.",
                         value=float((-d3).min() / se[:len(d3)].mean())))
        # per item: is the failure ladder along the resource a single step?
        # an exponential link per item (the refuted axiom) predicts a geometric
        # ladder, which has many transitions; a sharp threshold has exactly one.
        keep = (F.mean(1) > 0.05) & (F.mean(1) < 0.95)
        Fi = F[keep].astype(int)
        trans = np.abs(np.diff(Fi, axis=1)).sum(1)
        one_step = float(np.mean(trans <= 1))
        # matched exponential-link control: same per-item failure rate, i.i.d. draws
        pr = Fi.mean(1, keepdims=True)
        sim = (RNG.random(Fi.shape) < pr).astype(int)
        one_step_sim = float(np.mean(np.abs(np.diff(sim, axis=1)).sum(1) <= 1))
        rows += [dict(family=fam, test="E3 per item: fraction with a single transition",
                      value=one_step),
                 dict(family=fam, test="E3 same under a matched exponential link (control)",
                      value=one_step_sim),
                 dict(family=fam, test="E3 items tested", value=float(len(Fi)))]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(__doc__.split("Outputs")[0])
    T = run()
    print(T.to_string(index=False))
    save(T, "table_exchangeability.csv")

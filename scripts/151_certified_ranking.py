#!/usr/bin/env python3
"""151 -- ranking every candidate, honestly.  (Fixes F18 and F19.)

"Rank all the models by effectiveness" has a dishonest answer and an honest
one.  The dishonest one is a total order on sigma: it invents distinctions the
data cannot support, and Theorem `covering` says how many -- 13.8 consecutive
models sit inside one indifference band on a median published frontier.  The
honest one is a sequence of TIERS: within a tier no difference is certifiable
at the item count in hand, between tiers every difference is.  That is the
partial order the certificates of F12 actually establish, laid out in layers.

One case needs no price at all (F18).  If a candidate is no dearer on every
axis and certified no worse on accuracy, it dominates at EVERY non-negative
price, so the comparison is settled with no grain, no sweep and no derived
price.  That is the first thing a selector should try, and this script measures
how often it fires on the archive.

Measured on every frontier with a public evaluation-set size, treating its
Pareto points as the candidate list:

  * how many certified tiers the ranking has, and how large the top tier is;
  * how many candidates are certifiably out of contention;
  * how often the price-free (Pareto) certificate alone settles a pair.

Writes results/table_certified_ranking.csv and
results/table_certified_ranking_summary.csv.
"""
import math

import numpy as np

import _pipeline as P

pc = P.pc
ALPHA = 0.10
RHO = 0.50
NLAM = 41


def main():
    rows = []
    for f in P.frontiers():
        xs, es = list(f["x"]), list(f["e"])
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in o]; es = [es[i] for i in o]
        ls = [math.log(e) for e in es]
        S = [-l for l in ls]
        n = len(xs)
        if n < 5:
            continue
        ni = P.eval_items(f["name"])
        if not ni:
            continue
        sl = [(S[i + 1] - S[i]) / (xs[i + 1] - xs[i]) for i in range(n - 1)
              if xs[i + 1] > xs[i] and S[i + 1] > S[i]]
        if not sl:
            continue
        lo, hi = min(sl), max(sl)
        if not (hi > lo > 0):
            continue
        plane = pc.read_plane(xs, ls, 1.0)
        lam = plane.get("price") or math.sqrt(lo * hi)
        sds = [pc.sd_of_l(min(max(1 - e, 1e-6), 1 - 1e-9), ni) for e in es]
        med, plo, phi_, psd = pc.price_uncertainty(xs, ls, sds, 1.0,
                                                   draws=200, seed=4)
        rng_ = ({"r": (plo, phi_)} if (med and phi_ > plo > 0) else None)
        models = [{"name": i, "eps": es[i], "x": {"r": xs[i]}}
                  for i in range(n)]
        rk = pc.certified_ranking(models, {"r": lam}, price_range=rng_,
                                  alpha=ALPHA, n_test=ni, rho=RHO)
        pa = pc.certified_pareto(models, alpha=ALPHA, n_test=ni, rho=RHO)
        gate = pc.class_gate(xs, ls)
        rows.append({
            "corpus": f["corpus"], "frontier": f["name"], "n_candidates": n,
            "n_test": ni, "class_verdict": gate["verdict"],
            "n_tiers": rk["n_tiers"],
            "top_tier_size": len(rk["tiers"][0]),
            "out_of_contention": n - len(rk["tiers"][0]),
            "share_out_of_contention": (n - len(rk["tiers"][0])) / float(n),
            "contradictions": len(rk["contradictions"]),
            "pricefree_dominations": len(pa["dominations"]),
            "pricefree_dominated": len(pa["dominated"]),
            "share_pricefree_dominated": len(pa["dominated"]) / float(n),
            "top_tier_worst_regret": max(rk["worst_case_regret"][nm]
                                         for nm in rk["tiers"][0]),
        })
    df = P.save(rows, "table_certified_ranking.csv")

    summ = []
    for tag, d in (("all", df),
                   ("in class", df[df.class_verdict == "IN_CLASS"]),
                   ("out of class", df[df.class_verdict == "OUT_OF_CLASS"])):
        if not len(d):
            continue
        summ.append({"subset": tag, "n_frontiers": len(d),
                     "median_candidates": float(d.n_candidates.median()),
                     "median_tiers": float(d.n_tiers.median()),
                     "median_top_tier": float(d.top_tier_size.median()),
                     "median_share_out_of_contention": float(
                         d.share_out_of_contention.median()),
                     "median_share_pricefree_dominated": float(
                         d.share_pricefree_dominated.median()),
                     "frontiers_with_contradictions": int(
                         (d.contradictions > 0).sum())})
    sdf = P.save(summ, "table_certified_ranking_summary.csv")
    print()
    print(f"  the certified ranking on {len(df)} frontiers with a public item"
          f" count, alpha = {ALPHA}:")
    print(sdf.to_string(index=False))
    print()
    d = df
    print(f"  median candidate list {d.n_candidates.median():.0f};"
          f" median {d.n_tiers.median():.0f} certified tiers;"
          f" top tier {d.top_tier_size.median():.0f} candidates,"
          f" so {d.share_out_of_contention.median():.0%} of a published"
          f" frontier is certifiably out of contention.")
    print(f"  the PRICE-FREE certificate alone retires"
          f" {d.share_pricefree_dominated.median():.0%} of candidates"
          f" (median), with no price, no grain and no sweep:"
          f" {int((d.pricefree_dominations>0).sum())} of {len(d)} frontiers"
          f" have at least one such domination.")
    print(f"    (it fires on only {int((d.pricefree_dominations>0).sum())} of"
          f" {len(d)} here because an ARCHIVED FRONTIER is already"
          f" Pareto-filtered:")
    print(f"     dominated points were removed before publication. On a raw"
          f" candidate list --")
    print(f"     a competition leaderboard, a sweep of checkpoints -- it is the"
          f" first test to run,")
    print(f"     and the worked example shows it retiring a submission with no"
          f" price at all.)")
    print(f"  contradictory pairs (both directions certified, each an error of"
          f" probability <= alpha/(K-1)):"
          f" {int((d.contradictions>0).sum())} of {len(d)} frontiers.")
    print()
    print("  A tier is the honest unit of a ranking: it is what the data can")
    print("  separate.  Ordering inside a tier is presentation, not evidence.")


if __name__ == "__main__":
    main()

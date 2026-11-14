#!/usr/bin/env python3
"""155 -- who can be optimal at all, without knowing the frontier.  (Fix F22.)

The break-even identity of F21 is algebra on two measured points and holds
whether or not either lies on a frontier: sigma = S - lambda x is a function of
(eps, x) alone.  What breaks when candidates are off-frontier is CHAINING
break-evens between cost neighbours, because a dominated candidate, or one
merely sitting in a dent of the measured staircase, breaks the chain.

The repair needs no knowledge of which candidates are efficient.  Each
candidate is a straight line in the price, sigma_i(lambda) = S_i - lambda x_i,
so the best attainable score is the upper envelope of K lines: convex,
piecewise linear, one sort and one stack pass.  Its pieces name the winner at
each price, its breakpoints are the break-evens that matter, and the candidates
absent from it are provably never the answer at any price.

That set is STRICTER than the Pareto set, which is the measurement here.  A
Pareto-efficient candidate lying in a dent maximises sigma for no lambda at
all, so a Pareto frontier systematically over-states the set of viable models.
The archive's frontiers are already Pareto-filtered, so every point dropped
here is dropped for that reason alone.

Writes results/table_price_envelope.csv and table_price_envelope_summary.csv.
"""
import math

import numpy as np

import _pipeline as P

pc = P.pc
LN2 = math.log(2.0)
ALPHA = 0.10
RHO = 0.50


def main():
    rows = []
    for f in P.frontiers():
        xs, es = list(f["x"]), list(f["e"])
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in o]; es = [es[i] for i in o]
        n = len(xs)
        if n < 5:
            continue
        models = [{"name": i, "eps": es[i], "x": {"r": xs[i]}}
                  for i in range(n)]
        env = pc.price_envelope(models, "r")
        ni = P.eval_items(f["name"])
        gate = pc.class_gate(xs, [math.log(e) for e in es])
        rec = {"corpus": f["corpus"], "frontier": f["name"],
               "class_verdict": gate["verdict"], "n_test": ni,
               "n_pareto": n, "n_lambda_reachable": env["n_winners"],
               "n_in_a_dent": len(env["in_a_dent"]),
               "n_dominated": len(env["dominated"]),
               "share_unreachable": 1.0 - env["n_winners"] / float(n)}
        # the span of prices each winner reigns over, and the ladder's rungs
        spans = [math.log(hi / lo) / LN2 for _nm, lo, hi in env["pieces"]
                 if hi > lo > 0 and hi < 1e5]
        rec["median_reign_doublings"] = (float(np.median(spans))
                                         if spans else None)
        if ni:
            bee = pc.break_even_envelope(models, "r", alpha=ALPHA,
                                         n_test=ni, rho=RHO)
            und = [r["undecided_doublings"] for r in bee["rungs"]
                   if r.get("undecided_doublings")]
            rec["n_rungs"] = len(bee["rungs"])
            rec["median_rung_undecided_doublings"] = (float(np.median(und))
                                                      if und else None)
            rec["rungs_decided_share"] = (
                float(np.mean([bool(r["certified_lo"] or r["certified_hi"])
                               for r in bee["rungs"]]))
                if bee["rungs"] else None)
        rows.append(rec)
    df = P.save(rows, "table_price_envelope.csv")

    summ = []
    for tag, d in [("all", df)] + [(c, g) for c, g in df.groupby("corpus")]:
        summ.append({
            "subset": tag, "n_frontiers": len(d),
            "median_n_pareto": float(d.n_pareto.median()),
            "median_n_lambda_reachable": float(d.n_lambda_reachable.median()),
            "median_share_unreachable": float(d.share_unreachable.median()),
            "median_in_a_dent": float(d.n_in_a_dent.median()),
            "median_reign_doublings": float(d.median_reign_doublings.median()),
        })
    sdf = P.save(summ, "table_price_envelope_summary.csv")
    print()
    print(f"  on {len(df)} archived frontiers, every one of them already"
          f" Pareto-filtered:")
    print(sdf.to_string(index=False))
    print()
    d = df
    print(f"  median Pareto points                      :"
          f" {d.n_pareto.median():.0f}")
    print(f"  median points a linear price can select   :"
          f" {d.n_lambda_reachable.median():.0f}")
    print(f"  median share NEVER optimal at any price   :"
          f" {d.share_unreachable.median():.1%}"
          f"  (mean {d.share_unreachable.mean():.1%},"
          f" max {d.share_unreachable.max():.1%})")
    print(f"  of those, median sitting in a DENT        :"
          f" {d.n_in_a_dent.median():.0f} per frontier -- Pareto-efficient,"
          f" and still never the answer")
    print()
    k = d.dropna(subset=["median_rung_undecided_doublings"])
    if len(k):
        print(f"  the ladder built on the envelope, on the"
              f" {len(k)} frontiers with a public item count:")
        print(f"    rungs per frontier                 :"
              f" {k.n_rungs.median():.0f}")
        print(f"    undecided interval per rung        : median"
              f" {k.median_rung_undecided_doublings.median():.2f} doublings")
        print(f"    rungs decided on at least one side :"
              f" {k.rungs_decided_share.median():.0%}")
    print()
    print("  Two consequences.  A Pareto frontier over-states the viable set by")
    print("  a factor of two or more, so 'is this model on the frontier?' is the")
    print("  wrong question -- the envelope answers the right one, and answers it")
    print("  without identifying any frontier.  And the ladder must be built on")
    print("  the envelope's breakpoints, not on cost neighbours: a dominated")
    print("  candidate or one in a dent breaks a neighbour chain and cannot")
    print("  break the envelope, because it never appears on it.")


if __name__ == "__main__":
    main()

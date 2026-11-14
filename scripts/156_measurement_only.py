#!/usr/bin/env python3
"""156 -- what the candidates' own measurements decide, and nothing else.
(Fix F23.)

Three of the selector's inputs are not inputs of the decision, and the tool
should never have asked for them.

  q0, the chance level.  eps = (1-Q)/(1-q0), so ln eps = ln(1-Q) - ln(1-q0):
      a constant shift of l.  It cancels identically from every difference of
      sigma, from every break-even and from the whole envelope.  Part A checks
      this on every archived frontier, exactly.
  the grain t.  It enters only through lambda* = sqrt(kappa/t), the operating
      point and the wall and depth bounds.  Nothing in the envelope, the ladder
      or the price-free test refers to it.
  the price itself.  The envelope is max_i (S_i - lambda x_i) read as a
      FUNCTION of lambda, so it is computed without knowing lambda; the rungs
      are where that function changes winner; the price-free test is the rung
      at lambda = 0.

So the minimal invocation is

    select_model.py candidates.csv --items items.csv

and Part B measures what it returns on all 117 archived frontiers: how complete
the answer is, how many candidates it retires, and how sharp its rungs are.

Writes results/table_measurement_only.csv and
table_measurement_only_summary.csv.
"""
import math

import numpy as np

import _pipeline as P

pc = P.pc
LN2 = math.log(2.0)
ALPHA = 0.10
RHO = 0.50
Q0S = (0.0, 0.001, 0.01, 0.1, 0.5)


def main():
    # ---------------- A: q0 cancels, exactly
    worst_sigma, worst_be, worst_env = 0.0, 0.0, 0
    for f in P.frontiers():
        xs, es = list(f["x"]), list(f["e"])
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in o]; es = [es[i] for i in o]
        if len(xs) < 5:
            continue
        ref_sig, ref_be, ref_win = None, None, None
        for q0 in Q0S:
            # the archived eps is already chance-corrected at the corpus's own
            # q0; re-correcting by a further q0 is exactly the shift being
            # tested, which is what a user choosing a different convention does
            eps = [e / (1.0 - q0) for e in es]
            S = [-math.log(v) for v in eps]
            lam = 0.5
            sig = [S[i] - lam * xs[i] for i in range(len(xs))]
            dif = [sig[i] - sig[0] for i in range(len(xs))]
            models = [{"name": i, "eps": eps[i], "x": {"r": xs[i]}}
                      for i in range(len(xs))]
            env = pc.price_envelope(models, "r")
            be = [b for b in env["breakpoints"]]
            if ref_sig is None:
                ref_sig, ref_be, ref_win = dif, be, env["winners"]
            else:
                worst_sigma = max(worst_sigma,
                                  max(abs(a - b) for a, b in zip(dif, ref_sig)))
                if len(be) == len(ref_be) and be:
                    worst_be = max(worst_be,
                                   max(abs(a - b) for a, b in zip(be, ref_be)))
                worst_env += (0 if env["winners"] == ref_win else 1)
    print()
    print(f"  A. q0 invariance, over q0 in {Q0S}, on every frontier:")
    print(f"     largest deviation in any sigma DIFFERENCE : {worst_sigma:.3e}")
    print(f"     largest deviation in any break-even price : {worst_be:.3e}")
    print(f"     envelope winner sets that changed         : {worst_env}")
    print(f"     -> q0 is not an input of the decision. It moves the floor and"
          f" the depth readings only.")

    # ---------------- B: the measurement-only decision, on the whole corpus
    rows = []
    for f in P.frontiers():
        xs, es = list(f["x"]), list(f["e"])
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in o]; es = [es[i] for i in o]
        n = len(xs)
        if n < 5:
            continue
        ni = P.eval_items(f["name"])
        models = [{"name": i, "eps": es[i], "x": {"r": xs[i]}}
                  for i in range(n)]
        res = pc.decide_from_measurements(models, "r", alpha=ALPHA,
                                          n_test=ni, rho=RHO)
        env = res["envelope"]
        und = [g["undecided_doublings"] for g in res["rungs"]
               if g.get("undecided_doublings")]
        dec = [bool(g["certified_lo"] or g["certified_hi"])
               for g in res["rungs"]]
        rows.append({
            "corpus": f["corpus"], "frontier": f["name"], "n_test": ni,
            "n_candidates": n, "n_winners": env["n_winners"],
            "n_never_optimal": len(env["never"]),
            "share_retired": len(env["never"]) / float(n),
            "n_rungs": len(res["rungs"]),
            "n_tiers": len(res["tiers"]),
            "median_undecided_doublings": (float(np.median(und)) if und
                                           else None),
            "rungs_decided_share": (float(np.mean(dec)) if dec else None),
            "pricefree_dominations": len(res["price_free"]["dominations"]),
        })
    df = P.save(rows, "table_measurement_only.csv")
    k = df.dropna(subset=["median_undecided_doublings"])
    summ = [{
        "n_frontiers": len(df),
        "frontiers_with_items": len(k),
        "median_candidates": float(df.n_candidates.median()),
        "median_winners": float(df.n_winners.median()),
        "median_share_retired": float(df.share_retired.median()),
        "median_rungs": float(df.n_rungs.median()),
        "median_undecided_doublings": float(
            k.median_undecided_doublings.median()),
        "median_rungs_decided_share": float(k.rungs_decided_share.median()),
        "q0_max_sigma_deviation": worst_sigma,
        "q0_max_breakeven_deviation": worst_be,
        "q0_envelope_changes": worst_env,
    }]
    sdf = P.save(summ, "table_measurement_only_summary.csv")
    r = summ[0]
    print()
    print(f"  B. the measurement-only decision on {len(df)} frontiers"
          f" (accuracy, resource, n_test; no price, grain or q0):")
    print(f"     median candidates                 :"
          f" {r['median_candidates']:.0f}")
    print(f"     median that can ever be optimal   :"
          f" {r['median_winners']:.0f}"
          f"   -> {r['median_share_retired']:.0%} retired outright")
    print(f"     median rungs in the ladder        :"
          f" {r['median_rungs']:.0f}  (they tile the whole price axis, so the"
          f" answer is complete)")
    print(f"     median undecided interval per rung:"
          f" {r['median_undecided_doublings']:.2f} doublings")
    print(f"     rungs decided on at least one side:"
          f" {r['median_rungs_decided_share']:.0%}")
    print()
    print("  The answer is complete without a price: the rungs partition the")
    print("  positive price axis, so whatever the deployment's price turns out")
    print("  to be, exactly one candidate is named -- and the candidates that")
    print("  are named by no price at all are retired with a proof.")


if __name__ == "__main__":
    main()

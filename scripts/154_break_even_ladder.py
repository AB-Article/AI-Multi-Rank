#!/usr/bin/env python3
"""154 -- the break-even ladder: the simplest statement the theory supports.
(Fix F21.)

Two candidates, one evaluation each, and the per-item outcomes behind them.
Then

    sigma_D > sigma_C  <=>  lambda < lambda_be := ln(eps_C/eps_D)/(x_D - x_C),

so a single BREAK-EVEN price decides, and certifying it is F12 run as a
function of lambda: the certificate for "the dearer wins" falls monotonically
in lambda and the one for "the cheaper wins" rises, so a bisection returns an
interval outside which one candidate is certified better and inside which the
items in hand cannot separate them.  No sweep.  No grain.  No derived price.
No Law I.

The question this script answers is whether that is enough in practice: how
WIDE is the undecided interval, measured against the width of the derived
price's own interval -- the thing the sweep is for.  If the ladder's rungs are
as sharp as the derived price, then for a deployment that can name any price at
all the sweep is optional, and the theory's contribution is to supply a price
when nobody will name one.

Writes results/table_break_even.csv and table_break_even_summary.csv.
"""
import math

import numpy as np

import _pipeline as P

pc = P.pc
ALPHA = 0.10
RHO = 0.50
LN2 = math.log(2.0)


def main():
    rows = []
    for f in P.frontiers():
        xs, es = list(f["x"]), list(f["e"])
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in o]; es = [es[i] for i in o]
        ls = [math.log(e) for e in es]
        n = len(xs)
        if n < 5:
            continue
        ni = P.eval_items(f["name"])
        if not ni:
            continue
        # the derived price and its own interval, for the comparison
        sds = [pc.sd_of_l(min(max(1 - e, 1e-6), 1 - 1e-9), ni) for e in es]
        plane = pc.read_plane(xs, ls, 1.0)
        lam = plane.get("price")
        med, plo, phi_, _psd = pc.price_uncertainty(xs, ls, sds, 1.0,
                                                    draws=200, seed=6)
        price_width = (math.log(phi_ / plo) / LN2
                       if (med and phi_ > plo > 0) else None)

        models = [{"name": i, "eps": es[i], "x": {"r": xs[i]}}
                  for i in range(n)]
        chain = pc.break_even_chain(models, "r", alpha=ALPHA, n_test=ni,
                                    rho=RHO)
        for c in chain:
            if c["delta_log_resource"] <= 0:
                continue
            rows.append({
                "corpus": f["corpus"], "frontier": f["name"],
                "n_test": ni, "cheap": c["cheap"], "dear": c["dear"],
                "delta_log_resource": c["delta_log_resource"],
                "break_even": c["break_even"],
                "certified_lo": c["certified_lo"],
                "certified_hi": c["certified_hi"],
                "undecided_doublings": c["undecided_doublings"],
                "decided_somewhere": bool(c["certified_lo"]
                                          or c["certified_hi"]),
                "lambda_hat": lam,
                "price_interval_doublings": price_width,
                "break_even_above_lambda_hat": (
                    bool(c["break_even"] > lam) if lam else None),
            })
    df = P.save(rows, "table_break_even.csv")
    d = df.dropna(subset=["undecided_doublings"])
    summ = [{
        "n_adjacent_pairs": len(df),
        "pairs_with_an_interval": len(d),
        "median_undecided_doublings": float(d.undecided_doublings.median()),
        "p90_undecided_doublings": float(
            np.percentile(d.undecided_doublings, 90)),
        "median_price_interval_doublings": float(
            df.price_interval_doublings.median()),
        "ladder_sharper_than_price_share": float(
            (d.undecided_doublings < d.price_interval_doublings).mean()),
        "median_break_even": float(df.break_even.median()),
        "median_delta_log_resource": float(df.delta_log_resource.median()),
    }]
    sdf = P.save(summ, "table_break_even_summary.csv")
    r = summ[0]
    print()
    print(f"  {r['n_adjacent_pairs']:,} adjacent Pareto pairs on the frontiers"
          f" with a public item count.")
    print(f"    undecided interval, per pair      : median"
          f" {r['median_undecided_doublings']:.2f} doublings"
          f" (90th percentile {r['p90_undecided_doublings']:.2f})")
    print(f"    the DERIVED price's own interval  : median"
          f" {r['median_price_interval_doublings']:.2f} doublings")
    print(f"    the ladder is sharper than the derived price on"
          f" {r['ladder_sharper_than_price_share']:.0%} of pairs")
    print()
    print(f"    median break-even price"
          f" {r['median_break_even']:.3f} nats/e-fold, across a median"
          f" resource step of {r['median_delta_log_resource']:.3f} in log")
    print()
    print("  So the smallest statement is also the sharpest available one. For a")
    print("  deployment that can say ANYTHING about its own price -- even just"
          " 'below 0.5")
    print("  nats per e-fold' -- the ladder decides, from one evaluation per"
          " candidate and")
    print("  the per-item outcomes, with no sweep at all. The sweep earns its"
          " keep only when")
    print("  nobody will name a price: then lambda* = sqrt(kappa/t) supplies"
          " one, and F20 says")
    print("  how wide the sweep must be to make it sharper than these rungs.")


if __name__ == "__main__":
    main()

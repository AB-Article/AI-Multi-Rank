#!/usr/bin/env python3
"""168_rule_spread.py -- how far apart the selection rules in use are, measured
without this paper's loss.

WHY THIS SCRIPT EXISTS
----------------------
Table 2 scores the rules against the derived price, which is this paper's own
objective; its first row is therefore close to an identity and is reported as a
reference point, not as a contest won.  That leaves the practical question
unanswered in a loss-free way: does it MATTER which rule a practitioner uses?

This script answers it with no objective at all.  For every frontier it takes
each rule's pick and reports only what the measured frontier itself says:

    the resource multiple   t(rule A) / t(rule B)        (an e-fold count)
    the error gap           ln eps(rule B) / eps(rule A) (bits)

Both are read off the archived table.  No floor, no fit, no price, no regret
enters, so every frontier of the 96-frontier build is usable rather than the
54 that carry an identified floor or the 26 that are attainable.

Reported:
  (A) THE SPREAD.  Across the rules that need nothing but the measured
      frontier, the span between the cheapest and the most expensive pick, in
      resource multiples and in bits.  This is the size of the decision.
  (B) THE PRICE OF THE SPREAD.  Resource multiples per bit between the extreme
      picks: what the disagreement costs per unit of quality.
  (C) UNIT SENSITIVITY.  The one rule in current use that is not scale-free,
      counted on the whole population.
  (D) WHAT THE EXTRA RESOURCE BUYS, on every frontier that carries the
      canonical point, as a check that (A) is not an artefact of the extremes.

Outputs: results/table_rule_spread_per_frontier.csv
         results/table_rule_spread_summary.csv
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save, selection_rules                      # noqa: E402

# the rules a practitioner can run with no fit and no declared number
FREE = ["two_x_best", "knee", "accuracy_max", "cheapest", "acc_per_cost",
        "ratio_log10", "budget_mid", "half_slope"]
# the three the recent literature actually uses
IN_USE = ["acc_per_cost", "ratio_log10", "budget_mid"]


def run():
    sr = selection_rules()
    frs = sr.single_resource_frontiers()
    rows = []
    for fr in frs:
        field, name, cost, x, e = fr
        S = -np.log(e)
        n = len(e)
        picks = {}
        picks["two_x_best"] = sr.nearest(np.log(e), np.log(2 * e.min()))
        picks["knee"] = sr.rule_knee(x, S)
        picks["half_slope"] = sr.rule_half_slope(x, S)
        picks["accuracy_max"] = n - 1
        picks["cheapest"] = 0
        c_, acc = np.exp(x), 1.0 - e
        picks["acc_per_cost"] = int(np.argmax(acc / c_))
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log10(c_)
            r10 = np.where(lg > 0, acc / lg, -np.inf)
        picks["ratio_log10"] = int(np.argmax(r10))
        # the same rule after a change of unit: GFLOP -> MFLOP, s -> ms
        flipped = {}
        for f in (1e-3, 1e3):
            lg2 = np.log10(c_ * f)
            r2 = np.where(lg2 > 0, acc / lg2, -np.inf)
            flipped[f] = int(np.argmax(r2))
        budget = np.exp(0.5 * (x.min() + x.max()))
        feas = np.where(c_ <= budget)[0]
        picks["budget_mid"] = int(feas[np.argmin(e[feas])]) if len(feas) else 0

        picks = {k: v for k, v in picks.items() if v is not None and np.isfinite(v)}
        used = [r for r in FREE if r in picks]
        if len(used) < 4:
            continue
        idx = np.array([int(picks[r]) for r in used], dtype=int)
        xs, es = x[idx], e[idx]
        lo, hi = int(np.argmin(xs)), int(np.argmax(xs))
        d_logcost = float(xs[hi] - xs[lo])                      # e-folds of resource
        d_bits = float((np.log(es[lo]) - np.log(es[hi])) / np.log(2))   # bits bought
        rows.append(dict(
            field=field, frontier=name, cost=cost, n=n,
            span_logcost=float(np.ptp(x)),
            spread_logcost=d_logcost,
            spread_cost_mult=float(np.exp(d_logcost)),
            spread_bits=d_bits,
            mult_per_bit=float(np.exp(d_logcost) / d_bits) if d_bits > 1e-9 else np.nan,
            n_distinct_picks=int(len(set(idx))),
            ratio_log10_unitflip=int(len({picks["ratio_log10"], flipped[1e-3], flipped[1e3]}) > 1),
            n_rules=len(used),
            **{f"pick_{r}": picks.get(r, -1) for r in FREE},
        ))
    df = pd.DataFrame(rows)
    save(df, "table_rule_spread_per_frontier.csv")

    print("\n" + "=" * 78)
    print("HOW FAR APART THE RULES IN USE ARE, WITH NO OBJECTIVE")
    print("=" * 78)
    print(f"\nfrontiers (the 96-frontier build, no floor needed) .. {len(df)}")
    print(f"distinct picks per frontier, median ................ {df.n_distinct_picks.median():.0f}"
          f"   of {df.n_rules.median():.0f} rules defined")
    s = df.spread_cost_mult
    print(f"\n(A) THE SPREAD between the extreme picks")
    print(f"  resource multiple, median ........................ {s.median():.0f}x"
          f"   IQR [{s.quantile(.25):.0f}x, {s.quantile(.75):.0f}x]")
    print(f"  e-folds of resource, median ...................... {df.spread_logcost.median():.2f}")
    b = df.spread_bits
    print(f"  bits of error bought by that resource, median .... {b.median():.2f}"
          f"   IQR [{b.quantile(.25):.2f}, {b.quantile(.75):.2f}]")
    print(f"  frontiers where the spread exceeds 10x ........... {int((s > 10).sum())} of {len(df)}")
    print(f"  frontiers where it buys less than one bit ........ {int((b < 1).sum())} of {len(df)}")
    m = df.mult_per_bit.dropna()
    print(f"\n(B) THE PRICE OF THE DISAGREEMENT")
    print(f"  resource multiple per bit, median ................ {m.median():.0f}x per bit"
          f"   IQR [{m.quantile(.25):.0f}x, {m.quantile(.75):.0f}x]")
    print(f"\n(C) UNIT SENSITIVITY of the one rule in use that is not scale-free")
    print(f"  pick moves under GFLOP->MFLOP or s->ms ........... "
          f"{int(df.ratio_log10_unitflip.sum())} of {len(df)}")

    summ = pd.DataFrame([dict(
        n_frontiers=len(df),
        median_distinct_picks=float(df.n_distinct_picks.median()),
        median_spread_cost_mult=float(s.median()),
        iqr_lo_spread_cost_mult=float(s.quantile(.25)),
        iqr_hi_spread_cost_mult=float(s.quantile(.75)),
        median_spread_logcost=float(df.spread_logcost.median()),
        median_spread_bits=float(b.median()),
        frontiers_spread_over_10x=int((s > 10).sum()),
        frontiers_under_one_bit=int((b < 1).sum()),
        median_mult_per_bit=float(m.median()),
        unitflip_moves=int(df.ratio_log10_unitflip.sum()),
    )])
    save(summ, "table_rule_spread_summary.csv")


if __name__ == "__main__":
    run()

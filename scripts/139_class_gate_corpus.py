#!/usr/bin/env python3
"""139 -- the class gate on the whole corpus.  (Fix F2.)

Law I is the first thing to fail and takes everything with it, so a production
selector must test it BEFORE it prices anything and refuse to name a winner
when it fails.  This script runs exactly that gate -- convexity of ln eps on
every wide triple, one-sided, exact at any spacing -- on all 117 archived
single-resource frontiers, and reports the verdict by corpus.

The paper's claim this checks: the class announces its own breakdown, and on
the panel the breakdown is the distribution-shift benchmarks.

Writes results/table_class_gate.csv and results/table_class_gate_summary.csv.
No network, no fit, no bandwidth.
"""
import math
import _pipeline as P

pc = P.pc


def main():
    rows = []
    for f in P.frontiers():
        xs = list(f["x"])
        ls = [math.log(e) for e in f["e"]]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ls = [ls[i] for i in order]
        n_items = P.eval_items(f["name"])

        # the gate as the selector runs it: noise-aware when n_test is known
        sd = None
        if n_items:
            # sd(ln eps) at the median error of this frontier
            eps_mid = sorted(f["e"])[len(f["e"]) // 2]
            q = 1.0 - eps_mid
            sd = pc.sd_of_l(q, n_items)
        gate = pc.class_gate(xs, ls, sd=sd)
        rows.append({
            "corpus": f["corpus"], "frontier": f["name"],
            "n_points": gate["n_points"], "n_wide_triples": gate["n_wide_triples"],
            "min_defect": gate["min_defect"],
            "min_defect_z": gate["min_defect_z"],
            "widest_defect": gate["widest_defect"],
            "share_triples_violating": gate["share_triples_violating"],
            "n_violations": gate["n_violations"],
            "curvature_rises": gate["curvature_rises"],
            "divisibility": gate["divisibility"],
            "n_test": n_items, "sd_l": sd,
            "verdict": gate["verdict"],
        })
    df = P.save(rows, "table_class_gate.csv")

    g = (df.groupby("corpus")
           .apply(lambda d: {
               "n_frontiers": len(d),
               "out_of_class": int((d.verdict == "OUT_OF_CLASS").sum()),
               "envelope": int((d.verdict == "ENVELOPE").sum()),
               "in_class": int((d.verdict == "IN_CLASS").sum()),
               "share_out": float((d.verdict == "OUT_OF_CLASS").mean()),
               "median_min_defect": float(d.min_defect.median()),
               "median_widest_defect": float(d.widest_defect.median()),
               "widest_defect_negative": int((d.widest_defect < 0).sum()),
               "median_share_violating": float(d.share_triples_violating.median()),
               "divisibility_refuted": int((d.divisibility == "REFUTED").sum()),
           }, include_groups=False))
    summ = [dict(corpus=k, **v) for k, v in g.items()]
    sdf = P.save(summ, "table_class_gate_summary.csv")

    print()
    print("  class gate by corpus (OUT_OF_CLASS = the selector refuses a winner)")
    print(sdf.to_string(index=False))
    shift = df[df.corpus == "vision-shift"]
    rest = df[df.corpus != "vision-shift"]
    print()
    print(f"  distribution shift : {int((shift.verdict=='OUT_OF_CLASS').sum())}"
          f" of {len(shift)} out of class")
    print(f"  everything else    : {int((rest.verdict=='OUT_OF_CLASS').sum())}"
          f" of {len(rest)} out of class")
    print(f"  corpus-wide        : "
          f"{int((df.verdict=='OUT_OF_CLASS').sum())} of {len(df)}"
          f" ({(df.verdict=='OUT_OF_CLASS').mean():.1%}) refused outright, "
          f"{int((df.verdict=='ENVELOPE').sum())} flagged as envelopes")
    print()
    print("  the discriminating statistic is the SHARE of wide triples that")
    print("  violate convexity, not the minimum over them:")
    print(f"    distribution shift : median share "
          f"{shift.share_triples_violating.median():.3f}")
    print(f"    everything else    : median share "
          f"{rest.share_triples_violating.median():.3f}")


if __name__ == "__main__":
    main()

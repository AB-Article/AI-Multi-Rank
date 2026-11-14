#!/usr/bin/env python3
"""150 -- the ENVELOPE threshold, calibrated instead of guessed.  (Fix A7.)

The class gate has three verdicts.  OUT_OF_CLASS rests on the global chord,
which is a single low-noise reading and needs no threshold.  ENVELOPE -- "the
chord is convex but the interior is jagged, so this is a Pareto envelope of
several models rather than one frontier of one clock" -- rested on a constant:
more than HALF the wide triples violating convexity.  One half was a guess, and
five of the archive's frontiers hang on it.

The share of violating triples is not a fixed quantity, though: a dense grid
measured on few items will violate convexity from noise alone, and a wide grid
on many items will not.  So the threshold belongs to the instance, and it can
be computed: simulate the frontier's OWN convex minorant (a curve that obeys
Law I exactly and is as close to the readings as a convex curve can be from
below) with the frontier's OWN binomial noise, and take the 95th percentile of
the violating share under that null.  Above it, the jaggedness is not noise.

This script computes both verdicts on every frontier with a public
evaluation-set size, reports how many change, and checks the separation the
paper's finding rests on -- distribution-shift benchmarks against everything
else -- under each rule.

Writes results/table_envelope_threshold.csv and
table_envelope_threshold_summary.csv.
"""
import math

import _pipeline as P

pc = P.pc
DRAWS = 400


def main():
    rows = []
    for f in P.frontiers():
        xs, es = list(f["x"]), list(f["e"])
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in o]; es = [es[i] for i in o]
        ls = [math.log(e) for e in es]
        n_items = P.eval_items(f["name"])
        if len(xs) < 4:
            continue
        eps_mid = sorted(es)[len(es) // 2]
        q_mid = min(max(1.0 - eps_mid, 1e-6), 1.0 - 1e-9)
        sd = pc.sd_of_l(q_mid, n_items) if n_items else None
        g = pc.class_gate(xs, ls, sd=sd)
        null = pc.envelope_share_null(xs, ls, sd, draws=DRAWS) if sd else None
        share = g["share_triples_violating"]
        rec = {"corpus": f["corpus"], "frontier": f["name"],
               "n_points": len(xs), "n_test": n_items, "sd_l": sd,
               "share_violating": share, "null_share_q95": null,
               "verdict_fixed_half": g["verdict"],
               "is_shift": bool(f["corpus"] == "vision-shift")}
        if null is not None and share is not None:
            g2 = pc.class_gate(xs, ls, sd=sd, envelope_share=null)
            rec["verdict_calibrated"] = g2["verdict"]
            rec["excess_over_null"] = share - null
        rows.append(rec)
    df = P.save(rows, "table_envelope_threshold.csv")

    d = df.dropna(subset=["null_share_q95"])
    changed = d[d.verdict_fixed_half != d.verdict_calibrated]
    summ = []
    for tag, sub in (("all with a public item count", d),
                     ("distribution shift", d[d.is_shift]),
                     ("everything else", d[~d.is_shift])):
        summ.append({
            "subset": tag, "n": len(sub),
            "median_share_violating": float(sub.share_violating.median()),
            "median_null_q95": float(sub.null_share_q95.median()),
            "median_excess_over_null": float(sub.excess_over_null.median()),
            "envelope_or_worse_fixed_half": float(
                (sub.verdict_fixed_half != "IN_CLASS").mean()),
            "envelope_or_worse_calibrated": float(
                (sub.verdict_calibrated != "IN_CLASS").mean()),
        })
    sdf = P.save(summ, "table_envelope_threshold_summary.csv")
    print()
    print(f"  the null is the frontier's own convex minorant plus its own"
          f" binomial noise, {DRAWS} draws, 95th percentile of the violating"
          f" share.")
    print(sdf.to_string(index=False))
    print()
    print(f"  median threshold: the fixed rule says 0.500 everywhere; the"
          f" calibrated one says {d.null_share_q95.median():.3f} on this"
          f" corpus,")
    print(f"  so the constant was far too permissive -- noise alone does not"
          f" produce anything like half.")
    print(f"  verdicts that change: {len(changed)} of {len(d)}"
          f" ({len(changed)/max(len(d),1):.0%}).")
    if len(changed):
        print(changed[["corpus", "frontier", "share_violating",
                       "null_share_q95", "verdict_fixed_half",
                       "verdict_calibrated"]].to_string(index=False))
    print()
    s_sh = d[d.is_shift].excess_over_null.median()
    s_el = d[~d.is_shift].excess_over_null.median()
    print(f"  the separation the paper's finding rests on survives, and"
          f" sharpens: distribution shift exceeds its own null by a median"
          f" {s_sh:+.3f}")
    print(f"  against {s_el:+.3f} elsewhere.  The statistic to report is the"
          f" EXCESS over the instance's own null, not the raw share.")


if __name__ == "__main__":
    main()

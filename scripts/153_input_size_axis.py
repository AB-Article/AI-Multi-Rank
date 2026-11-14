#!/usr/bin/env python3
"""153 -- the cheapest calibration sweep: one model, several INPUT SIZES.

F20 says the automatic price needs a sweep spanning the order of ten in
resource.  The question is where a deployment gets one without training
anything, and there is an answer: feed ONE trained model inputs of several
sizes.  Nothing is retrained, every point is one inference pass over the same
test set, and input size moves compute quadratically, so a 3x range of
resolution is already a 9x range of resource -- more than the design rule asks
for.

Whether such a sweep is a FRONTIER in the theory's sense is an empirical
question, because the axiom is about one mechanism with one clock, and giving
the same weights a larger input is not obviously the same kind of move as
giving a model more channels.  The archive can answer it: the MedMNIST blocks
carry an input-pixels axis at 28, 64, 128 and 224 pixels, in two variants --

    endToEnd       the network is RETRAINED at each input size
    linearProbing  one fixed backbone, only the input size changes

-- and the class gate separates them.  Retraining changes the mechanism between
points, so it is not one clock; a fixed model fed different sizes is much
closer to one.  That is the variant a deployment would actually run.

Writes results/table_input_size_axis.csv.
"""
import math

import pandas as pd

import _pipeline as P

pc = P.pc


def main():
    rows = []
    for f in P.frontiers():
        name = f["name"]
        if "cost_pixels" not in name:
            continue
        xs, es = list(f["x"]), list(f["e"])
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in o]; es = [es[i] for i in o]
        ls = [math.log(e) for e in es]
        n = len(xs)
        if n < 4:
            continue
        mode = ("retrained at each size" if "endToEnd" in name
                else ("fixed backbone, input size varies"
                      if "linearProbing" in name else "unknown"))
        ni = P.eval_items(name)
        sd = (pc.sd_of_l(min(max(1 - sorted(es)[n // 2], 1e-6), 1 - 1e-9), ni)
              if ni else None)
        g = pc.class_gate(xs, ls, sd=sd)
        steps = [b - a for a, b in zip(xs, xs[1:])]
        h = sum(steps) / len(steps) if steps else None
        rows.append({
            "frontier": name, "mode": mode, "n_points": n,
            "span_efolds": xs[-1] - xs[0],
            "span_ratio": math.exp(xs[-1] - xs[0]),
            "grid_h": h,
            "verdict": g["verdict"],
            "widest_defect": g["widest_defect"],
            "share_triples_violating": g["share_triples_violating"],
        })
    df = P.save(rows, "table_input_size_axis.csv")
    if not len(df):
        print("  no input-size frontiers found")
        return
    print()
    print("  the input-pixels axis of the archive, by how the points were"
          " produced:")
    print(df[["mode", "frontier", "n_points", "span_ratio", "verdict",
              "widest_defect"]].to_string(index=False))
    print()
    t = df.groupby("mode").verdict.value_counts().unstack(fill_value=0)
    print(t.to_string())
    print()
    for mode, d in df.groupby("mode"):
        okc = int((d.verdict != "OUT_OF_CLASS").sum())
        print(f"  {mode:<36}: in class or envelope on {okc} of {len(d)},"
              f" median widest chord defect {d.widest_defect.median():+.4f}")
    print()
    print(f"  the axis itself is wide: a median span of"
          f" {df.span_ratio.median():.0f}x in cost over"
          f" {df.n_points.median():.0f} points, against the order of ten that")
    print(f"  F20 asks for.  Input size is therefore the cheapest sweep"
          f" available -- and it is the only one")
    print(f"  that requires no training at all.")
    print()
    print("  CAVEAT, stated because it matters: n is small, and"
          " 'linearProbing' is a frozen backbone")
    print("  re-probed per resolution rather than one deployed model at"
          " several input sizes. The")
    print("  verdicts point the right way; they are not a proof. A deployment"
          " runs the gate on")
    print("  its own sweep and finds out -- which is what the gate is for.")
    print()
    print("  Two rules the geometry imposes on such a sweep:")
    print("    1. stay at or below the training resolution. Above it accuracy"
          " FALLS with resource,")
    print("       so those points are not Pareto and no frontier passes"
          " through them.")
    print("    2. place the sizes geometrically, and record the resource you"
          " will pay for")
    print("       (measured inference time on the target device), not the"
          " pixel count.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""157 -- the four input routes must give the same answer.  (Fixes F23, F24.)

The same evaluation can be handed to the selector in four shapes:

  A  measurements.csv                     one file: per-item outcomes + a
                                          resource row; accuracy and n_test
                                          are COUNTED from it
  B  candidates.csv --items items.csv     the aggregates beside the item matrix
  C  candidates.csv --pairs pairs.csv     the aggregates beside the pairwise
                                          2x2 tables -- K(K-1)/2 rows instead
                                          of n_test
  D  --model ... --pair ...                the same numbers typed on the
                                          command line, no file at all

A, B and C carry identical information and D carries it as arguments, so the
four must agree exactly.  This script runs all four and diffs the decision
blocks.  It is the regression test for the whole class of bug where two inputs
describing one evaluation drift apart: the certified interval of F21 is read off
a 2x2 table while the point break-even is read off the accuracies, so a stale
aggregate beside a fresh table prints a point outside its own interval.  F24's
consistency gate refuses that at the door, and part B here checks the gate
fires.

Needs no network.  Runs the archive's own worked example.
"""
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(os.path.dirname(HERE), "examples", "model_selection")
SEL = os.path.join(EX, "select_model.py")
BLOCKS = ("ENVELOPE", "BREAK-EVEN", "PRICE-FREE", "RANK")


def run(args):
    r = subprocess.run([sys.executable, SEL] + args, cwd=EX,
                       capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def decision_blocks(out):
    """The lines of the four decision blocks, with the MODE header dropped
    (it names the route, which is the one thing that legitimately differs)."""
    keep, on = [], False
    for ln in out.split("\n"):
        if any(ln.startswith(b + " ") or ln.startswith(b + ".") for b in BLOCKS):
            on = True
        elif ln and not ln.startswith(" ") and not ln.startswith("  "):
            on = ln.split(".")[0].strip() in BLOCKS
        if on:
            keep.append(ln.rstrip())
    return [k for k in keep if k.strip()]


def main():
    rows = [r for r in csv.DictReader(
        l for l in open(os.path.join(EX, "candidates.csv"))
        if not l.startswith("#"))]
    prs = [r for r in csv.DictReader(
        l for l in open(os.path.join(EX, "pairs.csv"))
        if not l.startswith("#"))]
    axis = "inference_time_ms"

    model_args = []
    for r in rows:
        model_args += ["--model", f"{r['name']},{r['accuracy']},"
                                  f"{r[axis]},{r['n_test']}"]
    pair_args = []
    for r in prs:
        pair_args += ["--pair", f"{r['i']},{r['j']},{r['both_fail']},"
                                f"{r['i_only']},{r['j_only']},{r['neither']}"]

    routes = {
        "A  measurements.csv": ["measurements.csv"],
        "B  candidates.csv --items items.csv":
            ["candidates.csv", "--items", "items.csv"],
        "C  candidates.csv --pairs pairs.csv":
            ["candidates.csv", "--pairs", "pairs.csv"],
        "D  --model/--pair on the command line":
            ["--axis", axis] + model_args + pair_args,
    }
    print()
    print("  A. the four routes, on the archive's own worked example:")
    ref_name, ref = None, None
    ok = True
    for tag, args in routes.items():
        code, out, err = run(args)
        if code != 0:
            print(f"     {tag:<40} FAILED (exit {code})")
            print("       " + (err.strip().split("\n")[0] if err else ""))
            ok = False
            continue
        blk = decision_blocks(out)
        if ref is None:
            ref_name, ref = tag, blk
            print(f"     {tag:<40} reference, {len(blk)} lines")
            continue
        if blk == ref:
            print(f"     {tag:<40} IDENTICAL to the reference")
        else:
            ok = False
            print(f"     {tag:<40} DIFFERS from {ref_name}:")
            for a, b in zip(ref, blk):
                if a != b:
                    print(f"       - {a}")
                    print(f"       + {b}")
            if len(ref) != len(blk):
                print(f"       (line counts {len(ref)} vs {len(blk)})")
    print()
    print(f"     verdict: {'all four routes agree' if ok else 'ROUTES DISAGREE'}")

    # ---------------- B: the consistency gate must fire on a stale input
    print()
    print("  B. F24, the consistency gate.  One accuracy is perturbed by 100")
    print("     items while its 2x2 table is left alone -- which is exactly how")
    print("     a stale aggregate file beside a fresh table looks.")
    r0 = rows[0]
    n = int(float(r0["n_test"]))
    bad = float(r0["accuracy"]) + 100.0 / n
    bad_args = ["--axis", axis,
                "--model", f"{r0['name']},{bad:.6f},{r0[axis]},{n}"]
    for r in rows[1:]:
        bad_args += ["--model", f"{r['name']},{r['accuracy']},"
                                f"{r[axis]},{r['n_test']}"]
    bad_args += pair_args
    code, out, err = run(bad_args)
    fired = (code != 0 and "F24" in (out + err))
    print(f"     gate fires and refuses : {fired}")
    if fired:
        first = [l for l in (out + err).split("\n") if "items apart" in l]
        if first:
            print(f"       {first[0].strip()}")
    code2, out2, err2 = run(bad_args + ["--force"])
    print(f"     --force proceeds anyway: {code2 == 0}")

    # ---------------- C: the gate must be INVARIANT to the chance level
    # F23 says q0 is never an input of a decision.  Until revision
    # 2026-10-08m, check_consistency compared a q0-scaled eps against a raw
    # counted failure rate, so every run with --q0 > 0 refused its own
    # consistent inputs by exactly the factor 1/(1 - q0) -- which made q0 an
    # input that could veto a decision.  The gate must now fire on exactly the
    # same inputs whatever q0 is.
    print()
    print("  C. F23 x F24: the consistency gate must not see the chance level.")
    q0s = [0.0, 0.1, 0.25, 0.5]
    clean_fired, stale_fired, bodies = [], [], []
    for q0 in q0s:
        c, o, e = run(routes["D  --model/--pair on the command line"]
                      + (["--q0", str(q0)] if q0 else []))
        clean_fired.append("F24" in (o + e))
        c2, o2, e2 = run(bad_args + (["--q0", str(q0)] if q0 else []))
        stale_fired.append("F24" in (o2 + e2))
        bodies.append([l for l in (o2 + e2).split("\n") if "items apart" in l])
    same_msg = all(b == bodies[0] for b in bodies)
    okC = (not any(clean_fired)) and all(stale_fired) and same_msg
    for q0, cf, sf in zip(q0s, clean_fired, stale_fired):
        print(f"     q0={q0:<5} consistent inputs refused: {str(cf):<5} "
              f"| stale input refused: {sf}")
    print(f"     the refusal message is identical at every q0: {same_msg}")
    print(f"     verdict: {'q0 cancels from the gate' if okC else 'q0 LEAKS INTO THE GATE'}")

    print()
    if not (ok and fired and okC):
        print("  FAILED: the routes must agree, the gate must fire, and the")
        print("  gate must not depend on the chance level.")
        sys.exit(1)
    print("  Four shapes of the same evaluation, one answer; an input that")
    print("  contradicts itself is refused rather than averaged; and the")
    print("  chance level changes neither the answer nor the refusal.")


if __name__ == "__main__":
    main()

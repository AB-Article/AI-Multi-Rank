#!/usr/bin/env python3
"""Pick a model from a quality--resource frontier, with every gate the theory
requires before it is allowed to name a winner.

    eps = (1-Q)/(1-q0),  S = -ln eps,  x = ln(resource),
    N_eff = s^2/kappa,   lambda* = s/sqrt(t N_eff),   operate at N_eff = 1/t.

This is version 3 of the worked example (revision 2026-10-08m).  Version 1
returned a single winner from a single resource at an assumed grain of one,
with no test of whether the frontier was in the theory's class and no account
of test-set noise.  All four of those were load-bearing.

THE SHAPE OF THE PROGRAM, in three levels, is what to read first; the F-numbered
list below is the audit trail, not the interface.

  Level 0  one file, nothing declared.  ``select_model.py measurements.csv``
           -- or ``candidates.csv`` with ``--items`` / ``--pairs``, or
           ``--model`` / ``--pair`` on the command line, all four giving the
           same answer (F23, F24; tested by ``scripts/157``).  Returns the
           price envelope, the break-even ladder, the price-free domination
           test, and the RANK table whose tier column is the price-free
           partial order.  No price, no grain, no q0.
  Level 1  ``--price-interval LO:HI``: adds the certified set at that interval
           AND the certified tiers over it (the RANKING block).  Neither is
           price-free -- tier membership moves with the interval -- which is why
           a Level-0 run prints no RANKING block at all.  An interval is a bound
           the deployment already knows, not a price.
  Level 2  ``--calibration`` + ``--grain``: derives the price lambda* and reads
           N_eff, the two walls and the floor.  This is the only level that
           needs a sweep, and the only one that needs the grain.

Flags belonging to a level a run does not reach are listed back under
"NOT READ AT THIS LEVEL" rather than silently dropped.

The fixes this version carries, F1..F24:

  F1  DECLARES THE GRAIN WHERE IT IS USED.  --grain is required only with
      --calibration, where the price is derived at the operating point
      N_eff = 1/t; nothing else reads it.  The grain -- the number of
      independently drawn items a delivered unit must all fail -- is the one
      input no frontier can supply (Thm. noinvariant), and it is never
      defaulted silently where it matters (F23).
  F2  GATES on Law I before any SECOND-ORDER reading.  ln eps must be convex on
      every wide triple.  Out of class there is no N_eff, no wall, no floor and
      no derived price -- but the envelope, the ladder and the certified set
      are unaffected and are still returned, because none of them uses
      convexity (F12, F21, F22).
  F3  PROPAGATES test-set noise.  n_test gives sd(ln eps) = sqrt(Q/((1-Q)n)),
      which widens the indifference band to the resolution actually achieved,
      puts an interval on N_eff, and reports the n_test a decision would need.
  F4  REPORTS BOTH objectives.  The accuracy-only canonical value and the
      resource-aware scalarization agree on only 40.7% of the paper's panel
      cells (median regret 0.029 nats, ninetieth percentile 0.65).  Both are
      printed and their disagreement is priced.
  F5  PRICES n RESOURCES.  One lambda per axis, derived where that axis has a
      calibration curve and declared with --price otherwise.
  F6  READS A COUNT AXIS when given one (--passk).  There the axiom is a
      theorem of exchangeability, the complete-monotonicity test is exact and
      signed at every spacing, and the floor is bracketed by two measured
      numbers, P(ceil L) <= eps_inf <= P(floor L), instead of extrapolated to
      x = infinity.
  F7  CORRECTS THE UNDER-SIZING.  1/N^x = 1/N^k + s_m/s, so a compute-axis
      reading is a lower bound on the task's own channel count and the rule
      fires too early.  With a count axis the bias is identified and the target
      becomes N^x = 1/(t + s_m/s).
  F8  RETURNS A TIE SET.  13.8 consecutive models fit inside one indifference
      band on a median published frontier, so a point winner is a fiction.  The
      verdict is RESOLVED or TIE, and inside the band the tiebreak is delivered
      cost, not accuracy.
  F9  DOES NOT REQUIRE THE CANDIDATES TO BE ONE FAMILY.  The price comes from
      the calibration frontier; the score comes from the candidate.  A
      heterogeneous zoo is scored on a common declared price.
  F10 CONVERTS A RETRY BUDGET into the grain it actually delivers,
      S(t_r)/S(1) < t_r, from the measured discount D(t).

Usage
-----
    # Level 0 -- ONE file, nothing declared
    python3 select_model.py measurements.csv

    # Level 0 -- the same, entirely from the command line
    python3 select_model.py --axis inference_time_ms \
        --model net_a,0.853280,4,50000 --model net_b,0.955980,24,50000 \
        --pair net_a,net_b,1894,5442,307,42357

    # Level 1 -- plus a bound on your own price
    python3 select_model.py measurements.csv --price-interval 0.45:0.55

    # Level 2 -- plus the sweep, which derives the price
    python3 select_model.py candidates.csv --grain 1 \
        --calibration sweep.csv --calibration-axis inference_time_ms \
        --items items.csv

Inputs
------
measurements.csv  ONE file and the only one Level 0 needs: row 1 names the
                  candidates, exactly one further row has a non-numeric first
                  cell and carries the resource (its first cell names the
                  axis), every other row is one test item with 1 = correct.
                  Accuracy and n_test are COUNTED from the item rows.
pairs.csv         i,j,both_fail,i_only,j_only,neither -- the pairwise 2x2
                  tables, which is all the certificate ever reads.  K(K-1)/2
                  rows instead of n_test, identical result.
models.csv        name,accuracy[,n_test][,<resource columns>] -- aggregates
                  only; the certificate then falls back to the unpaired normal
                  one, which is conservative.
calibration.csv   ONE model re-run at 5-7 budgets of ONE axis (five is the
                  protocol's figure; three is the algebraic floor for a second
                  difference), on a geometric grid (the closed forms for the
                  walls assume equal steps in log-resource; the spread is
                  reported).
passk.csv         k,P  -- P(k) is the probability an item is still failed after
                  k independent attempts (a pass@k curve read as fail@k, or the
                  co-failure curve of k drawn models).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_core as pc                                      # noqa: E402

NON_RESOURCE = {"name", "accuracy", "acc", "q", "n_test", "eps", "k", "p"}


# ------------------------------------------------------------------- reading

def _rows(path):
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def read_models(path, q0):
    rows = _rows(path)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    res_cols = [c for c in rows[0] if c and c.lower() not in NON_RESOURCE]
    out = []
    for r in rows:
        q = float(r.get("accuracy", r.get("acc")))
        m = {"name": r["name"], "accuracy": q, "eps": pc.eps_of(q, q0),
             "n_test": int(float(r["n_test"])) if r.get("n_test") else None,
             "t": {}, "x": {}}
        for c in res_cols:
            v = r.get(c)
            if v in (None, ""):
                continue
            m["t"][c] = float(v)
            m["x"][c] = math.log(float(v))
        out.append(m)
    return out, res_cols


def read_passk(path):
    rows = _rows(path)
    ks, ps = [], []
    for r in rows:
        key_k = "k" if "k" in r else list(r)[0]
        key_p = "P" if "P" in r else ("p" if "p" in r else list(r)[1])
        ks.append(float(r[key_k])); ps.append(float(r[key_p]))
    order = sorted(range(len(ks)), key=lambda i: ks[i])
    return [ks[i] for i in order], [ps[i] for i in order]


def read_cli_models(specs, axis):
    """F23.  Candidates straight from the command line: one ``--model``
    per candidate, ``name,accuracy,resource[,n_test]``.  The same three input
    shapes as the file routes, with no file."""
    out = []
    for spec in specs:
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) < 3:
            raise SystemExit(
                f"--model {spec!r}: need at least name,accuracy,{axis}"
                f"[,n_test]")
        name, acc, res = parts[0], float(parts[1]), float(parts[2])
        n = int(float(parts[3])) if len(parts) > 3 and parts[3] else None
        if not (0.0 < acc < 1.0):
            raise SystemExit(f"--model {spec!r}: accuracy must be in (0,1)")
        if res <= 0:
            raise SystemExit(f"--model {spec!r}: the resource must be > 0")
        out.append({"name": name, "accuracy": acc, "eps": pc.eps_of(acc, 0.0),
                    "n_test": n, "t": {axis: res},
                    "x": {axis: math.log(res)}})
    if len({m["name"] for m in out}) != len(out):
        raise SystemExit("--model: duplicate candidate names")
    return out


def read_cli_pairs(specs, names):
    """F23.  Pairwise 2x2 tables from the command line:
    ``i,j,both_fail,i_only,j_only,neither``."""
    out = {}
    for spec in specs:
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 6:
            raise SystemExit(
                f"--pair {spec!r}: need i,j,both_fail,i_only,j_only,neither")
        i, j = parts[0], parts[1]
        if i not in names or j not in names:
            raise SystemExit(f"--pair {spec!r}: unknown candidate name")
        tb = tuple(int(float(v)) for v in parts[2:])
        out[(i, j)] = tb
        out[(j, i)] = (tb[0], tb[2], tb[1], tb[3])
    return out


def read_single(path):
    """F23.  ONE file: the per-item outcomes, with the resource on a row of its
    own.  Everything else is derived, so nothing can disagree.

        inference_time_ms,  4.0,  9.0, 18.0, 24.0      <- the resource row
        net_a_small,     net_d_int8, net_c_mobile, net_b@224   (header)
        1,1,1,1
        0,1,0,1
        ...

    The first row names the candidates.  Exactly one further row has a
    NON-NUMERIC first cell: that is the resource row, its first cell names the
    axis.  Every other row is one test item, 1 = correct.  The accuracy of each
    candidate is COUNTED from those rows and n_test is their number, so there
    is nothing to keep in step and nothing to get wrong.

    Returns (models, items, axis, n_test).
    """
    rows = _rows_raw(path)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    header = rows[0]
    names = [c.strip() for c in header if c.strip()]
    axis, res = None, None
    item_rows = []
    for r in rows[1:]:
        if not any(c.strip() for c in r):
            continue
        first = r[0].strip()
        try:
            float(first)
            numeric_first = True
        except ValueError:
            numeric_first = False
        if not numeric_first and axis is None:
            axis = first
            res = [float(c) for c in r[1:1 + len(names)]]
            continue
        vals = [c.strip() for c in r[:len(names)]]
        item_rows.append(vals)
    if axis is None or res is None:
        raise SystemExit(
            f"{path}: no resource row found. One row must start with the name "
            f"of the resource (e.g. 'inference_time_ms') followed by one value "
            f"per candidate; every other row is a test item.")
    if len(res) != len(names):
        raise SystemExit(f"{path}: the resource row has {len(res)} values for "
                         f"{len(names)} candidates")
    if not item_rows:
        raise SystemExit(f"{path}: no item rows")
    n = len(item_rows)
    items, correct = {}, {}
    for k, nm in enumerate(names):
        fails = []
        c = 0
        for r in item_rows:
            v = r[k].lower()
            ok = v in ("1", "true", "t", "yes", "y", "correct")
            fails.append(not ok)
            c += 1 if ok else 0
        items[nm] = fails
        correct[nm] = c
    models = []
    for k, nm in enumerate(names):
        q = correct[nm] / float(n)
        models.append({"name": nm, "accuracy": q, "eps": pc.eps_of(q, 0.0),
                       "n_test": n, "t": {axis: res[k]},
                       "x": {axis: math.log(res[k])}})
    return models, items, axis, n


def read_pairs(path, names):
    """F23.  The alternative to shipping the item matrix: its pairwise 2x2
    tables, which is all the certificate ever reads.

        i,j,both_fail,i_only,j_only,neither

    One row per unordered pair.  This is K(K-1)/2 rows instead of n_test, so a
    50,000-item evaluation of seven candidates becomes 21 lines -- and the
    certificate is exactly the same, because ``certify_domination_counts``
    reads only the table.
    """
    out = {}
    for r in _rows(path):
        i, j = r["i"].strip(), r["j"].strip()
        if i not in names or j not in names:
            continue
        tb = (int(float(r["both_fail"])), int(float(r["i_only"])),
              int(float(r["j_only"])), int(float(r["neither"])))
        out[(i, j)] = tb
        out[(j, i)] = (tb[0], tb[2], tb[1], tb[3])
    if not out:
        raise SystemExit(f"{path}: no pair matches a candidate name")
    return out


def _rows_raw(path):
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    return [r for r in csv.reader(lines) if r]


def read_items(path, names):
    """F11/F12.  Per-item outcomes on the shared test set: one column per
    candidate, one row per item, 1 = correct (0 = failed).  Returns
    {name: [failed?]} for the candidates present in both files."""
    rows = _rows(path)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    cols = [c for c in rows[0] if c and c in names]
    if not cols:
        raise SystemExit(f"{path}: no column matches a candidate name "
                         f"({', '.join(sorted(names))})")
    out = {c: [] for c in cols}
    for r in rows:
        for c in cols:
            v = str(r[c]).strip().lower()
            if v in ("", "na", "nan"):
                raise SystemExit(f"{path}: missing outcome for {c}")
            correct = v in ("1", "true", "t", "yes", "y", "correct")
            out[c].append(not correct)              # store FAILURE
    n = {len(v) for v in out.values()}
    if len(n) != 1:
        raise SystemExit(f"{path}: columns have different lengths")
    return out


def pareto(models, axes):
    """Pareto set on (eps, resources): keep m unless another is <= on every
    axis and < on one."""
    keep = []
    for a in models:
        dom = False
        for b in models:
            if b is a:
                continue
            le = (b["eps"] <= a["eps"]
                  and all(b["t"].get(k, 0) <= a["t"].get(k, 0) for k in axes))
            lt = (b["eps"] < a["eps"]
                  or any(b["t"].get(k, 0) < a["t"].get(k, 0) for k in axes))
            if le and lt:
                dom = True
                break
        if not dom:
            keep.append(a)
    return keep


# ------------------------------------------------------------------- the run

def run(a):
    q0 = a.q0
    # ONE-FILE MODE: if the file has no 'accuracy' column it is the per-item
    # table with a resource row, and everything else is derived from it.
    single = None
    cli_tables = None
    if a.model:
        if a.models:
            raise SystemExit("pass either a file or --model, not both")
        ax_cli = a.axis or "resource"
        cands = read_cli_models(a.model, ax_cli)
        res_cols = [ax_cli]
        if a.pair:
            cli_tables = read_cli_pairs(a.pair, {m["name"] for m in cands})
        hdr = []
    elif not a.models:
        raise SystemExit(
            "give a file (measurements.csv) or pass --model once per candidate")
    else:
        hdr = _rows_raw(a.models)[0] if _rows_raw(a.models) else []
    if a.model:
        pass
    elif not any(c.strip().lower() in ("accuracy", "acc") for c in hdr):
        cands, single_items, single_axis, single_n = read_single(a.models)
        single = {"items": single_items, "axis": single_axis,
                  "n_test": single_n}
        res_cols = [single_axis]
    else:
        cands, res_cols = read_models(a.models, q0)
    axes = a.resources.split(",") if a.resources else res_cols
    axes = [c for c in axes if c]

    # ---- F23: MEASUREMENT-ONLY MODE.
    # With no calibration sweep and no declared price there is still a
    # complete, certified answer, and it needs neither the grain nor q0:
    # the envelope is sigma_i(lambda) = S_i - lambda x_i read as a FUNCTION of
    # lambda, its breakpoints are the certified break-evens, and the rung at
    # lambda = 0 is the price-free test.  q0 cancels from every one of those
    # quantities identically.
    if not a.calibration and not a.price and not a.price_interval:
        ax = (single["axis"] if single else
              (a.axis or a.calibration_axis or
               (axes[0] if len(axes) == 1 else None)))
        if a.model:
            ax = a.axis or "resource"
        if ax is None:
            raise SystemExit(
                "several resource columns and no price: name the one to read "
                "the envelope on with --axis, or supply --price for each")
        items2, tables2 = None, cli_tables
        if single:
            items2 = single["items"]
        if a.pairs:
            tables2 = read_pairs(a.pairs, {m["name"] for m in cands})
        if a.items:
            items2 = read_items(a.items, {m["name"] for m in cands})
        if items2 and not tables2:
            tables2 = {}
            for i in cands:
                for j in cands:
                    if i["name"] == j["name"]:
                        continue
                    fi, fj = items2.get(i["name"]), items2.get(j["name"])
                    if fi is None or fj is None:
                        continue
                    tables2[(i["name"], j["name"])] = pc.contingency(fi, fj)
        # ---- F24: the inputs must agree with each other before anything is
        # read off them.  A stale accuracy beside a fresh table silently
        # produces a point break-even outside its own certified interval.
        probs = pc.check_consistency(cands, tables=tables2, items=items2)
        if probs and not a.force:
            raise SystemExit(
                "INPUTS DISAGREE (F24) -- refusing to decide:\n  "
                + "\n  ".join(probs)
                + "\n\nA 2x2 table fixes both error rates and the item count "
                  "exactly, and an item\nvector fixes the accuracy exactly, so "
                  "these are not roundings. Re-export the\naggregates from the "
                  "same evaluation, or drop them and let the tool count.\n"
                  "Use --force to proceed anyway (the point break-even may "
                  "then fall outside\nits own certified interval).")
        n_any = next((m["n_test"] for m in cands if m["n_test"]), None)
        res = pc.decide_from_measurements(
            cands, ax, alpha=a.alpha, items=items2, tables=tables2,
            n_test=n_any, rho=a.rho)
        # F17: nothing degrades silently.  Level 0 reads only the
        # accuracies, the resource column and the item counts; every other
        # flag belongs to a higher level and is recorded here so the report
        # can say it was ignored instead of dropping it (revision
        # 2026-10-08m).
        _given = set()
        for _tok in sys.argv[1:]:
            _t = str(_tok)
            if _t.startswith("--"):
                _given.add(_t.split("=", 1)[0])
        ignored = [nm for nm in ("--passk", "--floor", "--grain",
                                 "--retry-budget", "--attempt-budget",
                                 "--theta", "--resources", "--q0",
                                 "--tie-band", "--no-bias-correction")
                   if nm in _given]
        return {"mode": "measurement-only", "axis": ax, "result": res,
                "models": cands, "n_test": n_any, "ignored_flags": ignored,
                # which input actually supplied the pairing, so the provenance
                # line names the file the run received and not the one it did
                # not (revision 2026-10-08m)
                "from_items": bool(items2),
                "method": ("exact-paired" if tables2 else
                           ("normal-paired" if a.rho is not None else
                            "normal-unpaired" if n_any else "uncertified")),
                "warnings": ([] if (tables2 or n_any) else
                             ["no n_test and no --items: the ladder has no "
                              "certified interval, only point break-evens"])}

    grain_used = True
    if a.grain is None:
        if a.calibration:
            raise SystemExit(
                "--grain is required with --calibration, because the price is "
                "derived at the operating point N_eff = 1/t (Thm. "
                "noinvariant). It is NOT needed for the envelope, the ladder, "
                "the price-free test or a declared price interval.")
        a.grain = "1"              # a placeholder: nothing below reads it
        grain_used = False
    rep = {"grain": {}, "gates": {}, "count_axis": None, "plane": {},
           "prices": {}, "decision": {}, "warnings": []}

    # ---- F1: the grain, parsed and recorded; read only where a price is
    # derived
    if ":" in a.grain:
        t_lo, t_hi = (float(v) for v in a.grain.split(":"))
    else:
        t_lo = t_hi = float(a.grain)
    if not (t_lo > 0 and t_hi >= t_lo):
        raise SystemExit("--grain must be t>0 or t_min:t_max with t_min<=t_max")
    rep["grain"] = {"t_lo": t_lo, "t_hi": t_hi,
                    "t_eff": math.sqrt(t_lo * t_hi), "declared": True,
                    "used": grain_used}

    # ---- F6/F10: the count axis, if supplied
    undersize_c, floor_bracket = 0.0, None
    neff_count = None
    if a.passk:
        ks, ps = read_passk(a.passk)
        ca = pc.count_axis_read(ps, budget=a.attempt_budget or len(ps))
        rep["count_axis"] = {
            "K": ca["K"], "verdict": ca["verdict"],
            "refuted_orders": ca["alternation"]["refuted_orders"],
            "neff": ca["neff"],
            "floor_bracket": ca["floor_bracket"],
            "discount": {k: ca["discount"][k] for k in sorted(ca["discount"])
                         if k <= 4},
            "grain_of_retry": {k: ca["grain_of_retry"][k]
                               for k in sorted(ca["grain_of_retry"]) if k <= 4},
        }
        if ca["neff"]:
            vals = sorted(ca["neff"].values())
            neff_count = vals[len(vals) // 2]
        floor_bracket = ca["floor_bracket"]
        if a.retry_budget:
            tr = int(a.retry_budget)
            g = ca["grain_of_retry"].get(tr)
            if g:
                rep["grain"]["retry_budget"] = tr
                rep["grain"]["grain_delivered_by_retries"] = g
                t_lo = t_hi = max(t_lo, g)
                rep["grain"]["t_lo"] = t_lo
                rep["grain"]["t_hi"] = t_hi
                rep["grain"]["t_eff"] = math.sqrt(t_lo * t_hi)

    # ---- the calibration frontier, the gates, and the derived price
    cal_axis = a.calibration_axis
    prices, derived, price_sd, price_range = {}, {}, {}, {}
    rep["prices"] = {}
    if a.calibration:
        cal, cal_res = read_models(a.calibration, q0)
        if cal_axis is None:
            cal_axis = cal_res[0]
        pts = sorted(((m["x"][cal_axis], math.log(m["eps"]), m)
                      for m in cal if cal_axis in m["x"]))
        xs = [p[0] for p in pts]
        ls = [p[1] for p in pts]
        sds = [pc.sd_of_l(p[2]["accuracy"], p[2]["n_test"]) for p in pts]
        sd = (sum(sds) / len(sds)) if any(sds) else None

        # F3: grid spread -- the closed forms assume equal steps
        steps = [b - a_ for a_, b in zip(xs, xs[1:])]
        spread = (max(steps) - min(steps)) / (sum(steps) / len(steps)) \
            if steps else 0.0
        rep["gates"]["grid_spread"] = spread
        if spread > 0.25:
            rep["warnings"].append(
                f"calibration grid is not geometric (spread {spread:.2f}): the "
                f"closed forms for the two walls assume equal steps in "
                f"log-resource; use scripts/125_grid_corpus.py")

        # F2 + A7: the class gate, with the ENVELOPE threshold calibrated on
        # this frontier's own noise instead of the constant one half
        env_null = pc.envelope_share_null(xs, ls, sd) if sd else None
        gate = pc.class_gate(xs, ls, sd=sd,
                             envelope_share=(env_null if env_null is not None
                                             else pc.ENVELOPE_SHARE))
        gate["envelope_threshold"] = (env_null if env_null is not None
                                      else pc.ENVELOPE_SHARE)
        gate["envelope_threshold_source"] = (
            "calibrated on this frontier's own convex minorant and noise"
            if env_null is not None else "the constant 0.5 (no n_test given)")
        rep["gates"]["class"] = gate

        # F3: what N_eff is worth at this precision
        if len(xs) >= 3 and sd:
            mid = len(xs) // 2
            tri = (mid - 1, mid, mid + 1) if mid + 1 < len(xs) else (0, 1, 2)
            med, lo, hi, badfrac = pc.neff_uncertainty(
                [xs[i] for i in tri], [ls[i] for i in tri],
                [sds[i] for i in tri])
            rep["gates"]["neff_interval"] = {
                "median": med, "lo5": lo, "hi95": hi,
                "frac_out_of_class": badfrac}
            if med and lo and hi and med > 0:
                if hi / max(lo, 1e-12) > 4.0:
                    rep["warnings"].append(
                        "N_eff spans more than one doubling at this n_test: the "
                        "operating point is not resolved by the data. N_eff is a "
                        "square over a second difference and amplifies noise "
                        "(corpus median straightness R^2 = 0.22, no power "
                        "against a design-matched null)")

        if gate["verdict"] == "OUT_OF_CLASS" and not a.force:
            # Law I has failed, so nothing second-order may be read: no N_eff,
            # no walls, no floor, no derived price.  But the certified set of
            # F12 needs none of those, so a decision is still returned -- on a
            # DECLARED price only, which is the honest remaining input.
            rep["decision"] = {
                "verdict": "OUT_OF_CLASS", "pick": None,
                "pareto": [m["name"] for m in pareto(cands, axes)],
                "reason": gate["reason"]}
            rep["out_of_class"] = True
            rep["out_of_class_reason"] = gate["reason"]
            rep["pareto"] = [m["name"] for m in pareto(cands, axes)]

        # Everything below is SECOND ORDER -- N_eff, the walls, the
        # floor, the derived price -- and none of it may be read once
        # Law I has failed.  The certified set of F12 needs none of it.
        if not rep.get("out_of_class"):
            # F7: correct the compute-axis target before locating the point
            plane0 = pc.read_plane(xs, ls, t_lo, t_hi, sd=sd)
            target, undersize_c, note = pc.undersizing_correction(
                math.sqrt(t_lo * t_hi) and rep["grain"]["t_eff"],
                plane0.get("operate_neff"), neff_count)
            rep["gates"]["undersizing"] = {"s_m_over_s": undersize_c, "note": note}
            plane = pc.read_plane(xs, ls, t_lo, t_hi, sd=sd,
                                  undersize_c=undersize_c or 0.0)
            rep["plane"] = {k: v for k, v in plane.items() if k != "points"}
            if plane["price"] is not None:
                prices[cal_axis] = plane["price"]
                # the price range the frontier itself supports: its own local
                # prices, min to max.  This is what the minimax tiebreak ranges over.
                loc = [p for p in (
                    (-(ls[i + 1] - ls[i]) / (xs[i + 1] - xs[i]))
                    for i in range(len(xs) - 1)) if p > 0]
                if len(loc) >= 2:
                    price_range[cal_axis] = (min(loc), max(loc))
                derived[cal_axis] = "derived from the calibration frontier"
                if sd:
                    pm, plo, phi, psd = pc.price_uncertainty(
                        xs, ls, sds, t_lo, t_hi, undersize_c or 0.0)
                    rep["prices"]["uncertainty"] = {
                        cal_axis: {"median": pm, "lo5": plo, "hi95": phi,
                                   "sd": psd}}
                    rel = ((phi - plo) / pm) if (pm and pm > 0) else None
                    rep["prices"]["uncertainty"][cal_axis]["rel_spread"] = rel
                    if psd and rel is not None and rel <= 1.0:
                        price_sd[cal_axis] = psd
                    else:
                        rep["gates"]["price_identified"] = False
                        rep["warnings"].append(
                            "the derived price is NOT identified at this "
                            "precision: its 90% interval spans more than its own "
                            "value. The band is reported from accuracy noise "
                            "alone and the price must not be used as if measured.")
                # F16: the O(h^2) slope bias, corrected when it is bigger than
                # the price's own noise
                rich = pc.richardson_price(xs, ls, t_lo, t_hi,
                                           undersize_c or 0.0,
                                           price_sd=price_sd.get(cal_axis))
                rep["prices"]["bias_correction"] = rich
                if (not a.no_bias_correction and rich.get("recommend")
                        == "corrected" and rich.get("price_recommended")):
                    prices[cal_axis] = rich["price_recommended"]
                    derived[cal_axis] = ("derived, O(h^2) bias removed by "
                                         "Richardson")
                req = pc.calibration_requirement(
                    xs, ls, pts[len(pts) // 2][2]["accuracy"])
                if req and req.get("n_test_needed"):
                    have = pts[len(pts) // 2][2]["n_test"] or 0
                    rep["gates"]["calibration_requirement"] = {
                        "h": req["h"], "n_test_needed": req["n_test_needed"],
                        "n_test_have": have,
                        "widen_grid_by": (math.sqrt(req["n_test_needed"] / have)
                                          if have else None)}
                # F13: the PROSPECTIVE design rule, which is what should have been
                # checked before any of these runs happened
                mid_m = pts[len(pts) // 2][2]
                j_ = len(xs) // 2
                s_mid = pc.local_slope(xs[j_ - 1:j_ + 2], ls[j_ - 1:j_ + 2]) \
                    if len(xs) >= 3 else None
                steps_ = [b - a_ for a_, b in zip(xs, xs[1:])]
                h_have = (sum(steps_) / len(steps_)) if steps_ else None
                if s_mid and h_have and mid_m["n_test"]:
                    rep["gates"]["design"] = pc.design_calibration(
                        mid_m["accuracy"], mid_m["n_test"], s_mid,
                        t=rep["grain"]["t_eff"], K=len(xs),
                        h_available=h_have)
    else:
        rep["warnings"].append(
            "no --calibration: no price is derived, and none is needed -- the "
            "envelope and the ladder are read as functions of lambda (F21, "
            "F22). Supply a five-point sweep of one model AND --grain only if "
            "you want a point price, lambda* = sqrt(kappa/t).")

    # ---- a declared price INTERVAL with no point price: the geometric
    # midpoint is the point at which the ranking is reported, and the interval
    # is what the certificate must survive
    if a.price_interval and not a.price:
        lo_s, hi_s = a.price_interval.split(":")
        axk = a.axis or cal_axis or (axes[0] if axes else None)
        if axk:
            prices[axk] = math.sqrt(float(lo_s) * float(hi_s))
            derived[axk] = "declared-interval midpoint"

    # ---- F5: declared prices for the remaining axes
    for item in (a.price or []):
        for part in item.split(","):
            if not part:
                continue
            k, v = part.split("=")
            prices[k.strip()] = float(v)
            derived[k.strip()] = "declared on the command line"
    missing = [c for c in axes if c not in prices]
    if missing:
        rep["warnings"].append(
            f"axes priced at zero because neither a calibration curve nor a "
            f"--price was given: {', '.join(missing)}. A resource at price zero "
            f"is a resource the decision ignores.")
    rep["prices"].update({"lambda": prices, "provenance": derived,
                          "n_axes_priced": len(prices), "n_axes": len(axes)})
    if len(prices) > 1:
        th = a.theta
        n = len(prices)
        rep["prices"]["interaction"] = {
            "theta": th,
            "canonical_depth_nats": math.log(1.0 + n ** (1.0 / th)),
            "note": ("with n interacting resources the canonical point is "
                     "eps* = (1 + n^(1/theta)) eps_inf and the worst case is "
                     "ln(1 + n^(1/theta)); theta is declared, not fitted here")}

    # ---- F11: per-item outcomes, if supplied: the gap is PAIRED
    items = None
    tables = cli_tables
    if single:
        items = single["items"]
    if a.pairs:
        tables = read_pairs(a.pairs, {m["name"] for m in cands})
    if a.items:
        items = read_items(a.items, {m["name"] for m in cands})
    probs = pc.check_consistency(cands, tables=tables, items=items)
    if probs and not a.force:
        raise SystemExit(
            "INPUTS DISAGREE (F24) -- refusing to decide:\n  "
            + "\n  ".join(probs)
            + "\n\nRe-export the aggregates from the same evaluation, or drop "
              "them and let the\ntool count. --force proceeds anyway.")
    if items is not None and tables is None:
        n_items_file = len(next(iter(items.values())))
        tables = {}
        for i in cands:
            for j in cands:
                if i["name"] == j["name"]:
                    continue
                fi, fj = items.get(i["name"]), items.get(j["name"])
                if fi is None or fj is None:
                    continue
                tables[(i["name"], j["name"])] = pc.contingency(fi, fj)
        rep["paired"] = {"n_items": n_items_file,
                         "n_candidates_covered": len(items)}
        # the measured failure correlation of every adjacent pair
        phis = []
        for (ni, nj), tb in tables.items():
            aa, bb, cc, dd = tb
            n_ = aa + bb + cc + dd
            phis.append(pc.phi_correlation((aa + bb) / n_, (aa + cc) / n_,
                                           aa / n_))
        if phis:
            rep["paired"]["median_phi"] = sorted(phis)[len(phis) // 2]

    # ---- F3: resolution per candidate
    resolution = {m["name"]: pc.sd_of_l(m["accuracy"], m["n_test"])
                  for m in cands if m["n_test"]}
    resolution = {k: v for k, v in resolution.items() if v}
    # F11: the PAIRED resolution of each pair, which is what the band needs
    pair_sd = None
    if tables:
        pair_sd = {}
        for (ni, nj), tb in tables.items():
            aa, bb, cc, dd = tb
            n_ = aa + bb + cc + dd
            pair_sd[(ni, nj)] = pc.sd_of_gap((aa + bb) / n_, (aa + cc) / n_,
                                             n_, p_both=aa / n_)

    # ---- F6: the floor, measured not fitted, for the canonical value
    floor = None
    if floor_bracket:
        floor = 0.5 * (floor_bracket[0] + floor_bracket[1])
    elif a.floor is not None:
        floor = a.floor

    # ---- F4/F8/F9: the ranking and the indifference band (version 2)
    dec = pc.select(cands, prices, band=a.tie_band, floor=floor,
                    resolution=resolution or None,
                    price_sd=price_sd or None,
                    price_range=price_range or None)
    dec["pareto"] = [m["name"] for m in pareto(cands, axes)]
    rep["decision"] = dec

    # ---- F12: the CERTIFIED set, which is the verdict that carries a number.
    # The interval it ranges over is the PRICE UNCERTAINTY and nothing wider:
    # the derived price's own 90% interval, or a declared one.
    cert_range = None
    if a.price_interval:
        lo_s, hi_s = a.price_interval.split(":")
        axk = cal_axis or (axes[0] if axes else None)
        if axk:
            cert_range = {axk: (float(lo_s), float(hi_s))}
    elif price_sd and rep.get("prices", {}).get("uncertainty"):
        for axk, u in rep["prices"]["uncertainty"].items():
            if u.get("lo5") and u.get("hi95") and u["hi95"] > u["lo5"]:
                cert_range = {axk: (u["lo5"], u["hi95"])}
    n_test_any = next((m["n_test"] for m in cands if m["n_test"]), None)
    cert = pc.certified_set(
        cands, prices, price_range=cert_range, alpha=a.alpha,
        items=items, tables=tables, n_test=n_test_any,
        rho=(a.rho if (a.rho is not None and not tables) else None))
    cert["price_range"] = cert_range
    rep["certified"] = cert

    # ---- F12/F14: the schedule, when the price is free over a wide range,
    # and the number of networks it actually names
    if cert_range:
        axk, (lo_c, hi_c) = next(iter(cert_range.items()))
        others = {k: v for k, v in prices.items() if k != axk}
        sch = pc.certified_schedule(
            cands, axk, lo_c, hi_c, alpha=a.alpha, other_prices=others,
            n_cells=21, items=items, tables=tables, n_test=n_test_any,
            rho=(a.rho if (a.rho is not None and not tables) else None))
        rep["schedule"] = {
            "axis": axk, "lam_lo": lo_c, "lam_hi": hi_c,
            "n_networks": sch["n_networks"], "networks": sch["networks"],
            "one_network_pick": sch["one_network_pick"],
            "one_network_worst_case": sch["one_network_worst_case"],
            "cells": [{"lambda": c["lambda"], "pick": c["pick"],
                       "set_size": len(c["set"])} for c in sch["cells"]]}

    # ---- F18: the price-free question first -- it is the strongest statement
    # available and needs no price, no grain and no calibration curve at all.
    rep["pareto_certified"] = pc.certified_pareto(
        cands, alpha=a.alpha, items=items, tables=tables,
        n_test=n_test_any, rho=a.rho, axes=axes)

    # ---- F21: the price ladder -- the simplest statement of all, and it
    # needs no sweep, no grain and no derived price
    be_axis = cal_axis or (axes[0] if len(axes) == 1 else None)
    if be_axis:
        # F22: the ladder is built on the price ENVELOPE, not on cost
        # neighbours, so it needs no knowledge of which candidates are
        # efficient -- a chain of cost neighbours is wrong as soon as one
        # candidate is dominated or sits in a dent.
        bee = pc.break_even_envelope(
            cands, be_axis, alpha=a.alpha, items=items, tables=tables,
            n_test=n_test_any, rho=a.rho)
        rep["break_even"] = bee["rungs"]
        rep["envelope"] = bee["envelope"]

    # ---- F19: rank everything into certified tiers
    rep["ranking_certified"] = pc.certified_ranking(
        cands, prices, price_range=cert_range, alpha=a.alpha,
        items=items, tables=tables, n_test=n_test_any,
        rho=(a.rho if (a.rho is not None and not tables) else None))

    # ---- F20: is the derived price sharp enough to NAME one candidate?
    if cert_range:
        axk, (lo_c, hi_c) = next(iter(cert_range.items()))
        h_av = None
        if a.calibration:
            try:
                h_av = rep["gates"]["calibration_requirement"]["h"]
            except Exception:
                h_av = None
        rep["design_selection"] = pc.design_for_selection(
            cands, prices, axk, lo_c, hi_c, h_available=h_av)

    # ---- F14/F15: is the candidate list finished, and where does the next
    # network go?
    rep["stopping"] = pc.stopping_rule(cands, prices,
                                       cert_range or price_range or None)

    # ---- what a decision would need
    need = []
    top = dec["ranking"][0]["name"]
    for r in dec["ranking"][1:]:
        if 0 < r["gap"] <= dec["effective_band"]:
            m = next(c for c in cands if c["name"] == r["name"])
            if tables and (top, r["name"]) in tables:
                aa, bb, cc, dd = tables[(top, r["name"])]
                n_ = aa + bb + cc + dd
                v = pc.required_n_gap((aa + bb) / n_, (aa + cc) / n_,
                                      max(r["gap"], 1e-6), p_both=aa / n_)
            else:
                v = pc.required_n_gap(
                    next(c for c in cands
                         if c["name"] == top)["eps"], m["eps"],
                    max(r["gap"], 1e-6), rho=a.rho)
            need.append((r["name"], v))
    rep["decision"]["required_n_test"] = dict(need)
    rep["decision"]["required_n_test_paired"] = bool(tables)
    rep["depth_bounds"] = pc.depth_bounds(rep["grain"]["t_eff"])
    # ---- F17: one line per input, each saying what its absence costs
    rep["inputs"] = pc.inputs_audit(
        cands, grain=(rep["grain"]["t_eff"]
                      if rep["grain"].get("used") else None),
        q0=(q0 if a.q0_given else None), n_test=n_test_any, items=items,
        tables=tables,
        price_identified=rep.get("gates", {}).get("price_identified"),
        q0_convention=a.q0_convention)
    return rep


# ------------------------------------------------------------------- output

def show_measurement_only(rep):
    """F23.  The whole report that the candidates' own measurements support."""
    r = rep["result"]; env = r["envelope"]; ax = rep["axis"]
    by = {m["name"]: m for m in rep["models"]}
    print()
    print(f"MODE ........... measurement-only: {len(rep['models'])} candidates,"
          f" one evaluation each, resource '{ax}'")
    src = ""
    if rep["method"] == "exact-paired":
        src = (", per-item outcomes" if rep.get("from_items")
               else ", 2x2 tables")
    print(f"  inputs used .. accuracy, {ax}, n_test" + src
          + f"   [{rep['method']}]")
    print(f"  NOT used ..... no price, no calibration sweep, no grain, no"
          f" chance level q0 -- none of them")
    print(f"                 enters the envelope, the ladder or the price-free"
          f" test, and q0 cancels")
    print(f"                 identically from every difference of sigma.")
    print()
    print(f"ENVELOPE ....... who can be optimal at ALL -- {env['n_winners']}"
          f" of {env['n_candidates']} candidates:")
    for nm, lo_, hi_ in env["pieces"]:
        hi_s = "infinity" if hi_ > 1e5 else f"{hi_:.4f}"
        print(f"  {str(nm):<16} optimal for price in [{lo_:.4f}, {hi_s}]"
              f" nats/e-fold of {ax}")
    if env["never"]:
        print(f"  never optimal at any price:"
              f" {', '.join(str(x) for x in env['never'])}")
        if env["dominated"]:
            print(f"    dominated .... "
                  f"{', '.join(str(x) for x in env['dominated'])}")
        if env["in_a_dent"]:
            print(f"    in a dent .... "
                  f"{', '.join(str(x) for x in env['in_a_dent'])}"
                  f"  (Pareto-efficient, and still never selected)")
    print()
    print("BREAK-EVEN ..... the ladder, one rung per envelope breakpoint:")
    print(f"  {'cheaper':<16} {'dearer':<16} {'break-even':>10}"
          f" {'certified interval':>22} {'undecided':>11}")
    for g_ in r["rungs"]:
        lo_s = f"{g_['certified_lo']:.4f}" if g_["certified_lo"] else "    --"
        hi_s = f"{g_['certified_hi']:.4f}" if g_["certified_hi"] else "    --"
        und = (f"{g_['undecided_doublings']:.2f} dbl"
               if g_.get("undecided_doublings") else "   --")
        print(f"  {str(g_['cheap']):<16} {str(g_['dear']):<16}"
              f" {g_['break_even']:>10.4f}"
              f" {('[' + lo_s + ', ' + hi_s + ']'):>22} {und:>11}")
    print("  read it as: pay for the dearer model while your price per e-fold"
          " is below the rung.")
    pf = r["price_free"]
    if pf["dominations"]:
        print()
        print(f"PRICE-FREE ..... {len(pf['dominations'])} domination(s)"
              f" certified on accuracy alone, valid at EVERY price:")
        for w, l in pf["dominations"][:8]:
            print(f"  {w} beats {l}")
    print()
    print("RANK ........... tier from the price-free order; inside a tier,"
          " by where each is optimal")
    hdr = (f"  {'#':>2} {'name':<16}{'eps':>10}{ax[:12]:>13}{'tier':>6}"
           f"{'optimal for price in':>24}")
    print(hdr)
    i = 0
    for ti, tier in enumerate(r["tiers"], 1):
        for nm in tier:
            i += 1
            m = by[nm]
            w = r["optimal_interval"].get(nm)
            if w:
                hi_s = "inf" if w[1] > 1e5 else f"{w[1]:.3f}"
                wstr = f"[{w[0]:.3f}, {hi_s}]"
            else:
                wstr = "never optimal"
            res_v = math.exp(m["x"][ax]) if ax in m["x"] else float("nan")
            print(f"  {i:>2} {str(nm):<16}{m['eps']:>10.5f}{res_v:>13.4g}"
                  f"{ti:>6}{wstr:>24}")
    print("  no sigma column, because sigma needs a price. The envelope column"
          " is the ranking by")
    print("  trade-off, and it is complete: every price is covered by exactly"
          " one candidate.")
    print()
    print("WHAT MORE WOULD BUY ...")
    print("  a price interval (--price-interval LO:HI) -> the certified SET at"
          " that interval, and tiers")
    print("                                               that use it")
    print("  a 5-point sweep (--calibration + --grain)  -> the price itself,"
          " lambda* = sqrt(kappa/t),")
    print("                                               plus N_eff, the"
          " walls and the floor")
    print("  a count axis (--passk)                     -> the floor as two"
          " measured numbers")
    ign = rep.get("ignored_flags") or []
    if ign:
        print()
        print("NOT READ AT THIS LEVEL ...")
        print(f"  {', '.join(ign)}")
        print("  Level 0 reads the accuracies, the resource column and the"
              " item counts and nothing")
        print("  else.  The flags above belong to a level this run did not"
              " reach: --passk, --floor,")
        print("  --retry-budget, --attempt-budget, --theta and --q0 are read"
              " only once a price or a")
        print("  calibration sweep is supplied.  Nothing has been silently"
              " dropped; add"
              " --price-interval")
        print("  or --calibration to put them to work.")
    for w in rep.get("warnings", []):
        print(f"! {w}")


def show(rep):
    if rep.get("mode") == "measurement-only":
        return show_measurement_only(rep)
    g = rep["grain"]
    w = lambda d, k, dflt=None: d.get(k, dflt)
    print()
    if not g.get("used", True):
        print("grain .......... not used: no price is derived, so the"
              " operating point never enters")
    else:
        print(f"grain .......... t = {g['t_lo']:g}"
              + (f":{g['t_hi']:g} (operate at 1/sqrt(t_lo t_hi) = "
                 f"{1/g['t_eff']:.4f})" if g['t_hi'] > g['t_lo'] else
                 "  [declared, not defaulted]"))
    if "grain_delivered_by_retries" in g:
        print(f"  retries ...... budget {g['retry_budget']} delivers grain "
              f"{g['grain_delivered_by_retries']:.3f}, not "
              f"{g['retry_budget']} (the discount D)")

    cg = rep["gates"].get("class")
    if cg:
        print(f"class gate ..... {cg['verdict']}: {cg['reason']}")
        print(f"  divisibility . {cg['divisibility']}; curvature rises on "
              f"{cg['curvature_rises']} step(s) (a rise refutes; no rise "
              f"proves nothing)")
    ni = rep["gates"].get("neff_interval")
    if ni and ni["median"]:
        print(f"precision gate . N_eff = {ni['median']:.3f} "
              f"[{ni['lo5']:.3f}, {ni['hi95']:.3f}] at 90% from n_test alone"
              + (f"; {ni['frac_out_of_class']:.1%} of draws out of class"
                 if ni["frac_out_of_class"] else ""))
    us = rep["gates"].get("undersizing")
    if us:
        print(f"under-sizing ... {us['note']}")

    ca = rep.get("count_axis")
    if ca:
        fb = ca["floor_bracket"]
        print(f"count axis ..... {ca['verdict']} on orders 1-3"
              + (f" (refuted at {ca['refuted_orders']})"
                 if ca["refuted_orders"] else " (exact, signed, any spacing)"))
        print(f"  floor ........ {fb[0]:.5g} <= eps_inf <= {fb[1]:.5g} "
              f"at budget {fb[2]:g} attempts -- two measured numbers, not an "
              f"extrapolation")
        d2 = ca["discount"].get(2)
        if d2:
            print(f"  retry worth .. D(2) = {d2:.3f} (bound 0.5); the second "
                  f"attempt buys {100*(2*d2-1):.0f}% of the first")

    pl = rep.get("plane") or {}
    if pl.get("price") is not None:
        print(f"plane .......... {pl['n_readings']} readings; median N_eff "
              f"{pl['neff_median']:.3f}; implied grain "
              f"[{pl['implied_grain_range'][0]:.2f}, "
              f"{pl['implied_grain_range'][1]:.2f}]")
        print(f"operate at ..... resource {math.exp(pl['operate_x']):.4g} "
              f"(N_eff = {pl['target_neff']:.4f} solved on the plane's own "
              f"line, R^2 = {pl['line_r2']:.3f}; "
              f"{'inside' if pl['inside_window'] else 'EXTRAPOLATED beyond'} "
              f"the measured window)")
        print(f"  nearest ...... the closest measured point costs "
              f"{pl['nearest_reading_regret']:.4f} nats against the rule")
        if pl.get("grain_worst_case"):
            print(f"  grain range .. worst case over the declared interval "
                  f"{pl['grain_worst_case']:.4f} nats")
        print(f"walls .......... eps_inf <= {pl['floor_axiom']:.5g} (axiom), "
              f"<= {pl['floor_divisible']:.5g} (divisible) at the operating "
              f"reading")
        print(f"arc ............ {pl['arc']/math.pi:.3f} pi measured; "
              f"{pl['checkpoints_needed']:.2f} checkpoints' worth of pricing "
              f"information (five cover a complete frontier)")

    pr = rep["prices"]
    if pr.get("lambda"):
        print("price .......... " + ", ".join(
            f"{k} = {v:.4f} nats/e-fold [{pr['provenance'][k].split()[0]}]"
            for k, v in pr["lambda"].items()))
        for ax, u in (pr.get("uncertainty") or {}).items():
            if u.get("sd"):
                print(f"  {ax} ...... 90% [{u['lo5']:.4f}, {u['hi95']:.4f}] "
                      f"from test-set noise alone (sd {u['sd']:.4f}); the "
                      f"plug-in slope reads low, so this price is biased LOW "
                      f"and the operating depth biased DEEP")
        if "interaction" in pr:
            it = pr["interaction"]
            print(f"  interaction .. theta = {it['theta']:g}, canonical depth "
                  f"{it['canonical_depth_nats']:.4f} nats")

    cr = rep["gates"].get("calibration_requirement")
    if cr:
        msg = (f"calibration ..... grid step h = {cr['h']:.3f} in "
               f"log-resource; N_eff needs n_test >= "
               f"{cr['n_test_needed']:,.0f} items at this h")
        if cr.get("widen_grid_by") and cr["widen_grid_by"] > 1.0:
            msg += (f" -- or keep {cr['n_test_have']:,} items and widen the "
                    f"grid by {cr['widen_grid_by']:.1f}x, which is "
                    f"quadratically cheaper")
        print(msg)

    db = rep.get("depth_bounds")
    if db and rep.get("grain", {}).get("used", True):
        print(f"depth left ..... >= {db['axiom']:.4f} nats (axiom), "
              f">= {db['divisible']:.4f} (divisible); gap {db['gap']:.4f}")

    g2 = rep.get("gates", {}).get("class")
    if g2 and g2.get("envelope_threshold") is not None:
        print(f"envelope test .. threshold {g2['envelope_threshold']:.3f} on "
              f"the share of jagged wide triples [{g2['envelope_threshold_source']}];"
              f" measured {g2['share_triples_violating']:.3f}")
    de = rep.get("gates", {}).get("design")
    if de and de.get("h_min"):
        print(f"design rule .... this grid h = "
              f"{de['h_min']/max(de['widen_by'],1e-12):.3f}; a PRICE needs "
              f"h >= {de['h_min_price']:.3f} [{de['verdict_price']}], a "
              f"channel count h >= {de['h_min']:.3f} [{de['verdict']}]")
        if de["verdict_price"] == "INFEASIBLE":
            print(f"  widen by ..... x{de['widen_by_price']:.2f} in log-resource,"
                  f" or buy {de['n_test_min_at_h']:,.0f} items at this grid"
                  f" (the widening is quadratically cheaper)")
    bc = rep.get("prices", {}).get("bias_correction")
    if bc and bc.get("price_corrected"):
        print(f"bias (F16) ..... plug-in {bc['price_h']:.4f}, 2h grid "
              f"{bc['price_2h']:.4f}, Richardson {bc['price_corrected']:.4f};"
              f" estimated h^2 bias {bc['bias_estimate']:+.4f} -> using the"
              f" {bc['recommend']} price")
    pa = rep.get("paired")
    if pa:
        print(f"paired (F11) ... {pa['n_items']:,} per-item outcomes for"
              f" {pa['n_candidates_covered']} candidates;"
              + (f" median failure correlation {pa['median_phi']:.3f},"
                 f" so every band below is the PAIRED one"
                 if pa.get("median_phi") is not None else ""))

    d = rep["decision"]
    ce = rep.get("certified")
    print()
    if ce and ce.get("verdict") != "UNCERTIFIED":
        rng = ce.get("price_range")
        where = "at the price in force"
        if rng:
            ax, (lo_, hi_) = next(iter(rng.items()))
            where = (f"over {ax} in [{lo_:.4f}, {hi_:.4f}] nats/e-fold"
                     f" ({math.log(hi_/lo_):.2f} e-folds of price uncertainty)")
        print(f"CERTIFIED SET .. {{{', '.join(str(x) for x in ce['set'])}}}"
              f"  at {100*(1-ce['alpha']):.0f}% coverage, {where}")
        print(f"  method ....... {ce['method']} at alpha/(K-1) ="
              f" {ce['alpha_pair']:.4f} per comparison")
        print(f"  assumes ...... {ce['assumptions']}")
        print(f"  pick ......... {ce['pick']}  [minimax inside the certified"
              f" set]")
        beaten = ("at every price in the interval" if rng
                  else "at the single price in force")
        over = "over that interval" if rng else "at that price"
        if ce["rejected"]:
            print(f"  excluded ..... {', '.join(str(x) for x in ce['rejected'])}"
                  f"  (each beaten {beaten} by one rival, certified)")
        print(f"  the set can still cost {ce['regret_bound']:.4f} nats against"
              f" the best candidate, {over}")
        if not rng:
            print("  !            no price range was given, so this is a"
                  " statement at ONE price, not over")
            print("               a range.  For a statement that holds at"
                  " every price, read the ENVELOPE")
            print("               and PRICE-FREE blocks, or supply"
                  " --price-interval.")
    elif ce:
        print("CERTIFIED SET .. UNCERTIFIED: no n_test and no --items, so no"
              " set carries coverage.")

    pa2 = rep.get("pareto_certified")
    if pa2 and pa2["dominations"]:
        print()
        print(f"PRICE-FREE ..... {len(pa2['dominations'])} domination(s)"
              f" certified on accuracy alone, valid at EVERY price:")
        for wn, ln_ in pa2["dominations"][:8]:
            print(f"  {wn} beats {ln_}  (no dearer on any axis, and certified"
                  f" no worse)")
        print(f"  survivors .... {', '.join(str(x) for x in pa2['survivors'])}"
              f"   [needs no price, no grain, no sweep]")

    ev = rep.get("envelope")
    if ev:
        print()
        print(f"ENVELOPE ....... who can be optimal at ALL, on {ev['axis']}"
              f" -- {ev['n_winners']} of {ev['n_candidates']} candidates:")
        for nm, lo_, hi_ in ev["pieces"]:
            hi_s = ("infinity" if hi_ > 1e5 else f"{hi_:.4f}")
            print(f"  {str(nm):<16} optimal for price in"
                  f" [{lo_:.4f}, {hi_s}] nats/e-fold")
        if ev["never"]:
            print(f"  never optimal at any price:"
                  f" {', '.join(str(x) for x in ev['never'])}")
            if ev["dominated"]:
                print(f"    dominated .... "
                      f"{', '.join(str(x) for x in ev['dominated'])}"
                      f"  (something is both cheaper and more accurate)")
            if ev["in_a_dent"]:
                print(f"    in a dent .... "
                      f"{', '.join(str(x) for x in ev['in_a_dent'])}"
                      f"  (Pareto-efficient, but NO linear price selects it)")
        print("  this is read off K straight lines sigma_i = S_i - lambda x_i;"
              " no frontier has to be")
        print("  identified first, and nothing here assumes Law I.")

    be = rep.get("break_even")
    if be:
        print()
        print(f"BREAK-EVEN ..... the ladder, rung per envelope breakpoint"
              f" [{be[0].get('method') if be else ''}]:")
        print(f"  {'cheaper':<16} {'dearer':<16} {'break-even':>10}"
              f" {'certified interval':>22} {'undecided':>11}")
        for r in be:
            lo_s = (f"{r['certified_lo']:.4f}" if r["certified_lo"]
                    else "    --")
            hi_s = (f"{r['certified_hi']:.4f}" if r["certified_hi"]
                    else "    --")
            und = (f"{r['undecided_doublings']:.2f} dbl"
                   if r.get("undecided_doublings") else "   --")
            print(f"  {str(r['cheap']):<16} {str(r['dear']):<16}"
                  f" {r['break_even']:>10.4f} {('[' + lo_s + ', ' + hi_s + ']'):>22}"
                  f" {und:>11}")
        print("  below the lower end the dearer model is certified better,"
              " above the upper end the")
        print("  cheaper one is, and in between the items in hand cannot"
              " separate them. No sweep,")
        print("  no grain and no derived price are used anywhere in this"
              " block.")

    rk = rep.get("ranking_certified")
    if rk:
        print()
        print(f"RANKING ........ {rk['n_tiers']} certified tier(s)"
              f" ({rk['method']}, alpha/(K-1) = {rk['alpha_pair']:.4f}):")
        for i, tier in enumerate(rk["tiers"], 1):
            lo_r = min(rk["worst_case_regret"][nm] for nm in tier)
            hi_r = max(rk["worst_case_regret"][nm] for nm in tier)
            print(f"  tier {i} ....... {', '.join(str(x) for x in tier)}"
                  f"   (worst-case regret {lo_r:.4f}-{hi_r:.4f} nats)")
        print("  within a tier no difference is certifiable; between tiers"
              " every difference is.")
        print("  the tiers order certified DOMINANCE, not regret: a lower tier"
              " can carry a")
        print("  smaller worst case and still be certifiably beaten by"
              " something above it.")
        if rk["contradictions"]:
            print(f"  ! contradictory pairs (a certificate must be wrong):"
                  f" {rk['contradictions']}")

    ds = rep.get("design_selection")
    if ds and ds.get("verdict"):
        print()
        print(f"PRICE SHARPNESS  {ds['verdict']}: the derived price spans"
              f" {ds.get('price_interval_doublings', float('nan')):.2f}"
              f" doublings; a candidate reigns for"
              + (f" {ds['median_reign_doublings']:.2f}"
                 if ds.get("median_reign_doublings") else " the whole interval"))
        if ds["verdict"] == "INFEASIBLE":
            print(f"  remedy ....... widen the sweep by"
                  f" x{ds['widen_grid_by']:.1f} in log-resource"
                  + (f" (h = {ds['h_needed']:.3f}, a span of"
                     f" {ds['span_needed']:,.0f}x over five points)"
                     if ds.get("span_needed") else "")
                  + f", or buy {ds['more_items_by']:.0f}x more items")

    sc = rep.get("schedule")
    if sc:
        print()
        print(f"SCHEDULE ....... {sc['n_networks']} network(s) serve the whole"
              f" price interval: {', '.join(str(x) for x in sc['networks'])}")
        print(f"  one instead .. {sc['one_network_pick']}, costing"
              f" {sc['one_network_worst_case']:.4f} nats worst case over the"
              f" interval")

    st = rep.get("stopping")
    if st:
        print()
        wc = st["worst_case"]
        print(f"CANDIDATES ..... worst case still available to an UNSEEN"
              f" network: {wc:.4f} nats"
              f"  [{st['basis']}]")
        print(f"  verdict ...... {'STOP -- the list is finished' if st['stop'] else 'KEEP GOING'}"
              f" at rho_2 = {st['rho']:.4f}")
        if not st["stop"] and st.get("next_log_resource") is not None:
            print(f"  next one at .. log-resource"
                  f" {st['next_log_resource']:.4f}"
                  f" (resource {math.exp(st['next_log_resource']):.4g})"
                  f" -- where the worst case is attained")

    print()
    if rep.get("out_of_class"):
        print("LAW I .......... OUT OF CLASS -- no price may be derived, no"
              " N_eff, no walls, no floor.")
        print(f"  {rep['out_of_class_reason']}")
        ev2 = rep.get("envelope")
        if ev2:
            print(f"  price envelope: "
                  f"{', '.join(str(x) for x in ev2['winners'])}"
                  + (f"; never optimal at any price: "
                     f"{', '.join(str(x) for x in ev2['never'])}"
                     if ev2["never"] else ""))
        else:
            print(f"  Pareto set: {', '.join(rep['pareto'])}"
                  f"  (an over-statement: a median 62.5% of Pareto points are"
                  f" optimal at no price -- see the envelope, F22)")
        print("  The envelope and the ladder above need no price and no"
              " convexity.")
        print("  --price-interval LO:HI narrows the certified set; --passk"
              " buys the floor.")
        for ww in rep["warnings"]:
            print(f"  ! {ww}")
        return
    print(f"BAND (v2) ...... {d['verdict']}; tie set "
          f"{{{', '.join(str(x) for x in d['tie_set'])}}} at band "
          f"{d['effective_band']:.4f} nats"
          + ("" if d["effective_band"] <= d["band"] + 1e-12 else
             f" (widened from {d['band']:.4f} by test-set noise)"))
    print(f"  pick ......... {d['pick']}"
          + (f"  [tie broken on {d['tiebreak']}]"
             if d["verdict"] == "TIE" else ""))
    print("  note ......... an indifference band is not a confidence set."
          " Where the two disagree, the certified set is the one with a"
          " number attached.")
    if d.get("minimax_pick"):
        print(f"  minimax ...... {d['minimax_pick']} minimises worst-case "
              f"regret over the whole price range the frontier supports")
    if d["canonical_pick"]:
        print(f"  accuracy-only  {d['canonical_pick']}"
              + (f"  <-- DISAGREES, costs {d['blind_spot_regret']:.4f} nats"
                 if d["blind_spot"] else "  (agrees)"))
    if d.get("required_n_test"):
        print(f"  to resolve ... " + ", ".join(
            f"{k}: {v:,.0f} items" for k, v in d["required_n_test"].items())
            + ("  [paired]" if d.get("required_n_test_paired")
               else "  [unpaired: supply --items or --rho and these fall]"))
    # ------------------------------------------------------------------
    # THE RANKING TABLE: one object that answers "rank them all by the
    # trade-off".  Tier first, because that is what the data can separate;
    # then sigma at the price in force, which orders inside a tier without
    # claiming the order is evidence.  The envelope column says where each
    # candidate is optimal, so the table doubles as the schedule.
    # ------------------------------------------------------------------
    cset = set(ce["set"]) if ce else set()
    tier_of = {}
    if rk:
        for i, tier in enumerate(rk["tiers"], 1):
            for nm in tier:
                tier_of[nm] = i
    win = {}
    if ev:
        for nm, lo_, hi_ in ev["pieces"]:
            win[nm] = (lo_, hi_)
    ax_show = (rep.get("break_even") or [{}])[0].get("axis") if rep.get(
        "break_even") else None
    if ax_show is None:
        ax_show = next((k for k in d["ranking"][0]["x"]), None)
    print()
    print("RANK ........... tier first (what the data can separate), then"
          " sigma at the price in force")
    hdr = (f"  {'#':>2} {'name':<16}{'eps':>10}"
           f"{(ax_show or 'resource')[:12]:>13}{'sigma':>9}{'tier':>6}"
           f"{'optimal for price in':>24}  cert")
    print(hdr)
    order = sorted(d["ranking"],
                   key=lambda r: (tier_of.get(r["name"], 99), -r["sigma"]))
    for i, r in enumerate(order, 1):
        res = (math.exp(r["x"][ax_show]) if ax_show in r["x"] else
               float("nan"))
        w = win.get(r["name"])
        if w:
            hi_s = "inf" if w[1] > 1e5 else f"{w[1]:.3f}"
            wstr = f"[{w[0]:.3f}, {hi_s}]"
        else:
            wstr = "never optimal"
        print(f"  {i:>2} {str(r['name']):<16}{r['eps']:>10.5f}{res:>13.4g}"
              f"{r['sigma']:>9.4f}{tier_of.get(r['name'], 0):>6}"
              f"{wstr:>24}  {'*' if r['name'] in cset else ''}")
    print("  'cert' marks the certified set: the candidates that cannot be"
          " certifiably excluded")
    print("  at the price interval in force. 'never optimal' is proof, not"
          " noise -- no price selects it.")
    ia = rep.get("inputs")
    if ia:
        print()
        print(f"INPUTS ......... strongest statement these inputs support:"
              f" {ia['level']}")
        for k, st_, why in ia["notes"]:
            print(f"  {k:<20} {st_:<14} {why}")
    for ww in rep["warnings"]:
        print(f"! {ww}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="?",
                    help="measurements.csv, pairs-free aggregates, or omit it "
                         "entirely and pass --model on the command line")
    ap.add_argument("--model", action="append", metavar="N,ACC,RES[,NTEST]",
                    help="F23: one candidate, straight from the command line: "
                         "name, accuracy, the resource, and optionally the "
                         "item count. Repeat once per candidate. Name the "
                         "resource with --axis. No file needed.")
    ap.add_argument("--pair", action="append",
                    metavar="I,J,BOTH,I_ONLY,J_ONLY,NEITHER",
                    help="F23: one pair's 2x2 table from the command line -- "
                         "all the certificate reads. Repeat per pair; with "
                         "them the certificate is exact, without them it is "
                         "the unpaired normal one.")
    ap.add_argument("--grain", default=None,
                    help="t, or t_min:t_max. Required ONLY with "
                         "--calibration, where the price is derived at the "
                         "operating point N_eff = 1/t. Not read by the "
                         "envelope, the ladder, the price-free test or "
                         "--price-interval (F23).")
    ap.add_argument("--calibration",
                    help="one model at 5-7 budgets of one axis")
    ap.add_argument("--calibration-axis", help="which resource column it varies")
    ap.add_argument("--resources", help="comma-separated resource columns to price")
    ap.add_argument("--axis", help="F23: the single resource column the "
                                   "envelope and the ladder are read on, when "
                                   "no price and no sweep are supplied")
    ap.add_argument("--price", action="append",
                    help="axis=lambda, repeatable or comma-separated")
    ap.add_argument("--passk", help="count axis: csv with k,P")
    ap.add_argument("--attempt-budget", type=float,
                    help="Lambda, for the floor bracket P(ceil L)..P(floor L)")
    ap.add_argument("--retry-budget", type=float,
                    help="retries per item; converted to delivered grain")
    ap.add_argument("--theta", type=float, default=1.0,
                    help="power-mean interaction index for n resources")
    ap.add_argument("--floor", type=float,
                    help="eps_inf, if you insist; preferred route is --passk")
    ap.add_argument("--q0", type=float, default=0.0,
                    help="chance level: eps = (1-Q)/(1-q0). A constant shift "
                         "of ln eps, so it cancels from every ranking and "
                         "every derivative; it moves the floor and the depth.")
    ap.add_argument("--q0-convention", default=None,
                    help="F17: how q0 was chosen (e.g. 'uniform chance', "
                         "'majority class', 'declared'); printed with the "
                         "inputs so the choice is on the record")
    ap.add_argument("--tie-band", type=float, default=pc.RHO2,
                    help=f"indifference band in nats (default {pc.RHO2:.5f} = "
                         f"one doubling of psi)")
    ap.add_argument("--pairs",
                    help="F23: the pairwise 2x2 tables instead of the item "
                         "matrix: i,j,both_fail,i_only,j_only,neither. "
                         "K(K-1)/2 rows, and the certificate is identical.")
    ap.add_argument("--items",
                    help="F11/F12: per-item outcomes on the shared test set, "
                         "one column per candidate, 1 = correct. This is what "
                         "turns the verdict into a CERTIFIED set.")
    ap.add_argument("--rho", type=float, default=None,
                    help="F11: declared correlation of the candidates' failure "
                         "indicators, used when --items is not available. "
                         "Measured median on the archive's own panel: 0.50 "
                         "over all pairs, 0.77 for adjacent ones.")
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="F12: 1 - alpha is the guaranteed coverage of the "
                         "certified set (default 0.10, i.e. 90%%)")
    ap.add_argument("--price-interval", metavar="LO:HI",
                    help="F12: the price uncertainty the certified set must "
                         "survive, on the calibration axis. Default: the "
                         "derived price's own 90%% interval.")
    ap.add_argument("--no-bias-correction", action="store_true",
                    help="F16: keep the plug-in price even where the estimated "
                         "O(h^2) bias exceeds its noise")
    ap.add_argument("--force", action="store_true",
                    help="proceed anyway: name a winner even out of class, and "
                         "override the F24 input-consistency gate (neither is "
                         "recommended)")
    ap.add_argument("--json", action="store_true",
                    help="emit the whole report as JSON instead of text")
    a = ap.parse_args(argv)
    a.q0_given = any(str(v).startswith("--q0") for v in (argv or sys.argv[1:]))
    rep = run(a)
    if a.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        show(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())

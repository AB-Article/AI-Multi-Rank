#!/usr/bin/env python3
"""140 -- the precision gate on the whole corpus.  (Fix F3.)

N_eff = s^2/kappa is a square over a second difference.  Its signal is h^2 kappa
and its noise is sqrt(6) sd(ln eps), so there are exactly two ways to make it
mean anything: buy test items, or WIDEN THE GRID.  Because the signal grows as
h^2 while the noise does not, widening is quadratically cheaper -- a factor f
in grid width is worth a factor f^4 in items.

This script computes, for every archived frontier,

  * sd(ln eps) implied by the evaluation set it was measured on,
  * the n_test that frontier's own grid would need for sd(kappa)/kappa <= 1/4,
  * the factor by which its grid would have to be stretched to get there at the
    items it already has,
  * whether a one-doubling gap (RHO2 = 0.0589 nats) between adjacent Pareto
    models is resolvable at all at that precision.

Writes results/table_precision_gate.csv and results/table_precision_gate_summary.csv.
No network, no fit, no bandwidth.
"""
import math
import _pipeline as P

pc = P.pc


def main():
    rows = []
    for f in P.frontiers():
        xs = list(f["x"])
        es = list(f["e"])
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        es = [es[i] for i in order]
        ls = [math.log(e) for e in es]
        n_items = P.eval_items(f["name"])
        eps_mid = sorted(es)[len(es) // 2]
        q_mid = 1.0 - eps_mid

        req = pc.calibration_requirement(xs, ls, q_mid)
        sd = pc.sd_of_l(q_mid, n_items) if n_items else None
        need = pc.required_n_test(q_mid, pc.RHO2)

        # adjacent-pair resolvability: is a RHO2 gap bigger than its own noise?
        gaps = [abs(ls[i] - ls[i + 1]) for i in range(len(ls) - 1)]
        med_gap = sorted(gaps)[len(gaps) // 2] if gaps else None
        resolvable = None
        if sd and med_gap is not None:
            resolvable = bool(med_gap > pc.Z95 * math.sqrt(2.0) * sd)

        r = {"corpus": f["corpus"], "frontier": f["name"],
             "n_points": len(xs), "eps_median": eps_mid,
             "n_test": n_items, "sd_l": sd,
             "grid_h": req["h"] if req else None,
             "kappa_mid": req["kappa"] if req else None,
             "n_test_needed_for_Neff": req["n_test_needed"] if req else None,
             "n_test_needed_for_rho2_gap": need,
             "median_adjacent_gap_nats": med_gap,
             "rho2_gap_resolvable": resolvable}
        if req and req.get("n_test_needed") and n_items:
            r["widen_grid_by"] = math.sqrt(req["n_test_needed"] / n_items)
            r["Neff_resolved_now"] = bool(n_items >= req["n_test_needed"])
        else:
            r["widen_grid_by"] = None
            r["Neff_resolved_now"] = None
        rows.append(r)

    df = P.save(rows, "table_precision_gate.csv")

    known = df[df.n_test.notna()]
    print()
    print(f"  frontiers with a known evaluation-set size : {len(known)} of {len(df)}")
    if len(known):
        print(f"  median sd(ln eps)                          : "
              f"{known.sd_l.median()*1e3:.2f} milli-nats")
        print(f"  the paper's required precision             : 0.73 milli-nats")
        print(f"  N_eff resolved at the items they have      : "
              f"{int(known.Neff_resolved_now.sum())} of {len(known)}")
        print(f"  median items needed for N_eff              : "
              f"{known.n_test_needed_for_Neff.median():,.0f}")
        print(f"  median grid-widening factor instead        : "
              f"{known.widen_grid_by.median():.1f}x"
              f"  (cost in items scales as the 4th power, so this is the"
              f" cheap route)")
        print(f"  a {pc.RHO2:.4f}-nat gap is resolvable on        : "
              f"{int(known.rho2_gap_resolvable.sum())} of {len(known)}")
    print(f"  median items a {pc.RHO2:.4f}-nat decision needs : "
          f"{df.n_test_needed_for_rho2_gap.median():,.0f}")

    summ = []
    for corpus, d in df.groupby("corpus"):
        summ.append({
            "corpus": corpus, "n_frontiers": len(d),
            "median_grid_h": float(d.grid_h.median()),
            "median_n_test_needed_for_Neff": float(
                d.n_test_needed_for_Neff.median()),
            "median_n_test_needed_for_rho2_gap": float(
                d.n_test_needed_for_rho2_gap.median()),
            "median_widen_grid_by": (float(d.widen_grid_by.median())
                                     if d.widen_grid_by.notna().any() else None),
        })
    sdf = P.save(summ, "table_precision_gate_summary.csv")
    print()
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()

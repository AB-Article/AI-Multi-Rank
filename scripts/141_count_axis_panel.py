#!/usr/bin/env python3
"""141 -- the count axis, measured.  (Fixes F6, F10, and the floor.)

The paper's own conclusion is that the remedy for everything the compute axis
cannot decide is "a wider item census on a count axis, not a deeper corpus of
compute frontiers": on a count axis the axiom is a THEOREM of exchangeability
(Prop. countaxis), the complete-monotonicity test is exact and signed at every
spacing, and the floor is P(Lambda) -- a measured number -- rather than an
extrapolation to x = infinity.

This script builds that count axis from the archive's own per-item panel, with
no new data and no network.  The panel is 10,000 Fashion-MNIST items x 21
models x 2 families.  Drawing k of the M models without replacement, the
probability that an item is failed by all k is exactly C(f,k)/C(M,k) where f is
that item's failure count, so

    P(k) = mean_i C(f_i, k) / C(M, k),      k = 1..M

is available in closed form -- no simulation, no fit.  P is then tested for
complete monotonicity (Hausdorff: (-1)^j Delta^j P >= 0 for every j), read for
N_eff, and used to bracket the floor and to measure the retry discount.

Writes
  results/table_count_axis_panel.csv        per (family, k)
  results/table_count_axis_summary.csv      per family
  data/cofailure/passk_<family>.csv         the count axis itself, for reuse
  examples/model_selection/passk_fmnist_nested.csv
No network.
"""
import math
import os

import numpy as np

import _pipeline as P

pc = P.pc
CO = os.path.join(P.DATA, "cofailure")


def passk(correct):
    """P(k) = Pr[an item is failed by all of k models drawn without
    replacement], exactly, from the per-item failure counts."""
    fails = (~correct).sum(axis=1)                 # f_i, 0..M
    M = correct.shape[1]
    out = []
    for k in range(1, M + 1):
        num = np.array([math.comb(int(f), k) for f in fails], dtype=float)
        out.append(float(num.mean() / math.comb(M, k)))
    return out


def main():
    rows, summ = [], []
    for fam in ("nested", "indep"):
        path = os.path.join(CO, f"fmnist_{fam}_correct.npy")
        if not os.path.exists(path):
            print(f"  (skipping {fam}: {path} not present)")
            continue
        correct = np.load(path)
        n_items, M = correct.shape
        Pk = passk(correct)
        rep = pc.count_axis_read(Pk, budget=M)

        # the independence prediction, for the co-failure comparison: if models
        # failed an item independently given its own rate, P(k) would be
        # E[h^k] with h the per-item failure share, which is what we have --
        # so the comparison that matters is against the ONE-RATE model,
        # P_indep(k) = (P(1))^k, the law a single global failure rate implies.
        P1 = Pk[0]
        for k in range(1, M + 1):
            r = {"family": fam, "n_items": n_items, "n_models": M, "k": k,
                 "P": Pk[k - 1],
                 "P_one_rate": P1 ** k,
                 "excess_over_one_rate": Pk[k - 1] / (P1 ** k),
                 "S": -math.log(Pk[k - 1]),
                 "D": rep["discount"][k],
                 "grain_delivered": rep["grain_of_retry"][k],
                 "neff_count": rep["neff"].get(k)}
            for j in (1, 2, 3):
                v = rep["alternation"].get(j)
                r[f"alt{j}"] = v[k - 1] if v and k - 1 < len(v) else None
            rows.append(r)

        ns = [v for v in rep["neff"].values() if v]
        fb = rep["floor_bracket"]
        summ.append({
            "family": fam, "n_items": n_items, "n_models": M,
            "complete_monotone": rep["verdict"],
            "refuted_orders": ",".join(str(o) for o in
                                       rep["alternation"]["refuted_orders"]),
            "P1": Pk[0], "P2": Pk[1], "P_M": Pk[-1],
            "floor_lo": fb[0], "floor_hi": fb[1], "attempt_budget": fb[2],
            "floor_as_share_of_P1": fb[0] / Pk[0],
            "D2": rep["discount"][2], "D2_bound": 0.5,
            "second_attempt_worth": 2 * rep["discount"][2] - 1,
            "neff_count_median": (sorted(ns)[len(ns) // 2] if ns else None),
            "neff_count_min": min(ns) if ns else None,
            "neff_count_max": max(ns) if ns else None,
            "grain_of_2_retries": rep["grain_of_retry"][2],
            "grain_of_4_retries": rep["grain_of_retry"][min(4, M)],
        })

        # the count axis itself, for the selector and for reuse
        os.makedirs(CO, exist_ok=True)
        with open(os.path.join(CO, f"passk_{fam}.csv"), "w") as fh:
            fh.write("# P(k): probability an item is still failed after k\n"
                     "# independent attempts (k models drawn without\n"
                     f"# replacement from the {fam} family of the co-failure\n"
                     f"# panel, {n_items} items, exact from the failure counts).\n")
            fh.write("k,P\n")
            for k, p in enumerate(Pk, 1):
                fh.write(f"{k},{p:.10f}\n")
        print(f"  -> data/cofailure/passk_{fam}.csv")

    df = P.save(rows, "table_count_axis_panel.csv")
    sdf = P.save(summ, "table_count_axis_summary.csv")

    # ship the nested family's axis with the worked example
    src = os.path.join(CO, "passk_nested.csv")
    if os.path.exists(src):
        dst = os.path.join(P.EX, "passk_fmnist_nested.csv")
        with open(src) as a, open(dst, "w") as b:
            b.write(a.read())
        print(f"  -> examples/model_selection/passk_fmnist_nested.csv")

    print()
    print(sdf[["family", "complete_monotone", "refuted_orders", "P1",
               "floor_lo", "floor_hi", "D2", "second_attempt_worth",
               "neff_count_median"]].to_string(index=False))
    print()
    for _, r in sdf.iterrows():
        print(f"  {r['family']}: complete monotonicity {r['complete_monotone']}"
              f" on orders 1-3 (exact and signed at every spacing -- on the"
              f" compute axis no bandwidth-free statistic reaches power 0.5)")
        print(f"    floor      {r['floor_lo']:.5g} <= eps_inf <= "
              f"{r['floor_hi']:.5g} at {r['attempt_budget']:.0f} attempts, i.e."
              f" {r['floor_as_share_of_P1']:.1%} of the one-attempt error --"
              f" two measured numbers, not an extrapolation")
        print(f"    retries    D(2) = {r['D2']:.3f} against the bound 0.5: the"
              f" second attempt buys {100*r['second_attempt_worth']:.0f}% of"
              f" the first, so a budget of 2 retries delivers grain"
              f" {r['grain_of_2_retries']:.3f}, not 2")
        print(f"    channels   N_eff on the count axis, median "
              f"{r['neff_count_median']:.3f}"
              f"  (the task's own, not a lower bound)")


if __name__ == "__main__":
    main()

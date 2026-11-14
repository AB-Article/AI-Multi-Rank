#!/usr/bin/env python3
"""145 -- the band is a PAIRED quantity.  (Fix F11.)

Candidates are evaluated on the same test items, so the gap that a decision
turns on,  l_i - l_j  with  l = ln eps,  has variance

    V1/n,  V1 = (1-e_i)/e_i + (1-e_j)/e_j - 2 (p_ij - e_i e_j)/(e_i e_j),

and not the sum of two independent variances.  Version 2 of the selector used
the unpaired sum (``sd_i^2 + sd_best^2``), and the matching factor 2 in the
required item count.  Failures on shared items are positively associated -- the
archive's own panel measures Pr[both fail] = kappa min(h_i,h_j) with
kappa = 0.835 -- so the unpaired band is too WIDE and the item count it demands
too LARGE, by a factor this script measures.

It is measured, not assumed: the co-failure panel is 10,000 items x 21 models x
2 families of per-item correctness, so every pair's p_ij is a count.

Nothing new has to be collected in a real pipeline either: the per-item
correctness vector is already computed by whatever produced the accuracy.

Writes results/table_paired_band_pairs.csv and
results/table_paired_band_summary.csv.
"""
import itertools
import math
import os

import numpy as np

import _pipeline as P

pc = P.pc
FAM = ("nested", "indep")


def main():
    rows = []
    for fam in FAM:
        A = np.load(os.path.join(P.DATA, "cofailure",
                                 f"fmnist_{fam}_correct.npy"))
        fail = ~A                                   # True = failed this item
        n, M = fail.shape
        e = fail.mean(axis=0)
        for i, j in itertools.combinations(range(M), 2):
            fi, fj = fail[:, i], fail[:, j]
            e_i, e_j = float(e[i]), float(e[j])
            if not (0.0 < e_i < 1.0 and 0.0 < e_j < 1.0):
                continue
            p_both = float((fi & fj).mean())
            phi = pc.phi_correlation(e_i, e_j, p_both)
            sd_pair = pc.sd_of_gap(e_i, e_j, n, p_both=p_both)
            sd_unp = pc.sd_of_gap(e_i, e_j, n)
            n_pair = pc.required_n_gap(e_i, e_j, pc.RHO2, p_both=p_both)
            n_unp = pc.required_n_gap(e_i, e_j, pc.RHO2)
            gap = abs(math.log(e_j) - math.log(e_i))
            rows.append({
                "family": fam, "i": i, "j": j, "n_items": n,
                "e_i": e_i, "e_j": e_j, "p_both": p_both,
                "p_indep": e_i * e_j, "phi": phi,
                "gap_nats": gap,
                "sd_paired": sd_pair, "sd_unpaired": sd_unp,
                "sd_ratio": (sd_pair / sd_unp) if sd_unp > 0 else None,
                "n_needed_paired": n_pair, "n_needed_unpaired": n_unp,
                "n_ratio": (n_pair / n_unp) if n_unp > 0 else None,
                # a decision at the achieved items: is the gap resolved?
                "resolved_paired": bool(gap > pc.Z95 * sd_pair),
                "resolved_unpaired": bool(gap > pc.Z95 * sd_unp),
                "adjacent": bool(abs(i - j) == 1),
            })
    df = P.save(rows, "table_paired_band_pairs.csv")

    summ = []
    for tag, d in [("all", df)] + [(f, df[df.family == f]) for f in FAM] + \
            [("adjacent only", df[df.adjacent])]:
        gained = d[d.resolved_paired & ~d.resolved_unpaired]
        summ.append({
            "subset": tag, "n_pairs": len(d),
            "median_phi": float(d.phi.median()),
            "median_cofailure_excess": float(
                ((d.p_both - d.p_indep) / d.p_indep).median()),
            "median_sd_ratio": float(d.sd_ratio.median()),
            "median_n_ratio": float(d.n_ratio.median()),
            "resolved_unpaired": float(d.resolved_unpaired.mean()),
            "resolved_paired": float(d.resolved_paired.mean()),
            "pairs_newly_resolved": len(gained),
            "share_newly_resolved": len(gained) / float(max(len(d), 1)),
        })
    sdf = P.save(summ, "table_paired_band_summary.csv")
    print()
    print("  the gap between two candidates is PAIRED: same items, correlated"
          " failures.")
    print(sdf.to_string(index=False))
    print()
    a = df[df.family == "nested"]
    b = df[df.family == "indep"]
    print(f"  median failure correlation phi : nested {a.phi.median():.3f},"
          f" independent-draw {b.phi.median():.3f}")
    print(f"  median band shrinkage          : x{df.sd_ratio.median():.3f}"
          f"  (the unpaired band is {1.0/df.sd_ratio.median():.2f}x too wide)")
    print(f"  median item-count saving       : x{df.n_ratio.median():.3f}"
          f"  (the unpaired n_test is {1.0/df.n_ratio.median():.2f}x too large)")
    print(f"  comparisons resolved at the items in hand:"
          f" {df.resolved_unpaired.mean():.1%} unpaired ->"
          f" {df.resolved_paired.mean():.1%} paired"
          f"  ({int((df.resolved_paired & ~df.resolved_unpaired).sum())} of"
          f" {len(df)} pairs newly decided, at no extra cost)")
    print()
    print("  This is the cheapest correctness gain in the pipeline: it uses the")
    print("  per-item vector that already exists and no new input parameter.")


if __name__ == "__main__":
    main()

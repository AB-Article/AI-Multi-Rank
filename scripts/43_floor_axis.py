#!/usr/bin/env python3
"""T6c  The fitted floor is not a property of the evaluation set alone.

Proposition 'floor bound' bounds einf below by the chance-corrected Bayes
error, which is a property of the evaluation set.  If the fitted floor
estimated that quantity, fitting it along different resource axes on one
evaluation set would return one number.  This script tests that directly.

The test needs no sub-windows and no external reference, and it is sharper
than the nested-window drift of script 22 in one respect: the sharp window
bound eps(x_0+X) of Proposition 'running floor'(ii) is IDENTICAL across the
axes of one evaluation set, because the most accurate model is the same one,
so any spread between axes is a property of the fit and not of the window.

Writes results/table_floor_axis.csv.
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from _loaders import save, RES
DATA = os.path.join(os.path.dirname(RES), "data")

# Northcutt et al. (2021) audit 6% of ImageNet validation labels as erroneous.
# That is not itself a lower bound on the Bayes error (some label errors are
# predictable from the image), so it is used only as a plausibility anchor.
LABEL_ERROR = 0.06
# The audit covers the ImageNet validation images, so the ratio is quoted only
# for the evaluation sets built on them; ImageNet-V2 has its own labels.
AUDITED = ("ImageNet-1k", "ImageNet-ReaL")


def main():
    # 11_selection_rules.py writes this into results/; older copies of the
    # archive shipped it under data/.  Try both, as 23 and 26 do.
    src = os.path.join(RES, "table_floor_label_noise.csv")
    if not os.path.exists(src):
        src = os.path.join(DATA, "table_floor_label_noise.csv")
    F = pd.read_csv(src)
    rows = []
    for ev, g in F.groupby("eval_set"):
        v = g.eps_inf.to_numpy(float)
        lo, hi = float(g.ci_lo.max()), float(g.ci_hi.min())
        rows.append(dict(
            eval_set=ev, axes=len(g),
            eps_min=float(g.eps_min.iloc[0]),            # same for every axis
            eps_min_spread=float(np.ptp(g.eps_min)),     # exactly 0 if so
            floor_min=float(v.min()), floor_max=float(v.max()),
            floor_spread_ratio=float(v.max() / v.min()),
            common_ci_lo=lo, common_ci_hi=hi,
            axis_invariant=bool(lo <= hi),
            floor_over_label_error_min=float(v.min() / LABEL_ERROR),
            floor_over_label_error_max=float(v.max() / LABEL_ERROR),
            bound_respected=bool(v.min() >= LABEL_ERROR)))
    D = pd.DataFrame(rows)
    save(D, "table_floor_axis.csv")
    print("\nIf einf estimated the Bayes error, every axis of one evaluation set")
    print("would return the same floor.  The sharp window bound eps(x_0+X) does.\n")
    print(D[["eval_set", "axes", "eps_min", "eps_min_spread", "floor_min", "floor_max",
             "floor_spread_ratio", "axis_invariant"]].round(4).to_string(index=False))
    a = D[D.eval_set.isin(AUDITED)]
    print(f"\nall floors above the {100*LABEL_ERROR:.0f}% label-error anchor: "
          f"{bool(D.bound_respected.all())}")
    print(f"ratio to the anchor on the audited sets {AUDITED}: "
          f"{a.floor_over_label_error_min.min():.2f}x to "
          f"{a.floor_over_label_error_max.max():.2f}x")
    print(f"axis-invariance holds on {int(D.axis_invariant.sum())}/{len(D)} evaluation sets")


if __name__ == "__main__":
    main()

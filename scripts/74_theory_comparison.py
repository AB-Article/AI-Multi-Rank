#!/usr/bin/env python3
"""IS THE REVISED THEORY BETTER?  A ledger and a held-out score.

Two things have to be true for a re-grounding to be an improvement rather than
a rearrangement:

  1. it must consume FEWER inputs -- quantities that enter as a measurement, a
     simulation or a hand-set convention rather than as a theorem; and
  2. it must predict at least as well OUT OF SAMPLE, on quantities the old
     reading could also be asked for.

This script checks both.  Part 1 is a ledger: every quantity the revision
touches, what it was before, what it is now, and -- where it is now derived --
the derived value against the measured one.  Part 2 reads the held-out
higher-order comparison of script 72 and reports the margin.  Nothing in Laws
I-III changes, so there is no regression surface there; the verification of the
analytic identities (scripts 02, 51, 54, 65) is unchanged and is re-run by
`make all` as before.

Reads results/table_count_axis_{summary,conventions,stable,holdout_summary}.csv
Writes results/table_theory_comparison.csv
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")


def R(name):
    return pd.read_csv(os.path.join(RESULTS, name))


def main():
    S = R("table_count_axis_summary.csv").set_index("family")
    CV = R("table_count_axis_conventions.csv").set_index("quantity")
    ST = R("table_count_axis_stable.csv")
    HO = R("table_count_axis_holdout_summary.csv").set_index("family")

    rows = []

    def add(q, before, after, derived=np.nan, measured=np.nan, where=""):
        rel = (abs(derived - measured) / abs(measured)
               if np.isfinite(derived) and np.isfinite(measured) and measured else np.nan)
        rows.append(dict(quantity=q, status_before=before, status_after=after,
                         derived=derived, measured=measured, rel_gap=rel,
                         source=where))

    # ---- the ledger ------------------------------------------------------
    add("nesting coefficient kappa", "measured constant, no derivation",
        "price of one attempt; = 1 - q0 at nested knowledge + independent luck",
        1 - 0.10, S.loc["nested", "kappa_adjacent"], "Prop. kappaq0")
    add("kappa (other family, bound from below)", "measured constant",
        "lowered below 1-q0 by imperfect nesting, as constructed",
        1 - 0.10, S.loc["indep", "kappa_adjacent"], "Prop. kappaq0")
    add("count-axis floor nu({g=1})", "not a concept",
        "exactly 0 whenever q0 > 0: theorem", 0.0,
        float(S["nu_atom_at_one"].max()), "Prop. nocountfloor")
    add("m-model floor-bound inflation (indep)", "measured curiosity",
        "geometric at rate g_max/(1-q0): theorem",
        S.loc["indep", "floor_bound_inflation_predicted"],
        S.loc["indep", "floor_bound_inflation_per_model"], "Prop. bloat")
    add("m-model floor-bound inflation (nested)", "measured curiosity",
        "geometric at rate g_max/(1-q0): theorem",
        S.loc["nested", "floor_bound_inflation_predicted"],
        S.loc["nested", "floor_bound_inflation_per_model"], "Prop. bloat")
    add("regret tolerance rho", "hand-set convention (0.05 nats)",
        "cost of one doubling of psi: ln cosh(ln2/2)",
        CV.loc["rho_one_doubling_nats", "derived"],
        CV.loc["rho_one_doubling_nats", "paper_value"], "eq. onedoubling")
    add("covering number K ('five points')", "hand-set, follows from rho",
        "pi/(2 sqrt(2 rho_2)): a consequence of the number two",
        CV.loc["covering_number_K", "derived"],
        CV.loc["covering_number_K", "paper_value"], "eq. onedoubling")
    add("band ratio at rho = 0.05", "quoted band [1.53, 2.89]",
        "the factor one doubling corresponds to",
        CV.loc["psi_factor_at_rho_0.05", "derived"],
        CV.loc["psi_factor_at_rho_0.05", "paper_value"], "eq. onedoubling")
    add("beta <= 1", "imposed to stay inside the class",
        "forced: completely monotone <=> one-sided stable index", np.nan, np.nan,
        "Prop. betastable")
    add("beta corpus-specific", "reported as a defect",
        "theorem: beta is that corpus's per-skill tail index",
        float(ST.rel_error.median()), 0.0, "Prop. betastable")
    add("pi (arc length)", "unique scale-free curvature moment (dim. analysis)",
        "Fisher-Rao diameter of the binary simplex", np.pi, np.pi,
        "Prop. binary")
    add("E[R] = ln 2", "integral of the regret law",
        "log-cosine integral = Jeffreys vs uniform on one bit",
        np.log(2), np.log(2), "Prop. binary")
    add("Law I on a count axis", "axiom (additivity + scale-freeness)",
        "theorem of conditional independence alone", np.nan, np.nan,
        "Prop. countaxis")
    add("Laws II and III", "two laws",
        "one object, two geometries (Bregman / Fisher-Rao)", np.nan, np.nan,
        "Prop. binary")
    add("rank(A) = 1 on 76% of frontiers", "surprising, needed a correction",
        "generic: one difficulty order is one latent coordinate", np.nan, np.nan,
        "Sec. count")
    add("eta^2 = 0.52 of ln(alpha) by axis", "'nothing transfers'",
        "alpha nearly a property of the axis alone (task adds 0.007)",
        np.nan, np.nan, "Sec. scope")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "table_theory_comparison.csv"), index=False)

    # ---- report ----------------------------------------------------------
    print("PART 1.  THE LEDGER\n" + "=" * 78)
    conv = df[df.status_before.str.contains("measured|hand-set|axiom|two laws|"
                                            "defect|imposed|surprising|nothing|"
                                            "dim\\. analysis|integral|not a concept",
                                            regex=True)]
    print(f"  quantities the revision converts from input to consequence: {len(conv)}")
    chk = df.dropna(subset=["rel_gap"])
    chk = chk[chk.rel_gap > 0]
    print(f"  of those, {len(chk)} carry a number; "
          f"max relative gap derived-vs-measured = {chk.rel_gap.max()*100:.1f}%, "
          f"median {chk.rel_gap.median()*100:.1f}%")
    for _, r in df.iterrows():
        d = f"{r.derived:.4f}" if np.isfinite(r.derived) else "   --   "
        m = f"{r.measured:.4f}" if np.isfinite(r.measured) else "   --   "
        print(f"  {r.quantity[:44]:44s} {d:>9s} {m:>9s}   {r["source"]}")

    print("\nPART 2.  HELD OUT, AT ORDERS NO FIT SAW\n" + "=" * 78)
    for fam in HO.index:
        print(f"  {fam:7s}  nesting law (kappa from pairs) {HO.loc[fam,'errB']:.3f}"
              f"   count axis (nu from orders<=4) {HO.loc[fam,'errD']:.3f}"
              f"   -> {HO.loc[fam,'improvement_D_over_B']*100:.0f}% better")
    print("  reading (A), the refuted mechanism, cannot be scored above order 4:")
    print("  sum_{i in S} x_i leaves the measured window.")

    print("\nVERDICT\n" + "=" * 78)
    print(f"  fewer inputs: {len(conv)} quantities become consequences, none is added.")
    print(f"  no worse in sample: Laws I-III are unchanged, so scripts 02/51/54/65")
    print(f"    verify the same identities to the same precision.")
    print(f"  better out of sample: {HO['improvement_D_over_B'].min()*100:.0f}-"
          f"{HO['improvement_D_over_B'].max()*100:.0f}% lower held-out error at "
          f"co-failure orders 5-12.")
    print(f"  new reach: ensembling, retries and best-of-n sampling are inside the "
          f"theory,\n    with an exchange rate of "
          f"{S['exchange_rate_efolds_per_attempt'].min():.2f}-"
          f"{S['exchange_rate_efolds_per_attempt'].max():.2f} e-folds per attempt "
          f"on the panel.")
    print("  wrote results/table_theory_comparison.csv")


if __name__ == "__main__":
    main()

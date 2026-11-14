#!/usr/bin/env python3
"""DOES THE EXPONENT TRANSFER?  (Section 7 of paper.tex, the negative result.)

The exchange-rate test of Proposition 13 needs one number, alpha, and nothing
else about the frontier.  If alpha were a property of the resource -- which is
what the reciprocity of eq. (26) would suggest, since theta_t A_t is a
property of the resources and alpha_{t,i} = A_t / b_i -- then a benchmark
would not have to be fitted at all: a per-resource constant would decide any
two candidates from two accuracies and two resource values.

This script tests that.  It decomposes the variance of the fitted exponent into
a between-resource-axis and a within-axis part, and it measures what a borrowed
alpha actually costs when the borrowing is honest (other corpora) against when
it is not (other cells of the same corpus, which share a model zoo and
overlapping evaluation sets).

Writes (results/):
    table_exponent_transfer.csv        variance decomposition and transfer error
    table_exponent_by_axis.csv         fitted alpha per resource axis
"""
import os, sys, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import save, ROOT  # noqa: E402

PAIRS = os.path.join(ROOT, "results", "table_exchange_rate_pairs.csv")


def main():
    if not os.path.exists(PAIRS):
        raise SystemExit("run 47_exchange_rate.py first")
    d = pd.read_csv(PAIRS)
    A = d.groupby(["axis", "corpus", "cell"]).alpha_fit.first().reset_index()

    # --- variance decomposition ---------------------------------------------
    tot = float(A.alpha_fit.var(ddof=1))
    grand = float(A.alpha_fit.mean())
    gm = A.groupby("axis").alpha_fit.agg(["count", "mean", "median", "std"])
    within_ss = float(sum(((A.alpha_fit[A.axis == k] - gm.loc[k, "mean"]) ** 2).sum()
                          for k in gm.index))
    between_ss = float(sum(gm.loc[k, "count"] * (gm.loc[k, "mean"] - grand) ** 2
                           for k in gm.index))
    eta2 = between_ss / (between_ss + within_ss)
    print(f"  alpha over {len(A)} cells: total var {tot:.4f}")
    print(f"  between-axis share of variance (eta^2) = {eta2:.3f}")
    save(gm.reset_index().round(4), "table_exponent_by_axis.csv")
    print(gm.round(3).to_string())

    # --- what a borrowed alpha costs ----------------------------------------
    rows = [dict(quantity="cells", value=float(len(A))),
            dict(quantity="pairs", value=float(len(d))),
            dict(quantity="alpha_median", value=float(A.alpha_fit.median())),
            dict(quantity="alpha_min", value=float(A.alpha_fit.min())),
            dict(quantity="alpha_max", value=float(A.alpha_fit.max())),
            dict(quantity="var_total", value=tot),
            dict(quantity="between_axis_eta2", value=eta2),
            dict(quantity="within_axis_var_mean",
                 value=float(np.mean([A.alpha_fit[A.axis == k].var(ddof=1)
                                      for k in gm.index if gm.loc[k, "count"] > 1]))) ]

    def rel_err(cross_corpus):
        """Relative error of a borrowed alpha, cell by cell."""
        out = []
        for _, r in A.iterrows():
            v = A[(A.axis == r.axis) & (A.cell != r.cell)]
            if cross_corpus:
                v = v[v.corpus != r.corpus]
            if not len(v):
                continue
            out.append(abs(np.median(v.alpha_fit.values) - r.alpha_fit) / r.alpha_fit)
        return np.array(out)

    for nm, cc in (("same_axis_any_corpus", False), ("same_axis_other_corpora", True)):
        e = rel_err(cc)
        rows += [dict(quantity=f"borrowed_alpha_rel_err_median[{nm}]", value=float(np.median(e))),
                 dict(quantity=f"borrowed_alpha_rel_err_p90[{nm}]", value=float(np.percentile(e, 90))),
                 dict(quantity=f"borrowed_alpha_within_factor2[{nm}]",
                      value=float(np.mean(e < 1.0))),
                 dict(quantity=f"borrowed_alpha_n_cells[{nm}]", value=float(len(e)))]

    # how often does a factor-two bracket suffice, i.e. is alpha in [sbar, 2 sbar)?
    rows.append(dict(quantity="pairs_needing_factor2_resolution",
                     value=float(np.mean((d.alpha_fit >= d.sbar) & (d.alpha_fit < 2 * d.sbar)))))
    rows.append(dict(quantity="median_alpha_over_sbar",
                     value=float(np.median(d.alpha_fit / d.sbar))))
    T = pd.DataFrame(rows)
    save(T.round(6), "table_exponent_transfer.csv")
    print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

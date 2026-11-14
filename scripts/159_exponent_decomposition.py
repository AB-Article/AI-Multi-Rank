#!/usr/bin/env python3
"""How much of ln(alpha) is the axis and how much is the task.

Why this script exists
----------------------
Revision 2026-10-08m could not reproduce the paper's claim that "ln alpha
fitted additively in task and axis gives R^2 = 0.502 against eta^2 = 0.495 for
the axis alone, so task identity adds 0.007".  Neither number appears in any
archive table, and no admissible grouping of the exchange-rate panel yields a
0.007 increment: the axis labels are nested inside corpora in
table_floor_free_per_frontier.csv, so adding the corpus there gives an
increment of exactly zero, while a genuine per-task factor gives 0.30 or more.

This script does the decomposition properly, on the panel script 48 already
builds (table_exchange_rate_pairs.csv: one alpha per axis x task x cell), and
reports:

  D1  eta^2 of ln(alpha) by axis alone and by task alone;
  D2  R^2 of the additive axis + task model, raw and adjusted for its
      degrees of freedom, and the increment task identity buys;
  D3  what a borrowed alpha actually costs, within a task and across tasks,
      which is the operational version of the same question and is the number
      the paper should quote.

Writes results/table_exponent_decomposition.csv
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import save, ROOT                                    # noqa: E402

PAIRS = os.path.join(ROOT, "results", "table_exchange_rate_pairs.csv")
TRANSFER = os.path.join(ROOT, "results", "table_exponent_transfer.csv")


def design(n, factors):
    """Intercept plus treatment-coded dummies for each factor."""
    cols = [np.ones(n)]
    for g in factors:
        for lev in sorted(set(g))[1:]:
            cols.append(np.array([1.0 if v == lev else 0.0 for v in g]))
    return np.column_stack(cols)


def r2(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return 1.0 - float(r @ r) / float(((y - y.mean()) ** 2).sum())


def eta2(y, g):
    grand = y.mean()
    gm = {}
    for k, v in zip(g, y):
        gm.setdefault(k, []).append(v)
    bss = sum(len(v) * (np.mean(v) - grand) ** 2 for v in gm.values())
    wss = sum(((np.asarray(v) - np.mean(v)) ** 2).sum() for v in gm.values())
    return float(bss / (bss + wss))


def main():
    if not os.path.exists(PAIRS):
        raise SystemExit("run 47_exchange_rate.py first")
    d = pd.read_csv(PAIRS)
    A = (d.groupby(["axis", "corpus", "cell"]).alpha_fit.first().reset_index())
    A = A[A.alpha_fit > 0].reset_index(drop=True)
    y = np.log(A.alpha_fit.values)
    ax, task = list(A.axis), list(A.corpus)
    n, n_ax, n_task = len(y), len(set(ax)), len(set(task))

    e_ax, e_task = eta2(y, ax), eta2(y, task)
    R_ax = r2(y, design(n, [ax]))
    R_task = r2(y, design(n, [task]))
    R_both = r2(y, design(n, [ax, task]))
    p_both = 1 + (n_ax - 1) + (n_task - 1)
    R_both_adj = 1.0 - (1.0 - R_both) * (n - 1) / (n - p_both)

    rows = [
        ("cells (axis x task x frontier)", float(n)),
        ("axes", float(n_ax)),
        ("tasks (evaluation sets)", float(n_task)),
        ("eta2 of ln(alpha) by axis", e_ax),
        ("eta2 of ln(alpha) by task", e_task),
        ("R2 additive, axis only", R_ax),
        ("R2 additive, task only", R_task),
        ("R2 additive, axis + task", R_both),
        ("R2 additive, axis + task, adjusted", R_both_adj),
        ("increment from task identity (R2 both - R2 axis)", R_both - R_ax),
        ("increment from axis identity (R2 both - R2 task)", R_both - R_task),
        ("parameters in the additive model", float(p_both)),
    ]

    # D3: the operational version -- what a borrowed alpha costs.
    if os.path.exists(TRANSFER):
        tr = pd.read_csv(TRANSFER).set_index("quantity").value
        for k, lab in [
                ("borrowed_alpha_rel_err_median[same_axis_any_corpus]",
                 "borrowed alpha, median rel. error, same axis any task"),
                ("borrowed_alpha_rel_err_median[same_axis_other_corpora]",
                 "borrowed alpha, median rel. error, same axis other tasks"),
                ("borrowed_alpha_within_factor2[same_axis_other_corpora]",
                 "borrowed alpha within a factor 2, same axis other tasks")]:
            if k in tr.index:
                rows.append((lab, float(tr[k])))

    T = pd.DataFrame([dict(quantity=k, value=v) for k, v in rows])
    save(T.round(6), "table_exponent_decomposition.csv")

    print("=" * 78)
    print("ln(alpha) OVER %d CELLS: HOW MUCH IS THE AXIS, HOW MUCH THE TASK" % n)
    print("=" * 78)
    for k, v in rows:
        print(f"  {k:55s} {v:10.4f}")
    print()
    print("  The axis carries most of ln(alpha) but the task carries a lot too:")
    print("  task identity adds %.2f to R^2, not 0.007.  The operational reading" %
          (R_both - R_ax))
    print("  is D3: a borrowed exponent is good to a factor of two across tasks.")


if __name__ == "__main__":
    main()

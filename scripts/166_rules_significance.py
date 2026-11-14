#!/usr/bin/env python3
"""IS THE HEADLINE TABLE'S ORDERING REAL?  FRIEDMAN, NEMENYI AND BOOTSTRAP.
   (Paper, "Rules in use", Table 'namedrules'; the setting is Demsar 2006.)

Table 'namedrules' reports a median regret and an in-band fraction per rule
over the attainable frontiers, and the paper names \\citet{demsar2006} as the
setting -- comparing methods across datasets rather than within one -- without
running the test that paper prescribes.  This script runs it.

  F1  FRIEDMAN.  Rank the rules within each frontier by regret (ties averaged,
      an infinite regret ranks last) and test the null that all rules have the
      same mean rank.  Only rules applicable on EVERY frontier of the subset
      enter, because the test needs complete blocks; the rest are reported and
      excluded.

  F2  NEMENYI.  The post-hoc of Demsar 2006: a critical difference
      CD = q_alpha sqrt(k(k+1)/(6N)) on mean ranks, with q_alpha the studentised
      range quantile over sqrt(2).  Pairs closer than CD are not separated by
      this evidence.  The table reports every pair against the canonical rule.

  F3  BOOTSTRAP.  10000 resamples of the frontiers (the blocks, not the
      observations) give percentile intervals for each rule's median regret and
      in-band fraction -- the error bars Table 'namedrules' prints without.

Reads  results/table_rules_per_frontier.csv   (script 11)
Writes results/table_rules_significance.csv   (per-rule ranks and intervals)
       results/table_rules_nemenyi.csv        (pairwise, against canonical)
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import selection_rules, save          # noqa: E402

sr = selection_rules()
RES = os.path.join(os.path.dirname(HERE), "results")
B = 10000
SEED = 0
ALPHA = 0.05
# studentised range q_0.05 / sqrt(2), Demsar 2006 Table 5, k = 2..12
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
       8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}


def main():
    D = pd.read_csv(os.path.join(RES, "table_rules_per_frontier.csv"))
    D = D[D.attainable.astype(bool)].reset_index(drop=True)
    cols = {r: f"{r}_regret" for r in sr.RULES}
    full = [r for r, c in cols.items() if D[c].notna().all()]
    dropped = [r for r in sr.RULES if r not in full]
    N, k = len(D), len(full)
    print(f"     {N} attainable frontiers, {k} rules in complete blocks"
          + (f"; excluded for missing values: {', '.join(dropped)}" if dropped else ""))

    R = D[[cols[r] for r in full]].to_numpy(float)
    R = np.where(np.isinf(R), np.nanmax(R[np.isfinite(R)]) * 1e3, R)   # inf ranks last
    ranks = np.apply_along_axis(lambda v: stats.rankdata(v), 1, R)
    mean_rank = ranks.mean(0)

    chi2 = (12.0 * N / (k * (k + 1))) * (np.sum(mean_rank ** 2)
                                         - k * (k + 1) ** 2 / 4.0)
    p_chi = float(stats.chi2.sf(chi2, k - 1))      # sf, not 1-cdf: 1-cdf underflows
    Ff = (N - 1) * chi2 / (N * (k - 1) - chi2)          # Iman-Davenport
    p_F = float(stats.f.sf(Ff, k - 1, (k - 1) * (N - 1)))
    CD = Q05[min(k, 12)] * np.sqrt(k * (k + 1) / (6.0 * N))
    print(f"     Friedman chi2 = {chi2:.1f} (p = {p_chi:.3g}); "
          f"Iman-Davenport F = {Ff:.1f} (p = {p_F:.3g}); Nemenyi CD = {CD:.2f}")

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, N, size=(B, N))
    rows = []
    for j, r in enumerate(full):
        v = D[cols[r]].to_numpy(float)
        inb = D[f"{r}_inband"].astype(float).to_numpy()
        med = np.array([np.median(v[i]) for i in idx])
        frac = inb[idx].mean(1)
        rows.append(dict(
            rule=r, n=N, mean_rank=float(mean_rank[j]),
            median_regret=float(np.median(v)),
            median_regret_lo=float(np.nanpercentile(med, 2.5)),
            median_regret_hi=float(np.nanpercentile(med, 97.5)),
            frac_inband=float(inb.mean()),
            frac_inband_lo=float(np.percentile(frac, 2.5)),
            frac_inband_hi=float(np.percentile(frac, 97.5)),
            friedman_chi2=float(chi2), friedman_p=float(p_chi),
            iman_davenport_F=float(Ff), iman_davenport_p=float(p_F),
            nemenyi_CD=float(CD)))
    T = pd.DataFrame(rows).sort_values("mean_rank").reset_index(drop=True)
    save(T, "table_rules_significance.csv")

    base = T[T.rule == "canonical"].iloc[0]
    pair = []
    for _, r in T.iterrows():
        if r.rule == "canonical":
            continue
        d = float(r.mean_rank - base.mean_rank)
        w = stats.wilcoxon(D[cols["canonical"]].to_numpy(float),
                           np.where(np.isinf(D[cols[r.rule]].to_numpy(float)),
                                    np.nanmax(R), D[cols[r.rule]].to_numpy(float)),
                           zero_method="zsplit")
        pair.append(dict(rule=r.rule, mean_rank=float(r.mean_rank),
                         rank_gap_vs_canonical=d, nemenyi_CD=float(CD),
                         separated_by_nemenyi=bool(d > CD),
                         wilcoxon_stat=float(w.statistic),
                         wilcoxon_p=float(w.pvalue)))
    P = pd.DataFrame(pair)
    save(P, "table_rules_nemenyi.csv")
    sep = int(P.separated_by_nemenyi.sum())
    print(f"     canonical has the best mean rank ({base.mean_rank:.2f}) and is "
          f"separated from {sep} of {len(P)} other rules at the Nemenyi CD")
    print(f"     its in-band fraction is {base.frac_inband:.0%} "
          f"[{base.frac_inband_lo:.0%}, {base.frac_inband_hi:.0%}] and its median "
          f"regret {base.median_regret:.4f} "
          f"[{base.median_regret_lo:.4f}, {base.median_regret_hi:.4f}] nats")
    notsep = P[~P.separated_by_nemenyi].rule.tolist()
    if notsep:
        print(f"     NOT separated from canonical: {', '.join(notsep)}")


if __name__ == "__main__":
    main()

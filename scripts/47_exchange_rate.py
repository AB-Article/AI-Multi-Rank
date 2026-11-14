#!/usr/bin/env python3
"""THE EXCHANGE-RATE FORM OF THE TWO-CANDIDATE DECISION  (Section 7 of paper.tex).

For two candidates on a single-channel law with eps_1 > eps_2 at t_1 < t_2, the
pair's own exchange rate (the chord slope of log-error against log-resource) is

    sbar = ln(eps_1/eps_2) / ln(t_2/t_1).

Proposition 13:  sigma_lamstar(2) > sigma_lamstar(1)  <=>  sbar > alpha/2,
and sbar = S'(xi) for an interior xi, so sbar <= alpha (1 - p_0) <= alpha.

This script verifies both statements on the panel's own frontiers and compares
three decision rules under the resource-aware objective (eq. Rsigma):

  A  the harmonic test of Corollary 9           (reads the fitted floor)
  B  the exchange-rate test with a borrowed alpha (reads t_1, t_2 only)
  C  always take the cheaper candidate           (the constant baseline)

Writes (results/):
    table_exchange_rate_pairs.csv    one row per Pareto pair
    table_exchange_rate_rules.csv    rule x statistic, Table 3 of the paper
    table_exchange_rate_agree.csv    the agreement diagnostic
"""
import os, sys, glob, itertools, warnings
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import save, ROOT  # noqa: E402

FRONT = os.path.join(ROOT, "results", "frontiers")
SKIP = ("base", "img_size", "eps", "model", "name", "arch")


def pareto(eps, t):
    order = np.argsort(t)
    keep, best = [], np.inf
    for i in order:
        if eps[i] < best - 1e-12:
            keep.append(i); best = eps[i]
    return np.array(keep)


def fit_law(t, eps):
    """eps = eps_inf + c t^-alpha by least squares in log-error."""
    lt, le = np.log(t), np.log(eps)
    f = lambda l, li, lc, la: np.logaddexp(li, lc - np.exp(la) * l)
    best = None
    for i0 in (0.3, 0.6, 0.9, 0.99):
        for a0 in (0.05, 0.3, 1.0, 3.0):
            try:
                p, _ = curve_fit(f, lt, le, maxfev=40000,
                                 p0=[np.log(eps.min() * i0), np.log(eps.max()), np.log(a0)])
                r = float(np.sum((f(lt, *p) - le) ** 2))
                if best is None or r < best[1]:
                    best = (p, r)
            except Exception:
                pass
    if best is None:
        return None
    p, r = best
    return dict(eps_inf=float(np.exp(p[0])), c=float(np.exp(p[1])),
                alpha=float(np.exp(p[2])), resid=float(np.sqrt(r / len(eps))))


def build():
    rows = []
    for path in sorted(glob.glob(os.path.join(FRONT, "*.csv"))):
        tag = os.path.basename(path)[:-4]
        df = pd.read_csv(path)
        if "eps" not in df.columns:
            continue
        for rc in [c for c in df.columns
                   if c not in SKIP and pd.api.types.is_numeric_dtype(df[c])]:
            s = df[["eps", rc]].dropna()
            s = s[(s.eps > 0) & (s[rc] > 0)]
            if len(s) < 8:
                continue
            eps, t = s.eps.values, s[rc].values
            P = pareto(eps, t)
            if len(P) < 6:
                continue
            eps, t = eps[P], t[P]
            if np.ptp(np.log(t)) < 0.5:
                continue
            law = fit_law(t, eps)
            if law is None:
                continue
            a, f0 = law["alpha"], law["eps_inf"]
            if not (0.05 * eps.min() < f0 < 0.97 * eps.min()):
                continue                      # floor not identified on this window
            for i, j in itertools.combinations(range(len(P)), 2):
                hi, lo = (i, j) if eps[i] > eps[j] else (j, i)
                if eps[hi] <= eps[lo] or t[lo] <= t[hi]:
                    continue
                sbar = np.log(eps[hi] / eps[lo]) / np.log(t[lo] / t[hi])
                marg = np.log(eps[hi] / eps[lo]) - a / 2 * np.log(t[lo] / t[hi])
                rows.append(dict(
                    cell=f"{tag}|{rc}", corpus=tag.split("__")[0], axis=rc,
                    alpha_fit=a, eps_inf_fit=f0, resid=law["resid"],
                    eps_hi=eps[hi], eps_lo=eps[lo], t_hi=t[hi], t_lo=t[lo],
                    sbar=sbar, p0_pair=1.0 - sbar / a,
                    harmonic=2 / (1 / eps[hi] + 1 / eps[lo]),
                    truth=bool(marg > 0), cost=abs(marg)))
    return pd.DataFrame(rows)


def score(name, pick, d):
    wrong = pick != d.truth
    c = np.where(wrong, d.cost, 0.0)
    return dict(rule=name, accuracy=1 - wrong.mean(), mean_cost=c.mean(),
                p90_cost=np.percentile(c, 90), max_cost=c.max())


def main():
    d = build()
    save(d, "table_exchange_rate_pairs.csv")
    cells = d.groupby("cell").alpha_fit.first()
    print(f"  {d.cell.nunique()} cells with an identified floor, {len(d)} Pareto pairs")

    # --- Proposition 13, both statements -------------------------------------
    ident = float((d.truth == (d.sbar > d.alpha_fit / 2)).mean())
    bracket = float((d.sbar <= d.alpha_fit + 1e-9).mean())
    print(f"  identity  truth <=> sbar > alpha/2 : {ident:.4f} of pairs")
    print(f"  bracket   sbar <= alpha           : {bracket:.4f} of pairs")

    # --- rule comparison -----------------------------------------------------
    a_glob = float(cells.median())
    loo = d.cell.map(lambda c: float(np.median(cells.drop(c).values)))
    axis_a = d.groupby(["axis", "cell"]).alpha_fit.first()

    def borrow(r, cross_corpus):
        """Median alpha of other cells sharing the resource axis."""
        try:
            v = axis_a.loc[r.axis]
        except KeyError:
            v = pd.Series(dtype=float)
        v = v.drop(r.cell, errors="ignore")
        if cross_corpus:
            keep = [k for k in v.index if not k.startswith(r.corpus)]
            v = v.loc[keep]
        return float(np.median(v.values)) if len(v) else float(np.median(cells.drop(r.cell).values))

    a_axis = d.apply(lambda r: borrow(r, False), axis=1)
    a_strict = d.apply(lambda r: borrow(r, True), axis=1)

    rules = [
        score("harmonic test, fitted floor (Cor. 9)", d.harmonic > 2 * d.eps_inf_fit, d),
        score(f"exchange rate, global median alpha={a_glob:.2f}", d.sbar > a_glob / 2, d),
        score("exchange rate, alpha from other cells", d.sbar > loo / 2, d),
        score("exchange rate, alpha from same axis", d.sbar > a_axis / 2, d),
        score("exchange rate, alpha from other corpora, same axis", d.sbar > a_strict / 2, d),
        score("always the cheaper candidate", pd.Series(np.zeros(len(d), bool), index=d.index), d),
        score("always the more accurate candidate", pd.Series(np.ones(len(d), bool), index=d.index), d),
        score("exchange rate, in-sample alpha (the objective itself)", d.sbar > d.alpha_fit / 2, d),
    ]
    R = pd.DataFrame(rules)
    R["n_pairs"] = len(d)
    save(R, "table_exchange_rate_rules.csv")
    print(R.round(4).to_string(index=False))

    # --- agreement diagnostic ------------------------------------------------
    A = d.harmonic > 2 * d.eps_inf_fit
    B = d.sbar > a_axis / 2
    agree = (A == B)
    rows = [dict(subset="the two tests agree", frac_pairs=float(agree.mean()),
                 accuracy=float((A == d.truth)[agree].mean()),
                 mean_cost=float(np.where((A != d.truth)[agree], d.cost[agree], 0).mean())),
            dict(subset="the two tests disagree", frac_pairs=float((~agree).mean()),
                 accuracy=float((A == d.truth)[~agree].mean()),
                 mean_cost=float(np.where((A != d.truth)[~agree], d.cost[~agree], 0).mean()))]
    G = pd.DataFrame(rows)
    save(G, "table_exchange_rate_agree.csv")
    print(G.round(4).to_string(index=False))

    with open(os.path.join(ROOT, "results", "exchange_rate_summary.txt"), "w") as fh:
        fh.write(f"cells {d.cell.nunique()}\npairs {len(d)}\n"
                 f"identity {ident:.6f}\nbracket {bracket:.6f}\n"
                 f"alpha_global_median {a_glob:.6f}\n"
                 f"median_resid {d.resid.median():.6f}\n"
                 f"median_p0_pair {d.p0_pair.median():.6f}\n"
                 f"median_sbar_over_alpha {np.median(d.sbar/d.alpha_fit):.6f}\n")


if __name__ == "__main__":
    main()

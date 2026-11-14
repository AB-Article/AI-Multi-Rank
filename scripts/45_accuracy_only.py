#!/usr/bin/env python3
"""T8  The accuracy-only collapse of the canonical selection, and its cost.

With one reducible channel the canonical scalarization, evaluated along the
law, is a function of the error and the floor alone:

    J(eps) = 1/2 ln(eps - eps_inf) - ln eps ,
    sigma_{lambda*} = J(eps) - 1/2 ln c ,
    R(eps)          = -J(eps) - 1/2 ln(4 eps_inf)     [exactly]

so argmax_k J(eps_k) = argmin_k R(eps_k): the fitted c and alpha drop out of
the ranking.  For two candidates with eps_1 > eps_2 the more accurate one wins
iff  1/eps_inf > 1/eps_1 + 1/eps_2,  i.e. iff the harmonic mean of the two
errors exceeds 2 eps_inf.

That collapse is also a blind spot.  R is a function of (eps, eps_inf) only,
so a rule comparison conducted in R cannot see a candidate that attains its
error off the law.  This script therefore scores every rule under BOTH

    R        the error-space regret  ln cosh(zeta/2)                  and
    R_sigma  max_k' sigma(x_k') - sigma(x_k),
             sigma(x_k) = -ln eps_k - (alpha/2) x_k  at the MEASURED x,

which agree on the law and differ by each candidate's off-law residual.

Panel: the single-resource Pareto frontiers of script 11 whose fit returns an
identified floor (1e-3 < eps_inf < 0.5, 1e-3 < alpha < 5.99).  Cells whose
fitted floor exceeds the best observed error are KEPT: there J is -inf on
every candidate above the floor and the rule correctly returns the cheapest,
which is the feasibility logic of Corollary 'feasible', not a special case.

Writes (results/):
    table_acconly_cells.csv      one row per (frontier, resource) cell
    table_acconly_rules.csv      Table 2 of the note: rules x {R, R_sigma}
    table_acconly_pairs.csv      every Pareto pair, rule vs truth
    table_acconly_pairs_strata.csv   the stratified pairwise summary
    table_acconly_floor.csv      floor-misspecification profile
    table_acconly_threepoint.csv Aitken Delta^2 floor vs the fitted floor
"""
import os, sys, itertools, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import selection_rules, save                      # noqa: E402

sr = selection_rules()

FLOOR_LO, FLOOR_HI, A_LO, A_HI = 1e-3, 0.5, 1e-3, 5.99
BAND = 0.05                                     # nats, as in script 11


# ----------------------------------------------------------------- the rule ---
def J(e, einf):
    """Canonical value; -inf for candidates at or below the claimed floor."""
    e = np.asarray(e, float)
    red = e - einf
    out = np.full(e.shape, -np.inf)
    ok = red > 0
    out[ok] = 0.5 * np.log(red[ok]) - np.log(e[ok])
    return out


def pick_J(e, einf):
    """argmax J.  J is strictly concave in ln eps with its peak at 2 eps_inf,
    so this returns the most accurate candidate when the whole frontier lies
    above the reference and the cheapest when it lies below.  If no candidate
    exceeds the claimed floor there is no reducible error to buy and the
    cheapest candidate is correct."""
    jv = J(e, einf)
    return int(np.argmax(e)) if not np.isfinite(jv).any() else int(np.argmax(jv))


def harmonic_rule(e_hi, e_lo, einf):
    """True iff the MORE ACCURATE of the two candidates wins (e_hi > e_lo)."""
    return e_hi * e_lo > einf * (e_hi + e_lo)


def aitken_floor(x, e, tol=0.05):
    """eps_inf from the widest equally-log-spaced triple, Aitken's Delta^2."""
    best = None
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            for k in range(j + 1, len(x)):
                h1, h2 = x[j] - x[i], x[k] - x[j]
                if h1 <= 0 or h2 <= 0 or abs(np.log(h2 / h1)) > tol:
                    continue
                d1, d2 = e[i] - e[j], e[j] - e[k]
                if not (d1 > 0 and d2 > 0 and d1 > d2):
                    continue
                einf = e[k] - d2 * d2 / (d1 - d2)
                if not (0 < einf < e[k]):
                    continue
                if best is None or (x[k] - x[i]) > best[0]:
                    best = (x[k] - x[i], einf)
    return (np.nan, np.nan) if best is None else (best[1], best[0])


# -------------------------------------------------------------------- panel ---
def panel():
    out = []
    for field, name, cost, x, e in sr.single_resource_frontiers():
        einf, c, a, rms = sr.fit_law(x, e)
        if not (FLOOR_LO < einf < FLOOR_HI) or not (A_LO < a < A_HI):
            continue
        o = np.argsort(x)
        out.append((field, name, cost, x[o], e[o], einf, c, a, rms))
    return out


RULES = ["J", "nearest", "two_x_best", "knee", "half_slope", "best", "cheapest"]
READS_X = dict(J="no", nearest="no", two_x_best="no", knee="yes",
               half_slope="yes", best="no", cheapest="yes")


def run():
    P = panel()
    print(f"panel: {len(P)} cells, {len({p[1] for p in P})} frontiers, "
          f"{sum(1 for p in P if p[5] >= p[4].min())} with the floor above the best error")

    cells, pairs = [], []
    for field, name, cost, x, e, einf, c, a, rms in P:
        S = -np.log(e)
        sig = S - 0.5 * a * (x - x.min())        # measured-resource score
        jv = J(e, einf)
        iR, iS = pick_J(e, einf), int(np.argmax(sig))
        picks = dict(J=iR,
                     nearest=sr.nearest(np.log(e), np.log(2 * einf)),
                     two_x_best=sr.nearest(np.log(e), np.log(2 * e.min())),
                     knee=sr.rule_knee(x, S),
                     half_slope=sr.rule_half_slope(x, S),
                     best=int(np.argmin(e)),
                     cheapest=0)
        row = dict(field=field, frontier=name, cost=cost, n_pareto=len(e),
                   eps_inf=einf, alpha=a, law_rms_nats=rms,
                   attainable=bool(e.min() <= 2 * einf <= e.max()),
                   floor_above_best=bool(einf >= e.min()),
                   same_optimum=bool(iR == iS))
        for r, i in picks.items():
            row[f"R_{r}"] = np.nan if i is None else float(jv[iR] - jv[i])
            row[f"Rsigma_{r}"] = np.nan if i is None else float(sig[iS] - sig[i])
        # three-point floor
        e3, span3 = aitken_floor(x, e)
        row["eps_inf_3pt"] = e3
        row["triple_span_nats"] = span3
        if np.isfinite(e3):
            i3 = pick_J(e, e3)
            row["Rsigma_threepoint"] = float(sig[iS] - sig[i3])
            row["threepoint_same_as_fitted"] = bool(i3 == iR)
        else:
            row["Rsigma_threepoint"] = np.nan
            row["threepoint_same_as_fitted"] = np.nan
        # floor misspecification
        for mult in (0.25, 0.5, 0.7, 0.9, 1.25, 1.5, 2.0, 4.0):
            row[f"Rsigma_floor_x{mult}"] = float(sig[iS] - sig[pick_J(e, einf * mult)])
        cells.append(row)

        pred = einf + c * np.exp(-a * (x - x.min()))
        resid = np.log(e / pred)
        for i, k in itertools.combinations(range(len(e)), 2):
            hi, lo = (i, k) if e[i] > e[k] else (k, i)
            if e[hi] == e[lo]:
                continue
            marg = np.log(e[hi] / e[lo]) - 0.5 * a * (x[lo] - x[hi])
            pairs.append(dict(
                field=field, frontier=name, cost=cost, law_rms_nats=rms,
                eps_hi=float(e[hi]), eps_lo=float(e[lo]), eps_inf=einf,
                harmonic_mean=float(2 * e[hi] * e[lo] / (e[hi] + e[lo])),
                truth_take_accurate=bool(marg > 0),
                rule_take_accurate=bool(harmonic_rule(e[hi], e[lo], einf)),
                margin_nats=float(abs(marg)),
                log_sep_nats=float(np.log(e[hi] / e[lo])),
                pair_resid_gap_nats=float(abs(resid[hi] - resid[lo]))))

    C = pd.DataFrame(cells)
    D = pd.DataFrame(pairs)
    D["correct"] = D.truth_take_accurate == D.rule_take_accurate
    save(C, "table_acconly_cells.csv")
    save(D, "table_acconly_pairs.csv")

    # ---- Table 2 of the note -------------------------------------------------
    def block(sub, lab):
        out = []
        for r in RULES:
            A, B = sub[f"R_{r}"].dropna(), sub[f"Rsigma_{r}"].dropna()
            out.append(dict(subset=lab, rule=r, reads_x=READS_X[r], n=len(sub),
                            R_median=A.median(), R_p90=A.quantile(.9),
                            R_in_band=float(np.mean(A < BAND)),
                            Rsigma_median=B.median(), Rsigma_p90=B.quantile(.9),
                            Rsigma_in_band=float(np.mean(B < BAND)),
                            Rsigma_max=B.max()))
        return out

    T = pd.DataFrame(block(C, "all cells")
                     + block(C[C.attainable], "attainable")
                     + block(C[C.law_rms_nats <= C.law_rms_nats.median()],
                             "better-fitting half"))
    save(T, "table_acconly_rules.csv")

    # ---- stratified pairwise summary ----------------------------------------
    def skill(sub, lab):
        acc = sub.correct.mean()
        base = max(sub.truth_take_accurate.mean(), 1 - sub.truth_take_accurate.mean())
        t, r = sub.truth_take_accurate, sub.rule_take_accurate
        tp, fn = int((r & t).sum()), int((~r & t).sum())
        tn, fp = int((~r & ~t).sum()), int((r & ~t).sum())
        return dict(stratum=lab, n=len(sub), accuracy=acc, best_constant_rule=base,
                    balanced=0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
                    skill=(acc - base) / max(1 - base, 1e-12))

    q = D.law_rms_nats.quantile([1 / 3, 2 / 3]).values
    S = pd.DataFrame([
        skill(D, "all pairs"),
        skill(D[D.law_rms_nats <= q[0]], "law rms tercile 1 (best fit)"),
        skill(D[(D.law_rms_nats > q[0]) & (D.law_rms_nats <= q[1])], "law rms tercile 2"),
        skill(D[D.law_rms_nats > q[1]], "law rms tercile 3 (worst fit)"),
        skill(D[D.pair_resid_gap_nats < 0.05], "pair off-law gap < 0.05 nats"),
        skill(D[D.pair_resid_gap_nats > 0.30], "pair off-law gap > 0.30 nats"),
        skill(D[D.margin_nats > 0.05], "decision margin > 0.05 nats"),
        skill(D[D.margin_nats > 0.20], "decision margin > 0.20 nats"),
        skill(D[D.margin_nats <= 0.05], "near-tie, margin < 0.05 nats"),
        skill(D[(D.law_rms_nats <= q[0]) & (D.margin_nats > 0.05)],
              "best fit and margin > 0.05"),
    ] + [skill(D[D.field == f], f"field: {f}") for f in sorted(D.field.unique())])
    save(S, "table_acconly_pairs_strata.csv")

    # ---- floor misspecification ---------------------------------------------
    rows = [dict(floor_multiplier=1.0,
                 Rsigma_median=C.Rsigma_J.median(),
                 Rsigma_p90=C.Rsigma_J.quantile(.9),
                 in_band=float(np.mean(C.Rsigma_J < BAND)))]
    for mult in (0.25, 0.5, 0.7, 0.9, 1.25, 1.5, 2.0, 4.0):
        v = C[f"Rsigma_floor_x{mult}"]
        rows.append(dict(floor_multiplier=mult, Rsigma_median=v.median(),
                         Rsigma_p90=v.quantile(.9), in_band=float(np.mean(v < BAND))))
    save(pd.DataFrame(rows).sort_values("floor_multiplier"), "table_acconly_floor.csv")

    # ---- three-point floor ---------------------------------------------------
    A3 = C[C.eps_inf_3pt.notna()].copy()
    A3["ratio_3pt_to_fitted"] = A3.eps_inf_3pt / A3.eps_inf
    save(A3[["field", "frontier", "cost", "eps_inf", "eps_inf_3pt",
             "ratio_3pt_to_fitted", "triple_span_nats", "Rsigma_threepoint",
             "Rsigma_J", "threepoint_same_as_fitted"]], "table_acconly_threepoint.csv")

    # ---- console summary -----------------------------------------------------
    print(f"\nR-optimum == R_sigma-optimum on {C.same_optimum.mean():.1%} of cells")
    print(f"Spearman(law rms, R_sigma of accuracy-only) = "
          f"{C[['law_rms_nats','Rsigma_J']].corr(method='spearman').iloc[0,1]:.3f}")
    print(f"\npairs: {len(D)};  harmonic rule {D.correct.mean():.2%} vs best constant "
          f"{max(D.truth_take_accurate.mean(), 1-D.truth_take_accurate.mean()):.2%}")
    w = D[~D.correct].margin_nats
    print(f"cost of a wrong call: median {w.median():.4f}, p90 {w.quantile(.9):.4f} nats")
    v = D[D.truth_take_accurate]
    print(f"cheap-pick bound  <= ln(e1/e2): {np.mean(v.margin_nats <= v.log_sep_nats + 1e-12):.1%}")
    print(f"three-point floor: {len(A3)} cells, ratio median "
          f"{A3.ratio_3pt_to_fitted.median():.3f} "
          f"IQR [{A3.ratio_3pt_to_fitted.quantile(.25):.3f}, "
          f"{A3.ratio_3pt_to_fitted.quantile(.75):.3f}]; "
          f"R_sigma median {A3.Rsigma_threepoint.median():.4f} "
          f"(fitted floor {A3.Rsigma_J.median():.4f})")


if __name__ == "__main__":
    run()

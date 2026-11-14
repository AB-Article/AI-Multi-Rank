#!/usr/bin/env python3
"""125_grid_corpus.py -- the bandwidth-free readings, on the archive's own data.

No new data, no new fit.  The same 117 frontiers as scripts 68/122, read through
Proposition 'grid', Proposition 'floorbracket' and Proposition 'grainprice',
every one of which consumes *measured accuracies only*: no kernel, no
bandwidth, no smoother, no fitted chart.  Where a comparison with the incumbent
reading is possible it uses the archive's own jet at the bandwidth and grid of
scripts 81/97/104/109/122 (h = 0.40*span, 9-point grid), at the degree-4 order
of 104/109/122, so the comparison is like for like.

Four readings per frontier.

 (A) THE AXIOM, EXACTLY.  ln eps must be convex in x.  On every consecutive
     triple the second divided difference [x0,x1,x2] ln eps >= 0 is an exact,
     one-sided, bandwidth-free test (the three-accuracy member of the hierarchy
     of Prop. 'grid'; its first form is the log-convexity defect J of
     Prop. 'threeways'(iii)).

 (B) THE CHANNEL COUNT, WITHOUT THE ARC.  Cor. 'prony' says the order is a
     Hankel rank; Prop. 'unbounded' says a bounded spectrum is never divisible.
     The practical form needs no determinant: alpha recovered from each triple
     is CONSTANT along the frontier if and only if the window is two-atom.  We
     report the spread of alpha and compare with the incumbent certificate
     L < pi, which is a plug-in arc reading on a window (script 77).

 (C) THE FLOOR, BOUNDED TWICE.  eps_A (Aitken) is the largest floor compatible
     with the axiom; eps_W is the largest floor compatible with the composition
     law.  ln(eps_A/eps_W) is then the floor precision, in nats, that decides
     divisibility on that frontier -- to be compared with the count-axis bracket
     of 0.094 / 0.20 nats, which a pass@k harvest delivers with no extrapolation.

 (D) THE PRICE, WITHOUT A SMOOTHER.  lambda = alpha/(1+t) (Prop. 'grainprice')
     against the jet's lambda = sqrt(kappa/t).  Reported as a price difference
     AND as a decision: whether the two prices pick the same point of the
     frontier, and by what sigma gap in nats.

Outputs: results/table_grid_frontiers.csv
         results/table_grid_summary.csv
"""
from __future__ import annotations
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save, all_frontiers                        # noqa: E402

HFRAC, NGRID = 0.40, 9
BRACKET_NESTED, BRACKET_INDEP = 0.094, 0.20     # one attempt's price, sc. 113
TIE_BAND = 0.05                                 # nats, Thm. covering
GRAIN = 1.0


def jets(x, S, kmax=4, hfrac=HFRAC, ng=NGRID):
    """Identical to scripts 81/97/104/109/122."""
    Z = x[-1] - x[0]
    h = Z * hfrac
    grid = np.linspace(x[0] + 0.15 * Z, x[-1] - 0.15 * Z, ng)
    fact = [1.0]
    for j in range(1, kmax + 1):
        fact.append(fact[-1] * j)
    out = []
    for g in grid:
        d = x - g
        w = np.sqrt(np.exp(-0.5 * (d / h) ** 2))
        M = np.column_stack([d ** j / fact[j] for j in range(kmax + 1)])
        b, *_ = np.linalg.lstsq(M * w[:, None], S * w, rcond=None)
        out.append([g, b[0]] + [(-1) ** (k + 1) * b[k] for k in range(1, kmax + 1)])
    return np.array(out)


def fit_floor(x, e):
    """The incumbent three-parameter chart (scripts 109/122)."""
    def mdl(th, xx):
        return np.exp(th[0]) + np.exp(th[1]) * np.exp(-np.exp(th[2]) * np.atleast_1d(xx))
    th0 = [np.log(max(e.min() * .5, 1e-9)), np.log(e.max()), np.log(.4)]
    try:
        r = least_squares(lambda th: np.log(mdl(th, x)) - np.log(e), th0, max_nfev=40000)
        return float(np.exp(r.x[0]))
    except Exception:
        return np.nan


# ---------------------------------------------------------------- the closed forms
def alpha_floor(p):
    """alpha and eps_A for eps = eps_inf + c e^{-alpha x} through three points.

    Any spacing: (e0-e1)/(e1-e2) = (1-e^{-au})/(e^{-au}-e^{-av}) with u = x1-x0,
    v = x2-x0, the right side increasing from u/(v-u) to infinity.  Equal spacing
    reduces it to Aitken's Delta^2 (eq. aitken).
    """
    (x0, e0), (x1, e1), (x2, e2) = p
    u, v = x1 - x0, x2 - x0
    if not (e0 > e1 > e2) or not (0 < u < v):
        return None
    tgt = (e0 - e1) / (e1 - e2)
    if tgt <= u / (v - u):
        return None                                      # no positive floor
    f = lambda a: (1 - np.exp(-a * u)) / (np.exp(-a * u) - np.exp(-a * v)) - tgt
    lo = hi = 1e-9
    while f(hi) < 0 and hi < 1e7:
        hi *= 2.0
    if f(hi) < 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) >= 0:
            hi = mid
        else:
            lo = mid
    a = 0.5 * (lo + hi)
    c = (e0 - e1) / (np.exp(-a * x0) - np.exp(-a * x1))
    return a, e0 - c * np.exp(-a * x0)


def floor_W(p):
    """Largest floor compatible with the composition law: ln(eps/eps_inf) convex."""
    (x0, e0), (x1, e1), (x2, e2) = p
    l = np.log([e0, e1, e2]); h1, h2 = x1 - x0, x2 - x1
    def defect(c):
        d = l - c
        if d.min() <= 0:
            return -1.0
        return (np.log(d[2]) - np.log(d[1])) / h2 - (np.log(d[1]) - np.log(d[0])) / h1
    hi = l.min() - 1e-12; lo = l.min() - 60.0
    if defect(lo) < 0:
        return np.nan
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if defect(mid) >= 0:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def dd2(xs, ys):
    """Second divided difference (exact, any spacing)."""
    x0, x1, x2 = xs; y0, y1, y2 = ys
    return (y0 / ((x0 - x1) * (x0 - x2)) + y1 / ((x1 - x0) * (x1 - x2))
            + y2 / ((x2 - x0) * (x2 - x1)))


def run():
    rows = []
    for fr in all_frontiers():
        x, e = np.asarray(fr["x"], float), np.asarray(fr["e"], float)
        o = np.argsort(x); x, e = x[o], e[o]
        k = (e > 0) & np.isfinite(e)
        x, e = x[k], e[k]
        if len(x) < 5:
            continue
        S, L = -np.log(e), np.log(e)

        # ---- (A) the axiom, exactly, on every consecutive triple -------------
        defects = [dd2(x[j:j + 3], L[j:j + 3]) for j in range(len(x) - 2)]
        n_tri = len(defects)
        n_pass = int(sum(d >= 0 for d in defects))

        # ---- (A') the same test on WIDE triples (span >= 2/3 of the window) --
        Z = x[-1] - x[0]
        widx = [(i, j, m) for i in range(len(x)) for j in range(i + 1, len(x) - 1)
                for m in range(j + 1, len(x)) if x[m] - x[i] >= 0.667 * Z]
        dw = [dd2(x[[i, j, m]], L[[i, j, m]]) for i, j, m in widx]
        n_wide = len(dw)
        n_wide_pass = int(sum(d >= 0 for d in dw))
        alw = []
        for i, j, m in widx:
            sol = alpha_floor(list(zip(x[[i, j, m]], e[[i, j, m]])))
            if sol is not None:
                alw.append(sol[0])
        alw = np.asarray(alw, float)
        spread_wide = float((np.percentile(alw, 75) - np.percentile(alw, 25))
                            / np.median(alw)) if len(alw) >= 4 else np.nan

        # ---- (B) alpha per triple: constant <=> two atoms --------------------
        sols = [alpha_floor(list(zip(x[j:j + 3], e[j:j + 3]))) for j in range(len(x) - 2)]
        al = np.array([s[0] for s in sols if s is not None], float)
        n_alpha = len(al)
        spread = float((np.percentile(al, 75) - np.percentile(al, 25)) / np.median(al)) \
            if n_alpha >= 4 else np.nan

        # ---- (C) the floor, bounded twice, from the widest triple ------------
        wide = list(zip(x[[0, len(x) // 2, -1]], e[[0, len(x) // 2, -1]]))
        AW = alpha_floor(wide)
        eps_A = float(AW[1]) if AW else np.nan
        alpha_wide = float(AW[0]) if AW else np.nan
        eps_W = floor_W(wide)
        room = float(np.log(eps_A / eps_W)) if (AW and np.isfinite(eps_W)
                                                and eps_A > 0 and eps_W > 0) else np.nan
        einf_fit = fit_floor(x, e)

        # ---- (D) the price, with and without a smoother ----------------------
        J = jets(x, S, kmax=4)
        c1, c2 = J[:, 2], J[:, 3]
        ok = (c1 > 0) & (c2 > 0)
        lam_jet = float(np.sqrt(np.median(c2[ok]) / GRAIN)) if ok.sum() >= 3 else np.nan
        lam_new = alpha_wide / (1.0 + GRAIN) if np.isfinite(alpha_wide) else np.nan
        same, gap = np.nan, np.nan
        if np.isfinite(lam_jet) and np.isfinite(lam_new):
            i_jet = int(np.argmax(S - lam_jet * x))
            i_new = int(np.argmax(S - lam_new * x))
            same = int(i_jet == i_new)
            sj = S - lam_jet * x
            gap = float(sj[i_jet] - sj[i_new])            # cost of the new pick, jet's price

        rows.append(dict(
            corpus=fr["corpus"], name=fr["name"], n=len(x),
            span_nats=float(S.max() - S.min()),
            # (A)
            n_triples=n_tri, n_triples_axiom_pass=n_pass,
            axiom_refuted=int(n_pass < n_tri),
            n_wide_triples=n_wide, n_wide_triples_axiom_pass=n_wide_pass,
            wide_pass_share=float(n_wide_pass / n_wide) if n_wide else np.nan,
            alpha_iqr_over_median_wide=spread_wide,
            rank1_wide_at_10pct=int(np.isfinite(spread_wide) and spread_wide < 0.10),
            rank1_wide_at_25pct=int(np.isfinite(spread_wide) and spread_wide < 0.25),
            # (B)
            n_alpha=n_alpha, alpha_iqr_over_median=spread,
            rank1_at_10pct=int(np.isfinite(spread) and spread < 0.10),
            rank1_at_25pct=int(np.isfinite(spread) and spread < 0.25),
            # (C)
            alpha_wide=alpha_wide, eps_A=eps_A, eps_W=eps_W,
            floor_room_nats=room,
            room_within_nested_bracket=int(np.isfinite(room) and room < BRACKET_NESTED),
            room_within_indep_bracket=int(np.isfinite(room) and room < BRACKET_INDEP),
            einf_fitted=einf_fit,
            fitted_above_eps_A=int(np.isfinite(einf_fit) and np.isfinite(eps_A)
                                   and einf_fit > eps_A),
            fitted_above_eps_W=int(np.isfinite(einf_fit) and np.isfinite(eps_W)
                                   and einf_fit > eps_W),
            # (D)
            lambda_jet=lam_jet, lambda_grid=lam_new,
            same_pick=same, sigma_gap_nats=gap,
            pick_inside_tie_band=int(np.isfinite(gap) and abs(gap) < TIE_BAND)))
    return pd.DataFrame(rows)


def main():
    df = run()
    save(df, "table_grid_frontiers.csv")
    n = len(df)
    print("=" * 78)
    print("(A)  THE AXIOM, EXACTLY (no bandwidth, one-sided)")
    print(f"  frontiers ......................................... {n}")
    print(f"  exact three-accuracy triples tested ............... {int(df.n_triples.sum())}")
    print(f"  triples passing ln eps convex ..................... "
          f"{int(df.n_triples_axiom_pass.sum())} "
          f"({100*df.n_triples_axiom_pass.sum()/df.n_triples.sum():.1f}%)")
    print(f"  frontiers with at least one refuting triple ....... {int(df.axiom_refuted.sum())}")
    print(f"  WIDE triples (span >= 2/3 of the window): median share passing "
          f"{df.wide_pass_share.median():.3f}")
    print(f"      -- the defect scales as h^2 and the noise does not, so narrow")
    print(f"         consecutive triples measure noise, not curvature.")
    print()
    print("(B)  THE CHANNEL COUNT WITHOUT THE ARC")
    s = df.alpha_iqr_over_median.dropna()
    print(f"  frontiers with alpha on >= 4 triples .............. {len(s)}")
    print(f"  median IQR/median of alpha along the frontier ..... {s.median():.3f}")
    print(f"  two-atom at a 10% / 25% tolerance ................. "
          f"{int(df.rank1_at_10pct.sum())} / {int(df.rank1_at_25pct.sum())} of {len(s)}")
    sw = df.alpha_iqr_over_median_wide.dropna()
    print(f"  on WIDE triples: median spread {sw.median():.3f}, two-atom at 10% / 25%: "
          f"{int(df.rank1_wide_at_10pct.sum())} / {int(df.rank1_wide_at_25pct.sum())} of {len(sw)}")
    print(f"  incumbent certificate L < pi (script 77) .......... 55/55, a plug-in arc "
          f"reading at h = 0.40*span")
    print()
    print("(C)  THE FLOOR, BOUNDED TWICE FROM THREE ACCURACIES")
    r = df.floor_room_nats.dropna()
    print(f"  frontiers with both bounds available .............. {len(r)}")
    print(f"  median ln(eps_A/eps_W) -- the floor precision that decides "
          f"divisibility: {r.median():.3f} nats   IQR [{r.quantile(.25):.3f}, {r.quantile(.75):.3f}]")
    print(f"  decided by the count-axis bracket 0.20 / 0.094 .... "
          f"{int((r > BRACKET_INDEP).sum())} / {int((r > BRACKET_NESTED).sum())} of {len(r)}"
          f"   (a bracket narrower than the room decides)")
    print(f"  fitted floor above eps_A (axiom violated by the fit) {int(df.fitted_above_eps_A.sum())}")
    print(f"  fitted floor above eps_W (would refute divisibility) {int(df.fitted_above_eps_W.sum())}")
    print()
    print("(D)  THE PRICE, WITH AND WITHOUT A SMOOTHER")
    d = df.dropna(subset=["lambda_jet", "lambda_grid"])
    print(f"  frontiers with both prices ........................ {len(d)}")
    print(f"  median |ln(lambda_grid/lambda_jet)| ............... "
          f"{np.median(np.abs(np.log(d.lambda_grid/d.lambda_jet))):.3f}")
    print(f"  same point of the frontier selected ............... "
          f"{int(d.same_pick.sum())}/{len(d)}  ({100*d.same_pick.mean():.0f}%)")
    print(f"  disagreements inside the 0.05-nat tie band ........ "
          f"{int(d.pick_inside_tie_band.sum())}/{len(d)}")
    print(f"  median sigma gap of the new pick at the jet's price {d.sigma_gap_nats.median():.4f} nats")

    summ = pd.DataFrame([dict(
        n_frontiers=n,
        triples_tested=int(df.n_triples.sum()),
        triples_axiom_pass=int(df.n_triples_axiom_pass.sum()),
        frontiers_axiom_refuted=int(df.axiom_refuted.sum()),
        median_alpha_spread=float(sw.median()),
        median_alpha_spread_narrow=float(s.median()),
        wide_triples_tested=int(df.n_wide_triples.sum()),
        wide_pass_share_median=float(df.wide_pass_share.median()),
        rank1_wide_at_10pct=int(df.rank1_wide_at_10pct.sum()),
        rank1_wide_at_25pct=int(df.rank1_wide_at_25pct.sum()),
        rank1_at_10pct=int(df.rank1_at_10pct.sum()),
        rank1_at_25pct=int(df.rank1_at_25pct.sum()),
        n_floor_room=len(r),
        median_floor_room_nats=float(r.median()),
        floor_room_q25=float(r.quantile(.25)), floor_room_q75=float(r.quantile(.75)),
        decided_by_bracket_020=int((r > BRACKET_INDEP).sum()),
        decided_by_bracket_0094=int((r > BRACKET_NESTED).sum()),
        fitted_above_eps_A=int(df.fitted_above_eps_A.sum()),
        fitted_above_eps_W=int(df.fitted_above_eps_W.sum()),
        n_price=len(d),
        median_abs_log_price_ratio=float(np.median(np.abs(np.log(d.lambda_grid/d.lambda_jet)))),
        same_pick=int(d.same_pick.sum()),
        disagreements_inside_tie_band=int(d.pick_inside_tie_band.sum()),
        median_sigma_gap_nats=float(d.sigma_gap_nats.median()))])
    save(summ, "table_grid_summary.csv")


if __name__ == "__main__":
    main()

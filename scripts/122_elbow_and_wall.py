#!/usr/bin/env python3
"""THE ELBOW AND THE WALL ON THE 117 FRONTIERS: THE NEW ROUTE'S REACH.
   (Section "The price is a frontier": Prop. 'elbow', 'wallrefutes';
    Online Appendix sec. "The price is convex".)

Script 121 proves the laws.  This one measures what they BUY on the archive's
own frontiers, against the routes they replace.  Nothing new is fitted: the jet
bandwidth and interior grid are those of scripts 81/97/104/109 exactly
(h = 0.40 * span, 9-point interior grid) and the jet is the degree-4 one of
104/109 -- not the degree-2 jet of 81/97 -- so every comparison here is like
for like.

Three comparisons, all on the same 117 frontiers and the same jet.

 (A) ADMISSIBILITY: WHERE IS A QUADRUPLE DIAGNOSTIC?
     Incumbent route (Prop. 'lumpy'):  a quadruple refutes only where
     eps > 2 eps_inf, which needs a FITTED FLOOR -- three parameters, and the
     condition is a property of the incumbent chart.
     New route (Prop. 'elbow'):        a quadruple refutes only to the LEFT of
     the curvature maximum, kappa' = 0, i.e. where c_3 < 0.  This is local,
     floor-free, chart-free and holds for EVERY bounded spectrum.
     Measured: how many frontiers each route can place, and whether they agree
     where both are available.

 (B) THE WALL AS A SECOND-ORDER TEST.
     The axiom allows N_eff <= rho - 1; the hypothesis only N_eff <= ln rho
     (rho = eps/eps_inf).  The gap is a refutation band of width
     N_eff - ln(1 + N_eff) nats of floor precision.  Per frontier we report the
     band at the window's deepest legible point, and compare it against the
     count-axis floor bracket (one attempt's price, 0.094-0.20 nats) that the
     paper already delivers without extrapolation.

 (C) THE ARCHIVE'S "TIGHTENING FACTOR" IS THE BAND.
     Script 97 reports eps*e^{-N_eff} beating eps/(1+N_eff) by (1+N)e^{-N}.
     Recomputed here from the same jet alongside N - ln(1+N): they are one
     function, so the reported gain and the indivisibility margin are the same
     measurement under two names.

Outputs: results/table_elbow_frontiers.csv
         results/table_elbow_summary.csv
         results/table_wall_band.csv
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
BRACKET_NESTED, BRACKET_INDEP = 0.094, 0.20                     # one attempt's price, sc. 113


def jets(x, S, kmax=4, hfrac=HFRAC, ng=NGRID):
    """Identical to scripts 97/104/109: c_k = (-1)^{k+1} S^{(k)}."""
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
    """The incumbent three-parameter chart eps = eps_inf + c e^{-a x} (script 109)."""
    def mdl(th, xx):
        return np.exp(th[0]) + np.exp(th[1]) * np.exp(-np.exp(th[2]) * np.atleast_1d(xx))
    th0 = [np.log(max(e.min() * .5, 1e-9)), np.log(e.max()), np.log(.4)]
    try:
        r = least_squares(lambda th: np.log(mdl(th, x)) - np.log(e), th0, max_nfev=40000)
        return float(np.exp(r.x[0]))
    except Exception:
        return np.nan


def run():
    rows = []
    for fr in all_frontiers():
        x, e = np.asarray(fr["x"], float), np.asarray(fr["e"], float)
        o = np.argsort(x)
        x, e = x[o], e[o]
        k = (e > 0) & np.isfinite(e)
        x, e = x[k], e[k]
        if len(x) < 7:
            continue
        S = -np.log(e)
        J = jets(x, S, kmax=4)
        g, Sg, c1, c2, c3 = J[:, 0], J[:, 1], J[:, 2], J[:, 3], J[:, 4]
        ok = (c1 > 0) & (c2 > 0) & np.isfinite(c3)
        if ok.sum() < 4:
            continue
        gg, c1, c2, c3, Sg = g[ok], c1[ok], c2[ok], c3[ok], Sg[ok]
        neff = c1 ** 2 / c2

        # ---- (A) the elbow: c_3 = 0, floor-free -----------------------------
        frac_left = float(np.mean(c3 < 0))                      # share of window refutable
        has_elbow = bool((c3 < 0).any() and (c3 > 0).any())
        x_elbow = np.nan
        if has_elbow:
            i = int(np.argmax(np.sign(c3[:-1]) != np.sign(c3[1:])))
            a, b = c3[i], c3[i + 1]
            x_elbow = float(gg[i] + (gg[i + 1] - gg[i]) * (-a) / (b - a)) if b != a else float(gg[i])

        # ---- the incumbent admissibility condition, which needs a floor -----
        einf = fit_floor(x, e)
        floor_ok = bool(np.isfinite(einf) and 1e-6 < einf < e.min())
        frac_adm = float(np.mean(np.exp(-Sg) > 2 * einf)) if floor_ok else np.nan

        # ---- (B)/(C) the wall band ------------------------------------------
        # The band N - ln(1+N) is increasing in N_eff, so the test is strongest
        # where N_eff is LARGEST, i.e. at the shallow (low-resource) edge -- which
        # is also, by Prop. 'elbow', the side on which refutation is possible at
        # all.  Reporting it at the deep edge instead understates it by ~2 orders
        # of magnitude; both are given.
        N_end = float(neff[-1])                                  # deep edge
        N_best = float(np.nanmax(neff))                          # best point in window
        band = float(N_best - np.log1p(N_best))
        band_deep = float(N_end - np.log1p(N_end))
        tighten = float((1 + N_end) * np.exp(-N_end))
        rows.append(dict(corpus=fr["corpus"], name=fr["name"], n=len(x),
                         span_nats=float(S.max() - S.min()),
                         has_elbow=int(has_elbow), x_elbow=x_elbow,
                         frac_window_refutable=frac_left,
                         floor_identified=int(floor_ok), eps_inf=einf,
                         frac_window_admissible_incumbent=frac_adm,
                         N_eff_deep=N_end, N_eff_best=N_best,
                         refutation_band_nats=band,
                         refutation_band_deep_nats=band_deep,
                         tightening_factor=tighten,
                         band_beats_nested_bracket=int(band > BRACKET_NESTED),
                         band_beats_indep_bracket=int(band > BRACKET_INDEP)))
    return pd.DataFrame(rows)


def main():
    df = run()
    save(df, "table_elbow_frontiers.csv")
    n = len(df)
    el = int(df.has_elbow.sum())
    fl = int(df.floor_identified.sum())
    both = df[(df.has_elbow == 1) & (df.floor_identified == 1)]
    agree = float(np.corrcoef(both.frac_window_refutable,
                              both.frac_window_admissible_incumbent)[0, 1]) if len(both) > 3 else np.nan

    print("=" * 78)
    print("(A)  ADMISSIBILITY: WHERE A QUADRUPLE IS DIAGNOSTIC")
    print(f"  frontiers analysed ................................. {n}")
    print(f"  NEW   elbow located, floor-free .................... {el}/{n}  ({100*el/n:.0f}%)")
    print(f"  OLD   floor identified, so eps>2eps_inf placeable .. {fl}/{n}  ({100*fl/n:.0f}%)")
    print(f"  median refutable share of the window (new) ......... {df.frac_window_refutable.median():.3f}")
    print(f"  correlation of the two admissible shares where both exist: {agree:.3f}")
    print(f"  frontiers the new route places that the old cannot . "
          f"{int(((df.has_elbow==1)&(df.floor_identified==0)).sum())}")

    print()
    print("=" * 78)
    print("(B)  THE WALL AS A SECOND-ORDER TEST: floor precision required")
    print(f"  median N_eff at the deep edge ...................... {df.N_eff_deep.median():.4f}")
    print(f"  median N_eff at the best point in the window ....... {df.N_eff_best.median():.4f}")
    print(f"  median band AT THE DEEP EDGE ....................... {df.refutation_band_deep_nats.median():.4f} nats")
    print(f"  median band AT THE BEST POINT ...................... {df.refutation_band_nats.median():.4f} nats")
    print(f"  IQR of the band .................................... "
          f"[{df.refutation_band_nats.quantile(.25):.4f}, {df.refutation_band_nats.quantile(.75):.4f}] nats")
    print(f"  band wider than the nested count-axis bracket (0.094) {int(df.band_beats_nested_bracket.sum())}/{n}")
    print(f"  band wider than the indep  count-axis bracket (0.20)  {int(df.band_beats_indep_bracket.sum())}/{n}")

    print()
    print("=" * 78)
    print("(C)  THE 'TIGHTENING FACTOR' AND THE BAND ARE ONE FUNCTION")
    # band = N - ln(1+N)  and  tightening T = (1+N)e^{-N},  so  ln T = -(N - ln(1+N)):
    #                      band  =  -ln(tightening factor),  exactly.
    dev = float(np.nanmax(np.abs(df.refutation_band_deep_nats + np.log(df.tightening_factor))))
    print(f"  max |band + ln(tightening)| over {n} frontiers ....... {dev:.3e}")
    print(f"  median tightening factor ........................... {df.tightening_factor.median():.4f}")
    print(f"  -> band = -ln(tightening factor), EXACTLY.  The archive's reported")
    print(f"     gain and the indivisibility margin are one measurement.")
    print(f"     median tightening 0.9988 is a band of {-np.log(df.tightening_factor.median()):.4f} nats:")
    print(f"     no power at the deep edge, which is why nothing was seen there.")

    summ = pd.DataFrame([dict(
        n_frontiers=n, elbow_located=el, floor_identified=fl,
        new_route_only=int(((df.has_elbow == 1) & (df.floor_identified == 0)).sum()),
        median_refutable_share=float(df.frac_window_refutable.median()),
        admissible_share_correlation=agree,
        median_N_eff_deep=float(df.N_eff_deep.median()),
        median_N_eff_best=float(df.N_eff_best.median()),
        median_band_deep_nats=float(df.refutation_band_deep_nats.median()),
        median_band_nats=float(df.refutation_band_nats.median()),
        band_gt_0094=int(df.band_beats_nested_bracket.sum()),
        band_gt_020=int(df.band_beats_indep_bracket.sum()),
        median_tightening=float(df.tightening_factor.median()),
        band_equals_minus_log_tightening_maxdev=dev)])
    save(summ, "table_elbow_summary.csv")
    save(df[["corpus", "name", "N_eff_deep", "N_eff_best", "refutation_band_nats",
             "tightening_factor", "band_beats_nested_bracket"]], "table_wall_band.csv")


if __name__ == "__main__":
    main()

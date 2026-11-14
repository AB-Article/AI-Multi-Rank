#!/usr/bin/env python3
"""129_plane_corpus.py -- the 117 frontiers read on the channel-depth plane.

No new data, no new fit, no kernel, no bandwidth, no smoother, no chart.  The
same 117 frontiers as scripts 68/122/125, read through Proposition 'plane':

    plot  N_eff = s^2/kappa  against  S = -ln eps.

Everything the theory says is a property of that one curve (Prop. 'plane',
'pencil', 'tangent', 'minarc', 'localprice'), and the only estimator needed is

    N_eff = (l[+] - l[-])^2 / (4 (l[-] - 2 l[0] + l[+]))          (eq. nhat)

which is a ratio of a squared first difference to a second difference, so the
SPACING CANCELS IDENTICALLY: no h appears anywhere below.

Five readings per frontier.

 (A) THE CURVE.  N_eff on every triple, consecutive and wide.

 (B) THE PENCIL.  Prop. 'pencil': every constant-sigma chart is a STRAIGHT LINE
     on the plane, its slope is sigma and its S-intercept is -ln eps_inf.  We fit
     that line (two points would suffice; least squares over all triples is used
     so the residual is informative) and report
        slope sigma, R^2 (straightness), and the floor from the intercept.
     R^2 decides the regime: straight => the intercept is the floor; curved =>
     the frontier is in the axiom-wall regime and eps_A is the floor to use.
     NOTE the slope is NOT used as a divisibility test: it is biased low by
     O(h^2) (sc. 127/129 both show it), which is the archive's own 'D1 plug-in
     biased downward' finding.  The exact test is (D).

 (C) THE TANGENT.  Prop. 'tangent': the two walls are a curve and its own
     tangent at the origin, so the two floor bounds are those walls solved for
     the floor -- eps_A = eps/(1+N_eff) and eps_W = eps e^{-N_eff}.  We check
     that these reproduce the determinant forms of script 125 (Prop.
     'floorbracket'), i.e. that the 1006d layer adds nothing beyond (eps, N_eff),
     and we compare all three floor readings with the incumbent fitted chart.

 (D) THE EXACT DIVISIBILITY TEST, PLACED.  Prop. 'tangent'/'elbow': a divisible
     frontier's curvature never rises, so the third divided difference of S must
     be >= 0 for every quadruple, and the test has power only at or left of the
     curvature maximum.  We locate that maximum WITHOUT A KERNEL (the largest
     second divided difference of ln eps) and report the widest admissible
     quadruple's standardised statistic.  The null is in script 130.

 (E) THE PRICE.  Prop. 'localprice': lambda* = s/sqrt(t N_eff), and lambda* = s
     exactly at the operating point.  Reported against the jet's sqrt(kappa/t)
     computed here (line ~246) and the Prony price of script 125.

Outputs: results/table_plane_frontiers.csv
         results/table_plane_summary.csv
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

HFRAC, NGRID = 0.40, 9                 # the incumbent jet, for comparison only
WIDE = 0.667                           # a triple/quadruple is 'wide' at 2/3 of the window
GRAIN = 1.0
TIE_BAND = 0.05
BRACKET_NESTED, BRACKET_INDEP = 0.094, 0.20


# ----------------------------------------------------------------- primitives
def dd(xs, ys):
    """Divided difference of any order, exact at any spacing."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return float(sum(ys[i] / np.prod([xs[i] - xs[j] for j in range(len(xs)) if j != i])
                     for i in range(len(xs))))


def dd_weights(xs):
    xs = np.asarray(xs, float)
    return np.array([1.0 / np.prod([xs[i] - xs[j] for j in range(len(xs)) if j != i])
                     for i in range(len(xs))])


def nhat(xs, ls):
    """eq. nhat: N_eff from three log-errors. The spacing cancels; works at any
    spacing via divided differences (s = -dd1, kappa = 2*dd2 of ln eps)."""
    s = -dd(xs[:2], ls[:2]) if False else None
    s1 = -(ls[2] - ls[0]) / (xs[2] - xs[0])          # central first difference
    k = 2.0 * dd(xs, ls)                             # kappa, exact 2nd divided difference
    if not np.isfinite(s1) or not np.isfinite(k) or k <= 0 or s1 <= 0:
        return None, s1, k
    return s1 * s1 / k, s1, k


def line_fit(S, N):
    S, N = np.asarray(S, float), np.asarray(N, float)
    if len(S) < 2 or np.ptp(S) <= 0:
        return np.nan, np.nan, np.nan
    A = np.column_stack([S, np.ones_like(S)])
    (sig, b), *_ = np.linalg.lstsq(A, N, rcond=None)
    pred = A @ np.array([sig, b])
    ss, tot = float(np.sum((N - pred) ** 2)), float(np.sum((N - N.mean()) ** 2))
    return float(sig), float(b), (1 - ss / tot if tot > 0 else np.nan)


def jets(x, S, kmax=4, hfrac=HFRAC, ng=NGRID):
    """The incumbent kernel jet, identical to scripts 81/97/104/109/122/125."""
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
        bb, *_ = np.linalg.lstsq(M * w[:, None], S * w, rcond=None)
        out.append([g, bb[0]] + [(-1) ** (k + 1) * bb[k] for k in range(1, kmax + 1)])
    return np.array(out)


def fit_floor(x, e):
    """The incumbent three-parameter chart (scripts 109/122/125)."""
    def mdl(th, xx):
        return np.exp(th[0]) + np.exp(th[1]) * np.exp(-np.exp(th[2]) * np.atleast_1d(xx))
    th0 = [np.log(max(e.min() * .5, 1e-9)), np.log(e.max()), np.log(.4)]
    try:
        r = least_squares(lambda th: np.log(mdl(th, x)) - np.log(e), th0, max_nfev=40000)
        return float(np.exp(r.x[0]))
    except Exception:
        return np.nan


def resid_sigma(x, e):
    """Per-model noise proxy: residual of the incumbent chart, in nats."""
    def mdl(th, xx):
        return np.exp(th[0]) + np.exp(th[1]) * np.exp(-np.exp(th[2]) * np.atleast_1d(xx))
    th0 = [np.log(max(e.min() * .5, 1e-9)), np.log(e.max()), np.log(.4)]
    try:
        r = least_squares(lambda th: np.log(mdl(th, x)) - np.log(e), th0, max_nfev=40000)
        dof = max(len(x) - 3, 1)
        return float(np.sqrt(np.sum(r.fun ** 2) / dof))
    except Exception:
        return np.nan


# ----------------------------------------------------------------------- run
def run():
    rows = []
    for fr in all_frontiers():
        x, e = np.asarray(fr["x"], float), np.asarray(fr["e"], float)
        o = np.argsort(x)
        x, e = x[o], e[o]
        k = (e > 0) & np.isfinite(e)
        x, e = x[k], e[k]
        if len(x) < 5:
            continue
        L, S = np.log(e), -np.log(e)
        Z = x[-1] - x[0]
        n = len(x)

        # ---------- (A) the curve, on consecutive and on wide triples --------
        cons, wide = [], []
        for j in range(n - 2):
            N, s1, kp = nhat(x[j:j + 3], L[j:j + 3])
            if N is not None:
                cons.append((S[j + 1], N, x[j + 1], s1, kp))
        for i in range(n):
            for j in range(i + 1, n - 1):
                for m in range(j + 1, n):
                    if x[m] - x[i] >= WIDE * Z:
                        N, s1, kp = nhat(x[[i, j, m]], L[[i, j, m]])
                        if N is not None:
                            wide.append((S[j], N, x[j], s1, kp))

        # ---------- (B) the pencil: one straight line ------------------------
        sig_c, b_c, r2_c = line_fit([p[0] for p in cons], [p[1] for p in cons]) \
            if len(cons) >= 3 else (np.nan, np.nan, np.nan)
        sig_w, b_w, r2_w = line_fit([p[0] for p in wide], [p[1] for p in wide]) \
            if len(wide) >= 3 else (np.nan, np.nan, np.nan)
        floor_int = np.nan
        if np.isfinite(sig_c) and sig_c < 0:
            floor_int = float(np.exp(-(-b_c / sig_c)))

        # ---------- (C) the two walls: valid at EVERY triple, so the bound is
        #               the minimum over them, and the best place to TEST is the
        #               triple of largest room (the walls touch at the floor).
        eps_A = eps_W = room_max = np.nan
        allt = cons + wide
        if allt:
            As = [np.exp(-p[0]) / (1.0 + p[1]) for p in allt]
            Ws = [np.exp(-p[0]) * np.exp(-p[1]) for p in allt]
            eps_A, eps_W = float(np.nanmin(As)), float(np.nanmin(Ws))
            rooms = [np.log(a / w) for a, w in zip(As, Ws) if w > 0 and np.isfinite(a / w)]
            room_max = float(np.nanmax(rooms)) if rooms else np.nan
        f_fit = fit_floor(x, e)
        sig_noise = resid_sigma(x, e)

        # ---------- (D) the exact test, placed at the curvature maximum ------
        kap = [(x[j + 1], 2.0 * dd(x[j:j + 3], L[j:j + 3])) for j in range(n - 2)]
        kap = [(xx, kk) for xx, kk in kap if np.isfinite(kk)]
        x_elbow = max(kap, key=lambda p: p[1])[0] if kap else np.nan
        rises = sum(1 for a, b in zip(kap, kap[1:]) if b[1] > a[1])
        # widest quadruple whose right edge is at or left of the curvature max
        best_z, best_span = np.nan, np.nan
        if np.isfinite(x_elbow) and np.isfinite(sig_noise) and sig_noise > 0:
            adm = np.where(x <= x_elbow + 1e-12)[0]
            if len(adm) >= 4:
                for a in range(len(adm) - 3):
                    for d_ in range(len(adm) - 1, a + 2, -1):
                        q = [adm[a], adm[(a + d_) // 3 + (0 if d_ > 2 else 0)], 0, adm[d_]]
                        # equally-spaced-in-index quadruple inside the admissible set
                        i0, i3 = adm[a], adm[d_]
                        i1 = adm[a + max(1, (d_ - a) // 3)]
                        i2 = adm[a + max(2, (2 * (d_ - a)) // 3)]
                        if len({i0, i1, i2, i3}) < 4:
                            continue
                        idx = sorted({i0, i1, i2, i3})
                        if len(idx) < 4:
                            continue
                        w = dd_weights(x[idx])
                        z = dd(x[idx], S[idx]) / (sig_noise * np.linalg.norm(w))
                        span = x[idx[-1]] - x[idx[0]]
                        if not np.isfinite(best_z) or z < best_z:
                            best_z, best_span = z, span
        # the same statistic with no placement rule, for the like-for-like column
        z_unplaced = np.nan
        if np.isfinite(sig_noise) and sig_noise > 0 and n >= 4:
            idx = [0, max(1, (n - 1) // 3), max(2, (2 * (n - 1)) // 3), n - 1]
            idx = sorted(set(idx))
            if len(idx) == 4:
                w = dd_weights(x[idx])
                z_unplaced = dd(x[idx], S[idx]) / (sig_noise * np.linalg.norm(w))

        # ---------- (E) the price -------------------------------------------
        J = jets(x, S)
        Nj = J[:, 2] ** 2 / np.maximum(J[:, 3], 1e-18)
        ij = int(np.argmin(np.abs(np.log(np.maximum(Nj, 1e-18) * GRAIN))))
        lam_jet = float(np.sqrt(max(J[ij, 3], 1e-18) / GRAIN))      # sqrt(kappa/t) at ITS point
        lam_plane = np.nan
        if cons:
            tgt = 1.0 / GRAIN
            bp = min(cons, key=lambda p: abs(np.log(p[1] / tgt)))
            lam_plane = float(bp[3] / np.sqrt(GRAIN * bp[1]))
            x_star_plane, N_star = bp[2], bp[1]
        else:
            x_star_plane, N_star = np.nan, np.nan
        x_star_jet = float(J[ij, 0])

        rows.append(dict(
            corpus=fr["corpus"], name=fr["name"], n=n, span=Z,
            n_triples=len(cons), n_wide=len(wide),
            sigma_plane=sig_c, line_r2=r2_c, sigma_wide=sig_w, line_r2_wide=r2_w,
            straight=int(np.isfinite(r2_c) and r2_c > 0.95),
            has_floor=int(np.isfinite(sig_c) and sig_c < 0),
            floor_intercept=floor_int, eps_A=eps_A, eps_W=eps_W,
            floor_fitted=f_fit, eps_min=float(e.min()), sigma_noise=sig_noise,
            n_admissible=int(np.sum(x <= x_elbow + 1e-12)) if np.isfinite(x_elbow) else 0,
            admissible_span=float(x_elbow - x[0]) if np.isfinite(x_elbow) else np.nan,
            x_elbow=x_elbow, elbow_frac=(x_elbow - x[0]) / Z if np.isfinite(x_elbow) else np.nan,
            curvature_rises=rises, n_kappa=len(kap),
            z_placed=best_z, span_placed=best_span, z_unplaced=z_unplaced,
            lam_plane=lam_plane, lam_jet=lam_jet,
            x_star_plane=x_star_plane, N_star=N_star, x_star_jet=x_star_jet,
            floor_room_nats=room_max,
        ))
    df = pd.DataFrame(rows)

    # ------------------------------------------------------------- report
    print("\n(A/B)  THE PENCIL -- one straight line on the channel-depth plane")
    r2 = df.line_r2.dropna()
    print(f"  frontiers with a legible curve ............... {len(r2)} of {len(df)}")
    print(f"  median R^2 of N_eff against S (consecutive) .. {r2.median():.4f}"
          f"   IQR [{r2.quantile(.25):.4f}, {r2.quantile(.75):.4f}]")
    r2w = df.line_r2_wide.dropna()
    print(f"  median R^2 on wide triples ................... {r2w.median():.4f}")
    print(f"  straight at R^2 > 0.95 ....................... {int(df.straight.sum())} of {len(r2)}"
          "   (constant-sigma: the intercept IS the floor)")
    s_ = df.sigma_plane.dropna()
    print(f"  median slope sigma ........................... {s_.median():+.3f}"
          f"   IQR [{s_.quantile(.25):+.3f}, {s_.quantile(.75):+.3f}]")
    print(f"  slope < 0 (a floor) .......................... {int(df.has_floor.sum())} of {len(s_)}")

    print("\n(C)  THE TANGENT -- the two walls are the two floor bounds")
    ok = df.dropna(subset=["eps_A", "eps_W", "floor_fitted"])
    print(f"  frontiers with all three floor readings ...... {len(ok)}")
    print(f"  fitted floor above eps_A (no CM frontier may) . {int((ok.floor_fitted > ok.eps_A).sum())}")
    print(f"  fitted floor above eps_W (would refute div.) .. {int((ok.floor_fitted > ok.eps_W).sum())}")
    ii = df.dropna(subset=["floor_intercept", "floor_fitted"])
    ii = ii[ii.straight == 1]
    if len(ii):
        rel = np.log(ii.floor_intercept / ii.floor_fitted).abs()
        print(f"  on the straight frontiers, |ln(intercept/fitted floor)| median {rel.median():.3f} nats"
              f"  ({len(ii)} frontiers)")
    rm = df.floor_room_nats.dropna()
    print(f"  best-triple room ln(eps_A/eps_W), median ..... {rm.median():.4f} nats"
          f"   IQR [{rm.quantile(.25):.4f}, {rm.quantile(.75):.4f}]"
          f"   (sc. 125: 0.0161)")
    print(f"  exceeds the count-axis bracket 0.20 / 0.094 .. "
          f"{int((rm > BRACKET_INDEP).sum())} / {int((rm > BRACKET_NESTED).sum())} of {len(rm)}")

    print("\n(D)  THE EXACT TEST, PLACED AT THE CURVATURE MAXIMUM (no kernel)")
    print(f"  curvature maximum located on ................. "
          f"{int(df.x_elbow.notna().sum())} of {len(df)}"
          f"   median position {df.elbow_frac.median():.2f} of the window")
    print(f"  frontiers whose curvature rises at all ....... "
          f"{int((df.curvature_rises > 0).sum())} of {int(df.n_kappa.gt(1).sum())}")
    zp, zu = df.z_placed.dropna(), df.z_unplaced.dropna()
    print(f"  placed   z < -1.645 .......................... {int((zp < -1.645).sum())} of {len(zp)}"
          f"   median z {zp.median():+.2f}")
    print(f"  unplaced z < -1.645 .......................... {int((zu < -1.645).sum())} of {len(zu)}"
          f"   median z {zu.median():+.2f}")
    print(f"  median per-model noise proxy ................. {df.sigma_noise.median():.4f} nats"
          f"   (archive's nominal sigma: 0.020)")
    print(f"  admissible span left of the curvature max .... median "
          f"{df.admissible_span.median():.2f} e-folds of {df.span.median():.2f}"
          f"   ({int(df.n_admissible.median())} models)")

    print("\n(E)  THE PRICE -- lambda* = s/sqrt(t N_eff)")
    pp = df.dropna(subset=["lam_plane", "lam_jet"])
    print(f"  frontiers with both prices ................... {len(pp)}")
    print(f"  median |ln(lam_plane/lam_jet)| ............... "
          f"{np.log(pp.lam_plane / pp.lam_jet).abs().median():.3f}"
          "   (sc. 125 Prony price vs jet: 0.477)")

    save(df, "table_plane_frontiers.csv")
    summ = pd.DataFrame([dict(
        n_frontiers=len(df), n_legible=int(len(r2)),
        median_line_r2=float(r2.median()), median_line_r2_wide=float(r2w.median()),
        straight_at_095=int(df.straight.sum()), median_sigma=float(s_.median()),
        n_floor=int(df.has_floor.sum()),
        fitted_above_epsA=int((ok.floor_fitted > ok.eps_A).sum()),
        fitted_above_epsW=int((ok.floor_fitted > ok.eps_W).sum()),
        median_floor_room=float(rm.median()),
        elbow_located=int(df.x_elbow.notna().sum()),
        median_elbow_frac=float(df.elbow_frac.median()),
        curvature_rises=int((df.curvature_rises > 0).sum()),
        placed_refute=int((zp < -1.645).sum()), n_placed=int(len(zp)),
        unplaced_refute=int((zu < -1.645).sum()), n_unplaced=int(len(zu)),
        median_z_placed=float(zp.median()), median_z_unplaced=float(zu.median()),
        median_sigma_noise=float(df.sigma_noise.median()),
        median_price_gap=float(np.log(pp.lam_plane / pp.lam_jet).abs().median()),
    )])
    save(summ, "table_plane_summary.csv")


if __name__ == "__main__":
    run()

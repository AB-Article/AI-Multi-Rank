#!/usr/bin/env python3
"""THEORY REPLACEMENTS FOR THREE PREVIOUSLY EMPIRICAL STATEMENTS.

This script verifies, at 50-digit precision and on the archived pair table,
the three analytic statements that replace measurements in the revision of
2026-10-04.

  (1) MARGIN-RATIO IDENTITY (Proposition "margin and deciding ratio",
      Section 7 of paper.tex).  For ANY two candidates -- no law assumed --

          margin  =  ln(eps_1/eps_2) * (1 - alpha / (2 sbar)),
          sbar    =  ln(eps_1/eps_2) / ln(t_2/t_1),

      where margin = sigma_lambda*(x_2) - sigma_lambda*(x_1) is the quantity
      the resource-aware objective ranks by.  The decision margin is the
      accuracy gap times the deciding ratio's relative distance from unity, so
      "the ratio is near one" and "the margin is near zero" are one statement,
      not two.  In the floor coordinate the same structure is exact with
      u_i = eps_inf/eps_i:

          DeltaJ  = 1/2 ln[ u_2 (1-u_2) / (u_1 (1-u_1)) ],
          sign(DeltaJ) = sign(1 - u_1 - u_2) = sign(1 - 2 eps_inf/hm).

  (2) COST OF A MISQUOTED FLOOR (Proposition "cost of a misquoted floor",
      Section 9).  Quoting phi * eps_inf and applying the accuracy-only rule
      on the law costs exactly

          R(phi) = ln cosh( 1/2 ln(2 phi - 1) ),     phi > 1/2,

      with a pole at phi = 1/2 and R ~ 1/2 ln(2 phi) - ln 2 as phi -> oo.
      This replaces the measured floor-sensitivity profile.

  (3) COVARIANCE-FLOOR BIAS FOR A GENERAL SPECTRUM (Section 9 remark and
      Online Appendix).  Under the item-level spectral law with failures
      independent across models given the item, write the reducible failure
      probability h(x) = int e^{-ax} dnu(a).  Then the pairwise estimator
      obeys exactly

          hat_eps_inf - eps_inf = (1 - eps_inf) D / (1 - h_1 - h_2 + h_12),
          D = h(x_1 + x_2) - h(x_1) h(x_2)  >=  0,

      The sign requires the two models to be compared along ONE resource
      direction (x_1, x_2 collinear), where Chebyshev's integral inequality
      applies to the scalar a.v; D = 0 then holds iff the reducible spectrum
      is a single atom in that direction.  With non-parallel x_1, x_2 in
      n >= 2 dimensions the sign can fail, and this script exhibits a
      spectrum for which D < 0 and the estimator returns a NEGATIVE floor.
      Where D >= 0 holds, the bias is predicted by the fitted law
      (D = h_12 - h_1 h_2, with h_12 the reducible error at the product
      resource t_1 t_2) and bounded fit-free by D <= min(h_1, h_2).
      Equivariance to appended always-failed items holds for every spectrum.

      The harmonic test may be run on raw rates with a raw floor, since both
      sides carry the same factor 1 - q_0; the estimator itself is NOT
      homogeneous, so chance-corrected rates must not be substituted into it.
      Both halves of that statement are checked below.

Run:  python3 50_theory_replacements.py  [--pairs PATH]
"""

import argparse
import csv
import math
import os
import random
import sys

from mpmath import mp, mpf, log, cosh, exp, matrix  # noqa: F401

mp.dps = 50
TOL = mpf(10) ** (-40)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PAIRS = os.path.join(
    HERE, os.pardir, "results", "table_exchange_rate_pairs.csv"
)


# ----------------------------------------------------------------------
# (1) the margin-ratio identity
# ----------------------------------------------------------------------
def check_margin_identity(ntrials=20000, seed=20261004):
    """margin = ln(eps1/eps2) (1 - alpha/(2 sbar)), for arbitrary pairs."""
    rng = random.Random(seed)
    worst = mpf(0)
    for _ in range(ntrials):
        e1 = mpf(rng.uniform(1e-4, 1.0))
        e2 = e1 * mpf(rng.uniform(1e-3, 0.999))       # eps_2 < eps_1
        t1 = mpf(rng.uniform(1.0, 1e3))
        t2 = t1 * mpf(rng.uniform(1.001, 1e3))        # t_2 > t_1
        alpha = mpf(rng.uniform(1e-3, 3.0))
        margin = log(e1 / e2) - alpha / 2 * log(t2 / t1)
        sbar = log(e1 / e2) / log(t2 / t1)
        ident = log(e1 / e2) * (1 - alpha / (2 * sbar))
        worst = max(worst, abs(margin - ident))
    return worst


def check_floor_coordinate(ntrials=20000, seed=11):
    """DeltaJ form and the equivalence of its sign with 1 - 2 eps_inf/hm."""
    rng = random.Random(seed)
    worst = mpf(0)
    signs_ok = True
    for _ in range(ntrials):
        einf = mpf(rng.uniform(1e-5, 0.3))
        e1 = einf * mpf(rng.uniform(1.001, 50.0))
        e2 = einf * mpf(rng.uniform(1.001, 50.0))
        if e2 >= e1:
            e1, e2 = e2, e1
        if e1 == e2:
            continue
        dJ = (log(e2 - einf) / 2 - log(e2)) - (log(e1 - einf) / 2 - log(e1))
        u1, u2 = einf / e1, einf / e2
        form = log(u2 * (1 - u2) / (u1 * (1 - u1))) / 2
        worst = max(worst, abs(dJ - form))
        hm = 2 * e1 * e2 / (e1 + e2)
        if (dJ > 0) != (1 - 2 * einf / hm > 0):
            signs_ok = False
    return worst, signs_ok


def check_margin_on_panel(path):
    """The identity on every archived Pareto pair (float precision)."""
    if not os.path.exists(path):
        return None
    worst, n = 0.0, 0
    stats = {"gap": [], "rel": [], "margin": [], "relfloor": [],
             "tol": [], "tolfloor": []}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            e1, e2 = float(r["eps_hi"]), float(r["eps_lo"])
            t1, t2 = float(r["t_hi"]), float(r["t_lo"])
            a, s = float(r["alpha_fit"]), float(r["sbar"])
            einf = float(r["eps_inf_fit"])
            gap = math.log(e1 / e2)
            margin = gap - a / 2 * math.log(t2 / t1)
            ident = gap * (1 - a / (2 * s))
            worst = max(worst, abs(margin - ident))
            n += 1
            hm = 2 * e1 * e2 / (e1 + e2)
            stats["gap"].append(gap)
            stats["rel"].append(abs(1 - a / (2 * s)))
            stats["margin"].append(abs(margin))
            stats["relfloor"].append(abs(1 - hm / (2 * einf)))
            stats["tol"].append(abs(1 - 2 * s / a))
            stats["tolfloor"].append(abs(1 - 2 * einf / hm))
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    return {
        "n": n,
        "max_dev": worst,
        "median_gap": med(stats["gap"]),
        "median_rel_alpha": med(stats["rel"]),
        "median_tol_alpha": med(stats["tol"]),
        "median_rel_floor": med(stats["relfloor"]),
        "median_tol_floor": med(stats["tolfloor"]),
        "median_margin": med(stats["margin"]),
        "median_gap_times_rel": med(
            [g * r for g, r in zip(stats["gap"], stats["rel"])]
        ),
        "product_of_medians": med(stats["gap"]) * med(stats["rel"]),
        "frac_in_band": sum(m < 0.05 for m in stats["margin"]) / n,
    }


# ----------------------------------------------------------------------
# (2) the cost of a misquoted floor
# ----------------------------------------------------------------------
def R_phi(phi):
    """Analytic on-law regret of quoting phi * eps_inf."""
    phi = mpf(phi)
    if phi <= mpf(1) / 2:
        return mp.inf
    return log(cosh(log(2 * phi - 1) / 2))


def check_misquoted_floor(ntrials=4000, seed=7):
    """Brute force: maximise J with the wrong floor, score with the right one."""
    rng = random.Random(seed)
    worst = mpf(0)
    for _ in range(ntrials):
        einf = mpf(rng.uniform(1e-4, 0.2))
        phi = mpf(rng.uniform(0.51, 20.0))
        # the rule driven by the quoted floor selects eps = 2 phi eps_inf
        eps_sel = 2 * phi * einf
        # true regret at that operating point, row 8 of Table 1
        p0 = einf / eps_sel
        zeta = log(p0 / (1 - p0))
        true_R = log(cosh(zeta / 2))
        worst = max(worst, abs(true_R - R_phi(phi)))
    return worst


# ----------------------------------------------------------------------
# (3) the covariance floor under a general spectrum
# ----------------------------------------------------------------------
def covfloor_estimator(e1, e2, e12):
    return (e12 - e1 * e2) / (1 - e1 - e2 + e12)


def spectrum_h(nu, x):
    """h(x) = int e^{-a x} d nu(a) for a discrete probability measure nu."""
    return sum(w * exp(-a * x) for a, w in nu)


def check_covfloor_general(ntrials=4000, seed=3):
    """Exact bias formula, the sign of D, and the single-atom equality case."""
    rng = random.Random(seed)
    worst_bias, worst_single, min_D = mpf(0), mpf(0), mpf(1)
    for _ in range(ntrials):
        einf = mpf(rng.uniform(1e-4, 0.4))
        k = rng.randint(1, 5)
        ws = [mpf(rng.uniform(0.05, 1.0)) for _ in range(k)]
        tot = sum(ws)
        nu = [(mpf(rng.uniform(0.02, 3.0)), w / tot) for w in ws]
        x1 = mpf(rng.uniform(0.05, 6.0))
        x2 = mpf(rng.uniform(0.05, 6.0))
        h1, h2 = spectrum_h(nu, x1), spectrum_h(nu, x2)
        h12 = spectrum_h(nu, x1 + x2)
        # scalar x: this is the collinear case, where D >= 0 is a theorem
        e1 = einf + (1 - einf) * h1
        e2 = einf + (1 - einf) * h2
        e12 = einf + (1 - einf) * h12
        est = covfloor_estimator(e1, e2, e12)
        D = h12 - h1 * h2
        pred = (1 - einf) * D / (1 - h1 - h2 + h12)
        worst_bias = max(worst_bias, abs((est - einf) - pred))
        min_D = min(min_D, D)                      # Chebyshev: D >= 0
        if k == 1:                                 # one reducible channel
            worst_single = max(worst_single, abs(est - einf))
    return worst_bias, min_D, worst_single


def check_covfloor_gamma(ntrials=2000, seed=5):
    """Closed form for the maximum-entropy (Gamma) reducible spectrum."""
    rng = random.Random(seed)
    worst = mpf(0)
    for _ in range(ntrials):
        kk = mpf(rng.uniform(0.3, 6.0))
        beta = mpf(rng.uniform(0.2, 5.0))
        x1 = mpf(rng.uniform(0.05, 8.0))
        x2 = mpf(rng.uniform(0.05, 8.0))
        h = lambda x: (1 + x / beta) ** (-kk)       # noqa: E731
        D_direct = h(x1 + x2) - h(x1) * h(x2)
        D_form = h(x1) * h(x2) * (
            ((1 + x1 / beta) * (1 + x2 / beta) / (1 + (x1 + x2) / beta)) ** kk - 1
        )
        worst = max(worst, abs(D_direct - D_form))
    return worst


def check_covfloor_equivariance(ntrials=4000, seed=9):
    """f -> (f+u)/(1+u) for an arbitrary spectrum, not just two classes."""
    rng = random.Random(seed)
    worst = mpf(0)
    for _ in range(ntrials):
        e1 = mpf(rng.uniform(1e-3, 0.6))
        e2 = mpf(rng.uniform(1e-3, 0.6))
        lo = max(e1 + e2 - 1, e1 * e2)
        e12 = mpf(rng.uniform(float(lo) + 1e-9, float(min(e1, e2))))
        u = mpf(rng.uniform(1e-4, 0.5))
        base = covfloor_estimator(e1, e2, e12)
        shifted = covfloor_estimator(
            (e1 + u) / (1 + u), (e2 + u) / (1 + u), (e12 + u) / (1 + u)
        )
        worst = max(worst, abs(shifted - (base + u) / (1 + u)))
    return worst


def check_covfloor_bound(ntrials=4000, seed=13):
    """Fit-free bound: D <= min(h_1, h_2), hence bias <= min_i(eps_i-eps_inf)/..."""
    rng = random.Random(seed)
    ok = True
    for _ in range(ntrials):
        k = rng.randint(1, 4)
        ws = [mpf(rng.uniform(0.05, 1.0)) for _ in range(k)]
        tot = sum(ws)
        nu = [(mpf(rng.uniform(0.02, 3.0)), w / tot) for w in ws]
        x1 = mpf(rng.uniform(0.05, 6.0))
        x2 = mpf(rng.uniform(0.05, 6.0))
        h1, h2 = spectrum_h(nu, x1), spectrum_h(nu, x2)
        D = spectrum_h(nu, x1 + x2) - h1 * h2
        if D > min(h1, h2) + TOL:
            ok = False
    return ok


def check_noncollinear_counterexample():
    """D can be NEGATIVE when x_1, x_2 are not parallel and n >= 2.

    Chebyshev's integral inequality orders a scalar.  In several resources,
    a -> e^{-a.x_1} and a -> e^{-a.x_2} are only coordinatewise decreasing,
    which does not make them positively correlated under an arbitrary nu.
    """
    nu = [((mpf(2), mpf("0.2")), mpf("0.5")), ((mpf("0.2"), mpf(2)), mpf("0.5"))]
    x1 = (mpf("1.5"), mpf("0.1"))
    x2 = (mpf("0.1"), mpf("1.5"))
    hv = lambda x: sum(                                   # noqa: E731
        w * exp(-(a[0] * x[0] + a[1] * x[1])) for a, w in nu
    )
    h1, h2 = hv(x1), hv(x2)
    h12 = hv((x1[0] + x2[0], x1[1] + x2[1]))
    D = h12 - h1 * h2
    einf = mpf("0.1")
    e1, e2 = einf + (1 - einf) * h1, einf + (1 - einf) * h2
    e12 = einf + (1 - einf) * h12
    est = covfloor_estimator(e1, e2, e12)
    pred = einf + (1 - einf) * D / (1 - h1 - h2 + h12)
    return D, est, einf, abs(est - pred)


def check_collinear_sign(ntrials=4000, seed=17, dim=3):
    """With x_1, x_2 parallel in n dimensions, D >= 0 is restored."""
    rng = random.Random(seed)
    worst = mpf(1)
    for _ in range(ntrials):
        k = rng.randint(1, 4)
        ws = [mpf(rng.uniform(0.05, 1.0)) for _ in range(k)]
        tot = sum(ws)
        nu = [
            (tuple(mpf(rng.uniform(0.01, 3.0)) for _ in range(dim)), w / tot)
            for w in ws
        ]
        v = tuple(mpf(rng.uniform(0.01, 2.0)) for _ in range(dim))
        c1, c2 = mpf(rng.uniform(0.05, 5.0)), mpf(rng.uniform(0.05, 5.0))
        hv = lambda c: sum(                                # noqa: E731
            w * exp(-c * sum(ai * vi for ai, vi in zip(a, v))) for a, w in nu
        )
        worst = min(worst, hv(c1 + c2) - hv(c1) * hv(c2))
    return worst


def check_chance_correction(ntrials=4000, seed=23):
    """Two statements, one true and one false, about the chance correction.

    True:  the harmonic test run on raw rates with a raw floor gives the same
           decision as on chance-corrected rates, since hm and the floor are
           both divided by 1 - q_0 (hm is 1-homogeneous).
    False: the estimator commutes with the chance correction.  It is not
           1-homogeneous, so chance-corrected rates must NOT be fed to it.
    """
    rng = random.Random(seed)
    decision_flips_homog = 0
    estimator_not_homogeneous = 0
    for _ in range(ntrials):
        e1 = mpf(rng.uniform(1e-3, 0.5))
        e2 = mpf(rng.uniform(1e-3, 0.5))
        lo = max(e1 + e2 - 1, e1 * e2)
        e12 = mpf(rng.uniform(float(lo) + 1e-9, float(min(e1, e2))))
        q0 = mpf(rng.uniform(0.05, 0.5))
        kappa = 1 - q0
        einf = covfloor_estimator(e1, e2, e12)
        hm = 2 * e1 * e2 / (e1 + e2)
        # (a) rescale the three quantities that enter the test
        raw = hm > 2 * einf
        corr = (hm / kappa) > 2 * (einf / kappa)
        if raw != corr:
            decision_flips_homog += 1
        # (b) feed chance-corrected rates to the estimator instead
        bad = covfloor_estimator(e1 / kappa, e2 / kappa, e12 / kappa)
        if abs(bad - einf / kappa) > mpf("1e-12"):
            estimator_not_homogeneous += 1
    return decision_flips_homog, estimator_not_homogeneous, ntrials


def check_chance_models(ntrials=2000, seed=29, q0=mpf("0.25")):
    """Two readings of a chance level, with opposite consequences.

    Shared: a q0 fraction of items is answered correctly by EVERY model, so
      raw rates are eps_i^r = (1-q0) eps_i and eps_12^r = (1-q0) eps_12.
      The raw population is again an item-level model, with atom
      (1-q0) eps_inf, so (*) on raw rates still bounds the raw floor above.
    Independent: each model independently guesses right with probability q0
      on the items it would otherwise fail, so eps_12^r = (1-q0)^2 eps_12.
      The shared always-failed mass is then smaller and (*) UNDERSTATES.
    """
    rng = random.Random(seed)
    below_shared = below_indep = 0
    for _ in range(ntrials):
        einf = mpf(rng.uniform(1e-3, 0.3))
        k = rng.randint(1, 4)
        ws = [mpf(rng.uniform(0.05, 1.0)) for _ in range(k)]
        tot = sum(ws)
        nu = [(mpf(rng.uniform(0.02, 3.0)), w / tot) for w in ws]
        x1 = mpf(rng.uniform(0.3, 5.0))
        x2 = mpf(rng.uniform(0.3, 5.0))
        h1, h2 = spectrum_h(nu, x1), spectrum_h(nu, x2)
        h12 = spectrum_h(nu, x1 + x2)
        e1 = einf + (1 - einf) * h1
        e2 = einf + (1 - einf) * h2
        e12 = einf + (1 - einf) * h12
        kap = 1 - q0
        raw_floor = kap * einf
        if covfloor_estimator(kap * e1, kap * e2, kap * e12) < raw_floor:
            below_shared += 1
        if covfloor_estimator(kap * e1, kap * e2, kap * kap * e12) < raw_floor:
            below_indep += 1
    return below_shared, below_indep, ntrials


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []

    def rec(name, value, target=""):
        rows.append({"check": name, "value": value, "target": target})
        print(f"{name:<52s} {value!s:<26s} {target}")

    print("=" * 96)
    print("(1) margin-ratio identity")
    print("=" * 96)
    rec("random pairs, max |margin - identity| (50 dps)",
        mp.nstr(check_margin_identity(), 8), "0")
    dJ_dev, signs_ok = check_floor_coordinate()
    rec("floor coordinate, max |DeltaJ - closed form|",
        mp.nstr(dJ_dev, 8), "0")
    rec("sign(DeltaJ) == sign(1 - 2 eps_inf/hm) always", str(signs_ok), "True")

    panel = check_margin_on_panel(args.pairs)
    if panel:
        rec("archived pairs checked", panel["n"], "13278")
        rec("max |margin - identity| on archived pairs",
            f"{panel['max_dev']:.3e}", "0 (machine precision)")
        rec("median accuracy gap (nats)", f"{panel['median_gap']:.4f}", "")
        rec("median |1 - 2 sbar/alpha|  (tolerance on alpha)",
            f"{panel['median_tol_alpha']:.4f}", "paper 0.307")
        rec("median |1 - hm/(2 eps_inf)| (tolerance on the floor)",
            f"{panel['median_rel_floor']:.4f}", "paper 0.235")
        rec("median |1 - alpha/(2 sbar)| (margin/gap)",
            f"{panel['median_rel_alpha']:.4f}", "0.3089, not the same stat")
        rec("median |1 - 2 eps_inf/hm|   (floor counterpart)",
            f"{panel['median_tol_floor']:.4f}", "0.2538")
        rec("median |margin| (nats)", f"{panel['median_margin']:.4f}",
            "paper 0.093")
        rec("median of gap * |1 - alpha/(2 sbar)|",
            f"{panel['median_gap_times_rel']:.4f}", "equals the line above")
        rec("product of the two medians (NOT the margin)",
            f"{panel['product_of_medians']:.4f}", "medians do not multiply")
        rec("fraction of pairs inside the 0.05-nat band",
            f"{panel['frac_in_band']:.4f}", "paper 0.32")
    else:
        print(f"  (pair table not found at {args.pairs}; skipped)")

    print()
    print("=" * 96)
    print("(2) cost of a misquoted floor")
    print("=" * 96)
    rec("max |brute force - ln cosh(ln(2 phi -1)/2)|",
        mp.nstr(check_misquoted_floor(), 8), "0")
    for phi, panel_val in [("0.5", "1.03 (grid-truncated)"),
                           ("0.7", "0.111"), ("1.5", "0.083"),
                           ("2.0", "0.120")]:
        val = R_phi(phi)
        rec(f"R(phi = {phi})",
            "inf" if val == mp.inf else mp.nstr(val, 6),
            f"panel R_sigma {panel_val}")
    rec("asymmetry R(0.6)/R(1/0.6)",
        mp.nstr(R_phi("0.6") / R_phi(1 / mpf("0.6")), 6), "> 1, diverges at 1/2")
    rec("asymmetry R(0.55)/R(1/0.55)",
        mp.nstr(R_phi("0.55") / R_phi(1 / mpf("0.55")), 6), "> previous line")

    print()
    print("=" * 96)
    print("(3) covariance floor under a general spectrum")
    print("=" * 96)
    wb, minD, wsingle = check_covfloor_general()
    rec("max |bias - (1-eps_inf) D / (1-h1-h2+h12)|", mp.nstr(wb, 8), "0")
    rec("min D, scalar (= collinear) resource", mp.nstr(minD, 8), ">= 0")
    rec("min D, collinear x1,x2 in 3 resources",
        mp.nstr(check_collinear_sign(), 8), ">= 0")
    Dn, estn, einfn, devn = check_noncollinear_counterexample()
    rec("NON-collinear counterexample: D", mp.nstr(Dn, 6), "< 0 (sign fails)")
    rec("  its estimate vs true floor",
        f"{mp.nstr(estn, 6)} vs {mp.nstr(einfn, 6)}", "estimator goes negative")
    rec("  bias formula still exact there", mp.nstr(devn, 8), "0")
    rec("max |bias| for a single reducible atom", mp.nstr(wsingle, 8), "0")
    rec("Gamma spectrum, max |D - closed form|",
        mp.nstr(check_covfloor_gamma(), 8), "0")
    rec("max equivariance residual, general spectrum",
        mp.nstr(check_covfloor_equivariance(), 8), "0")
    rec("fit-free bound D <= min(h1,h2) holds",
        str(check_covfloor_bound()), "True")
    flips, nonhomog, ntot = check_chance_correction()
    rec("harmonic test: decisions changed by rescaling",
        f"{flips}/{ntot}", "0 (hm and floor share 1-q0)")
    rec("estimator fed corrected rates: disagreements",
        f"{nonhomog}/{ntot}", "most: it is not 1-homogeneous")
    bs, bi, nt = check_chance_models()
    rec("shared chance mass: est below the raw floor",
        f"{bs}/{nt}", "0 (still an upper bound)")
    rec("independent guessing: est below the raw floor",
        f"{bi}/{nt}", "many (bound lost)")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["check", "value", "target"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

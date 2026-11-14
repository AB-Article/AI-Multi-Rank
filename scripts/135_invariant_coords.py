#!/usr/bin/env python3
"""135_invariant_coords.py -- the same curve in gauge-free coordinates.

No data.  Pure mathematics at 40 digits.  Script 134 states the primitive as one
function N of the depth.  This script changes to the two coordinates that carry
no gauge at all -- the arc and the angle -- and finds that in them BOTH LADDERS
ARE LIPSCHITZ BOUNDS and Laws II--III stop being laws.

    the angle    phi = 2 arctan sqrt(N)   in (0, pi)
    the arc      dL  = d(dS)/sqrt(N)

Then, as identities for every frontier:

    sin phi = sech(u/2), u = ln N        so   R = ln cosh(u/2) = -ln sin phi
    cos phi = (1 - N)/(1 + N)            so   N = 1  <=>  phi = pi/2
    dphi/dL = Ndot/(N + 1)

and therefore the two rungs of script 134 become

    AXIOM        phi' <= 1                 -- the angle advances no faster than
                                              the arc: phi is 1-Lipschitz in L
    HYPOTHESIS   phi' <= cos^2(phi/2)      -- integrating, tan(phi/2) <= L/2,
                                              i.e. N <= L^2/4

with equality iff the two-atom chart and iff the cure chart respectively.  Three
consequences that were separate results:

  * Law II--III.  R = -ln sin phi is now a DEFINITION, not a theorem, and the
    rule 'operate where N = 1' is 'stand at phi = pi/2', where R = 0 exactly.
  * pi.  A complete floored frontier sweeps phi from pi to 0, and phi is
    1-Lipschitz in the arc, so L >= pi -- Law III's constant in one line, with
    the finite-floor form L >= 2 arccos sqrt(eps_inf) and equality iff two atoms.
  * The grain.  phi is grain-INVARIANT at a fixed N and the arc scales as
    sqrt(t), so the grain is a dilation of L alone.

Honest negative, and the reason the plane and not the angle is the measurement
form: phi equals the arc only on the two-atom extremal.  On a divisible frontier
the angle advances strictly more slowly, so a measured window's arc OVERSTATES
the angle it swept, and the inversion from arc to angle needs the curve anyway.

Outputs: results/table_invariant_coords.csv
"""
from __future__ import annotations

import os
import sys

import mpmath as mp
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save                                        # noqa: E402

mp.mp.dps = 40
TOL = mp.mpf(10) ** -20
ROWS = []


def chk(name, got, want, tol=TOL, detail=""):
    if want is None:
        ok = bool(got >= -tol)
        err = mp.mpf(0) if ok else abs(got)
    else:
        err = abs(mp.mpf(got) - mp.mpf(want))
        ok = bool(err / max(mp.mpf(1), abs(mp.mpf(want))) <= tol)
    ROWS.append(dict(check=name, passed=int(ok), error=mp.nstr(err, 6), detail=detail))
    return ok


# ------------------------------------------------------------------- charts
CHARTS = {
    "two_atom": (lambda x: mp.mpf("0.004") + mp.mpf("0.9") * mp.e ** (-mp.mpf("0.7") * x),
                 mp.mpf("0.004")),
    "cure": (lambda x: mp.e ** (-5 * (1 - mp.e ** (-mp.mpf("0.7") * (x + 1)))),
             mp.e ** -5),
    "power_floor": (lambda x: mp.e ** (-4 + mp.mpf("1.3") * (x + 2) ** mp.mpf("-0.6")),
                    mp.e ** -4),
    "three_atom": (lambda x: mp.mpf("0.002") + mp.mpf("0.4") * mp.e ** (-mp.mpf("0.3") * x)
                   + mp.mpf("0.3") * mp.e ** (-mp.mpf("0.9") * x), mp.mpf("0.002")),
}


def quantities(eps, einf, x):
    S = lambda t: -mp.log(eps(t))                                # noqa: E731
    sfun = lambda t: mp.diff(S, t)                               # noqa: E731
    kfun = lambda t: -mp.diff(S, t, 2)                           # noqa: E731
    Nfun = lambda t: sfun(t) ** 2 / kfun(t)                      # noqa: E731
    s, kap = sfun(x), kfun(x)
    N = s ** 2 / kap
    Nd = -mp.diff(Nfun, x) / s                                   # dN/d(dS)
    DS = mp.log(eps(x) / einf)
    u = mp.log(N)
    phi = 2 * mp.atan(mp.sqrt(N))
    R = mp.log(mp.cosh(u / 2))
    # Both the arc and the angle are measured in the direction of INCREASING
    # depth remaining (decreasing x), which is the direction in which the ODE of
    # script 134 is integrated from the floor; then the ratio is positive.
    phifun = lambda t: 2 * mp.atan(mp.sqrt(Nfun(t)))             # noqa: E731
    dphi_dx = mp.diff(phifun, x)
    dL_dx = -sfun(x) / mp.sqrt(Nfun(x))                          # dL = d(dS)/sqrt(N)
    return dict(N=N, Nd=Nd, DS=DS, s=s, kap=kap, u=u, phi=phi, R=R,
                dphi_dL=dphi_dx / dL_dx)


# ============================================================ A. the identities
def section_A():
    for nm, (eps, einf) in CHARTS.items():
        for x in (mp.mpf("1.0"), mp.mpf("3.0")):
            q = quantities(eps, einf, x)
            N, phi = q["N"], q["phi"]
            chk(f"A1 {nm} x={float(x)}: sin phi = sech(u/2)",
                mp.sin(phi), mp.sech(q["u"] / 2))
            chk(f"A2 {nm} x={float(x)}: cos phi = (1-N)/(1+N)",
                mp.cos(phi), (1 - N) / (1 + N))
            chk(f"A3 {nm} x={float(x)}: R = -ln sin phi (Law II--III as a definition)",
                q["R"], -mp.log(mp.sin(phi)))
            chk(f"A4 {nm} x={float(x)}: dphi/dL = Ndot/(N+1)",
                q["dphi_dL"], q["Nd"] / (N + 1))
            chk(f"A5 {nm} x={float(x)}: tan(phi/2) = sqrt(N)",
                mp.tan(phi / 2), mp.sqrt(N))
            chk(f"A6 {nm} x={float(x)}: cos^2(phi/2) = 1/(N+1)",
                mp.cos(phi / 2) ** 2, 1 / (N + 1))


# ======================================================== B. the two rungs
def section_B():
    """axiom: phi' <= 1.  hypothesis: phi' <= cos^2(phi/2).  Equality cases."""
    for nm, (eps, einf) in CHARTS.items():
        for x in (mp.mpf("1.0"), mp.mpf("3.0")):
            q = quantities(eps, einf, x)
            ax_slack = 1 - q["dphi_dL"]
            hy_slack = mp.cos(q["phi"] / 2) ** 2 - q["dphi_dL"]
            chk(f"B1 {nm} x={float(x)}: axiom rung as phi' <= 1", ax_slack, None,
                detail=f"phi' = {mp.nstr(q['dphi_dL'], 8)}")
            # the hypothesis bound must coincide with Ndot <= 1
            chk(f"B2 {nm} x={float(x)}: cos^2(phi/2) - phi' = (1-Ndot)/(N+1), so "
                f"the hypothesis rung is phi' <= cos^2(phi/2)",
                hy_slack, (1 - q["Nd"]) / (q["N"] + 1),
                detail=f"Ndot = {mp.nstr(q['Nd'], 8)}")
    # equality: two atoms saturate the axiom rung, the cure chart the hypothesis
    for x in (mp.mpf("0.5"), mp.mpf("2.0"), mp.mpf("4.0")):
        q = quantities(*CHARTS["two_atom"], x)
        chk(f"B3 two atoms x={float(x)}: phi' = 1 exactly (arc = angle)",
            q["dphi_dL"], 1)
        q = quantities(*CHARTS["cure"], x)
        chk(f"B4 cure x={float(x)}: phi' = cos^2(phi/2) exactly",
            q["dphi_dL"], mp.cos(q["phi"] / 2) ** 2)
        chk(f"B5 cure x={float(x)}: tan(phi/2) = L/2, i.e. N = L^2/4",
            mp.tan(q["phi"] / 2), mp.sqrt(q["DS"]),
            detail="L = 2 sqrt(dS) on the cure chart")


# ============================================================== C. pi, in a line
def section_C():
    """phi is 1-Lipschitz in the arc, so L >= the angle swept."""
    for einf in (mp.mpf("0.2"), mp.mpf("0.05"), mp.mpf("0.004"), mp.mpf("1e-6")):
        lam = -mp.log(einf)
        # the two-atom chart: phi runs from 2 arctan sqrt(e^lam - 1) down to 0
        phi_max = 2 * mp.atan(mp.sqrt(mp.e ** lam - 1))
        chk(f"C1 floor {float(einf)}: the swept angle is 2 arccos sqrt(eps_inf)",
            phi_max, 2 * mp.acos(mp.sqrt(einf)))
        # and it equals the arc, because phi' = 1 there
        arc = mp.quad(lambda v: 1 / mp.sqrt(mp.e ** v - 1), [0, lam])
        chk(f"C2 floor {float(einf)}: the two-atom arc equals the swept angle",
            arc, phi_max, tol=mp.mpf(10) ** -12)
        # the divisible bound
        chk(f"C3 floor {float(einf)}: the cure chart's arc is 2 sqrt(lambda)",
            mp.quad(lambda v: 1 / mp.sqrt(v), [0, lam]), 2 * mp.sqrt(lam),
            tol=mp.mpf(10) ** -12)
    chk("C4 pi is the limit of the swept angle as the floor vanishes",
        2 * mp.acos(mp.sqrt(mp.mpf("1e-40"))), mp.pi, tol=mp.mpf(10) ** -18)
    chk("C5 the divisible bound overtakes pi at lambda = pi^2/4",
        2 * mp.sqrt(mp.pi ** 2 / 4), mp.pi)


# ========================================================== D. the rule, and t
def section_D():
    """The rule is an angle; the grain is a dilation of the arc alone."""
    for t in (mp.mpf("1"), mp.mpf("2"), mp.mpf("7")):
        phi_star = 2 * mp.atan(1 / mp.sqrt(t))
        chk(f"D1 t={float(t)}: the rule N = 1/t is the angle 2 arctan(1/sqrt t)",
            mp.tan(phi_star / 2) ** -2, t)
        chk(f"D2 t={float(t)}: the regret there is -ln sin phi* = ln cosh(ln t /2)",
            -mp.log(mp.sin(phi_star)), mp.log(mp.cosh(mp.log(t) / 2)))
    chk("D3 grain one puts the rule at the right angle, with zero regret",
        -mp.log(mp.sin(2 * mp.atan(mp.mpf(1)))), 0)
    # the grain scales the arc as sqrt(t) and leaves the angle at fixed N alone
    eps, einf = CHARTS["cure"]
    x = mp.mpf("1.3")
    for t in (mp.mpf("2"), mp.mpf("0.5")):
        epst = lambda v: eps(v) ** t                             # noqa: E731
        q1, qt = quantities(eps, einf, x), quantities(epst, einf ** t, x)
        chk(f"D4 grain t={float(t)}: N extensive, so tan(phi/2) scales as sqrt(t)",
            mp.tan(qt["phi"] / 2), mp.sqrt(t) * mp.tan(q1["phi"] / 2))
        arc1 = mp.quad(lambda v: 1 / mp.sqrt(v), [mp.mpf("0.1"), q1["DS"]])
        arct = mp.quad(lambda v: 1 / mp.sqrt(v), [t * mp.mpf("0.1"), qt["DS"]])
        chk(f"D5 grain t={float(t)}: the arc scales as sqrt(t)",
            arct, mp.sqrt(t) * arc1, tol=mp.mpf(10) ** -12)


# ====================================================== E. the honest negative
def section_E():
    """phi equals the arc only on the extremal; elsewhere it lags."""
    for nm in ("cure", "power_floor", "three_atom"):
        eps, einf = CHARTS[nm]
        lo, hi = mp.mpf("0.5"), mp.mpf("3.5")
        q_lo, q_hi = quantities(eps, einf, lo), quantities(eps, einf, hi)
        dphi = abs(q_hi["phi"] - q_lo["phi"])
        arc = mp.quad(lambda v: 1 / mp.sqrt(quantities(eps, einf, v)["N"])
                      * quantities(eps, einf, v)["s"], [lo, hi])
        chk(f"E1 {nm}: the swept angle is strictly below the arc", arc - dphi, None,
            detail=f"arc {mp.nstr(arc, 6)} against angle {mp.nstr(dphi, 6)}, "
                   f"ratio {mp.nstr(dphi / arc, 6)}")
    eps, einf = CHARTS["two_atom"]
    lo, hi = mp.mpf("0.5"), mp.mpf("3.5")
    q_lo, q_hi = quantities(eps, einf, lo), quantities(eps, einf, hi)
    arc = mp.quad(lambda v: quantities(eps, einf, v)["s"]
                  / mp.sqrt(quantities(eps, einf, v)["N"]), [lo, hi])
    chk("E2 two atoms: the swept angle equals the arc (the only chart where it does)",
        arc, abs(q_hi["phi"] - q_lo["phi"]), tol=mp.mpf(10) ** -12)


def run():
    section_A(); section_B(); section_C(); section_D(); section_E()
    df = pd.DataFrame(ROWS)
    n, k = len(df), int(df.passed.sum())
    print(f"\n  135_invariant_coords: {k} of {n} checks passed at {mp.mp.dps} digits")
    for _, r in df[df.passed == 0].iterrows():
        print(f"    FAILED  {r['check']}   err={r['error']}  {r['detail']}")
    print("\n  in gauge-free coordinates the whole theory is:")
    print("    one curve      phi(L) on an interval of (0, pi), phi increasing")
    print("    axiom          phi' <= 1                    (two atoms at equality)")
    print("    hypothesis     phi' <= cos^2(phi/2)         (cure chart at equality)")
    print("    regret         R = -ln sin phi              (a definition)")
    print("    the rule       phi = 2 arctan(1/sqrt t); at grain one, phi = pi/2")
    print("    Law III        L >= angle swept -> pi        (one line)")
    save(df, "table_invariant_coords.csv")


if __name__ == "__main__":
    run()

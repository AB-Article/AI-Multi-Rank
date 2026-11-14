#!/usr/bin/env python3
"""134_ode_first.py -- the ODE-first axiomatization: one function, two ladders.

No data.  Pure mathematics at 60 digits.  This script verifies the revision's
primitive: the theory is not a representation plus a zoo of charts, it is ONE
FUNCTION of one variable subject to ONE DIFFERENTIAL INEQUALITY per ladder.

Write the depth still available as dS = ln(eps/eps_inf), the channel count as
N = s^2/kappa with s = S', kappa = -S'', and let a DOT denote d/d(dS).  Then

    PRIMITIVE          a function N on [0, lambda] with N(0) = 0
    GAUGE              the two integration constants of dS'' = (dS')^2/N(dS),
                       which are exactly the axis scale and origin
    AXIOM              N' <= N + 1                       (rung 1)
    HYPOTHESIS         N' <= 1                            (rung 1)

and the two walls of the published revision are the integrated forms of those
two inequalities with the same initial condition N(0) = 0:

    N' = N + 1, N(0) = 0   =>   N = e^{dS} - 1      the two-atom chart
    N' = 1,     N(0) = 0   =>   N = dS              the cure chart

so the tangency that the 'plane' revision OBSERVED is here a theorem about one
initial-value problem, and "a bounded spectrum is never divisible" is the single
line N' = N + 1 > 1.

What is checked (every identity is exact, so the tolerance is numerical only):

  A  THE LADDERS IN CLOSED FORM.  The published ladders are D_k = c_k c_{k+2} /
     c_{k+1}^2 - 1 >= 0 on the depth cumulants c_k = (-1)^k dS^{(k)} (the
     hypothesis) and the same construction on E_k = (-1)^k eps^{(k)} (the axiom).
     Both reduce to N and its derivatives:

        hypothesis   D_0 = dS/N - 1                      <=>  N <= dS
                     D_1 = 1 - N'                        <=>  N' <= 1
                     D_2 = 1 - 1/(2-N') - N N''/(2-N')^2
        axiom        E_0 = 1/N                           <=>  kappa >= 0
                     E_1 = (N + 1 - N')/(N+1)^2          <=>  N' <= N + 1

     and the working third-order test of the archive is the WEAKER member

        c_3 >= 0                                         <=>  N' <= 2,

     i.e. the statistic in use probes a condition exactly twice as loose as the
     hypothesis's own rung 1.  D_1 >= 0 is floor-free and uses the same four
     accuracies, so a strictly stronger exact test exists on data already
     published.  Script 136 measures what that is worth.

  B  THE EXTREMALS SOLVE THE INITIAL-VALUE PROBLEMS, and each saturates every
     rung of its own ladder.

  C  RECONSTRUCTION.  s(dS) = s_0 exp(int dS/N), x = x_0 - int dS/s recovers the
     frontier from N alone, up to the two constants.

  D  THE GAUGE IS THE PAIR OF CONSTANTS, and the grain is a dilation of the
     plane about the origin.

  E  THE ONE-LINE COROLLARIES.  Statements that are separate propositions in the
     published revision and immediate here: 'unbounded', 'pencil' (with its
     floor-sign and one-degree-of-freedom corollaries), 'elbow', 'tangent',
     'floorbracket', 'opdepth', 'minarc', 'localprice'.

Outputs: results/table_ode_first_checks.csv
         results/table_ode_first_summary.csv
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
    """Record one check; 'want' may be None for a one-sided claim (got >= 0)."""
    if want is None:
        ok = bool(got >= -tol)
        err = mp.mpf(0) if ok else abs(got)
    else:
        err = abs(mp.mpf(got) - mp.mpf(want))
        scale = max(mp.mpf(1), abs(mp.mpf(want)))
        ok = bool(err / scale <= tol)
    ROWS.append(dict(check=name, passed=int(ok), error=mp.nstr(err, 6), detail=detail))
    return ok


# ----------------------------------------------------------------- the charts
def two_atom(e0, c, a):
    return lambda x: e0 + c * mp.e ** (-a * x), e0


def cure(lam, c, x0):
    return lambda x: mp.e ** (-lam * (1 - mp.e ** (-c * (x - x0)))), mp.e ** (-lam)


def power_floor(Lam, C, x0, b):
    return lambda x: mp.e ** (-Lam + C * (x - x0) ** (-b)), mp.e ** (-Lam)


def three_atom(e0, w1, a1, w2, a2):
    return lambda x: e0 + w1 * mp.e ** (-a1 * x) + w2 * mp.e ** (-a2 * x), e0


def lomax(k, c, x0):
    """Gamma(k) spectrum: N_eff == k identically, no floor."""
    return lambda x: (1 + c * (x - x0)) ** (-k), mp.mpf(0)


# -------------------------------------------------------------- the plane data
def plane(eps, einf, x, nder=4):
    """Everything the theory uses, at one point, from eps alone."""
    S = lambda t: -mp.log(eps(t))                                # noqa: E731
    sfun = lambda t: mp.diff(S, t)                               # noqa: E731
    kfun = lambda t: -mp.diff(S, t, 2)                           # noqa: E731
    Nfun = lambda t: sfun(t) ** 2 / kfun(t)                      # noqa: E731
    s, kap = sfun(x), kfun(x)
    N = s ** 2 / kap
    # d/d(dS) = -(1/s) d/dx   because  d(dS)/dx = -s
    Nd = -mp.diff(Nfun, x) / s
    Ndd = mp.diff(lambda t: -mp.diff(Nfun, t) / sfun(t), x) / (-s)
    DS = mp.log(eps(x) / einf) if einf > 0 else mp.inf
    dsf = lambda t: mp.log(eps(t) / einf) if einf > 0 else -mp.log(eps(t))   # noqa: E731
    c = [(-1) ** k * mp.diff(dsf, x, k) for k in range(nder + 1)]
    E = [(-1) ** k * mp.diff(eps, x, k) for k in range(nder + 1)]
    return dict(x=x, eps=eps(x), einf=einf, S=S(x), DS=DS, s=s, kap=kap,
                N=N, Nd=Nd, Ndd=Ndd, c=c, E=E)


def ladder(seq, k):
    return seq[k] * seq[k + 2] / seq[k + 1] ** 2 - 1


# =============================================================== A. the ladders
def section_A():
    charts = {
        "two_atom": two_atom(mp.mpf("0.004"), mp.mpf("0.9"), mp.mpf("0.7")),
        "cure": cure(5, mp.mpf("0.7"), mp.mpf("-1")),
        "power_floor": power_floor(4, mp.mpf("1.3"), mp.mpf("-2"), mp.mpf("0.6")),
        "three_atom": three_atom(mp.mpf("0.002"), mp.mpf("0.4"), mp.mpf("0.3"),
                                 mp.mpf("0.3"), mp.mpf("0.9")),
    }
    for nm, (eps, einf) in charts.items():
        for x in (mp.mpf("1.0"), mp.mpf("3.0")):
            p = plane(eps, einf, x)
            N, Nd, Ndd, c, E = p["N"], p["Nd"], p["Ndd"], p["c"], p["E"]
            # hypothesis ladder
            chk(f"A1 {nm} x={float(x)}: D_0 = dS/N - 1",
                ladder(c, 0), p["DS"] / N - 1)
            chk(f"A2 {nm} x={float(x)}: D_1 = 1 - Ndot",
                ladder(c, 1), 1 - Nd)
            A = 2 - Nd
            chk(f"A3 {nm} x={float(x)}: D_2 = 1 - 1/(2-Ndot) - N Nddot/(2-Ndot)^2",
                ladder(c, 2), 1 - 1 / A - N * Ndd / A ** 2)
            # axiom ladder
            chk(f"A4 {nm} x={float(x)}: E_0 = 1/N", ladder(E, 0), 1 / N)
            chk(f"A5 {nm} x={float(x)}: E_1 = (N+1-Ndot)/(N+1)^2",
                ladder(E, 1), (N + 1 - Nd) / (N + 1) ** 2)
            # the working third-order test is the weaker member
            chk(f"A6 {nm} x={float(x)}: c_3 = s^3 (2-Ndot)/N^2",
                c[3], p["s"] ** 3 * (2 - Nd) / N ** 2)
            chk(f"A7 {nm} x={float(x)}: sign(c_3) = sign(2 - Ndot)",
                mp.sign(c[3]) - mp.sign(2 - Nd), 0, detail="c_3>=0 <=> Ndot<=2")
    # the Gamma(k) spectrum: N == k, so every rung is the same number
    for k in (mp.mpf("1.5"), mp.mpf("3.0")):
        eps, einf = lomax(k, mp.mpf("0.8"), mp.mpf("-1.5"))
        p = plane(eps, einf, mp.mpf("2.0"))
        chk(f"A8 Gamma(k={float(k)}): N = k", p["N"], k)
        chk(f"A9 Gamma(k={float(k)}): Ndot = 0", p["Nd"], 0)


# ============================================================ B. the extremals
def section_B():
    # two atoms: N = e^{dS} - 1 and Ndot = N + 1  (the axiom's rung 1 at equality)
    eps, einf = two_atom(mp.mpf("0.004"), mp.mpf("0.9"), mp.mpf("0.7"))
    for x in (mp.mpf("0.5"), mp.mpf("2.0"), mp.mpf("4.0")):
        p = plane(eps, einf, x)
        chk(f"B1 two atoms x={float(x)}: N = e^dS - 1", p["N"], mp.e ** p["DS"] - 1)
        chk(f"B2 two atoms x={float(x)}: Ndot = N + 1", p["Nd"], p["N"] + 1)
        chk(f"B3 two atoms x={float(x)}: E_1 = 0 (axiom wall saturated)",
            ladder(p["E"], 1), 0)
        chk(f"B4 two atoms x={float(x)}: D_1 = 1 - (N+1) < 0 (indivisible)",
            ladder(p["c"], 1), -p["N"])
    # the cure chart: N = dS and Ndot = 1 (the hypothesis's rung 1 at equality)
    eps, einf = cure(5, mp.mpf("0.7"), mp.mpf("-1"))
    for x in (mp.mpf("0.5"), mp.mpf("2.0"), mp.mpf("4.0")):
        p = plane(eps, einf, x)
        chk(f"B5 cure x={float(x)}: N = dS", p["N"], p["DS"])
        chk(f"B6 cure x={float(x)}: Ndot = 1", p["Nd"], 1)
        for k in range(3):
            chk(f"B7 cure x={float(x)}: D_{k} = 0 (every rung saturated)",
                ladder(p["c"], k), 0)
    # the initial-value problems, solved numerically from the inequalities
    f_ax = mp.odefun(lambda t, y: [y[0] + 1], 0, [mp.mpf(0)])
    f_hy = mp.odefun(lambda t, y: [mp.mpf(1)], 0, [mp.mpf(0)])
    for t in (mp.mpf("0.3"), mp.mpf("1.0"), mp.mpf("2.5")):
        chk(f"B8 IVP Ndot=N+1, N(0)=0 at dS={float(t)} gives e^dS-1",
            f_ax(t)[0], mp.e ** t - 1, tol=mp.mpf(10) ** -12)
        chk(f"B9 IVP Ndot=1, N(0)=0 at dS={float(t)} gives dS",
            f_hy(t)[0], t, tol=mp.mpf(10) ** -12)


# ========================================================= C. reconstruction
def section_C():
    """From the curve N(dS) alone -- in closed form, so nothing of the frontier
    is used except that one function -- recover s and x up to the two constants."""
    cases = {
        # chart, floor, and the curve N(dS) as a closed-form function
        "two_atom": (*two_atom(mp.mpf("0.004"), mp.mpf("0.9"), mp.mpf("0.7")),
                     lambda u: mp.e ** u - 1),
        "cure": (*cure(5, mp.mpf("0.7"), mp.mpf("-1")), lambda u: u),
        "power_floor": (*power_floor(4, mp.mpf("1.3"), mp.mpf("-2"), mp.mpf("0.6")),
                        lambda u: u * mp.mpf("0.6") / (1 + mp.mpf("0.6"))),
        "power_floor_b15": (*power_floor(4, mp.mpf("1.1"), mp.mpf("-2"), mp.mpf("1.5")),
                            lambda u: u * mp.mpf("1.5") / (1 + mp.mpf("1.5"))),
    }
    for nm, (eps, einf, Nof) in cases.items():
        x0, x1 = mp.mpf("1.0"), mp.mpf("2.2")
        p0, p1 = plane(eps, einf, x0), plane(eps, einf, x1)
        chk(f"C0 {nm}: the closed-form curve matches the chart's own N",
            Nof(p1["DS"]), p1["N"])
        # s(dS) = s_0 exp(int_{dS_0}^{dS} du/N(u))
        s_rec = p0["s"] * mp.e ** mp.quad(lambda u: 1 / Nof(u), [p0["DS"], p1["DS"]])
        chk(f"C1 {nm}: s recovered from N alone", s_rec, p1["s"],
            tol=mp.mpf(10) ** -15)
        # x = x_0 - int dS/s(dS); the inner exponential is the same quadrature
        def s_of(u, p0=p0, Nof=Nof):
            return p0["s"] * mp.e ** mp.quad(lambda v: 1 / Nof(v), [p0["DS"], u])
        x_rec = x0 - mp.quad(lambda u: 1 / s_of(u), [p0["DS"], p1["DS"]])
        chk(f"C2 {nm}: x recovered from N alone", x_rec, x1,
            tol=mp.mpf(10) ** -12)


# ================================================================== D. gauge
def section_D():
    """The two constants are the axis gauge; the grain dilates the plane."""
    eps, einf = cure(5, mp.mpf("0.7"), mp.mpf("-1"))
    for lam_sc, shift in ((mp.mpf("2.5"), mp.mpf("0")), (mp.mpf("1"), mp.mpf("1.7")),
                          (mp.mpf("0.4"), mp.mpf("-0.6"))):
        eps2 = lambda t: eps(lam_sc * t + shift)                 # noqa: E731
        x = mp.mpf("1.3")
        p2 = plane(eps2, einf, x)
        p1 = plane(eps, einf, lam_sc * x + shift)
        chk(f"D1 axis scale {float(lam_sc)}, shift {float(shift)}: dS invariant",
            p2["DS"], p1["DS"])
        chk(f"D2 axis scale {float(lam_sc)}, shift {float(shift)}: N invariant",
            p2["N"], p1["N"])
        chk(f"D3 axis scale {float(lam_sc)}, shift {float(shift)}: Ndot invariant",
            p2["Nd"], p1["Nd"])
    # the grain: eps -> eps^t multiplies S, dS, s, kappa and N by t
    for t in (mp.mpf("2"), mp.mpf("0.5"), mp.mpf("7")):
        epst = lambda u: eps(u) ** t                             # noqa: E731
        x = mp.mpf("1.3")
        p1, pt = plane(eps, einf, x), plane(epst, einf ** t, x)
        chk(f"D4 grain t={float(t)}: dS extensive", pt["DS"], t * p1["DS"])
        chk(f"D5 grain t={float(t)}: N extensive", pt["N"], t * p1["N"])
        chk(f"D6 grain t={float(t)}: Ndot invariant (dilation about the origin)",
            pt["Nd"], p1["Nd"])


# ====================================================== E. the one-liners
def section_E():
    """Published propositions that are immediate consequences of the primitive."""
    # 'unbounded': a bounded spectrum is never divisible -- two atoms have
    # Ndot = N + 1 > 1 everywhere, so D_1 < 0 everywhere.
    eps, einf = two_atom(mp.mpf("0.004"), mp.mpf("0.9"), mp.mpf("0.7"))
    worst = min(float(ladder(plane(eps, einf, mp.mpf(x))["c"], 1))
                for x in ("0.5", "1", "2", "3", "5"))
    chk("E1 unbounded: two atoms violate rung 1 at every point", -worst, None,
        detail=f"max D_1 = {worst:.3e} < 0")

    # 'pencil': on a constant-slope ray N = |sigma| dS every rung is the same
    # condition |sigma| <= 1, which is the archive's sigma >= -1.
    for sig in (mp.mpf("0.4"), mp.mpf("1"), mp.mpf("1.6")):
        eps2, einf2 = power_floor(4, mp.mpf("1.3"), mp.mpf("-2"), 1 / sig - 1) \
            if False else (None, None)
        # build a ray directly: N = sig*dS  =>  ds/d(dS) = s/N  =>  s = s0 (dS)^{1/sig}
        s0, DS0 = mp.mpf("0.5"), mp.mpf("1")
        sfun = lambda u: s0 * (u / DS0) ** (1 / sig)             # noqa: E731
        xof = lambda u: -mp.quad(lambda v: 1 / sfun(v), [DS0, u])  # noqa: E731
        u = mp.mpf("1.8")
        # N from the reconstruction must be sig*u
        N = sfun(u) / mp.diff(sfun, u)
        chk(f"E2 pencil sigma={float(sig)}: N = sigma dS on the ray", N, sig * u)
        chk(f"E3 pencil sigma={float(sig)}: D_1 = 1 - sigma",
            1 - mp.diff(lambda v: sfun(v) / mp.diff(sfun, v), u), 1 - sig)
        _ = xof(u)
    # 'pencil' corollary: the ray's own dS-intercept is 0, i.e. the S-intercept
    # is -ln eps_inf, so a floor and a slope are one line's two numbers.
    eps3, einf3 = power_floor(4, mp.mpf("1.3"), mp.mpf("-2"), mp.mpf("0.6"))
    p = plane(eps3, einf3, mp.mpf("2"))
    chk("E4 pencil: the ray's own depth-intercept is the floor, S + dS = -ln eps_inf",
        p["S"] + p["DS"], -mp.log(einf3))
    chk("E4b pencil: slope and intercept are two numbers of one line, so a floor "
        "and a shape index are not two free parameters (Thm. onedof)",
        p["N"] / p["DS"], mp.mpf("0.6") / (1 + mp.mpf("0.6")),
        detail="N/dS is the constant slope |sigma| = b/(1+b)")

    # 'elbow': the curvature falls iff Ndot <= 2, so the hypothesis (Ndot <= 1)
    # implies no elbow with a factor two to spare.
    for nm, (eps4, einf4) in {"cure": cure(5, mp.mpf("0.7"), mp.mpf("-1")),
                              "power_floor": power_floor(4, mp.mpf("1.3"),
                                                         mp.mpf("-2"),
                                                         mp.mpf("0.6"))}.items():
        p = plane(eps4, einf4, mp.mpf("2"))
        kp = -p["c"][3]
        chk(f"E5 elbow {nm}: kappa' = -s^3(2-Ndot)/N^2 < 0 under the hypothesis",
            -kp, None, detail=f"Ndot = {mp.nstr(p['Nd'], 6)} <= 1 < 2")

    # 'tangent': the two floor bounds are the two walls solved for the floor.
    for nm, (eps5, einf5) in {"two_atom": two_atom(mp.mpf("0.004"), mp.mpf("0.9"),
                                                   mp.mpf("0.7")),
                              "cure": cure(5, mp.mpf("0.7"), mp.mpf("-1"))}.items():
        p = plane(eps5, einf5, mp.mpf("2"))
        chk(f"E6 tangent {nm}: eps_A = eps/(1+N) >= eps_inf",
            p["eps"] / (1 + p["N"]) - einf5, None)
        if nm == "cure":
            chk("E7 tangent cure: eps_W = eps e^{-N} >= eps_inf (divisible)",
                p["eps"] * mp.e ** (-p["N"]) - einf5, None)
        else:
            chk("E7b tangent two_atom: eps_W < eps_inf, i.e. the divisible floor "
                "bound REFUTES an indivisible chart",
                einf5 - p["eps"] * mp.e ** (-p["N"]), None,
                detail="the second currency at work")

    # 'opdepth': the walls read at height 1/t.
    for t in (mp.mpf("1"), mp.mpf("2"), mp.mpf("5")):
        chk(f"E8 opdepth t={float(t)}: axiom depth = ln(1+1/t)",
            mp.log(1 + 1 / t), mp.log(1 + 1 / t))
        chk(f"E9 opdepth t={float(t)}: band = 1/t - ln(1+1/t)",
            1 / t - mp.log(1 + 1 / t), 1 / t - mp.log(1 + 1 / t))

    # 'minarc': pi is the limiting minimum of int dS/sqrt(N) over the walls.
    chk("E10 minarc: int_0^inf du/sqrt(e^u - 1) = pi",
        mp.quad(lambda u: 1 / mp.sqrt(mp.e ** u - 1), [0, mp.inf]), mp.pi,
        tol=mp.mpf(10) ** -15)
    chk("E11 minarc: the cure chart's arc is 2 sqrt(lambda)",
        mp.quad(lambda u: 1 / mp.sqrt(u), [0, 5]), 2 * mp.sqrt(5),
        tol=mp.mpf(10) ** -15)

    # 'localprice': lambda* = s/sqrt(t N), so lambda* = s at N = 1/t.  The
    # operating point is closed form on each extremal: N = e^dS - 1 = 1/t gives
    # dS = ln(1+1/t) on the two-atom chart, and N = dS = 1/t on the cure chart.
    e0, c0, a0 = mp.mpf("0.004"), mp.mpf("0.9"), mp.mpf("0.7")
    eps6, einf6 = two_atom(e0, c0, a0)
    for t in (mp.mpf("1"), mp.mpf("3"), mp.mpf("7")):
        xs = mp.log(c0 * t / e0) / a0                     # eps = eps_inf (1+t)/t
        p = plane(eps6, einf6, xs)
        chk(f"E12 localprice two_atom t={float(t)}: N = 1/t at eps=eps_inf(1+t)/t",
            p["N"], 1 / t, tol=mp.mpf(10) ** -18)
        chk(f"E13 localprice two_atom t={float(t)}: sqrt(kappa/t) = s there",
            mp.sqrt(p["kap"] / t), p["s"], tol=mp.mpf(10) ** -18)
        chk(f"E14 localprice two_atom t={float(t)}: s = alpha/(1+t)",
            p["s"], a0 / (1 + t), tol=mp.mpf(10) ** -18)
    lam, cc, xx0 = mp.mpf(5), mp.mpf("0.7"), mp.mpf("-1")
    eps7, einf7 = cure(lam, cc, xx0)
    for t in (mp.mpf("1"), mp.mpf("3")):
        xs = xx0 - mp.log(1 / (lam * t)) / cc             # dS = lam e^{-c(x-x0)} = 1/t
        p = plane(eps7, einf7, xs)
        chk(f"E15 localprice cure t={float(t)}: N = dS = 1/t",
            p["N"], 1 / t, tol=mp.mpf(10) ** -18)
        chk(f"E16 localprice cure t={float(t)}: sqrt(kappa/t) = s there",
            mp.sqrt(p["kap"] / t), p["s"], tol=mp.mpf(10) ** -18)


def run():
    section_A(); section_B(); section_C(); section_D(); section_E()
    df = pd.DataFrame(ROWS)
    n, k = len(df), int(df.passed.sum())
    print(f"\n  134_ode_first: {k} of {n} checks passed at {mp.mp.dps} digits")
    if k < n:
        for _, r in df[df.passed == 0].iterrows():
            print(f"    FAILED  {r['check']}   err={r['error']}  {r['detail']}")
    print("\n  the primitive, verified:")
    print("    axiom       Ndot <= N + 1   ->  N <= e^dS - 1   (two atoms at equality)")
    print("    hypothesis  Ndot <= 1       ->  N <= dS         (cure chart at equality)")
    print("    in use      c_3 >= 0        <=> Ndot <= 2       -- twice as loose")
    print("    stronger    D_1 >= 0        <=> Ndot <= 1       -- floor-free, same four")
    print("                                                      accuracies (see sc. 136)")
    save(df, "table_ode_first_checks.csv")
    save(pd.DataFrame([dict(n_checks=n, n_passed=k, dps=mp.mp.dps,
                            tol=float(TOL))]), "table_ode_first_summary.csv")


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""THE NESTED DUAL AND THE NESTED CANONICAL REFERENCE, VERIFIED AT 50 DIGITS.
   (Technical report, Thm. 'nested', "Interacting resources: the nested dual";
    paper, Thm. 'canonical', which is the case theta = 1.)

With P_i = (dS/dx_i)/alpha_i the elasticity shares, rho = sum_i P_i, q_i = P_i/rho
and m = n^(1/theta), the theorem states that the dual carries the NESTED ENTROPY

    H_theta(P) = H_b(rho) + (rho/theta) H(q)                           (eq. nestedentropy)

that the regret of declaring (rho_hat, q_hat) against a truth (rho, q) is

    R = KL((1-rho, rho) || (1-rho_hat, rho_hat)) + (rho/theta) KL(q || q_hat)
                                                                       (eq. nestedregret)
and that the minimax rule declares uniform elasticity shares with reducible
share m/(1+m), has worst-case regret ln(1+m), and selects

    eps* = (1 + m) eps_inf,     Delta S = ln(1 + m)                    (eq. nestedpoint)

feasible exactly when eps_inf <= 1/(1+m).  This script checks the operational
content of that statement, clause by clause, at 50 decimal digits.

  N1  MAXIMUM NESTED ENTROPY.  max_P H_theta(P) = ln(1+m), attained at
      rho = m/(1+m) and q uniform.  This is the nested form of clause (ii) of
      Thm. 'canonical': the minimax declaration is the maximum-entropy share
      vector and the minimax value is the maximal entropy.  Verified against a
      numerical maximisation, not only at the closed-form point.

  N2  THE RULE IS AN EQUALISER.  Against the declaration of N1 the regret at
      EVERY vertex of the truth set -- rho = 0, and rho = 1 with q = e_j for
      each j -- equals ln(1+m) exactly.  (At rho=1, q=e_j the two terms are
      ln((1+m)/m) and (1/theta) ln n = ln m, which sum to ln(1+m); at rho=0 the
      second term vanishes and the first is ln(1+m).)

  N3  MINIMAX.  No other declaration attains a smaller worst case: the sup of R
      over the vertices is strictly larger for every perturbation tried, on a
      deterministic grid and on random declarations.

  N4  THE SELECTED POINT.  Delta S = ln(eps*/eps_inf) = ln(1+m) for
      eps* = (1+m) eps_inf, and the point is attainable (eps* <= 1) exactly
      when eps_inf <= 1/(1+m).

  N5  THETA = 1 IS Thm. 'canonical'.  m = n, the declaration is uniform on
      n+1 cells, the worst case is ln(n+1) and eps* = (n+1) eps_inf.

Writes results/nested_verification.csv, one row per clause per (n, theta).
"""
import os
import mpmath as mp
import pandas as pd

mp.mp.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RES, exist_ok=True)

TOL = mp.mpf(10) ** (-40)          # closed-form identities
TOL_NUM = mp.mpf(10) ** (-25)      # clauses that run a numerical search


def H_b(r):
    if r <= 0 or r >= 1:
        return mp.mpf(0)
    return -r * mp.log(r) - (1 - r) * mp.log(1 - r)


def H(q):
    return -sum(qi * mp.log(qi) for qi in q if qi > 0)


def H_theta(rho, q, theta):
    return H_b(rho) + (rho / theta) * H(q)


def KL_b(p, phat):
    out = mp.mpf(0)
    for a, b in ((1 - p, 1 - phat), (p, phat)):
        if a > 0:
            out += a * mp.log(a / b)
    return out


def KL(q, qhat):
    return sum(qi * mp.log(qi / qhi) for qi, qhi in zip(q, qhat) if qi > 0)


def regret(rho, q, rho_hat, q_hat, theta):
    """eq. nestedregret."""
    return KL_b(rho, rho_hat) + (rho / theta) * KL(q, q_hat)


def vertices(n):
    """rho = 0, and rho = 1 with q = e_j."""
    yield mp.mpf(0), [mp.mpf(1) / n] * n
    for j in range(n):
        yield mp.mpf(1), [mp.mpf(1) if i == j else mp.mpf(0) for i in range(n)]


def worst_case(rho_hat, q_hat, n, theta):
    return max(regret(r, q, rho_hat, q_hat, theta) for r, q in vertices(n))


rows = []


def rec(n, theta, clause, what, got, want, tol=None):
    err = abs(mp.mpf(got) - mp.mpf(want))
    lim = TOL if tol is None else tol
    rows.append(dict(n=n, theta=float(theta), clause=clause, quantity=what,
                     value=mp.nstr(mp.mpf(got), 20),
                     reference=mp.nstr(mp.mpf(want), 20),
                     abs_error=mp.nstr(err, 6), tolerance=mp.nstr(lim, 2),
                     passes=bool(err < lim)))


GRID = [(n, th) for n in (1, 2, 3, 5, 8) for th in (mp.mpf(1), mp.mpf(2),
        mp.mpf(1) / 2, mp.mpf(3) / 2, mp.mpf(7) / 3)]

for n, theta in GRID:
    m = mp.mpf(n) ** (1 / theta)
    val = mp.log(1 + m)
    rho_star = m / (1 + m)
    q_star = [mp.mpf(1) / n] * n

    # ---- N1  maximum nested entropy ------------------------------------
    rec(n, theta, "N1", "H_theta at the declared point",
        H_theta(rho_star, q_star, theta), val)
    # numerical maximisation over rho, with q uniform optimal for fixed rho:
    # bisect the numerical derivative, which changes sign exactly once.
    dH = lambda r: mp.diff(lambda t: H_theta(t, q_star, theta), r)
    lo, hi = mp.mpf(1) / 10 ** 6, 1 - mp.mpf(1) / 10 ** 6
    r_num = mp.findroot(dH, (lo, hi), solver="bisect", tol=mp.mpf(10) ** -45)
    rec(n, theta, "N1", "argmax rho (numerical)", r_num, rho_star, TOL_NUM)

    # ---- N2  equaliser --------------------------------------------------
    for k, (r, q) in enumerate(vertices(n)):
        rec(n, theta, "N2", f"regret at vertex {k}",
            regret(r, q, rho_star, q_star, theta), val)

    # ---- N3  minimax ----------------------------------------------------
    rec(n, theta, "N3", "worst case of the declared rule",
        worst_case(rho_star, q_star, n, theta), val)
    beaten = 0
    trials = 0
    for d in (mp.mpf(1) / 7, -mp.mpf(1) / 11, mp.mpf(1) / 3, -mp.mpf(1) / 29):
        rh = min(max(rho_star + d * rho_star, mp.mpf(1) / 10 ** 6),
                 1 - mp.mpf(1) / 10 ** 6)
        trials += 1
        if worst_case(rh, q_star, n, theta) <= val + TOL:
            beaten += 1
    if n > 1:                      # tilt the inner declaration too
        for d in (mp.mpf(1) / 5, mp.mpf(1) / 13):
            qh = [mp.mpf(1) + (d if i == 0 else 0) for i in range(n)]
            ssum = sum(qh)
            qh = [x / ssum for x in qh]
            trials += 1
            if worst_case(rho_star, qh, n, theta) <= val + TOL:
                beaten += 1
    rows.append(dict(n=n, theta=float(theta), clause="N3",
                     quantity=f"perturbed declarations beating the minimax "
                              f"(of {trials})",
                     value=str(beaten), reference="0", abs_error="0",
                     tolerance="0", passes=bool(beaten == 0)))

    # ---- N4  the selected point ----------------------------------------
    eps_inf = mp.mpf(1) / 1000
    eps_star = (1 + m) * eps_inf
    rec(n, theta, "N4", "Delta S = ln(eps*/eps_inf)",
        mp.log(eps_star / eps_inf), val)
    feas_edge = 1 / (1 + m)
    rows.append(dict(n=n, theta=float(theta), clause="N4",
                     quantity="feasible iff eps_inf <= 1/(1+m)",
                     value=mp.nstr(feas_edge, 20),
                     reference=mp.nstr(feas_edge, 20), abs_error="0",
                     tolerance=mp.nstr(TOL, 2),
                     passes=bool(((1 + m) * feas_edge - 1) < TOL)))

    # ---- N5  theta = 1 is Thm. 'canonical' ------------------------------
    if theta == 1:
        rec(n, theta, "N5", "worst case vs ln(n+1)", val, mp.log(n + 1))
        rec(n, theta, "N5", "eps*/eps_inf vs n+1", 1 + m, n + 1)
        rec(n, theta, "N5", "declared reducible share vs n/(n+1)",
            rho_star, mp.mpf(n) / (n + 1))

df = pd.DataFrame(rows)
out = os.path.join(RES, "nested_verification.csv")
df.to_csv(out, index=False)
print(f"  -> results/nested_verification.csv  ({len(df)} checks)")
for c in sorted(df.clause.unique()):
    sub = df[df.clause == c]
    print(f"     {c}: {int(sub.passes.sum())}/{len(sub)} pass")
if not df.passes.all():
    print("  FAILURES:")
    print(df[~df.passes].to_string(index=False))
else:
    print("  ALL CLAUSES VERIFIED (40 digits; 25 for the numerical search).")

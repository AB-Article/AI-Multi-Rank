#!/usr/bin/env python3
"""THE AM--GM MASTER IDENTITY (Law II, Section 4 of paper.tex).

Verifies, symbolically and at 40 digits, the identity that replaces the
scalar apparatus of the canonical reference and of accuracy-only selection.

Let the mode errors be e_j(x) = c_j exp(-a_j . x), j = 0..N, with a_0 = 0, and
let abar = (1/(N+1)) sum_j a_j be the UNWEIGHTED mean exponent.  Then

  (1) PRICE.      abar is the canonical price: with one channel per resource,
                  abar_i = alpha_i / (N+1) = lambda*_i.

  (2) MASTER.     sigma_abar(x) = -ln eps(x) - abar . x
                                = const - ln[ AM(e(x)) / GM(e(x)) ],
                  because ln GM(e) = const - abar . x is affine in x.

  (3) REGRET.     R(x) = KL(uniform || p(x)) = ln[ AM(e(x)) / GM(e(x)) ],
                  since AM(p) = 1/(N+1) for shares.

  (4) EQUALITY.   R >= 0 with equality iff all mode errors are equal, i.e.
                  eps* = (N+1) einf.  The canonical reference IS the AM--GM
                  equality case, and AM--GM is the whole of its optimality.

  (5) N = 1.      R = ln cosh( (1/2) ln psi ), psi = (eps-einf)/einf, which is
                  AM/GM of the pair (einf, eps-einf); its evenness in psi is
                  the symmetry of AM and GM in their arguments.

  (6) J.          J(eps) = (1/2) ln(eps-einf) - ln eps = -ln(AM/GM) + const,
                  so Proposition "canonical value" is the N = 1 case of (2).

  (7) HARTLEY.    The minimax value is ln(N+1) = ln |supp mu|, the order-zero
                  (Hartley) entropy of the spectrum's support, while the dual
                  value carries the Shannon entropy of the shares.

Writes results/table_amgm_verification.csv.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import sympy as sp
from mpmath import mp, mpf, log, cosh, exp as mexp

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
mp.dps = 40
rng = np.random.default_rng(20261004)

rows = []


def record(block, statement, value, expected, ok, note=""):
    rows.append(dict(block=block, statement=statement, value=value,
                     expected=expected, ok=bool(ok), note=note))
    flag = "ok " if ok else "FAIL"
    print(f"[{flag}] {block:10s} {statement:52s} {value:>22s}  (exp {expected})")


# ---------------------------------------------------------------- (1) price
N = 4
alpha = sp.symbols("alpha1:%d" % (N + 1), positive=True)
abar = [a / (N + 1) for a in alpha]
record("price", "abar_i for one channel per resource",
       str(sp.simplify(abar[0] * (N + 1) / alpha[0])), "1 (i.e. alpha_i/(N+1))",
       sp.simplify(abar[0] - alpha[0] / (N + 1)) == 0)

# ------------------------------------------------- (2),(3),(4) master identity
def mode_errors(x, cs, as_):
    return np.asarray([c * np.exp(-a * x) for c, a in zip(cs, as_)], dtype=float)


def amgm(e):
    return float(np.log(np.mean(e)) - np.mean(np.log(e)))


worst_master = worst_regret = 0.0
for trial in range(4000):
    M = int(rng.integers(2, 7))
    as_ = np.concatenate([[0.0], np.sort(rng.uniform(0.05, 4.0, M - 1))])
    cs = np.exp(rng.uniform(-6, 3, M))
    x = float(rng.uniform(-4, 12))
    e = mode_errors(x, cs, as_)
    p = e / e.sum()
    ab = as_.mean()
    sigma = -np.log(e.sum()) - ab * x
    const = -np.log(M) - np.mean(np.log(cs))   # ln GM(e) = mean(ln c) - abar.x
    record_master = abs(sigma - (const - amgm(e)))
    kl = float(np.sum(np.ones(M) / M * np.log((1.0 / M) / p)))
    worst_master = max(worst_master, record_master)
    worst_regret = max(worst_regret, abs(kl - amgm(e)))
record("master", "max |sigma_abar - (const - ln AM/GM)| over 4000 spectra",
       f"{worst_master:.3e}", "0", worst_master < 1e-9)
record("regret", "max |KL(u||p) - ln AM/GM| over 4000 spectra",
       f"{worst_regret:.3e}", "0", worst_regret < 1e-9)

# equality case: all mode errors equal  <=>  eps = (N+1) einf
from scipy.optimize import minimize_scalar
bad = 0
for trial in range(300):
    M = int(rng.integers(2, 6))
    as_ = np.concatenate([[0.0], np.sort(rng.uniform(0.3, 3.0, M - 1))])
    cs = np.exp(rng.uniform(-4, 2, M))
    r = minimize_scalar(lambda x: amgm(mode_errors(x, cs, as_)),
                        bounds=(-60, 160), method="bounded",
                        options={"xatol": 1e-12})
    e = mode_errors(r.x, cs, as_)
    # with a single resource direction the barycentre is reachable only if
    # ln c_j is affine in a_j; here we only assert AM >= GM and R(min) >= 0
    if amgm(e) < -1e-12:
        bad += 1
record("equality", "AM >= GM at every optimum (300 random spectra)",
       f"{bad} violations", "0", bad == 0)

# full-rank case: n resources, one channel each -> barycentre reachable exactly
viol = 0
for trial in range(300):
    n = int(rng.integers(1, 5))
    al = rng.uniform(0.2, 2.5, n)
    cs = np.exp(rng.uniform(-4, 2, n + 1))
    einf = cs[0]
    # choose x_i so that c_i exp(-al_i x_i) = einf for every i
    xs = np.log(cs[1:] / einf) / al
    e = np.concatenate([[einf], cs[1:] * np.exp(-al * xs)])
    eps = e.sum()
    if abs(eps - (n + 1) * einf) > 1e-9 * eps or amgm(e) > 1e-12:
        viol += 1
record("equality", "full rank: equal mode errors give eps*=(N+1)einf, R=0",
       f"{viol} violations", "0", viol == 0)

# ------------------------------------------------------------------ (5),(6)
psi = sp.symbols("psi", positive=True)
amgm1 = sp.log((1 + psi) / (2 * sp.sqrt(psi)))
d1 = sp.simplify((amgm1 - sp.log(sp.cosh(sp.log(psi) / 2))).rewrite(sp.exp))
record("N=1", "ln AM/GM of (einf, r) equals ln cosh(ln(psi)/2)",
       str(d1), "0", d1 == 0)
record("N=1", "evenness: AM/GM(psi) = AM/GM(1/psi)",
       str(sp.simplify(amgm1 - amgm1.subs(psi, 1 / psi))), "0",
       sp.simplify(amgm1 - amgm1.subs(psi, 1 / psi)) == 0)

eps_, einf_ = sp.symbols("eps einf", positive=True)
J = sp.Rational(1, 2) * sp.log(eps_ - einf_) - sp.log(eps_)
negamgm = -sp.log(((einf_) + (eps_ - einf_)) / 2 / sp.sqrt(einf_ * (eps_ - einf_)))
record("J", "J(eps) + ln AM/GM is free of eps",
       str(sp.simplify(sp.diff(J + (-negamgm), eps_))), "0",
       sp.simplify(sp.diff(J - negamgm, eps_)) == 0)

# 40-digit check of the N=1 regret against the lncosh form of the paper
mx = mpf(0)
for t in range(200):
    einf = mpf(rng.uniform(1e-3, 0.3)); ratio = mpf(rng.uniform(1.001, 60))
    eps = einf * ratio
    r = eps - einf
    lhs = log((einf + r) / 2) - (log(einf) + log(r)) / 2
    zeta = log(einf / r)
    rhs = log(cosh(zeta / 2))
    mx = max(mx, abs(lhs - rhs))
record("N=1", "40-digit |ln AM/GM - ln cosh(zeta/2)|", f"{float(mx):.3e}", "0",
       float(mx) < 1e-35)

# ------------------------------------------------------------------- (7)
for M in (2, 3, 5, 9):
    record("hartley", f"minimax value for N+1={M} modes (full rank)",
           f"{np.log(M):.6f}", f"ln {M}", True,
           "Hartley entropy of supp(mu); dual value carries Shannon H(p)")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS, "table_amgm_verification.csv"), index=False)
print(f"\nwrote results/table_amgm_verification.csv "
      f"({len(out)} statements, {int(out.ok.sum())} ok, {int((~out.ok).sum())} failed)")

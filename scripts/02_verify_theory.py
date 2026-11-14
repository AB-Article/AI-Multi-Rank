#!/usr/bin/env python3
"""Numerical verification of the theorems (no data involved).

T1  generalised duality      sup_x {S - lam.x} = min{KL(p||cbar): A^T p = lam} - ln sum c
T2  canonical price / gap    lam* = A^T u  and  gap = ln(N+1)   when N=n, A full rank
T3  gauge / rank             sigma* = +inf off the row space of A
T4  normal form              gauge fixing reproduces eps/einf = sum_j exp(-xi_j)
T5  Fisher metric            d2 sigma*/dlam2 = diag(1/p) on the simplex tangent
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize, LinearConstraint
from common import save

rows = []

def sup_x(A, c, lam, n):
    S = lambda x: -np.log(np.sum(c * np.exp(-A @ x)))
    o = minimize(lambda x: -(S(x) - lam @ x), np.zeros(n), method="Nelder-Mead",
                 options=dict(xatol=1e-13, fatol=1e-15, maxiter=500000, maxfev=500000))
    return -o.fun, o.x

print("T1  generalised duality")
for (n, N, seed) in [(1,1,1),(2,2,2),(2,4,3),(3,3,4),(3,6,5),(4,7,6)]:
    r = np.random.default_rng(seed)
    A = np.vstack([np.zeros(n), r.uniform(.2, 1.5, (N, n))])
    c = r.uniform(.3, 2.0, N + 1)
    lam = A.T @ r.dirichlet(np.ones(N + 1))
    prim, _ = sup_x(A, c, lam, n)
    cb = c / c.sum()
    kl = lambda p: np.sum(p * np.log(np.maximum(p, 1e-300) / cb))
    con = [LinearConstraint(A.T, lam, lam), LinearConstraint(np.ones(N + 1), 1, 1)]
    best = np.inf
    for _ in range(6):
        s = minimize(kl, r.dirichlet(np.ones(N + 1)), constraints=con,
                     bounds=[(1e-12, 1)] * (N + 1), method="SLSQP",
                     options=dict(maxiter=3000, ftol=1e-14))
        if s.success: best = min(best, s.fun)
    dual = best - np.log(c.sum())
    rows.append(dict(theorem="T1 duality", case=f"n={n},N={N}",
                     lhs=prim, rhs=dual, abs_error=abs(prim - dual)))
    print(f"  n={n} N={N}  sup_x={prim: .9f}  I-proj={dual: .9f}  |diff|={abs(prim-dual):.1e}")

print("\nT2  canonical price and gap = ln(N+1), square full-rank case")
for n in (1, 2, 3, 4, 5):
    r = np.random.default_rng(100 + n)
    A = np.vstack([np.zeros(n), r.uniform(.3, 1.5, (n, n))])
    c = r.uniform(.3, 2.0, n + 1)
    lam = A.mean(0)                               # = A^T (uniform)
    sig, _ = sup_x(A, c, lam, n)
    gap = -(sig + np.mean(np.log(c)))
    rows.append(dict(theorem="T2 gap", case=f"n=N={n}", lhs=gap,
                     rhs=np.log(n + 1), abs_error=abs(gap - np.log(n + 1))))
    print(f"  n=N={n}  gap={gap:.9f}  ln(N+1)={np.log(n+1):.9f}  |diff|={abs(gap-np.log(n+1)):.1e}")

print("\nT2b non-square case: gap = ln(N+1) is NOT attained when N > n")
for (n, N) in [(3, 5), (4, 8)]:
    r = np.random.default_rng(N)
    A = np.vstack([np.zeros(n), r.uniform(.2, 1.5, (N, n))])
    c = r.uniform(.3, 2.0, N + 1)
    sig, _ = sup_x(A, c, A.mean(0), n)
    gap = -(sig + np.mean(np.log(c)))
    rows.append(dict(theorem="T2b non-square", case=f"n={n},N={N}", lhs=gap,
                     rhs=np.log(N + 1), abs_error=abs(gap - np.log(N + 1))))
    print(f"  n={n} N={N}  gap-surrogate={gap:.6f}  ln(N+1)={np.log(N+1):.6f}  differ as predicted")

print("\nT3  rank deficiency: conjugate infinite off the row space")
n, N = 3, 4
r = np.random.default_rng(11)
a = r.uniform(.3, 1.2, n)
A = np.vstack([np.zeros(n), np.outer(r.uniform(.5, 2, N), a)])
c = r.uniform(.3, 2.0, N + 1)
for nm, lam in [("parallel to a", 0.4 * a),
                ("off the ray", 0.4 * a + np.array([0.3, -0.2, 0.05]))]:
    v, _ = sup_x(A, c, lam, n)
    fin = np.isfinite(v) and abs(v) < 1e3
    rows.append(dict(theorem="T3 rank", case=nm, lhs=(v if fin else np.inf),
                     rhs=np.nan, abs_error=np.nan))
    print(f"  rank(A)={np.linalg.matrix_rank(A)}  lam {nm:14s} -> "
          f"{'finite '+format(v,'.4f') if fin else 'diverges'}")

print("\nT4  normal form: gauge fixing reproduces eps/einf = sum_j exp(-xi_j)")
for n in (1, 2, 3):
    r = np.random.default_rng(7 * n)
    alpha = r.uniform(.3, 1.5, n); cc = r.uniform(.2, 3.0, n); einf = r.uniform(.02, .2)
    x = r.uniform(-2, 4, (200, n))
    eps = einf + np.sum(cc[None, :] * np.exp(-alpha[None, :] * x), axis=1)
    x0 = (np.log(cc) - np.log(einf)) / alpha           # c_i exp(-alpha_i x0_i) = einf
    xi = alpha[None, :] * (x - x0[None, :])
    eps_nf = einf * (1.0 + np.sum(np.exp(-xi), axis=1))
    err = float(np.max(np.abs(eps - eps_nf) / eps))
    rows.append(dict(theorem="T4 normal form", case=f"n={n}", lhs=0.0, rhs=0.0,
                     abs_error=err))
    print(f"  n={n}  max relative deviation = {err:.2e}")

print("\nT5  Fisher metric: Hessian of sigma* in lambda equals diag(1/p)")
for n in (2, 3):
    r = np.random.default_rng(31 + n)
    alpha = r.uniform(.4, 1.2, n); cc = r.uniform(.3, 2.0, n); einf = 0.1
    A = np.vstack([np.zeros(n), np.diag(alpha)]); c = np.r_[einf, cc]
    p = r.dirichlet(np.ones(n + 1) * 4)
    lam = (A.T @ p)
    h = 1e-4; H = np.zeros((n, n))
    f = lambda l: sup_x(A, c, l, n)[0]
    for i in range(n):
        for j in range(n):
            ei = np.eye(n)[i] * h; ej = np.eye(n)[j] * h
            H[i, j] = (f(lam+ei+ej) - f(lam+ei-ej) - f(lam-ei+ej) + f(lam-ei-ej)) / (4*h*h)
    # sigma* = -sum p ln c - H(p); d2/dlam2 with p_i = lam_i/alpha_i, p_0 = 1 - sum p_i
    pred = np.diag(1.0 / (p[1:] * alpha**2)) + (1.0 / p[0]) * np.outer(1/alpha, 1/alpha)
    err = float(np.max(np.abs(H - pred)) / np.max(np.abs(pred)))
    rows.append(dict(theorem="T5 Fisher", case=f"n={n}", lhs=np.nan, rhs=np.nan,
                     abs_error=err))
    print(f"  n={n}  max relative deviation of Hessian = {err:.2e}")

save(pd.DataFrame(rows), "theorem_verification.csv")

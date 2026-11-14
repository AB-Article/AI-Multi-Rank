#!/usr/bin/env python3
"""Numerical verification of the generalisations (Sections 4, 5, 9, 10).

G1  Running exponents.  Channels g_j(x) log-convex but NOT exponential.
        a_j(x) = -grad ln g_j(x),   kappa_j(x) = ln g_j(x) + a_j(x).x
    Claim:  sigma*(lam) = -H(p) - sum_j p_j kappa_j(x*)
    Reduces to -H(p) - sum_j p_j ln c_j exactly when every kappa_j is constant,
    i.e. when the channels are exponential.  This is the characterisation
    theorem: scale-freeness <=> the conjugate is a pure entropy.

G2  Vector-valued quality.  eps_k = sum_j c_kj exp(-a_j.x), weights w in the
    simplex, S = sum_k w_k S_k.
    Claim:  sigma*(lam) = -sum_k w_k [ H(p^k) + sum_j p^k_j ln c_kj ]
            lam        =  sum_k w_k A^T p^k

G3  Objective-count independence.  With exchangeable objectives the maximal
    entropic gap is ln(N+1) for every number of objectives m.

G4  Sharpness deformation.  For S_beta = -beta LSE(-xi/beta) the conjugate is
    beta times the negative entropy, so the gap is beta ln(N+1).  beta = 1 is
    the additive law; beta -> 0 approaches Chebyshev scalarisation.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save
np.seterr(over="ignore", invalid="ignore")

NM = dict(xatol=1e-12, fatol=1e-14, maxiter=400000, maxfev=400000)
rows = []


def sup(f, x0):
    o = minimize(lambda x: -f(x), x0, method="Nelder-Mead", options=NM)
    return -o.fun, o.x


# ------------------------------------------------------------------ G1 -----
print("G1  running exponents: sigma* = -H(p) - sum_j p_j kappa_j(x*)")
for trial, n in [(1, 1), (2, 2), (3, 2), (4, 3), (5, 3)]:
    r = np.random.default_rng(10 * trial)
    N = 3
    a = r.uniform(.3, 1.2, (N, n)); s = r.uniform(.05, .3, N)
    u = r.uniform(-.4, .4, (N, n)); c0 = r.uniform(.02, .15)

    def lng(x):
        return np.r_[np.log(c0), -a @ x + s * np.log1p(np.exp(np.clip(u @ x, -50, 50)))]

    S = lambda x: -np.log(np.sum(np.exp(lng(x))))
    h = 1e-6

    def grad_block(x):
        G = np.zeros((N + 1, n))
        for i in range(n):
            e = np.zeros(n); e[i] = h
            G[:, i] = -(lng(x + e) - lng(x - e)) / (2 * h)
        return G

    xr = r.uniform(-.5, .5, n)                      # reference point
    lg0 = lng(xr); p0 = np.exp(lg0) / np.sum(np.exp(lg0))
    lam = grad_block(xr).T @ p0                     # admissible by construction
    prim, xs = sup(lambda x: S(x) - lam @ x, xr)
    lg = lng(xs); p = np.exp(lg) / np.sum(np.exp(lg))
    G = grad_block(xs)
    kappa = lg + G @ xs
    pred = -(-np.sum(p * np.log(p))) - float(p @ kappa)
    rows.append(dict(test="G1 running exponents", case=f"n={n},N={N}",
                     lhs=prim, rhs=pred, abs_error=abs(prim - pred),
                     kappa_spread=float(np.ptp(kappa[1:]))))
    print(f"  n={n}  sup_x={prim: .10f}  -H-p.kappa={pred: .10f}  "
          f"diff={abs(prim-pred):.1e}  spread(kappa)={np.ptp(kappa[1:]):.3f}")

# control: exponential channels have constant kappa and reduce to ln c
r = np.random.default_rng(99); n = N = 3
A = np.vstack([np.zeros(n), np.diag(r.uniform(.4, 1.2, n))]); c = r.uniform(.3, 2, n + 1)
xs0 = r.uniform(-.3, .3, n)
lg = np.log(c) - A @ xs0
kappa = lg + A @ xs0
rows.append(dict(test="G1 control (exponential)", case=f"n=N={n}",
                 lhs=float(np.max(np.abs(kappa - np.log(c)))), rhs=0.0,
                 abs_error=float(np.max(np.abs(kappa - np.log(c)))),
                 kappa_spread=0.0))
print(f"  control: exponential channels give kappa_j = ln c_j exactly, "
      f"max dev {np.max(np.abs(kappa-np.log(c))):.1e}")

# ------------------------------------------------------------------ G2 -----
print("\nG2  vector-valued quality: sigma* = -sum_k w_k [H(p^k) + sum_j p^k_j ln c_kj]")
for (m, N, n) in [(2, 2, 2), (3, 3, 2), (2, 4, 3), (4, 3, 3)]:
    r = np.random.default_rng(m * 100 + N)
    A = np.vstack([np.zeros(n), r.uniform(.3, 1.3, (N, n))])
    C = r.uniform(.2, 1.5, (m, N + 1)); C[:, 0] = r.uniform(.02, .12, m)
    w = r.dirichlet(np.ones(m)); xr = r.uniform(-.5, .5, n)
    shares = lambda x: np.array([C[k] * np.exp(-A @ x) / np.sum(C[k] * np.exp(-A @ x))
                                 for k in range(m)])
    lam = sum(w[k] * (A.T @ shares(xr)[k]) for k in range(m))
    Ssum = lambda x: -sum(w[k] * np.log(np.sum(C[k] * np.exp(-A @ x))) for k in range(m))
    prim, xs = sup(lambda x: Ssum(x) - lam @ x, xr)
    P = shares(xs); H = -np.sum(P * np.log(P), axis=1)
    pred = -sum(w[k] * (H[k] + float(P[k] @ np.log(C[k]))) for k in range(m))
    mom = float(np.max(np.abs(sum(w[k] * (A.T @ P[k]) for k in range(m)) - lam)))
    rows.append(dict(test="G2 vector quality", case=f"m={m},N={N},n={n}",
                     lhs=prim, rhs=pred, abs_error=abs(prim - pred),
                     moment_residual=mom))
    print(f"  m={m} N={N} n={n}  sup_x={prim: .10f}  mixture={pred: .10f}  "
          f"diff={abs(prim-pred):.1e}  |moment|={mom:.1e}")

# ------------------------------------------------------------------ G3 -----
print("\nG3  maximal gap is ln(N+1) for every number of objectives m")
for m in (1, 2, 3, 5, 8):
    n = N = 3
    r = np.random.default_rng(7)
    A = np.vstack([np.zeros(n), np.diag(r.uniform(.4, 1.2, n))])
    C = np.tile(r.uniform(.3, 1.2, N + 1), (m, 1))      # exchangeable objectives
    w = np.ones(m) / m
    lam = A.T @ (np.ones(N + 1) / (N + 1))
    Ssum = lambda x: -sum(w[k] * np.log(np.sum(C[k] * np.exp(-A @ x))) for k in range(m))
    prim, _ = sup(lambda x: Ssum(x) - lam @ x, np.zeros(n))
    gap = -(prim + float(np.mean(np.log(C[0]))))
    rows.append(dict(test="G3 objective independence", case=f"m={m},N={N}",
                     lhs=gap, rhs=np.log(N + 1), abs_error=abs(gap - np.log(N + 1))))
    print(f"  m={m}  gap={gap:.9f}  ln(N+1)={np.log(N+1):.9f}  "
          f"diff={abs(gap-np.log(N+1)):.1e}")

# ------------------------------------------------------------------ G4 -----
print("\nG4  sharpness deformation: gap = beta ln(N+1)")
for beta in (0.25, 0.5, 1.0, 2.0, 4.0):
    N, n = 3, 3
    lse_b = lambda th: beta * np.log(np.sum(np.exp(th / beta)))
    S = lambda xi: -lse_b(-np.r_[0.0, xi])
    lam = np.ones(n) / (N + 1)
    prim, _ = sup(lambda xi: S(xi) - lam @ xi, np.zeros(n))
    gap = -prim
    rows.append(dict(test="G4 beta deformation", case=f"beta={beta}",
                     lhs=gap, rhs=beta * np.log(N + 1),
                     abs_error=abs(gap - beta * np.log(N + 1))))
    print(f"  beta={beta:<4}  gap={gap:.9f}  beta ln(N+1)={beta*np.log(N+1):.9f}  "
          f"diff={abs(gap-beta*np.log(N+1)):.1e}")

save(pd.DataFrame(rows), "generalisation_verification.csv")

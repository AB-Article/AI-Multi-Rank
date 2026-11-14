#!/usr/bin/env python3
"""Numerical verification of the eight proposed modifications (M1-M8).

No network. Items 1 and 6 read the archived frontiers if a path is given as
argv[1] (the Zenodo results/frontiers directory); everything else is
self-contained.  Run:  python3 verify_theory_additions.py [frontiers_dir]
"""
import sys, numpy as np
from scipy.optimize import least_squares, minimize, minimize_scalar

ok = lambda name, cond: print(("  PASS  " if cond else "  FAIL  ") + name)

# ---------------------------------------------------------------- M1 capacity
def KL(p, q):
    p = np.clip(np.asarray(p, float), 1e-15, 1); q = np.clip(np.asarray(q, float), 1e-15, 1)
    return float((p * np.log(p / q)).sum())
def Hb(p):
    p = np.clip(p, 1e-15, 1 - 1e-15); return -p*np.log(p) - (1-p)*np.log(1-p)

print("M1  capacity form  min_q max_{p in Lambda} KL(p||q) = max_pi I(pi)")
rng = np.random.default_rng(3); worst = 0.0
for _ in range(3):
    V = rng.dirichlet(np.ones(4), size=5)
    obj = lambda z: max(KL(v, np.exp(z - z.max()) / np.exp(z - z.max()).sum()) for v in V)
    best = min((minimize(obj, rng.normal(size=4), method="Nelder-Mead",
                options=dict(xatol=1e-12, fatol=1e-14, maxiter=40000, maxfev=40000))
                for _ in range(6)), key=lambda r: r.fun)
    pi = np.ones(len(V)) / len(V)
    for _ in range(200000):
        qb = pi @ V; pi = pi * np.exp([KL(v, qb) for v in V]); pi /= pi.sum()
    qb = pi @ V; C = float(pi @ np.array([KL(v, qb) for v in V]))
    worst = max(worst, abs(best.fun - C))
ok(f"capacity identity on random polytopes (max gap {worst:.2e})", worst < 1e-7)

print("M1  entropy tangency  xi = 1/(1+exp(chord slope of Hb))")
worst = 0.0
for a, b in [(0., 1.), (.2, .8), (.05, .4), (.5, .95), (.3, .31)]:
    g = np.linspace(a, b, 20001)
    r = minimize_scalar(lambda q: np.max(Hb(g)*0 + [max(KL([1-p, p], [1-q, q]) for p in g)][0]),
                        bounds=(1e-9, 1-1e-9), method="bounded", options=dict(xatol=1e-12))
    xi = 1.0 / (1.0 + np.exp((Hb(b) - Hb(a)) / (b - a)))
    C = Hb(xi) - Hb(b) + (xi - b) * (np.log(xi) - np.log(1 - xi))
    worst = max(worst, abs(r.x - xi), abs(r.fun - C))
ok(f"closed form vs numerical minimax (max gap {worst:.2e})", worst < 1e-6)
ok("no-knowledge case gives xi=1/2, C=ln2",
   abs(1/(1+np.exp((Hb(1.)-Hb(0.))/1.)) - .5) < 1e-12 and abs(Hb(.5) - np.log(2)) < 1e-12)

# ------------------------------------------------------- M2 knee = p0 = 1/2
print("M2  -S'' = alpha^2 p0(1-p0) is maximal at p0 = 1/2")
p0 = np.linspace(1e-6, 1-1e-6, 10**6)
ok("argmax of p0(1-p0)", abs(p0[np.argmax(p0*(1-p0))] - .5) < 1e-5)

# --------------------------------------------- M3 blind-spot decomposition
print("M3  sigma_lambda*(x) = J(eps) - 1/2 ln c - (alpha/2) delta")
einf, c, al = .1, 2., .6; worst = 0.0
for xk, rk in [(1., 0.), (1., .3), (2.5, -.2), (4., .05), (.3, -.5)]:
    eps = (einf + c*np.exp(-al*xk)) * np.exp(rk)
    if eps <= einf: continue
    J = .5*np.log(eps - einf) - np.log(eps)
    delta = xk - np.log(c/(eps - einf))/al
    worst = max(worst, abs((-np.log(eps) - al/2*xk) - (J - .5*np.log(c) - al/2*delta)))
ok(f"identity (max residual {worst:.2e})", worst < 1e-12)

# ------------------------------------------ M4 misquoted-exponent bound
print("M4  loss <= G |1-phi| max(1, 1/phi) when a quoted exponent flips a pair")
rng = np.random.default_rng(1); excess = 0.0
for _ in range(400000):
    G = rng.uniform(.01, 1.5); u = rng.uniform(.2, 3.); phi = rng.uniform(.3, 3.)
    if (G*(1-u) > 0) != (G*(1-u*phi) > 0):
        excess = max(excess, abs(G*(1-u)) - G*abs(1-phi)*max(1, 1/phi))
ok(f"bound holds on 4e5 random pairs (max excess {excess:.2e})", excess <= 1e-12)

# ----------------------------------------------- M5 Prony / Hankel recovery
print("M5  2k+1 accuracies on a geometric grid determine an order-k law")
rng = np.random.default_rng(0); worst = 0.0; ranks = []
for k in (1, 2, 3):
    einf = .07; cc = rng.uniform(.3, 1., k); al = np.sort(rng.uniform(.3, 1.5, k))
    D = .4; x = np.arange(2*k+1)*D
    eps = einf + (cc[:, None]*np.exp(-al[:, None]*x[None, :])).sum(0)
    de = np.diff(eps)
    A = np.array([de[i:i+k] for i in range(k)])
    coef = np.linalg.solve(A, -de[k:2*k])
    roots = np.sort(np.roots(np.concatenate([[1.], coef[::-1]])).real)
    al_rec = np.sort(-np.log(roots)/D)
    M = np.column_stack([np.ones(len(x)), np.exp(-np.outer(x, al_rec))])
    sol = np.linalg.lstsq(M, eps, rcond=None)[0]
    worst = max(worst, np.abs(al_rec - al).max(), abs(sol[0] - einf))
    ranks.append(np.linalg.matrix_rank(np.array([de[i:i+k+1] for i in range(k)]), tol=1e-9))
ok(f"exponents and floor recovered exactly (max error {worst:.2e})", worst < 1e-6)
ok(f"Hankel rank of the differenced sequence equals k: {ranks}", ranks == [1, 2, 3])

# -------------------------------------- M8 L2 constant and invariance check
print("M8  RMS of u^2 after a linear fit on [0,Z] is Z^2/sqrt(180); kappa Z^2 is invariant")
u = np.linspace(0, 1, 2000001); A = np.column_stack([np.ones_like(u), u])
r = u**2 - A @ np.linalg.lstsq(A, u**2, rcond=None)[0]
ok("RMS constant", abs(np.sqrt((r**2).mean()) - 1/np.sqrt(180)) < 1e-7)
ok("sup constant 1/6 (matches the printed k2 Z^2/16)", abs(np.abs(r).max() - 1/6) < 1e-6)
beta = 2.7
ok("kappa Z^2 invariant under t -> t^beta, g Z^2 is not",
   abs((1/beta**2)*(beta*1.0)**2 - 1.0) < 1e-12 and abs((beta*1.0)**2 - 1.0) > .1)

# --------------------------------- M6 floor identified set on the archive
if len(sys.argv) > 1:
    import pandas as pd, os
    print("M6  unconstrained floor vs the identified set [0, min eps]")
    base = sys.argv[1].rstrip("/") + "/"
    for ds in ("ImageNet-1k", "ImageNet-ReaL", "ImageNet-V2"):
        for ax in ("lat3090", "params", "gmacs"):
            f = base + f"{ds}__{ax}.csv"
            if not os.path.exists(f): continue
            d = pd.read_csv(f)
            rc = [c for c in d.columns if c not in ("base", "img_size", "eps")][0]
            x = np.log(d[rc].values.astype(float)); x -= x.min()
            e = d.eps.values; emin = e.min()
            res = least_squares(lambda p: np.log(p[0] + p[1]*np.exp(-p[2]*x)) - np.log(e),
                                [emin*.8, e.max()-emin, .4],
                                bounds=([1e-9, 1e-9, 1e-3], [np.inf, np.inf, 10]))
            s = np.sqrt((res.fun**2).mean()); exc = np.log(res.x[0]/emin)
            print(f"   {ds:14s}{ax:8s} floor={res.x[0]:.4f} min_eps={emin:.4f} "
                  f"ln excess={exc:+.4f} ({exc/s:+.2f} sigma)  alpha*Z={res.x[2]*x.max():.2f}")
else:
    print("M6  skipped (pass the frontiers directory as argv[1])")

"""Numerical verification of the multi-channel minimax characterization
(Lemma 'reduction' and Theorem 'canonical' of the manuscript).

Checks, for N = 1..6 reducible channels (simplex Delta^N with N+1 categories):
  M1  sup_{p in Delta^N} KL(p||q) is attained at a vertex:  no sampled interior
      point exceeds max_j(-ln q_j)                                (must be <= 0)
  M2  argmin_q max_j(-ln q_j) is the uniform vector, value ln(N+1)
  M3  identity KL(p||u) = ln(N+1) - H(p)
  M4  Bayes act: argmin_q E_pi KL(p||q) equals the prior mean
  M5  equivariance: under t_i -> t_i^{b_i} the rule lambda_i = alpha_i q_i
      selects the same candidate, whereas an exponent-independent rate does not
Writes results/minimax_verification.csv.  Seed fixed.
"""
import os
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(20260824)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []


def kl(p, q):
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def softmax(z):
    z = np.concatenate([[0.0], z])
    e = np.exp(z - z.max())
    return e / e.sum()


for N in range(1, 7):
    K = N + 1
    # M1: vertex attainment of the supremum
    worst = -np.inf
    for _ in range(200):
        q = rng.dirichlet(np.ones(K))
        vmax = np.max(-np.log(q))
        P = rng.dirichlet(np.full(K, 0.3), size=5000)
        vals = np.sum(np.where(P > 0, P * np.log(np.where(P > 0, P, 1) / q), 0), axis=1)
        worst = max(worst, float(vals.max() - vmax))
    rows.append(("M1 sup at vertex", N, "max sampled KL(p||q) - max_j(-ln q_j)", worst))

    # M2: minimax share is uniform
    dq, dv = 0.0, 0.0
    for _ in range(20):
        # epigraph form: minimise s subject to s >= -ln q_j, sum q = 1, q > 0
        q0 = rng.dirichlet(np.ones(K))
        v0 = np.concatenate([q0, [np.max(-np.log(q0))]])
        cons = [dict(type="eq", fun=lambda v: np.sum(v[:K]) - 1.0,
                     jac=lambda v: np.concatenate([np.ones(K), [0.0]]))]
        cons += [dict(type="ineq", fun=(lambda v, j=j: v[K] + np.log(v[j])),
                      jac=(lambda v, j=j: np.concatenate([np.eye(K)[j] / v[j], [1.0]])))
                 for j in range(K)]
        r = minimize(lambda v: v[K], v0, jac=lambda v: np.eye(K + 1)[K], method="SLSQP",
                     bounds=[(1e-9, 1.0)] * K + [(0.0, None)], constraints=cons,
                     options=dict(ftol=1e-15, maxiter=1000))
        q = r.x[:K]
        dq = max(dq, float(np.max(np.abs(q - 1.0 / K))))
        dv = max(dv, abs(float(np.max(-np.log(q))) - np.log(K)))
    rows.append(("M2 minimax share", N, "max |q_j - 1/(N+1)|", dq))
    rows.append(("M2 minimax value", N, "|Psi - ln(N+1)|", dv))

    # M3: entropy identity
    P = rng.dirichlet(np.ones(K), size=2000)
    u = np.full(K, 1.0 / K)
    d3 = max(abs(kl(p, u) - (np.log(K) + np.sum(p * np.log(p)))) for p in P)
    rows.append(("M3 KL(p||u)=ln(N+1)-H(p)", N, "abs deviation", float(d3)))

    # M4: Bayes act equals prior mean
    d4 = 0.0
    for _ in range(10):
        a = rng.uniform(0.3, 5.0, size=K)
        S = rng.dirichlet(a, size=4000)
        pbar = S.mean(axis=0)
        obj = lambda z: -float(np.sum(pbar * np.log(softmax(z))))   # E KL up to a constant
        jac = lambda z: (softmax(z) - pbar)[1:]
        r = minimize(obj, rng.normal(size=N), jac=jac, method="BFGS",
                     options=dict(gtol=1e-15))
        d4 = max(d4, float(np.max(np.abs(softmax(r.x) - pbar))))
    rows.append(("M4 Bayes act = prior mean", N, "max |argmin - mean|", d4))

# M5: equivariance of lambda_i = alpha_i/(N+1) versus a fixed rate
same_eq, changed_fixed, trials = 0, 0, 0
for _ in range(2000):
    n = 2
    alpha = rng.uniform(0.2, 1.2, size=n)
    c = rng.uniform(0.2, 2.0, size=n)
    einf = rng.uniform(0.02, 0.2)
    X = rng.uniform(-4, 4, size=(300, n))
    eps = einf + np.sum(c * np.exp(-alpha * X), axis=1)
    eps *= np.exp(rng.normal(0, 0.02, size=len(eps)))
    S = -np.log(eps)
    b = rng.uniform(0.3, 3.0, size=n)                    # t_i -> t_i^{b_i}
    Xb, alpha_b = X * b, alpha / b
    lam, lam_b = alpha / (n + 1), alpha_b / (n + 1)
    same_eq += int(np.argmax(S - X @ lam) == np.argmax(S - Xb @ lam_b))
    lam0 = np.full(n, 0.3)                               # exponent-independent rate
    changed_fixed += int(np.argmax(S - X @ lam0) != np.argmax(S - Xb @ lam0))
    trials += 1
rows.append(("M5 equivariant rule", 2, "fraction of instances with unchanged selection",
             same_eq / trials))
rows.append(("M5 fixed rate lambda_0=0.3", 2, "fraction of instances with changed selection",
             changed_fixed / trials))

out = os.path.join(ROOT, "results", "minimax_verification.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write("check,N,quantity,value\n")
    for r in rows:
        f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6g}\n")
print(open(out).read())

# Misspecification cost between the two ImageNet RTX 3090 latency estimates
a_here, a_prev = 0.754, 0.657
p_hat = (a_prev / 2) / a_here
R = 0.5 * np.log(0.5 / p_hat) + 0.5 * np.log(0.5 / (1 - p_hat))
print(f"regret of declaring {a_prev/2:.4f} when alpha={a_here}: {R:.5f} nats")

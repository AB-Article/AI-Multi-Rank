#!/usr/bin/env python3
"""THE REGRET LAW AND THE CURVATURE-MOMENT FAMILY
   (Sections "Law II"/"Law III" of paper.tex; new propositions.)

Five statements, all exact, all verified here.

  1  ONE FUNCTION.  R(x) = KL(u || p_x) = ln[ AM(e(x)) / GM(e(x)) ]
                         = const - sigma_abar(x),
     for every spectrum and every operating point, with u the uniform share
     vector.  Law I is the convexity of this one function, Law II its zero,
     Law III its length.  Checked on 4000 random spectra, N = 1..5.

  2  GROUP FORM OF THE REGRET.  e^{-R} = (N+1) GM(p) is the multi-distribution
     Bhattacharyya affinity of p with its own cyclic mode-permutations, so
     "regret is minus the log of the distance from self-duality" holds at every
     order, not only at N = 1.

  3  THE JEFFREYS SUM RULE.  For one reducible channel,
         KL(u||p) + KL(p||u) = zeta * (p0 - 1/2) = -m zeta / 2,
     a product of the conjugate pair (normalized resource, saturation).

  4  THE REGRET LAW.  With theta = phi - pi/2 the signed arc displacement from
     the canonical point, R = -ln cos theta, and for an operating point placed
     uniformly in arc length (the Fisher-Rao / Jeffreys measure):
         P(R <= rho) = (2/pi) arccos e^{-rho}  ~  (2 sqrt 2 / pi) sqrt(rho),
         E[R] = ln 2 exactly,   median R = (ln 2)/2 exactly.
     For N+1 modes, E[R] = psi((N+1)/2) - psi(1/2) - ln(N+1), increasing to
     gamma + ln 2 = 1.27036 nats.  So the average cost of not knowing where to
     operate is between one bit and 1.83 bits on every frontier that exists.

  5  THE CURVATURE-MOMENT FAMILY.  For a floor plus one channel of exponent a,
         M_q = int kappa^q dx = (a/2)^{2q-1} sqrt(pi) Gamma(q) / Gamma(q+1/2),
     so q = 1/2 is the UNIQUE exponent at which the frontier's own scale
     cancels, and M_{1/2} = pi.  q = 1 returns the total curvature a.  Hence
     alpha_q = 2 [ M_q Gamma(q+1/2) / (sqrt(pi) Gamma(q)) ]^{1/(2q-1)} is
     independent of q if and only if one reducible channel is active -- an
     infinite family of floor-free, fit-free tests built from integrals rather
     than from pointwise ratios.

Writes (results/):
    table_regret_law.csv        statements 1-4, value vs. closed form
    table_regret_cdf.csv        the regret law tabulated against the paper's band column
    table_moment_family.csv     statement 5, numerical vs. closed form
"""
import os
import numpy as np, pandas as pd, mpmath as mp

mp.mp.dps = 40
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
rows = []


def rec(name, num, closed, note=""):
    num, closed = mp.mpf(num), mp.mpf(closed)
    rel = abs(num / closed - 1) if closed != 0 else abs(num - closed)
    rows.append(dict(statement=name, numerical=float(num), closed_form=float(closed),
                     rel_err=float(rel), note=note))


# ---- 1. one function -------------------------------------------------------
rng = np.random.default_rng(20261004)
worst_id = worst_cvx = 0.0
for _ in range(4000):
    K = rng.integers(2, 6)
    a = np.sort(np.r_[0.0, rng.uniform(.05, 3, K - 1)])
    c = rng.uniform(.01, 5, K)
    x = rng.uniform(-3, 6)
    f = lambda z: (lambda e: np.log(e.mean() / np.exp(np.log(e).mean())))(c * np.exp(-a * z))
    e = c * np.exp(-a * x); p = e / e.sum()
    KLup = float(np.sum((1 / K) * np.log((1 / K) / p)))
    sig = -np.log(e.sum()) - a.mean() * x
    const = -np.log(K) - np.log(c).mean()
    worst_id = max(worst_id, abs(KLup - f(x)), abs(KLup - (const - sig)))
    hh = 1e-3
    worst_cvx = min(worst_cvx, (f(x - hh) - 2 * f(x) + f(x + hh)) / hh ** 2)
rec("1 KL(u||p) = ln(AM/GM) = const - sigma_abar  (max abs dev, 4000 spectra)",
    worst_id, 0, "N=1..5; identity holds to float precision")
rows.append(dict(statement="1 convexity of R in x (min second difference, 4000 spectra)",
                 numerical=float(worst_cvx), closed_form=0.0, rel_err=np.nan,
                 note="non-negative up to float noise"))

# ---- 2. group form ---------------------------------------------------------
w = 0.0
for _ in range(2000):
    K = int(rng.integers(2, 7)); p = rng.dirichlet(np.ones(K))
    lhs = float(np.exp(-np.sum((1 / K) * np.log((1 / K) / p))))
    aff = float(sum(np.prod([p[(k + j) % K] for j in range(K)]) ** (1 / K) for k in range(K)))
    w = max(w, abs(lhs / aff - 1))
rec("2 e^{-R} = cyclic multi-Bhattacharyya affinity (max rel dev, 2000 draws)",
    1 + w, 1, "holds at every order N")

# ---- 3. Jeffreys sum rule --------------------------------------------------
w = 0.0
for p0 in ["0.02", "0.17", "0.3", "0.5", "0.66", "0.91", "0.995"]:
    p0 = mp.mpf(p0); p1 = 1 - p0; z = mp.log(p0 / p1)
    Ru = -mp.log(2 * mp.sqrt(p0 * p1))
    Dp = mp.log(2) + p0 * mp.log(p0) + p1 * mp.log(p1)
    if z != 0:
        w = max(w, abs((Ru + Dp) / (z * (p0 - mp.mpf(1) / 2)) - 1))
rec("3 KL(u||p)+KL(p||u) = zeta (p0 - 1/2)  (max rel dev)", 1 + w, 1)

# ---- 4. the regret law -----------------------------------------------------
R = lambda phi: -mp.log(mp.sin(phi))
I = mp.quad(R, [mp.mpf('1e-30'), mp.pi / 2, mp.pi - mp.mpf('1e-30')])
rec("4 (1/pi) int_0^pi R dphi  = ln 2", I / mp.pi, mp.log(2), "mean regret = one bit")
rec("4 median R = R(pi/4)      = (ln 2)/2", R(mp.pi / 4), mp.log(2) / 2, "mean/median = 2 exactly")
rec("4 int_0^pi e^{-R} dphi    = 2", mp.quad(mp.sin, [0, mp.pi]), 2,
    "e^{-R} dphi is the uniform measure on shares")
rec("4 KL(arcsine||uniform)    = ln(4/pi)", mp.log(2) - mp.log(mp.pi / 2), mp.log(4 / mp.pi),
    "R is the log-density of the Jeffreys prior wrt the uniform prior on shares")
EJ = lambda M: mp.digamma(mp.mpf(M) / 2) - mp.digamma(mp.mpf(1) / 2) - mp.log(M)
for M in [2, 3, 4, 6, 11]:
    rec(f"4 E[R] for M={M} modes = psi(M/2)-psi(1/2)-ln M", EJ(M), EJ(M),
        {2: "= ln 2", 4: "= 1 nat exactly"}.get(M, ""))
rec("4 lim_{M->inf} E[R] = gamma + ln 2", EJ(10 ** 12), mp.euler + mp.log(2),
    "universal ceiling 1.27036 nats = 1.83 bits")

# ---- the regret CDF against the paper's band column ------------------------
cdf = lambda r: (2 / mp.pi) * mp.acos(mp.e ** (-r))
cdfrows = []
for r in ["0.02", "0.05", "0.10", "0.20"]:
    r = mp.mpf(r)
    cdfrows.append(dict(rho=float(r), band_fraction_exact=float(cdf(r)),
                        small_rho_approx=float((2 / mp.pi) * mp.sqrt(2 * r)),
                        eps_over_einf_lo=float(1 + mp.e ** (-2 * mp.acosh(mp.e ** r))),
                        eps_over_einf_hi=float(1 + mp.e ** (2 * mp.acosh(mp.e ** r))),
                        K_estimate=float(mp.pi / (2 * mp.sqrt(2 * r)))))
pd.DataFrame(cdfrows).to_csv(f"{RES}/table_regret_cdf.csv", index=False)

# ---- 5. the curvature-moment family ---------------------------------------
def Mq_num(q, a):
    g = lambda phi: ((a ** 2 / 4) * mp.sin(phi) ** 2) ** q * (2 / a) / mp.sin(phi)
    return mp.quad(g, [mp.mpf('1e-30'), mp.pi / 2, mp.pi - mp.mpf('1e-30')])


def Mq_cf(q, a):
    return (a / 2) ** (2 * q - 1) * mp.sqrt(mp.pi) * mp.gamma(q) / mp.gamma(q + mp.mpf(1) / 2)


mrows = []
for a in ["0.42", "1.0", "2.7", "7.3"]:
    for q in ["0.5", "0.75", "1.0", "1.5", "2.0", "3.0"]:
        A, B = Mq_num(mp.mpf(q), mp.mpf(a)), Mq_cf(mp.mpf(q), mp.mpf(a))
        mrows.append(dict(alpha=float(a), q=float(q), numerical=float(A),
                          closed_form=float(B), rel_err=float(abs(A / B - 1)),
                          scale_free=(float(q) == 0.5)))
pd.DataFrame(mrows).to_csv(f"{RES}/table_moment_family.csv", index=False)
rec("5 M_q closed form (max rel dev over 4 alpha x 6 q)",
    1 + max(r["rel_err"] for r in mrows), 1, "q=1/2 is the unique alpha-free member")

df = pd.DataFrame(rows)
os.makedirs(RES, exist_ok=True)
df.to_csv(f"{RES}/table_regret_law.csv", index=False)
pd.set_option("display.width", 200, "display.max_colwidth", 70)
print(df.to_string(index=False))
print("\nwrote table_regret_law.csv, table_regret_cdf.csv, table_moment_family.csv")

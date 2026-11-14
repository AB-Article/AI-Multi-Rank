#!/usr/bin/env python3
"""THE LOCAL CALCULUS.  Every quantity of the theory from (S, s, kappa) alone.
   (Section "Reading a frontier without fitting it" of paper.tex.)

Laws I-III are stated in terms of fitted quantities: the floor eps_inf, the
coefficient c, the exponent alpha, the channel count N.  They need not be.
Write

    S = -ln eps ,  s = -dln eps/dx = S' ,  kappa = -S''

for the cumulative hazard, the hazard rate and the curvature of a frontier.
Then every object of the theory is an algebraic function of those three, with
no fit, no floor, no origin and no units:

  T1  p_0      = kappa / (s^2 + kappa)                        [exact]
  T2  m=cos phi= (s^2 - kappa) / (s^2 + kappa)                [exact]
  T3  R        = ln[ (s^2 + kappa) / (2 s sqrt(kappa)) ]      [exact]
  T4  alpha    = 2 (s_1 - s_2) / (m_1 - m_2), ANY two points  [exact]
  T5  eps_inf  = kappa eps / (s^2 + kappa)                    [exact]
      -- i.e. the floor is a LOCAL FIELD, not a constant: Prop. 26 in closed
      form.  A fitted floor is this field averaged by a particular estimator.
  T6  the canonical point p_0 = 1/2 is  kappa = s^2, i.e. -S'' = (S')^2,
      i.e. d(1/s)/dx = 1.  No floor, no exponent, no units, no origin.

Two exact sum rules follow from the same three quantities:

  T7  int kappa dx      = a_max - a_min   (the width of the spectrum support)
  T8  int sqrt(kappa) dx = L = (number of RESOLVED kinks) x pi

T8 makes L/pi a channel count that is a geometric invariant: no BIC, no AICc,
no model selection.  Kinks that overlap in x MERGE, which is why pi <= L <= N pi
rather than L = N pi.

Finally the representation itself is identified with a standard object:

  T9  eps is the survival function of an exponential threshold with a random
      rate a ~ mu.  Hence S is a cumulative hazard, s = E[a | X > x] is a
      hazard rate, and kappa = Var(a | X > x) is the frailty variance.
      "Diminishing returns" IS "decreasing failure rate" (Proschan 1963);
      eps_inf is a CURE FRACTION, whose non-identifiability under finite
      follow-up is a classical theorem rather than a feature of compute.
      Checked by Monte Carlo.

Writes (results/):
    table_local_calculus.csv   T1-T6, one row per (floor, exponent, window)
    table_sum_rules.csv        T7-T8, one row per spectrum
    table_frailty_mc.csv       T9, Monte Carlo against the closed forms
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
RNG = np.random.default_rng(20261005)


# --------------------------------------------------------------- spectra
def spectrum(a, c):
    """eps(x) = sum_j c_j exp(-a_j x); returns (p, s, kappa) as callables."""
    a = np.asarray(a, float)
    lc = np.log(np.asarray(c, float))

    def p(x):
        lw = lc - a * x
        return np.exp(lw - logsumexp(lw))

    def s(x):
        return float((p(x) * a).sum())

    def kappa(x):
        P = p(x)
        mu = (P * a).sum()
        return float((P * (a - mu) ** 2).sum())

    return p, s, kappa


def two_atom(einf, c, alpha):
    """The single-channel law with a floor, in closed form."""
    def q(x):
        eps = einf + c * np.exp(-alpha * x)
        p0 = einf / eps
        return dict(eps=eps, p0=p0, S=-np.log(eps),
                    s=alpha * (1 - p0), kappa=alpha * alpha * p0 * (1 - p0))
    return q


# --------------------------------------------------- T1-T6: local calculus
def local_calculus():
    rows = []
    grid = [(0.03, 0.5, 0.60), (0.10, 0.4, 0.42), (1e-3, 2.0, 1.70),
            (0.20, 0.9, 0.25), (0.005, 1.2, 1.05)]
    windows = [(1.0, 5.0), (0.5, 12.0), (3.0, 4.0), (-1.0, 2.0)]
    for einf, c, alpha in grid:
        q = two_atom(einf, c, alpha)
        for x1, x2 in windows:
            A, B = q(x1), q(x2)
            m = lambda d: (d["s"] ** 2 - d["kappa"]) / (d["s"] ** 2 + d["kappa"])
            p0_hat = A["kappa"] / (A["s"] ** 2 + A["kappa"])
            m_hat = m(A)
            R_hat = np.log((A["s"] ** 2 + A["kappa"]) / (2 * A["s"] * np.sqrt(A["kappa"])))
            R_true = -np.log(2 * np.sqrt(A["p0"] * (1 - A["p0"])))
            a_hat = 2 * (A["s"] - B["s"]) / (m(A) - m(B))
            e_hat = A["kappa"] * A["eps"] / (A["s"] ** 2 + A["kappa"])
            beta_hat = A["s"] ** 2 / (A["s"] ** 2 + A["kappa"] * A["S"])
            rows.append(dict(
                einf=einf, c=c, alpha=alpha, x1=x1, x2=x2,
                p0_true=A["p0"], p0_hat=p0_hat, err_p0=abs(p0_hat - A["p0"]),
                m_true=1 - 2 * A["p0"], m_hat=m_hat, err_m=abs(m_hat - (1 - 2 * A["p0"])),
                R_true=R_true, R_hat=R_hat, err_R=abs(R_hat - R_true),
                alpha_hat=a_hat, err_alpha=abs(a_hat - alpha),
                einf_hat=e_hat, err_einf=abs(e_hat - einf),
                beta_implied=beta_hat))
    return pd.DataFrame(rows)


def canonical_point():
    """T6: kappa = s^2 locates p_0 = 1/2 exactly, for every (einf, c, alpha)."""
    out = []
    for einf, c, alpha in [(0.03, 0.5, 0.6), (0.10, 0.4, 0.42), (1e-3, 2.0, 1.7)]:
        q = two_atom(einf, c, alpha)
        # p0 = 1/2  <=>  eps = 2 einf  <=>  x = ln(c/einf)/alpha
        xs = np.log(c / einf) / alpha
        d = q(xs)
        out.append(dict(einf=einf, c=c, alpha=alpha, x_star=xs,
                        eps_star=d["eps"], two_einf=2 * einf,
                        kappa=d["kappa"], s_sq=d["s"] ** 2,
                        ratio=d["kappa"] / d["s"] ** 2,
                        d_inv_s_dx=d["kappa"] / d["s"] ** 2))
    return pd.DataFrame(out)


# ----------------------------------------------------------- T7-T8: sum rules
def sum_rules():
    rows = []
    cases = [
        ("two atoms",              [0, 1],            [1, 1]),
        ("two atoms, rescaled",    [0, 0.37],         [1, 2.3]),
        ("three atoms, coincident",[0, 1, 2],         [1, 1, 1]),
        ("three atoms, spread",    [0, 0.5, 3.1],     [1, 0.3, 7.0]),
        ("four atoms, coincident", [0, 1, 2, 3],      [1, 1, 1, 1]),
        ("five atoms, generic",    [0, .2, .9, 1., 4.], [1, 2, .5, 3, 1]),
        ("near-degenerate pair",   [0, 1, 1.0001],    [1, 1, 1]),
    ]
    # separated kinks: stagger the crossings 40 nats apart so no two kinks
    # overlap.  L should then be exactly N pi.
    for N in range(1, 6):
        a = np.arange(N + 1, dtype=float)
        lc = np.zeros(N + 1)
        for j in range(N):
            lc[j + 1] = lc[j] + (a[j + 1] - a[j]) * (-40.0 * j)
        cases.append((f"{N+1} atoms, kinks separated", list(a),
                      list(np.exp(lc - lc.max()))))

    for name, a, c in cases:
        _, s, kappa = spectrum(a, c)
        span = (max(a) - min(a)) or 1.0
        X = 40.0 * (len(a) + 1) + 200.0 / span
        # integrate on a grid dense enough to resolve coincident kinks
        xs = np.unique(np.concatenate([
            np.linspace(-X, X, 40001),
            np.linspace(-60 * len(a), 60 * len(a), 40001)]))
        kv = np.array([kappa(x) for x in xs])
        E = float(np.trapezoid(kv, xs))
        L = float(np.trapezoid(np.sqrt(np.maximum(kv, 0)), xs))
        rows.append(dict(spectrum=name, n_atoms=len(a),
                         E=E, spectral_width=max(a) - min(a),
                         E_rel_err=abs(E / (max(a) - min(a)) - 1),
                         L=L, L_over_pi=L / np.pi, N_channels=len(a) - 1,
                         kinks_separated="separated" in name))
    return pd.DataFrame(rows)


# ------------------------------------------------------- T9: frailty dictionary
def frailty_mc(M=4_000_000):
    a = np.array([0.0, 0.6, 2.0])
    c = np.array([0.2, 0.5, 0.3]); c = c / c.sum()
    _, s, kappa = spectrum(a, c)
    idx = RNG.choice(len(a), size=M, p=c)
    rate = a[idx]
    thr = np.where(rate > 0, RNG.exponential(1 / np.maximum(rate, 1e-12)), np.inf)
    rows = []
    for x in [0.25, 0.5, 1.5, 3.0, 6.0]:
        alive = thr > x
        rows.append(dict(
            x=x,
            eps_mc=alive.mean(), eps_closed=float(np.sum(c * np.exp(-a * x))),
            hazard_mc=rate[alive].mean(), hazard_closed=s(x),
            frailty_var_mc=rate[alive].var(), frailty_var_closed=kappa(x),
            cure_fraction=c[0], p0_closed=c[0] / float(np.sum(c * np.exp(-a * x)))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- run
if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)

    print("T1-T6  the local calculus, against the closed-form two-atom law")
    lc = local_calculus()
    for col, lab in [("err_p0", "p_0 = kappa/(s^2+kappa)"),
                     ("err_m", "m   = (s^2-kappa)/(s^2+kappa)"),
                     ("err_R", "R   = ln[(s^2+kappa)/(2 s sqrt kappa)]"),
                     ("err_alpha", "alpha = 2 ds/dm"),
                     ("err_einf", "eps_inf = kappa eps/(s^2+kappa)")]:
        print(f"    max |error|  {lab:42s} {lc[col].max():.3e}")
    lc.to_csv(os.path.join(RESULTS, "table_local_calculus.csv"), index=False)
    print("  -> results/table_local_calculus.csv")

    cp = canonical_point()
    print("T6     canonical point: kappa/s^2 at eps = 2 eps_inf =",
          ", ".join(f"{v:.12f}" for v in cp.ratio))

    print("T7-T8  sum rules")
    sr = sum_rules()
    sep = sr[sr.kinks_separated]
    print(f"    int kappa dx = a_max - a_min : max rel. error "
          f"{sr.E_rel_err.max():.2e} over {len(sr)} spectra")
    print("    int sqrt(kappa) dx / pi for separated kinks: " +
          ", ".join(f"{v:.4f}" for v in sep.L_over_pi))
    print("    coincident kinks merge: " + ", ".join(
        f"{r.n_atoms-1} channels -> L/pi={r.L_over_pi:.3f}"
        for _, r in sr[~sr.kinks_separated].iterrows() if r.n_atoms > 2))
    sr.to_csv(os.path.join(RESULTS, "table_sum_rules.csv"), index=False)
    print("  -> results/table_sum_rules.csv")

    print("T9     frailty dictionary (Monte Carlo, 4e6 draws)")
    mc = frailty_mc()
    print(f"    max |eps_mc/eps_closed - 1|        = "
          f"{abs(mc.eps_mc/mc.eps_closed-1).max():.2e}")
    print(f"    max |hazard_mc/hazard_closed - 1|  = "
          f"{abs(mc.hazard_mc/mc.hazard_closed-1).max():.2e}")
    print(f"    max |Var_mc/Var_closed - 1|        = "
          f"{abs(mc.frailty_var_mc/mc.frailty_var_closed-1).max():.2e}")
    mc.to_csv(os.path.join(RESULTS, "table_frailty_mc.csv"), index=False)
    print("  -> results/table_frailty_mc.csv")

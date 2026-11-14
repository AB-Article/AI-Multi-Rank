#!/usr/bin/env python3
"""ONE INEQUALITY IS THE SECOND RUNG OF A LADDER.
   (Section "The floor is not a parameter", Thm. 'monofloor' and Prop. 'ladder';
   Online Appendix, the moment-problem proofs.)

Writing c_k for the cumulants of the frailty posterior P_x (density prop. to
e^{-ax} dmu), the derivatives of S = -ln eps are

    S^{(k)} = (-1)^{k+1} c_k           (L1, verified below)

so s = c_1, kappa = -S'' = c_2, S''' = mu_3 and S'''' = -(mu_4 - 3 mu_2^2).
The axiom is therefore EXACTLY the statement that the moment sequence of P_x is
a Stieltjes moment sequence, i.e. that all its Hankel determinants are
non-negative.  That gives three things at once.

  L2  Thm. 'monofloor' IS CAUCHY-SCHWARZ.  The shifted 2x2 Hankel condition is
      E[a] E[a^3] >= E[a^2]^2, i.e. Cauchy-Schwarz in the a-weighted inner
      product <f,g> = E[a f g]; dividing by s^2 kappa^2 turns it into

          d ln kappa / dS  <=  1 - kappa/s^2 ,

      and equality in Cauchy-Schwarz holds iff a is constant on {a>0}, i.e.
      iff the spectrum is {0, alpha}.  The truncated-moment-problem argument of
      the first proof is not needed, and the equality case is immediate.

  L3  THE LADDER.  Rung k is the k-th Hankel condition, needs S^{(k+1)}, and is
      TIGHT exactly on a k-atom spectrum:
          rung 1 :  kappa >= 0                         (1 atom)
          rung 2 :  w2 <= 1 - w                        (2 atoms, one at 0)
          rung 3 :  S''''/(s^2 kappa) <= 2w - w2^2     (any 2 atoms)
          rung k :  Hankel_k >= 0                      (k atoms)
      so model selection is a statement about one higher jet and the channel
      count is the RANK of the Hankel matrix of (s, kappa, S''', S'''', ...).

  L4  THE MIRROR BOUND.  The upper principal representation (an atom pinned at
      a_max) bounds mu_3 from above, so with rho = (a_max - s)/s,

          -rho + w/rho  <=  w2  <=  1 - w ,

      both walls tight exactly on two-atom spectra.  Two corollaries that are
      measurable from the local jet alone:
          rho >= w , i.e.  a_max >= s(1+w) = s + kappa/s   (the fastest mode is
          bounded below by local data), and, with the sum rule,
          int kappa dx = a_max - a_min >= sup_x (s + kappa/s) - a_min .

Outputs: results/table_ladder_identities.csv
         results/table_ladder_tightness.csv
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from mpmath import mp, mpf, exp as mexp

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _loaders import save

mp.dps = 50
RNG = np.random.default_rng(20261005)


def post(a, p, x):
    """exact moments and cumulants of the frailty posterior at x."""
    a = np.array([mpf(str(v)) for v in a]); p = np.array([mpf(str(v)) for v in p])
    w = p * np.array([mexp(-v * mpf(str(x))) for v in a]); w = w / sum(w)
    mom = [sum(w * a ** k) for k in range(0, 8)]
    m1 = mom[1]
    cen = [sum(w * (a - m1) ** k) for k in range(0, 8)]
    c = [None, mom[1], cen[2], cen[3], cen[4] - 3 * cen[2] ** 2,
         cen[5] - 10 * cen[3] * cen[2]]
    return mom, cen, c


def jet(a, p, x):
    mom, cen, c = post(a, p, x)
    s, kap, mu3, mu4 = c[1], c[2], cen[3], cen[4]
    if kap == 0:                                    # single atom: no curvature
        return dict(s=s, kap=kap, w=mpf(0), w2=mpf(0), V=mpf(0), mom=mom, cen=cen)
    return dict(s=s, kap=kap, w=kap / s ** 2, w2=-mu3 / (s * kap),
                V=(3 * kap ** 2 - mu4) / (s ** 2 * kap), mom=mom, cen=cen)


def S_fd(a, p, x, k):
    h = mpf('1e-6')
    def S(t):
        return -mp.log(sum(mpf(str(pi)) * mexp(-mpf(str(ai)) * t)
                           for ai, pi in zip(a, p)))
    C = {1: ([1, -1], [1, -1], 2), 2: ([1, -2, 1], [1, 0, -1], 1),
         3: ([1, -2, 2, -1], [2, 1, -1, -2], 2), 4: ([1, -4, 6, -4, 1], [2, 1, 0, -1, -2], 1)}
    cc, oo, d = C[k]
    return sum(ci * S(mpf(str(x)) + oi * h) for ci, oi in zip(cc, oo)) / (d * h ** k)


def spec(k, floor):
    a = np.sort(RNG.uniform(0.02, 3.0, k))
    if floor:
        a[0] = 0.0
    return a, RNG.dirichlet(np.ones(k))


def run():
    ids = []
    # --------------------------------------------------------- L1 ----------
    dev = 0.0
    for _ in range(12):
        a, p = spec(int(RNG.integers(2, 5)), bool(RNG.random() < .5))
        x = float(RNG.uniform(0.3, 4.0)); _, _, c = post(a, p, x)
        for k, sg in [(1, 1), (2, -1), (3, 1), (4, -1)]:
            dev = max(dev, abs(float((S_fd(a, p, x, k) - sg * c[k]) / c[k])))
    ids.append(dict(result="L1  S^(k) = (-1)^{k+1} c_k (cumulants of the frailty posterior)",
                    max_deviation=dev))
    # --------------------------------------------------------- L2 ----------
    d_cs, d_eq = 0.0, 0.0
    for _ in range(400):
        a, p = spec(2, True); x = float(RNG.uniform(0.05, 8.0))
        mom, cen, c = post(a, p, x); J = jet(a, p, x)
        lhs = float(mom[1] * mom[3] - mom[2] ** 2)          # Cauchy-Schwarz gap
        rhs = float((1 - J['w']) - J['w2'])                 # monofloor slack
        # the two are the same inequality: gap = s^2 kappa^2 * slack
        d_cs = max(d_cs, abs(lhs - float(J['s'] ** 2 * J['kap'] ** 2) * rhs))
        d_eq = max(d_eq, abs(lhs))
    ids.append(dict(result="L2  E[a]E[a^3]-E[a^2]^2 = s^2 kappa^2 [(1-w) - w2]  (identity)",
                    max_deviation=d_cs))
    ids.append(dict(result="L2  both vanish on {0,alpha}: max |E[a]E[a^3]-E[a^2]^2|",
                    max_deviation=d_eq))
    # --------------------------------------------------------- L4 ----------
    dmirror = 0.0
    for _ in range(400):
        a, p = spec(2, bool(RNG.random() < .5)); x = float(RNG.uniform(0.05, 8.0))
        J = jet(a, p, x); rho = (mpf(str(a[-1])) - J['s']) / J['s']
        dmirror = max(dmirror, abs(float(J['w2'] - (-rho + J['w'] / rho))))
    ids.append(dict(result="L4  mirror wall w2 = -rho + w/rho is TIGHT on every two-atom spectrum",
                    max_deviation=dmirror))
    save(pd.DataFrame(ids), "table_ladder_identities.csv")

    # ------------------------------------------------- tightness table -----
    rows = []
    fams = [("1 atom (pure power law)", 1, False), ("floor + 1 channel", 2, True),
            ("2 atoms, no floor", 2, False), ("floor + 2 channels", 3, True),
            ("3 atoms, no floor", 3, False), ("5 atoms, no floor", 5, False)]
    for name, k, fl in fams:
        r1, r2, r3, mir, amax, = [], [], [], [], []
        for _ in range(400):
            if k == 1:
                a = np.array([RNG.uniform(0.1, 3.0)]); p = np.array([1.0])
            else:
                a, p = spec(k, fl)
            x = float(RNG.uniform(0.05, 8.0)); J = jet(a, p, x)
            r1.append(float(J['kap']))
            if float(J['kap']) > 0:
                r2.append(float((1 - J['w']) - J['w2']))
                r3.append(float(2 * J['w'] - J['w2'] ** 2 - J['V']))
                rho = (mpf(str(a[-1])) - J['s']) / J['s']
                mir.append(float(J['w2'] - (-rho + J['w'] / rho)))
                amax.append(float(mpf(str(a[-1])) / (J['s'] * (1 + J['w']))))
        rows.append(dict(family=name, atoms=k, floor=fl,
                         rung1_min_kappa=min(r1),
                         rung2_min_slack=(min(r2) if r2 else np.nan),
                         rung3_min_slack=(min(r3) if r3 else np.nan),
                         mirror_min_slack=(min(mir) if mir else np.nan),
                         min_amax_over_s_1plusw=(min(amax) if amax else np.nan)))
    T = pd.DataFrame(rows)
    save(T, "table_ladder_tightness.csv")
    print(T.to_string(index=False))
    print()
    print(pd.DataFrame(ids).to_string(index=False))

    # ------------------------------------------- the sum-rule inequality ---
    srows = []
    for name, k, fl in [("floor + 1 channel", 2, True), ("floor + 2 channels", 3, True),
                        ("4 atoms, no floor", 4, False)]:
        ratio = []
        for _ in range(60):
            a, p = spec(k, fl)
            best = max(float(jet(a, p, x)['s'] * (1 + jet(a, p, x)['w']))
                       for x in np.exp(np.linspace(-4, 6, 40)))
            width = float(a[-1] - a[0])
            if best > float(a[0]):
                ratio.append(width / (best - float(a[0])))
        srows.append(dict(family=name, n=len(ratio), min_ratio=min(ratio),
                          median_ratio=float(np.median(ratio))))
    S = pd.DataFrame(srows)
    save(S, "table_sumrule_inequality.csv")
    print()
    print("sum-rule inequality  int kappa dx >= sup_x (s + kappa/s) - a_min  (ratio must be >= 1)")
    print(S.to_string(index=False))


if __name__ == "__main__":
    run()

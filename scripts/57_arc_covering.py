#!/usr/bin/env python3
"""THE COVERING LAW: HOW MANY CHECKPOINTS A FRONTIER NEEDS (Section 5).

Because the Fisher-Rao length of a complete single-channel transition is pi
regardless of the exponent, the floor, the resource or the units, so is the
answer to "how many operating points must a provider publish?".

Set phi_mirror = 2 arcsin sqrt(p_0) in (0,pi) as the internal greedy
coordinate.  NOTE (round 13): this is the MIRROR of the paper's arc
coordinate phi = 2 arccos sqrt(phat_0) = 2 arctan sqrt(N_eff), which runs to
0 at the floor.  The two are related by phi = pi - phi_mirror, and the regret
-ln sin is symmetric under it, so every covering number and band width below
is the same in either convention.  Only POSITIONS differ, so
table_arc_placement.csv now carries both columns and anything that plots a
position against a phi axis must use phi_paper.  The regret of
holding the checkpoint at phi-hat when the realized price selects phi is
KL(p_phi || p_phi-hat), and the 0.05-nat indifference band about the canonical
point phi = pi/2 is phi in [1.2610, 1.8806], a fixed 19.7% of every frontier.

  (1) BAND.        R(phi) = -ln sin phi; the R-nat band is
                   |phi - pi/2| <= arccos(...)-free closed form
                   phi in [arcsin e^{-R}, pi - arcsin e^{-R}].
  (2) COVERING.    The minimal K with max_phi min_k KL <= R, by exact greedy
                   covering.  K ~ L / (2 sqrt(2R)) = pi / (2 sqrt(2R)).
  (3) PLACEMENT.   The optimal checkpoints are equally spaced in phi, NOT in
                   log-resource; the table gives eps/einf and zeta for each.
  (4) BUDGET.      The log-resource span needed to cover a fraction f of the
                   arc is (2/alpha) ln[ tan(pi(1+f)/4) / tan(pi(1-f)/4) ],
                   which diverges as f -> 1: the floor is at finite arc
                   distance but infinite resource distance.

Writes results/table_arc_covering.csv and results/table_arc_placement.csv.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")


def p_of(phi):
    return np.array([np.sin(phi / 2) ** 2, np.cos(phi / 2) ** 2])


def KL(phi, phih):
    p = np.clip(p_of(phi), 1e-300, None)
    q = np.clip(p_of(phih), 1e-300, None)
    return float(np.sum(p * np.log(p / q)))


def regret_from_canonical(phi):
    return -np.log(np.sin(phi))


def greedy(R, lo, hi):
    """Minimal covering of [lo,hi] by KL-balls of radius R, left to right."""
    pts, cur = [], lo
    while cur < hi - 1e-12:
        f = lambda ph: KL(cur, ph) - R
        ph = brentq(f, cur, hi) if f(hi) > 0 else hi
        pts.append(ph)
        g = lambda q: KL(q, ph) - R
        nxt = brentq(g, ph, hi) if g(hi) > 0 else hi
        if nxt <= cur + 1e-12:
            break
        cur = nxt
    return pts


rows = []
for R in (0.02, 0.05, 0.10, 0.20, 0.50):
    lo_band = np.arcsin(min(np.exp(-R), 1.0))
    full = greedy(R, 1e-6, np.pi - 1e-6)
    mid = greedy(R, 0.05 * np.pi, 0.95 * np.pi)
    rows.append(dict(R_nats=R,
                     band_lo=lo_band, band_hi=np.pi - lo_band,
                     band_frac_of_arc=(np.pi - 2 * lo_band) / np.pi,
                     eps_over_einf_lo=1.0 / np.sin((np.pi - lo_band) / 2) ** 2,
                     eps_over_einf_hi=1.0 / np.sin(lo_band / 2) ** 2,
                     K_full_arc=len(full), K_central_90=len(mid),
                     K_sqrt_rule=np.pi / (2 * np.sqrt(2 * R))))
C = pd.DataFrame(rows)
C.to_csv(os.path.join(RESULTS, "table_arc_covering.csv"), index=False)
print("COVERING NUMBERS")
print(C[["R_nats", "band_frac_of_arc", "K_full_arc", "K_central_90",
         "K_sqrt_rule"]].to_string(index=False,
       float_format=lambda v: f"{v:.3f}"))

pts = greedy(0.05, 1e-6, np.pi - 1e-6)
pl = []
for k, ph in enumerate(pts, 1):
    p0 = np.sin(ph / 2) ** 2
    pl.append(dict(k=k, phi=ph, phi_paper=np.pi - ph, p0=p0,
                   eps_over_einf=1.0 / p0,
                   zeta=np.log(p0 / (1 - p0)),
                   m=np.cos(ph), regret_from_canonical=regret_from_canonical(ph)))
P = pd.DataFrame(pl)
P.to_csv(os.path.join(RESULTS, "table_arc_placement.csv"), index=False)
print(f"\nOPTIMAL PLACEMENT at R = 0.05 nats (K = {len(pts)}); "
      "x_k = zeta_k/alpha + const")
print(P.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\nRESOURCE BUDGET to cover a fraction f of the arc (units of 1/alpha)")
bud = []
for f in (0.25, 0.5, 0.8, 0.9, 0.95, 0.99):
    X = 2 * np.log(np.tan(np.pi * (1 + f) / 4) / np.tan(np.pi * (1 - f) / 4))
    bud.append(dict(arc_fraction=f, span_times_alpha=X,
                    efolds_at_alpha_0p42=X / 0.42, decades_at_alpha_0p42=X / 0.42 / np.log(10)))
B = pd.DataFrame(bud)
print(B.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
B.to_csv(os.path.join(RESULTS, "table_arc_budget.csv"), index=False)
print("\nwrote results/table_arc_covering.csv, table_arc_placement.csv, "
      "table_arc_budget.csv")

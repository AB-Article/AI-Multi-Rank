#!/usr/bin/env python3
"""SELF-DUALITY OF THE CANONICAL OPERATING POINT.

Verifies, symbolically and at 40 digits, the statements added in the revision
of 2026-10-04, and checks the two that touch data on the archived pair tables.

  (1) SELF-DUALITY (Theorem "Self-duality", Section 5 of paper.tex).  With one
      reducible channel, eps = einf + c exp(-alpha x) and r = eps - einf, the
      reflection x -> 2 x0 - x induces r -> r0^2 / r with r0 = r(x0).  The
      scalarization sigma_lambda = -ln(einf + c e^{-alpha x}) - lambda x is
      invariant under it IFF lambda = alpha/2 AND r0 = einf.  Both halves are
      forced: the x -> oo limit of d/dx[sigma(2x0-x) - sigma(x)] is 2 lambda -
      alpha, and at lambda = alpha/2 the remainder vanishes for all x only at
      c = einf e^{alpha x0}.

  (2) EVENNESS AND "NEAREST IS EXACT" (eq. nearestexact).  With
      zeta = ln(einf/r), J = const - ln cosh(zeta/2) is even, so
      argmax J = argmin |zeta| = argmin |m| identically.  Checked symbolically
      and by random candidate sets.  The same identity in ln(eps) is false,
      and the disagreement rate is draw-dependent -- reported here for two
      draws to show that it is a property of the draw, not of the theory.

  (3) PSI-EVENNESS OF THE MISQUOTED-FLOOR REGRET (eq. misquotepsi).
      R(phi) = ln cosh(1/2 ln(2 phi - 1)) is exactly even under psi -> 1/psi
      with psi = 2 phi - 1, so R(0.7) = R(1.75).

  (4) AITKEN = THE DUALITY RELATION (eq. aitken).  For a mirror triple
      x0 +- h on the law, r_- r_+ = r0^2 identically; solved for the floor it
      is Aitken's Delta^2.

  (5) THE m-FORM OF THE TWO-CANDIDATE TEST (eq. mmid).  m = 1 - 2 einf/eps;
      m_1 + m_2 > 0 is algebraically identical to the harmonic test, and
      unlike zeta it is defined when eps <= einf.  Checked on all archived
      pairs, including those where zeta does not exist.

  (6) MIRROR-TEST FEASIBILITY (Section 5, "A symmetry test").  Counts, per
      single-resource cell with an identified floor, how many candidates have
      their reflection about the knee inside the observed window and how far
      the nearest measured candidate is from that reflection.

Writes results/table_duality.csv.  Items (1)-(4) read no data.

Run:  python3 53_duality.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import sympy as sp
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FRONTIERS = os.path.join(RESULTS, "frontiers")
mp.mp.dps = 40

rows = []


def record(block, statement, value, target, ok, note=""):
    rows.append(dict(block=block, statement=statement, value=value,
                     target=target, ok=bool(ok), note=note))
    flag = "ok " if ok else "FAIL"
    print(f"  [{flag}] {statement}: {value}")


# ----------------------------------------------------------------- (1)
print("\n(1) SELF-DUALITY, symbolic")
x, x0 = sp.symbols("x x0", real=True)
al, c, ei = sp.symbols("alpha c epsinf", positive=True)
lam = sp.Symbol("lam", positive=True)

sigma = lambda X: -sp.log(ei + c * sp.exp(-al * X)) - lam * X
D = sp.simplify(sigma(2 * x0 - x) - sigma(x))

lim = sp.simplify(sp.limit(sp.diff(D, x), x, sp.oo))
record("selfduality", "lim_{x->oo} d/dx[sigma(2x0-x)-sigma(x)]", str(lim),
       "2*lam - alpha", sp.simplify(lim - (2 * lam - al)) == 0,
       "forces lambda = alpha/2")

D_half = sp.simplify(D.subs(lam, al / 2))
sols = sp.solve(sp.Eq(sp.simplify(sp.exp(D_half)), 1), c, dict=True)
got = sp.simplify(sols[0][c]) if len(sols) == 1 else None
record("selfduality", "invariance at lambda=alpha/2 holds iff c =", str(got),
       "epsinf*exp(alpha*x0)", got is not None and
       sp.simplify(got - ei * sp.exp(al * x0)) == 0,
       "i.e. r(x0) = epsinf")

resid = sp.simplify(D_half.subs(c, ei * sp.exp(al * x0)))
record("selfduality", "sigma(2x0-x) - sigma(x) at lambda=alpha/2, r0=epsinf",
       str(resid), "0", sp.simplify(resid) == 0)

inv = sp.simplify(c * sp.exp(-al * x) * c * sp.exp(-al * (2 * x0 - x))
                  - (c * sp.exp(-al * x0)) ** 2)
record("selfduality", "r(x) r(2x0-x) - r(x0)^2", str(inv), "0", inv == 0)

# ----------------------------------------------------------------- (2)
print("\n(2) J EVEN IN zeta; nearest is exact")
z = sp.Symbol("zeta", real=True)
eps_z = ei * (1 + sp.exp(-z))
J = sp.Rational(1, 2) * sp.log(eps_z - ei) - sp.log(eps_z)
even = sp.simplify(sp.expand_log(J.subs(z, -z) - J, force=True))
record("evenness", "J(-zeta) - J(zeta)", str(even), "0", even == 0)

fJ = sp.lambdify((z, ei), J + sp.log(sp.cosh(z / 2)) + sp.log(4 * ei) / 2, "mpmath")
dev = max(abs(fJ(mp.mpf(a), mp.mpf(b)))
          for a in ("-3.7", "-0.4", "0", "1.3", "5.1") for b in ("0.01", "0.07", "0.3"))
record("evenness", "max |J + ln cosh(zeta/2) + ln(4 einf)/2| at 40 digits",
       mp.nstr(dev, 5), "0", dev < mp.mpf("1e-35"),
       "J = const - ln cosh(zeta/2)")

rng = np.random.default_rng(20261004)
def draw(n_sets, sd):
    bad_z = bad_log = 0
    worst = 0.0
    for _ in range(n_sets):
        e0 = rng.uniform(0.01, 0.30)
        k = int(rng.integers(2, 9))
        e = e0 * (1 + np.exp(rng.normal(0, sd, k)))
        j = 0.5 * np.log(e - e0) - np.log(e)
        zz = np.log(e0 / (e - e0))
        iJ = int(np.argmax(j))
        if iJ != int(np.argmin(np.abs(zz))):
            bad_z += 1
        iL = int(np.argmin(np.abs(np.log(e / (2 * e0)))))
        if iJ != iL:
            bad_log += 1
            worst = max(worst, j[iJ] - j[iL])
    return bad_z, bad_log, worst, n_sets

for sd, tag in ((1.0, "sd=1.0"), (2.0, "sd=2.0")):
    bz, bl, w, n = draw(200_000, sd)
    record("evenness", f"argmax J != argmin|zeta|, {tag}", f"{bz}/{n}", "0", bz == 0)
    record("evenness", f"argmax J != nearest in ln(eps), {tag}",
           f"{bl}/{n} = {100*bl/n:.2f}%, worst loss {w:.3f} nats", "draw-dependent", True,
           "not a property of the theory; no figure quoted in the paper")

# ----------------------------------------------------------------- (3)
print("\n(3) PSI-EVENNESS OF THE MISQUOTED-FLOOR REGRET")
ph, psi = sp.symbols("phi psi", positive=True)
Rpsi = sp.simplify(sp.log(sp.cosh(sp.log(2 * ph - 1) / 2)).subs(ph, (1 + psi) / 2))
record("psi", "R in psi", str(Rpsi), "log(cosh(log(psi)/2))",
       sp.simplify(Rpsi - sp.log(sp.cosh(sp.log(psi) / 2))) == 0)
d = sp.simplify(Rpsi.subs(psi, 1 / psi) - Rpsi)
record("psi", "R(1/psi) - R(psi)", str(d), "0", d == 0)
Rn = lambda v: mp.log(mp.cosh(mp.log(2 * mp.mpf(v) - 1) / 2))
gap = abs(Rn("0.7") - Rn("1.75"))
record("psi", "R(phi=0.7) - R(phi=1.75)", mp.nstr(gap, 5), "0",
       gap < mp.mpf("1e-35"), f"both = {mp.nstr(Rn('0.7'), 8)} nats")

# ----------------------------------------------------------------- (4)
print("\n(4) AITKEN = THE DUALITY RELATION")
em, e0s, ep = sp.symbols("e_minus e_0 e_plus", positive=True)
sol = sp.solve(sp.Eq((em - ei) * (ep - ei), (e0s - ei) ** 2), ei)
aitken = (em * ep - e0s ** 2) / (em + ep - 2 * e0s)
record("aitken", "solve r_- r_+ = r_0^2 for the floor", str(sp.simplify(sol[0])),
       "Aitken Delta^2", sp.simplify(sol[0] - aitken) == 0)

mp.mp.dps = 40
einf_t, c_t, al_t, x0_t, h_t = (mp.mpf("0.0731"), mp.mpf("2.17"),
                                mp.mpf("0.43"), mp.mpf("3.1"), mp.mpf("1.37"))
law = lambda X: einf_t + c_t * mp.e ** (-al_t * X)
Em, E0, Ep = law(x0_t - h_t), law(x0_t), law(x0_t + h_t)
err = abs((Em * Ep - E0 ** 2) / (Em + Ep - 2 * E0) - einf_t)
record("aitken", "40-digit recovery of the floor from a mirror triple",
       mp.nstr(err, 5), "0", err < mp.mpf("1e-35"))

# ----------------------------------------------------------------- (5)
print("\n(5) THE m-FORM ON THE ARCHIVED PAIRS")
ac = pd.read_csv(os.path.join(RESULTS, "table_acconly_pairs.csv"))
f, e1, e2 = ac.eps_inf.values, ac.eps_hi.values, ac.eps_lo.values
m1, m2 = 1 - 2 * f / e1, 1 - 2 * f / e2
harm = (1 / f) > (1 / e1 + 1 / e2)
mrule = (m1 + m2) > 0
record("mform", "m-rule vs harmonic test, all archived pairs",
       f"{int((mrule != harm).sum())} disagreements in {len(ac)}", "0",
       (mrule == harm).all())
record("mform", "m-rule vs the archived rule_take_accurate column",
       f"{int((mrule != ac.rule_take_accurate.values).sum())} disagreements", "0",
       (mrule == ac.rule_take_accurate.values).all())
undef = (e2 <= f) | (e1 <= f)
record("mform", "pairs on which zeta is undefined but m is not",
       f"{int(undef.sum())}/{len(ac)} = {100*undef.mean():.1f}%", "reported",
       True, f"{ac[undef].groupby(['frontier','cost']).ngroups} of "
             f"{ac.groupby(['frontier','cost']).ngroups} cells")
record("mform", "median pair midpoint (m1+m2)/2, 54-cell panel",
       f"{np.median((m1 + m2) / 2):.4f}", "reported", True)

ex = pd.read_csv(os.path.join(RESULTS, "table_exchange_rate_pairs.csv"))
g, E1, E2 = ex.eps_inf_fit.values, ex.eps_hi.values, ex.eps_lo.values
M = ((1 - 2 * g / E1) + (1 - 2 * g / E2)) / 2
Z = (np.log(g / (E1 - g)) + np.log(g / (E2 - g))) / 2
record("mform", "median pair midpoint (m1+m2)/2, 41-cell panel",
       f"{np.median(M):.4f}", "reported", True,
       f"median zeta-midpoint {np.median(Z):.4f}")

# ----------------------------------------------------------------- (6)
print("\n(6) MIRROR-TEST FEASIBILITY")
feas = []
for cell, grp in ex.groupby("cell"):
    fn, col = cell.split("|")
    path = os.path.join(FRONTIERS, fn + ".csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    if col not in df.columns:
        continue
    einf, alpha = grp.eps_inf_fit.iloc[0], grp.alpha_fit.iloc[0]
    xs = np.log(df[col].values.astype(float))
    es = df["eps"].values.astype(float)
    keep = np.isfinite(xs) & (es > einf)
    xs, es = xs[keep], es[keep]
    if len(xs) < 4 or alpha <= 0:
        continue
    lnc = float(np.mean(np.log(es - einf) + alpha * xs))
    knee = (lnc - np.log(einf)) / alpha
    mirror = 2 * knee - xs
    inside = (mirror >= xs.min()) & (mirror <= xs.max())
    gaps = np.min(np.abs(mirror[:, None] - xs[None, :]), axis=1)
    feas.append(dict(cell=cell, n=len(xs), knee_in_window=bool(xs.min() <= knee <= xs.max()),
                     frac_mirror_inside=float(inside.mean()),
                     med_gap_nats=float(np.median(gaps[inside])) if inside.any() else np.nan,
                     med_spacing_nats=float(np.median(np.diff(np.sort(xs))))))
F = pd.DataFrame(feas)
record("mirror", "cells with the knee inside the observed window",
       f"{int(F.knee_in_window.sum())}/{len(F)}", "reported", True)
record("mirror", "median fraction of candidates whose mirror is in-window",
       f"{F.frac_mirror_inside.median():.2f}", "reported", True)
record("mirror", "median gap to the nearest measured mirror partner",
       f"{F.med_gap_nats.median():.3f} nats", "reported", True,
       f"median point spacing {F.med_spacing_nats.median():.3f} nats")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS, "table_duality.csv"), index=False)
F.to_csv(os.path.join(RESULTS, "table_duality_mirror_feasibility.csv"), index=False)
print(f"\nwrote results/table_duality.csv ({len(out)} statements, "
      f"{int(out.ok.sum())} ok, {int((~out.ok).sum())} failed)")
print("wrote results/table_duality_mirror_feasibility.csv")

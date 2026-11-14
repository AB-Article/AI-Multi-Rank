#!/usr/bin/env python3
"""
128_power_audit.py -- simulation on KNOWN charts only. No corpus, no data.

Design-matched throughout: every test is scored BOTH on an indivisible truth
(power) and on the extremal divisible truth (level).  Sign-of-plug-in decisions
are reported next to studentised ones, because the extremal divisible truth sits
exactly ON the boundary and a sign reading therefore has level 1/2 there.

P.  COMPUTE AXIS.  The archive's four-point divided-difference test (proper
    unequal-spacing weights, z = dd3 / (sigma*||w||_2)) as a function of where
    the window sits relative to the elbow.  Question: is the archive's "no
    power, needs an 18-22x noise reduction" a property of the TEST or of the
    PLACEMENT of its windows?

Q.  COUNT AXIS.  Is the free datum P(0)=1 worth a rung?  Rung 1 and rung 2 of
    the price ladder at offset 0 (P(1..3), P(1..5)) against the archive's
    offset 1 (P(1..4), P(1..6)) -- studentised by parametric bootstrap.
"""
import csv
import math
import os

import numpy as np

rng = np.random.default_rng(20261006)
Z95 = 1.6448536269514722
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RES, exist_ok=True)
ROWS_P, ROWS_Q = [], []


def _save(rows, header, name):
    with open(os.path.join(RES, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  -> results/{name}")

A, EPS_INF, C0 = 0.9, 0.004, 0.996
X_ELBOW = math.log(C0 / EPS_INF) / A          # eps = 2 eps_inf


def eps_two_atom(x):
    return EPS_INF + C0 * np.exp(-A * x)


def eps_cure(x, lam=5.0, c=0.7):
    return np.exp(-lam * (1 - np.exp(-c * x)))


def eps_powerfloor(x, Lam=4.0, C=2.0, beta=-0.5):
    return np.exp(-Lam + C * x ** beta)


def dd_weights(xs):
    """exact divided-difference weights, any spacing: dd_k = sum_i w_i f(x_i)."""
    xs = np.asarray(xs, float)
    return np.array([1.0 / np.prod([xi - xj for j, xj in enumerate(xs) if j != i])
                     for i, xi in enumerate(xs)])


def refute_rate(chart, x0, span, M, sigma, reps=20000, widest=True):
    """archive's test: S Bernstein => [x0..x3]S >= 0 ; refute if z < -z95."""
    x = x0 + np.linspace(0.0, span, M)
    S0 = -np.log(chart(x))
    idx = [0, (M - 1) // 3, (2 * (M - 1)) // 3, M - 1] if widest else [0, 1, 2, 3]
    w = dd_weights(x[idx])
    noise = sigma * rng.standard_normal((reps, 4))
    dd = (S0[idx] + noise) @ w
    z = dd / (sigma * np.linalg.norm(w))
    return float(np.mean(z < -Z95))


def sigma_for_power(chart, x0, span, M, target=0.80):
    lo, hi = 1e-6, 2.0
    for _ in range(45):
        mid = math.sqrt(lo * hi)
        if refute_rate(chart, x0, span, M, mid, reps=6000) >= target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


print("=" * 80)
print("P.  COMPUTE AXIS -- the archive's own four-point test, placed correctly")
print(f"    two-atom truth, alpha={A}, eps_inf={EPS_INF}, elbow at x = {X_ELBOW:.3f}")
print("    (proper divided-difference weights; z = dd3/(sigma*||w||_2); one-sided 95%)")
print("=" * 80)
print(f"\n  {'window right edge':>18} {'span':>5} {'M':>3} | {'power':>7} {'level(cure)':>11}"
      f" {'level(pflr)':>11} | {'sigma_80 (milli-nats)':>21}")
for edge, span, M in [(X_ELBOW, 5.0, 10), (X_ELBOW, 5.0, 7), (X_ELBOW, 3.0, 10),
                      (X_ELBOW, 1.5, 10), (X_ELBOW - 2.0, 5.0, 10),
                      (X_ELBOW + 2.0, 5.0, 10), (X_ELBOW + 4.0, 5.0, 10),
                      (X_ELBOW + 6.0, 5.0, 10)]:
    x0 = edge - span
    pw = refute_rate(eps_two_atom, x0, span, M, 0.020)
    lc = refute_rate(eps_cure, x0, span, M, 0.020)
    lp = refute_rate(eps_powerfloor, max(x0, 0.05), span, M, 0.020)
    s80 = sigma_for_power(eps_two_atom, x0, span, M)
    tag = "elbow" if abs(edge - X_ELBOW) < 1e-9 else f"elbow{edge-X_ELBOW:+.0f}"
    print(f"  {tag:>18} {span:>5.1f} {M:>3} | {pw:>7.3f} {lc:>11.3f} {lp:>11.3f} |"
          f" {s80*1000:>21.2f}")
    ROWS_P.append([tag, span, M, round(pw, 4), round(lc, 4), round(lp, 4), round(s80 * 1000, 3)])

print("\n  narrow consecutive quadruples instead of the widest one (same windows):")
for edge, span, M in [(X_ELBOW, 5.0, 10), (X_ELBOW + 4.0, 5.0, 10)]:
    x0 = edge - span
    print(f"    right edge {'elbow' if edge==X_ELBOW else 'elbow+4'}: "
          f"power {refute_rate(eps_two_atom, x0, span, M, 0.020, widest=False):.3f}"
          f"   sigma_80 = {sigma_for_power(eps_two_atom, x0, span, M)*1000:.2f} milli-nats"
          f"  (widest: {sigma_for_power(eps_two_atom, x0, span, M)*1000:.2f})")

print("\n  TELESCOPED two-ended form (sum of every third difference at spacing h")
print("  = D2 log-eps at the right end minus at the left end), same data:")
for edge, span, M in [(X_ELBOW, 5.0, 10), (X_ELBOW, 5.0, 7)]:
    x0 = edge - span
    x = x0 + np.linspace(0.0, span, M)
    l0 = np.log(eps_two_atom(x))
    L = l0 + 0.020 * rng.standard_normal((20000, M))
    g = (L[:, M - 3] - 2 * L[:, M - 2] + L[:, M - 1]) - (L[:, 0] - 2 * L[:, 1] + L[:, 2])
    print(f"    span {span}, M={M}: power {np.mean(g/(0.020*math.sqrt(12)) > Z95):.3f}"
          f"   (the identity is exact; the ARRANGEMENT is weaker -- recorded)")

print()
print("=" * 80)
print("Q.  COUNT AXIS -- is P(0)=1 worth a rung?   (studentised, parametric bootstrap)")
print("=" * 80)

KMAX = 6


def mix(ws, gs, kmax=KMAX):
    ws, gs = np.asarray(ws, float), np.asarray(gs, float)
    ws = ws / ws.sum()
    p = np.array([np.sum(ws * gs ** f * (1 - gs)) for f in range(kmax)]
                 + [np.sum(ws * gs ** kmax)])
    return p / p.sum()


def poisson_mix(lam=4.0, c=0.6, npois=80):
    k = np.arange(npois)
    w = np.exp(-lam) * lam ** k / np.array([float(math.factorial(int(i))) for i in k])
    return mix(w, np.exp(-c * k))


TRUTHS = [
    ("indivisible: floor+one class (w=.05,g=.45)", mix([0.05, 0.95], [1.0, 0.45]), False),
    ("indivisible: 3-class census (.02/.55/.43)", mix([0.02, 0.55, 0.43], [1.0, 0.17, 0.95]), False),
    ("EXTREMAL divisible: compound Poisson", poisson_mix(), True),
    ("interior divisible: Lomax k=2 th=1.5", None, True),
]


def lomax_counts(n, size, th=1.5, k=2.0, kmax=KMAX):
    """P(m) = (1+m/th)^-k ; build the F-distribution directly."""
    P = np.array([(1 + m / th) ** (-k) for m in range(kmax + 1)])
    p = np.concatenate([P[:-1] - P[1:], [P[-1]]])
    return rng.multinomial(n, p / p.sum(), size=size).astype(float)


def stats_from_counts(counts, n):
    tail = np.cumsum(counts[:, ::-1], axis=1)[:, ::-1] / n
    P = np.concatenate([np.ones((counts.shape[0], 1)), tail[:, 1:KMAX + 1]], axis=1)
    P = np.clip(P, 1e-15, None)
    L = np.log(P[:, :-1] / P[:, 1:])

    def det3(off):
        a, b, c_, d, e = (L[:, off + i] for i in range(5))
        return a * (c_ * e - d * d) - b * (b * e - d * c_) + c_ * (b * d - c_ * c_)
    return (L[:, 0] * L[:, 2] - L[:, 1] ** 2,
            L[:, 1] * L[:, 3] - L[:, 2] ** 2,
            det3(0), det3(1))


def audit(p, n, reps=600, B=400, lomax=False):
    """returns, per statistic: sign-rate and studentised (bootstrap) rate."""
    counts = lomax_counts(n, reps) if lomax else rng.multinomial(n, p, size=reps).astype(float)
    obs = stats_from_counts(counts, n)
    sign = [float(np.mean(o < 0)) for o in obs]
    stud = [0.0] * 4
    hits = np.zeros((4, reps))
    for r in range(reps):
        phat = counts[r] / n
        phat = np.clip(phat, 1e-12, None); phat /= phat.sum()
        bc = rng.multinomial(n, phat, size=B).astype(float)
        bs = stats_from_counts(bc, n)
        for j in range(4):
            se = np.std(bs[j])
            hits[j, r] = 1.0 if (se > 0 and obs[j][r] + Z95 * se < 0) else 0.0
    stud = [float(hits[j].mean()) for j in range(4)]
    return sign, stud


print(f"\n  {KMAX} attempts per item.  'sign' = plug-in minor < 0 (the archive's reading);")
print("  'stud' = one-sided 95% bootstrap upper bound < 0.")
print(f"\n  {'truth':<42} {'n':>9} | {'r1@0':>13} {'r1@1':>13} {'r2@0':>13} {'r2@1':>13}")
print(f"  {'':<42} {'':>9} | {'sign  stud':>13} {'sign  stud':>13} {'sign  stud':>13} {'sign  stud':>13}")
for name, p, div in TRUTHS:
    for n in (10_000, 100_000, 1_000_000):
        sign, stud = audit(p, n, lomax=(p is None))
        row = "  ".join(f"{sign[j]:>5.2f} {stud[j]:>5.2f}" for j in range(4))
        print(f"  {name:<42} {n:>9} | {row}" + ("   <- LEVEL" if div else ""))
        ROWS_Q.append([name, int(div), n] + [round(v, 4) for v in sign]
                      + [round(v, 4) for v in stud])
print("\n  a column is usable only where both divisible rows are <= 0.05 under 'stud'.")
_save(ROWS_P, ["window_right_edge", "span_efolds", "n_models", "power", "level_cure",
               "level_powerfloor", "sigma80_millinats"], "table_compute_power.csv")
_save(ROWS_Q, ["truth", "divisible", "n_items", "sign_r1_off0", "sign_r1_off1",
               "sign_r2_off0", "sign_r2_off1", "stud_r1_off0", "stud_r1_off1",
               "stud_r2_off0", "stud_r2_off1"], "table_count_rung.csv")

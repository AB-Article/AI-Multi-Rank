"""pipeline_core -- the reading layer of the theory, hardened for a production
prediction pipeline.

Stdlib only, offline, no fit, no kernel, no bandwidth, no smoother, no root-find.
Everything is read off one curve,

    N_eff = s^2 / kappa     against     S = -ln eps ,      l := ln eps
    N_eff = (l[+] - l[-])^2 / (4 (l[-] - 2 l[0] + l[+]))          (h cancels)

This module implements the twenty-four fixes of the practical-pipeline audit
(2026-10-08, extended through revision m).  Each is tagged F1..F24 at its
definition and in ``PRACTICAL_PIPELINE.md``; F1..F17 are summarised below and
F18..F24 -- price-free domination, the certified tiers, the selection-grade
price, the break-even ladder, the price envelope, measurement-only Level 0 and
the input-consistency gate -- are documented at their definitions.

  F1  the grain t is a mandatory declaration; it is never defaulted silently
  F2  class gate: Law I is tested before anything is priced; out of class the
      selector refuses to name a winner
  F3  precision gate: binomial test-set noise is propagated into eps, s, kappa,
      N_eff and into every gap, and the tie band is widened to the measured
      resolution; the n_test a decision needs is reported
  F4  dual objective: the accuracy-only canonical value and the resource-aware
      scalarization are always both reported, and their disagreement is priced
  F5  multiresource: one price per axis, derived where that axis has a
      calibration curve and declared where it has not
  F6  count axis: given a pass@k / co-failure curve, N_eff is read there, where
      the test is exact and signed, and the floor is bracketed by two measured
      numbers instead of extrapolated
  F7  under-sizing: 1/N^x = 1/N^k + s_m/s, so a compute-axis reading is a lower
      bound; the corrected target is N^x = 1/(t + s_m/s)
  F8  a tie SET with a resource-aware tiebreak, or UNRESOLVED.  Retained, but
      it is no longer the verdict: see F12, the certified set.  The band is
      printed as ``BAND (v2)`` for comparison
  F9  candidates need not lie on the calibration frontier: the price comes from
      the frontier, the score from the candidate
  F10 the retry discount D(t) is measured on the count axis and converts a retry
      budget into the grain it actually delivers

Revision 2026-10-08d adds the seven fixes the audit of the audit found to be
critical, tagged F11..F17.  F11-F12 are the load-bearing pair: they replace an
uncalibrated indifference band with a SET THAT CARRIES A COVERAGE GUARANTEE,
computed from per-item outcomes and valid with no appeal to Law I.

  F11 paired resolution: candidates are scored on the SAME items, so the gap
      between two of them has variance
          V1/n,  V1 = (1-e_i)/e_i + (1-e_j)/e_j - 2 (p_ij - e_i e_j)/(e_i e_j),
      not the sum of two independent variances.  The archive's own co-failure
      panel measures the covariance term, and dropping it inflates both the
      band and the item count a decision needs.
  F12 certified set: candidate j is excluded only when some i is CERTIFIED to
      beat it at every price the frontier supports, by a one-sided
      distribution-free finite-sample bound (empirical Bernstein) on the mean
      of a bounded per-item variable, at level alpha/(K-1).  The surviving set
      then contains the price-optimal model with probability at least 1-alpha,
      by a union bound -- and the certificate uses NO convexity, no N_eff, no
      floor, so it still returns an answer where the class gate refuses.
  F13 prospective design: the grid width a calibration curve must span for
      kappa to be measurable at all is computed BEFORE the runs are spent,
      from the items, the accuracy, the expected slope and the grain.
  F14 instance-level stopping rule for the number of networks: add candidates
      until the worst-case regret over the supported price range falls below
      rho_2, measured on the candidates in hand, not on a corpus median.
  F15 next-candidate placement: the resource to evaluate next is the one that
      most reduces that worst case -- a sequential minimax design, which
      measures better than both equal-arc and equal-log-resource placement.
  F16 the O(h^2) slope bias is CORRECTED, not annotated: the plane is read at
      step h and at step 2h and combined by Richardson, (4 A_h - A_2h)/3.
  F17 nothing degrades silently: a missing n_test, a missing per-item file, a
      missing grain or an unidentified price each downgrade the verdict and are
      named in the output.
"""
from __future__ import annotations

import math
import random
import statistics

# ---------------------------------------------------------------- constants

#: one doubling of psi = (eps - eps_inf)/eps_inf, ln cosh(0.5 ln 2).
#: the paper's own indifference band and the tolerance behind "five checkpoints".
RHO2 = math.log(math.cosh(0.5 * math.log(2.0)))          # 0.0588915...
#: pi / (2 sqrt(2 RHO2)) = 4.577 -> five checkpoints.
CHECKPOINTS = math.pi / (2.0 * math.sqrt(2.0 * RHO2))
#: z for a two-sided 95% statement.
Z95 = 1.959963984540054
#: share of violating wide triples above which the object is called a Pareto
#: ENVELOPE rather than one frontier.  THIS IS THE ORIGINAL GUESS and is used
#: only as the fallback when ``n_test`` is absent and no per-instance threshold
#: can be computed.  ``scripts/150_envelope_threshold.py`` calibrated the
#: corpus-wide value at 0.183 (``table_envelope_threshold_summary.csv``,
#: ``median_null_q95``) and found 0.5 far too permissive; ``select_model.py``
#: computes the per-instance threshold from the frontier's own convex minorant
#: and noise whenever it can.
ENVELOPE_SHARE = 0.5


# ---------------------------------------------------------------- utilities

def ln_eps(accuracy: float, q0: float = 0.0) -> float:
    """l = ln eps with eps = (1 - Q)/(1 - q0).  q0 shifts l by a constant, so it
    cancels from every derivative and from every difference of sigma."""
    eps = (1.0 - accuracy) / (1.0 - q0)
    if not (0.0 < eps <= 1.0):
        raise ValueError(f"accuracy {accuracy} outside (q0, 1] for q0={q0}")
    return math.log(eps)


def eps_of(accuracy: float, q0: float = 0.0) -> float:
    return (1.0 - accuracy) / (1.0 - q0)


def _triples(xs, wide_frac: float = 0.667, min_span_frac: float = 0.10):
    """Index triples (i, j, k) with i < j < k, arms balanced to within
    ``wide_frac`` and each arm at least ``min_span_frac`` of the window.

    Read the axiom WIDE: the convexity defect is O(h^2) and the noise is not.
    The absolute floor on the arm matters as much as the balance: a Pareto
    envelope of a heterogeneous zoo contains near-duplicate resources, and a
    divided difference over a vanishing arm is noise divided by nothing.  On
    the archived corpus dropping that guard turns a median defect of -0.05 into
    one of -111.
    """
    n = len(xs)
    out = []
    span = xs[-1] - xs[0] if n > 1 else 0.0
    if span <= 0:
        return out
    floor = min_span_frac * span
    for i in range(n):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                left, right = xs[j] - xs[i], xs[k] - xs[j]
                if left < floor or right < floor:
                    continue
                if min(left, right) >= wide_frac * max(left, right):
                    out.append((i, j, k))
    return out


def _dd2(xs, ys):
    """Second divided difference 2 * f[x0,x1,x2]; equals f'' for equal spacing
    up to O(h^2) and has the sign of f'' exactly for any spacing."""
    x0, x1, x2 = xs
    y0, y1, y2 = ys
    a = (y1 - y0) / (x1 - x0)
    b = (y2 - y1) / (x2 - x1)
    return 2.0 * (b - a) / (x2 - x0)


def channels_from_triple(xs, ls):
    """N_eff = s^2/kappa from three points of l = ln eps.

    For equal spacing this is exactly (l[+]-l[-])^2 / (4 (l[-]-2l[0]+l[+])),
    in which the step h cancels identically.  For unequal spacing the divided
    differences are used, which is the same number to O(h^2) and keeps the sign
    exact.  Returns None where kappa is not positive (out of class locally).
    """
    x0, x1, x2 = xs
    l0, l1, l2 = ls
    s = -(l2 - l0) / (x2 - x0)              # s = -dl/dx > 0 on a frontier
    kappa = _dd2(xs, ls)                    # kappa = l'' = -S''
    if kappa <= 0.0:
        return None
    return (s * s) / kappa


def local_slope(xs, ls):
    """s = -dl/dx from the two outer points of a triple."""
    return -(ls[-1] - ls[0]) / (xs[-1] - xs[0])


# ------------------------------------------------------------ F2 class gate

def class_gate(xs, ls, sd=None, wide_frac: float = 0.667, z: float = Z95,
               envelope_share: float = ENVELOPE_SHARE):
    """F2.  Law I before anything is priced.

    Law I (complete monotonicity) implies l = ln eps is convex, so every wide
    triple must satisfy l[-] - 2 l[0] + l[+] >= 0.  This is exact, one-sided and
    free of any spacing assumption.  ``sd`` is the per-point noise of l in nats
    (from the precision gate); when given, a defect only counts as a refutation
    if it exceeds ``z`` of its own propagated noise, so the gate does not fire
    on measurement error.

    Divisibility (the stronger hypothesis) additionally requires that the second
    difference never grows.  A rise refutes; no rise proves nothing -- by
    Proposition ``wallivp`` the two walls are tangent at the floor, so this test
    has no power there, by theorem.

    Returns a dict; ``verdict`` is one of IN_CLASS, ENVELOPE, OUT_OF_CLASS,
    UNDECIDABLE.
    """
    tri = _triples(xs, wide_frac)
    rep = {
        "n_points": len(xs),
        "n_wide_triples": len(tri),
        "min_defect": None,
        "min_defect_z": None,
        "widest_defect": None,
        "widest_defect_z": None,
        "share_triples_violating": None,
        "n_violations": 0,
        "curvature_rises": 0,
        "divisibility": "not refuted",
        "verdict": "UNDECIDABLE",
        "reason": "",
    }
    if not tri:
        rep["reason"] = "fewer than three usable points: nothing to test"
        return rep

    defects, zs = [], []
    for (i, j, k) in tri:
        d = _dd2([xs[i], xs[j], xs[k]], [ls[i], ls[j], ls[k]])
        defects.append(d)
        if sd:
            # var of the divided difference with independent per-point noise
            x0, x1, x2 = xs[i], xs[j], xs[k]
            c0 = 2.0 / ((x1 - x0) * (x2 - x0))
            c2 = 2.0 / ((x2 - x1) * (x2 - x0))
            c1 = -(c0 + c2)
            var = (c0 * c0 + c1 * c1 + c2 * c2) * sd * sd
            zs.append(d / math.sqrt(var) if var > 0 else float("inf"))
    rep["min_defect"] = min(defects)
    # the widest triple alone: the global chord, the lowest-noise single read
    i, j, k = max(tri, key=lambda t: xs[t[2]] - xs[t[0]])
    rep["widest_defect"] = _dd2([xs[i], xs[j], xs[k]], [ls[i], ls[j], ls[k]])
    rep["share_triples_violating"] = (
        sum(1 for d in defects if d < 0.0) / float(len(defects)))
    if zs:
        rep["min_defect_z"] = min(zs)
        rep["n_violations"] = sum(1 for v in zs if v < -z)
    else:
        rep["n_violations"] = sum(1 for d in defects if d < 0.0)

    # divisibility: the second difference must not grow along the axis.
    # Read it on the consecutive centred triples only -- that is a sequence in
    # x, which is what "never grows" is a statement about.  Exact at every
    # spacing (Widder); a rise refutes, no rise proves nothing, and by
    # Prop. wallivp the test has no power at the floor, by theorem.
    seq = [(xs[j], _dd2(xs[j - 1:j + 2], ls[j - 1:j + 2]))
           for j in range(1, len(xs) - 1)]
    rises, rise_z = 0, []
    for (xa, da), (xb, db) in zip(seq, seq[1:]):
        if db > da:
            if sd:
                # crude but honest: the two divided differences share one point
                var = 2.0 * (6.0 * sd * sd) / max(
                    ((xb - xa) ** 4) / 4.0, 1e-300)
                z = (db - da) / math.sqrt(var) if var > 0 else float("inf")
                rise_z.append(z)
                if z > z:
                    rises += 1
            else:
                rises += 1
    rep["curvature_rises"] = rises
    rep["curvature_rise_max_z"] = max(rise_z) if rise_z else None
    rep["divisibility"] = "REFUTED" if rises else "not refuted"

    # The verdict rests on the WIDEST triple -- the global chord, the single
    # lowest-noise read of convexity -- because a per-triple minimum over an
    # envelope is a minimum over noise.  The share of violating wide triples is
    # then the severity, and it separates two different failures:
    #
    #   chord negative           Law I itself fails.  Nothing downstream holds.
    #   chord positive, interior
    #   jagged (share of violating wide triples above ``envelope_share``,
    #   which the caller computes per instance -- median 0.183 on the corpus,
    #   not the 0.5 fallback) the object is not one frontier of one clock
    #                            but a Pareto ENVELOPE of a heterogeneous zoo.
    #                            A price can still be quoted; the second-order
    #                            reading (N_eff, the walls, the floor) cannot.
    wd = rep["widest_defect"]
    wd_bad = wd < 0.0
    if sd is not None:
        x0, x1, x2 = xs[i], xs[j], xs[k]
        c0 = 2.0 / ((x1 - x0) * (x2 - x0))
        c2 = 2.0 / ((x2 - x1) * (x2 - x0))
        c1 = -(c0 + c2)
        v = (c0 * c0 + c1 * c1 + c2 * c2) * sd * sd
        rep["widest_defect_z"] = wd / math.sqrt(v) if v > 0 else float("inf")
        wd_bad = rep["widest_defect_z"] < -z
    if wd_bad:
        rep["verdict"] = "OUT_OF_CLASS"
        rep["reason"] = (
            f"the global chord of ln eps is concave (defect {wd:+.5f}): Law I "
            f"fails, and Law I is the first thing to fail and takes everything "
            f"with it")
    elif rep["share_triples_violating"] > envelope_share:
        rep["verdict"] = "ENVELOPE"
        rep["reason"] = (
            f"the chord is convex (defect {wd:+.5f}) but "
            f"{rep['share_triples_violating']:.0%} of wide triples are not: "
            f"this is a Pareto envelope of several models, not one frontier of "
            f"one clock. A price may be quoted; N_eff, the walls and the floor "
            f"may not be read off it")
    else:
        rep["verdict"] = "IN_CLASS"
        rep["reason"] = (
            f"ln eps convex on the global chord (defect {wd:+.5f}) and on "
            f"{1-rep['share_triples_violating']:.0%} of {len(tri)} wide triples")
    return rep


# -------------------------------------------------------- F3 precision gate

def sd_of_l(accuracy: float, n_test: int) -> float:
    """F3.  Standard deviation of l = ln eps in nats from a finite test set.

    With e = 1 - Q the error rate and n items, Var(e) = e(1-e)/n and
    sd(ln e) = sd(e)/e = sqrt(Q / ((1-Q) n)).  q0 drops out: it is a constant
    shift of l.
    """
    if not n_test or n_test <= 0:
        return 0.0
    e = 1.0 - accuracy
    if e <= 0.0:
        return float("inf")
    return math.sqrt(accuracy / (e * n_test))


def required_n_test(accuracy: float, delta_nats: float = RHO2,
                    z: float = Z95) -> float:
    """F3.  Items needed so that a gap of ``delta_nats`` between two candidates
    of this accuracy is resolvable at ``z``:  n = 2 z^2 Q / ((1-Q) delta^2)."""
    e = 1.0 - accuracy
    if e <= 0.0 or delta_nats <= 0.0:
        return float("inf")
    return 2.0 * z * z * accuracy / (e * delta_nats * delta_nats)


def neff_uncertainty(xs, ls, sds, draws: int = 2000, seed: int = 0):
    """F3.  Monte-Carlo interval for N_eff on a triple, from the per-point noise.

    N_eff is a square over a second difference, so it amplifies noise: this is
    the quantitative form of the corpus finding that the curve is
    noise-dominated on published windows (median straightness R^2 = 0.22) and
    has no power against a design-matched null.  Returns
    (median, lo5, hi95, frac_out_of_class).
    """
    rng = random.Random(seed)
    vals, bad = [], 0
    for _ in range(draws):
        pert = [l + rng.gauss(0.0, sd) for l, sd in zip(ls, sds)]
        n = channels_from_triple(xs, pert)
        if n is None:
            bad += 1
        else:
            vals.append(n)
    if not vals:
        return (None, None, None, 1.0)
    vals.sort()
    q = lambda p: vals[min(len(vals) - 1, max(0, int(p * len(vals))))]
    return (q(0.50), q(0.05), q(0.95), bad / float(draws))


# --------------------------------------------------------- F6/F10 count axis

def count_axis_read(P, budget=None):
    """F6, F10.  Read the frontier on a count axis.

    ``P`` is the measured sequence P(1), P(2), ... : the probability that an
    item is still failed after k independent attempts (a pass@k curve read as a
    fail@k curve, or the co-failure curve of k drawn models).  On a count axis
    the axiom is a *theorem* of exchangeability, not an assumption, and the test
    is exact and signed at every spacing (Hausdorff: P must be a moment
    sequence, so (-1)^j Delta^j P >= 0 for every j).

    Returns a dict with
      ``alternation``   the signed differences and whether any is refuted
      ``neff``          N_eff per interior k, read with h = 1
      ``floor_bracket`` P(ceil Lambda) <= eps_inf <= P(floor Lambda): the floor
                        is two measured numbers, not an extrapolation to x = inf
      ``discount``      D(t) = S(t) / (t S(1)) in [1/t, 1], non-increasing;
                        D = 1 means a retry is worth a fresh item, D = 1/t means
                        it is worthless
      ``grain_of_retry`` the grain a retry budget t_r actually delivers,
                        S(t_r)/S(1) < t_r
    """
    P = [float(p) for p in P]
    K = len(P)
    out = {"K": K, "P": P, "alternation": {}, "neff": {}, "floor_bracket": None,
           "discount": {}, "grain_of_retry": {}, "verdict": "UNDECIDABLE"}
    if K < 2:
        return out

    def delta(j, k):
        """(-1)^j Delta^j P at k (0-based), which must be >= 0 for all j."""
        s = 0.0
        for i in range(j + 1):
            c = math.comb(j, i) * ((-1) ** i)
            s += c * P[k + i]
        return s                                    # = (-1)^j Delta^j P(k)

    refuted = []
    for j in range(1, min(4, K)):
        vals = [delta(j, k) for k in range(K - j)]
        out["alternation"][j] = vals
        if min(vals) < -1e-12:
            refuted.append(j)
    out["alternation"]["refuted_orders"] = refuted
    out["verdict"] = "OUT_OF_CLASS" if refuted else "IN_CLASS"

    l = [math.log(p) for p in P]
    for k in range(1, K - 1):
        n = channels_from_triple([k, k + 1, k + 2], [l[k - 1], l[k], l[k + 1]])
        if n is not None:
            out["neff"][k + 1] = n

    if budget is None:
        budget = float(K)
    lo_i = min(K, max(1, int(math.ceil(budget))))
    hi_i = min(K, max(1, int(math.floor(budget))))
    out["floor_bracket"] = (P[lo_i - 1], P[hi_i - 1], budget)

    S1 = -l[0]
    for t in range(1, K + 1):
        St = -l[t - 1]
        out["discount"][t] = St / (t * S1) if S1 > 0 else float("nan")
        out["grain_of_retry"][t] = St / S1 if S1 > 0 else float("nan")
    return out


# ------------------------------------------------------- F7 under-sizing

def undersizing_correction(t: float, neff_compute: float,
                           neff_count: float = None):
    """F7.  A compute-axis channel count is a LOWER bound on the task's own.

    Identically  1/N^x = 1/N^k + s_m/s  with s_m = -d ln m'/dx the clock's own
    price, so N^x <= N^k with equality iff the axis *is* a count.  Where the
    compute-axis rule fires it therefore under-sizes against the task's own
    criterion by s_m/s in inverse units.  Given a measured count-axis N^k the
    bias is identified and the corrected compute-axis target is

        N^x_target = 1 / (t + s_m/s).

    Returns (target, s_m_over_s, note).
    """
    if neff_count is None or neff_count <= 0 or neff_compute is None \
            or neff_compute <= 0:
        return (1.0 / t, None,
                "no count axis supplied: the compute-axis target 1/t is a "
                "lower bound on the task's own, so this reading under-sizes")
    c = 1.0 / neff_compute - 1.0 / neff_count
    if c <= 0.0:
        return (1.0 / t, 0.0,
                "the compute-axis reading is NOT below the count-axis reading, "
                "which the identity forbids: the attempts counted on that axis "
                "are not exchangeable (on the archive's own panel, nested "
                "difficulty, kappa = 0.835). No correction is applied and the "
                "count axis should be rebuilt from fresh items")
    return (1.0 / (t + c), c,
            f"s_m/s = {c:.4f} measured against the count axis; the uncorrected "
            f"target 1/t = {1.0/t:.4f} under-sizes, so the corrected target is "
            f"{1.0/(t+c):.4f}")


# ---------------------------------------------------- the plane, and the price

def _fit_line(pts):
    """Least squares N_eff = sigma * S + b over the readings.  Every
    constant-sigma chart is a straight line on this plane (Cor. odepencil):
    slope sigma, S-intercept -ln eps_inf.  The slope is biased low by O(h^2) --
    a chart whose true sigma is -1 reads -0.91 to -1.08 at published spacings --
    so it is used to LOCATE, never to TEST."""
    n = len(pts)
    if n < 2:
        return (None, None, None)
    sx = sum(p["S"] for p in pts); sy = sum(p["neff"] for p in pts)
    mx, my = sx / n, sy / n
    sxx = sum((p["S"] - mx) ** 2 for p in pts)
    sxy = sum((p["S"] - mx) * (p["neff"] - my) for p in pts)
    if sxx <= 0:
        return (None, None, None)
    sl = sxy / sxx
    b = my - sl * mx
    syy = sum((p["neff"] - my) ** 2 for p in pts)
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else 0.0
    return (sl, b, r2)


def read_plane(xs, ls, t_lo, t_hi=None, sd=None, undersize_c: float = 0.0,
               wide_frac: float = 0.5, min_line_r2: float = 0.30):
    """Read (N_eff, S) on a window and locate the operating point.

    ``t_lo``/``t_hi`` are the grain, mandatory (F1).  Given an interval the
    operating height is 1/sqrt(t_lo t_hi) and the worst case over the interval
    is exactly ln cosh((1/4) ln(t_hi/t_lo)).  ``undersize_c`` is s_m/s from F7,
    which moves the target from 1/t to 1/(t + s_m/s).

    The operating point is found by solving the plane's own straight line
    (Cor. odepencil) for the target height, not by clamping to the nearest
    measured model, and the report says whether that solution is inside the
    measured window or an extrapolation.  Readings are taken on WIDE triples:
    the convexity defect is O(h^2) and the noise is not.
    """
    if t_hi is None:
        t_hi = t_lo
    if not (t_lo > 0 and t_hi >= t_lo):
        raise ValueError("grain must satisfy 0 < t_lo <= t_hi")
    t_eff = math.sqrt(t_lo * t_hi)
    target = 1.0 / (t_eff + max(undersize_c, 0.0))

    # Readings on the CENTRED consecutive triples: equal arms, so every
    # reading carries the same O(h^2) bias and the line through them is not
    # tilted by mixing arm widths.  Fall back to wide triples if there are too
    # few points for that.
    def _read(idx):
        out = []
        for (i, j, k) in idx:
            n = channels_from_triple([xs[i], xs[j], xs[k]],
                                     [ls[i], ls[j], ls[k]])
            if n is None:
                continue
            out.append({"x": xs[j], "S": -ls[j], "neff": n,
                        "s": local_slope([xs[i], xs[k]], [ls[i], ls[k]])})
        return out

    pts = _read([(j - 1, j, j + 1) for j in range(1, len(xs) - 1)])
    if len(pts) < 2:
        pts = _read(_triples(xs, wide_frac))
    pts = [p for p in pts if p["neff"] > 0.0 and p["s"] > 0.0]
    pts.sort(key=lambda p: p["S"])

    rep = {"t_lo": t_lo, "t_hi": t_hi, "t_eff": t_eff, "target_neff": target,
           "n_readings": len(pts), "neff_median": None, "inside_window": False,
           "operate_x": None, "operate_neff": None, "operate_s": None,
           "price": None, "nearest_reading_regret": None,
           "grain_worst_case": None, "floor_axiom": None,
           "floor_divisible": None, "floor_from_intercept": None,
           "line_slope": None, "line_r2": None, "arc": None,
           "checkpoints_needed": None, "implied_grain_range": None,
           "extrapolated": False, "line_usable": None}
    if not pts:
        return rep
    if t_hi > t_lo:
        rep["grain_worst_case"] = math.log(
            math.cosh(0.25 * math.log(t_hi / t_lo)))

    ns = sorted(p["neff"] for p in pts)
    rep["neff_median"] = ns[len(ns) // 2]
    rep["implied_grain_range"] = (1.0 / ns[-1], 1.0 / ns[0])

    sl, b, r2 = _fit_line(pts)
    rep["line_slope"], rep["line_r2"] = sl, r2
    if sl is not None and sl < 0:
        # the line's own S-intercept is -ln eps_inf (used only when straight)
        rep["floor_from_intercept"] = math.exp(-(-b / sl)) if sl else None

    # Solve the line for the target height, then map S* back to x* -- but only
    # if the line is usable.  A chart whose readings do not lie on a line is
    # not a constant-sigma chart, and solving a line with R^2 = 0.02 produced
    # prices of 1494 nats per e-fold on the archive's own param_count axis.
    # Where the line is unusable the nearest reading is used instead and the
    # report says so.
    Ss = [p["S"] for p in pts]
    S_star = None
    rep["line_usable"] = bool(sl is not None and sl < 0.0
                              and r2 is not None and r2 >= min_line_r2)
    if rep["line_usable"] and abs(sl) > 1e-12:
        S_star = (target - b) / sl
    if S_star is None:
        chosen = min(pts, key=lambda p: abs(math.log(p["neff"] / target)))
        rep["operate_x"], rep["operate_neff"] = chosen["x"], chosen["neff"]
        rep["operate_s"] = chosen["s"]
    else:
        rep["inside_window"] = (min(Ss) <= S_star <= max(Ss))
        rep["extrapolated"] = not rep["inside_window"]
        # x* by monotone interpolation of S(x) on the raw window
        S_raw = [-l for l in ls]
        x_star = None
        for (xa, Sa), (xb, Sb) in zip(zip(xs, S_raw), zip(xs[1:], S_raw[1:])):
            if (Sa - S_star) * (Sb - S_star) <= 0 and Sb != Sa:
                x_star = xa + (S_star - Sa) * (xb - xa) / (Sb - Sa)
                break
        if x_star is None:            # extrapolate on the last/first segment
            if S_star > max(S_raw):
                xa, Sa, xb, Sb = xs[-2], S_raw[-2], xs[-1], S_raw[-1]
            else:
                xa, Sa, xb, Sb = xs[0], S_raw[0], xs[1], S_raw[1]
            x_star = xa + (S_star - Sa) * (xb - xa) / (Sb - Sa) \
                if Sb != Sa else xs[-1]
        # s at x*: the local price read off its own straight line in S.
        # s is smooth in S and this is one line fit, no bandwidth; the nearest
        # single reading is used only if the line is degenerate.
        sl_s, b_s, _ = _fit_line([{"S": p["S"], "neff": p["s"]} for p in pts])
        s_star = (sl_s * S_star + b_s) if sl_s is not None else None
        if s_star is None or s_star <= 0:
            s_star = min(pts, key=lambda p: abs(p["x"] - x_star))["s"]
        rep["operate_x"] = x_star
        rep["operate_neff"] = target
        rep["operate_s"] = s_star

    # lambda* = s / sqrt(t N_eff); equals s exactly when N_eff = 1/t
    rep["price"] = rep["operate_s"] / math.sqrt(t_eff * rep["operate_neff"])
    # what the closest MEASURED point on the frontier costs against the rule
    near = min(pts, key=lambda p: abs(math.log(p["neff"] / target)))
    rep["nearest_reading_regret"] = math.log(
        math.cosh(0.5 * math.log(near["neff"] / target)))

    eps_here = math.exp(-near["S"])
    N = near["neff"]
    rep["floor_axiom"] = eps_here / (1.0 + N)
    rep["floor_divisible"] = eps_here * math.exp(-N)

    # arc: dphi = dS / sqrt(N_eff); trapezoid over readings
    arc = 0.0
    for a, bb in zip(pts, pts[1:]):
        dS = bb["S"] - a["S"]
        inv = 0.5 * (1.0 / math.sqrt(a["neff"]) + 1.0 / math.sqrt(bb["neff"]))
        arc += abs(dS) * inv
    rep["arc"] = arc
    rep["checkpoints_needed"] = max(1.0, arc / (2.0 * math.sqrt(2.0 * RHO2)))
    return rep


def price_uncertainty(xs, ls, sds, t_lo, t_hi=None, undersize_c=0.0,
                      draws: int = 400, seed: int = 1):
    """F3.  The derived price inherits the test-set noise of the calibration
    curve, and a price error moves every candidate's score by lambda-error
    times its log-resource.  This re-reads the whole plane on perturbed
    calibration points and returns (median, lo5, hi95, sd) of lambda*.

    It is also the honest report on the O(h^2) slope bias: every constant-sigma
    chart is a straight line on the plane and the plug-in slope reads low in
    magnitude, so the solved operating depth is biased DEEP and the derived
    price biased LOW.  The direction is known; the size is what this returns.
    """
    rng = random.Random(seed)
    vals = []
    for _ in range(draws):
        pert = [l + rng.gauss(0.0, sd) for l, sd in zip(ls, sds)]
        try:
            r = read_plane(xs, pert, t_lo, t_hi, undersize_c=undersize_c)
        except Exception:
            continue
        if r.get("price") and r["price"] > 0:
            vals.append(r["price"])
    if len(vals) < 10:
        return (None, None, None, None)
    vals.sort()
    q = lambda pr: vals[min(len(vals) - 1, max(0, int(pr * len(vals))))]
    # robust scale: the draws are heavy-tailed (a near-zero second difference
    # sends the price to infinity), so a moment estimate is meaningless here.
    sd = (q(0.95) - q(0.05)) / (2.0 * 1.6448536269514722)
    return (q(0.50), q(0.05), q(0.95), sd)


def calibration_requirement(xs, ls, accuracy_mid, tol: float = 0.25):
    """F3.  What the calibration curve would have to be for N_eff to mean
    anything, and the two ways to get there.

    kappa is read as a second difference, whose noise is sqrt(6) sd(l) while
    its signal is h^2 kappa.  Requiring sd(kappa)/kappa <= ``tol`` gives

        sd(l) <= tol * h^2 * kappa / sqrt(6),
        n_test >= Q / ((1-Q) sd(l)^2),

    and because the signal grows as h^2 while the noise does not, WIDENING THE
    GEOMETRIC GRID is quadratically cheaper than adding items: the factor the
    current grid would have to be stretched by is sqrt of the shortfall in n.
    """
    if len(xs) < 3:
        return None
    steps = [b - a for a, b in zip(xs, xs[1:])]
    h = sum(steps) / len(steps)
    j = len(xs) // 2
    kappa = _dd2(xs[j - 1:j + 2], ls[j - 1:j + 2])
    if kappa <= 0:
        return None
    sd_needed = tol * h * h * kappa / math.sqrt(6.0)
    e = 1.0 - accuracy_mid
    n_needed = accuracy_mid / (e * sd_needed * sd_needed) if e > 0 else None
    return {"h": h, "kappa": kappa, "sd_l_needed": sd_needed,
            "n_test_needed": n_needed}


# ------------------------------------------ F4/F5/F8/F9 the decision itself

def select(models, prices, band: float = RHO2, floor: float = None,
           resolution: dict = None, price_sd: dict = None,
           price_range: dict = None, n_grid: int = 41):
    """F4, F5, F8, F9.  Score, then report a tie SET rather than a winner.

    ``models``   list of dicts with ``name``, ``eps``, and ``x`` a dict of
                 log-resources per axis.  Candidates need NOT lie on the
                 calibration frontier (F9): the price comes from the frontier,
                 the score from the candidate.
    ``prices``   dict axis -> lambda in nats per e-fold (F5: one per axis,
                 derived where that axis has a calibration curve, declared
                 otherwise).
    ``band``     indifference band in nats; widen it with the measured
                 resolution (F3) before calling.
    ``floor``    eps_inf, for the accuracy-only canonical value J (F4).  When
                 None, J is omitted rather than guessed: the floor is not a
                 quantity (Prop. nofloor) and is classically non-identified.
    ``resolution`` optional dict name -> sd of l in nats, used to widen the band
                 pairwise and to mark unresolvable gaps.
    """
    out = []
    for m in models:
        S = -math.log(m["eps"])
        cost = sum(prices.get(k, 0.0) * v for k, v in m["x"].items())
        out.append({"name": m["name"], "eps": m["eps"], "x": dict(m["x"]),
                    "S": S, "cost": cost, "sigma": S - cost})
    out.sort(key=lambda r: -r["sigma"])
    best = out[0]["sigma"]
    for r in out:
        r["gap"] = best - r["sigma"]

    # F3: widen the band to the resolution actually achieved -- test-set
    # noise on both candidates, plus the derived price's own uncertainty times
    # the log-resource lever arm.  A gap smaller than this is not a gap.
    eff_band = band
    tie = []
    for r in out:
        v = 0.0
        if resolution:
            v += resolution.get(out[0]["name"], 0.0) ** 2
            v += resolution.get(r["name"], 0.0) ** 2
        if price_sd:
            for ax, sdl in price_sd.items():
                if not sdl:
                    continue
                dx = r["x"].get(ax, 0.0) - out[0]["x"].get(ax, 0.0)
                v += (dx * sdl) ** 2
        r["resolution"] = Z95 * math.sqrt(v)
        r["band_here"] = max(band, r["resolution"])
        if r["gap"] <= r["band_here"]:
            tie.append(r)
        eff_band = max(eff_band, r["band_here"]) if r in tie else eff_band
    if out[0] not in tie:
        tie.insert(0, out[0])
    for r in out:
        r["in_tie_band"] = r in tie

    # F8: inside the band, break the tie on WORST-CASE REGRET over the price
    # range the frontier itself supports -- not on cost, and not on accuracy.
    # Measured on the archive (script 143, table_selector_audit*.csv): the
    # minimax tiebreak cuts the median excess over the minimax-optimal model
    # from 0.075 to 0.029 nats frontier by frontier (the difference of the two
    # medians is the larger 0.329 against 0.269; they are not the same
    # statistic), because a tie in sigma at one price is not a tie in
    # sigma over a range of prices.  Cost breaks any remaining tie.
    worst = None
    if price_range:
        grid = []
        for ax, (lo, hi) in price_range.items():
            if not (hi > lo > 0):
                continue
            grid.append((ax, [lo * (hi / lo) ** (a / (n_grid - 1.0))
                              for a in range(n_grid)]))
        if grid:
            ax, lams = grid[0]          # the derived axis; others stay declared
            worst = {r["name"]: 0.0 for r in out}
            for lam in lams:
                pr = dict(prices)
                pr[ax] = lam
                sg = {r["name"]: r["S"] - sum(pr.get(k, 0.0) * v
                                              for k, v in r["x"].items())
                      for r in out}
                b = max(sg.values())
                for nm, v in sg.items():
                    worst[nm] = max(worst[nm], b - v)
            for r in out:
                r["worst_case_regret"] = worst[r["name"]]
    if worst:
        pick = min(tie, key=lambda r: (worst[r["name"]], r["cost"]))
    else:
        pick = min(tie, key=lambda r: (r["cost"], r["eps"]))

    # F4: the accuracy-only canonical value, and what following it would cost
    canon = None
    if floor is not None and floor > 0:
        ok = [r for r in out if r["eps"] > floor]
        if ok:
            for r in out:
                r["J"] = (0.5 * math.log(max(r["eps"] - floor, 1e-300))
                          - math.log(r["eps"]))
            canon = max(ok, key=lambda r: r["J"])

    rep = {
        "ranking": out,
        "band": band,
        "effective_band": eff_band,
        "tie_set": [r["name"] for r in tie],
        "pick": pick["name"],
        "verdict": "RESOLVED" if len(tie) == 1 else "TIE",
        "canonical_pick": canon["name"] if canon else None,
        "tiebreak": ("worst-case regret over the supported price range"
                     if worst else "delivered cost"),
        "minimax_pick": (min(out, key=lambda r: worst[r["name"]])["name"]
                         if worst else None),
        "blind_spot": None,
        "blind_spot_regret": None,
    }
    if canon is not None:
        rep["blind_spot"] = (canon["name"] != pick["name"])
        rep["blind_spot_regret"] = best - canon["sigma"]
    return rep


# ------------------------------------------------------------ depth bounds

def depth_bounds(t: float):
    """Minimum remaining depth at the operating point: ln(1 + 1/t) under the
    axiom, 1/t under divisibility.  At grain 1 that is one bit against one nat,
    and the gap 1 - ln 2 = 0.307 nats is the divisibility test read at the
    rule's own point.  Also: never operate within one nat of the floor."""
    return {"axiom": math.log(1.0 + 1.0 / t), "divisible": 1.0 / t,
            "gap": 1.0 / t - math.log(1.0 + 1.0 / t)}


# ==========================================================================
#  F11 -- PAIRED RESOLUTION
# ==========================================================================
#
# Every candidate is evaluated on the SAME test items.  The quantity a decision
# turns on is a DIFFERENCE, l_i - l_j, and the difference of two means taken on
# shared items does not have the variance of the sum of two independent means.
# With f_i(k) = 1 if model i fails item k, e_i = E f_i and p_ij = Pr[both fail],
# the delta method on l = ln e gives
#
#     Var(l_i - l_j) = V1 / n,
#     V1 = (1-e_i)/e_i + (1-e_j)/e_j - 2 (p_ij - e_i e_j) / (e_i e_j).
#
# The unpaired formula keeps only the first two terms.  Since failures on the
# same items are positively associated -- the archive's own panel measures
# Pr[both fail] = kappa min(h_i,h_j) with kappa = 0.835 -- the third term is
# strictly negative, so the unpaired band is too WIDE and the n_test it demands
# too LARGE.  Nothing new has to be measured: the per-item correctness vector is
# already computed by whatever produced the accuracy.

def failure_stats(fail_i, fail_j):
    """Per-item failure statistics of a pair, from two boolean sequences.

    Returns ``(n, e_i, e_j, p_both, b, c)`` where ``b`` counts items that only
    ``i`` fails and ``c`` items that only ``j`` fails (the McNemar discordants).
    """
    n = len(fail_i)
    if n != len(fail_j) or n == 0:
        raise ValueError("per-item vectors must be the same non-zero length")
    a = b = c = 0
    for u, v in zip(fail_i, fail_j):
        u = bool(u); v = bool(v)
        if u and v:
            a += 1
        elif u:
            b += 1
        elif v:
            c += 1
    return (n, (a + b) / n, (a + c) / n, a / n, b, c)


def phi_correlation(e_i, e_j, p_both):
    """The correlation of the two failure indicators; 0 is independence."""
    d = math.sqrt(max(e_i * (1.0 - e_i) * e_j * (1.0 - e_j), 0.0))
    if d <= 0.0:
        return 0.0
    return (p_both - e_i * e_j) / d


def _v1(e_i, e_j, p_both=None, rho=None):
    """The per-item variance factor of l_i - l_j (see the section comment)."""
    if not (0.0 < e_i < 1.0 and 0.0 < e_j < 1.0):
        return float("inf")
    v = (1.0 - e_i) / e_i + (1.0 - e_j) / e_j
    if p_both is None and rho is not None:
        p_both = e_i * e_j + rho * math.sqrt(
            e_i * (1.0 - e_i) * e_j * (1.0 - e_j))
    if p_both is not None:
        v -= 2.0 * (p_both - e_i * e_j) / (e_i * e_j)
    return max(v, 0.0)


def sd_of_gap(e_i, e_j, n_test, p_both=None, rho=None):
    """F11.  sd of l_i - l_j in nats.  ``p_both`` from per-item data, or
    ``rho`` as a declared failure correlation; neither given is the unpaired
    (independent-errors) formula, which is an upper bound."""
    if not n_test or n_test <= 0:
        return 0.0
    return math.sqrt(_v1(e_i, e_j, p_both, rho) / float(n_test))


def required_n_gap(e_i, e_j, delta_nats=RHO2, p_both=None, rho=None, z=Z95):
    """F11.  Items needed to resolve a gap of ``delta_nats`` between this pair.
    With the covariance term this is (1-rho)-ish times the unpaired count."""
    if delta_nats <= 0.0:
        return float("inf")
    return _v1(e_i, e_j, p_both, rho) * z * z / (delta_nats * delta_nats)


# ==========================================================================
#  F12 -- THE CERTIFIED SET
# ==========================================================================
#
# The band of F3/F8 is an indifference region, not a confidence set: on the
# archive's own audit the minimax-optimal model lies inside it on 45% of
# frontiers, so "maximum correctness" was an unquantified claim.  F12 replaces
# it with a set that carries a guarantee.
#
# Write the score of a candidate as sigma_i = -l_i - c_i with c_i = sum_a
# lambda_a x_{ia}.  Then
#
#     sigma_i > sigma_j   <=>   e_j > R0 e_i ,     R0 = exp(c_i - c_j),
#
# so domination is a statement about the mean of the BOUNDED per-item variable
#
#     v_k = 1{j fails item k} - R0 * 1{i fails item k}   in  [-R0, 1],
#     E v = e_j - R0 e_i .
#
# A one-sided distribution-free finite-sample lower bound on E v (empirical
# Bernstein, Maurer and Pontil) therefore CERTIFIES domination with no
# distributional assumption, no asymptotics, no convexity, no N_eff and no
# floor.  Candidate j is excluded only when some i certifies domination at
# every price the frontier supports, each pair tested at alpha/(K-1); the
# price-optimal model survives unless one of those K-1 certificates is wrong,
# so by a union bound
#
#     Pr[ the price-optimal model is in the returned set ] >= 1 - alpha.
#
# This is the only statement in the pipeline that holds out of class, and it is
# the one the decision needs.

def _z_of(p):
    """Normal quantile, stdlib."""
    return statistics.NormalDist().inv_cdf(p)


def eb_lower(values, alpha, rng_width):
    """One-sided (1-alpha) empirical-Bernstein lower bound on the mean of an
    i.i.d. sample bounded in an interval of width ``rng_width``.

    Maurer and Pontil: with probability at least 1 - alpha,
        E X >= mean - sqrt(2 Vhat ln(1/alpha) / n) - 7 R ln(1/alpha) / (3(n-1)).
    Finite sample and distribution free; the only assumption is that the test
    items are drawn independently, which is the same assumption the accuracy
    itself needs.
    """
    n = len(values)
    if n < 2 or alpha <= 0.0 or alpha >= 1.0:
        return float("-inf")
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1.0)
    L = math.log(1.0 / alpha)
    return m - math.sqrt(2.0 * var * L / n) - 7.0 * rng_width * L / (3.0 * (n - 1))


def contingency(fail_i, fail_j):
    """The 2x2 table (a, b, c, d) of a pair on shared items: a = both fail,
    b = only i fails, c = only j fails, d = neither."""
    a = b = c = d = 0
    for u, v in zip(fail_i, fail_j):
        if u:
            if v:
                a += 1
            else:
                b += 1
        elif v:
            c += 1
        else:
            d += 1
    return (a, b, c, d)


def certify_domination_counts(table, dcost, alpha):
    """Certify sigma_i > sigma_j from the pair's 2x2 table alone.

    The per-item variable v = 1{j fails} - R0 1{i fails}, R0 = exp(dcost),
    takes only four values, so its mean and variance are closed forms of the
    table -- no per-item loop, and exactly the same number:

        both fail -> 1 - R0 ,  i only -> -R0 ,  j only -> 1 ,  neither -> 0 .

    E v = e_j - R0 e_i, and E v > 0 is exactly sigma_i > sigma_j.  Returns
    (certified, lower_bound, mean).
    """
    a, b, c, d = table
    n = a + b + c + d
    if n < 2:
        return (False, float("-inf"), None)
    # a cost difference of more than 50 nats makes R0 either astronomical or
    # negligible; clamp so the bound stays computable at the ends of a
    # bisection over price (F21)
    R0 = math.exp(max(-50.0, min(50.0, dcost)))
    m = ((a * (1.0 - R0)) + (b * -R0) + c) / n
    m2 = ((a * (1.0 - R0) ** 2) + (b * R0 * R0) + c) / n
    var = (m2 - m * m) * n / (n - 1.0)
    L = math.log(1.0 / alpha)
    lb = (m - math.sqrt(max(2.0 * var * L / n, 0.0))
          - 7.0 * (1.0 + R0) * L / (3.0 * (n - 1)))
    return (lb > 0.0, lb, m)


def certify_domination(fail_i, fail_j, dcost, alpha):
    """Certify sigma_i > sigma_j at level ``alpha``, distribution-free.

    ``dcost`` = c_i - c_j in nats (what i costs over j at the price in force).
    Returns (certified, lower_bound, margin) where ``lower_bound`` is the
    (1-alpha) lower bound on e_j - exp(dcost) e_i; positive means certified.
    """
    return certify_domination_counts(contingency(fail_i, fail_j), dcost, alpha)


def _costs(model, prices):
    return sum(prices.get(k, 0.0) * v for k, v in model["x"].items())


def _price_grid(prices, price_range, n_grid):
    """The list of price dicts the frontier itself supports.  One axis varies
    (the derived one); the others stay as declared."""
    grids = [dict(prices)]
    if not price_range:
        return grids
    for ax, (lo, hi) in price_range.items():
        if not (hi > lo > 0):
            continue
        out = []
        for a in range(n_grid):
            pr = dict(prices)
            pr[ax] = lo * (hi / lo) ** (a / (n_grid - 1.0))
            out.append(pr)
        grids = out
        break
    return grids


def certified_set(models, prices, price_range=None, alpha: float = 0.10,
                  items: dict = None, n_test: int = None, rho: float = None,
                  pair_items: dict = None, tables: dict = None,
                  n_grid: int = 41):
    """F12.  Return the set of candidates that cannot be certifiably excluded,
    together with the coverage level that set carries.

    ``models``      as in ``select``: name, eps, x (log-resources per axis).
    ``prices``      one price per axis.
    ``price_range`` {axis: (lo, hi)} -- the PRICE UNCERTAINTY, and nothing
                    wider.  A candidate is excluded only if one rival beats it
                    at every price in the interval, so the interval must be the
                    range the price is genuinely not known within: the derived
                    price's own confidence interval (F3), or a declared +/-.
                    Passing the whole range the frontier supports is a
                    different question -- then no single model is optimal
                    throughout and the honest output is the certified SCHEDULE
                    below, not a set.
    ``alpha``       1 - alpha is the guaranteed coverage of the price-optimal
                    model.  Each of the K-1 pairwise certificates is run at
                    alpha/(K-1).
    ``items``       {name: per-item failure vector}.  Present -> the exact
                    distribution-free certificate (method ``exact-paired``).
    ``n_test``,
    ``rho``         absent per-item data, the normal paired certificate with a
                    declared failure correlation (``normal-paired``), or the
                    unpaired one if rho is None (``normal-unpaired``, the
                    conservative fallback).

    The returned dict names its own method and its own assumption set, so a
    downgrade can never be silent (F17).
    """
    K = len(models)
    rows = []
    for m in models:
        rows.append({"name": m["name"], "eps": m["eps"], "x": dict(m["x"]),
                     "S": -math.log(m["eps"]),
                     "cost": _costs(m, prices)})
        rows[-1]["sigma"] = rows[-1]["S"] - rows[-1]["cost"]
    rows.sort(key=lambda r: -r["sigma"])
    by = {r["name"]: r for r in rows}
    grids = _price_grid(prices, price_range, n_grid)

    if K < 2:
        return {"method": "trivial", "alpha": alpha, "coverage": 1.0,
                "set": [r["name"] for r in rows], "rejected": [],
                "pick": rows[0]["name"] if rows else None,
                "assumptions": "none", "alpha_pair": alpha,
                "regret_bound": 0.0, "certificates": []}

    a_pair = alpha / (K - 1.0)
    if items or tables:
        method, assumptions = "exact-paired", (
            "independent test items only: no distributional form, no "
            "asymptotics, no Law I, no N_eff, no floor")
    elif rho is not None and n_test:
        method, assumptions = "normal-paired", (
            "asymptotic normality of a difference of means, with a DECLARED "
            "failure correlation rho = %.3f; no Law I" % rho)
    elif n_test:
        method, assumptions = "normal-unpaired", (
            "asymptotic normality AND independent errors between candidates, "
            "which the archive's own panel refutes: this is conservative "
            "(the band is too wide), never anti-conservative")
    else:
        method, assumptions = "uncertified", "no n_test and no per-item data"

    z_pair = _z_of(1.0 - a_pair) if method.startswith("normal") else None
    certs, rejected = [], set()
    for j in rows:
        for i in rows:
            if i["name"] == j["name"] or j["name"] in rejected:
                continue
            # the hardest price for i: the one where i's cost advantage is least
            worst = max(grids, key=lambda pr: _costs(i, pr) - _costs(j, pr))
            dcost = _costs(i, worst) - _costs(j, worst)
            dsigma = (i["S"] - dcost) - j["S"]
            ok, lb, stat = False, None, None
            if method == "exact-paired":
                tb = None
                if tables:
                    tb = tables.get((i["name"], j["name"]))
                    if tb is None and tables.get((j["name"], i["name"])):
                        a2, b2, c2, d2 = tables[(j["name"], i["name"])]
                        tb = (a2, c2, b2, d2)
                if tb is None:
                    fi, fj = items.get(i["name"]), items.get(j["name"])
                    if fi is None or fj is None:
                        continue
                    tb = contingency(fi, fj)
                ok, lb, stat = certify_domination_counts(tb, dcost, a_pair)
            elif method.startswith("normal"):
                e_i, e_j = i["eps"], j["eps"]
                p_both = None
                if pair_items:
                    p_both = pair_items.get((i["name"], j["name"]),
                                            pair_items.get((j["name"], i["name"])))
                sd = sd_of_gap(e_i, e_j, n_test, p_both=p_both,
                              rho=(rho if method == "normal-paired" else None))
                stat = dsigma / sd if sd > 0 else float("inf")
                lb = dsigma - z_pair * sd
                ok = stat > z_pair
            if ok:
                rejected.add(j["name"])
                certs.append({"loser": j["name"], "winner": i["name"],
                              "dsigma_worst": dsigma, "lower_bound": lb,
                              "statistic": stat, "alpha_pair": a_pair})
    keep = [r["name"] for r in rows if r["name"] not in rejected]

    # the regret the SET can still cost: the worst gap, over the supported
    # prices, between the best member and the best candidate overall.
    bound = 0.0
    for pr in grids:
        sg = {r["name"]: r["S"] - _costs(r, pr) for r in rows}
        bound = max(bound, max(sg.values()) - max(sg[k] for k in keep))
    # inside the set, the minimax member -- F8's tiebreak, now applied to a set
    # that carries a guarantee.
    worst_of = {}
    for nm in keep:
        w = 0.0
        for pr in grids:
            sg = {r["name"]: r["S"] - _costs(r, pr) for r in rows}
            w = max(w, max(sg.values()) - sg[nm])
        worst_of[nm] = w
    pick = min(keep, key=lambda nm: (worst_of[nm], by[nm]["cost"])) if keep else None
    return {"method": method, "assumptions": assumptions, "alpha": alpha,
            "alpha_pair": a_pair, "coverage": 1.0 - alpha,
            "set": keep, "rejected": sorted(rejected), "certificates": certs,
            "pick": pick, "worst_case_regret": worst_of,
            "regret_bound": bound, "ranking": rows,
            "verdict": ("CERTIFIED" if method != "uncertified" else
                        "UNCERTIFIED")}


# ==========================================================================
#  F13 -- PROSPECTIVE DESIGN OF THE CALIBRATION CURVE
# ==========================================================================

def design_calibration(accuracy, n_test, s_expected, t=1.0, K=5,
                       tol: float = 0.25, kappa_expected=None,
                       h_available=None):
    """F13.  What the calibration grid must span, BEFORE the runs are spent.

    kappa is a second difference: its noise is sqrt(6) sd(l) / h^2 and its
    signal is kappa, so requiring sd(kappa)/kappa <= tol gives

        h >= sqrt( sqrt(6) sd(l) / (tol kappa) ) ,    sd(l) = sqrt(Q/((1-Q) n)).

    Before any run kappa is unknown, but at the operating point the rule itself
    fixes it: N_eff = 1/t and N_eff = s^2/kappa give kappa = s^2 t, so an
    expected slope and the grain are enough.  Returns the minimum step, the
    resource ratio the whole grid then spans, and -- if ``h_available`` is
    given -- a FEASIBLE/INFEASIBLE verdict with both remedies.
    """
    sd_l = sd_of_l(accuracy, n_test) if n_test else None
    kappa = kappa_expected if kappa_expected else (s_expected ** 2) * t
    out = {"sd_l": sd_l, "kappa_expected": kappa, "tol": tol, "K": K,
           "h_min": None, "h_min_price": None, "span": None,
           "resource_ratio": None, "resource_ratio_price": None,
           "n_test_min_at_h": None, "widen_by": None, "widen_by_price": None,
           "verdict": None, "verdict_price": None}
    if not sd_l or sd_l <= 0 or kappa <= 0:
        out["verdict"] = out["verdict_price"] = "UNDETERMINED"
        return out
    h_min = math.sqrt(math.sqrt(6.0) * sd_l / (tol * kappa))
    # The PRICE is a weaker target than kappa itself.  lambda* = s/sqrt(t N_eff)
    # and N_eff = s^2/kappa, so a relative error tol in lambda needs only 2 tol
    # in kappa -- and h scales as tol^{-1/2}, so the grid may be sqrt(2)
    # narrower for a price than for a channel count.  Reported separately,
    # because a pipeline that only needs a price need not pay for N_eff.
    h_price = h_min / math.sqrt(2.0)
    out["h_min"], out["h_min_price"] = h_min, h_price
    out["span"] = (K - 1) * h_min
    out["resource_ratio"] = math.exp((K - 1) * h_min)
    out["resource_ratio_price"] = math.exp((K - 1) * h_price)
    if h_available and h_available > 0:
        # items needed to reach tol at the grid actually available
        sd_need = tol * h_available * h_available * kappa / math.sqrt(6.0)
        e = 1.0 - accuracy
        out["n_test_min_at_h"] = (accuracy / (e * sd_need * sd_need)
                                  if e > 0 and sd_need > 0 else float("inf"))
        out["widen_by"] = h_min / h_available
        out["widen_by_price"] = h_price / h_available
        out["verdict"] = "FEASIBLE" if h_available >= h_min else "INFEASIBLE"
        out["verdict_price"] = ("FEASIBLE" if h_available >= h_price
                                else "INFEASIBLE")
    else:
        out["verdict"] = out["verdict_price"] = "FEASIBLE"
    return out


# ==========================================================================
#  F14/F15 -- HOW MANY NETWORKS, AND WHERE THE NEXT ONE GOES
# ==========================================================================

def worst_case_regret(models, prices, price_range, subset=None, n_grid=41):
    """Worst case, over the prices the frontier supports, of the loss from
    being restricted to ``subset`` instead of all of ``models``."""
    rows = [{"name": m["name"], "S": -math.log(m["eps"]), "x": dict(m["x"])}
            for m in models]
    names = [r["name"] for r in rows] if subset is None else list(subset)
    grids = _price_grid(prices, price_range, n_grid)
    worst, lam_worst = 0.0, None
    for pr in grids:
        sg = {r["name"]: r["S"] - _costs(r, pr) for r in rows}
        gap = max(sg.values()) - max(sg[n] for n in names)
        if gap > worst:
            worst, lam_worst = gap, pr
    return worst, lam_worst


def envelope_upper(points, lam, window_only: bool = True):
    """The highest score an UNSEEN model could have at price ``lam``, given the
    candidates in hand and Law I.

    S = -ln eps is concave in log-resource under Law I, so the curve lies below
    the extension of every secant: for y > x_i,  S(y) <= S_i + s_prev (y - x_i)
    with s_prev the slope of the secant arriving at x_i, and for y < x_j,
    S(y) <= S_j + s_next (y - x_j) with s_next the slope of the secant leaving
    x_j.  Inside each gap between adjacent candidates the upper envelope is the
    minimum of those two lines, a piecewise-affine function whose maximum of
    S - lam x is at an endpoint or at the kink -- a closed form, no search.

    Outside the span of the candidates concavity bounds nothing (a frontier may
    keep rising), so with ``window_only`` the bound is stated for the resource
    window the candidates themselves cover, which is the window a deployment
    operates in anyway.  Returns (upper_bound, at_x).
    """
    pts = sorted(points, key=lambda p: p[0])
    m = len(pts)
    best, at = max(((S - lam * x, x) for x, S in pts), default=(None, None))
    if m < 2:
        return (best, at)
    for i in range(m - 1):
        xa, Sa = pts[i]
        xb, Sb = pts[i + 1]
        if xb <= xa:
            continue
        lines = []
        if i >= 1 and pts[i][0] > pts[i - 1][0]:
            s_prev = (Sa - pts[i - 1][1]) / (xa - pts[i - 1][0])
            lines.append((s_prev, Sa - s_prev * xa))
        if i + 2 <= m - 1 and pts[i + 2][0] > xb:
            s_next = (pts[i + 2][1] - Sb) / (pts[i + 2][0] - xb)
            lines.append((s_next, Sb - s_next * xb))
        if not lines:
            # no secant on either side: the chord's own slope is the only
            # information, and concavity puts the curve above it, not below.
            # Nothing can be bounded here.
            return (float("inf"), None)
        cand = [xa, xb]
        if len(lines) == 2:
            (m1, b1), (m2, b2) = lines
            if abs(m1 - m2) > 1e-15:
                xc = (b2 - b1) / (m1 - m2)
                if xa <= xc <= xb:
                    cand.append(xc)
        for x in cand:
            U = min(mm * x + bb for mm, bb in lines)
            v = U - lam * x
            if best is None or v > best:
                best, at = v, x
    return (best, at)


def stopping_rule(models, prices, price_range, rho=RHO2, n_grid=41,
                  assume_concave: bool = True):
    """F14.  Do the candidates in hand already decide the question?

    The quantity is the worst case, over the price interval in force, of what
    an UNSEEN network could still be worth over the best candidate already
    evaluated.  Once it is below one doubling (rho_2 = 0.0589 nats) no further
    network can change the decision by more than the paper's own indifference,
    and the search stops.  This is an INSTANCE-level rule -- read off the
    candidates in hand, not off a corpus median -- and it is what answers "how
    few networks" for a particular deployment.

    With ``assume_concave`` the bound is the concave upper envelope of Law I
    (``envelope_upper``), which is what Law I is FOR; without it, the rule falls
    back to the gap between the best candidate and its neighbours, which is
    sound with no assumption at all but much weaker.  Both are returned, so the
    cost of the assumption is visible.

    Also returns (F15) the log-resource at which the next candidate should be
    evaluated: the point where that worst case is attained.
    """
    rows = sorted(({"name": m["name"], "S": -math.log(m["eps"]),
                    "x": dict(m["x"])} for m in models),
                  key=lambda r: sum(r["x"].values()))
    grids = _price_grid(prices, price_range, n_grid)
    axes = sorted({k for r in rows for k in r["x"]})
    # the envelope argument is a statement about ONE curve: S concave along one
    # log-resource axis.  With several axes the candidates do not lie on one
    # curve, so the envelope is not available and the rule says so.
    single = (len(axes) == 1)
    worst_env, worst_gap, pr_at, x_at = 0.0, 0.0, None, None
    for pr in grids:
        cost = [_costs(r, pr) for r in rows]
        sg = [rows[a]["S"] - cost[a] for a in range(len(rows))]
        best = max(sg)
        if single:
            ax = axes[0]
            lam = pr.get(ax, 0.0)
            pts = [(r["x"].get(ax, 0.0), r["S"]) for r in rows]
            ub, at = envelope_upper(pts, lam)
            if ub is not None and ub - best > worst_env:
                worst_env, pr_at, x_at = ub - best, pr, at
        iw = max(range(len(sg)), key=lambda a: sg[a])
        nb = [a for a in (iw - 1, iw + 1) if 0 <= a < len(sg)]
        local = min((sg[iw] - sg[a]) for a in nb) if nb else 0.0
        if local > worst_gap:
            worst_gap = local
            if not single:
                pr_at = pr
    worst = worst_env if (assume_concave and single) else worst_gap
    return {"worst_case": worst, "worst_case_envelope": worst_env,
            "worst_case_gap": worst_gap, "stop": worst <= rho, "rho": rho,
            "price_at_worst": pr_at, "next_log_resource": x_at,
            "basis": ("concave envelope (Law I)"
                      if (assume_concave and single)
                      else "neighbour gap (no assumption; the envelope needs "
                           "one axis)"),
            "n_candidates": len(rows)}


def greedy_subset(models, prices, price_range, K, n_grid=41):
    """F15.  The K candidates a sequential minimax design would have chosen:
    start from the two extremes and add, each time, the candidate that most
    reduces the worst case over the supported price range.  Measured against
    equal-arc and equal-log-resource placement in
    ``scripts/148_sequential_design.py``."""
    rows = sorted(({"name": m["name"], "S": -math.log(m["eps"]),
                    "x": dict(m["x"])} for m in models),
                  key=lambda r: sum(r["x"].values()))
    names = [r["name"] for r in rows]
    if K >= len(names):
        return names
    chosen = [names[0], names[-1]][:K]
    while len(chosen) < K:
        best_nm, best_w = None, None
        for nm in names:
            if nm in chosen:
                continue
            w, _ = worst_case_regret(models, prices, price_range,
                                     subset=chosen + [nm], n_grid=n_grid)
            if best_w is None or w < best_w:
                best_nm, best_w = nm, w
        if best_nm is None:
            break
        chosen.append(best_nm)
    return chosen


# ==========================================================================
#  F16 -- THE O(h^2) SLOPE BIAS, CORRECTED
# ==========================================================================

def richardson_price(xs, ls, t_lo, t_hi=None, undersize_c: float = 0.0,
                     price_sd=None):
    """F16.  The plane's plug-in slope is biased low in magnitude by O(h^2), so
    the derived price reads LOW and the operating depth DEEP.  The bias has a
    known order, so it can be removed rather than reported: read the plane on
    the full grid (step h) and on the half grid (step 2h) and combine

        A_corrected = (4 A_h - A_2h) / 3,

    which cancels the h^2 term exactly for a smooth chart.  The 2h grid must
    itself be readable, so this needs at least seven points (four on the coarse
    grid); with fewer, the correction is refused rather than attempted.

    Removing bias amplifies variance -- the combination has a larger noise
    coefficient than the plug-in -- so the correction is worth taking only when
    the bias it removes, estimated by (A_h - A_2h)/3, is larger than the price's
    own noise.  Pass ``price_sd`` (from ``price_uncertainty``) and the report
    says which to use; the decision rule is measured in
    ``scripts/149_slope_bias_corrected.py``.
    """
    full = read_plane(xs, ls, t_lo, t_hi, undersize_c=undersize_c)
    out = {"price_h": full.get("price"), "price_2h": None,
           "price_corrected": None, "bias_estimate": None,
           "recommend": "plug-in", "price_recommended": full.get("price"),
           "note": ""}
    if len(xs) < 7 or not full.get("price"):
        out["note"] = ("fewer than seven readings: the 2h grid is too short to"
                       " read, so no correction is attempted")
        return out
    coarse = read_plane(xs[::2], ls[::2], t_lo, t_hi, undersize_c=undersize_c)
    if not coarse.get("price"):
        out["note"] = "the 2h grid is unreadable: no correction"
        return out
    out["price_2h"] = coarse["price"]
    out["price_corrected"] = (4.0 * full["price"] - coarse["price"]) / 3.0
    out["bias_estimate"] = (full["price"] - coarse["price"]) / 3.0
    if price_sd and price_sd > 0:
        if abs(out["bias_estimate"]) > price_sd:
            out["recommend"] = "corrected"
            out["price_recommended"] = out["price_corrected"]
        out["note"] = (f"Richardson: estimated h^2 bias"
                       f" {out['bias_estimate']:+.4f} against a noise sd of"
                       f" {price_sd:.4f}, so the {out['recommend']} price is"
                       f" the one to use")
    else:
        out["recommend"] = "corrected"
        out["price_recommended"] = out["price_corrected"]
        out["note"] = ("Richardson on the price: (4 A_h - A_2h)/3 removes the"
                       " leading h^2 term; no noise estimate was supplied, so"
                       " the variance it costs is not accounted for")
    return out


# ==========================================================================
#  F17 -- NOTHING DEGRADES SILENTLY
# ==========================================================================

def inputs_audit(models, grain=None, q0=None, n_test=None, items=None,
                 price_identified=None, q0_convention=None, tables=None):
    """F17.  One line per input, each saying what its absence costs.  The
    returned ``level`` is the strongest statement the inputs support.

    ``tables`` is the dict of 2x2 contingency tables, if any.  A 2x2 table
    fixes the joint failure counts exactly, so it buys the SAME exact-paired
    certificate a per-item vector does; auditing only ``items`` reported
    "per-item outcomes ABSENT" and downgraded the level on the ``--pairs``
    route even though the certificate actually used was exact-paired.  Fixed
    in revision 2026-10-08m; regression-tested by ``scripts/157``.
    """
    notes = []
    level = "CERTIFIED"
    if grain is None:
        notes.append(("grain", "NOT SUPPLIED",
                      "not needed for the envelope, the ladder, the price-free "
                      "test or the certified set; without it there is no "
                      "lambda*, no wall and no depth reading (Thm. "
                      "noinvariant)"))
    if q0 is None:
        notes.append(("q0", "NOT SUPPLIED",
                      "not an input of any decision: a constant shift of l "
                      "that cancels from every difference of sigma (verified "
                      "to 1e-14, F23). q0 = 0 is assumed for the floor and "
                      "depth readings, which are the only things that use it"))
    else:
        notes.append(("q0", q0_convention or "declared",
                      "a constant shift of l: it cancels from every derivative "
                      "and from every difference of sigma"))
    if items:
        notes.append(("per-item outcomes", "PRESENT",
                      "the gap is paired and the exclusion certificate is "
                      "distribution-free and finite-sample"))
    elif tables:
        notes.append(("2x2 tables", "PRESENT",
                      "the joint failure counts are fixed exactly, so the gap "
                      "is paired and the exclusion certificate is the same "
                      "distribution-free finite-sample one the per-item "
                      "vectors give -- no per-item file is needed"))
    elif n_test:
        notes.append(("per-item outcomes", "ABSENT",
                      "falling back to a normal certificate; without a "
                      "declared failure correlation the band is conservative"))
        level = "CERTIFIED (normal)"
    else:
        notes.append(("n_test", "MISSING",
                      "no resolution can be computed, so no gap is a gap and "
                      "no set carries coverage"))
        level = "UNCERTIFIED"
    if price_identified is False:
        notes.append(("price", "NOT IDENTIFIED",
                      "the derived price's interval spans more than its own "
                      "value: it is not used as if measured"))
        if level == "CERTIFIED":
            level = "CERTIFIED (price range only)"
    return {"level": level, "notes": notes}


def certified_schedule(models, axis, lam_lo, lam_hi, alpha: float = 0.10,
                       other_prices: dict = None, n_cells: int = 41,
                       **cert_kw):
    """F12/F14.  The decision as a function of price, each cell certified.

    When the price is not known within a doubling or two, no single network is
    optimal across the range and a single pick is the wrong object.  This
    partitions the price interval into cells, certifies a set in each cell at
    level ``alpha``, and returns the schedule together with

      ``n_networks``  the number of DISTINCT picks in the schedule: the minimum
                      number of networks that serves the whole price range.
                      This is the instance-level answer to "how few networks",
                      read off the candidates in hand with no corpus median.
      ``one_network`` the worst case, over the range, of serving it with the
                      single best compromise network instead -- what the
                      schedule buys.

    The guarantee is per cell: within a cell the price-optimal network is in
    that cell's set with probability at least 1 - alpha.
    """
    other_prices = dict(other_prices or {})
    lams = [lam_lo * (lam_hi / lam_lo) ** (a / (n_cells - 1.0))
            for a in range(n_cells)]
    rows = [{"name": m["name"], "S": -math.log(m["eps"]), "x": dict(m["x"])}
            for m in models]
    cells, picks = [], []
    for lam in lams:
        pr = dict(other_prices); pr[axis] = lam
        cs = certified_set(models, pr, price_range=None, alpha=alpha, **cert_kw)
        cells.append({"lambda": lam, "set": cs["set"], "pick": cs["pick"],
                      "method": cs["method"]})
        picks.append(cs["pick"])
    distinct = []
    for p in picks:
        if p is not None and p not in distinct:
            distinct.append(p)
    # what one compromise network would cost instead
    worst = {}
    for r in rows:
        w = 0.0
        for lam in lams:
            pr = dict(other_prices); pr[axis] = lam
            best = max(q["S"] - _costs(q, pr) for q in rows)
            w = max(w, best - (r["S"] - _costs(r, pr)))
        worst[r["name"]] = w
    one = min(worst.values()) if worst else None
    return {"cells": cells, "n_networks": len(distinct),
            "networks": distinct, "one_network_worst_case": one,
            "one_network_pick": (min(worst, key=worst.get) if worst else None),
            "alpha": alpha, "lam_lo": lam_lo, "lam_hi": lam_hi}


# ==========================================================================
#  A7 -- THE ENVELOPE THRESHOLD, CALIBRATED INSTEAD OF GUESSED
# ==========================================================================

def convex_minorant(xs, ls):
    """The greatest convex minorant of the points, by lower convex hull.  It is
    convex by construction and touches the data, so it is the natural IN-CLASS
    null: a frontier that obeys Law I exactly and is as close to these readings
    as a convex curve can be from below."""
    pts = sorted(zip(xs, ls))
    hull = []
    for x, y in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (x - x2) >= (y - y2) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append((x, y))
    out = []
    for x in xs:
        for (xa, ya), (xb, yb) in zip(hull, hull[1:]):
            if xa <= x <= xb:
                out.append(ya + (yb - ya) * (x - xa) / (xb - xa)
                           if xb > xa else ya)
                break
        else:
            out.append(hull[-1][1] if x > hull[-1][0] else hull[0][1])
    return out


def envelope_share_null(xs, ls, sd, draws: int = 400, q: float = 0.95,
                        wide_frac: float = 0.667, seed: int = 11):
    """A7.  The share of wide triples that violate convexity PURELY FROM NOISE,
    on a frontier of this shape, this grid and this precision.

    The 0.5 of the original ENVELOPE rule was a guess.  This replaces it with
    the ``q``-quantile of the null distribution of that share, simulated from
    the data's own convex minorant plus its own binomial noise -- an
    instance-level threshold, so a dense grid with few items is not called an
    envelope for being noisy, and a clean wide grid is not excused for being
    jagged.  Returns None when there is no noise estimate to simulate from.
    """
    if not sd or sd <= 0:
        return None
    g = convex_minorant(xs, ls)
    tri = _triples(xs, wide_frac)
    if not tri:
        return None
    rng = random.Random(seed)
    shares = []
    for _ in range(draws):
        pert = [v + rng.gauss(0.0, sd) for v in g]
        bad = 0
        for (i, j, k) in tri:
            if _dd2([xs[i], xs[j], xs[k]], [pert[i], pert[j], pert[k]]) < 0.0:
                bad += 1
        shares.append(bad / float(len(tri)))
    shares.sort()
    return shares[min(len(shares) - 1, max(0, int(q * len(shares))))]


# ==========================================================================
#  F18 -- PRICE-FREE DOMINATION
# ==========================================================================
#
# One case needs no price at all.  If candidate i is certified to be no worse
# in accuracy AND is no dearer on every priced axis, it beats j for EVERY
# non-negative price vector: the comparison is Pareto, not scalarized, and the
# whole apparatus of the grain, the plane and the derived price is unnecessary.
#
# This is the case a batch-size or throughput study produces, where the axis
# moves cost without moving accuracy, and it is the first thing the selector
# should try: it is the strongest statement available and the cheapest.

def certified_pareto(models, alpha: float = 0.10, items: dict = None,
                     tables: dict = None, n_test: int = None,
                     rho: float = None, axes=None):
    """F18.  Pairs (i, j) for which i is certified to dominate j at every
    non-negative price: not dearer on any axis, and certified no worse on
    accuracy by the same bound as F12 with the cost difference set to zero.

    Returns a dict with ``dominations`` (list of (winner, loser)), the
    ``dominated`` set, and the ``survivors`` -- the certified Pareto set, which
    needs no price, no grain and no calibration curve.
    """
    rows = [{"name": m["name"], "eps": m["eps"], "x": dict(m["x"])}
            for m in models]
    if axes is None:
        axes = sorted({k for r in rows for k in r["x"]})
    K = len(rows)
    a_pair = alpha / max(K - 1.0, 1.0)
    z = _z_of(1.0 - a_pair)
    doms = []
    for i in rows:
        for j in rows:
            if i["name"] == j["name"]:
                continue
            # i must be no dearer on every axis
            if any(i["x"].get(ax, 0.0) > j["x"].get(ax, 0.0) for ax in axes):
                continue
            ok = False
            if tables or items:
                tb = None
                if tables:
                    tb = tables.get((i["name"], j["name"]))
                    if tb is None and tables.get((j["name"], i["name"])):
                        a2, b2, c2, d2 = tables[(j["name"], i["name"])]
                        tb = (a2, c2, b2, d2)
                if tb is None and items:
                    fi, fj = items.get(i["name"]), items.get(j["name"])
                    if fi is not None and fj is not None:
                        tb = contingency(fi, fj)
                if tb is not None:
                    ok, _lb, _m = certify_domination_counts(tb, 0.0, a_pair)
            elif n_test:
                sd = sd_of_gap(i["eps"], j["eps"], n_test, rho=rho)
                ok = sd > 0 and (math.log(j["eps"] / i["eps"]) / sd) > z
            if ok:
                doms.append((i["name"], j["name"]))
    dominated = {j for _i, j in doms}
    return {"dominations": doms, "dominated": sorted(dominated),
            "survivors": [r["name"] for r in rows
                          if r["name"] not in dominated],
            "alpha": alpha, "alpha_pair": a_pair,
            "basis": ("no price, no grain, no calibration curve: Pareto "
                      "domination certified on accuracy alone")}


# ==========================================================================
#  F19 -- THE CERTIFIED RANKING
# ==========================================================================
#
# "Rank all the candidates by effectiveness" has an honest answer and a
# dishonest one.  The dishonest one is a total order on sigma, which invents
# distinctions the data cannot support: 13.8 consecutive models sit inside one
# indifference band on a median published frontier.  The honest one is a
# sequence of TIERS -- within a tier no difference is certifiable, between
# tiers every difference is -- which is the partial order the certificates
# actually establish, laid out in layers.

def certified_ranking(models, prices, price_range=None, alpha: float = 0.10,
                      items: dict = None, tables: dict = None,
                      n_test: int = None, rho: float = None, n_grid: int = 41):
    """F19.  Rank the candidates into certified tiers.

    Tier 1 is the set of candidates nothing is certified to beat; tier 2 is the
    same after removing tier 1, and so on.  Everything in tier k is certified
    worse than something in tier k-1, and nothing inside a tier is separable at
    this item count and this price interval.

    Returns the tiers, the pairwise relation, any contradictory pairs (both
    directions certified -- an error of probability at most alpha/(K-1) per
    pair, reported rather than hidden), and the per-candidate worst-case regret
    over the price interval.
    """
    rows = [{"name": m["name"], "eps": m["eps"], "x": dict(m["x"]),
             "S": -math.log(m["eps"])} for m in models]
    K = len(rows)
    a_pair = alpha / max(K - 1.0, 1.0)
    z = _z_of(1.0 - a_pair) if not (items or tables) else None
    grids = _price_grid(prices, price_range, n_grid)

    beats = {r["name"]: set() for r in rows}
    for i in rows:
        for j in rows:
            if i["name"] == j["name"]:
                continue
            worst = max(grids, key=lambda pr: _costs(i, pr) - _costs(j, pr))
            dcost = _costs(i, worst) - _costs(j, worst)
            ok = False
            if items or tables:
                tb = None
                if tables:
                    tb = tables.get((i["name"], j["name"]))
                    if tb is None and tables.get((j["name"], i["name"])):
                        a2, b2, c2, d2 = tables[(j["name"], i["name"])]
                        tb = (a2, c2, b2, d2)
                if tb is None and items:
                    fi, fj = items.get(i["name"]), items.get(j["name"])
                    if fi is not None and fj is not None:
                        tb = contingency(fi, fj)
                if tb is not None:
                    ok, _lb, _m = certify_domination_counts(tb, dcost, a_pair)
            elif n_test:
                sd = sd_of_gap(i["eps"], j["eps"], n_test, rho=rho)
                ds = (i["S"] - dcost) - j["S"]
                ok = sd > 0 and (ds / sd) > z
            if ok:
                beats[i["name"]].add(j["name"])

    contradictions = sorted({tuple(sorted((a, b)))
                             for a in beats for b in beats[a]
                             if a in beats.get(b, ())})
    for a, b in contradictions:                 # a certificate must be wrong
        beats[a].discard(b)
        beats[b].discard(a)

    # layer the partial order: tier 1 = nothing beats it
    remaining = {r["name"] for r in rows}
    tiers = []
    while remaining:
        top = [nm for nm in remaining
               if not any(nm in beats[o] for o in remaining if o != nm)]
        if not top:                              # should not happen; be safe
            top = sorted(remaining)
        tiers.append(sorted(top, key=lambda nm: -next(
            r["S"] - _costs(r, prices) for r in rows if r["name"] == nm)))
        remaining -= set(top)

    worst = {}
    for r in rows:
        w = 0.0
        for pr in grids:
            best = max(q["S"] - _costs(q, pr) for q in rows)
            w = max(w, best - (r["S"] - _costs(r, pr)))
        worst[r["name"]] = w
    return {"tiers": tiers, "beats": {k: sorted(v) for k, v in beats.items()},
            "contradictions": contradictions, "worst_case_regret": worst,
            "alpha": alpha, "alpha_pair": a_pair, "n_tiers": len(tiers),
            "method": ("exact-paired" if (items or tables) else
                       ("normal-paired" if rho is not None else
                        "normal-unpaired"))}


# ==========================================================================
#  F20 -- SELECTION-GRADE DESIGN: how sharp must the derived price be?
# ==========================================================================
#
# F13 asks how wide the calibration grid must be for kappa to be measurable to
# a relative tolerance.  That is the wrong target for a selector, and a much
# stricter one.  What a selector needs is that the derived price's own interval
# be narrower than the interval of prices over which the WINNER DOES NOT
# CHANGE -- a candidate's "reign".  Reigns are a property of the candidates in
# hand and cost nothing to measure, so this is the design rule a deployment
# should actually check.
#
# Measured on the archive (table_selection_grade_summary.csv, 36 frontiers
# with a public item count): the derived price's 90% interval is a median 4.92
# doublings wide against a median reign of 1.00 doublings; the median
# sharpening factor is 4.04 over all 36 (the 7 already sharp enough entering at
# 1.0) and 6.49 over the 29 that are not.  Since the operating price is
# lambda* = sqrt(kappa/t),
# sd(lambda)/lambda = (1/2) sd(kappa)/kappa and sd(kappa) = sqrt(6) sd(l)/h^2,
# so a factor F of sharpening is sqrt(F) in grid width and F^2 in items:
# sqrt(6.49) = 2.55x wider -- a five-point curve spanning about 4.7x in
# resource -- or 42x more items.  The 853x span of F13 is for kappa itself
# (table_selection_grade_summary.median_span_needed_neff_F13); the 110x that
# appeared here in earlier revisions is F13's PRICE target, not kappa's.

def reign_widths(models, prices, axis, lam_lo, lam_hi, n_grid: int = 241):
    """The widths, in doublings of price, of the intervals over which each
    candidate is the argmax of sigma.  Measured on the candidates in hand."""
    rows = [{"name": m["name"], "x": dict(m["x"]), "S": -math.log(m["eps"])}
            for m in models]
    if not (lam_hi > lam_lo > 0):
        return []
    lams = [lam_lo * (lam_hi / lam_lo) ** (k / (n_grid - 1.0))
            for k in range(n_grid)]
    win = []
    for lam in lams:
        pr = dict(prices); pr[axis] = lam
        win.append(max(rows, key=lambda r: r["S"] - _costs(r, pr))["name"])
    out, cur, start = [], win[0], 0
    ln2 = math.log(2.0)
    for k in range(1, n_grid):
        if win[k] != cur:
            out.append(math.log(lams[k - 1] / lams[start]) / ln2)
            cur, start = win[k], k
    out.append(math.log(lams[-1] / lams[start]) / ln2)
    return [w for w in out if w > 0]


def design_for_selection(models, prices, axis, price_lo, price_hi,
                         lam_lo=None, lam_hi=None, h_available=None,
                         n_grid: int = 241):
    """F20.  Is the derived price sharp enough to name a single candidate?

    ``price_lo``/``price_hi`` are the derived price's own confidence interval;
    ``lam_lo``/``lam_hi`` the range of prices to scan for reigns (default: the
    interval itself, widened to the span the candidates support).  Returns the
    verdict and, where it fails, the factor by which the grid must widen and
    the factor by which the items must grow.
    """
    ln2 = math.log(2.0)
    if not (price_hi > price_lo > 0):
        return {"verdict": "UNDETERMINED",
                "note": "no price interval to judge"}
    if lam_lo is None or lam_hi is None:
        lam_lo, lam_hi = price_lo, price_hi
    reigns = reign_widths(models, prices, axis, lam_lo, lam_hi, n_grid)
    interval = math.log(price_hi / price_lo) / ln2
    out = {"price_interval_doublings": interval,
           "n_reigns_in_interval": len(reigns),
           "median_reign_doublings": (sorted(reigns)[len(reigns) // 2]
                                      if reigns else None),
           "verdict": None, "sharpen_by": None,
           "widen_grid_by": None, "more_items_by": None, "note": ""}
    reign = out["median_reign_doublings"]
    if reign is None or reign <= 0:
        out["verdict"] = "FEASIBLE"
        out["note"] = ("one candidate is optimal across the whole price "
                       "interval: the price is already sharp enough to decide")
        return out
    out["sharpen_by"] = interval / reign
    if out["sharpen_by"] <= 1.0:
        out["verdict"] = "FEASIBLE"
        out["note"] = ("the price interval is narrower than a candidate's "
                       "reign, so it names one candidate")
    else:
        out["verdict"] = "INFEASIBLE"
        out["widen_grid_by"] = math.sqrt(out["sharpen_by"])
        out["more_items_by"] = out["sharpen_by"] ** 2
        out["note"] = (
            f"the price interval spans {interval:.2f} doublings against a "
            f"median reign of {reign:.2f}: sharpen by "
            f"{out['sharpen_by']:.1f}x, which is a "
            f"{math.sqrt(out['sharpen_by']):.1f}x wider sweep or "
            f"{out['sharpen_by']**2:.0f}x more items. The sweep is the cheap "
            f"route: lambda* = sqrt(kappa/t), so sd(lambda)/lambda goes as "
            f"sd(l)/h^2")
        if h_available:
            out["h_needed"] = h_available * out["widen_grid_by"]
            out["span_needed"] = math.exp(4.0 * out["h_needed"])
    return out


# ==========================================================================
#  F21 -- THE CERTIFIED BREAK-EVEN PRICE
# ==========================================================================
#
# This is the smallest useful statement the whole apparatus can make, and it
# needs NO sweep, NO grain, NO derived price and NO Law I.  Two candidates, one
# evaluation each, the per-item outcomes behind them:
#
#     sigma_D > sigma_C  <=>  ln(eps_C/eps_D) > lambda (x_D - x_C)
#                        <=>  lambda < lambda_be := ln(eps_C/eps_D)/(x_D-x_C),
#
# so there is a single BREAK-EVEN price at which the dearer candidate stops
# being worth it.  Certifying it is F12 run as a function of lambda: the
# certificate for "the dearer wins" is monotone decreasing in lambda and the
# one for "the cheaper wins" is monotone increasing, so a bisection returns an
# interval
#
#     lambda < lambda_lo   the DEARER candidate is certifiably better
#     lambda > lambda_hi   the CHEAPER candidate is certifiably better
#     in between           not separable at this item count
#
# at total level alpha.  A deployment then needs only to know which side of
# that interval its own willingness to pay falls on -- a question it can answer
# without the theory.  The theory's contribution is to supply lambda itself,
# lambda* = sqrt(kappa/t), when nobody wants to declare one.

def _dear_wins_lb(tb_DC, dx, lam, alpha):
    """Lower bound for 'the dearer candidate wins' at price ``lam``.
    ``tb_DC`` is the 2x2 table with i = dear, j = cheap."""
    _ok, lb, _m = certify_domination_counts(tb_DC, lam * dx, alpha)
    return lb


def _cheap_wins_lb(tb_CD, dx, lam, alpha):
    """Lower bound for 'the cheaper candidate wins' at price ``lam``.
    ``tb_CD`` is the 2x2 table with i = cheap, j = dear."""
    _ok, lb, _m = certify_domination_counts(tb_CD, -lam * dx, alpha)
    return lb


def _bisect_sign_change(f, lo, hi, want_positive_at, iters: int = 60):
    """Largest (or smallest) lambda at which ``f`` is still positive.

    ``want_positive_at`` is "lo" when f is decreasing in lambda and "hi" when
    it is increasing.  Returns None when f has no sign change in [lo, hi].
    """
    flo, fhi = f(lo), f(hi)
    if want_positive_at == "lo":
        if flo <= 0:
            return None                      # never certified, even at lam->0
        if fhi > 0:
            return hi                        # certified everywhere in range
    else:
        if fhi <= 0:
            return None
        if flo > 0:
            return lo
    a, b = lo, hi
    for _ in range(iters):
        m = math.sqrt(a * b) if (a > 0 and b > 0) else 0.5 * (a + b)
        if (f(m) > 0) == (want_positive_at == "lo"):
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def certified_break_even(dear, cheap, axis, alpha: float = 0.10,
                         items: dict = None, tables: dict = None,
                         n_test: int = None, rho: float = None,
                         lam_lo: float = 1e-4, lam_hi: float = None):
    """F21.  The price interval inside which two candidates cannot be
    separated, and outside which one of them is certified better.

    ``dear`` and ``cheap`` are model dicts as in ``select``; ``cheap`` must
    have the smaller log-resource on ``axis``.  Returns the point break-even,
    the certified interval, and what to do on each side.
    """
    dx = dear["x"].get(axis, 0.0) - cheap["x"].get(axis, 0.0)
    out = {"dear": dear["name"], "cheap": cheap["name"], "axis": axis,
           "delta_log_resource": dx, "break_even": None,
           "certified_lo": None, "certified_hi": None,
           "undecided_doublings": None, "method": None, "note": ""}
    if dx <= 0:
        out["note"] = ("the 'dear' candidate is not dearer on this axis: this "
                       "is a price-free comparison (F18), not a break-even")
        return out
    if lam_hi is None:
        lam_hi = 40.0 / dx                 # keeps lambda*dx inside the clamp
    out["break_even"] = math.log(cheap["eps"] / dear["eps"]) / dx
    out["lambda_searched"] = (lam_lo, lam_hi)

    tb_DC = None
    if tables:
        tb_DC = tables.get((dear["name"], cheap["name"]))
        if tb_DC is None and tables.get((cheap["name"], dear["name"])):
            a2, b2, c2, d2 = tables[(cheap["name"], dear["name"])]
            tb_DC = (a2, c2, b2, d2)
    if tb_DC is None and items:
        fi, fj = items.get(dear["name"]), items.get(cheap["name"])
        if fi is not None and fj is not None:
            tb_DC = contingency(fi, fj)

    if tb_DC is not None:
        out["method"] = "exact-paired"
        a_, b_, c_, d_ = tb_DC
        tb_CD = (a_, c_, b_, d_)
        out["certified_lo"] = _bisect_sign_change(
            lambda lam: _dear_wins_lb(tb_DC, dx, lam, alpha / 2.0),
            lam_lo, lam_hi, "lo")
        out["certified_hi"] = _bisect_sign_change(
            lambda lam: _cheap_wins_lb(tb_CD, dx, lam, alpha / 2.0),
            lam_lo, lam_hi, "hi")
    elif n_test:
        out["method"] = ("normal-paired" if rho is not None
                         else "normal-unpaired")
        z = _z_of(1.0 - alpha / 2.0)
        sd = sd_of_gap(dear["eps"], cheap["eps"], n_test, rho=rho) / dx
        out["certified_lo"] = out["break_even"] - z * sd
        out["certified_hi"] = out["break_even"] + z * sd
        if out["certified_lo"] <= 0:
            out["certified_lo"] = None
    else:
        out["method"] = "uncertified"
        out["note"] = "no n_test and no per-item data: no interval"
        return out

    lo, hi = out["certified_lo"], out["certified_hi"]
    if lo and hi and hi > lo > 0:
        out["undecided_doublings"] = math.log(hi / lo) / math.log(2.0)
    out["note"] = (
        f"buy {dear['name']} if your price is below "
        f"{lo:.4f} nats/e-fold" if lo else
        f"{dear['name']} is never certifiably worth it at these items")
    if hi:
        out["note"] += (f"; take {cheap['name']} above {hi:.4f}")
    return out


def break_even_chain(models, axis, alpha: float = 0.10, items: dict = None,
                     tables: dict = None, n_test: int = None,
                     rho: float = None):
    """F21.  The break-even prices of every adjacent pair, cheapest first.

    Read downwards it is a price ladder: each rung is the price at which the
    next model up stops paying for itself.  This is the decision for K
    candidates with no sweep at all -- the only thing left to supply is which
    rung you stand on.
    """
    rows = sorted(models, key=lambda m: m["x"].get(axis, 0.0))
    out = []
    for a, b in zip(rows, rows[1:]):
        out.append(certified_break_even(b, a, axis, alpha=alpha, items=items,
                                        tables=tables, n_test=n_test,
                                        rho=rho))
    return out


# ==========================================================================
#  F22 -- THE PRICE ENVELOPE: who can ever be optimal, without knowing the
#         Pareto frontier
# ==========================================================================
#
# The break-even identity of F21 is algebra on two measured points and holds
# whether or not either point lies on any frontier: sigma = S - lambda x is a
# function of (eps, x) alone, with no appeal to Law I, convexity or an
# envelope.  What does NOT survive off-frontier candidates is chaining
# break-evens between COST NEIGHBOURS, because a candidate that is dominated,
# or merely sits in a dent of the measured staircase, breaks the chain.
#
# The fix needs no knowledge of which candidates are efficient.  Each candidate
# contributes a straight line in the price,
#
#       sigma_i(lambda) = S_i - lambda x_i ,
#
# so the best achievable score is the UPPER ENVELOPE of K lines: a convex
# piecewise-linear function whose pieces name the winner at each price and
# whose breakpoints are exactly the break-evens that matter.  Computing it is
# one sort and one stack pass, and it answers the question directly -- the
# candidates that appear on it are the ones that can be optimal at some price,
# and every other candidate is provably never the answer, whatever the price.
#
# That set is STRICTER than the Pareto set.  A Pareto-efficient candidate lying
# in a dent of the staircase maximises sigma for no lambda at all.  Measured on
# the archive's own (already Pareto-filtered) frontiers, a median of 62.5% of
# Pareto points are never optimal at any price: the median frontier of 11
# Pareto models has 5 that a linear price can ever select.

def price_envelope(models, axis, lam_lo: float = 1e-6, lam_hi: float = 1e6):
    """F22.  The upper envelope of sigma_i(lambda) = S_i - lambda x_i.

    Returns a dict with
      ``pieces``        [(name, lambda_from, lambda_to)] in ascending price:
                        who is optimal where, for the whole positive price axis
      ``winners``       the candidates that are optimal somewhere
      ``never``        those that are optimal nowhere, split into
      ``dominated``     (another candidate is both cheaper and more accurate)
      ``in_a_dent``     (Pareto-efficient, but no linear price selects it)
      ``breakpoints``   the break-even prices between successive winners

    No frontier has to be identified first, and nothing here assumes Law I.
    """
    lines = []
    for m in models:
        lines.append((-m["x"].get(axis, 0.0), -math.log(m["eps"]), m["name"]))
    lines.sort(key=lambda t: (t[0], t[1]))
    ded = []
    for b, a, n in lines:                      # one line per slope, best one
        if ded and abs(ded[-1][0] - b) < 1e-15:
            if a > ded[-1][1]:
                ded[-1] = (b, a, n)
        else:
            ded.append((b, a, n))

    def cross(p, q):
        return (q[1] - p[1]) / (p[0] - q[0])

    hull = []
    for ln in ded:
        while hull:
            if hull[-1][0] == ln[0]:
                hull.pop(); continue
            if len(hull) >= 2 and cross(hull[-1], ln) <= cross(hull[-2],
                                                               hull[-1]):
                hull.pop(); continue
            break
        hull.append(ln)
    bounds = [cross(hull[i], hull[i + 1]) for i in range(len(hull) - 1)]
    pieces, lo = [], lam_lo
    for i, ln in enumerate(hull):
        hi = bounds[i] if i < len(bounds) else lam_hi
        if hi > lo:
            pieces.append((ln[2], lo, hi))
        lo = hi
    winners = [p[0] for p in pieces]
    wset = set(winners)

    rows = [{"name": m["name"], "eps": m["eps"],
             "x": m["x"].get(axis, 0.0)} for m in models]
    dominated, dent = [], []
    for r in rows:
        if r["name"] in wset:
            continue
        if any(o["eps"] <= r["eps"] and o["x"] <= r["x"]
               and (o["eps"] < r["eps"] or o["x"] < r["x"])
               for o in rows if o["name"] != r["name"]):
            dominated.append(r["name"])
        else:
            dent.append(r["name"])
    return {"pieces": pieces, "winners": winners,
            "never": sorted(dominated + dent),
            "dominated": sorted(dominated), "in_a_dent": sorted(dent),
            "breakpoints": bounds, "axis": axis,
            "n_candidates": len(rows), "n_winners": len(wset),
            "note": ("the winners are the candidates a linear price can ever "
                     "select; this is stricter than the Pareto set, and it is "
                     "computed without identifying any frontier")}


def break_even_envelope(models, axis, alpha: float = 0.10, items: dict = None,
                        tables: dict = None, n_test: int = None,
                        rho: float = None):
    """F21 + F22.  The price ladder built on the ENVELOPE rather than on cost
    neighbours: one rung per breakpoint, between the two candidates that are
    actually optimal on either side of it, each with its certified interval.

    This is the ladder to read when it is not known which candidates are
    efficient -- which is always.
    """
    env = price_envelope(models, axis)
    by = {m["name"]: m for m in models}
    rungs = []
    for (n1, _l1, h1), (n2, l2, _h2) in zip(env["pieces"], env["pieces"][1:]):
        # ascending price: n1 is the dearer winner below the breakpoint
        dear, cheap = by[n1], by[n2]
        if dear["x"].get(axis, 0.0) < cheap["x"].get(axis, 0.0):
            dear, cheap = cheap, dear
        r = certified_break_even(dear, cheap, axis, alpha=alpha, items=items,
                                 tables=tables, n_test=n_test, rho=rho)
        r["breakpoint"] = h1
        rungs.append(r)
    return {"envelope": env, "rungs": rungs}


# ==========================================================================
#  F23 -- MEASUREMENT-ONLY MODE: no price, no grain, no chance level
# ==========================================================================
#
# Three of the selector's inputs turn out not to be inputs of the DECISION at
# all, and the tool should not ask for them.
#
#   q0.  With eps = (1-Q)/(1-q0), ln eps = ln(1-Q) - ln(1-q0), so q0 is a
#        constant shift of l and cancels identically from every difference of
#        sigma, from every break-even and from the whole envelope.  Verified
#        numerically over q0 in {0, 0.001, 0.1, 0.5}: the sigma differences and
#        the break-even prices agree to ten decimals.  q0 moves only the floor
#        and the depth readings.
#   the grain.  It enters only through lambda* = sqrt(kappa/t), the operating
#        point and the wall/depth bounds.  Nothing in the envelope, the ladder
#        or the price-free test refers to it.
#   the price.  The envelope is the max over candidates of S_i - lambda x_i as
#        a FUNCTION of lambda, so it is computed without knowing lambda; the
#        break-even rungs are where that function changes winner; and the
#        price-free test is the rung at lambda = 0.
#
# So a complete, certified answer exists from the candidates' own measurements
# alone: accuracy, the resource, n_test, and (for the exact certificate) the
# per-item outcomes.  This function returns it.

def tiers_from_relation(names, beats):
    """Layer a strict partial order into tiers: tier 1 is what nothing beats.

    The iteration runs over ``names`` in the order given, never over a set, so
    the result is deterministic: iterating a set of strings would make the
    reported order depend on the process's hash seed, which
    ``scripts/157_input_routes.py`` is there to catch.
    """
    order = list(names)
    remaining, tiers = set(order), []
    while remaining:
        top = [n for n in order
               if n in remaining
               and not any(n in beats.get(o, ()) for o in order
                           if o in remaining and o != n)]
        if not top:
            top = [n for n in order if n in remaining]
        tiers.append(top)
        remaining -= set(top)
    return tiers


def decide_from_measurements(models, axis, alpha: float = 0.10,
                             items: dict = None, tables: dict = None,
                             n_test: int = None, rho: float = None):
    """F23.  Everything the candidates' own measurements support, and nothing
    that needs an input they do not contain.

    Needs only: ``eps`` and ``x[axis]`` per candidate, ``n_test`` (or the
    per-item outcomes) for the certificates.  No price, no grain, no q0.

    Returns the envelope, the certified ladder at its breakpoints, the
    price-free dominations, and the ranking those two induce: tiers from the
    price-free partial order, ordered inside a tier by where each candidate is
    optimal.
    """
    env = price_envelope(models, axis)
    bee = break_even_envelope(models, axis, alpha=alpha, items=items,
                              tables=tables, n_test=n_test, rho=rho)
    pf = certified_pareto(models, alpha=alpha, items=items, tables=tables,
                          n_test=n_test, rho=rho, axes=[axis])
    beats = {}
    for w, l in pf["dominations"]:
        beats.setdefault(w, set()).add(l)
    names = [m["name"] for m in models]
    tiers = tiers_from_relation(names, beats)
    win = {nm: (lo, hi) for nm, lo, hi in env["pieces"]}
    # inside a tier, order by where the candidate is optimal (dearest first),
    # and put the never-optimal ones last: that is an ordering by evidence, not
    # by an invented score
    ordered = []
    for t in tiers:
        # deterministic: the envelope position first, then the error, then the
        # name -- never an accident of input order
        eps_of_name = {m["name"]: m["eps"] for m in models}
        ordered.append(sorted(
            t, key=lambda nm: (0 if nm in win else 1,
                               win.get(nm, (0.0, 0.0))[0],
                               eps_of_name.get(nm, 0.0), str(nm))))
    return {"envelope": env, "rungs": bee["rungs"],
            "price_free": pf, "tiers": ordered,
            "optimal_interval": win,
            "basis": ("the candidates' own measurements only: no price, no "
                      "grain, no chance level")}


# ==========================================================================
#  F24 -- THE INPUT CONSISTENCY GATE
# ==========================================================================
#
# The certified interval of F21 is computed from a pair's 2x2 table; the point
# break-even is computed from the two accuracies.  If those two inputs come
# from different places -- a stale aggregate file beside a fresh table, a
# mistyped accuracy on the command line -- the tool will happily print a point
# that lies outside its own certified interval.  That is not a theory problem
# and it is not detectable downstream, so it is checked at the door.
#
# Nothing has to be assumed: a 2x2 table determines both marginal error rates
# and the item count exactly, and an item vector determines the accuracy
# exactly.  The declared values must match to within the rounding of the
# accuracy as written.

def _declared_failure_rate(m):
    """The RAW failure rate a declared accuracy implies.

    F24 compares declared aggregates against counted ones.  An item vector and
    a 2x2 table both count RAW failures, so the declared side must be the raw
    rate 1 - accuracy and NOT m["eps"] = (1 - accuracy)/(1 - q0), which carries
    the chance level.  Comparing the two scales made every run with --q0 > 0
    refuse its own consistent inputs by exactly the factor 1/(1 - q0), which
    also made q0 an input that could veto a decision -- the one thing F23 says
    it never is.  Fixed in revision 2026-10-08m.
    """
    if m.get("accuracy") is not None:
        return 1.0 - float(m["accuracy"])
    return float(m["eps"])          # q0 = 0, so the two agree


def check_consistency(models, tables: dict = None, items: dict = None):
    """F24.  Cross-check the declared accuracies and item counts against any
    supplied 2x2 tables and per-item vectors.  Returns a list of strings, one
    per disagreement; an empty list is a clean bill.

    The comparison is on RAW failure rates on both sides, so it is invariant to
    the chance level q0: q0 rescales every eps by the same factor and cancels
    from this gate exactly as it cancels from every decision (F23).
    """
    out = []
    by = {m["name"]: m for m in models}
    if items:
        for nm, fails in items.items():
            m = by.get(nm)
            if m is None:
                continue
            n = len(fails)
            e_counted = sum(1 for f in fails if f) / float(n)
            if m.get("n_test") and m["n_test"] != n:
                out.append(f"{nm}: n_test declared {m['n_test']:,} but the "
                           f"item vector has {n:,} entries")
            tol = 1.5 / n + 1e-9
            e_decl = _declared_failure_rate(m)
            if abs(e_decl - e_counted) > tol:
                out.append(
                    f"{nm}: accuracy implies a failure rate of {e_decl:.6f} "
                    f"but the item vector counts {e_counted:.6f} "
                    f"({abs(e_decl-e_counted)*n:.0f} items apart)")
    if tables:
        seen = set()
        for (ni, nj), tb in tables.items():
            key = tuple(sorted((str(ni), str(nj))))
            if key in seen:
                continue
            seen.add(key)
            mi, mj = by.get(ni), by.get(nj)
            if mi is None or mj is None:
                continue
            a, b, c, d = tb
            n = a + b + c + d
            if n <= 0:
                out.append(f"{ni}/{nj}: the 2x2 table sums to zero")
                continue
            tol = 1.5 / n + 1e-9
            for m, e_tab in ((mi, (a + b) / float(n)), (mj, (a + c) / float(n))):
                if m.get("n_test") and m["n_test"] != n:
                    out.append(
                        f"{m['name']}: n_test declared {m['n_test']:,} but the "
                        f"{ni}/{nj} table sums to {n:,}")
                e_decl = _declared_failure_rate(m)
                if abs(e_decl - e_tab) > tol:
                    out.append(
                        f"{m['name']}: accuracy implies a failure rate of "
                        f"{e_decl:.6f} but the {ni}/{nj} table implies "
                        f"{e_tab:.6f} "
                        f"({abs(e_decl-e_tab)*n:.0f} items apart)")
    # de-duplicate, keep order
    seen2, uniq = set(), []
    for s in out:
        if s not in seen2:
            seen2.add(s); uniq.append(s)
    return uniq

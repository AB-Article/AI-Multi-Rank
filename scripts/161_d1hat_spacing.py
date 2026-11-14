#!/usr/bin/env python3
"""The plug-in rung D1hat: what node spacing does to it, measured.

Why this script exists
----------------------
Revision 2026-10-09 (round 13) found that the paper asserted of

    D1hat = 1.5 * dd1 * dd3 / dd2^2 - 1      (eq:d1hat)

that it is "exact at any spacing", and attributed the ~50% rate at which its
SIGN refutes a divisible truth to measurement noise ("at every noise level").
Both halves are wrong, and the second is wrong in a way that matters for how
the statistic should be read on this archive.

D1hat is a PLUG-IN for the invariant D_1 = s * S''' / kappa^2 - 1.  Each
divided difference is a derivative at its own mean-value node,
dd_k = S^(k)(xi_k)/k!, with a DIFFERENT xi_k per order, so the ratio is not
D_1 at any finite spacing.  Two separate facts follow.

  (a) It is exact on the cure chart at EQUAL spacing, and only there.  For
      S = Lam (1 - e^{-c x}) on nodes x0 + j h one has
      dd_k = A (q-1)^k / (k! h^k) with q = e^{-ch}, so
      1.5 * dd1 * dd3 / dd2^2 = 1.5 * (4/6) = 1 and D1hat = 0 identically, for
      every h and every (Lam, c).  That is the extremal of the rung, which is
      why the cure chart saturates it.

  (b) Away from the cure chart, or away from equal spacing, it is biased.  On a
      Gamma(2) truth S = 2 ln(1+x), where the exact D_1 = 1, an equally spaced
      quadruple of span 3 returns 0.42 and one of span 9 returns 0.27.  And on a divisible truth at ZERO
      noise, unequal nodes drive the sign negative, i.e. refute a truth that
      satisfies the rung exactly.

The reason (b) is not a footnote: the archive's quadruples are never equally
spaced.  quad_indices (script 130) takes the widest four-point subset at index
thirds of the measured grid of LOG-resource, and published model sets are not
geometric in the resource -- models cost whatever they happen to cost -- so the
chosen nodes are irregular.  (Were the resource grid geometric, the log-resource
nodes WOULD be equally spaced and this whole effect would vanish.)

What is tested
--------------
T1  Exactness on the cure chart at equal spacing, and its loss off it:
    D1hat on a cure chart (exact D_1 = 0) and on Gamma(2) (exact D_1 = 1),
    at equal spacing over four spans and at four unequal node sets.

T2  The archive's own node geometry: for each of the 117 frontiers, the gap
    ratio max(dx)/min(dx) of the quadruple that quad_indices selects.

T3  The zero-noise false-refutation rate attributable to SPACING ALONE.  On
    each frontier's own quadruple x-values, plant a divisible truth, evaluate S
    with NO noise added, and record the sign of D1hat.  Two truths are planted
    and the distinction between them is the point:

      - the cure chart, which SATURATES the rung (exact D_1 = 0).  Here the
        sign is a coin flip by construction, at any spacing and at any noise
        level, because the truth sits exactly on the boundary.  That is the
        paper's own second honest negative, and it is not a spacing effect.

      - compound-Poisson mixtures of 2-4 atoms, kept to those whose exact
        D_1 > 0.05, so they are STRICTLY INTERIOR with margin to spare.  A
        negative here refutes a truth the rung does not bind, and at zero
        noise the only thing left that can produce it is the node geometry.

T4  The same counts on equally spaced nodes spanning the same interval, with
    the same planted truths, as the control that isolates spacing from
    everything else about the frontier.

Writes results/table_d1hat_spacing.csv
       results/table_d1hat_archive_nodes.csv
       results/table_d1hat_spacing_summary.csv
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import save                                          # noqa: E402
import _loaders as L                                             # noqa: E402

RNG = np.random.default_rng(20261009)
V130 = L._module("130_plane_verdict.py", "v130_for_161")
quad_indices = V130.quad_indices


def stable_seed(name, mod=2 ** 31):
    """A per-frontier seed that does not move between interpreter processes.

    Python's built-in hash() of a str is salted per process (PYTHONHASHSEED),
    and the archive's run convention `python3 -I` implies -E, so the salt
    cannot be pinned from the environment.  Seeding from hash() therefore made
    this script's output change from run to run; sha256 of the UTF-8 name is
    stable across processes, platforms and Python versions.  Round 13.
    """
    import hashlib
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:16], 16) % mod


# ------------------------------------------------------------- the statistic
def dd_chain(xq, Sq):
    """(dd1, dd2, dd3) by the Newton table on four nodes."""
    x = np.asarray(xq, float); t = np.asarray(Sq, float)
    out = []
    for k in range(1, 4):
        t = (t[1:] - t[:-1]) / (x[k:] - x[:-k])
        out.append(t[0])
    return out


TOL = 1e-9        # D1hat is 0 exactly on the saturating chart; below this is roundoff


def d1_exact(S, x, h=1e-4):
    # The invariant D_1 = s S''' / kappa^2 - 1, by central differences.
    # (eq:d1hat is exactly this with s = dd1, kappa = -2 dd2, S''' = 6 dd3,
    #  which is why the plug-in carries the factor 3/2 and the invariant does not.)
    g = lambda t: float(np.asarray(S(np.asarray([t], float)), float).ravel()[0])
    s1 = (g(x + h) - g(x - h)) / (2 * h)
    s2 = (g(x + h) - 2 * g(x) + g(x - h)) / h ** 2
    s3 = (g(x + 2 * h) - 2 * g(x + h) + 2 * g(x - h) - g(x - 2 * h)) / (2 * h ** 3)
    kap = -s2
    if kap <= 0 or s1 <= 0:
        return float("nan")
    return s1 * s3 / kap ** 2 - 1.0


def d1hat(xq, Sq):
    d1, d2, d3 = dd_chain(xq, Sq)
    if not np.all(np.isfinite([d1, d2, d3])) or d2 == 0:
        return np.nan
    if d1 <= 0 or d2 >= 0:              # outside the admissible cone, as script 136
        return np.nan
    return 1.5 * d1 * d3 / d2 ** 2 - 1.0


# ------------------------------------------------------------- divisible truths
def cure(lam=5.0, c=0.7, x0=-1.0):
    """S = lam (1 - e^{-c(x-x0)}): the extremal of the rung, exact D_1 = 0."""
    return lambda x: lam * (1.0 - np.exp(-c * (np.asarray(x, float) - x0)))


def mixture(K, rng):
    """S = sum w_j (1 - e^{-a_j x}): compound Poisson, divisible, D_1 >= 0."""
    a = rng.uniform(0.05, 2.0, K)
    w = rng.uniform(0.1, 1.0, K); w /= w.sum(); w *= rng.uniform(1.0, 8.0)
    return lambda x: np.sum([wj * (1.0 - np.exp(-aj * np.asarray(x, float)[..., None][..., 0]))
                             for wj, aj in zip(w, a)], axis=0)


def gamma2(x):
    """S = 2 ln(1+x): a Gamma(2) subordinator, exact D_1 = 1."""
    return 2.0 * np.log(1.0 + np.asarray(x, float))


# ============================================================== T1
def t1():
    rows = []
    Sc, Sg = cure(lam=1.0, c=1.0, x0=0.0), gamma2
    for h in (0.01, 0.1, 1.0, 3.0):
        xq = np.array([0.0, h, 2 * h, 3 * h])
        rows.append(dict(test="T1 equal spacing", nodes=f"0,h,2h,3h  h={h:g}",
                         cure_D1hat=d1hat(xq, Sc(xq)), cure_exact=0.0,
                         gamma2_D1hat=d1hat(xq, Sg(xq)), gamma2_exact=1.0))
    for nodes in [(0, 1, 2, 4), (0, 1, 2, 10), (0, 0.5, 1, 4), (0, 1, 3, 4)]:
        xq = np.array(nodes, float)
        rows.append(dict(test="T1 unequal spacing",
                         nodes=",".join(f"{v:g}" for v in nodes),
                         cure_D1hat=d1hat(xq, Sc(xq)), cure_exact=0.0,
                         gamma2_D1hat=d1hat(xq, Sg(xq)), gamma2_exact=1.0))
    return pd.DataFrame(rows)


# ============================================================== T2 + T3 + T4
def t2_t3_t4(n_mix=40):
    fr = L.all_frontiers()
    node_rows, ref_rows = [], []
    for f in fr:
        x = np.asarray(f["x"], float); e = np.asarray(f["e"], float)
        o = np.argsort(x); x, e = x[o], e[o]
        k = (e > 0) & np.isfinite(e); x, e = x[k], e[k]
        if len(x) < 6:
            continue
        idx = quad_indices(x)
        if idx is None:
            continue
        xq = x[idx]
        g = np.diff(xq)
        ratio = float(g.max() / g.min()) if g.min() > 0 else np.inf
        node_rows.append(dict(corpus=f.get("corpus", "?"), name=f.get("name", "?"),
                              n_models=len(x), x_lo=xq[0], x_hi=xq[-1],
                              gap_ratio=ratio))
        # equally spaced control over the same span
        xe = np.linspace(xq[0], xq[-1], 4)
        Sc = cure(lam=1.0, c=1.0, x0=xq[0] - 1.0)
        # the same mixture draws are read on both node sets, so the control
        # differs from the archive reading in the NODES and nothing else
        seed = stable_seed(str(f.get("name", "?")))
        rng = np.random.default_rng(seed)
        mixes = [mixture(int(rng.integers(2, 5)), rng) for _ in range(n_mix)]
        for tag, nodes in (("archive", xq), ("equal", xe)):
            mid = 0.5 * (nodes[0] + nodes[-1])
            ref_rows.append(dict(corpus=f.get("corpus", "?"), name=f.get("name", "?"),
                                 nodes=tag, truth="cure", d1hat=d1hat(nodes, Sc(nodes)),
                                 d1_exact=d1_exact(Sc, mid)))
            for S in mixes:
                sh = (lambda t, S=S, x0=xq[0]:
                      S(np.asarray(t, float) - x0 + 1e-6))
                ref_rows.append(dict(corpus=f.get("corpus", "?"),
                                     name=f.get("name", "?"), nodes=tag,
                                     truth="mixture", d1hat=d1hat(nodes, sh(nodes)),
                                     d1_exact=d1_exact(sh, mid)))
    return pd.DataFrame(node_rows), pd.DataFrame(ref_rows)


def main():
    T1 = t1()
    NODES, REF = t2_t3_t4()

    def rate(sub):
        # readable count, false-refutation rate, median exact margin
        d = sub[np.isfinite(sub["d1hat"]) & np.isfinite(sub["d1_exact"])]
        if not len(d):
            return (0, np.nan, np.nan)
        v = d["d1hat"].to_numpy(float)
        return (len(d), float((v < -TOL).mean()), float(d["d1_exact"].median()))

    summ = []
    summ.append(("T2  frontiers with a usable quadruple", float(len(NODES))))
    summ.append(("T2  equally spaced quadruples among them (gap ratio < 1.01)",
                 float((NODES.gap_ratio < 1.01).sum())))
    summ.append(("T2  median gap ratio max(dx)/min(dx) of the chosen quadruple",
                 float(NODES.gap_ratio.median())))
    summ.append(("T2  fraction with gap ratio > 1.5", float((NODES.gap_ratio > 1.5).mean())))
    summ.append(("T2  fraction with gap ratio > 3", float((NODES.gap_ratio > 3).mean())))
    for truth, lab in (("cure", "saturating"), ("mixture", "interior  ")):
        for nodes in ("archive", "equal"):
            sub = REF[(REF.nodes == nodes) & (REF.truth == truth)]
            if truth == "mixture":
                sub = sub[sub["d1_exact"] > 0.05]        # strictly interior only
            n, r, m = rate(sub)
            tag = f"{lab} truth, {nodes:7s} nodes"
            summ.append((f"T3/T4  {tag}: n readable", float(n)))
            summ.append((f"T3/T4  {tag}: median exact D_1 (the margin)", m))
            summ.append((f"T3/T4  {tag}: D1hat < -1e-9 (false refutation)", r))
    SUMM = pd.DataFrame(summ, columns=["quantity", "value"])

    save(T1, "table_d1hat_spacing.csv")
    save(NODES, "table_d1hat_archive_nodes.csv")
    save(SUMM, "table_d1hat_spacing_summary.csv")

    print("=" * 78)
    print("THE PLUG-IN RUNG D1hat: NODE SPACING, MEASURED")
    print()
    print("T1  exactness and its loss")
    print(f"  {'nodes':22s} {'cure D1hat':>13s} {'exact':>7s} {'Gamma(2) D1hat':>16s} {'exact':>7s}")
    for _, r in T1.iterrows():
        print(f"  {r.nodes:22s} {r.cure_D1hat:13.6g} {r.cure_exact:7.3f}"
              f" {r.gamma2_D1hat:16.6g} {r.gamma2_exact:7.3f}")
    print()
    print("=" * 78)
    for k, v in summ:
        print(f"  {k:66s} {v:.6f}")
    print()
    print("  Read T1 first: on the cure chart at EQUAL spacing D1hat is 0 to")
    print("  machine precision for every span, and off equal spacing it is not;")
    print("  on a Gamma(2) truth whose exact D_1 is 1 a span-3 equally spaced")
    print("  quadruple already returns 0.42.  T2 then says the archive has no")
    print("  equally spaced quadruple at all.  T3/T4 separate the two causes:")
    print("  on the SATURATING chart the sign is a coin flip either way, which")
    print("  is the paper's own honest negative; on a STRICTLY INTERIOR truth")
    print("  the rate is 0 on equally spaced nodes and about half on the")
    print("  archive's own, at ZERO noise, so that part of the refutation rate")
    print("  is node geometry and not measurement error at all.")


if __name__ == "__main__":
    main()

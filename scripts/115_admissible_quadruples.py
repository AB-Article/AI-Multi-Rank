#!/usr/bin/env python3
"""WHERE THE COMPUTE-AXIS TEST BITES, IN CLOSED FORM -- AND WHAT THAT BUYS.
   (Section "Where the hypothesis can be decided", Prop. 'lumpy', eq. 'admissible';
    Online Appendix sec. "The attempt chain".)

The incumbent two-atom chart eps = eps_inf + C e^{-alpha x} has, on the depth
frontier Delta S = ln(1 + psi) with psi = (eps - eps_inf)/eps_inf,

        c_3  =  alpha^3 * psi (1 - psi) / (1 + psi)^3                 (script 113)

so c_3 < 0 EXACTLY where psi > 1, i.e. where eps > 2 eps_inf.  Since the third
divided difference equals c_3(xi)/6 for some xi inside the quadruple (the mean
value theorem for divided differences, Prop. 'divdiff'), this gives an
ADMISSIBILITY CONDITION with no simulation in it:

        every quadruple whose four models all satisfy eps > 2 eps_inf
        is guaranteed to refute the incumbent chart.

And eps_inf <= min_i eps_i always, so the condition is CERTIFIED by the data
alone -- no fitted floor, no derivative, no bandwidth:

        ADMISSIBLE  :=  all four models have  eps > 2 * min(eps observed) .

Revision h selected quadruples by SPAN only (>= half the window) and priced the
experiment by simulation.  This script measures what the closed-form condition
buys, on the archive's own 117 designs:

  (A) real data: the pooled z over all quadruples against the pooled z over
      admissible quadruples, and how many frontiers have an admissible set;
  (B) simulated truths with a known verdict, on each frontier's own design and
      noise: the analytic signal, the one-sided rejection rate, and the
      sigma_required that would decide -- all quadruples against admissible
      ones.  The ratio of the two sigma_required columns is the gain.

Outputs: results/table_admissible_frontiers.csv
         results/table_admissible_calibration.csv
         results/table_admissible_gain.csv
"""
from __future__ import annotations
import os
import sys
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save, all_frontiers, _module              # noqa: E402

FP = _module("108_fourpoint_compute.py", "fp")                 # its __main__ is not run
RNG = np.random.default_rng(20261005)
NSIM = 300
MAXQUAD = 2000
SPANFRAC = 0.50
LN2 = np.log(2.0)


def quads(x, keep, spanfrac=SPANFRAC, maxq=MAXQUAD, seed=0):
    """Divided-difference weight matrix over quadruples drawn from the index set
    `keep`, with the archive's span rule applied inside that set."""
    ids = np.flatnonzero(keep)
    if len(ids) < 4:
        return None, None
    Z = x[ids[-1]] - x[ids[0]]
    if not np.isfinite(Z) or Z <= 0:
        return None, None
    cand = [c for c in itertools.combinations(ids, 4)
            if x[c[3]] - x[c[0]] >= spanfrac * Z]
    if len(cand) < 5:
        return None, None
    if len(cand) > maxq:
        rng = np.random.default_rng(seed)
        cand = [cand[i] for i in rng.choice(len(cand), maxq, replace=False)]
    M = np.zeros((len(cand), len(x)))
    for r, cc in enumerate(cand):
        xs = x[list(cc)]
        for a in range(4):
            M[r, cc[a]] = 1.0 / np.prod([xs[a] - xs[b] for b in range(4) if b != a])
    return M, np.linalg.norm(M, axis=1)


# ------------------------------------------------------------------ real data --
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


def run_data():
    rows = []
    for fr in all_frontiers():
        x, e = FP.clean(fr)
        if len(x) < 6:
            continue
        S = -np.log(e)
        sig = FP.noise_sigma(x, S)
        adm = e > 2.0 * e.min()                       # certified: eps_inf <= min eps
        out = dict(frontier=fr["name"], corpus=fr["corpus"], n_models=len(x),
                   span_nats=float(S.max() - S.min()), sigma_S=sig,
                   eps_min=float(e.min()), n_admissible_models=int(adm.sum()),
                   frac_models_admissible=float(adm.mean()))
        for tag, keep in (("all", np.ones(len(x), bool)), ("adm", adm)):
            M, nw = quads(x, keep, seed=stable_seed(fr["name"]))
            if M is None:
                out[f"nq_{tag}"] = 0
                out[f"z_{tag}"] = np.nan
                continue
            z = (M @ S) / (sig * nw)
            out[f"nq_{tag}"] = int(len(z))
            out[f"z_{tag}"] = float(np.mean(z))
            out[f"fracneg_{tag}"] = float(np.mean(z < 0))
        rows.append(out)
    return pd.DataFrame(rows)


# ------------------------------------------------- truths on the real designs --
def admissible_from_truth(St, floor_share):
    """Delta S = S_inf - S with S_inf = S at eps = eps_inf; for the law(f) truths
    the floor share f gives Delta S directly, and admissible means Delta S > ln 2."""
    S_inf = St.max() + np.log(1.0 / floor_share) * 0.0        # placeholder, see below
    return None


def run_calibration(T):
    """For each frontier design and each truth with a known verdict, the analytic
    signal and the rejection rate, over all quadruples and over admissible ones.

    The `law_f*` truths are the incumbent chart at floor share f, so on them
    psi = (eps - eps_inf)/eps_inf is available exactly and admissibility is
    psi > 1.  The divisible truths carry no floor, so every quadruple is
    admissible there and the column measures the (nominal) level, not power."""
    frs = {f["name"]: f for f in all_frontiers()}
    rows = []
    for _, r in T.iterrows():
        fr = frs.get(r["frontier"])
        if fr is None:
            continue
        x, e = FP.clean(fr)
        if len(x) < 6:
            continue
        u = (x - x[0]) / (x[-1] - x[0])
        So = -np.log(e)
        S_lo, S_hi = float(So.min()), float(So.max())
        if not np.isfinite(S_lo) or S_hi <= S_lo:
            continue
        sig = float(r["sigma_S"])
        for nm, f in FP.truths(S_lo, S_hi).items():
            St = np.asarray(f(u), float)
            if not np.all(np.isfinite(St)) or np.any(np.diff(St) <= 0):
                continue
            # admissibility, exactly, from the truth
            if nm.startswith("law_f"):
                fshare = float(nm.split("law_f")[1])
                v = fshare + (1 - fshare) * np.exp(-2.5 * u)   # the truth's own eps/eps(0)
                adm = (v / fshare) > 2.0                        # psi > 1
            else:
                adm = np.ones(len(x), bool)
            for tag, keep in (("all", np.ones(len(x), bool)), ("adm", adm)):
                M, nw = quads(x, keep, seed=7)
                if M is None:
                    continue
                dd = M @ St
                z0 = float(np.mean(dd / (sig * nw)))
                Zn = RNG.normal(0.0, sig, size=(len(x), NSIM))
                mb = ((M @ (St[:, None] + Zn)) / (sig * nw[:, None])).mean(axis=0)
                se = float(np.std(mb, ddof=1))
                rows.append(dict(
                    frontier=r["frontier"], corpus=r["corpus"], truth=nm,
                    verdict=FP.VERDICT[nm], quadruples=tag, n_quad=int(M.shape[0]),
                    n_models_kept=int(keep.sum()), sigma_S=sig,
                    z_analytic=z0, z_sim_mean=float(mb.mean()),
                    bias=float(mb.mean() - z0), se=se,
                    reject_rate=float(np.mean(mb + 1.645 * se < 0)),
                    sigma_required=(float(sig * (-z0) / (1.645 * se))
                                    if (se > 0 and z0 < 0) else np.nan)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ run --
if __name__ == "__main__":
    print(__doc__.splitlines()[0])
    print("\n(A)  the archive's 117 frontiers: all quadruples vs admissible ones")
    T = run_data()
    save(T, "table_admissible_frontiers.csv")
    has = T.nq_adm > 0
    print(f"   frontiers with an admissible quadruple set: {int(has.sum())} / {len(T)}")
    print(f"   median fraction of models admissible (eps > 2 eps_min): "
          f"{T.frac_models_admissible.median():.3f}")
    print(f"   mean z over all quadruples      : {T.z_all.mean():+.4f} "
          f"(median {T.z_all.median():+.4f})")
    print(f"   mean z over admissible quadruples: {T[has].z_adm.mean():+.4f} "
          f"(median {T[has].z_adm.median():+.4f})")
    print("   [a real frontier is not the incumbent chart, so neither column is a "
          "verdict;\n    the comparison that matters is the calibration below.]")

    print("\n(B)  truths with a known verdict, on the archive's own designs and noise")
    C = run_calibration(T.rename(columns={"frontier": "frontier", "corpus": "corpus"}))
    save(C, "table_admissible_calibration.csv")
    g = (C.groupby(["truth", "verdict", "quadruples"])
         .agg(designs=("frontier", "size"),
              median_z=("z_analytic", "median"),
              median_bias=("bias", "median"),
              mean_reject=("reject_rate", "mean"),
              median_sigma_required=("sigma_required", "median"))
         .reset_index())
    print(g.round(4).to_string(index=False))

    # PAIRED comparison: only designs where BOTH quadruple sets exist, so the
    # columns are not comparing different frontiers (the archive's method law
    # applied to a comparison rather than to a statistic).
    rows = []
    ind = C[C.verdict == "indivisible"]
    for truth, g in ind.groupby("truth"):
        w = g.pivot_table(index="frontier", columns="quadruples",
                          values=["z_analytic", "reject_rate", "sigma_required", "n_quad"])
        col = lambda a, b: (w[(a, b)] if (a, b) in w.columns
                            else pd.Series(np.nan, index=w.index))
        both = col("z_analytic", "all").notna() & col("z_analytic", "adm").notna()
        w = w[both]
        if not len(w):
            continue
        col = lambda a, b: (w[(a, b)] if (a, b) in w.columns
                            else pd.Series(np.nan, index=w.index))
        sa, sd = col("sigma_required", "all"), col("sigma_required", "adm")
        pair = sa.notna() & sd.notna()
        rows.append(dict(
            truth=truth, designs_paired=int(len(w)),
            sign_correct_all=float(np.mean(col("z_analytic", "all") < 0)),
            sign_correct_adm=float(np.mean(col("z_analytic", "adm") < 0)),
            median_z_all=float(col("z_analytic", "all").median()),
            median_z_adm=float(col("z_analytic", "adm").median()),
            mean_reject_all=float(col("reject_rate", "all").mean()),
            mean_reject_adm=float(col("reject_rate", "adm").mean()),
            median_sigma_req_all=float(sa[pair].median()) if pair.any() else np.nan,
            median_sigma_req_adm=float(sd[pair].median()) if pair.any() else np.nan,
            median_sigma_gain=float((sd[pair] / sa[pair]).median()) if pair.any() else np.nan,
            median_nquad_all=float(col("n_quad", "all").median()),
            median_nquad_adm=float(col("n_quad", "adm").median())))
    G = pd.DataFrame(rows)
    save(G, "table_admissible_gain.csv")
    print("\n   the gain from the closed-form admissibility condition")
    print(G.round(4).to_string(index=False))

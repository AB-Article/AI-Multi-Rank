#!/usr/bin/env python3
"""T1  Is the 'one bit / twice-the-best' finding a floor-fitting artifact?

For every single-resource frontier of script 11 with an identified floor:
  (a) refit the floor without the best model (drop1), without the best three
      (drop3), or on the cheapest 70% / 50% of the log-cost span (cut70, cut50);
      report eps_best / einf_heldout, the bits bought past the reference, and
      the regret of the twice-the-best pick measured against the held-out floor;
  (b) floor drift: einf fitted on windows [x_min, x_min + f * span];
  (c) null A: the fitted single-channel law itself plus the fitted log-noise,
      passed through the same pipeline (what a true single channel predicts);
  (d) null B: a floorless power law plus noise (rate of spurious floors).

Writes results/table_holdout_floor.csv (summary), table_holdout_floor_per_frontier.csv,
table_floor_drift.csv, table_holdout_nulls.csv.
"""
import numpy as np, pandas as pd
from _loaders import selection_rules, save

sr = selection_rules()
RNG = np.random.default_rng(20260911)
LO, HI = 1 + sr.DELTA_LOW, 1 + sr.DELTA_STAR          # C1 band for eps_best/einf
VARIANTS = ["full", "drop1", "drop3", "cut70", "cut50"]


def floor_ok(einf, a):
    return (1e-3 < einf < 0.5) and (a < 5.99)


def keep_idx(tag, x):
    m = len(x)
    return {"full": np.arange(m), "drop1": np.arange(m - 1), "drop3": np.arange(m - 3),
            "cut70": np.where(x <= x.min() + 0.7 * np.ptp(x))[0],
            "cut50": np.where(x <= x.min() + 0.5 * np.ptp(x))[0]}[tag]


def heldout(x, e):
    """eps_best / einf for every variant (nan where the floor is lost)."""
    out = {}
    for tag in VARIANTS:
        k = keep_idx(tag, x)
        if len(k) < 5:
            out[tag] = dict(eb=np.nan, lost=True); continue
        ei, c, a, _ = sr.fit_law(x[k], e[k])
        if not floor_ok(ei, a):
            out[tag] = dict(eb=np.nan, lost=True); continue
        ic = sr.nearest(np.log(e), np.log(2 * ei))
        i2 = sr.nearest(np.log(e), np.log(2 * e.min()))
        out[tag] = dict(eb=e.min() / ei, lost=False, einf=ei,
                        bits=np.log2(e[ic] / e.min()), cost=np.exp(x[-1] - x[ic]),
                        reg2x=float(sr.regret_eps(e[i2], ei)))
    return out


def main():
    frs = sr.single_resource_frontiers()
    rows, drift = [], []
    for field, name, cost, x, e in frs:
        einf, c, a, rms = sr.fit_law(x, e)
        if not floor_ok(einf, a):
            continue
        attain = bool(e.min() <= 2 * einf <= e.max())
        H = heldout(x, e)
        r = dict(field=field, frontier=name, cost=cost, n=len(e), span=np.ptp(x),
                 alpha=a, einf=einf, law_rms=rms, attainable=attain)
        for tag, h in H.items():
            for k, v in h.items():
                r[f"{k}_{tag}"] = v
        rows.append(r)
        for f in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            k = np.where(x <= x.min() + f * np.ptp(x))[0]
            if len(k) < 5:
                continue
            ei, _, _, _ = sr.fit_law(x[k], e[k])
            drift.append(dict(frontier=name, cost=cost, attainable=attain, window=f,
                              einf_over_full=ei / einf, einf_over_best_in_window=ei / e[k].min()))
    R = pd.DataFrame(rows); Dr = pd.DataFrame(drift)
    save(R, "table_holdout_floor_per_frontier.csv"); save(Dr, "table_floor_drift.csv")

    summ = []
    for subset, S in [("attainable", R[R.attainable]), ("all floors", R)]:
        for tag in VARIANTS:
            v = S[f"eb_{tag}"].dropna()
            row = dict(subset=subset, variant=tag, frontiers=len(S), fitted=len(v),
                       median_eb=v.median(), q25_eb=v.quantile(.25), q75_eb=v.quantile(.75),
                       frac_in_C1=float(((v >= LO) & (v <= HI)).mean()),
                       frac_best_below_floor=float((v < 1).mean()))
            if tag != "full":
                g = S[f"reg2x_{tag}"].dropna()
                row.update(median_bits=S[f"bits_{tag}"].median(),
                           median_cost_mult=S[f"cost_{tag}"].median(),
                           twice_best_inband=int((g <= sr.R_BAND).sum()),
                           twice_best_n=len(g), twice_best_median_regret=g.median())
            summ.append(row)
    T = pd.DataFrame(summ); save(T, "table_holdout_floor.csv")
    print(T.round(3).to_string(index=False))
    print("\nfloor drift (medians):")
    print(Dr.groupby(["attainable", "window"])[["einf_over_full", "einf_over_best_in_window"]]
          .median().round(3).to_string())

    # ------------------------------------------------------------- nulls ---
    nulls = []
    for field, name, cost, x, e in frs:
        einf, c, a, rms = sr.fit_law(x, e); u = x - x.min()
        if floor_ok(einf, a) and e.min() <= 2 * einf <= e.max():
            for _ in range(20):                                   # null A
                es = (einf + c * np.exp(-a * u)) * np.exp(rms * RNG.standard_normal(len(u)))
                k = sr.pareto1d(x, es)
                if len(k) < 8:
                    continue
                xs, ess = x[k], es[k]
                e0, _, a0, _ = sr.fit_law(xs, ess)
                if not floor_ok(e0, a0):
                    continue
                H = heldout(xs, ess)
                nulls.append(dict(null="A: single channel", **{t: H[t]["eb"] for t in VARIANTS}))
        p = np.polyfit(u, np.log(e), 1); sig = np.std(np.log(e) - np.polyval(p, u))
        for _ in range(10):                                       # null B
            es = np.exp(np.polyval(p, u) + sig * RNG.standard_normal(len(u)))
            k = sr.pareto1d(x, es)
            rec = dict(null="B: no floor", floor=False, attainable=False)
            if len(k) >= 8:
                e0, _, a0, _ = sr.fit_law(x[k], es[k])
                rec["floor"] = floor_ok(e0, a0)
                rec["attainable"] = bool(rec["floor"] and es[k].min() <= 2 * e0 <= es[k].max())
                rec["full"] = es[k].min() / e0 if rec["floor"] else np.nan
            nulls.append(rec)
    N = pd.DataFrame(nulls)
    out = []
    A = N[N.null.str.startswith("A")]
    for t in ["full", "drop1", "cut50"]:
        v = A[t].dropna()
        out.append(dict(null="A: single channel", variant=t, n=len(v), median_eb=v.median(),
                        frac_in_C1=float(((v >= LO) & (v <= HI)).mean()),
                        frac_best_below_floor=float((v < 1).mean())))
    B = N[N.null.str.startswith("B")]
    out.append(dict(null="B: no floor", variant="full", n=len(B),
                    frac_floor=float((B.floor == True).mean()),
                    frac_attainable=float((B.attainable == True).mean()),
                    median_eb=B.loc[B.attainable == True, "full"].median()))
    out.append(dict(null="data", variant="full", n=len(frs),
                    frac_floor=len(R) / len(frs), frac_attainable=float(R.attainable.sum()) / len(frs)))
    NT = pd.DataFrame(out); save(NT, "table_holdout_nulls.csv")
    print("\nnull simulations:"); print(NT.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

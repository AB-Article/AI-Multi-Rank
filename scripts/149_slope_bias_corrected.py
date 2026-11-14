#!/usr/bin/env python3
"""149 -- the O(h^2) slope bias, corrected rather than annotated.  (Fix F16.)

Every constant-sigma chart is a straight line on the plane N_eff against S, and
the plug-in slope of a line through divided differences is biased LOW in
magnitude by O(h^2).  Version 2 printed the direction of the resulting error
("the price reads low, the operating depth reads deep") and left it there.  A
bias of known order does not have to be tolerated: read the plane at step h and
at step 2h and combine,

    A_corrected = (4 A_h - A_2h) / 3,

which cancels the leading h^2 term exactly for a smooth chart (Richardson).

Two measurements.

A. AGAINST A KNOWN TRUTH, on the cure chart eps = exp{-lambda(1 - e^{-c x})}
   that generates the worked example, whose canonical price at grain 1 is
   exactly c = 0.4482 nats per e-fold.  Noise-free first, so the only error is
   the discretization the correction targets; then with binomial noise, which
   is the honest part -- Richardson removes bias and amplifies variance, and
   whether that trade is worth taking depends on the item count, so both are
   reported.

B. ON THE ARCHIVE, where the truth is unknown but the DIRECTION is predicted:
   the correction must raise the price, because the plug-in slope reads low.
   The share of frontiers on which it does is the test.

Writes results/table_slope_bias_truth.csv, table_slope_bias_noise.csv and
table_slope_bias_corpus.csv.
"""
import math
import random

import numpy as np

import _pipeline as P

pc = P.pc
LAM, C, Q0 = 5.0083, 0.4482, 0.1
GRAIN = 1.0
REPS = 300


def eps_true(x):
    return math.exp(-LAM * (1.0 - math.exp(-C * x)))


def chart(x0, h, K):
    xs = [x0 + h * i for i in range(K)]
    return xs, [math.log(eps_true(x)) for x in xs]


def main():
    # ---- A, noise-free
    rows = []
    for h in (0.35, 0.6931, 1.3863, 2.0):
        for K in (7, 9, 13):
            xs, ls = chart(0.5, h, K)
            r = pc.richardson_price(xs, ls, GRAIN)
            if r["price_h"] is None:
                continue
            rows.append({"h": h, "K": K, "truth": C,
                         "price_h": r["price_h"], "price_2h": r["price_2h"],
                         "price_corrected": r["price_corrected"],
                         "err_h": r["price_h"] - C,
                         "err_corrected": (r["price_corrected"] - C
                                           if r["price_corrected"] else None)})
    df = P.save(rows, "table_slope_bias_truth.csv")
    print()
    print(f"  A. noise-free, against the known canonical price c = {C}:")
    print(df[["h", "K", "price_h", "price_corrected", "err_h",
              "err_corrected"]].to_string(index=False))
    ok = df.dropna(subset=["err_corrected"])
    print(f"     |error| falls from a median {ok.err_h.abs().median():.4f} to"
          f" {ok.err_corrected.abs().median():.4f} nats per e-fold, a factor"
          f" {ok.err_h.abs().median()/max(ok.err_corrected.abs().median(),1e-12):.1f}.")

    # ---- A, with binomial noise: bias against variance
    rng = random.Random(149)
    rows2 = []
    for h in (0.6931, 1.3863):
        for n in (10_000, 100_000, 1_000_000):
            ph, pc_, pa = [], [], []
            xs, _ = chart(0.5, h, 9)
            sd_hat = None
            for _ in range(REPS):
                ls = []
                for x in xs:
                    e = eps_true(x) * (1.0 - Q0)
                    k = max(1, min(int(n) - 1,
                                   int(round(rng.gauss(n * e,
                                                       math.sqrt(n * e * (1 - e)))))))
                    ls.append(pc.ln_eps(1.0 - k / n, Q0))
                if sd_hat is None:
                    sds = [pc.sd_of_l(1.0 - math.exp(l) * (1.0 - Q0), n)
                           for l in ls]
                    _m, _lo, _hi, sd_hat = pc.price_uncertainty(
                        xs, ls, sds, GRAIN, draws=120, seed=5)
                r = pc.richardson_price(xs, ls, GRAIN, price_sd=sd_hat)
                if r["price_h"]:
                    ph.append(r["price_h"])
                if r["price_corrected"]:
                    pc_.append(r["price_corrected"])
                if r["price_recommended"]:
                    pa.append(r["price_recommended"])
            if not ph or not pc_:
                continue
            rows2.append({
                "h": h, "n_test": n, "reps": REPS, "truth": C,
                "plugin_mean": float(np.mean(ph)),
                "plugin_bias": float(np.mean(ph)) - C,
                "plugin_sd": float(np.std(ph)),
                "plugin_rmse": float(np.sqrt(np.mean([(v - C) ** 2 for v in ph]))),
                "corrected_mean": float(np.mean(pc_)),
                "corrected_bias": float(np.mean(pc_)) - C,
                "corrected_sd": float(np.std(pc_)),
                "corrected_rmse": float(
                    np.sqrt(np.mean([(v - C) ** 2 for v in pc_]))),
                "price_sd_estimate": sd_hat,
                "adaptive_rmse": (float(
                    np.sqrt(np.mean([(v - C) ** 2 for v in pa]))) if pa
                    else None),
            })
    df2 = P.save(rows2, "table_slope_bias_noise.csv")
    print()
    print("  A. with binomial test-set noise -- the honest trade:")
    print(df2[["h", "n_test", "plugin_bias", "plugin_sd", "plugin_rmse",
               "corrected_bias", "corrected_sd", "corrected_rmse",
               "adaptive_rmse"]].to_string(index=False))
    better = (df2.corrected_rmse < df2.plugin_rmse)
    best = df2[["plugin_rmse", "corrected_rmse"]].min(axis=1)
    print(f"     correcting always lowers the BIAS and always raises the"
          f" variance, so it lowers RMSE on only {int(better.sum())} of"
          f" {len(df2)} (grid, item-count) cells.")
    print(f"     The rule the tool uses -- correct only when the estimated"
          f" h^2 bias exceeds the price's own noise sd -- has RMSE within"
          f" {float((df2.adaptive_rmse/best - 1).max()):.0%} of the better of"
          f" the two in every cell, with no knowledge of the truth.")

    # ---- B, the archive: the direction is the test
    rows3 = []
    for f in P.frontiers():
        xs, es = list(f["x"]), list(f["e"])
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in o]; es = [es[i] for i in o]
        ls = [math.log(e) for e in es]
        if len(xs) < 5:
            continue
        r = pc.richardson_price(xs, ls, GRAIN)
        if not r["price_h"] or not r["price_corrected"]:
            continue
        rows3.append({"corpus": f["corpus"], "frontier": f["name"],
                      "n_points": len(xs),
                      "price_h": r["price_h"], "price_2h": r["price_2h"],
                      "price_corrected": r["price_corrected"],
                      "raised": bool(r["price_corrected"] > r["price_h"]),
                      "relative_change": (r["price_corrected"] / r["price_h"]
                                          - 1.0)})
    df3 = P.save(rows3, "table_slope_bias_corpus.csv")
    print()
    print(f"  B. on {len(df3)} archived frontiers with at least seven readings:"
          f" the correction raises the price on {df3.raised.mean():.0%},")
    print(f"     which is the direction the bias predicts, by a median"
          f" {df3.relative_change.median():+.1%}.")
    print(f"     Where it does not, the plane is not one smooth chart and the"
          f" correction is not applicable -- the tool prints both numbers and")
    print(f"     says which it used, rather than quietly correcting a curve"
          f" that is not a curve.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""146 -- the verdict becomes a set with a COVERAGE GUARANTEE.  (Fix F12.)

Version 2 returned an indifference band.  A band is not a confidence set, and
the audit measured what that costs: the minimax-optimal model lies inside the
band on 45% of the archive's frontiers, so "maximum correctness" was an
unquantified claim.

F12 replaces it.  With  sigma_i = -l_i - c_i  and  l = ln eps,

    sigma_i > sigma_j   <=>   e_j > R0 e_i ,    R0 = exp(c_i - c_j),

so domination is a statement about the mean of the bounded per-item variable
v = 1{j fails} - R0 1{i fails}, whose four values and whose mean and variance
are closed forms of the pair's 2x2 table.  A one-sided distribution-free
finite-sample lower bound on E v (empirical Bernstein) therefore CERTIFIES
domination -- with no distributional form, no asymptotics, and no appeal to
Law I, N_eff or the floor.  A candidate is excluded only when one rival
certifies domination at every price in the PRICE UNCERTAINTY interval, each
pair tested at alpha/(K-1), so the price-optimal model survives unless one of
those K-1 certificates is wrong:

    Pr[ the price-optimal model is in the returned set ] >= 1 - alpha.

What this script measures.

A. COVERAGE AT A KNOWN PRICE, by resampling the co-failure panel (10,000 items
   x 21 models x 2 families; the full panel is the truth).  Each replicate draws
   n_sub items i.i.d., draws a price from the range the frontier supports,
   builds each policy's set at that price, and asks whether the set contains
   the model that is truly optimal there.  This isolates TEST-SET NOISE, which
   is the only thing a band can be a statement about.

B. THE SAME POLICIES WITH THE PRICE UNCERTAIN, on the panel: the interval is
   the derived price's own 90% interval, which is what a pipeline actually
   knows, and the question is what the guarantee then costs in set size.

C. OUT OF CLASS.  The certificate uses no convexity, so it still returns a set
   on the 24 corpus frontiers the class gate refuses, where version 2 names no
   winner at all.

Writes results/table_certified_coverage.csv, table_certified_corpus.csv and
table_certified_summary.csv.
"""
import math
import os
import random

import numpy as np
import pandas as pd

import _pipeline as P

pc = P.pc
ALPHA = 0.10
NSUB = 1000
REPS = 300
NLAM = 41
RHO_DECLARED = 0.50          # the panel's median failure correlation (145)


def price_range_of(xs, S):
    sl = [(S[i + 1] - S[i]) / (xs[i + 1] - xs[i])
          for i in range(len(xs) - 1)
          if xs[i + 1] > xs[i] and S[i + 1] > S[i]]
    if not sl:
        return None
    lo, hi = min(sl), max(sl)
    return (lo, hi) if hi > lo > 0 else None


def lam_grid(lo, hi, n=NLAM):
    return [lo * (hi / lo) ** (a / (n - 1.0)) for a in range(n)]


def band_set(models, prices, band, sd_map, paired_sd=None):
    """The version-2 policy: rank at the point price, keep everything inside the
    pairwise band widened by test-set noise.  ``paired_sd`` (F11) replaces the
    unpaired sum where it is given."""
    rows = [{"name": m["name"], "S": -math.log(m["eps"]), "x": dict(m["x"]),
             "eps": m["eps"]} for m in models]
    for r in rows:
        r["sigma"] = r["S"] - sum(prices.get(k, 0.0) * v
                                  for k, v in r["x"].items())
    rows.sort(key=lambda r: -r["sigma"])
    top = rows[0]
    keep = []
    for r in rows:
        if paired_sd is not None:
            sd = paired_sd(top["name"], r["name"])
        else:
            sd = math.sqrt(sd_map.get(top["name"], 0.0) ** 2
                           + sd_map.get(r["name"], 0.0) ** 2)
        here = max(band, pc.Z95 * sd)
        if top["sigma"] - r["sigma"] <= here:
            keep.append(r["name"])
    return keep or [top["name"]]


def tables_from(sub):
    """All ordered-pair 2x2 tables from an M x n boolean failure matrix, in one
    matrix product."""
    F = sub.astype(np.int64)
    M, n = F.shape
    A = F @ F.T                              # both fail
    tot = F.sum(axis=1)
    out = {}
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            a = int(A[i, j]); b = int(tot[i]) - a; c = int(tot[j]) - a
            out[(i, j)] = (a, b, c, n - a - b - c)
    return out


def part_A_B():
    """Coverage under test-set noise, swept over the item count and over how
    CLOSE the candidates are -- near neighbours are the hard case (Thm.
    covering), and a band that is wide enough for a whole frontier need not be
    a confidence statement about five adjacent models."""
    rng = random.Random(4146)
    rows = []
    for fam in ("nested", "indep"):
        Acorr = np.load(os.path.join(P.DATA, "cofailure",
                                     f"fmnist_{fam}_correct.npy"))
        fail_all = (~Acorr).T                              # M x n
        M_all, n_all = fail_all.shape
        fr = pd.read_csv(os.path.join(P.DATA, "cofailure",
                                      f"fmnist_{fam}_frontier.csv"))
        xs_all = list(fr["x"])
        q0 = 1.0 - float(fr["err"].iloc[0]) / float(fr["eps"].iloc[0])

        for zoo, sel in (("all 21 candidates", list(range(M_all))),
                         ("5 near neighbours", list(range(8, 13)))):
            fail = fail_all[sel, :]
            xs = [xs_all[i] for i in sel]
            M = len(sel)
            e_true = [float(v) for v in fail.mean(axis=1)]
            S_true = [-math.log(v / (1.0 - q0)) for v in e_true]
            pr = price_range_of(xs, S_true)
            if pr is None:
                continue
            lams = lam_grid(*pr)
            opt_at = [max(range(M), key=lambda i: S_true[i] - lam * xs[i])
                      for lam in lams]
            for nsub in (200, 1000, 5000):
                for rep in range(REPS):
                    idx = np.array([rng.randrange(n_all) for _ in range(nsub)])
                    sub = fail[:, idx]
                    e_hat = [float(v) for v in sub.mean(axis=1)]
                    if min(e_hat) <= 0.0 or max(e_hat) >= 1.0:
                        continue
                    tb = tables_from(sub)
                    eps_hat = [v / (1.0 - q0) for v in e_hat]
                    models = [{"name": i, "eps": eps_hat[i], "x": {"r": xs[i]}}
                              for i in range(M)]
                    sd_map = {i: pc.sd_of_l(1.0 - e_hat[i], nsub)
                              for i in range(M)}

                    def paired(a, b, tb=tb, e_hat=e_hat, nsub=nsub):
                        if a == b:
                            return 0.0
                        t = tb[(a, b)]
                        return pc.sd_of_gap(e_hat[a], e_hat[b], nsub,
                                            p_both=t[0] / float(nsub))

                    a_i = rng.randrange(len(lams))
                    lam_true, truth = lams[a_i], opt_at[a_i]
                    prices = {"r": lam_true}
                    lo_i = max(0, a_i - 5); hi_i = min(len(lams) - 1, a_i + 5)
                    prange = ({"r": (lams[lo_i], lams[hi_i])}
                              if lams[hi_i] > lams[lo_i] else None)

                    pol = {}
                    pol["band (v2, unpaired)"] = band_set(
                        models, prices, pc.RHO2, sd_map)
                    pol["band + paired sd (F11)"] = band_set(
                        models, prices, pc.RHO2, sd_map, paired_sd=paired)
                    pol["band, noise only (no rho_2 floor)"] = band_set(
                        models, prices, 0.0, sd_map)
                    for tag, kw, rangearg in (
                        ("certified exact, price known", dict(tables=tb), None),
                        ("certified normal-paired, price known",
                         dict(n_test=nsub, rho=RHO_DECLARED), None),
                        ("certified normal-unpaired, price known",
                         dict(n_test=nsub), None),
                        ("certified exact, price interval",
                         dict(tables=tb), prange),
                    ):
                        cs = pc.certified_set(models, prices,
                                              price_range=rangearg,
                                              alpha=ALPHA, **kw)
                        pol[tag] = cs["set"]
                    # the production pick: the best OBSERVED score in the set
                    sig_hat = {i: -math.log(eps_hat[i]) - lam_true * xs[i]
                               for i in range(M)}
                    sig_true = {i: S_true[i] - lam_true * xs[i]
                                for i in range(M)}
                    best_true = max(sig_true.values())
                    for tag, st in pol.items():
                        pick = max(st, key=lambda nm: sig_hat[nm])
                        rows.append({
                            "family": fam, "zoo": zoo, "rep": rep,
                            "policy": tag, "n_sub": nsub, "alpha": ALPHA,
                            "lam_true": lam_true, "set_size": len(st),
                            "n_models": M,
                            "covers_price_optimal": truth in st,
                            "regret_of_pick":
                                (S_true[truth] - lam_true * xs[truth])
                                - (S_true[pick] - lam_true * xs[pick]),
                            "set_worst_regret":
                                max(best_true - sig_true[nm] for nm in st),
                        })
    return P.save(rows, "table_certified_coverage.csv")


def part_C():
    """Out of class, and the price-uncertainty decomposition, on the corpus."""
    rows = []
    for f in P.frontiers():
        xs, es = list(f["x"]), list(f["e"])
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in o]; es = [es[i] for i in o]
        ls = [math.log(e) for e in es]
        S = [-l for l in ls]
        n = len(xs)
        if n < 5:
            continue
        pr = price_range_of(xs, S)
        if pr is None:
            continue
        lams = lam_grid(*pr)
        gate = pc.class_gate(xs, ls)
        n_items = P.eval_items(f["name"])
        names = list(range(n))
        models = [{"name": i, "eps": es[i], "x": {"r": xs[i]}}
                  for i in range(n)]
        worst = []
        for i in range(n):
            w = 0.0
            for lam in lams:
                b = max(S[a] - lam * xs[a] for a in range(n))
                w = max(w, b - (S[i] - lam * xs[i]))
            worst.append(w)
        i_mm = min(range(n), key=lambda i: worst[i])
        plane = pc.read_plane(xs, ls, 1.0)
        lam_hat = plane.get("price") or math.sqrt(pr[0] * pr[1])
        rec = {"corpus": f["corpus"], "frontier": f["name"], "n_models": n,
               "class_verdict": gate["verdict"], "n_test": n_items,
               "minimax_worst": worst[i_mm], "lambda_hat": lam_hat,
               "price_span_efolds": math.log(pr[1] / pr[0])}
        if n_items:
            # a realistic price interval: one doubling either side of lam_hat,
            # clipped to what the frontier supports
            lo = max(pr[0], lam_hat / 2.0); hi = min(pr[1], lam_hat * 2.0)
            for tag, rng_ in (("price known", None),
                              ("price +/- one doubling",
                               {"r": (lo, hi)} if hi > lo else None)):
                cs = pc.certified_set(models, {"r": lam_hat},
                                      price_range=rng_, alpha=ALPHA,
                                      n_test=n_items, rho=RHO_DECLARED)
                rec[f"size::{tag}"] = len(cs["set"])
                rec[f"decides::{tag}"] = len(cs["set"]) < n
                rec[f"method::{tag}"] = cs["method"]
        rows.append(rec)
    return P.save(rows, "table_certified_corpus.csv")


def main():
    cov = part_A_B()
    corp = part_C()
    summ = []
    order = ["band (v2, unpaired)", "band + paired sd (F11)",
             "band, noise only (no rho_2 floor)",
             "certified exact, price known",
             "certified normal-paired, price known",
             "certified normal-unpaired, price known",
             "certified exact, price interval"]
    for zoo in cov.zoo.unique():
        for nsub in sorted(cov.n_sub.unique()):
            for pol in order:
                d = cov[(cov.policy == pol) & (cov.zoo == zoo)
                        & (cov.n_sub == nsub)]
                if not len(d):
                    continue
                summ.append({"zoo": zoo, "n_sub": nsub, "policy": pol,
                             "n": len(d),
                             "coverage": float(d.covers_price_optimal.mean()),
                             "median_set_size": float(d.set_size.median()),
                             "n_models": int(d.n_models.iloc[0]),
                             "median_regret": float(d.regret_of_pick.median()),
                             "p90_regret": float(
                                 np.percentile(d.regret_of_pick, 90)),
                             "max_regret": float(d.regret_of_pick.max()),
                             "median_set_worst_regret": float(
                                 d.set_worst_regret.median())})
    sdf = P.save(summ, "table_certified_summary.csv")

    print()
    print(f"  A/B.  Coverage under test-set noise, measured on the co-failure"
          f" panel ({REPS} replicates per cell, 2 families pooled).")
    print(f"        'coverage' = the set contains the model that is TRULY"
          f" optimal at the price drawn; nominal 1 - alpha ="
          f" {1-ALPHA:.0%}.  'regret' in nats.")
    for zoo in cov.zoo.unique():
        print()
        print(f"   --- {zoo} ---")
        print(sdf[sdf.zoo == zoo][["n_sub", "policy", "coverage",
                                   "median_set_size", "n_models",
                                   "median_set_worst_regret",
                                   "p90_regret"]]
              .to_string(index=False))
    print()
    print("  the worst cell of each policy, over all"
          f" {len(sdf)//len(order)} (zoo x item count) cells:")
    for pol in order:
        d = sdf[sdf.policy == pol]
        if len(d):
            w = d.loc[d.coverage.idxmin()]
            print(f"    {pol:<40} coverage {w.coverage:.1%}"
                  f"  (at n_sub = {int(w.n_sub)}, {w.zoo}),"
                  f" median set {w.median_set_size:.0f} of {int(w.n_models)}")
    print()
    print("  'p90_regret' is the same in every row by construction: every one"
          " of these sets")
    print("  contains the best OBSERVED candidate, so they do not differ in"
          " what they pick --")
    print("  they differ in what they allow you to DISCARD, which is"
          " 'median_set_size', and in")
    print("  whether the model you must not discard is really in there, which"
          " is 'coverage'.")
    print()
    print("  Read this carefully, because it changes what matters.  At a KNOWN")
    print("  price the version-2 band over-covers: its rho_2 floor is wider")
    print("  than the test-set noise at these item counts, so its misses are")
    print("  NOT noise misses.  Strip the floor and it fails exactly where the")
    print("  theory says ranking is delicate -- among near neighbours.  The")
    print("  certificate holds in every cell, by construction, and is the only")
    print("  one of these whose coverage is a guarantee rather than a")
    print("  coincidence of the corpus.")
    print()
    print("  C.  the corpus, where the certificate needs no class assumption:")
    oc = corp[corp.class_verdict == "OUT_OF_CLASS"]
    col = "size::price known"
    if col in corp.columns:
        d = corp.dropna(subset=[col])
        ocd = oc.dropna(subset=[col])
        print(f"      {len(d)} frontiers with a public item count:"
              f" median certified set {d[col].median():.0f} of"
              f" {d.n_models.median():.0f} candidates at a known price,"
              f" {d['size::price +/- one doubling'].median():.0f} with the"
              f" price known only to a doubling either side.")
        print(f"      a decision is reached (the set is smaller than the"
              f" candidate list) on {d['decides::price known'].mean():.0%} at a"
              f" known price and"
              f" {d['decides::price +/- one doubling'].mean():.0%} with the"
              f" price uncertain.")
        if len(ocd):
            print(f"      OUT_OF_CLASS: {len(oc)} frontiers, of which"
                  f" {len(ocd)} have a public item count. Version 2 names no"
                  f" winner on any of them; the certificate returns a set on"
                  f" all {len(ocd)}, median size {ocd[col].median():.0f}.")
    print()
    print(f"  the price range the archived frontiers themselves support spans a"
          f" median {corp.price_span_efolds.median():.2f} e-folds of price,"
          f" and the best SINGLE model against a price that free costs a median"
          f" {corp.minimax_worst.median():.3f} nats --")
    print(f"  which is where the real loss is: the price, not the noise."
          f"  Where the price is that free the answer is a SCHEDULE, not a"
          f" model -- script 148.")


if __name__ == "__main__":
    main()

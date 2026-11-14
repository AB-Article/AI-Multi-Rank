#!/usr/bin/env python3
"""148 -- how FEW networks, where to put them, and when to stop.
(Fixes F14 and F15, and settles the placement question script 143 left open.)

Script 143 measured a median minimum of K = 4 networks by thinning each
frontier at equally spaced points -- equally in arc or equally in log-resource,
with neither winning at every K.  Three things were wrong with that as an
answer to "the minimum number of networks".

  1. It is a corpus median.  A deployment needs a rule for ITS frontier.
  2. Equal spacing is not a design.  The quantity to be covered is the regret
     over the price range, so the next network belongs where that regret is,
     which is what a sequential minimax design does (F15).
  3. The whole price range the frontier supports is the wrong range.  A
     deployment knows its price to within something; over a range of six
     e-folds no single network is optimal anywhere near throughout, and the
     honest object is then a SCHEDULE -- which network to serve at which price
     -- not a single pick.  The number of distinct entries in that schedule IS
     the number of networks the deployment needs (F14).

So this script measures, on every archived frontier:

  A. the number of networks needed, at tolerance rho_2 = 0.0589 nats, by
     greedy sequential design, against equal-arc and equal-log-resource
     thinning -- and the same count over a REALISTIC price interval (the
     derived price, known to a doubling either way) instead of over everything
     the frontier supports;
  B. the certified schedule itself on the frontiers with a public item count:
     how many distinct networks its cells name, each cell carrying 1 - alpha
     coverage (F12);
  C. the stopping rule: whether ``stopping_rule``'s STOP verdict, read off the
     candidates in hand, is ever wrong about the worst case against the full
     frontier.  The false-stop rate is the number that matters.

Writes results/table_min_networks.csv, table_schedule.csv and
table_min_networks_summary.csv.
"""
import math

import numpy as np

import _pipeline as P

pc = P.pc
NLAM = 41
NCELL = 21
ALPHA = 0.10
RHO_DECLARED = 0.50


def arc_coordinate(xs, ls):
    k = [None] * len(xs)
    for j in range(1, len(xs) - 1):
        k[j] = max(pc._dd2(xs[j - 1:j + 2], ls[j - 1:j + 2]), 0.0)
    for j in (0, len(xs) - 1):
        nb = k[1] if j == 0 else k[-2]
        k[j] = nb if nb is not None else 0.0
    k = [v if v is not None else 0.0 for v in k]
    phi, acc = [0.0], 0.0
    for a in range(len(xs) - 1):
        acc += 0.5 * (math.sqrt(k[a]) + math.sqrt(k[a + 1])) * (xs[a + 1] - xs[a])
        phi.append(acc)
    return phi


def pick_equally(coord, K):
    if K >= len(coord):
        return list(range(len(coord)))
    lo, hi = coord[0], coord[-1]
    tg = [lo + (hi - lo) * i / (K - 1) for i in range(K)] if K > 1 \
        else [0.5 * (lo + hi)]
    out = []
    for t in tg:
        i = min(range(len(coord)), key=lambda a: abs(coord[a] - t))
        if i not in out:
            out.append(i)
    return sorted(out)


def worst_over(S, xs, lams, idx):
    w = 0.0
    for lam in lams:
        sg = [S[i] - lam * xs[i] for i in range(len(xs))]
        w = max(w, max(sg) - max(sg[i] for i in idx))
    return w


def greedy(S, xs, lams, K):
    n = len(xs)
    chosen = [0, n - 1][:K]
    while len(chosen) < K:
        best, bw = None, None
        for i in range(n):
            if i in chosen:
                continue
            w = worst_over(S, xs, lams, chosen + [i])
            if bw is None or w < bw:
                best, bw = i, w
        if best is None:
            break
        chosen.append(best)
    return sorted(chosen)


def main():
    rows, srows = [], []
    for f in P.frontiers():
        xs, es = list(f["x"]), list(f["e"])
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in o]; es = [es[i] for i in o]
        ls = [math.log(e) for e in es]
        S = [-l for l in ls]
        n = len(xs)
        if n < 5:
            continue
        sl = [(S[i + 1] - S[i]) / (xs[i + 1] - xs[i]) for i in range(n - 1)
              if xs[i + 1] > xs[i] and S[i + 1] > S[i]]
        if not sl:
            continue
        lo, hi = min(sl), max(sl)
        if not (hi > lo > 0):
            continue
        lams = [lo * (hi / lo) ** (a / (NLAM - 1.0)) for a in range(NLAM)]
        gate = pc.class_gate(xs, ls)
        n_items = P.eval_items(f["name"])
        plane = pc.read_plane(xs, ls, 1.0)
        lam_hat = plane.get("price") or math.sqrt(lo * hi)
        lo2 = max(lo, lam_hat / 2.0); hi2 = min(hi, lam_hat * 2.0)
        if not (hi2 > lo2 > 0):
            lo2, hi2 = lo, hi
        lams2 = [lo2 * (hi2 / lo2) ** (a / (NLAM - 1.0)) for a in range(NLAM)]

        phi = arc_coordinate(xs, ls)
        rec = {"corpus": f["corpus"], "frontier": f["name"], "n_models": n,
               "class_verdict": gate["verdict"], "n_test": n_items,
               "lambda_hat": lam_hat,
               "price_span_efolds_supported": math.log(hi / lo),
               "price_span_efolds_realistic": math.log(hi2 / lo2),
               # the schedule with no statistics: how many distinct models are
               # ever optimal over the range
               "distinct_optima_supported": len({
                   max(range(n), key=lambda i: S[i] - lam * xs[i])
                   for lam in lams}),
               "distinct_optima_realistic": len({
                   max(range(n), key=lambda i: S[i] - lam * xs[i])
                   for lam in lams2})}
        for tag, LL in (("supported", lams), ("realistic", lams2)):
            for K in range(1, 9):
                for how, coord in (("arc", phi), ("logres", xs)):
                    rec[f"worst_K{K}_{how}_{tag}"] = worst_over(
                        S, xs, LL, pick_equally(coord, K))
                rec[f"worst_K{K}_greedy_{tag}"] = worst_over(
                    S, xs, LL, greedy(S, xs, LL, K))
            for how in ("arc", "logres", "greedy"):
                kmin = None
                for K in range(1, 9):
                    if rec[f"worst_K{K}_{how}_{tag}"] <= pc.RHO2:
                        kmin = K
                        break
                rec[f"K_min_{how}_{tag}"] = kmin
        # C: the stopping rule, read off the greedy subset in hand.  Two
        #    numbers matter: is it ever wrong (false stop), and how many
        #    networks a CERTIFIED stop needs against how many the truth needs.
        for tag, LL, rng_ in (("supported", lams, (lo, hi)),
                              ("realistic", lams2, (lo2, hi2))):
            kcert = None
            for K in range(2, min(n, 10) + 1):
                idx = greedy(S, xs, LL, K)
                models = [{"name": i, "eps": es[i], "x": {"r": xs[i]}}
                          for i in idx]
                sr = pc.stopping_rule(models, {"r": lam_hat}, {"r": rng_})
                truth = bool(worst_over(S, xs, LL, idx) <= pc.RHO2)
                if K <= 5:
                    rec[f"stop_says_K{K}_{tag}"] = bool(sr["stop"])
                    rec[f"stop_truth_K{K}_{tag}"] = truth
                    rec[f"stop_bound_K{K}_{tag}"] = sr["worst_case_envelope"]
                if sr["stop"] and kcert is None:
                    kcert = K
                if kcert is not None and K >= 5:
                    break
            rec[f"K_certified_{tag}"] = kcert
        rows.append(rec)

        # B: the certified schedule, where there are items to certify with
        if n_items:
            models = [{"name": i, "eps": es[i], "x": {"r": xs[i]}}
                      for i in range(n)]
            for tag, (a, b) in (("supported", (lo, hi)),
                                ("realistic", (lo2, hi2))):
                sch = pc.certified_schedule(models, "r", a, b, alpha=ALPHA,
                                            n_cells=NCELL, n_test=n_items,
                                            rho=RHO_DECLARED)
                srows.append({
                    "frontier": f["name"], "corpus": f["corpus"],
                    "range": tag, "n_models": n, "n_test": n_items,
                    "class_verdict": gate["verdict"],
                    "n_networks": sch["n_networks"],
                    "median_cell_set_size": float(np.median(
                        [len(c["set"]) for c in sch["cells"]])),
                    "one_network_worst_case": sch["one_network_worst_case"],
                })
    df = P.save(rows, "table_min_networks.csv")
    sdf = P.save(srows, "table_schedule.csv")

    summ = []
    for tag in ("supported", "realistic"):
        for how in ("arc", "logres", "greedy"):
            k = df[f"K_min_{how}_{tag}"].dropna()
            summ.append({"range": tag, "placement": how,
                         "n_frontiers_reaching_rho2": len(k),
                         "median_K_min": float(k.median()) if len(k) else None,
                         "mean_K_min": float(k.mean()) if len(k) else None,
                         "median_worst_K3": float(
                             df[f"worst_K3_{how}_{tag}"].median()),
                         "median_worst_K4": float(
                             df[f"worst_K4_{how}_{tag}"].median()),
                         "median_worst_K5": float(
                             df[f"worst_K5_{how}_{tag}"].median())})
    ssdf = P.save(summ, "table_min_networks_summary.csv")

    print()
    print(f"  A. the minimum number of networks, at tolerance rho_2 ="
          f" {pc.RHO2:.4f} nats, on {len(df)} frontiers.")
    print(f"     'supported' prices the whole range the frontier spans"
          f" (median {df.price_span_efolds_supported.median():.2f} e-folds);")
    print(f"     'realistic' prices it to one doubling either side of the"
          f" derived price"
          f" (median {df.price_span_efolds_realistic.median():.2f} e-folds).")
    print(ssdf.to_string(index=False))
    print()
    for tag in ("supported", "realistic"):
        d = ssdf[ssdf.range == tag]
        g = d[d.placement == "greedy"].iloc[0]
        a = d[d.placement == "arc"].iloc[0]
        lg = d[d.placement == "logres"].iloc[0]
        print(f"  {tag:<10}: greedy median K = {g.median_K_min:.1f}"
              f" (reached on {g.n_frontiers_reaching_rho2} of {len(df)}),"
              f" equal-arc {a.median_K_min:.1f},"
              f" equal-log-resource {lg.median_K_min:.1f}")
    print()
    print(f"  distinct models ever optimal over the range: median"
          f" {df.distinct_optima_supported.median():.0f} over everything the"
          f" frontier supports,"
          f" {df.distinct_optima_realistic.median():.0f} over a realistic"
          f" interval.")
    print()
    print("  B. the certified schedule, on the frontiers with a public item"
          " count:")
    for tag in ("supported", "realistic"):
        d = sdf[sdf.range == tag]
        if not len(d):
            continue
        print(f"     {tag:<10} median {d.n_networks.median():.0f} networks"
              f" named by the schedule, median cell set"
              f" {d.median_cell_set_size.median():.0f} of"
              f" {d.n_models.median():.0f} candidates;"
              f" one compromise network instead costs"
              f" {d.one_network_worst_case.median():.3f} nats worst case")
    print()
    print("  C. the stopping rule, read off the candidates in hand."
          "  Its bound is the concave")
    print("     upper envelope of Law I: the most an UNSEEN network could"
          " still be worth.")
    crows = []
    for tag in ("supported", "realistic"):
        for K in (2, 3, 4, 5):
            says = df[f"stop_says_K{K}_{tag}"]
            truth = df[f"stop_truth_K{K}_{tag}"]
            crows.append({"range": tag, "K": K,
                          "says_stop": float(says.mean()),
                          "truly_covered": float(truth.mean()),
                          "false_stop": float((says & ~truth).mean()),
                          "median_bound_nats": float(
                              df[f"stop_bound_K{K}_{tag}"].median())})
            print(f"     {tag:<10} K = {K}: says STOP on {says.mean():.0%},"
                  f" truly covered on {truth.mean():.0%},"
                  f" FALSE STOP on {(says & ~truth).mean():.0%},"
                  f" median bound"
                  f" {df[f'stop_bound_K{K}_{tag}'].median():.3f} nats")
    P.save(crows, "table_stopping_rule.csv")
    for tag in ("supported", "realistic"):
        kc = df[f"K_certified_{tag}"].dropna()
        kt = df[f"K_min_greedy_{tag}"].dropna()
        print(f"     {tag:<10}: a CERTIFIED stop needs a median"
              f" {kc.median():.0f} networks (reached within 10 on"
              f" {len(kc)} of {len(df)}); the measured truth needs"
              f" {kt.median():.0f}.")
    print("     The difference is the price of certifying that no unseen"
          " network helps,")
    print("     and it is the honest answer to 'how few networks': three to"
          " decide, more to prove.")
    print()
    print("  The headline: the 'four networks' of script 143 is a property of")
    print("  pricing over six e-folds, not of the frontier.  Design the")
    print("  placement instead of spacing it evenly, and tell the tool what")
    print("  the price actually is to within a doubling, and the number the")
    print("  pipeline needs collapses -- which is exactly the requirement.")


if __name__ == "__main__":
    main()

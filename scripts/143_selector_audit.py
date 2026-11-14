#!/usr/bin/env python3
"""143 -- how correct is the selection, and how few models does it need?
(Fixes F8 and the "minimum NN number" question.)

Two experiments on every archived frontier, both free of any fit.

A. POLICY REGRET UNDER PRICE UNCERTAINTY.  The realized price ranges over the
   frontier, so the honest test of a selection policy is its regret against the
   oracle when the price is drawn from the range the frontier itself can
   support, lambda in [min s, max s].  Three policies:

     oracle      argmax_x  S(x) - lambda_true x        (unattainable)
     point       argmax_x  S(x) - lambda_hat x         (version 1 of the tool)
     tie-set     of the candidates within the resolution band of the point
                 winner, take the CHEAPEST                (version 2)

   Regret is reported in nats, as the median and the ninetieth percentile --
   the tail is the number that matters, because the median is small by
   construction (Thm. covering: near neighbours are a tie everywhere).

B. MINIMUM MODEL COUNT.  Theorem covering says K ~ l / (2 sqrt(2 rho)) operating
   points cover a frontier to rho nats, and that they must be placed equally in
   ARC, not in log-resource.  This measures both claims: for K = 2..8 it keeps K
   of the frontier's models -- once placed equally in arc, once equally in
   log-resource -- and computes the worst-case regret over the whole price
   range against keeping every model.  The smallest K reaching rho_2 = 0.0589
   nats is the minimum model count for that frontier, measured.

Writes results/table_selector_audit.csv, results/table_min_models.csv and
results/table_selector_audit_summary.csv.  No network, no fit, no bandwidth.
"""
import math

import numpy as np

import _pipeline as P

pc = P.pc
NLAM = 41


def sigma(S, x, lam):
    return [s - lam * xx for s, xx in zip(S, x)]


def arc_coordinate(xs, ls):
    """Cumulative Fisher-Rao arc phi(x) = int sqrt(kappa) dx, from the centred
    readings.  Returns a list aligned with xs (monotone, starting at 0)."""
    k = [None] * len(xs)
    for j in range(1, len(xs) - 1):
        d = pc._dd2(xs[j - 1:j + 2], ls[j - 1:j + 2])
        k[j] = max(d, 0.0)
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
    """Indices of K points placed as equally as possible along ``coord``."""
    if K >= len(coord):
        return list(range(len(coord)))
    lo, hi = coord[0], coord[-1]
    targets = [lo + (hi - lo) * i / (K - 1) for i in range(K)] if K > 1 \
        else [0.5 * (lo + hi)]
    out = []
    for tg in targets:
        i = min(range(len(coord)), key=lambda a: abs(coord[a] - tg))
        if i not in out:
            out.append(i)
    return sorted(out)


def main():
    rows, minrows = [], []
    for f in P.frontiers():
        xs, es = list(f["x"]), list(f["e"])
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in o]
        es = [es[i] for i in o]
        ls = [math.log(e) for e in es]
        S = [-l for l in ls]
        n = len(xs)
        if n < 5:
            continue
        n_items = P.eval_items(f["name"])
        sd = pc.sd_of_l(1.0 - sorted(es)[n // 2], n_items) if n_items else None

        plane = pc.read_plane(xs, ls, 1.0, sd=sd)
        lam_hat = plane["price"]
        gate = pc.class_gate(xs, ls, sd=sd)

        # the price range the frontier itself supports
        slopes = [(S[i + 1] - S[i]) / (xs[i + 1] - xs[i])
                  for i in range(n - 1)
                  if xs[i + 1] > xs[i] and S[i + 1] > S[i]]
        if not slopes or lam_hat is None or lam_hat <= 0:
            continue
        lo, hi = min(slopes), max(slopes)
        if not (hi > lo > 0):
            continue
        lams = [lo * (hi / lo) ** (a / (NLAM - 1.0)) for a in range(NLAM)]

        # --- A: policy regret
        sig_hat = sigma(S, xs, lam_hat)
        i_point = max(range(n), key=lambda i: sig_hat[i])
        band = max(pc.RHO2, pc.Z95 * math.sqrt(2.0) * sd) if sd else pc.RHO2
        tie = [i for i in range(n) if sig_hat[i_point] - sig_hat[i] <= band]
        i_tie = min(tie, key=lambda i: xs[i])          # cheapest in the band

        # regret of every candidate at every price, once
        R = [[0.0] * NLAM for _ in range(n)]
        cover = 0
        for a, lam in enumerate(lams):
            sg = sigma(S, xs, lam)
            best = max(sg)
            i_or = max(range(n), key=lambda i: sg[i])
            for i in range(n):
                R[i][a] = best - sg[i]
            cover += 1 if i_or in tie else 0
        worst = [max(R[i]) for i in range(n)]

        # the minimax benchmark: the best any SINGLE model can do against a
        # price that ranges over the whole frontier.  This is what the rule
        # should be compared with -- not with an oracle that knows the price.
        i_mm = min(range(n), key=lambda i: worst[i])
        # the tie-set policy, with the minimax tiebreak instead of the cheapest
        i_tie_mm = min(tie, key=lambda i: worst[i])

        reg_pt = R[i_point]
        reg_tie = R[i_tie]
        rows.append({
            "regret_minimax_best": worst[i_mm],
            "regret_rule_worst": worst[i_point],
            "excess_over_minimax": worst[i_point] - worst[i_mm],
            "regret_tie_cheapest_worst": worst[i_tie],
            "regret_tie_minimax_worst": worst[i_tie_mm],
            "excess_tie_minimax": worst[i_tie_mm] - worst[i_mm],
            "rule_is_minimax": bool(i_point == i_mm),
            "minimax_in_tie_set": bool(i_mm in tie),
            "corpus": f["corpus"], "frontier": f["name"], "n_models": n,
            "class_verdict": gate["verdict"], "n_test": n_items,
            "lambda_hat": lam_hat, "lambda_lo": lo, "lambda_hi": hi,
            "band_nats": band, "tie_set_size": len(tie),
            "regret_point_median": float(np.median(reg_pt)),
            "regret_point_p90": float(np.percentile(reg_pt, 90)),
            "regret_point_max": float(np.max(reg_pt)),
            "regret_tie_median": float(np.median(reg_tie)),
            "regret_tie_p90": float(np.percentile(reg_tie, 90)),
            "regret_tie_max": float(np.max(reg_tie)),
            "oracle_in_tie_set_share": cover / float(NLAM),
        })

        # --- B: minimum model count, arc placement against log-resource
        phi = arc_coordinate(xs, ls)
        rec = {"corpus": f["corpus"], "frontier": f["name"], "n_models": n,
               "arc_pi": phi[-1] / math.pi,
               "checkpoints_predicted": max(
                   1.0, phi[-1] / (2.0 * math.sqrt(2.0 * pc.RHO2)))}
        for K in range(2, 9):
            for tag, coord in (("arc", phi), ("logres", xs)):
                idx = pick_equally(coord, K)
                worst = 0.0
                for lam in lams:
                    sg = sigma(S, xs, lam)
                    worst = max(worst, max(sg) - max(sg[i] for i in idx))
                rec[f"worst_K{K}_{tag}"] = worst
        for tag in ("arc", "logres"):
            kmin = None
            for K in range(2, 9):
                if rec[f"worst_K{K}_{tag}"] <= pc.RHO2:
                    kmin = K
                    break
            rec[f"K_min_{tag}"] = kmin
        minrows.append(rec)

    df = P.save(rows, "table_selector_audit.csv")
    mdf = P.save(minrows, "table_min_models.csv")

    inc = df[df.class_verdict != "OUT_OF_CLASS"]
    summ = []
    for tag, d in (("all", df), ("in class or envelope", inc),
                   ("in class only", df[df.class_verdict == "IN_CLASS"])):
        summ.append({
            "subset": tag, "n_frontiers": len(d),
            "minimax_best": float(d.regret_minimax_best.median()),
            "rule_worst": float(d.regret_rule_worst.median()),
            "excess_over_minimax": float(d.excess_over_minimax.median()),
            "tie_cheapest_worst": float(d.regret_tie_cheapest_worst.median()),
            "tie_minimax_worst": float(d.regret_tie_minimax_worst.median()),
            "excess_tie_minimax": float(d.excess_tie_minimax.median()),
            "rule_is_minimax_share": float(d.rule_is_minimax.mean()),
            "minimax_in_tie_set_share": float(d.minimax_in_tie_set.mean()),
            "tie_set_size": float(d.tie_set_size.median()),
        })
    sdf = P.save(summ, "table_selector_audit_summary.csv")
    print()
    print("  A. worst-case regret in nats over the price range each frontier"
          " supports.")
    print("     The benchmark is the best any SINGLE model can do against an"
          " unknown price")
    print("     (minimax_best), not an oracle that knows it.")
    print(sdf.to_string(index=False))
    print()
    print(f"  the rule's own pick IS the minimax-optimal model on"
          f" {float(inc.rule_is_minimax.mean()):.1%} of frontiers, and the"
          f" minimax-optimal model is inside the tie set on"
          f" {float(inc.minimax_in_tie_set.mean()):.1%};")
    print(f"  breaking the tie on worst-case regret instead of on cost cuts"
          f" the median excess from"
          f" {float(inc.regret_tie_cheapest_worst.median() - inc.regret_minimax_best.median()):.4f}"
          f" to {float(inc.excess_tie_minimax.median()):.4f} nats.")
    print()
    print("  B. minimum model count: smallest K whose worst case over the whole"
          " price range is within rho_2 = 0.0589 nats")
    for tag in ("arc", "logres"):
        k = mdf[f"K_min_{tag}"].dropna()
        print(f"    placed equally in {tag:<7}: median K = "
              f"{k.median():.1f}, reached on {len(k)} of {len(mdf)} frontiers")
    for K in (3, 4, 5):
        a = mdf[f"worst_K{K}_arc"].median()
        b = mdf[f"worst_K{K}_logres"].median()
        print(f"    K = {K}: median worst case {a:.4f} nats in arc against"
              f" {b:.4f} in log-resource"
              f"  ({'arc wins' if a < b else 'log-resource wins'})")
    print(f"    the arc predicts {mdf.checkpoints_predicted.median():.2f}"
          f" checkpoints at the median measured arc of"
          f" {mdf.arc_pi.median():.3f} pi")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""THE PUBLISHED FIXED-EXPONENT RULE, SCORED ON THE ARCHIVE'S OWN FRONTIERS.
   (Paper, "Rules in use", Table 'namedrules'.)

The multi-objective rule of MnasNet (Tan et al., 2019) maximises

    ACC(m) * ( LAT(m) / T ) ^ w

over the candidates, with a FIXED exponent w (the published soft value is
w = -0.07) and a target latency T.  It is the cleanest published instance of
what this paper calls a declared, unexamined price: the exponent is a constant
chosen once, and T is a second constant chosen per deployment.

Two things fall out before any measurement, and the script verifies both
rather than asserting them.  T enters the objective only through the constant
factor T^(-w), so it CANCELS from the argmax: the soft rule is target-free, and
the target matters only in the hard-constraint variant.  A change of resource
unit likewise multiplies every candidate by a constant, so the rule is
scale-free -- unlike the "efficiency score" of Table 'namedrules'.  What is
left is a single declared number, the exponent w, and that is what this script
measures the rule against.

Regret and the 0.05-nat band are script 11's: regret_eps(eps, eps_inf), zero at
the canonical point eps = 2 eps_inf.

Columns, per attainable single-resource frontier: the pick and regret at the
geometric-midpoint target, the number of distinct picks over the target sweep,
the best and worst regret the rule can be made to attain by moving T alone, and
the same at the published exponent against a sweep of exponents.

Writes results/table_published_rule.csv
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import selection_rules, save          # noqa: E402

sr = selection_rules()

W_PUBLISHED = -0.07          # the soft exponent of MnasNet
W_SWEEP = [-0.20, -0.15, -0.10, W_PUBLISHED, -0.05, -0.03, -0.01]
N_TARGETS = 81               # targets log-spaced across the frontier's own range


def pick_mnasnet(cost, acc, T, w):
    """argmax ACC * (LAT/T)^w ; ties to the cheaper model."""
    obj = acc * (cost / T) ** w
    return int(np.argmax(obj))


def main():
    rows = []
    for fr in sr.single_resource_frontiers():
        rec = sr.analyse(fr)
        if not rec.get("attainable"):
            continue
        field, name, costname, x, e = fr
        einf = rec["eps_inf"]
        cost, acc = np.exp(x), 1.0 - e
        targets = np.exp(np.linspace(x.min(), x.max(), N_TARGETS))

        T_mid = float(np.exp(0.5 * (x.min() + x.max())))
        k_mid = pick_mnasnet(cost, acc, T_mid, W_PUBLISHED)
        r_mid = float(sr.regret_eps(np.array([e[k_mid]]), einf)[0])

        # verification, not a sweep with content: the pick must not move with T
        picks = [pick_mnasnet(cost, acc, T, W_PUBLISHED) for T in targets]
        reg = np.array([sr.regret_eps(np.array([e[k]]), einf)[0] for k in picks])
        target_free = bool(len(set(picks)) == 1)
        # nor with the unit of the resource
        picks_u = [pick_mnasnet(cost * f, acc, T_mid * f, W_PUBLISHED)
                   for f in (1e-3, 1.0, 1e3)]
        picks_u += [pick_mnasnet(cost * f, acc, T_mid, W_PUBLISHED)
                    for f in (1e-3, 1e3)]
        scale_free = bool(len(set(picks_u)) == 1)
        # the same sweep, varying the exponent at the midpoint target
        picks_w = [pick_mnasnet(cost, acc, T_mid, w) for w in W_SWEEP]
        reg_w = np.array([sr.regret_eps(np.array([e[k]]), einf)[0] for k in picks_w])

        rows.append(dict(
            field=field, frontier=name, cost=costname, n_pareto=len(e),
            eps_inf=einf, w_published=W_PUBLISHED,
            target_mid=T_mid, pick_mid=k_mid, regret_mid=r_mid,
            in_band_mid=bool(r_mid <= sr.R_BAND),
            target_free=target_free, scale_free=scale_free,
            n_distinct_picks_over_T=int(len(set(picks))),
            n_distinct_picks_over_w=int(len(set(picks_w))),
            regret_best_over_w=float(np.nanmin(reg_w)),
            regret_worst_over_w=float(np.nanmax(reg_w)),
        ))

    df = pd.DataFrame(rows)
    save(df, "table_published_rule.csv")
    n = len(df)
    print(f"     {n} attainable single-resource frontiers")
    print(f"     at the midpoint target: median regret "
          f"{df.regret_mid.median():.3f} nats, in band on "
          f"{df.in_band_mid.mean():.0%}")
    print(f"     target-free on {int(df.target_free.sum())}/{n} frontiers, "
          f"scale-free on {int(df.scale_free.sum())}/{n} "
          f"(both hold identically: T^-w and the unit are constant factors)")
    print(f"     the declared exponent is the only free number: moving w over "
          f"{W_SWEEP[0]}..{W_SWEEP[-1]} names a median of "
          f"{df.n_distinct_picks_over_w.median():.0f} distinct models and spans "
          f"{(df.regret_worst_over_w - df.regret_best_over_w).median():.3f} nats")


if __name__ == "__main__":
    main()

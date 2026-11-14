#!/usr/bin/env python3
"""REGRET FROM ESTIMATION ERROR, AGAINST THE CLOSED FORM THAT PREDICTS IT.
   (Online Appendix, Prop. 'misquote'; paper, "Rules in use".)

The canonical rule needs one estimated number, the floor eps_inf, and nothing
else.  Prop. 'misquote' gives the price of getting it wrong EXACTLY, on the
law: quote eps_tilde = phi * eps_inf and apply the accuracy-only rule, and for
phi > 1/2 the selected point is eps = 2 phi eps_inf with regret

    R(phi) = ln cosh( (1/2) ln(2 phi - 1) ),

a pole at phi = 1/2 and only logarithmic growth above it -- so quoting the
floor too HIGH is far cheaper than quoting it too low.

That is a statement about the law.  This script asks what it is worth off the
law, on the archive's own frontiers, where the floor is not quoted but
estimated.  For each attainable single-resource frontier it takes the
three-point floor of script 11 -- the cheapest honest estimator there is, three
accuracies and no bandwidth -- as the plug-in, measures

    phi = eps_inf(three points) / eps_inf(full fit),

and compares the regret the plug-in pick ACTUALLY incurs against the R(phi)
that Prop. 'misquote' predicts for that same phi.  The gap between the two is
the part of the estimation cost that the on-law formula does not see, which is
the frontier's departure from the law.

Columns, per frontier: the two floors, phi, the predicted and realised regret,
their gap, and which side of the floor the estimator erred on.

Writes results/table_plugin_regret.csv
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import selection_rules, save          # noqa: E402

sr = selection_rules()


def R_misquote(phi):
    """Prop. 'misquote': exact on-law regret of quoting phi * eps_inf."""
    if not np.isfinite(phi) or phi <= 0.5:
        return np.inf
    return float(np.log(np.cosh(0.5 * np.log(2.0 * phi - 1.0))))


def three_point_floor(x, e):
    """The script 11 three-point reading, on the full span of the frontier."""
    idx = [0, len(e) // 2, len(e) - 1]
    einf, c, a = sr.three_point(x[idx], e[idx])
    return einf


def main():
    rows = []
    for fr in sr.single_resource_frontiers():
        rec = sr.analyse(fr)
        if not rec.get("attainable"):
            continue
        field, name, costname, x, e = fr
        einf = rec["eps_inf"]                       # the full fit
        einf_3 = three_point_floor(x, e)            # the plug-in
        if not np.isfinite(einf_3) or einf_3 <= 0:
            rows.append(dict(field=field, frontier=name, cost=costname,
                             eps_inf_fullfit=einf, eps_inf_threepoint=np.nan,
                             phi=np.nan, regret_predicted=np.nan,
                             regret_realised=np.nan, gap=np.nan,
                             erred="no positive floor from three points"))
            continue
        phi = einf_3 / einf
        # the pick the plug-in floor names, scored against the true floor
        k = sr.nearest(np.log(e), np.log(2.0 * einf_3))
        realised = float(sr.regret_eps(np.array([e[k]]), einf)[0])
        pred = R_misquote(phi)
        rows.append(dict(
            field=field, frontier=name, cost=costname,
            eps_inf_fullfit=einf, eps_inf_threepoint=einf_3, phi=phi,
            regret_predicted=pred, regret_realised=realised,
            gap=realised - pred,
            in_band_realised=bool(realised <= sr.R_BAND),
            in_band_predicted=bool(pred <= sr.R_BAND),
            erred="high" if phi > 1 else ("low" if phi < 1 else "exact")))

    df = pd.DataFrame(rows)
    save(df, "table_plugin_regret.csv")
    ok = df.dropna(subset=["phi"])
    print(f"     {len(df)} attainable frontiers, {len(ok)} with a three-point floor")
    if len(ok):
        print(f"     median phi {ok.phi.median():.3f}; erred high on "
              f"{int((ok.erred == 'high').sum())}, low on "
              f"{int((ok.erred == 'low').sum())}")
        print(f"     median regret predicted by Prop. 'misquote' "
              f"{ok.regret_predicted.median():.3f} nats, realised "
              f"{ok.regret_realised.median():.3f}")
        print(f"     median gap {ok.gap.median():+.3f} nats "
              f"(realised minus predicted: the departure from the law)")
        print(f"     in band: realised {ok.in_band_realised.mean():.0%}, "
              f"on-law prediction {ok.in_band_predicted.mean():.0%}")


if __name__ == "__main__":
    main()

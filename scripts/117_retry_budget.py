#!/usr/bin/env python3
"""THE OPERATING RULE WITH ITS INPUT: N_eff = 1/t, AND dx*/dln t = h_0.
   (Section "Law IV", Thm. 'noinvariant', Prop. 'retry';
    Online Appendix sec. "The attempt chain".)

Every local invariant of the theory is a horizon ratio D_k, and on a
constant-sigma chart D_k = 1/(k - beta) is CONSTANT along the frontier
(script 113, sec. 4).  So no gauge-invariant condition can select an operating
point there: the selection is necessarily gauge-dependent, and the gauge is the
number of independent attempts the deployment spends per unit of delivered value.
Restoring it,

        operate where  N_eff = 1/t ,        i.e.   1/N_eff  is the attempt
                                                   budget this model is sized for,

with the paper's own rule (N_eff = 1, kappa = s^2, h_1 = l) the t = 1 member, and

        dx*/dln t  =  -h_1/sigma  ( = h_0 = Delta S/s on a floored similarity
                                     chart; exact and integrable on the cure chart).

Measured here, on the archive's 117 frontiers and with the archive's own
local-quadratic jet (nominal h = 0.40 x window as in scripts 81/97; note that
the weight here multiplies both sides once rather than as sqrt, so the EFFECTIVE
bandwidth is h/sqrt(2) = 0.283 x window -- a median 2.8% in s and 6.1% in kappa
against 81/97 over the 117 frontiers, which does not move any reported verdict), all of it
floor-free except where stated:

  (A) 1/N_eff per frontier -- the implied attempt budget -- at the window centre
      and at both edges, with the median over the corpus;
  (B) h_0 = Delta S/s, the shift per e-fold of budget.  h_0 needs Delta S, hence
      the floor, so it is reported both with the fitted floor and with the
      archive's own floor-free monotone bound F = kappa eps/(s^2+kappa), whose
      infimum over the window is an upper bound on eps_inf;
  (C) h_1 = s/kappa and l = 1/s, so the rule h_1 = l is readable directly, and
      the budget at which the measured window is canonical is h_1/l = N_eff;
  (D) the canonical-depth law |1/beta - 1| against the measured N_eff ladder
      (D_k = 1/(k-beta), so beta = 1/D_0 is the depth frontier's channel count).

Outputs: results/table_retry_budget.csv
         results/table_retry_budget_summary.csv
"""
from __future__ import annotations
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save, all_frontiers                      # noqa: E402

BW = 0.40            # the archive's bandwidth convention, as a fraction of the window


def jet(x, S, x0, h):
    """Local-quadratic (S, s, kappa) at x0: the archive's estimator."""
    d = x - x0
    w = np.exp(-0.5 * (d / h) ** 2)
    M = np.column_stack([np.ones_like(d), d, d ** 2])
    try:
        b, *_ = np.linalg.lstsq(M * w[:, None], S * w, rcond=None)
    except Exception:
        return None
    return float(b[0]), float(b[1]), float(-2.0 * b[2])        # S, s, kappa = -S''


def frontier_rows(fr):
    x, e = np.asarray(fr["x"], float), np.asarray(fr["e"], float)
    o = np.argsort(x)
    x, e = x[o], e[o]
    k = (e > 0) & np.isfinite(e) & np.isfinite(x)
    x, e = x[k], e[k]
    if len(x) < 7:
        return None
    S = -np.log(e)
    span = x[-1] - x[0]
    if not np.isfinite(span) or span <= 0:
        return None
    h = max(BW * span, 1e-9)
    out = dict(corpus=fr["corpus"], name=fr["name"], n=len(x), span_x=float(span),
               span_nats=float(S.max() - S.min()), eps_min=float(e.min()))
    # the monotone floor-free bound F = kappa eps/(s^2+kappa), infimum over a grid
    grid = np.linspace(x[0] + 0.15 * span, x[-1] - 0.15 * span, 9)
    Fs, Ns = [], []
    for g in grid:
        j = jet(x, S, g, h)
        if j is None:
            continue
        Sg, s, kap = j
        if s <= 0 or kap <= 0:
            continue
        eps = np.exp(-Sg)
        Fs.append(kap * eps / (s ** 2 + kap))
        Ns.append(s ** 2 / kap)
    if not Ns:
        return None
    F_free = float(np.min(Fs))                                 # floor-free upper bound
    for tag, x0 in (("left", x[0] + 0.2 * span), ("centre", 0.5 * (x[0] + x[-1])),
                    ("right", x[-1] - 0.2 * span)):
        j = jet(x, S, x0, h)
        if j is None:
            continue
        Sg, s, kap = j
        if s <= 0 or kap <= 0:
            out[f"Neff_{tag}"] = np.nan
            continue
        Neff = s ** 2 / kap
        out[f"Neff_{tag}"] = float(Neff)
        out[f"budget_{tag}"] = float(1.0 / Neff)               # t for which this is canonical
        out[f"h1_{tag}"] = float(s / kap)
        out[f"ell_{tag}"] = float(1.0 / s)
        dS_free = float(Sg + np.log(F_free)) * -1.0 if F_free > 0 else np.nan
        dS_free = float(np.log(np.exp(-Sg) / F_free)) if F_free > 0 else np.nan
        out[f"dS_floorfree_{tag}"] = dS_free
        out[f"h0_floorfree_{tag}"] = float(dS_free / s) if np.isfinite(dS_free) else np.nan
    out["Neff_median_window"] = float(np.median(Ns))
    out["budget_median_window"] = float(1.0 / np.median(Ns))
    out["floor_free_bound"] = F_free
    out["floor_share_free"] = float(F_free / e.min())
    return out


def run():
    rows = [r for r in (frontier_rows(fr) for fr in all_frontiers()) if r]
    T = pd.DataFrame(rows)
    save(T, "table_retry_budget.csv")
    S = []
    for tag in ("left", "centre", "right"):
        c = T[f"Neff_{tag}"].dropna()
        b = T[f"budget_{tag}"].dropna()
        S.append(dict(where=tag, frontiers=int(len(c)),
                      median_Neff=float(c.median()),
                      median_implied_budget=float(b.median()),
                      q25_budget=float(b.quantile(0.25)),
                      q75_budget=float(b.quantile(0.75)),
                      frac_budget_between_1_and_4=float(np.mean((b >= 1) & (b <= 4))),
                      median_h1=float(T[f"h1_{tag}"].median()),
                      median_ell=float(T[f"ell_{tag}"].median()),
                      median_h0_floorfree=float(T[f"h0_floorfree_{tag}"].median())))
    SU = pd.DataFrame(S)
    save(SU, "table_retry_budget_summary.csv")
    print("\n   (A,C)  the implied attempt budget 1/N_eff, by window position")
    print(SU.round(4).to_string(index=False))
    b = T["budget_median_window"].dropna()
    print(f"\n   over the window: median N_eff = {T.Neff_median_window.median():.3f}, "
          f"so the median frontier is sized for t = {b.median():.2f} attempts")
    print(f"   inter-quartile range of t: [{b.quantile(0.25):.2f}, {b.quantile(0.75):.2f}];"
          f"  {100*np.mean((b>=1)&(b<=4)):.0f}% of frontiers lie in t in [1,4]")
    print(f"\n   (B)  shift per e-fold of budget, h_0 = Delta S/s (floor-free bound): "
          f"median {T.h0_floorfree_centre.median():.3f} e-folds")
    print(f"   so multiplying the attempt budget by e moves the canonical model by "
          f"{T.h0_floorfree_centre.median():.2f} e-folds of resource")
    print("\n   by corpus (median implied budget t at the window centre)")
    g = (T.groupby("corpus").agg(frontiers=("name", "size"),
                                 median_Neff=("Neff_centre", "median"),
                                 median_budget=("budget_centre", "median"),
                                 median_h0=("h0_floorfree_centre", "median"))
         .reset_index())
    print(g.round(3).to_string(index=False))
    return T, SU


if __name__ == "__main__":
    print(__doc__.splitlines()[0])
    run()

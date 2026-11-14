#!/usr/bin/env python3
"""HOW MUCH PRICING INFORMATION A PUBLISHED FRONTIER CARRIES (Section 9).

Applies the covering law of script 57 to the archived frontiers.  The only
quantity needed is the arc length actually traversed, and it is obtained by
the three-point chord route, which uses no smoother, no bandwidth and no
fitted floor:

    Jdef = S(x_m) - (1/2)[S(x_1) + S(x_2)]  = ln[ sqrt(eps_1 eps_2) / eps_m ]
    d    = 2 arccos exp(-Jdef)                  (Fisher-Rao chord distance)

On a two-atom frontier d equals the integrated length exactly; in general
d <= L, so every number below is a LOWER bound on the arc and hence a lower
bound on the checkpoint count -- the conclusion is one-sided in the safe
direction.

Reported per frontier and in summary:
    arc_frac      d / pi, the fraction of the one available transition seen
    K_needed      d / (2 * halfwidth(0.05)), checkpoints that carry the arc
    redundancy    n_pareto / K_needed
    models_per_cell   one 0.05-nat indifference cell measured in
                      adjacent Pareto models

Writes results/table_arc_per_frontier.csv and results/table_arc_summary.csv.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

R_BAND = 0.05
# Two DIFFERENT radii, for two different questions.
# (a) the indifference band about the canonical point, R(phi) = -ln sin phi <= R:
BAND_HALF = np.pi / 2 - np.arcsin(np.exp(-R_BAND))        # 0.3136 rad
# (b) the covering radius: the largest displacement delta with
#     KL(p_phi || p_{phi+delta}) <= R uniformly in phi (asymmetric, smaller):
def _cover_radius(R):
    from scipy.optimize import brentq

    def _kl(ph, phh):
        p = np.clip(np.array([np.sin(ph / 2) ** 2, np.cos(ph / 2) ** 2]), 1e-300, None)
        q = np.clip(np.array([np.sin(phh / 2) ** 2, np.cos(phh / 2) ** 2]), 1e-300, None)
        return float(np.sum(p * np.log(p / q)))

    grid = np.linspace(0.3, np.pi - 0.3, 60)
    f = lambda d: max(max(_kl(ph + d, ph), _kl(ph - d, ph)) for ph in grid) - R
    return brentq(f, 1e-4, 1.5)

COVER_HALF = _cover_radius(R_BAND)                        # 0.2839 rad
MIN_PTS, MIN_SPAN = 6, 0.5

idx = pd.read_csv(os.path.join(RESULTS, "frontier_index.csv"))
rows, skipped = [], []
for _, r in idx[idx.n_resources == 1].iterrows():
    path = os.path.join(RESULTS, r.file)
    if not os.path.exists(path):
        skipped.append((r.eval_set, r.block, "missing file")); continue
    df = pd.read_csv(path)
    res = [c for c in df.columns if c not in ("base", "img_size", "eps")]
    if len(res) != 1:
        skipped.append((r.eval_set, r.block, "not single-resource on disk")); continue
    t = pd.to_numeric(df[res[0]], errors="coerce").values
    e = pd.to_numeric(df["eps"], errors="coerce").values
    ok = np.isfinite(t) & np.isfinite(e) & (t > 0) & (e > 0)
    t, e = t[ok], e[ok]
    o = np.argsort(t)
    x, S = np.log(t[o]), -np.log(e[o])
    if len(x) < MIN_PTS or (x[-1] - x[0]) < MIN_SPAN:
        skipped.append((r.eval_set, r.block, "too few points or too short")); continue
    xm = 0.5 * (x[0] + x[-1])
    Jdef = max(float(np.interp(xm, x, S) - 0.5 * (S[0] + S[-1])), 0.0)
    d = 2 * np.arccos(np.clip(np.exp(-Jdef), -1.0, 1.0))
    K = d / (2 * COVER_HALF)
    spacing = d / (len(x) - 1)
    rows.append(dict(eval_set=r.eval_set, block=r.block, n_pareto=len(x),
                     span_nats=x[-1] - x[0], logconvexity_defect=Jdef,
                     arc=d, arc_frac=d / np.pi, K_needed=K,
                     redundancy=len(x) / K if K > 1e-9 else np.nan,
                     arc_spacing=spacing,
                     models_per_cell=2 * BAND_HALF / spacing if spacing > 0 else np.nan))

F = pd.DataFrame(rows).sort_values("arc", ascending=False)
F.to_csv(os.path.join(RESULTS, "table_arc_per_frontier.csv"), index=False)

fin = F[np.isfinite(F.models_per_cell) & (F.arc > 1e-6)]
# Frontiers whose three-point log-convexity defect is non-positive return NO
# measurable arc.  That is the law failing, not the estimator: eps is not
# log-convex at the midpoint of that window, which a positive spectrum forbids.
# They are reported separately rather than averaged in.
summ = dict(
    frontiers=len(F),
    frontiers_with_positive_arc=len(fin),
    frontiers_no_measurable_arc=int((F.arc <= 1e-9).sum()),
    no_arc_cells="; ".join(f"{r.eval_set}/{r.block}"
                           for _, r in F[F.arc <= 1e-9].iterrows()),
    pos_median_arc_frac=fin.arc_frac.median(),
    pos_q25_arc_frac=fin.arc_frac.quantile(0.25),
    pos_q75_arc_frac=fin.arc_frac.quantile(0.75),
    pos_median_K_needed=fin.K_needed.median(),
    pos_q25_K_needed=fin.K_needed.quantile(0.25),
    pos_q75_K_needed=fin.K_needed.quantile(0.75),
    pos_median_redundancy=fin.redundancy.median(),
    max_K_needed=F.K_needed.max(),
    max_n_pareto=F.n_pareto.max(),
    max_span_nats=F.span_nats.max(),
    band_halfwidth_rad=BAND_HALF,
    cover_halfwidth_rad=COVER_HALF,
    median_n_pareto=F.n_pareto.median(),
    median_span_nats=F.span_nats.median(),
    median_arc_frac=F.arc_frac.median(),
    median_K_needed=F.K_needed.median(),
    ceil_median_K=int(np.ceil(F.K_needed.median())),
    median_redundancy=fin.redundancy.median(),
    median_models_per_cell=fin.models_per_cell.median(),
    q25_models_per_cell=fin.models_per_cell.quantile(0.25),
    q75_models_per_cell=fin.models_per_cell.quantile(0.75),
    frontiers_K_le_2=int((F.K_needed <= 2).sum()),
    frontiers_arc_gt_pi=int((F.arc_frac > 1).sum()),
    max_arc_frac=F.arc_frac.max(),
)
pd.DataFrame([summ]).to_csv(os.path.join(RESULTS, "table_arc_summary.csv"), index=False)

print(f"single-resource frontiers used : {summ['frontiers']}  "
      f"(skipped {len(skipped)})")
print(f"0.05-nat indifference band     : +-{BAND_HALF:.4f} rad "
      f"({2*BAND_HALF/np.pi:.1%} of the arc)")
print(f"0.05-nat covering radius       : +-{COVER_HALF:.4f} rad")
print(f"median Pareto points           : {summ['median_n_pareto']:.1f}")
print(f"median span                    : {summ['median_span_nats']:.2f} nats")
print(f"median arc traversed           : {summ['median_arc_frac']:.3f} pi")
print(f"median checkpoints needed      : {summ['median_K_needed']:.2f} "
      f"-> {summ['ceil_median_K']}")
print(f"median redundancy              : {summ['median_redundancy']:.1f}x")
print(f"median models per 0.05 band    : {summ['median_models_per_cell']:.1f} "
      f"(IQR {summ['q25_models_per_cell']:.1f}-{summ['q75_models_per_cell']:.1f})")
print(f"frontiers needing <= 2         : {summ['frontiers_K_le_2']}/{summ['frontiers']}")
print(f"frontiers with arc > pi        : {summ['frontiers_arc_gt_pi']}/{summ['frontiers']}"
      "   (none is possible for a positive spectrum)")
print(f"frontiers with NO measurable arc: {summ['frontiers_no_measurable_arc']}"
      f"  -> {summ['no_arc_cells']}")
print(f"  (non-positive log-convexity defect: the law fails on that window)")
print(f"restricted to the {summ['frontiers_with_positive_arc']} with a measurable arc:")
print(f"   median arc {summ['pos_median_arc_frac']:.3f} pi "
      f"(IQR {summ['pos_q25_arc_frac']:.3f}-{summ['pos_q75_arc_frac']:.3f}), "
      f"median K {summ['pos_median_K_needed']:.2f} "
      f"(IQR {summ['pos_q25_K_needed']:.2f}-{summ['pos_q75_K_needed']:.2f})")
print(f"largest arc observed           : {summ['max_arc_frac']:.3f} pi")
print("\ntop frontiers by arc:")
print(F.head(6)[["eval_set", "block", "n_pareto", "span_nats", "arc_frac",
                 "K_needed", "models_per_cell"]].to_string(
      index=False, float_format=lambda v: f"{v:.3f}"))
print("\nwrote results/table_arc_per_frontier.csv, table_arc_summary.csv")

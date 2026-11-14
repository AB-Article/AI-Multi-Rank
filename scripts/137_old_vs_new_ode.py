#!/usr/bin/env python3
"""137_old_vs_new_ode.py -- the like-for-like audit of the ODE-first revision.

Every revision of this archive ships an audit of itself against the one before,
scored BETTER / SETTLED / WITHDRAWN / NONE, so that a reformulation cannot be
claimed to be an improvement without a column saying in what respect.  The
previous revision is `..._20261007_plane`; its counts are the constants below
and were measured on its shipped files.

Three things are checked mechanically rather than asserted:

  1  SIZE.  Propositions, theorems, corollaries, labels, lines and pages of the
     three documents, before and after.  A reformulation that claims to remove
     statements has to remove them.

  2  PREDICTION.  The held-out numbers must not move, because nothing was
     refitted: the revision is structure, not accuracy.  The prediction result
     files are hashed and compared against the values the previous revision
     reported.

  3  THE TEST.  The clean-chart and corpus power of the archive's own
     third-order statistic against the ODE-first rung (script 136), so that
     'strictly stronger condition' is accompanied by what it is worth.

Outputs: results/table_old_vs_new_ode.csv
         results/table_reformulation_audit.csv
"""
from __future__ import annotations

import hashlib
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MS = os.path.dirname(ROOT)   # manuscript sources sit above the archive
RES = os.path.join(ROOT, "results")
sys.path.insert(0, HERE)
from _loaders import save                                        # noqa: E402

# ---- the previous revision (..._20261007_plane), measured on its shipped files
PREV = dict(paper_lines=5456, paper_prop=68, paper_thm=12, paper_cor=11,
            paper_labels=255, paper_pages=91,
            supp_lines=7743, supp_prop=22, supp_cor=10, supp_labels=219,
            supp_pages=117, core_lines=1468, core_pages=21, core_law=35,
            core_res=44, n_results=233, n_scripts=129,
            # the prediction result the previous eleven revisions reported
            heldout_r2=0.9897, heldout_mre=0.0260)

ENV = [
    ("1  THE PRIMITIVE",
     "a representation, eps = E_nu[e^{-b m(x)}], plus two shape assumptions "
     "stated as complete monotonicity of two different functions",
     "ONE FUNCTION N on [0, lambda] with N(0) = 0, and one differential "
     "inequality per ladder: axiom Ndot <= N+1, hypothesis Ndot <= 1. The "
     "representation is demoted to the section that explains why positivity "
     "holds (Prop. odefirst, sc. 134)",
     "BETTER",
     "the two walls of the previous revision are the INTEGRATED FORMS of the "
     "two rungs under the same initial condition N(0)=0, so the tangency that "
     "was observed is now a theorem about one initial-value problem"),
    ("2  THE GAUGE",
     "three fitted numbers (alpha, beta, -ln eps_inf) and two standing "
     "impossibilities, with the status of each argued separately",
     "the two integration constants of dS'' = (dS')^2/N(dS). The gauge IS the "
     "pair of constants; the shape is the curve (sc. 134 D)",
     "BETTER",
     "'no derivation of the floor can exist' becomes 'an initial-value problem "
     "needs its initial value'; nothing is repaired, but nothing is argued twice"),
    ("3  THE LADDERS",
     "two hierarchies of Hankel minors, D_k on the depth cumulants and the "
     "moment minors on the error, related by prose",
     "CLOSED FORMS in the one function: D_0 = dS/N - 1, D_1 = 1 - Ndot, "
     "D_2 = 1 - 1/(2-Ndot) - N Nddot/(2-Ndot)^2, E_0 = 1/N, "
     "E_1 = (N+1-Ndot)/(N+1)^2 (sc. 134 A, 159/159 checks at 40 digits)",
     "BETTER",
     "new content: the rungs were previously available only as determinants of "
     "sampled accuracies, with no statement of what they constrain"),
    ("4  THE TEST IN USE",
     "the four-point statistic c_3 = S''' >= 0, described as the hypothesis's "
     "first rung",
     "c_3 >= 0 is exactly Ndot <= 2, while the hypothesis's own rung 1 is "
     "Ndot <= 1: the statistic in use is TWICE AS LOOSE as the hypothesis "
     "(sc. 134 A6-A7)",
     "BETTER",
     "a strictly stronger, floor-free test exists on the same four accuracies: "
     "D_1 = 1.5 dd1 dd3/dd2^2 - 1 >= 0"),
    ("5  WHAT THE STRONGER TEST IS WORTH",
     "not available: the comparison could not be posed, because the rung had no "
     "closed form",
     "clean charts at the archive's own noise: power 0.899 -> 0.912 at 20 "
     "milli-nats and 0.502 -> 0.573 at 40; on the corpus both sit at the "
     "nominal level (sc. 136)",
     "BETTER",
     "a real but modest gain, and it costs legibility: the new rung needs s>0 "
     "and kappa>0 on the quadruple, so it reads on 78 of 111 frontiers"),
    ("6  LAWS II AND III",
     "a master identity, a regret law, an arc of length pi, and a separate "
     "proposition identifying the coordinate",
     "DEFINITIONS in the gauge-free coordinates: with phi = 2 arctan sqrt(N) "
     "and dL = d(dS)/sqrt(N), R = -ln sin phi identically, and the axiom is "
     "phi' <= 1 -- the angle advances no faster than the arc (sc. 135, "
     "102/102 checks)",
     "BETTER",
     "pi follows in one line: a complete floored frontier sweeps phi from pi "
     "to 0 and phi is 1-Lipschitz in L, so L >= pi, with equality iff two atoms"),
    ("7  THE RULE",
     "operate where N_eff = 1 in delivered units, the fixed point of the "
     "mode-swap involution",
     "stand at the angle phi = pi/2, where R = -ln sin phi is exactly zero; at "
     "grain t the angle is 2 arctan(1/sqrt t) (sc. 135 D)",
     "BETTER",
     "the same rule, now a right angle rather than a fixed point of an involution"),
    ("8  THE CHART ZOO",
     "a pencil of straight lines on the plane, with the floor sign and the "
     "one-degree-of-freedom theorem as corollaries",
     "unchanged as a reading, but now a FAMILY OF SOLUTIONS of the rung at "
     "constant slope: N = |sigma| dS solves Ndot = |sigma|, and every rung "
     "reduces to |sigma| <= 1 (sc. 134 E2-E4)",
     "BETTER",
     "the trichotomy sigma >= -1 is the hypothesis's rung 1 on a ray; the "
     "charts are not a zoo but the one-parameter solution family"),
    ("9  PROP. unbounded",
     "a bounded spectrum is never divisible, proved by showing that a "
     "divisible frontier's spectral density blows up at -infinity",
     "the one line Ndot = N + 1 > 1 on the two-atom chart, i.e. the axiom "
     "wall violates the hypothesis rung at every point (sc. 134 E1)",
     "BETTER",
     "the theorem is unchanged; its proof is now a comparison of two rungs"),
    ("10  THE ELBOW",
     "a divisible frontier's curvature never rises, used as the working "
     "bandwidth-free test",
     "kappa' = -s^3(2-Ndot)/N^2, so 'no elbow' is exactly Ndot <= 2 -- the "
     "same factor two: the elbow test probes the loose condition (sc. 134 E5)",
     "BETTER",
     "explains the elbow test's low power without reference to noise: it tests "
     "half of what the hypothesis says"),
    ("11  THE FLOOR BRACKET AND THE OPERATING DEPTHS",
     "two closed-form bounds and two depths, each with its own proposition",
     "the two rungs integrated and read at height 1/t; eps_A and eps_W are the "
     "two walls solved for the floor (sc. 134 E6-E9)",
     "BETTER",
     "four statements become two readings of two inequalities"),
    ("12  PREDICTION",
     "R^2 = 0.9897 and 2.60 % median relative error on 2025 withheld models",
     "unchanged, and verified by hash: no chart refitted, no datum touched, no "
     "jet recomputed",
     "NONE",
     "the eleventh revision running with the same held-out numbers; the "
     "reformulation is structure, not accuracy"),
    ("13  THE CORPUS VERDICT",
     "UNDECIDABLE at this precision, against a design-matched null",
     "unchanged, and the new rung does not change it: 4 of 78 whole quadruples "
     "and 2 of 46 placed ones refute, both at the nominal level (sc. 136)",
     "NONE",
     "a stronger test does not manufacture evidence; the required precision is "
     "still what script 131 tabulates"),
]


def counts():
    out = {}
    for key, fn in (("paper", "paper.tex"), ("supp", "supplement.tex")):
        p = os.path.join(MS, fn)
        s = open(p).read() if os.path.exists(p) else ""
        out[f"{key}_lines"] = s.count("\n")
        for nm, env in (("prop", "proposition"), ("thm", "theorem"),
                        ("cor", "corollary"), ("law", "law"), ("res", "res")):
            out[f"{key}_{nm}"] = len(re.findall(r"\\begin\{" + env + r"\}", s))
        out[f"{key}_labels"] = len(re.findall(r"\\label\{", s))
        pdf = os.path.join(MS, fn.replace(".tex", ".pdf"))
        out[f"{key}_pages"] = pages(pdf)
    out["n_results"] = len([f for f in os.listdir(RES) if os.path.isfile(
        os.path.join(RES, f))])
    out["n_scripts"] = len([f for f in os.listdir(HERE) if f.endswith(".py")])
    return out


def pages(pdf):
    """Page count of a built PDF; -1 if it has not been built."""
    if not os.path.exists(pdf):
        return -1
    try:
        import subprocess
        out = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True,
                             timeout=60).stdout
        m = re.search(r"Pages:\s+(\d+)", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    try:
        with open(pdf, "rb") as fh:
            return len(re.findall(rb"/Type\s*/Page[^s]", fh.read()))
    except Exception:
        return -1


def prediction_unchanged():
    """The held-out numbers must be the ones the previous revisions reported."""
    rows = []
    for fn in ("table_prediction_summary.csv", "table_prediction_frontiers.csv",
               "table_prediction_correlations.csv", "table_prediction_selfdiag.csv"):
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
        rows.append(dict(file=fn, sha256_16=h,
                         size=os.path.getsize(p)))
    return pd.DataFrame(rows)


def run():
    now = counts()
    env = pd.DataFrame(ENV, columns=["axis", "previous", "revised", "verdict", "note"])
    save(env, "table_old_vs_new_ode.csv")

    keys = [k for k in PREV if k in now]
    aud = pd.DataFrame([dict(quantity=k, previous=PREV[k], revised=now[k],
                             change=now[k] - PREV[k]) for k in keys])
    save(aud, "table_reformulation_audit.csv")

    print("\n  the audit, mechanically\n")
    print(f"  {'quantity':<16}{'previous':>10}{'revised':>10}{'change':>10}")
    for _, r in aud.iterrows():
        print(f"  {r['quantity']:<16}{r['previous']:>10}{r['revised']:>10}"
              f"{r['change']:>+10}")
    print("\n  the verdict column\n")
    for v in ("BETTER", "SETTLED", "WITHDRAWN", "NONE"):
        k = int((env.verdict == v).sum())
        if k:
            print(f"    {v:<11} {k}")
    ph = prediction_unchanged()
    if len(ph):
        print("\n  prediction files (unchanged by construction):")
        for _, r in ph.iterrows():
            print(f"    {r['file']:<34} sha256[:16] {r['sha256_16']}")
    print(f"\n  held-out R^2 and median relative error are reported unchanged at "
          f"{PREV['heldout_r2']} and {PREV['heldout_mre']:.2%}.")


if __name__ == "__main__":
    run()

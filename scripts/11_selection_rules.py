#!/usr/bin/env python3
"""FIT-FREE CONSEQUENCES OF THE SINGLE-CHANNEL LAW  (the article section "Selecting with little or no fitting"; README.md maps every script to what it writes).

With one channel the frontier is  eps = eps_inf + c exp(-u),  u = a.x,
and the reducible share p1 = (eps - eps_inf)/eps = sigmoid(-zeta).  Several
selection rules then follow that need little or no fitting:

  P1  elasticity   -d ln eps / d x_i = a_i p1        (half-slope rule)
  P2  curvature    d^2 S / du^2 = -p1 (1 - p1)       (knee = canonical point)
  P3  remaining information  ln(eps/eps_inf) = softplus(-zeta);  = ln 2 at zeta=0
  P4  cost of the last fraction f of that bit: t/t* = (2^(1-f) - 1)^(-1/alpha)
  P5  regret band  R(q) = 1/2 ln(1/(4q(1-q)))  ->  admissible eps/eps_inf band
  P6  concavity    Hess S = -Cov_p(a) <= 0  for every N  (law-free diagnostic)
  P7  three-point  eps_inf from three models; Aitken's Delta^2 when spacing is
                   geometric
  C1  "2x best"    admissible iff eps_best <= (1+delta*) eps_inf
  C2  half-slope   admissible iff eps_cheapest >= K* eps_inf

Part 1 verifies P1-P7, C1-C2 numerically at 50-digit precision.
Part 2 tests every rule on every single-resource Pareto frontier of the seven
corpora, against the canonical choice of the full least-squares fit.

Writes (results/):
    rules_theorem_verification.csv    Table 7
    table_rules_per_frontier.csv      one row per single-resource frontier
    table_rules_summary.csv           Table 8 (rule x statistic)
    table_rules_by_field.csv          per-field breakdown
    table_rules_admissibility.csv     C1/C2 tested out of sample
    table_half_slope_sensitivity.csv  reference-window sensitivity
    table_nonconcavity_by_field.csv   Diagnostic 4 by field
    table_floor_label_noise.csv       Section 10.6 (ImageNet floors)
"""
import os, sys, warnings
import numpy as np, pandas as pd
import mpmath as mp
from scipy.optimize import least_squares, lsq_linear, brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import save, ROOT  # noqa: E402

warnings.filterwarnings("ignore")
mp.mp.dps = 50
RNG = np.random.default_rng(20260910)
R_BAND = 0.05                                   # regret tolerance, nats


# ============================================================ closed forms ===
def q_minus(r=R_BAND):
    """Lower reducible share at which the regret equals r nats."""
    return (1.0 - np.sqrt(1.0 - np.exp(-2.0 * r))) / 2.0


def regret_share(q):
    """Regret (nats) of operating at reducible share q under the canonical price."""
    q = np.asarray(q, float)
    out = np.full(q.shape, np.inf)
    ok = (q > 0) & (q < 1)
    out[ok] = 0.5 * np.log(1.0 / (4.0 * q[ok] * (1.0 - q[ok])))
    return out


def regret_eps(eps, einf):
    return regret_share(1.0 - einf / np.asarray(eps, float))


QM = q_minus()
BAND = (1.0 / (1.0 - QM), 1.0 / QM)             # eps/eps_inf admissible band
DELTA_STAR = BAND[1] / 2.0 - 1.0                # C1: eps_best <= (1+d*) eps_inf
DELTA_LOW = BAND[0] / 2.0 - 1.0                 # C1 lower side (binds only if
                                                # eps_best < eps_inf, i.e. noise)
K_STAR = 1.0 / (1.0 - 2.0 * QM)                 # C2: eps_0   >= K* eps_inf


# ============================================== Part 1: exact verification ===
def verify():
    rows = []

    def add(res, case, qty, dev):
        rows.append(dict(result=res, cases=case, quantity=qty,
                         max_deviation=float(dev)))

    inst = []
    for _ in range(20):
        inst.append(dict(e0=mp.mpf(RNG.uniform(0.01, 0.3)),
                         c=mp.mpf(RNG.uniform(0.2, 3.0)),
                         a=mp.mpf(RNG.uniform(0.2, 2.0))))

    d1 = d2 = d3 = d4 = d5 = d6 = d7 = 0
    for P in inst:
        e0, c, a = P["e0"], P["c"], P["a"]
        S = lambda x: -mp.log(e0 + c * mp.e ** (-a * x))
        p1 = lambda x: c * mp.e ** (-a * x) / (e0 + c * mp.e ** (-a * x))
        x0 = mp.log(c / e0) / a                          # canonical, zeta = 0
        for x in mp.linspace(x0 - 6 / a, x0 + 6 / a, 13):
            d1 = max(d1, abs(mp.diff(S, x) - a * p1(x)))                  # P1
            d2 = max(d2, abs(mp.diff(S, x, 2) + a ** 2 * p1(x) * (1 - p1(x))))  # P2
        # P2: argmax |S''| is the canonical point (S''' = 0 there)
        xk = mp.findroot(lambda x: mp.diff(S, x, 3), x0 + mp.mpf("0.3") / a)
        d3 = max(d3, abs(a * xk - a * x0))
        # P1: slope at canonical point is exactly half the asymptotic slope
        d4 = max(d4, abs(mp.diff(S, x0) - a / 2))
        # P3: remaining information at canonical point is ln 2
        d5 = max(d5, abs(-mp.log(e0) - S(x0) - mp.log(2)))
        # P4: buying fraction f of the last bit costs (2^(1-f)-1)^(-1/a)
        for f in (mp.mpf("0.5"), mp.mpf("0.9"), mp.mpf("0.99")):
            m = (2 ** (1 - f) - 1) ** (-1 / a)
            d6 = max(d6, abs((-mp.log(e0) - S(x0 + mp.log(m))) - (1 - f) * mp.log(2)))
        # P5: regret identity  sigma*(lam*) - sigma_{lam*}(x_q) = 1/2 ln(1/(4q(1-q)))
        lam = a / 2
        for q in (mp.mpf("0.1"), mp.mpf("0.3"), mp.mpf("0.5"), mp.mpf("0.8")):
            xq = x0 - mp.log(q / (1 - q)) / a          # share q at xq
            R = (S(x0) - lam * x0) - (S(xq) - lam * xq)
            d7 = max(d7, abs(R - mp.log(1 / (4 * q * (1 - q))) / 2))
    add("P1 elasticity", "20 random (eps_inf,c,alpha), 13 points each",
        "-dS/dx vs alpha p1", d1)
    add("P2 curvature", "same", "d2S/dx2 vs -alpha^2 p1(1-p1)", d2)
    add("P2 knee", "20 instances", "argmax|S''| vs canonical zeta=0", d3)
    add("P1 half-slope", "20 instances", "slope at zeta=0 vs alpha/2", d4)
    add("P3 one bit", "20 instances", "S_inf - S(zeta=0) vs ln 2", d5)
    add("P4 last bit", "20 instances x f in {.5,.9,.99}",
        "remaining info vs (1-f) ln 2", d6)
    add("P5 regret", "20 instances x 4 shares",
        "sup-gap vs 1/2 ln(1/(4q(1-q)))", d7)
    # P5 band end points
    qm = mp.mpf(QM)
    add("P5 band", f"r={R_BAND}", "R(q_-) vs r",
        abs(mp.log(1 / (4 * qm * (1 - qm))) / 2 - R_BAND))

    # P6: Hessian = -Cov_p(a) and <= 0, general (n, N)
    d8, worst = 0, -np.inf
    for n, N in [(1, 1), (2, 2), (2, 4), (3, 3), (3, 5)]:
        for _ in range(5):
            A = [[mp.mpf(RNG.uniform(0, 2)) for _ in range(n)] for _ in range(N)]
            A = [[mp.mpf(0)] * n] + A
            cc = [mp.mpf(RNG.uniform(0.05, 2)) for _ in range(N + 1)]
            x = [mp.mpf(RNG.uniform(-2, 2)) for _ in range(n)]
            v = [mp.mpf(RNG.normal()) for _ in range(n)]
            def S_dir(s):
                return -mp.log(sum(cc[j] * mp.e ** (-sum(A[j][i] * (x[i] + s * v[i])
                                                           for i in range(n)))
                                   for j in range(N + 1)))
            w = [cc[j] * mp.e ** (-sum(A[j][i] * x[i] for i in range(n)))
                 for j in range(N + 1)]
            Z = sum(w); p = [wj / Z for wj in w]
            av = [sum(A[j][i] * v[i] for i in range(n)) for j in range(N + 1)]
            mean = sum(p[j] * av[j] for j in range(N + 1))
            var = sum(p[j] * (av[j] - mean) ** 2 for j in range(N + 1))
            h = mp.diff(S_dir, 0, 2)
            d8 = max(d8, abs(h + var)); worst = max(worst, float(h))
    add("P6 concavity", "(n,N) in (1,1),(2,2),(2,4),(3,3),(3,5); 5 each",
        "v'Hv vs -Var_p(a.v)", d8)
    add("P6 concavity", "same", "max v'Hv (must be <= 0)", worst)

    # P7: Aitken exactness on geometric spacing; general three-point solve
    d9 = d10 = 0
    for P in inst:
        e0, c, a = (float(P[k]) for k in ("e0", "c", "a"))
        eps = lambda t: e0 + c * t ** (-a)
        r = RNG.uniform(1.5, 8.0)
        e1, e2, e3 = eps(1.0), eps(r), eps(r * r)
        d9 = max(d9, abs((e1 * e3 - e2 ** 2) / (e1 + e3 - 2 * e2) - e0) / e0)
        t = np.sort(RNG.uniform(0.5, 50.0, 3))
        est = three_point(np.log(t), eps(t))
        d10 = max(d10, abs(est[0] - e0) / e0)
    add("P7 Aitken", "20 instances, geometric spacing", "relative error of eps_inf", d9)
    add("P7 three-point", "20 instances, arbitrary spacing",
        "relative error of eps_inf", d10)

    # C1, C2: thresholds make the regret exactly r
    add("C1 2x-best", f"delta*={DELTA_STAR:.4f}",
        "R at eps_best=(1+delta*)eps_inf vs r",
        abs(regret_eps(2 * (1 + DELTA_STAR), 1.0) - R_BAND))
    add("C2 half-slope", f"K*={K_STAR:.4f}",
        "R at share p1(0)/2 vs r",
        abs(regret_share((1 - 1 / K_STAR) / 2) - R_BAND))
    return pd.DataFrame(rows)


# ====================================================== frontier utilities ===
def pareto1d(x, e):
    order = np.lexsort((e, x)); best = np.inf; keep = []
    for i in order:
        if e[i] < best - 1e-12:
            keep.append(i); best = e[i]
    return np.array(keep, int)


def fit_law(x, e):
    """eps = einf + c exp(-alpha (x - x_min)); x centred for conditioning only."""
    u = x - x.min()
    r = least_squares(lambda t: t[0] + t[1] * np.exp(-t[2] * u) - e,
                      [0.8 * e.min(), max(e.max() - e.min(), 1e-3), 0.5],
                      bounds=([0, 0, 1e-4], [0.9, 1e3, 6.0]), max_nfev=100000)
    einf, c, a = r.x
    return einf, c, a, np.sqrt(np.mean(np.log(e / (einf + c * np.exp(-a * u))) ** 2))


def three_point(x3, e3):
    """eps_inf, c, alpha (x measured from x3[0]) from three frontier points.
    Returns (nan, nan, nan) when the three points admit no positive floor."""
    x3 = np.asarray(x3, float); e3 = np.asarray(e3, float)
    u2, u3 = x3[1] - x3[0], x3[2] - x3[0]
    d12, d23 = e3[0] - e3[1], e3[1] - e3[2]
    if not (u2 > 0 and u3 > u2 and d12 > 0 and d23 > 0):
        return np.nan, np.nan, np.nan
    rho = d12 / d23
    g = lambda a: (1 - np.exp(-a * u2)) / (np.exp(-a * u2) - np.exp(-a * u3)) - rho
    lo, hi = 1e-6, 50.0 / u3
    try:
        if g(lo) * g(hi) > 0:
            return np.nan, np.nan, np.nan
        a = brentq(g, lo, hi, xtol=1e-14)
    except ValueError:
        return np.nan, np.nan, np.nan
    c = d12 / (1 - np.exp(-a * u2))
    einf = e3[0] - c
    return (einf, c, a) if einf > 0 else (np.nan, np.nan, np.nan)


def upper_hull(x, S):
    h = []
    for i in range(len(x)):
        while len(h) >= 2:
            o, p = h[-2], h[-1]
            if (x[p] - x[o]) * (S[i] - S[o]) - (S[p] - S[o]) * (x[i] - x[o]) >= 0:
                h.pop()
            else:
                break
        h.append(i)
    return np.array(h, int)


def rule_half_slope(x, S, ref_frac=0.25):
    H = upper_hull(x, S)
    if len(H) < 3:
        return None
    xr = x.min() + ref_frac * np.ptp(x)
    s_ref = (np.interp(xr, x[H], S[H]) - S[H[0]]) / (xr - x[H[0]])
    s = np.diff(S[H]) / np.diff(x[H])
    below = np.where(s < s_ref / 2)[0]
    return int(H[below[0]]) if len(below) else int(H[-1])


def rule_knee(x, S):
    H = upper_hull(x, S)
    if len(H) < 3:
        return None
    s = np.diff(S[H]) / np.diff(x[H])
    d2 = (s[1:] - s[:-1]) / ((x[H[2:]] - x[H[:-2]]) / 2)
    return int(H[1 + np.argmin(d2)])


def nonconcavity(x, S):
    """RMS residual (nats) of the least-squares concave fit (Hildreth 1954)."""
    knots = np.unique(x)[1:-1]
    M = np.column_stack([np.ones_like(x), x] +
                        [-np.maximum(x - k, 0) for k in knots])
    lb = np.r_[-np.inf, -np.inf, np.zeros(len(knots))]
    r = lsq_linear(M, S, bounds=(lb, np.full(M.shape[1], np.inf)))
    return float(np.sqrt(np.mean((M @ r.x - S) ** 2)))


def nearest(x, target):
    return int(np.argmin(np.abs(x - target)))


def pick_by_three(x, e, idx):
    einf, c, a = three_point(x[idx], e[idx])
    if not np.isfinite(einf):
        return None, np.nan
    xs = x[idx[0]] + np.log(c / einf) / a          # t* from the three models
    return nearest(x, xs), einf


# ================================================= Part 2: the corpora ======
def single_resource_frontiers():
    src = open(os.path.join(HERE, "09_all_fields.py")).read()
    ns = {"__name__": "loader"}
    exec(src.split("rows_f, rows_law, rows_N")[0], ns)
    seen = {}
    for field, d, costs in ns["load_standard"]() + ns["load_legacy"]():
        for name, sub in d.groupby("frontier"):
            name = str(name)
            base = "|".join(name.split("|")[:2]) if name.startswith("timm|") else name
            for c in costs:
                s = sub.dropna(subset=["eps", c]); s = s[s[c] > 0]
                x = np.log(s[c].to_numpy(float)); e = s.eps.to_numpy(float)
                if len(e) < 8:
                    continue
                k = pareto1d(x, e)
                if len(k) < 8:
                    continue
                key = (field, base, c)
                if key not in seen or len(k) > len(seen[key][3]):
                    seen[key] = (field, base, c, x[k], e[k])
    return list(seen.values())


def analyse(fr, floor_override=None):
    """One frontier, every rule.

    floor_override replaces the fitted floor everywhere it is used -- the
    canonical pick, the regrets, the admissibility flags -- and is how script
    165 re-reads this table with the bandwidth-free three-point floor in place
    of the three-parameter fit.  Left at None (the default) nothing changes and
    the output is the fitted-floor table this script has always written.
    """
    field, name, cost, x, e = fr
    S = -np.log(e)
    einf, c, a, law_rms = fit_law(x, e)
    if floor_override is not None:
        einf = float(floor_override)
    rec = dict(field=field, frontier=name, cost=cost, n_pareto=len(e),
               eps_min=e.min(), eps_max=e.max(), log_cost_span=np.ptp(x),
               eps_inf=einf, alpha=a, law_rms_nats=law_rms,
               nonconcavity_nats=nonconcavity(x, S))
    floor = (1e-3 < einf < 0.5) and (a < 5.99)
    attain = floor and (e.min() <= 2 * einf <= e.max())
    rec.update(floor=floor, attainable=attain)
    if not floor:
        return rec
    u = x - x.min()
    zeta = a * u - np.log(c / einf)
    x_star = x.min() + np.log(c / einf) / a
    picks = dict(
        canonical=nearest(np.log(e), np.log(2 * einf)),
        two_x_best=nearest(np.log(e), np.log(2 * e.min())),
        half_slope=rule_half_slope(x, S),
        knee=rule_knee(x, S),
        accuracy_max=len(e) - 1,
        cheapest=0)
    # ---- the rules named in the literature, for the head-to-head table -----
    # (a) accuracy per unit resource: scale-free in its argmax, so a change of
    #     unit cannot move it; (b) the ratio of accuracy to log10 cost used as
    #     an "efficiency score", which is NOT scale-free and is undefined below
    #     one in its own unit; (c) a hard budget, the rule that returns the
    #     feasible end of the frontier rather than a point inside it.
    cost, acc = np.exp(x), 1.0 - e

    def _ratio_log10(c):
        L = np.log10(c)
        if not (L > 0).any():
            return None
        return int(np.argmax(np.where(L > 0, acc / np.where(L > 0, L, 1.0),
                                      -np.inf)))

    picks["acc_per_cost"] = int(np.argmax(acc / cost))
    picks["ratio_log10"] = _ratio_log10(cost)
    picks["budget_mid"] = int(np.max(np.where(x <= x.min()
                                              + 0.5 * np.ptp(x))[0]))
    alt = _ratio_log10(cost * 1e3)          # same frontier, resource x1000
    rec["ratio_log10_unitflip"] = bool(picks["ratio_log10"] is not None
                                       and alt is not None
                                       and alt != picks["ratio_log10"])
    full = np.array([0, nearest(x, x.min() + np.ptp(x) / 2), len(x) - 1])
    xc = x.min() + 0.6 * np.ptp(x)
    cheap = np.array([0, nearest(x, (x.min() + xc) / 2), nearest(x, xc)])
    picks["three_point"], e3f = pick_by_three(x, e, full)
    picks["three_point_cheap60"], e3c = (pick_by_three(x, e, cheap)
                                         if len(set(cheap)) == 3 else (None, np.nan))
    rec.update(eps_inf_3pt=e3f, eps_inf_3pt_cheap60=e3c,
               x_star=x_star, eps_cheapest_over_einf=e[0] / einf,
               eps_best_over_einf=e.min() / einf,
               adm_2x=(1 + DELTA_LOW) * einf <= e.min() <= (1 + DELTA_STAR) * einf,
               adm_half=e[0] >= K_STAR * einf)
    for r, i in picks.items():
        if i is None:
            rec.update({f"{r}_regret": np.nan, f"{r}_inband": np.nan,
                        f"{r}_dlogcost": np.nan, f"{r}_same": np.nan})
            continue
        R = float(regret_eps(e[i], einf))
        rec.update({f"{r}_regret": R, f"{r}_inband": R <= R_BAND,
                    f"{r}_dlogcost": float(x[i] - x[picks['canonical']]),
                    f"{r}_same": i == picks["canonical"]})
    for rf in (0.10, 0.50):                       # sensitivity, not tuning
        i = rule_half_slope(x, S, ref_frac=rf)
        rec[f"half_slope_ref{int(rf*100)}_regret"] = (np.nan if i is None
                                                     else float(regret_eps(e[i], einf)))
    ic = picks["canonical"]
    rec.update(bits_past_canonical=float(np.log2(e[ic] / e.min())),
               cost_mult_past_canonical=float(np.exp(x[-1] - x[ic])),
               zeta_best=float(zeta[-1]), zeta_canonical=float(zeta[ic]),
               bits_best_vs_2einf=float(np.log2(2 * einf / e.min())))
    return rec


RULES = ["canonical", "two_x_best", "half_slope", "knee", "three_point",
         "three_point_cheap60", "accuracy_max", "cheapest",
         "acc_per_cost", "ratio_log10", "budget_mid"]
FIT_FREE = {"canonical": "full fit", "two_x_best": "none", "half_slope": "none",
            "knee": "none", "three_point": "3 models", "three_point_cheap60":
            "3 cheap models", "accuracy_max": "none", "cheapest": "none",
            "acc_per_cost": "none", "ratio_log10": "none",
            "budget_mid": "a budget"}


def summarise(D):
    rows = []
    for r in RULES:
        R = D[f"{r}_regret"]
        ok = R.notna()
        rows.append(dict(rule=r, information=FIT_FREE[r], applicable=int(ok.sum()),
                         median_regret=float(np.median(R[ok])),   # inf allowed
                         n_below_fitted_floor=int(np.isinf(R[ok]).sum()),
                         frac_inband=float(D.loc[ok, f"{r}_inband"].astype(float).mean()),
                         frac_same_as_canonical=float(D.loc[ok, f"{r}_same"].astype(float).mean()),
                         median_abs_dlogcost=float(D.loc[ok, f"{r}_dlogcost"].abs().median())))
    return pd.DataFrame(rows)


def floor_vs_labels(frs):
    """Section 10.6: ImageNet floors on original and ReaL labels, with bootstrap."""
    out = []
    for field, name, cost, x, e in frs:
        if not name.startswith("timm|"):
            continue
        ev = name.split("|")[1]
        est = []
        for _ in range(300):
            b = np.sort(RNG.choice(len(e), len(e), replace=True))
            try:
                est.append(fit_law(x[b], e[b])[0])
            except Exception:
                pass
        ei = fit_law(x, e)[0]
        out.append(dict(eval_set=ev, cost=cost, n_pareto=len(e), eps_min=e.min(),
                        eps_inf=ei, ci_lo=np.quantile(est, 0.05),
                        ci_hi=np.quantile(est, 0.95),
                        canonical_eps=2 * ei, canonical_top1=1 - 2 * ei * (1 - 1e-3)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    print("=== Part 1: exact verification (50-digit arithmetic) ===")
    V = verify(); save(V, "rules_theorem_verification.csv")
    print(V.to_string(index=False))
    print(f"\nband eps/eps_inf in [{BAND[0]:.3f}, {BAND[1]:.3f}] at R <= {R_BAND}")
    print(f"C1 delta in [{DELTA_LOW:.4f}, {DELTA_STAR:.4f}];  C2 K* = {K_STAR:.4f}")

    print("\n=== Part 2: single-resource frontiers ===")
    frs = single_resource_frontiers()
    D = pd.DataFrame([analyse(f) for f in frs])
    save(D, "table_rules_per_frontier.csv")
    print(f"frontiers {len(D)}; floor identified {int(D.floor.sum())}; "
          f"canonical attainable {int(D.attainable.sum())}")
    A = D[D.attainable].copy()
    T = summarise(A); T.insert(0, "subset", "all attainable")
    Tm = summarise(A[A.field != "hpo"]); Tm.insert(0, "subset", "measured (no YAHPO)")
    TT = pd.concat([T, Tm]); save(TT, "table_rules_summary.csv")
    print(TT.round(3).to_string(index=False))

    byf = []
    for f, g in A.groupby("field"):
        s = summarise(g); s.insert(0, "field", f); s.insert(1, "frontiers", len(g))
        byf.append(s)
    save(pd.concat(byf), "table_rules_by_field.csv")

    adm = []
    for rule, col in [("two_x_best", "adm_2x"), ("half_slope", "adm_half")]:
        for flag in (True, False):
            g = A[(A[col] == flag) & A[f"{rule}_regret"].notna()]
            adm.append(dict(rule=rule, admissible=flag, frontiers=len(g),
                            frac_inband=float(g[f"{rule}_inband"].astype(float).mean())
                            if len(g) else np.nan,
                            median_regret=float(g[f"{rule}_regret"].median())
                            if len(g) else np.nan))
    ADM = pd.DataFrame(adm); save(ADM, "table_rules_admissibility.csv")
    print("\nadmissibility conditions C1, C2 (tested, not fitted):")
    print(ADM.round(3).to_string(index=False))

    sens = [dict(ref_frac=rf, frac_inband=float((A[col] <= R_BAND).mean()),
                 median_regret=float(A[col].median()))
            for rf, col in [(0.10, "half_slope_ref10_regret"),
                            (0.25, "half_slope_regret"),
                            (0.50, "half_slope_ref50_regret")]]
    SENS = pd.DataFrame(sens); save(SENS, "table_half_slope_sensitivity.csv")
    print("\nhalf-slope sensitivity to the reference window:")
    print(SENS.round(3).to_string(index=False))

    print("\nthree-point floor, |eps_inf_3pt/eps_inf - 1| (median):",
          f"full {np.nanmedian(np.abs(A.eps_inf_3pt / A.eps_inf - 1)):.3f}",
          f"cheap60 {np.nanmedian(np.abs(A.eps_inf_3pt_cheap60 / A.eps_inf - 1)):.3f}")
    print("information bought past the canonical point: median",
          f"{A.bits_past_canonical.median():.2f} bits at median cost x"
          f"{A.cost_mult_past_canonical.median():.1f};",
          f"max {A.bits_past_canonical.max():.2f};",
          f"frac > 1 bit {(A.bits_past_canonical > 1).mean():.2f}")
    print("non-concavity (nats): median", f"{D.nonconcavity_nats.median():.4f};",
          "law rms median", f"{D.law_rms_nats.median():.4f};",
          "frac NCI > 0.05:", f"{(D.nonconcavity_nats > 0.05).mean():.2f}")
    print("NCI floor vs no-floor medians:",
          D.groupby("floor").nonconcavity_nats.median().round(4).to_dict())

    NC = D.groupby("field").agg(frontiers=("frontier", "size"),
                                median_nci_nats=("nonconcavity_nats", "median"),
                                frac_nci_above_005=("nonconcavity_nats",
                                                    lambda v: float((v > 0.05).mean())),
                                median_law_rms_nats=("law_rms_nats", "median")).reset_index()
    save(NC, "table_nonconcavity_by_field.csv")
    print(NC.round(3).to_string(index=False))

    L = floor_vs_labels(frs); save(L, "table_floor_label_noise.csv")
    print("\nImageNet floors:"); print(L.round(4).to_string(index=False))

#!/usr/bin/env python3
"""T10  The frontier as a path: Fisher-Rao length and the cost of saturation.

Theory (Proposition 'path length' and Corollary 'cost of saturation', main
text).  Along a resource ray, with p_x the tilted spectral measure,
S = -ln eps, s = S' and kappa = -S'' in x = ln t, the tilted family is a
one-parameter exponential family whose Fisher information is exactly kappa.
Its Fisher-Rao length over a window is

      L(x1,x2) = int_{x1}^{x2} sqrt(kappa(u)) du                       (L1)

which needs no floor.  Then

  (i)   two atoms (one channel + floor):
            L = 2[ arcsin sqrt(p(x1)) - arcsin sqrt(p(x2)) ],  p = 1 - einf/eps
        and the length of the complete transition is exactly pi, for every
        alpha, every einf and every scale;
  (ii)  a single atom (no floor, pure power law): L = 0;
  (iii) K+1 atoms: L_total >= pi, with L_total -> K*pi as the atoms separate;
  (iv)  Gamma spectrum (no atom at 0): g = kappa/s^2 is constant = 1/k and
        L_total = infinity.  So g constant <=> gapless Gamma, and g
        increasing is the signature of an atom at the origin;
  (v)   speed limit (Cauchy-Schwarz):  L^2 <= (x2-x1) * (s(x1)-s(x2)),
        hence a transition that still has Fisher-Rao length pi - L to run
        needs at least (pi-L)^2 / s(x2) further nats of log-resource.

Part 0 verifies (i)-(v) at 50-digit precision on random spectra.
Part 1 computes L, g-drift and the floor-free local floor eps*g/(1+g) on
every single-resource frontier of the corpus, tests the dichotomy of (iv) by
a sign test, and reports the remaining-resource bound of (v).
Part 2 runs the truncation experiment (floor-free local floor versus a
three-parameter refit on a truncated window).
Part 3 emits the two-candidate worked example of Section 'A student example'.

Writes results/table_path_verification.csv, table_path_per_frontier.csv,
table_path_summary.csv, table_path_truncation.csv, table_path_example.csv.
"""
import numpy as np, pandas as pd, mpmath as mp, warnings
warnings.filterwarnings("ignore")
from scipy import stats, optimize
from _loaders import selection_rules, save

sr = selection_rules()
RNG = np.random.default_rng(20260915)
mp.mp.dps = 50


# ===================================================== Part 0: identities ==
def _S_factory(a, c):
    eps = lambda x: sum(cj * mp.e ** (-aj * x) for cj, aj in zip(c, a))
    S = lambda x: -mp.log(eps(x))
    kap = lambda x: -mp.diff(S, x, 2)
    return eps, S, kap


def verify():
    d_two = d_pi = d_single = d_gamma = 0.0
    worst_cs = -mp.inf
    min_total = mp.inf
    d_sep = 0.0

    # (i) + (ii) + (v): two atoms, random alpha / floor / scale
    for _ in range(25):
        alpha = mp.mpf(RNG.uniform(0.15, 3.0))
        einf = mp.mpf(RNG.uniform(0.01, 0.4))
        C = mp.mpf(RNG.uniform(0.1, 5.0))
        a, c = [mp.mpf(0), alpha], [einf, C]
        eps, S, kap = _S_factory(a, c)
        x1 = mp.mpf(RNG.uniform(-3, 0)); x2 = x1 + mp.mpf(RNG.uniform(0.5, 5))
        L = mp.quad(lambda u: mp.sqrt(abs(kap(u))), [x1, x2])
        p = lambda x: 1 - einf / eps(x)
        Lc = 2 * (mp.asin(mp.sqrt(p(x1))) - mp.asin(mp.sqrt(p(x2))))
        d_two = max(d_two, float(abs(L - Lc)))
        # complete transition (window wide enough that both ends are flat)
        w = 60 / alpha
        T = mp.quad(lambda u: mp.sqrt(abs(kap(u))), [-w, 0, w])
        d_pi = max(d_pi, float(abs(T - mp.pi)))
        # Cauchy-Schwarz speed limit
        cs = L ** 2 - (x2 - x1) * (mp.diff(S, x1) - mp.diff(S, x2))
        worst_cs = max(worst_cs, cs)

    # (ii) single atom, no floor
    for _ in range(10):
        a, c = [mp.mpf(RNG.uniform(0.1, 3))], [mp.mpf(RNG.uniform(0.1, 3))]
        _, S, kap = _S_factory(a, c)
        d_single = max(d_single, float(abs(mp.quad(lambda u: mp.sqrt(abs(kap(u))), [-2, 2]))))

    # (iii) K+1 atoms: total length >= pi ; separated atoms -> K*pi
    for _ in range(10):
        K = int(RNG.integers(1, 4))
        a = [mp.mpf(0)] + [mp.mpf(RNG.uniform(0.2, 3.0)) for _ in range(K)]
        c = [mp.mpf(RNG.uniform(0.02, 0.4))] + [mp.mpf(RNG.uniform(0.1, 3)) for _ in range(K)]
        _, S, kap = _S_factory(a, c)
        w = 60 / min(float(v) for v in a[1:])
        T = mp.quad(lambda u: mp.sqrt(abs(kap(u))), [-w, 0, w])
        min_total = min(min_total, T)
    for K in (2, 3):
        # a_j = j and crossovers placed at x = 0, -D, -2D, ... : well separated
        D = mp.mpf(40)
        a = [mp.mpf(j) for j in range(K + 1)]
        lc = [mp.mpf(0)]
        for j in range(K):
            lc.append(lc[-1] - j * D)
        c = [mp.e ** v for v in lc]
        _, S, kap = _S_factory(a, c)
        nodes = sorted({-float((K - 1) * D) - 30.0, 30.0} |
                       {-float(j * D) for j in range(K)})
        T = mp.quad(lambda u: mp.sqrt(abs(kap(u))), nodes)
        d_sep = max(d_sep, float(abs(T - K * mp.pi)))

    # (iv) Gamma spectrum: eps(x) = (1 + x/beta)^-k  =>  g = 1/k exactly
    for _ in range(10):
        k = mp.mpf(RNG.uniform(0.3, 6)); beta = mp.mpf(RNG.uniform(0.5, 4))
        S = lambda x: k * mp.log(1 + x / beta)
        for x in mp.linspace(mp.mpf('0.1'), 6, 6):
            g = -mp.diff(S, x, 2) / mp.diff(S, x) ** 2
            d_gamma = max(d_gamma, float(abs(g - 1 / k)))

    return pd.DataFrame([
        dict(result="L = 2[asin sqrt(p1) - asin sqrt(p2)] (two atoms)", max_deviation=d_two),
        dict(result="complete transition length = pi (two atoms)", max_deviation=d_pi),
        dict(result="L = 0 for a single atom (pure power law)", max_deviation=d_single),
        dict(result="min total length over K+1 atoms minus pi (must be >= 0)",
             max_deviation=float(min_total - mp.pi)),
        dict(result="separated K+1 atoms: total length = K*pi", max_deviation=d_sep),
        dict(result="Gamma spectrum: g = 1/k constant", max_deviation=d_gamma),
        dict(result="speed limit L^2 - dx*(s1-s2) <= 0 (max value)", max_deviation=float(worst_cs))])


# ================================================== local estimator (as 24/32)
def local(x, S, xc, h):
    w = np.exp(-0.5 * ((x - xc) / h) ** 2)
    M = np.column_stack([np.ones_like(x), x - xc, 0.5 * (x - xc) ** 2])
    W = np.sqrt(w)[:, None]
    b, *_ = np.linalg.lstsq(M * W, S * W[:, 0], rcond=None)
    return b[0], b[1], -b[2]                                   # S, s, kappa


def profile(x, S, ng=41, trim=0.10):
    span = np.ptp(x); h = span / 4
    gr = np.linspace(x.min() + trim * span, x.max() - trim * span, ng)
    o = np.array([local(x, S, xc, h) for xc in gr])
    return gr, o[:, 0], o[:, 1], o[:, 2]


def summarise(x, e):
    S = -np.log(e)
    gr, Sf, s, kap = profile(x, S)
    ok = s > 1e-6
    if ok.sum() < 10:
        return None
    u, Sf, s, kap = gr[ok], Sf[ok], s[ok], kap[ok]
    L = np.trapezoid(np.sqrt(np.clip(kap, 0, None)), u)
    g = kap / s ** 2
    ah = s + kap / s
    eps = np.exp(-Sf)
    floor_loc = eps * kap / (s ** 2 + kap)
    k3 = max(1, len(u) // 3)
    return dict(L=L, span=np.ptp(x), frac_pi=L / np.pi,
                g_med=float(np.median(g)), g_lo=float(np.median(g[:k3])),
                g_hi=float(np.median(g[-k3:])),
                dlng_du=float(np.polyfit(u, np.log(np.clip(g, 1e-12, None)), 1)[0]),
                alpha_hat=float(np.median(ah)), s1=float(s[0]), s2=float(s[-1]),
                floor_loc=float(np.median(floor_loc)),
                eps1=float(eps[0]), eps2=float(eps[-1]),
                dx_remaining=float((np.pi - L) ** 2 / s[-1]) if s[-1] > 0 else np.nan,
                p_top=float(min(max(s[-1] / max(np.median(ah), 1e-9), 0.0), 1.0)),
                p_bot=float(min(max(s[0] / max(np.median(ah), 1e-9), 0.0), 1.0)),
                dx_to_5pct=_to_target(s[-1], float(np.median(ah)), 0.05),
                dx_to_1pct=_to_target(s[-1], float(np.median(ah)), 0.01))


def _to_target(s_top, alpha_hat, p_target):
    """Floor-free log-resource still needed to reach error share p_target,
    i.e. eps within p_target/(1-p_target) of the floor.  Two-atom model:
    p = s/alpha_hat and dp/dx = -alpha_hat p(1-p), so the logit is linear."""
    if not (alpha_hat > 0):
        return np.nan
    p = min(max(s_top / alpha_hat, 1e-9), 1 - 1e-9)
    if p <= p_target:
        return 0.0
    return float((np.log(p / (1 - p)) - np.log(p_target / (1 - p_target))) / alpha_hat)


def fit3(x, e):
    """eps = einf + C exp(-alpha x), fitted on log-error."""
    f = lambda q, x: np.exp(q[0]) + np.exp(q[1]) * np.exp(-np.exp(q[2]) * x)
    r = lambda q: np.log(np.clip(f(q, x), 1e-12, None)) - np.log(e)
    best = None
    for a0 in (0.2, 0.5, 1.0, 2.0):
        try:
            sol = optimize.least_squares(r, [np.log(max(e.min() * .5, 1e-8)), np.log(e.max()), np.log(a0)])
            if best is None or sol.cost < best.cost:
                best = sol
        except Exception:
            pass
    return (np.exp(best.x[0]), np.exp(best.x[2])) if best else (np.nan, np.nan)


# ================================================ Part 1 + 2: real frontiers
def frontiers():
    rows, trows = [], []
    for field, name, cost, x, e in sr.single_resource_frontiers():
        if len(e) < 12:
            continue
        x = np.asarray(x, float); e = np.asarray(e, float)
        r = summarise(x, e)
        if r is None:
            continue
        Ls, ds = [], []
        for _ in range(200):
            i = np.sort(RNG.choice(len(e), len(e), replace=True))
            if len(np.unique(x[i])) < 8:
                continue
            rr = summarise(x[i], e[i])
            if rr:
                Ls.append(rr["L"]); ds.append(rr["dlng_du"])
        r.update(field=field, frontier=name, cost=cost, n=len(e),
                 L_se=float(np.std(Ls, ddof=1)) if len(Ls) > 20 else np.nan,
                 p_drift_pos=float(np.mean(np.array(ds) > 0)) if len(ds) > 20 else np.nan)
        # floor-free length prediction from the two end slopes
        q1 = np.clip(r["s1"] / r["alpha_hat"], 0, 1); q2 = np.clip(r["s2"] / r["alpha_hat"], 0, 1)
        r["L_pred"] = 2 * (np.arcsin(np.sqrt(q1)) - np.arcsin(np.sqrt(q2)))
        einf_full, alpha_full = fit3(x, e)
        r["einf_fit"], r["alpha_fit"] = einf_full, alpha_full
        rows.append(r)

        if np.ptp(x) >= 1.0 and len(e) >= 15:
            for frac in (0.7, 0.5):
                m = x <= x.min() + frac * np.ptp(x)
                if m.sum() < 12 or np.ptp(x[m]) < 0.8:
                    continue
                rc = summarise(x[m], e[m])
                if rc is None:
                    continue
                ec, _ = fit3(x[m], e[m])
                trows.append(dict(field=field, frontier=name, cost=cost, frac=frac,
                                  einf_full=einf_full, einf_refit=ec,
                                  einf_free=rc["floor_loc"], L_cut=rc["L"]))
    return pd.DataFrame(rows), pd.DataFrame(trows)


# ============================================== Part 3: the student example
def example():
    """Two candidates, one number.  Section 'A student example'."""
    cand = [dict(model="EVA-02-L/448", eps=0.0995, params_M=305.08),
            dict(model="EVA-02-B/448", eps=0.1133, params_M=87.12)]
    a, b = cand
    d_eps = np.log(b["eps"] / a["eps"])                   # nats of error bought
    d_x = np.log(a["params_M"] / b["params_M"])           # log-resource spent
    alpha_star = d_eps / d_x                              # break-even price
    alpha = 0.4583229751448533                            # frontier price, params
    rows = []
    for c in cand:
        rows.append(dict(model=c["model"], eps=c["eps"], params_M=c["params_M"],
                         score=c["eps"] * c["params_M"] ** alpha))
    D = pd.DataFrame(rows)
    D["regret_nats"] = np.log(D.score / D.score.min())
    D["alpha_star"] = alpha_star
    D["alpha_frontier"] = alpha
    D["winner"] = D.score == D.score.min()
    return D


# =============================================================== run =======
def main():
    V = verify(); save(V, "table_path_verification.csv")
    print(V.to_string(index=False))

    F, T = frontiers()
    save(F, "table_path_per_frontier.csv")
    W = F[(F.span >= 1.0) & (F.n >= 15)].copy()

    lam = np.isfinite(W.L) & np.isfinite(W.L_pred)
    k_pos = int((W.dlng_du > 0).sum())
    mm = np.isfinite(W.einf_fit) & (W.einf_fit > 1e-6) & np.isfinite(W.floor_loc)
    lr = np.log(W.floor_loc[mm] / W.einf_fit[mm])

    S = pd.DataFrame([
        dict(statistic="frontiers analysed", value=len(F)),
        dict(statistic="frontiers with span>=1 nat and n>=15", value=len(W)),
        dict(statistic="fraction with L < pi", value=float((W.L < np.pi).mean())),
        dict(statistic="max L / pi", value=float((W.L / np.pi).max())),
        dict(statistic="median L / pi (fraction of transition traversed)",
             value=float((W.L / np.pi).median())),
        dict(statistic="corr(L, floor-free prediction)",
             value=float(np.corrcoef(W.L[lam], W.L_pred[lam])[0, 1])),
        dict(statistic="median |L - prediction|",
             value=float(np.median(np.abs(W.L[lam] - W.L_pred[lam])))),
        dict(statistic="median d ln g / dx", value=float(W.dlng_du.median())),
        dict(statistic="fraction with d ln g / dx > 0", value=float((W.dlng_du > 0).mean())),
        dict(statistic="sign test p (g constant, i.e. Gamma)",
             value=float(stats.binomtest(k_pos, len(W), 0.5).pvalue)),
        dict(statistic="median g, lower third", value=float(W.g_lo.median())),
        dict(statistic="median g, upper third", value=float(W.g_hi.median())),
        dict(statistic="median log-ratio local floor / fitted floor", value=float(lr.median())),
        dict(statistic="corr(ln local floor, ln fitted floor)",
             value=float(np.corrcoef(np.log(W.floor_loc[mm]), np.log(W.einf_fit[mm]))[0, 1])),
        dict(statistic="median error share p at window bottom", value=float(W.p_bot.median())),
        dict(statistic="median error share p at window top", value=float(W.p_top.median())),
        dict(statistic="median further log-resource to reach p=0.05 (nats)",
             value=float(W.dx_to_5pct.median())),
        dict(statistic="median further resource to reach p=0.01 (decades)",
             value=float((W.dx_to_1pct / np.log(10)).median())),
        dict(statistic="median Cauchy-Schwarz bound on remaining log-resource (nats)",
             value=float(W.dx_remaining.median())),
        dict(statistic="median observed span (nats)", value=float(W.span.median()))])
    save(S, "table_path_summary.csv"); print("\n" + S.to_string(index=False))

    save(T, "table_path_truncation.csv")
    print("\ntruncation experiment")
    trows = []
    for frac in (0.7, 0.5):
        c = T[(T.frac == frac) & (T.einf_full > 1e-6)].dropna(subset=["einf_refit", "einf_free"])
        ef = np.abs(np.log(np.clip(c.einf_refit, 1e-12, None) / c.einf_full))
        el = np.abs(np.log(np.clip(c.einf_free, 1e-12, None) / c.einf_full))
        trows.append(dict(frac=frac, n=len(c), refit_median=float(ef.median()),
                          refit_q75=float(ef.quantile(.75)), free_median=float(el.median()),
                          free_q75=float(el.quantile(.75)),
                          wilcoxon_p=float(stats.wilcoxon(ef, el).pvalue),
                          spearman_L_err=float(stats.spearmanr(c.L_cut, el).statistic)))
    TT = pd.DataFrame(trows); save(TT, "table_path_truncation_summary.csv")
    print(TT.to_string(index=False))

    E = example(); save(E, "table_path_example.csv")
    print("\nworked example\n" + E.to_string(index=False))


if __name__ == "__main__":
    main()

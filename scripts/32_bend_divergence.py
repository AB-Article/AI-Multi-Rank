#!/usr/bin/env python3
"""T9  The selection law and the bend divergence.

Theory (Propositions 'selection law' and 'bend divergence', main text).
Along a resource ray, with p_x the tilted spectral measure, S = -ln eps,
s = S' and kappa = -S'':

  master equation   d p_x / dx = -(a - E_p[a]) p_x
  Fisher form       d s / d x  = -Var_p(a) = -kappa
  integrated form   B(x1,x2) := [S(x2)-S(x1)] - (x2-x1) s(x2)
                              = KL( p_{x2} || p_{x1} )
                              = int_{x1}^{x2} (u - x1) kappa(u) du   >= 0,
  with B = 0 iff mu is a single atom.

B is the Bregman divergence of the convex function -S.  It is floor-free
(no epsinf) and needs only S at two points plus ONE first derivative, where
the pointwise width g = kappa/s^2 needs a second derivative.  The normalised
bend is

  Delta := B / [S(x2) - S(x1)]  in  [0,1),

zero for a single exponential, tending to 1 as the frontier reaches its floor.

Part 0 verifies the three identities at 50-digit precision on random spectra.
Part 1 computes B, Delta and a bootstrap z-score Z_B on every single-resource
frontier and compares Z_B with the pointwise criterion g*V of
Proposition 'spectral resolution' as a predictor of the BIC channel count.

Writes results/table_bend_verification.csv,
results/table_bend_per_frontier.csv, results/table_bend_summary.csv.
"""
import numpy as np, pandas as pd, mpmath as mp
from _loaders import selection_rules, all_fields, save

sr = selection_rules()
fit_channels = all_fields()["fit_channels"]
RNG = np.random.default_rng(20260914)
mp.mp.dps = 50


# ------------------------------------------------------ Part 0: identities --
def verify():
    """Check B = KL(p2||p1) = int (u-x1) kappa du, and B = 0 iff single atom."""
    d_kl = d_int = d_single = 0.0
    worst_neg = np.inf
    for _ in range(40):
        K = int(RNG.integers(1, 5))
        a = [mp.mpf(0)] + [mp.mpf(RNG.uniform(0.1, 3.0)) for _ in range(K)]
        c = [mp.mpf(RNG.uniform(0.02, 0.4))] + [mp.mpf(RNG.uniform(0.1, 3)) for _ in range(K)]
        eps = lambda x: sum(cj * mp.e ** (-aj * x) for cj, aj in zip(c, a))
        S = lambda x: -mp.log(eps(x))
        kap = lambda x: -mp.diff(S, x, 2)

        def shares(x):
            w = [cj * mp.e ** (-aj * x) for cj, aj in zip(c, a)]
            Z = sum(w)
            return [wj / Z for wj in w]

        x1 = mp.mpf(RNG.uniform(-2.0, 0.5))
        x2 = x1 + mp.mpf(RNG.uniform(0.5, 4.0))
        B = (S(x2) - S(x1)) - (x2 - x1) * mp.diff(S, x2)
        p1, p2 = shares(x1), shares(x2)
        KL = sum(q * mp.log(q / r) for q, r in zip(p2, p1) if q > 0)
        I = mp.quad(lambda u: (u - x1) * kap(u), [x1, x2])
        d_kl = max(d_kl, float(abs(B - KL)))
        d_int = max(d_int, float(abs(B - I)))
        worst_neg = min(worst_neg, float(B))

    # single atom (no floor, one channel): B must vanish identically
    for _ in range(20):
        a1 = mp.mpf(RNG.uniform(0.1, 3.0))
        c1 = mp.mpf(RNG.uniform(0.1, 3.0))
        S = lambda x: -mp.log(c1 * mp.e ** (-a1 * x))
        x1 = mp.mpf(RNG.uniform(-2, 1)); x2 = x1 + mp.mpf(RNG.uniform(0.5, 4))
        B = (S(x2) - S(x1)) - (x2 - x1) * mp.diff(S, x2)
        d_single = max(d_single, float(abs(B)))

    return pd.DataFrame([
        dict(result="B = KL(p_x2 || p_x1)", max_deviation=d_kl),
        dict(result="B = int (u-x1) kappa(u) du", max_deviation=d_int),
        dict(result="min B over draws (must be >= 0)", max_deviation=worst_neg),
        dict(result="B = 0 for a single atom", max_deviation=d_single)])


# ------------------------------------------------------ local estimator ----
def local(x, S, xc, h):
    """Kernel-weighted local quadratic: returns (S, s, kappa) at xc."""
    w = np.exp(-0.5 * ((x - xc) / h) ** 2)
    M = np.column_stack([np.ones_like(x), x - xc, 0.5 * (x - xc) ** 2])
    W = np.sqrt(w)[:, None]
    b, *_ = np.linalg.lstsq(M * W, S * W[:, 0], rcond=None)
    return b[0], b[1], -b[2]


def bend(x, S):
    """B and Delta from the two ends of the window."""
    h = np.ptp(x) / 4
    x1, x2 = x.min(), x.max()
    S1, _, _ = local(x, S, x1, h)
    S2, s2, _ = local(x, S, x2, h)
    dS = S2 - S1
    B = dS - (x2 - x1) * s2
    return B, (B / dS if dS > 1e-9 else np.nan), dS, s2


def nhat1(x, e):
    b = {}
    for K in (1, 2, 3):
        try:
            b[K] = fit_channels(x[:, None], e, K, shared=False)["bic"]
        except Exception:
            pass
    return min(b, key=b.get) if b else np.nan


def auc(score, label):
    s = np.asarray(score, float); y = np.asarray(label, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    V = verify(); save(V, "table_bend_verification.csv"); print(V.to_string(index=False))

    width = None
    try:
        width = pd.read_csv(f"{sr.ROOT}/results/table_spectral_width.csv")
        width = width.set_index(["frontier", "cost"])
    except Exception:
        pass

    rows = []
    for field, name, cost, x, e in sr.single_resource_frontiers():
        if len(e) < 10:
            continue
        S = -np.log(e)
        B, D, dS, s2 = bend(x, S)
        boot = []
        for _ in range(300):
            i = np.sort(RNG.choice(len(e), len(e), replace=True))
            if len(np.unique(x[i])) < 6:
                continue
            b, *_ = bend(x[i], S[i])
            boot.append(b)
        boot = np.array([v for v in boot if np.isfinite(v)])
        se = boot.std(ddof=1) if len(boot) > 20 else np.nan
        rec = dict(field=field, frontier=name, cost=cost, n=len(e), span=np.ptp(x),
                   dS=dS, s_top=s2, B=B, Delta=D,
                   B_se=se, Z_B=(B / se if se and se > 0 else np.nan),
                   B_lo=np.quantile(boot, .05) if len(boot) > 20 else np.nan,
                   B_hi=np.quantile(boot, .95) if len(boot) > 20 else np.nan,
                   N_hat=nhat1(x, e))
        if width is not None and (name, cost) in width.index:
            w = width.loc[(name, cost)]
            w = w.iloc[0] if isinstance(w, pd.DataFrame) else w
            rec.update(g=float(w["g"]), V=float(w["V"]), gV=float(w["gV"]))
        rows.append(rec)

    P = pd.DataFrame(rows); save(P, "table_bend_per_frontier.csv")

    P["two"] = P.N_hat >= 2
    summ = [dict(statistic="frontiers", value=float(len(P))),
            dict(statistic="frac B >= 0 (point estimate)",
                 value=float((P.B >= -1e-9).mean())),
            dict(statistic="frac B < 0 significantly (membership failure)",
                 value=float((P.B_hi < 0).mean())),
            dict(statistic="frac s(x2) <= 0 (window has reached the floor)",
                 value=float((P.s_top <= 0).mean())),
            dict(statistic="median Delta (all)", value=float(P.Delta.median())),
            dict(statistic="median Delta | s(x2) > 0",
                 value=float(P.loc[P.s_top > 0, "Delta"].median())),
            dict(statistic="median Delta | s(x2) > 0, N_hat = 1",
                 value=float(P.loc[(P.s_top > 0) & ~P.two, "Delta"].median())),
            dict(statistic="median Delta | s(x2) > 0, N_hat >= 2",
                 value=float(P.loc[(P.s_top > 0) & P.two, "Delta"].median())),
            dict(statistic="AUC  Z_B -> N_hat >= 2", value=auc(P.Z_B, P.two)),
            dict(statistic="AUC  g*V -> N_hat >= 2",
                 value=auc(P.gV, P.two) if "gV" in P else np.nan),
            dict(statistic="AUC  V -> N_hat >= 2",
                 value=auc(P.V, P.two) if "V" in P else np.nan),
            dict(statistic="median Z_B | N_hat = 1",
                 value=float(P.loc[~P.two, "Z_B"].median())),
            dict(statistic="median Z_B | N_hat >= 2",
                 value=float(P.loc[P.two, "Z_B"].median()))]
    T = pd.DataFrame(summ); save(T, "table_bend_summary.csv")
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

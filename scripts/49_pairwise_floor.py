#!/usr/bin/env python3
"""THE FLOOR FROM ONE PAIR OF MODELS  (Remark in Section 9 of paper.tex).

Under the two-class reading of the spectral law -- a fraction eps_inf of
evaluation items carries the atom at the origin and is failed by every model,
the rest is failed independently by model i with probability w_i -- the joint
failure rate of two models identifies the floor with no fit, no resource and no
exponent:

    eps_1   = f + (1-f) w_1
    eps_2   = f + (1-f) w_2
    eps_12  = f + (1-f) w_1 w_2
  =>  f = Cov(F_1,F_2) / P(both correct)
         = (eps_12 - eps_1 eps_2) / (1 - eps_1 - eps_2 + eps_12).            (*)

Two further facts, both exact:
  * (*) must be fed raw failure rates, and returns a raw floor; the decision of
    Corollary 9 is then invariant to chance correction, because the harmonic
    mean and the floor are divided by the same 1-q_0;
  * appending a fraction u of always-failed items maps the estimator by
    f -> (f + u)/(1 + u), the same map the true floor undergoes.

Part 1 verifies (*) and the equivariance at 50-digit precision and by
simulation.  Part 2 (optional, --panel, needs network) builds an auxiliary
panel of real model pairs with per-item correctness, since the frontiers
archived here record aggregate accuracies only and cannot test (*).

Writes (results/):
    table_pairwise_floor_verification.csv   Part 1
    table_pairwise_floor_panel.csv          Part 2, per (dataset, family, axis)
    table_pairwise_floor_rules.csv          Part 2, ranking under eq. Rsigma
    table_pairwise_floor_groundtruth.csv    Part 2, injected-floor recovery
"""
import os, sys, argparse, itertools, warnings
import numpy as np, pandas as pd
import mpmath as mp

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import save, ROOT  # noqa: E402

mp.mp.dps = 50


def cov_floor(e1, e2, e12):
    D = 1 - e1 - e2 + e12
    return (e12 - e1 * e2) / D if D > 0 else float("nan")


# --------------------------------------------------------------------------- 1
def verify():
    rows = []
    rng = np.random.default_rng(20261003)
    worst_id, worst_eq, worst_scale = mp.mpf(0), mp.mpf(0), mp.mpf(0)
    for _ in range(2000):
        f = mp.mpf(float(rng.uniform(1e-3, 0.5)))
        w1 = mp.mpf(float(rng.uniform(1e-3, 0.99)))
        w2 = mp.mpf(float(rng.uniform(1e-3, 0.99)))
        e1, e2 = f + (1 - f) * w1, f + (1 - f) * w2
        e12 = f + (1 - f) * w1 * w2
        est = (e12 - e1 * e2) / (1 - e1 - e2 + e12)
        worst_id = max(worst_id, abs(est - f))
        u = mp.mpf(float(rng.uniform(0, 0.9)))             # appended unlearnable mass
        E1, E2, E12 = (e1 + u) / (1 + u), (e2 + u) / (1 + u), (e12 + u) / (1 + u)
        est2 = (E12 - E1 * E2) / (1 - E1 - E2 + E12)
        worst_eq = max(worst_eq, abs(est2 - (f + u) / (1 + u)))
        k = mp.mpf(float(rng.uniform(0.2, 0.95)))          # chance correction eps -> eps/k
        hm = 2 / (1 / e1 + 1 / e2)
        side_raw = 1 if hm > 2 * est else 0
        side_cc = 1 if (hm / k) > 2 * (est / k) else 0      # same inequality, rescaled
        worst_scale = max(worst_scale, mp.mpf(abs(side_raw - side_cc)))
    rows += [dict(statement="identity (*) recovers the floor", max_abs_error=float(worst_id)),
             dict(statement="equivariance f -> (f+u)/(1+u)", max_abs_error=float(worst_eq)),
             dict(statement="Cor. 9 decision invariant to chance correction", max_abs_error=float(worst_scale))]
    # finite-sample behaviour: binomial items, conditional independence holds
    rng = np.random.default_rng(7)
    for M in (1000, 10000, 100000):
        f, w1, w2 = 0.08, 0.30, 0.15
        atom = rng.random(M) < f
        F1 = atom | (rng.random(M) < w1)
        F2 = atom | (rng.random(M) < w2)
        est = cov_floor(F1.mean(), F2.mean(), (F1 & F2).mean())
        rows.append(dict(statement=f"simulation, M={M} items", max_abs_error=abs(est - f)))
    V = pd.DataFrame(rows)
    save(V.round(12), "table_pairwise_floor_verification.csv")
    print(V.to_string(index=False))


# --------------------------------------------------------------------------- 2
TAB = {"adult": "adult-all.csv", "phoneme": "phoneme.csv",
       "mammography": "mammography.csv", "abalone": "abalone.csv", "german": "german.csv"}
RAW = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
FASHION = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"


def _fetch(url, dest):
    import urllib.request
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def load(name, cache, seed=0, ntr=20000, nte=10000):
    import gzip
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    if name == "fashion":
        def idx_i(p):
            b = gzip.open(p, "rb").read()
            n = int.from_bytes(b[4:8], "big"); r = int.from_bytes(b[8:12], "big")
            c = int.from_bytes(b[12:16], "big")
            return np.frombuffer(b[16:], np.uint8).reshape(n, r * c)
        def idx_l(p):
            return np.frombuffer(gzip.open(p, "rb").read()[8:], np.uint8)
        X = idx_i(_fetch(FASHION + "train-images-idx3-ubyte.gz", f"{cache}/f_tri.gz")).astype(float) / 255
        y = idx_l(_fetch(FASHION + "train-labels-idx1-ubyte.gz", f"{cache}/f_trl.gz")).astype(int)
        Xt = idx_i(_fetch(FASHION + "t10k-images-idx3-ubyte.gz", f"{cache}/f_tei.gz")).astype(float) / 255
        yt = idx_l(_fetch(FASHION + "t10k-labels-idx1-ubyte.gz", f"{cache}/f_tel.gz")).astype(int)
    else:
        df = pd.read_csv(_fetch(RAW + TAB[name], f"{cache}/{TAB[name]}"), header=None,
                         na_values="?").dropna()
        yv = df.iloc[:, -1].values
        Xd = df.drop(df.columns[-1], axis=1)
        cols = []
        for c in Xd.columns:
            v = pd.to_numeric(Xd[c], errors="coerce")
            if v.isna().mean() > 0.05:
                v = pd.Series(pd.factorize(Xd[c].astype(str))[0], index=Xd.index)
            cols.append(v.fillna(v.median()).astype(float).values)
        Xa = np.column_stack(cols)
        _, ya = np.unique(yv, return_inverse=True)
        if name == "abalone":
            ya = np.digitize(ya, np.quantile(ya, [1 / 3, 2 / 3]))
        X, Xt, y, yt = train_test_split(Xa, ya, test_size=0.35, random_state=seed, stratify=ya)
    rng = np.random.default_rng(seed)
    if len(y) > ntr:
        i = rng.choice(len(y), ntr, replace=False); X, y = X[i], y[i]
    if len(yt) > nte:
        i = rng.choice(len(yt), nte, replace=False); Xt, yt = Xt[i], yt[i]
    sc = StandardScaler().fit(X)
    return sc.transform(X), y.astype(int), sc.transform(Xt), yt.astype(int)


def ladders(name, ntr):
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    big = name == "fashion"
    L = {"mlp_width": [MLPClassifier((w,), max_iter=20 if big else 300, random_state=0,
                                     learning_rate_init=3e-3) for w in (2, 4, 8, 16, 32, 64, 128, 256)],
         "rf_trees": [RandomForestClassifier(n_estimators=k, random_state=0, n_jobs=-1)
                      for k in ((1, 2, 4, 8, 16, 32, 64) if big else (1, 2, 4, 8, 16, 32, 64, 128))],
         "knn_n": [("sub", n, KNeighborsClassifier(5, algorithm="brute", n_jobs=-1))
                   for n in ((250, 500, 1000, 2000, 4000, 8000) if big else
                             (100, 200, 400, 800, 1600, 3200)) if n <= ntr]}
    if not big:
        L["gb_iter"] = [HistGradientBoostingClassifier(max_iter=m, random_state=0,
                        early_stopping=False) for m in (2, 4, 8, 16, 32, 64, 128, 256)]
        L["tree_depth"] = [DecisionTreeClassifier(max_depth=d, random_state=0)
                           for d in (1, 2, 3, 4, 6, 8, 12, None)]
    return L


def size_of(clf, nfeat):
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    if isinstance(clf, MLPClassifier):
        return float(sum(c.size for c in clf.coefs_) + sum(b.size for b in clf.intercepts_))
    if isinstance(clf, RandomForestClassifier):
        return float(sum(t.tree_.node_count for t in clf.estimators_))
    if isinstance(clf, DecisionTreeClassifier):
        return float(clf.tree_.node_count)
    if isinstance(clf, HistGradientBoostingClassifier):
        return float(sum(p.nodes.shape[0] for ps in clf._predictors for p in ps))
    if isinstance(clf, KNeighborsClassifier):
        return float(clf._fit_X.shape[0] * nfeat)
    return np.nan


def panel(cache):
    import time
    from scipy.optimize import curve_fit

    def fit_law(t, e):
        lt, le = np.log(t), np.log(e)
        f = lambda l, li, lc, la: np.logaddexp(li, lc - np.exp(la) * l)
        best = None
        for i0 in (0.3, 0.6, 0.9, 0.99):
            for a0 in (0.05, 0.3, 1.0, 3.0):
                try:
                    p, _ = curve_fit(f, lt, le, maxfev=40000,
                                     p0=[np.log(e.min() * i0), np.log(e.max()), np.log(a0)])
                    r = float(np.sum((f(lt, *p) - le) ** 2))
                    if best is None or r < best[1]:
                        best = (p, r)
                except Exception:
                    pass
        if best is None:
            return None
        p, r = best
        return (float(np.exp(p[0])), float(np.exp(p[2])), float(np.sqrt(r / len(e))))

    cells, pairs, gt = [], [], []
    for name in list(TAB) + ["fashion"]:
        Xtr, ytr, Xte, yte = load(name, cache)
        K = len(np.unique(ytr))
        for fam, specs in ladders(name, len(ytr)).items():
            C, T, Sz, PRED = [], [], [], []
            for m in specs:
                if isinstance(m, tuple):
                    _, n, clf = m
                    i = np.random.default_rng(0).choice(len(ytr), n, replace=False)
                    clf.fit(Xtr[i], ytr[i])
                else:
                    clf = m; clf.fit(Xtr, ytr)
                reps = 3 if len(yte) > 5000 else 7
                ts = []
                for _ in range(reps):
                    s = time.perf_counter(); yp = clf.predict(Xte); ts.append(time.perf_counter() - s)
                C.append(yp == yte); T.append(float(np.median(ts)))
                Sz.append(size_of(clf, Xtr.shape[1])); PRED.append(yp)
            C, T, Sz, PRED = np.array(C), np.array(T), np.array(Sz), np.array(PRED)
            eps = 1 - C.mean(1); M = len(yte)
            for axis, t_all in (("t_inf", T), ("size", Sz)):
                if not np.all(np.isfinite(t_all)) or np.ptp(np.log(t_all)) < 0.5:
                    continue
                o = np.argsort(t_all); keep, b = [], np.inf
                for i in o:
                    if eps[i] < b - 1e-12:
                        keep.append(i); b = eps[i]
                P = np.array(keep)
                if len(P) < 6:
                    continue
                e, t, Cp = eps[P], t_all[P], C[P]
                law = fit_law(t, e)
                if law is None:
                    continue
                f0, a, res = law
                interior = 0.05 * e.min() < f0 < 0.97 * e.min()
                fl = [cov_floor((~Cp[i]).mean(), (~Cp[j]).mean(), ((~Cp[i]) & (~Cp[j])).mean())
                      for i, j in itertools.combinations(range(len(P)), 2)]
                cells.append(dict(dataset=name, family=fam, axis=axis, M=M, n_pareto=len(P),
                                  eps_min=e.min(), eps_inf_fit=f0, alpha=a, resid=res,
                                  interior=interior, cov_floor_median=float(np.nanmedian(fl))))
                for i, j in itertools.combinations(range(len(P)), 2):
                    hi, lo = (i, j) if e[i] > e[j] else (j, i)
                    if e[hi] <= e[lo] or t[lo] <= t[hi]:
                        continue
                    ft2 = cov_floor((~Cp[hi]).mean(), (~Cp[lo]).mean(),
                                    ((~Cp[hi]) & (~Cp[lo])).mean())
                    marg = np.log(e[hi] / e[lo]) - a / 2 * np.log(t[lo] / t[hi])
                    pairs.append(dict(dataset=name, family=fam, axis=axis, interior=interior,
                                      resid=res, eps_hi=e[hi], eps_lo=e[lo],
                                      harmonic=2 / (1 / e[hi] + 1 / e[lo]),
                                      cov_floor=ft2, eps_inf_fit=f0,
                                      truth=bool(marg > 0), cost=abs(marg)))
            # ground truth: corrupt a known fraction of TEST labels
            o = np.argsort(T); keep, b = [], np.inf
            for i in o:
                if eps[i] < b - 1e-12:
                    keep.append(i); b = eps[i]
            P = np.array(keep)
            if len(P) < 3:
                continue
            base = None
            for eta in (0.0, 0.05, 0.10, 0.20):
                rng = np.random.default_rng(7)
                y2 = yte.copy()
                idx = rng.choice(M, int(eta * M), replace=False)
                y2[idx] = (yte[idx] + rng.integers(1, K, len(idx))) % K
                eff = float(np.mean(y2 != yte))
                Cc = PRED[P] == y2[None, :]
                v = [cov_floor((~Cc[i]).mean(), (~Cc[j]).mean(), ((~Cc[i]) & (~Cc[j])).mean())
                     for i, j in itertools.combinations(range(len(P)), 2)]
                med = float(np.nanmedian(v))
                if eta == 0.0:
                    base = med
                gt.append(dict(dataset=name, family=fam, eta_eff=eff, cov_floor_median=med,
                               predicted=eff + (1 - eff) * base,
                               rel_dev=(med / (eff + (1 - eff) * base) - 1) if base is not None else np.nan))
    Ce, Pa, Gt = pd.DataFrame(cells), pd.DataFrame(pairs), pd.DataFrame(gt)
    save(Ce.round(6), "table_pairwise_floor_panel.csv")
    save(Gt.round(6), "table_pairwise_floor_groundtruth.csv")
    q = Pa[Pa.interior]
    rows = []
    for nm, pick in (("harmonic test, pairwise covariance floor", q.harmonic > 2 * q.cov_floor),
                     ("harmonic test, fitted floor", q.harmonic > 2 * q.eps_inf_fit),
                     ("always the cheaper", pd.Series(np.zeros(len(q), bool), index=q.index))):
        w = pick != q.truth
        rows.append(dict(rule=nm, accuracy=1 - w.mean(),
                         mean_cost=float(np.where(w, q.cost, 0).mean()), n_pairs=len(q)))
    rows.append(dict(rule="median |cov_floor/fitted_floor - 1|",
                     accuracy=np.nan,
                     mean_cost=float(np.median(abs(q.cov_floor / q.eps_inf_fit - 1))), n_pairs=len(q)))
    R = pd.DataFrame(rows)
    save(R.round(6), "table_pairwise_floor_rules.csv")
    print(f"  cells {len(Ce)} (identified {int(Ce.interior.sum())}), pairs {len(Pa)}")
    print(R.round(4).to_string(index=False))
    print(f"  ground truth: median relative deviation from (f+u)/(1+u): "
          f"{np.nanmedian(Gt[Gt.eta_eff>0].rel_dev):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="store_true", help="build the auxiliary real panel (network)")
    ap.add_argument("--cache", default=os.path.join(ROOT, "data", "pairwise_cache"))
    a = ap.parse_args()
    verify()
    if a.panel:
        panel(a.cache)

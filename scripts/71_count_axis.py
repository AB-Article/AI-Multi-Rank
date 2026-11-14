#!/usr/bin/env python3
"""THE COUNT AXIS.  Law I as a theorem, and the nesting coefficient derived.
   (Section "The count axis" of paper.tex.)

Script 64 refuted the item-level mechanism behind the pairwise floor estimator
and found the nesting law Pr[both fail] = kappa min(h_i,h_j) with kappa = 0.835.
That left kappa as a measured constant with no derivation.  It has one.

Fix any resource that is a COUNT of independent attempts -- ensemble members,
retries, best-of-n samples, committee votes -- and let the procedure succeed if
any attempt succeeds.  Let g be the per-item probability that one attempt fails
and nu the law of g over items.  Conditional independence of attempts given the
item -- the one assumption the panel CONFIRMED (64: eight latent classes
reproduce pairs to 0.6%, triples to 0.9%) -- gives

    P(m) := Pr[ all m attempts fail ] = int_0^1 g^m dnu(g) ,

a Hausdorff moment sequence.  That is Law I with the count m in place of
log-resource and the per-item attempt-failure rate in place of the spectrum,
and it is a THEOREM, not an axiom: no additivity, no scale-freeness, no
Bernstein.  Consequences, all tested here:

  T1  P(m) is completely monotone in m.                       [panel, vs noise]
  T2  kappa = P(2)/P(1) = e^{-lambda_count}: the price of one attempt, in the
      theory's own units.  With knowledge nested and residual guessing
      independent, kappa = 1 - q0 exactly; correlated guessing raises it
      toward 1, imperfect nesting lowers it.                  [panel, bracket]
  T3  nu({g=1}) = 0 on any task with a chance level.          [panel, exact 0]
  T4  the chance-corrected m-model floor bound inflates geometrically at rate
      g_max/(1-q0), which is why 64 found an optimum at m=2.  [panel, rate]
  T5  the two hand-set conventions of the paper are consequences of the number
      two: one doubling of (eps-einf)/einf costs ln cosh(ln2/2) = 0.0589 nats,
      and pi/(2 sqrt(2 rho)) at that rho is 4.58 -> five operating points.
  T6  beta of the stretched-exponential correction is a one-sided stable index,
      hence the per-skill tail index of the task: simulated, recovered.

Also recorded: the exchange rate between an attempt and an e-fold of compute,
which Theorem 2 prices exactly once both are in nats.

Reads  data/cofailure/fmnist_{indep,nested}_correct.npy  (from 63)
       data/cofailure/fmnist_{indep,nested}_frontier.csv
Writes (results/):
    table_count_axis_summary.csv   one row per family: every number above
    table_count_axis_chain.csv     P(m) for the m best models, m = 1..21
    table_count_axis_spectrum.csv  the Hausdorff inversion of P(m): nu
    table_count_axis_kappa.csv     kappa by nesting and by resource gap
    table_count_axis_conventions.csv  T5, exact
    table_count_axis_stable.csv    T6, the stable-index simulation
"""
from __future__ import annotations
import os
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "cofailure")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

Q0 = 0.10          # Fashion-MNIST: 10 classes, uniform chance
RNG = np.random.default_rng(20261005)
FAMILIES = ("indep", "nested")


def save(df, name):
    p = os.path.join(RESULTS, name)
    df.to_csv(p, index=False)
    print(f"  wrote {name}  ({len(df)} rows)")


# ----------------------------------------------------------------------------
# T1 / T3: Law I on the count axis, and the Hausdorff inversion
# ----------------------------------------------------------------------------
def chain_curve(F, order):
    """P(m) over the m BEST models.  Along this chain the hardest member is
    fixed (it is the best model), so under nesting P(m) is a pure moment
    sequence of one and the same nu -- which is what makes T1 a clean test."""
    return np.array([F[:, order[:m]].all(1).mean() for m in range(1, F.shape[1] + 1)])


def invert(P, ngrid=501):
    """Hausdorff inversion: find nu >= 0 on a grid with int g^m dnu = P(m)."""
    g = np.linspace(0.0, 1.0, ngrid)
    A = np.stack([g ** m for m in range(1, len(P) + 1)])
    nu, res = nnls(A, P)
    return g, nu, res


def cm_violations(P, orders=8):
    """Raw finite-difference check.  Reported for completeness: with 1e4 items
    the high-order differences are below the sampling noise, so the NNLS
    residual, not this, is the test."""
    D, out = P.copy(), []
    for r in range(1, orders + 1):
        D = np.diff(D)
        out.append(int(np.sum((-1) ** r * D < 0)))
    return out


# ----------------------------------------------------------------------------
# T2: kappa, by nesting and by gap
# ----------------------------------------------------------------------------
def kappa_table(F, C, x):
    h = F.mean(0)
    rows = []
    for i, j in itertools.combinations(range(F.shape[1]), 2):
        b, w = (i, j) if h[i] < h[j] else (j, i)        # b = better model
        both = (F[:, b] & F[:, w]).mean()
        # containment of the KNOWN sets: Pr[worse correct and better correct]
        # / Pr[worse correct] -> 1 when knowledge is nested.  Independent of
        # kappa, so it is a legitimate covariate.
        cont = (C[:, b] & C[:, w]).mean() / C[:, w].mean()
        rows.append(dict(better=b, worse=w, h_better=h[b], h_worse=h[w],
                         gap=abs(x[i] - x[j]), containment=cont,
                         co_failure=both, kappa=both / h[b]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# T5: the conventions
# ----------------------------------------------------------------------------
def conventions():
    import scipy.optimize as so
    rho = np.log(np.cosh(0.5 * np.log(2.0)))            # one doubling of psi
    K = np.pi / (2.0 * np.sqrt(2.0 * rho))
    r = so.brentq(lambda r: np.log(np.cosh(0.5 * np.log(r))) - 0.05, 1.0001, 10)
    # the paper's band at rho = 0.05: eps/einf in [1.53, 2.89]
    band_lo, band_hi = 1.53, 2.89
    return pd.DataFrame([
        dict(quantity="rho_one_doubling_nats", derived=rho,
             paper_value=0.05, note="ln cosh(ln2/2): cost of a factor 2 in psi"),
        dict(quantity="covering_number_K", derived=K,
             paper_value=5.0, note="pi/(2 sqrt(2 rho)) at rho = ln cosh(ln2/2)"),
        dict(quantity="psi_factor_at_rho_0.05", derived=r,
             paper_value=band_hi / band_lo,
             note="the paper's own band ratio, which fixes the convention"),
    ])


# ----------------------------------------------------------------------------
# T6: beta is a one-sided stable index
# ----------------------------------------------------------------------------
def stable_index(tails=(0.4, 0.5, 0.7, 0.9), nskill=400, nsim=120_000):
    """Item difficulty rate a = sum of nskill i.i.d. per-skill rates with tail
    index b, normalised: a -> one-sided stable(b).  Then
    eps(x) = E e^{-a x} = exp(-c x^b): the stretched exponential, with beta = b.
    """
    x = np.linspace(0.05, 3.0, 40)
    rows = []
    for b in tails:
        U = RNG.pareto(b, size=(nsim, nskill)) + 1.0
        a = U.sum(1) / nskill ** (1.0 / b)
        eps = np.array([np.mean(np.exp(-a * xx)) for xx in x])
        beta = np.polyfit(np.log(x), np.log(-np.log(eps)), 1)[0]
        rows.append(dict(per_skill_tail_index=b, fitted_beta=beta,
                         rel_error=abs(beta - b) / b))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
def main():
    print(__doc__.split("Reads")[0].strip()[:0] or "", end="")
    print("THE COUNT AXIS\n" + "=" * 68)
    summary, chains, spectra, kappas = [], [], [], []

    for fam in FAMILIES:
        C = np.load(os.path.join(DATA, f"fmnist_{fam}_correct.npy"))
        F = ~C
        n, M = F.shape
        h = F.mean(0)
        order = np.argsort(h)                      # best model first
        fr = pd.read_csv(os.path.join(DATA, f"fmnist_{fam}_frontier.csv"))
        x, eps = fr["x"].to_numpy(), fr["eps"].to_numpy()

        # ---- T1 -------------------------------------------------------------
        P = chain_curve(F, order)
        se = np.sqrt(P * (1 - P) / n)
        g, nu, res = invert(P)
        rms = res / np.sqrt(M)
        viol = cm_violations(P)
        print(f"\n[{fam}]  {n} items x {M} models,  h in [{h.min():.4f},{h.max():.4f}]")
        print(f"  T1 Law I on the count axis: NNLS rms {rms:.2e} vs binomial se "
              f"{se.mean():.2e}  -> {'HOLDS' if rms < se.mean() else 'FAILS'}")

        # ---- T3 -------------------------------------------------------------
        core = nu[g > 0.999].sum()
        sup = g[nu > 1e-5]
        print(f"  T3 nu({{g=1}}) = {core:.5f}   (predicted exactly 0 for q0 > 0)")
        print(f"     nu support {np.round(sup, 3)}  masses {np.round(nu[nu > 1e-5], 4)}")

        # ---- T2 -------------------------------------------------------------
        kap = P[1] / P[0]
        kt = kappa_table(F, C, x)
        hi = kt.loc[kt.containment > kt.containment.quantile(0.75), "kappa"].median()
        adj = kt.loc[kt.gap < 0.4, "kappa"].median()
        print(f"  T2 kappa = P(2)/P(1) = {kap:.4f};  -ln kappa = {-np.log(kap):.4f} "
              f"nats per attempt")
        print(f"     pairwise median {kt.kappa.median():.4f} | most-nested quartile "
              f"{hi:.4f} | adjacent {adj:.4f}   [1-q0 = {1-Q0:.4f}]")

        # ---- T4 -------------------------------------------------------------
        bound = P / (1 - Q0) ** np.arange(1, M + 1)
        mstar = int(np.argmin(bound) + 1)
        infl = (bound[-1] / bound[0]) ** (1.0 / (M - 1))
        gmax = sup.max()
        print(f"  T4 chance-corrected bound: optimum at m = {mstar}, gain "
              f"{(1-bound[mstar-1]/bound[0])*100:.1f}%, then inflation {infl:.4f}/model "
              f"vs predicted g_max/(1-q0) = {gmax/(1-Q0):.4f}")

        # ---- exchange rate --------------------------------------------------
        s_count = -np.diff(np.log(P))
        s_comp = -np.diff(np.log(eps)) / np.diff(x)
        xr = s_count[-1] / s_comp[-1]
        print(f"  exchange rate at the top of the frontier: one attempt = "
              f"{xr:.3f} e-folds = x{np.exp(xr):.2f} resource")

        chains.append(pd.DataFrame(dict(
            family=fam, m=np.arange(1, M + 1), P=P, se=se,
            bound_chance_corrected=bound,
            price_per_attempt=np.r_[np.nan, s_count])))
        spectra.append(pd.DataFrame(dict(family=fam, g=g[nu > 1e-9],
                                         mass=nu[nu > 1e-9])))
        kt.insert(0, "family", fam)
        kappas.append(kt)

        # does kappa depend on the gap once the better model is fixed?
        lm = np.log(kt.h_better.to_numpy())
        Z = np.vstack([np.ones_like(lm), lm, lm ** 2]).T
        rsd = lambda y: y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        rho_gap = spearmanr(rsd(kt.gap.to_numpy()), rsd(kt.kappa.to_numpy())).statistic

        summary.append(dict(
            family=fam, n_items=n, n_models=M,
            law_I_count_axis_nnls_rms=rms, binomial_se=se.mean(),
            law_I_holds=bool(rms < se.mean()),
            cm_violations_order1to8=";".join(map(str, viol)),
            kappa_P2_over_P1=kap, price_per_attempt_nats=-np.log(kap),
            kappa_pairwise_median=kt.kappa.median(),
            kappa_most_nested_quartile=hi, kappa_adjacent=adj,
            kappa_predicted_1_minus_q0=1 - Q0,
            nu_atom_at_one=core, g_max=gmax,
            n_atoms=int(((nu > 1e-4).astype(int)[1:] *
                         (np.diff((nu > 1e-4).astype(int)) >= 0)).sum() + (nu[0] > 1e-4)),
            floor_bound_m_star=mstar,
            floor_bound_inflation_per_model=infl,
            floor_bound_inflation_predicted=gmax / (1 - Q0),
            exchange_rate_efolds_per_attempt=xr,
            kappa_vs_gap_partial_spearman=rho_gap))

    print("\n" + "=" * 68)
    save(pd.DataFrame(summary), "table_count_axis_summary.csv")
    save(pd.concat(chains, ignore_index=True), "table_count_axis_chain.csv")
    save(pd.concat(spectra, ignore_index=True), "table_count_axis_spectrum.csv")
    save(pd.concat(kappas, ignore_index=True), "table_count_axis_kappa.csv")

    cv = conventions()
    print("\nT5 the conventions:")
    for _, r in cv.iterrows():
        print(f"  {r.quantity:28s} derived {r.derived:.5f}   paper {r.paper_value:.4f}")
    save(cv, "table_count_axis_conventions.csv")

    st = stable_index()
    print("\nT6 beta as a one-sided stable index:")
    for _, r in st.iterrows():
        print(f"  per-skill tail {r.per_skill_tail_index:.2f} -> fitted beta "
              f"{r.fitted_beta:.3f}  ({r.rel_error*100:.1f}%)")
    save(st, "table_count_axis_stable.csv")

    # T1 again, on arbitrary nu, where it is a theorem with no data at all
    print("\nT1 on random difficulty laws (where it cannot fail):")
    ok = 0
    for _ in range(200):
        gg = RNG.beta(RNG.uniform(0.2, 3), RNG.uniform(0.2, 3), 40_000)
        Pm = np.array([(gg ** m).mean() for m in range(1, 15)])
        D, good = Pm.copy(), True
        for r in range(1, 10):
            D = np.diff(D)
            if np.any((-1) ** r * D < -1e-12):
                good = False
        ok += good
    print(f"  complete monotonicity to order 9: {ok}/200 draws")


if __name__ == "__main__":
    main()

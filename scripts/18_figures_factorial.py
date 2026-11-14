#!/usr/bin/env python3
"""Figure for Section 9.10 (factorial test of N = n).

Reads results/table_factorial_N_test.csv (written by 14/17) and
data/grid_factorial.csv, and writes
    figures/fig_factorial.pdf   (+ .png preview)

(a) power-mean exponent theta_hat with 95% profile interval, every frontier;
    theta = 0 is one shared channel, theta = 1 one channel per resource.
(b) Pythia's exactly crossed grid: every 2x2 interaction contrast, normalised
    by the error drop across its rectangle -- observed, and as predicted by the
    single-channel fit (theta = 0).  One channel per resource predicts 0.
"""
import os, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
# law and fitter, shared with the analysis
exec(open(os.path.join(HERE, "14_test_N_equals_n.py")).read().split(
    "# ----------------------------------------------------------------- run")[0])

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.2, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, GREY = "#1f5fa8", "#c0392b", "#2e8b57", "#8a8a8a"
COL = {"Chinchilla": "#1f5fa8", "Porian": "#4a90d9", "Gadre loss": "#7b3fa0",
       "Gadre 14-task": "#b48cc8", "Pythia": "#c0392b", "Pythia dedup": "#e07b6f",
       "NATS-SSS": "#2e8b57"}
THMAX = 15.0


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print("  wrote", os.path.join(FIG, name + ".pdf"))


def label(fr):
    p = fr.split("|")
    if p[0] == "chinchilla":
        return "Chinchilla", "MassiveText"
    if p[0] == "porian":
        return "Porian", {"rw": "RefinedWeb", "owt2": "OpenWebText2"}[p[1]]
    if p[0] == "gadre":
        ds = {"rw_original": "RefinedWeb", "rpj": "RedPajama", "c4_original": "C4"}[p[1]]
        return ("Gadre loss" if "loss" in p[2] else "Gadre 14-task"), ds
    if p[0].startswith("pythia"):
        t = {"lambada_openai": "LAMBADA", "piqa": "PIQA", "arc_easy": "ARC-e",
             "sciq": "SciQ", "winogrande": "WinoGrande", "5-task": "5-task mean"}[p[1]]
        return ("Pythia dedup" if "deduped" in p[0] else "Pythia"), t
    return "NATS-SSS", p[1].replace("cifar", "CIFAR-")


R = pd.read_csv(os.path.join(RES, "table_factorial_N_test.csv"))
R[["corpus", "short"]] = R.frontier.apply(lambda f: pd.Series(label(f)))
order = list(COL)
R["o"] = R.corpus.map(order.index)
R = R.sort_values(["o", "theta_hat"], kind="stable").reset_index(drop=True)

fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.9), gridspec_kw=dict(width_ratios=[1.45, 1]))

# ------------------------------------------------------------------ (a) -----
a = ax[0]
y = np.arange(len(R))[::-1]
a.axvspan(-0.05, 0.05, color=C3, alpha=0.10, lw=0)
a.axvline(0, color=C3, lw=0.8, ls="--")
a.axvline(1, color=C1, lw=0.8, ls="--")
for yi, (_, r) in zip(y, R.iterrows()):
    c = COL[r.corpus]
    a.plot([r.theta_lo, r.theta_hi], [yi, yi], color=c, lw=1.6, solid_capstyle="butt")
    a.plot(r.theta_hat, yi, "o", color=c, ms=3.6, mec="white", mew=0.4)
    if r.theta_hi >= THMAX - 1e-9:
        a.annotate("", xy=(THMAX * 1.35, yi), xytext=(THMAX, yi),
                   arrowprops=dict(arrowstyle="->", color=c, lw=1.0))
a.set_xscale("symlog", linthresh=1.0, linscale=1.0)
a.set_xlim(-0.08, 22)
a.set_xticks([0, 0.5, 1, 2, 5, 10])
a.set_xticklabels(["0", "0.5", "1", "2", "5", "10"])
a.set_yticks(y)
a.set_yticklabels([f"{r.corpus}: {r.short}" for _, r in R.iterrows()], fontsize=6.3)
for t, (_, r) in zip(a.get_yticklabels(), R.iterrows()):
    t.set_color(COL[r.corpus])
a.set_ylim(-0.8, len(R) - 0.2)
a.set_xlabel(r"power-mean exponent $\hat\theta$ (95% profile interval)")
a.set_title(r"(a) channel structure, 24 factorial frontiers", loc="left", pad=20)
top = len(R) - 0.1
a.text(0.0, top, "one\nchannel", color=C3, fontsize=6.5, ha="center", va="bottom")
a.text(1.0, top, "one per\nresource", color=C1, fontsize=6.5, ha="center", va="bottom")
a.text(9.0, top, "toward\nChebyshev", color=GREY, fontsize=6.5, ha="center", va="bottom")
a.spines["left"].set_visible(False)
a.tick_params(axis="y", length=0)

# ------------------------------------------------------------------ (b) -----
D = pd.read_csv(os.path.join(ROOT, "data", "grid_factorial.csv"))
obs, n1, med_obs, med_n1 = [], [], [], []
for fr, d in D[D.frontier.str.startswith("pythia")].groupby("frontier"):
    X = np.log(d[["cost_params", "cost_tokens"]].to_numpy(float))
    X = X - X.mean(0)
    e = d.eps.to_numpy(float)
    p0, _ = fit_theta(X, e, 0.0)
    df = pd.DataFrame(dict(a=X[:, 0].round(6), b=X[:, 1].round(6), e=e,
                           m=powmean_eps(p0, X, 0.0)))
    Pe = df.pivot_table(index="a", columns="b", values="e").values
    Pm = df.pivot_table(index="a", columns="b", values="m").values
    fo, fm = [], []
    for i, j in itertools.combinations(range(Pe.shape[0]), 2):
        for k, l in itertools.combinations(range(Pe.shape[1]), 2):
            for P, out in ((Pe, fo), (Pm, fm)):
                tot = P[i, k] - P[j, l]
                if tot > 0:
                    out.append((P[i, k] - P[j, k] - P[i, l] + P[j, l]) / tot)
    obs += fo; n1 += fm
    med_obs.append(np.median(fo)); med_n1.append(np.median(fm))
obs, n1 = np.clip(obs, -1, 1), np.clip(n1, -1, 1)

b = ax[1]
for v, c, lab in ((obs, C2, "observed"), (n1, C3, "one channel, fitted")):
    xs = np.sort(v)
    b.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=c, lw=1.4, label=lab)
b.axvline(0, color=C1, lw=1.0, ls="--", label="one per resource (exact)")
b.axhline(0.5, color=GREY, lw=0.5, ls=":")
b.plot(med_obs, np.full(len(med_obs), 0.5), "|", color=C2, ms=8, mew=1.2)
b.plot(med_n1, np.full(len(med_n1), 0.5), "|", color=C3, ms=8, mew=1.2)
b.text(-0.97, 0.53, "ticks: medians of\nthe 12 frontiers", fontsize=6.3, color=GREY, va="bottom")
b.set_xlim(-1, 1); b.set_ylim(0, 1.0)
b.set_xlabel("interaction contrast / error drop")
b.set_ylabel("cumulative fraction")
b.set_title(f"(b) Pythia, {len(obs):,} 2$\\times$2 contrasts", loc="left", pad=20)
b.legend(frameon=False, loc="lower right", fontsize=6.6)
print(f"  pythia contrasts: {len(obs)}; obs median {np.median(obs):+.3f}, "
      f"one-channel median {np.median(n1):+.3f}, frac obs>0 {np.mean(obs > 0):.2f}")

fig.tight_layout(w_pad=1.5)
save(fig, "fig_factorial")

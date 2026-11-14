#!/usr/bin/env python3
"""Law comparison, exponent table, unit covariance, and the frontier spectrum."""
import os, itertools, numpy as np, pandas as pd
from common import (frontier, load_resources, load_accuracy, fit_additive,
                    fit_multiplicative, EVAL_FILES, BENCH_FILES, save)
B = load_resources()

# ---- A. additive vs multiplicative, with bootstrap ------------------------
PAIRS = [("lat_rtx3090","param_count"), ("infer_gmacs","param_count"),
         ("lat_rtx3090","lat_cpu_i9"), ("lat_rtx3090","infer_gmacs"),
         ("train_us","lat_rtx3090"),   ("lat_cpu_i9","param_count"),
         ("infer_macts","param_count")]
rows=[]
for c1,c2 in PAIRS:
    d,X,e = frontier("ImageNet-1k",[c1,c2],resources=B)
    if len(e)<25: continue
    A_=fit_additive(X,e); M_=fit_multiplicative(X,e); Bt=fit_additive(X,e,beta_free=True)
    rng=np.random.default_rng(0); bs=[]
    for _ in range(300):
        i=rng.integers(0,len(e),len(e))
        try: bs.append(fit_multiplicative(X[i],e[i])["aic"]-fit_additive(X[i],e[i])["aic"])
        except Exception: pass
    bs=np.array(bs)
    rows.append(dict(corpus="vision", resources=f"{c1} x {c2}", n_pareto=len(e),
        corr_logres=np.corrcoef(X[:,0],X[:,1])[0,1], beta_hat=Bt["beta"],
        dAIC_add_minus_mul=M_["aic"]-A_["aic"],
        boot_p5=np.percentile(bs,5), boot_p95=np.percentile(bs,95),
        p_additive_wins=float(np.mean(bs>0)),
        einf=A_["einf"], alpha_1=A_["alpha"][0], alpha_2=A_["alpha"][1]))
# materials
mb=pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data","matbench_models.csv"))
def par(M):
    k=np.ones(len(M),bool)
    for i in range(len(M)):
        if k[i]:
            dd=np.all(M<=M[i],axis=1)&np.any(M<M[i],axis=1)
            if dd.any(): k[i]=False
    return k
for qn,q in [("1-F1",1-mb.F1),("MAE",mb.MAE)]:
    d=mb.assign(eps=q).dropna(subset=["eps","params","ntrain"])
    M=d[["eps","params","ntrain"]].to_numpy(float); P=d[par(M)]
    X=np.log(P[["params","ntrain"]].to_numpy(float)); e=P.eps.to_numpy(float)
    A_=fit_additive(X,e); M_=fit_multiplicative(X,e)
    rows.append(dict(corpus="materials", resources=f"params x ntrain ({qn})",
        n_pareto=len(e), corr_logres=np.corrcoef(X[:,0],X[:,1])[0,1], beta_hat=np.nan,
        dAIC_add_minus_mul=M_["aic"]-A_["aic"], boot_p5=np.nan, boot_p95=np.nan,
        p_additive_wins=np.nan, einf=A_["einf"],
        alpha_1=A_["alpha"][0], alpha_2=A_["alpha"][1]))
law=pd.DataFrame(rows); save(law,"table_additive_vs_multiplicative.csv")
print(law[["resources","n_pareto","corr_logres","dAIC_add_minus_mul"]].round(3).to_string(index=False))

# ---- B. exponent table and the two-way factorisation ----------------------
RES=["lat_rtx3090","param_count","infer_gmacs"]
tab={}
for ev in EVAL_FILES:
    row={}
    for r_ in RES:
        try:
            d,X,e = frontier(ev,[r_],resources=B)
            if len(e)<15: continue
            row[r_]=fit_additive(X,e)["alpha"][0]
        except Exception: pass
    if len(row)==len(RES): tab[ev]=row
T=pd.DataFrame(tab).T
lg=np.log(T.to_numpy()); u=lg.mean(1,keepdims=True); v=(lg-u).mean(0,keepdims=True)
resid=lg-u-v
T2=T.copy(); T2.loc["resource offset exp(v)"]=np.exp(v[0])
save(T2.reset_index().rename(columns={"index":"eval_set"}),"table_exponents.csv")
fac=pd.DataFrame([dict(total_log_var=lg.var(), task_var=float(np.var(u)),
                       resource_var=float(np.var(v)), residual_var=resid.var(),
                       share_explained=1-resid.var()/lg.var())])
save(fac,"table_exponent_factorisation.csv")
print(T.round(3).to_string()); print(fac.round(4).to_string(index=False))

# ---- C. unit covariance across processors ---------------------------------
acc=load_accuracy("ImageNet-1k"); alpha1={}
for tag in BENCH_FILES:
    d,X,e=frontier("ImageNet-1k",["lat_"+tag],resources=B)
    alpha1[tag]=fit_additive(X,e)["alpha"][0]
merged=acc.merge(B,on=["base","img_size"]); rows=[]
for h1,h2 in itertools.combinations(BENCH_FILES,2):
    s=merged[["lat_"+h1,"lat_"+h2]].dropna()
    if len(s)<40: continue
    b=np.polyfit(np.log(s.iloc[:,1]),np.log(s.iloc[:,0]),1)[0]
    rows.append(dict(pair=f"{h1}/{h2}", slope_b=b, alpha_ratio=alpha1[h1]/alpha1[h2],
                     observed_over_predicted=(alpha1[h2]/alpha1[h1])/b,
                     same_class=("cpu" in h1)==("cpu" in h2)))
uc=pd.DataFrame(rows); save(uc,"table_unit_covariance.csv")
save(pd.DataFrame([dict(processor=k, alpha=v) for k,v in alpha1.items()]),
     "table_single_resource_alpha.csv")
print(uc.groupby("same_class").observed_over_predicted.agg(["mean","std","count"]).round(3).to_string())

# ---- D. spectrum of the log-resource block --------------------------------
COLS=["lat_rtx3090","lat_cpu_i9","param_count","infer_gmacs","infer_macts"]
d,X,e=frontier("ImageNet-1k",COLS,resources=B)
Z=(X-X.mean(0))/X.std(0); ev=np.linalg.eigvalsh(np.cov(Z.T))[::-1]; w=ev/ev.sum()
k=float(np.exp(-np.sum(w*np.log(w+1e-300))))
sp=pd.DataFrame([dict(n_pareto=len(e), n_resources=len(COLS),
                      **{f"share_pc{i+1}":w[i] for i in range(len(w))},
                      participation_k=k, ln_k_plus_1=np.log(k+1),
                      ln_n_plus_1=np.log(len(COLS)+1))])
save(sp,"table_frontier_spectrum.csv"); print(sp.round(4).to_string(index=False))

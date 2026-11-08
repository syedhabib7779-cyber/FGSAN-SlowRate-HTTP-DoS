# ===================== CELL 36 (NEW) — M7: statistical uncertainty =====================
# Dispersion for the headline metrics: (A) 5 training seeds -> mean +/- SD and 95% CI for
# F1/AUC/FPR/FNR/ESR with exact ESR numerator/denominator; (B) bootstrap 95% CIs on test
# F1 and ESR (test-sampling variance); (C) transfer-rate dispersion across seeds; plus a
# Holm-Bonferroni note for the family of baseline comparisons. Split-protocol variance is
# quantified separately in the Split Independence analysis (M4).
# Needs: df_wed, LABEL_KEEP, present_features, cfg, group_indices, set_seed,
#   FeatureGroupedAttentionNet, train_model, evaluate_torch, predict_torch, pgd_attack, to_t,
#   SEED, DEVICE, RATIO_BENIGN_TO_ATTACK.
import os, json, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from google.colab import drive; drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/fgsan_ckpt"
except Exception:
    OUT = "./fgsan_ckpt"
os.makedirs(OUT, exist_ok=True)
RES = f"{OUT}/m7_results.json"
R = json.load(open(RES)) if os.path.exists(RES) else {"seeds": {}}
def save(): json.dump(R, open(RES, "w"), indent=1)

N_SEEDS = 5
def mean_ci(vals, conf=0.95):
    v=np.asarray(vals,float); n=len(v); m=float(v.mean()); sd=float(v.std(ddof=1)) if n>1 else 0.0
    tcrit={2:12.706,3:4.303,4:3.182,5:2.776,6:2.571}.get(n,1.96)
    h=tcrit*sd/np.sqrt(n) if n>1 else 0.0
    return m, sd, (m-h, m+h)

# ---- fixed balanced split (split seed = SEED); only the TRAINING seed varies ----
d = df_wed.copy(); d["Label"]=d["Label"].astype(str).str.strip()
d = d[d["Label"].isin(LABEL_KEEP.keys())].copy(); d["y"]=d["Label"].map(LABEL_KEEP)
d[present_features]=d[present_features].apply(pd.to_numeric, errors="coerce")
d = d.replace([np.inf,-np.inf],np.nan).dropna(subset=present_features)
d = d[["y"]+present_features].reset_index(drop=True)
d = d[~d.duplicated(subset=present_features, keep="first")].reset_index(drop=True)
da,db=d[d.y==1],d[d.y==0]; nb=min(len(db),len(da)*RATIO_BENIGN_TO_ATTACK)
task=pd.concat([da,db.sample(nb,random_state=SEED)]).sample(frac=1,random_state=SEED).reset_index(drop=True)
X=task[present_features].to_numpy(np.float32); y=task.y.to_numpy(np.int64)
itr,itmp=train_test_split(np.arange(len(y)),test_size=cfg.val_split+cfg.test_split,stratify=y,random_state=SEED)
iva,ite=train_test_split(itmp,test_size=cfg.test_split/(cfg.val_split+cfg.test_split),stratify=y[itmp],random_state=SEED)
sc=StandardScaler().fit(X[itr])
Xtr,Xva,Xte=sc.transform(X[itr]),sc.transform(X[iva]),sc.transform(X[ite])
ytr,yva,yte=y[itr],y[iva],y[ite]
atk=yte==1; Xatk,yatk=Xte[atk],yte[atk]
yatk_t=torch.tensor(yatk,dtype=torch.long,device=DEVICE)

# ---- baselines trained once on this split (for transfer dispersion) ----
def make_baselines():
    return {"Random Forest":RandomForestClassifier(n_estimators=300,random_state=SEED,n_jobs=-1),
            "XGBoost":XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.1,subsample=0.9,
                                    colsample_bytree=0.9,eval_metric="logloss",random_state=SEED,n_jobs=-1),
            "LightGBM":LGBMClassifier(n_estimators=300,max_depth=-1,learning_rate=0.1,random_state=SEED,n_jobs=-1,verbose=-1),
            "SVM (RBF)":SVC(kernel="rbf",probability=True,random_state=SEED),
            "MLP (sklearn)":MLPClassifier(hidden_layer_sizes=(128,64),max_iter=300,random_state=SEED)}
if "base_fit" not in globals():
    base_fit={}
    SVM_MAX=int(globals().get("SVM_MAX_TRAIN",20000))
    for nm,clf in make_baselines().items():
        if nm=="SVM (RBF)" and len(ytr)>SVM_MAX:
            s=np.random.default_rng(SEED).choice(len(ytr),SVM_MAX,replace=False); clf.fit(Xtr[s],ytr[s])
        else: clf.fit(Xtr,ytr)
        base_fit[nm]=clf

def esr_counts(model, adv_np, clean_preds):
    adv_preds,_=predict_torch(model, adv_np)
    adv_preds=np.asarray(adv_preds); cc=clean_preds==1
    num=int(((adv_preds==0)&cc).sum()); den=int(cc.sum())
    return num, den

# ---- run N seeds ----
for i in range(N_SEEDS):
    s=SEED+i
    if str(s) in R["seeds"]:
        print(f"seed {s} cached"); continue
    set_seed(s)
    m=FeatureGroupedAttentionNet(group_indices,emb_dim=cfg.emb_dim,n_heads=cfg.n_heads,
                                 dropout=cfg.dropout,use_attention=True)
    m,_=train_model(m,Xtr,ytr,Xva,yva,cfg,use_adv=True,use_focal=True)
    met=evaluate_torch(m,Xte,yte)
    cp,_=predict_torch(m,Xatk); cp=np.asarray(cp)
    adv=pgd_attack(m,to_t(Xatk),yatk_t,cfg.adv_epsilon,cfg.adv_alpha,cfg.adv_steps).cpu().numpy()
    num,den=esr_counts(m,adv,cp); esr=num/den if den else float("nan")
    # transfer to baselines
    transfer={}
    for nm,clf in base_fit.items():
        bc=clf.predict(Xatk); ba=clf.predict(adv); cc=bc==1
        transfer[nm]=float(((ba!=1)&cc).sum()/max(cc.sum(),1))
    R["seeds"][str(s)]={"f1":float(met["f1"]),"auc":float(met.get("auc",float("nan"))),
        "fpr":float(met["fpr"]),"fnr":float(met["fnr"]),"esr":float(esr),
        "esr_num":num,"esr_den":den,"transfer":transfer}
    save()
    print(f"seed {s}: F1={met['f1']:.4f} ESR={esr:.4f} ({num}/{den})")
    if i==0:  # Part B: bootstrap on the first model
        from sklearn.metrics import f1_score as _f1
        pr,_=predict_torch(m,Xte); pr=np.asarray(pr); g=np.random.default_rng(0)
        bf=[]; be=[]
        for _b in range(1000):
            idx=g.integers(0,len(yte),len(yte)); bf.append(_f1(yte[idx],pr[idx],zero_division=0))
            aidx=g.integers(0,den if den else 1,den if den else 1)
            # bootstrap ESR over clean-correct attack flows
        ccmask=cp==1; advp,_=predict_torch(m,adv); advp=np.asarray(advp)
        ev=((advp==0)&ccmask)[ccmask]
        for _b in range(1000):
            ridx=g.integers(0,len(ev),len(ev)); be.append(ev[ridx].mean())
        R["bootstrap"]={"f1":[float(np.percentile(bf,2.5)),float(np.mean(bf)),float(np.percentile(bf,97.5))],
                        "esr":[float(np.percentile(be,2.5)),float(np.mean(be)),float(np.percentile(be,97.5))]}
        save()

# ---- summary ----
seeds=[R["seeds"][k] for k in R["seeds"]]
print(f"\n=== M7 dispersion over {len(seeds)} seeds (fixed split) ===")
for key,lab in [("f1","F1"),("auc","AUC"),("fpr","FPR"),("fnr","FNR"),("esr","ESR")]:
    m,sd,ci=mean_ci([s[key] for s in seeds])
    print(f"  {lab:4}: {m:.4f} +/- {sd:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
tot_num=sum(s["esr_num"] for s in seeds); tot_den=sum(s["esr_den"] for s in seeds)
print(f"  ESR pooled exact: {tot_num}/{tot_den} = {tot_num/tot_den:.4f}")
if "bootstrap" in R:
    bf=R["bootstrap"]["f1"]; be=R["bootstrap"]["esr"]
    print(f"\n  bootstrap (test-sampling) 95% CI: F1 [{bf[0]:.4f}, {bf[2]:.4f}]  ESR [{be[0]:.4f}, {be[2]:.4f}]")
print("\n  transfer evasion, mean +/- SD across seeds:")
for nm in seeds[0]["transfer"]:
    vals=[s["transfer"][nm] for s in seeds]; m,sd,ci=mean_ci(vals)
    print(f"    {nm:16}: {m:.4f} +/- {sd:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
print("\nNote: split-protocol dispersion is reported in the Split Independence analysis;")
print("multiplicity across the baseline family is controlled with Holm-Bonferroni in the text.")
print("saved to", RES)

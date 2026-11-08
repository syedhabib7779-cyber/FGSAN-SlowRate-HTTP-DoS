# ========== RECOMPUTE B (M9) — transfer-success rates with bootstrap confidence intervals ==========
# For adversarial examples crafted white-box against FGSAN and transferred to each baseline, reports the
# evasion success rate (ESR, conditional on the target being clean-correct) AND the unconditional attack
# miss rate, each with a 95% bootstrap CI over attack flows, plus exact numerator/denominator. Rebuilds
# baselines. Needs restored: X_train,y_train,X_atk,y_atk,X_atk_adv_np, main_model; defined: predict_torch.
import os, json, numpy as np
try:
    from google.colab import drive; drive.mount("/content/drive"); OUT="/content/drive/MyDrive/fgsan_ckpt"
except Exception: OUT="./fgsan_ckpt"
os.makedirs(OUT, exist_ok=True); RES=f"{OUT}/r2_B_transfer_ci.json"
SEED=42; rng=np.random.default_rng(SEED)
Xtr,ytr=np.asarray(X_train),np.asarray(y_train)
Xa=np.asarray(X_atk); Xadv=np.asarray(X_atk_adv_np); n=len(Xa)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
defs={
 "Random Forest":RandomForestClassifier(n_estimators=300,random_state=SEED,n_jobs=-1),
 "XGBoost":XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.1,subsample=0.9,
            colsample_bytree=0.9,eval_metric="logloss",random_state=SEED,n_jobs=-1),
 "LightGBM":LGBMClassifier(n_estimators=300,max_depth=-1,learning_rate=0.1,random_state=SEED,n_jobs=-1,verbose=-1),
 "SVM (RBF)":SVC(kernel="rbf",C=1.0,gamma="scale",random_state=SEED),
 "MLP (sklearn)":MLPClassifier(hidden_layer_sizes=(128,64),max_iter=300,random_state=SEED),
}
def preds(model,X):
    if hasattr(model,"predict"): return np.asarray(model.predict(X))
mods={}
for nm,m in defs.items(): print("training",nm,"..."); m.fit(Xtr,ytr); mods[nm]=m
# FGSAN (white-box, its own worst case) + baselines (transfer)
def fg_pred(X): p,_=predict_torch(main_model,X); return np.asarray(p)
PRED={"FGSAN (proposed, white-box)":(fg_pred(Xa),fg_pred(Xadv))}
for nm,m in mods.items(): PRED[nm]=(preds(m,Xa),preds(m,Xadv))

def esr_counts(cp,ap):
    cc=(cp==1)                      # clean-correct attack flows (denominator)
    evaded=cc&(ap==0)
    return int(evaded.sum()), int(cc.sum())
def boot(cp,ap,B=2000):
    esr=[]; miss=[]
    for _ in range(B):
        idx=rng.integers(0,n,n)
        c=cp[idx]; a=ap[idx]; cc=(c==1)
        esr.append((cc&(a==0)).sum()/max(cc.sum(),1))     # conditional ESR
        miss.append((a==0).sum()/n)                       # unconditional miss (adv predicted benign)
    q=lambda v:(float(np.percentile(v,2.5)),float(np.percentile(v,97.5)))
    return q(esr), q(miss)

R={}
print(f"\n=== Transfer success with 95% bootstrap CI (attack flows n={n}) ===")
print(f"{'model':28} {'ESR':>7} {'ESR 95% CI':>18} {'num/den':>12} {'uncond.miss':>11}")
for nm,(cp,ap) in PRED.items():
    num,den=esr_counts(cp,ap); esr=num/max(den,1)
    (elo,ehi),(mlo,mhi)=boot(cp,ap)
    uncond=float((ap==0).mean())
    R[nm]={"esr":esr,"esr_ci":[elo,ehi],"num":num,"den":den,"uncond_miss":uncond,"uncond_ci":[mlo,mhi]}
    print(f"{nm:28} {esr:7.4f} [{elo:.4f}, {ehi:.4f}] {num:6d}/{den:<5d} {uncond:7.4f}")
json.dump(R,open(RES,"w"),indent=1); print("\nsaved ->",RES)
print("Note: FGSAN faces its own white-box examples (worst case); baselines face the SAME examples transferred")
print("(a weaker, non-adaptive attack), so their ESR is an optimistic upper bound on their robustness.")

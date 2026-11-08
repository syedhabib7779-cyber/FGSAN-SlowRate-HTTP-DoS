# ========== RECOMPUTE A (m14 fix) — operating points with thresholds selected on VALIDATION ==========
# Fixes the leakage in the matched-FPR analysis: the 0.5% (and other) FPR thresholds are now chosen on the
# VALIDATION benign flows and only applied to TEST. Reports, per model and target FPR: the validation-selected
# threshold, the ACTUAL test FPR it produces, and clean vs adversarial recall on the test attack flows.
# Rebuilds baselines self-contained. Needs restored: X_train,y_train,X_val,y_val,X_test,y_test,X_atk,y_atk,
# X_atk_adv_np, main_model; defined: predict_torch.
import os, json, numpy as np
try:
    from google.colab import drive; drive.mount("/content/drive"); OUT="/content/drive/MyDrive/fgsan_ckpt"
except Exception: OUT="./fgsan_ckpt"
os.makedirs(OUT, exist_ok=True); RES=f"{OUT}/r2_A_val_thresholds.json"
SEED=42
Xtr,ytr=np.asarray(X_train),np.asarray(y_train)
Xva,yva=np.asarray(X_val),np.asarray(y_val)
Xte,yte=np.asarray(X_test),np.asarray(y_test)
Xa=np.asarray(X_atk); Xadv=np.asarray(X_atk_adv_np)
val_benign=Xva[yva==0]; test_benign=Xte[yte==0]

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
 "SVM (RBF)":SVC(kernel="rbf",C=1.0,gamma="scale",random_state=SEED),   # full-data (M14 parity)
 "MLP (sklearn)":MLPClassifier(hidden_layer_sizes=(128,64),max_iter=300,random_state=SEED),
}
models={}
for n,m in defs.items():
    print("training",n,"..."); m.fit(Xtr,ytr); models[n]=m

def sc(model,X):
    if hasattr(model,"predict_proba"):
        try: return model.predict_proba(X)[:,1]
        except Exception: pass
    return model.decision_function(X)
def fg(X):
    _,p=predict_torch(main_model,X); return np.asarray(p)

SC={"FGSAN (proposed)":(fg(val_benign),fg(Xa),fg(Xadv),fg(test_benign))}
for n,m in models.items():
    SC[n]=(sc(m,val_benign),sc(m,Xa),sc(m,Xadv),sc(m,test_benign))

R={}
print("\n=== Recall at matched FPR (threshold chosen on VALIDATION, applied to TEST) ===")
for fpr in [0.001,0.005,0.01]:
    print(f"\n-- target FPR = {fpr:.3f} (selected on validation) --")
    print(f"{'model':20} {'thr(val)':>9} {'testFPR':>8} {'recall_clean':>13} {'recall_adv':>11}")
    for n,(svb,sca,sa,stb) in SC.items():
        thr=np.quantile(svb,1-fpr)                 # threshold from VALIDATION benign
        rc=float((sca>=thr).mean()); ra=float((sa>=thr).mean()); tfpr=float((stb>=thr).mean())
        R[f"{n}|{fpr}"]={"thr":float(thr),"test_fpr":tfpr,"recall_clean":rc,"recall_adv":ra}
        print(f"{n:20} {thr:9.3f} {tfpr:8.4f} {rc:13.4f} {ra:11.4f}")

# effective-recall crossover FGSAN vs XGBoost at target FPR 0.005 (val-selected)
F=R["FGSAN (proposed)|0.005"]; Xg=R["XGBoost|0.005"]
num=(Xg['recall_clean']-F['recall_clean']); den=num+(F['recall_adv']-Xg['recall_adv'])
phi=num/den if den else float('nan')
print(f"\nFGSAN vs XGBoost @ val-selected 0.5% FPR: test FPRs {F['test_fpr']:.4f}/{Xg['test_fpr']:.4f}")
print(f"effective-recall crossover phi* = {phi:.3f}")
json.dump(R,open(RES,"w"),indent=1); print("saved ->",RES)
print("\nNOTE: compare test_fpr to the 0.005 target; small gaps are expected and honest (val!=test).")

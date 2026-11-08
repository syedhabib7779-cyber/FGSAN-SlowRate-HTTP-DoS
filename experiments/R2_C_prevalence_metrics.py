# ========== RECOMPUTE C (M5) — precision / PR-AUC / alerts-per-million across realistic prevalences ==========
# Recomputes deployment metrics for FGSAN at several attack prevalences by reweighting the test set, at two
# operating points: the default 0.5 threshold and a validation-selected 0.5%-FPR threshold. No retraining.
# Needs restored: X_val,y_val,X_test,y_test, main_model; defined: predict_torch.
import os, json, numpy as np
from sklearn.metrics import average_precision_score
try:
    from google.colab import drive; drive.mount("/content/drive"); OUT="/content/drive/MyDrive/fgsan_ckpt"
except Exception: OUT="./fgsan_ckpt"
os.makedirs(OUT, exist_ok=True); RES=f"{OUT}/r2_C_prevalence.json"
Xva,yva=np.asarray(X_val),np.asarray(y_val)
Xte,yte=np.asarray(X_test),np.asarray(y_test)
def fg(X): _,p=predict_torch(main_model,X); return np.asarray(p)
s_te=fg(Xte); s_valben=fg(Xva[yva==0])
thr_val=float(np.quantile(s_valben,1-0.005))            # 0.5% FPR threshold from validation
att=(yte==1); ben=~att
PREVS=[("balanced ~1:5",1/6),("native ~3%",0.03),("1%",0.01),("0.1%",0.001)]
R={}
print(f"{'prevalence':14} {'thr':>8} {'precision':>10} {'recall':>8} {'PR-AUC':>8} {'alerts/1e6':>11}")
for name,pi in PREVS:
    # reweight so attacks are a pi fraction of an effective 1e6-flow stream
    w=np.where(att, pi/att.mean(), (1-pi)/ben.mean())
    prauc=float(average_precision_score(yte, s_te, sample_weight=w))
    for thr,tag in [(0.5,"default-0.5"),(thr_val,"val-0.5%FPR")]:
        pred=(s_te>=thr).astype(int)
        tp=float((w*((pred==1)&att)).sum()); fp=float((w*((pred==1)&ben)).sum()); fn=float((w*((pred==0)&att)).sum())
        prec=tp/(tp+fp) if tp+fp>0 else 0.0; rec=tp/(tp+fn) if tp+fn>0 else 0.0
        alerts=fp/ (w*ben).sum() *1e6                    # false alerts per million benign flows
        R[f"{name}|{tag}"]={"thr":thr,"precision":prec,"recall":rec,"pr_auc":prauc,"alerts_per_1e6_benign":alerts}
        if tag=="val-0.5%FPR":
            print(f"{name:14} {thr:8.3f} {prec:10.4f} {rec:8.4f} {prauc:8.4f} {alerts:11.0f}")
json.dump(R,open(RES,"w"),indent=1)
print("\n(Showing the val-selected 0.5%-FPR operating point; default-0.5 values are saved in the JSON too.)")
print("PR-AUC falls as attacks get rarer; alerts/million is the per-time false-alarm load for the ops team.")
print("saved ->",RES)

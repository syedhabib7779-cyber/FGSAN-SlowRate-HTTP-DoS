# ===================== CELL (NEW) — M14: SVM training parity (learning curve + full-data SVM) =====================
# The other baselines train on the full split; the RBF-SVM was capped at 20,000 rows. This measures whether
# that cap handicapped the SVM: it trains the SVM on increasing subset sizes up to the FULL training split and
# reports test F1/precision/recall/ROC-AUC/PR-AUC, training time and support-vector count. Uses decision_function
# for scores (no Platt scaling) so the fits stay fast. Per-size results are saved for resume.
# Needs restored: X_train, y_train, X_test, y_test.
import os, json, time, numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
try:
    from google.colab import drive; drive.mount("/content/drive"); OUT="/content/drive/MyDrive/fgsan_ckpt"
except Exception: OUT="./fgsan_ckpt"
os.makedirs(OUT, exist_ok=True); RES=f"{OUT}/m14_results.json"
R=json.load(open(RES)) if os.path.exists(RES) else {}
def save(): json.dump(R,open(RES,"w"),indent=1)

SEED=42
Xtr=np.asarray(X_train); ytr=np.asarray(y_train); Xte=np.asarray(X_test); yte=np.asarray(y_test)
N=len(Xtr)
print(f"full training split: {N} rows  (class balance {np.bincount(ytr)})")
sizes=sorted({s for s in [5000,10000,20000,40000,N] if s<=N})
rng=np.random.default_rng(SEED)

def run(n):
    key=str(n)
    if key in R: return R[key]
    sub = np.arange(N) if n>=N else rng.choice(N, n, replace=False)
    t0=time.time()
    svm=SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)   # paper's config (no probability -> fast)
    svm.fit(Xtr[sub], ytr[sub]); fit_s=time.time()-t0
    pred=svm.predict(Xte); sc=svm.decision_function(Xte)
    pr,rc,f1,_=precision_recall_fscore_support(yte,pred,average="binary",pos_label=1,zero_division=0)
    R[key]=dict(n=int(n),f1=float(f1),precision=float(pr),recall=float(rc),
                roc_auc=float(roc_auc_score(yte,sc)),pr_auc=float(average_precision_score(yte,sc)),
                n_sv=int(svm.n_support_.sum()),fit_s=float(fit_s)); save()
    print(f"  n={n:>6}: F1={f1:.4f} P={pr:.4f} R={rc:.4f} ROC-AUC={R[key]['roc_auc']:.4f} "
          f"PR-AUC={R[key]['pr_auc']:.4f}  SV={R[key]['n_sv']}  fit={fit_s:.1f}s")
    return R[key]

print("SVM (RBF) learning curve to full training data:")
for n in sizes: run(n)

print("\n=== M14 SVM parity summary ===")
print(f"{'train rows':>10} {'F1':>7} {'Prec':>7} {'Rec':>7} {'ROC-AUC':>8} {'PR-AUC':>7} {'SV':>6} {'fit(s)':>7}")
for n in sizes:
    r=R[str(n)]
    print(f"{n:>10} {r['f1']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f} {r['roc_auc']:8.4f} "
          f"{r['pr_auc']:7.4f} {r['n_sv']:6d} {r['fit_s']:7.1f}")
r20=R.get("20000"); rfull=R[str(N)]
if r20:
    print(f"\n20,000 vs full ({N}):  dF1={rfull['f1']-r20['f1']:+.4f}  dROC-AUC={rfull['roc_auc']-r20['roc_auc']:+.4f}  dPR-AUC={rfull['pr_auc']-r20['pr_auc']:+.4f}")
    print("Reading: if ~0, the 20k cap did not handicap the SVM (performance had saturated); report the full-data")
    print("SVM as the parity baseline and state the saturation.")
print("saved ->",RES)

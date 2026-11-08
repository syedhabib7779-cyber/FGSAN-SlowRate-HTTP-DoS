# ===================== CELL 33 (NEW) — M5: native-prevalence deployment evaluation =====================
# Training stays class-balanced (5:1), but the TEST set keeps the NATIVE attack prevalence
# (~2.5%) instead of being resampled, and deployment metrics are reported: PR-AUC, precision,
# recall, F1, FPR, alerts per million benign flows, and threshold operating points.
# Needs: df_wed, LABEL_KEEP, present_features, cfg, group_indices, set_seed,
#   FeatureGroupedAttentionNet, train_model, predict_torch, SEED, RATIO_BENIGN_TO_ATTACK.
import os, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix)

try:
    from google.colab import drive; drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/fgsan_ckpt"
except Exception:
    OUT = "./fgsan_ckpt"
RES = f"{OUT}/m5_results.json"

# ---------- native (unbalanced) filtered data ----------
d = df_wed.copy(); d["Label"] = d["Label"].astype(str).str.strip()
d = d[d["Label"].isin(LABEL_KEEP.keys())].copy(); d["y"] = d["Label"].map(LABEL_KEEP)
d[present_features] = d[present_features].apply(pd.to_numeric, errors="coerce")
d = d.replace([np.inf,-np.inf], np.nan).dropna(subset=present_features)
d = d[["y"] + present_features].reset_index(drop=True)
d = d[~d.duplicated(subset=present_features, keep="first")].reset_index(drop=True)   # dedup (M4-consistent)
prev = d.y.mean()
print(f"native filtered data: {len(d)} flows, attack prevalence {prev:.3%} "
      f"(benign {int((d.y==0).sum())}, attack {int((d.y==1).sum())})")

# ---------- split at NATIVE prevalence; balance TRAIN+VAL only ----------
X = d[present_features].to_numpy(np.float32); y = d.y.to_numpy(np.int64)
itr, itmp = train_test_split(np.arange(len(y)), test_size=cfg.val_split+cfg.test_split,
                             stratify=y, random_state=SEED)
iva, ite = train_test_split(itmp, test_size=cfg.test_split/(cfg.val_split+cfg.test_split),
                            stratify=y[itmp], random_state=SEED)
def balance(idx, seed=SEED):
    a = idx[y[idx]==1]; b = idx[y[idx]==0]
    k = min(len(b), len(a)*RATIO_BENIGN_TO_ATTACK)
    b = np.random.default_rng(seed).choice(b, k, replace=False)
    return np.concatenate([a, b])
itr_b, iva_b = balance(itr), balance(iva, SEED+1)
Xtr, ytr = X[itr_b], y[itr_b]
Xva, yva = X[iva_b], y[iva_b]
Xte, yte = X[ite], y[ite]                      # NATIVE prevalence test
print(f"train {len(ytr)} (atk {ytr.mean():.1%}) | val {len(yva)} (atk {yva.mean():.1%}) | "
      f"TEST {len(yte)} (atk {yte.mean():.3%}, NATIVE)")

sc = StandardScaler().fit(Xtr)
Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
set_seed(SEED)
m = FeatureGroupedAttentionNet(group_indices, emb_dim=cfg.emb_dim, n_heads=cfg.n_heads,
                               dropout=cfg.dropout, use_attention=True)
m, _ = train_model(m, Xtr, ytr, Xva, yva, cfg, use_adv=True, use_focal=True)
_, scores = predict_torch(m, Xte); scores = np.asarray(scores, dtype=np.float64)

# ---------- deployment metrics on the native test set ----------
def op(thr):
    pred = (scores > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0,1]).ravel()
    prec = tp/(tp+fp) if tp+fp else 0.0; rec = tp/(tp+fn) if tp+fn else 0.0
    fpr  = fp/(fp+tn) if fp+tn else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return {"threshold":float(thr),"precision":prec,"recall":rec,"f1":f1,"fpr":fpr,
            "alerts_per_million_benign":fpr*1e6,"TP":int(tp),"FP":int(fp),"FN":int(fn)}

prauc = average_precision_score(yte, scores); rocauc = roc_auc_score(yte, scores)
prec_c, rec_c, thr_c = precision_recall_curve(yte, scores)

# operating points
pts = {"default (0.5)": op(0.5)}
ok = np.where(rec_c[:-1] >= 0.95)[0]           # highest threshold giving recall >= 0.95
pts["recall>=0.95"] = op(thr_c[ok[-1]]) if len(ok) else op(0.5)
grid = np.unique(np.quantile(scores, np.linspace(0.5, 0.9999, 500)))
budget = next((op(t) for t in grid if op(t)["alerts_per_million_benign"] <= 1000), None)
pts["alert budget <=1000/million"] = budget if budget else {"note":"not reachable"}

# contrast: precision at the paper's balanced 5:1 test prevalence
g = np.random.default_rng(SEED)
at = np.where(yte==1)[0]; bn = np.where(yte==0)[0]
keep = g.choice(bn, min(len(bn), len(at)*RATIO_BENIGN_TO_ATTACK), replace=False)
bi = np.concatenate([at, keep]); pb = (scores[bi] > 0.5).astype(int)
tn,fp,fn,tp = confusion_matrix(yte[bi], pb, labels=[0,1]).ravel()
prec_bal = tp/(tp+fp) if tp+fp else 0.0

res = {"native_prevalence":float(prev),"pr_auc":float(prauc),"roc_auc":float(rocauc),
       "precision_balanced_test":float(prec_bal),"operating_points":pts}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(RES,"w"), indent=1)

print(f"\nNATIVE-PREVALENCE TEST ({prev:.2%} attacks): PR-AUC={prauc:.4f}  ROC-AUC={rocauc:.4f}")
print(f"{'operating point':28}{'thr':>7}{'prec':>8}{'recall':>8}{'F1':>8}{'alerts/1e6':>12}")
for name,p in pts.items():
    if "precision" in p:
        print(f"{name:28}{p['threshold']:7.3f}{p['precision']:8.4f}{p['recall']:8.4f}{p['f1']:8.4f}{p['alerts_per_million_benign']:12.0f}")
print(f"\nprecision @0.5 — balanced 5:1 test (16.7% prev): {prec_bal:.4f}  vs  "
      f"native ({prev:.1%} prev): {pts['default (0.5)']['precision']:.4f}")
print("saved to", RES)

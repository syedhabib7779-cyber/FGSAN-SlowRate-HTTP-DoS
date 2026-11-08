# ===================== CELL 34 (NEW) — M6(b): per-tool (Slowloris vs SlowHTTPTest) results =====================
# Reports the model's detection performance SEPARATELY for each slow-rate tool, on the
# standard balanced split (matching the paper's main evaluation). Benign-side metrics
# (FPR, false alerts) are shared; detection (recall) and adversarial evasion are per-tool.
# Needs: df_wed, LABEL_KEEP, present_features, cfg, group_indices, set_seed,
#   FeatureGroupedAttentionNet, train_model, predict_torch, evasion_success_rate, to_t,
#   SEED, RATIO_BENIGN_TO_ATTACK, DEVICE.
import os, json, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

try:
    from google.colab import drive; drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/fgsan_ckpt"
except Exception:
    OUT = "./fgsan_ckpt"
RES = f"{OUT}/m6_pertool.json"

# rebuild balanced task retaining the attack subtype label
d = df_wed.copy(); d["Label"] = d["Label"].astype(str).str.strip()
d = d[d["Label"].isin(LABEL_KEEP.keys())].copy(); d["y"] = d["Label"].map(LABEL_KEEP)
d[present_features] = d[present_features].apply(pd.to_numeric, errors="coerce")
d = d.replace([np.inf,-np.inf], np.nan).dropna(subset=present_features)
d = d[["y","Label"] + present_features].reset_index(drop=True)
d = d[~d.duplicated(subset=present_features, keep="first")].reset_index(drop=True)
da, db = d[d.y==1], d[d.y==0]
nb = min(len(db), len(da)*RATIO_BENIGN_TO_ATTACK)
task = pd.concat([da, db.sample(nb, random_state=SEED)]).sample(frac=1, random_state=SEED).reset_index(drop=True)

idx = np.arange(len(task))
itr, itmp = train_test_split(idx, test_size=cfg.val_split+cfg.test_split, stratify=task.y, random_state=SEED)
iva, ite = train_test_split(itmp, test_size=cfg.test_split/(cfg.val_split+cfg.test_split),
                            stratify=task.y.values[itmp], random_state=SEED)
Xall = task[present_features].to_numpy(np.float32); yall = task.y.to_numpy(np.int64)
lab = task["Label"].to_numpy()
sc = StandardScaler().fit(Xall[itr])
Xtr, Xva, Xte = sc.transform(Xall[itr]), sc.transform(Xall[iva]), sc.transform(Xall[ite])
ytr, yva, yte = yall[itr], yall[iva], yall[ite]; lab_te = lab[ite]
set_seed(SEED)
m = FeatureGroupedAttentionNet(group_indices, emb_dim=cfg.emb_dim, n_heads=cfg.n_heads,
                               dropout=cfg.dropout, use_attention=True)
m, _ = train_model(m, Xtr, ytr, Xva, yva, cfg, use_adv=True, use_focal=True)
preds, scores = predict_torch(m, Xte); preds = np.asarray(preds)

# shared benign-side metrics
bmask = yte == 0
tn = int((preds[bmask]==0).sum()); fp = int((preds[bmask]==1).sum())
fpr = fp/max(bmask.sum(),1)
print(f"shared benign side: FPR={fpr:.4f} ({fp} FP / {int(bmask.sum())} benign)\n")

# per-tool detection + adversarial evasion
tools = [t for t in np.unique(lab_te) if t != "BENIGN" and (yte[lab_te==t]==1).all()]
res = {"fpr": fpr, "fp": fp, "benign_test": int(bmask.sum()), "tools": {}}
print(f"{'tool':20}{'n_test':>8}{'recall':>9}{'missed':>8}{'evasion':>9}")
for t in tools:
    mk = (lab_te == t)
    yt = yte[mk]; pt = preds[mk]
    recall = (pt==1).mean()
    # adversarial evasion for this tool's clean-correct flows
    xt = Xte[mk]; cc = pt==1
    if cc.sum() > 0:
        from numpy import asarray
        xadv = pgd_attack(m, to_t(xt[cc]), torch.tensor(yt[cc],dtype=torch.long,device=DEVICE),
                          cfg.adv_epsilon, cfg.adv_alpha, cfg.adv_steps)
        ap,_ = predict_torch(m, xadv.cpu().numpy())
        esr = float((np.asarray(ap)!=1).mean())
    else:
        esr = float("nan")
    res["tools"][t] = {"n_test": int(mk.sum()), "recall": float(recall),
                       "missed": int((pt==0).sum()), "evasion": esr}
    print(f"{t:20}{int(mk.sum()):8d}{recall:9.4f}{int((pt==0).sum()):8d}{esr:9.4f}")

os.makedirs(OUT, exist_ok=True); json.dump(res, open(RES,"w"), indent=1)
print("\nsaved to", RES)
print("Note: precision/F1 are prevalence- and threshold-dependent and are reported jointly in")
print("Table 5 / Table 13; recall and evasion are the per-tool quantities.")

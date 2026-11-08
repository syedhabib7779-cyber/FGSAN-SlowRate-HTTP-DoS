# ===================== CELL 32 (NEW) — M4 add-on: cross-tool threshold recalibration =====================
# The cross-campaign F1 (0.43-0.54 at the default 0.5 threshold) sits against a HIGH AUC
# (0.91-0.96): the model ranks the unseen tool's flows above benign, but the training-tool
# threshold is miscalibrated. This cell retrains the two cross-campaign models, captures test
# scores, and reports F1 at (i) the default 0.5, (ii) an oracle best threshold (ranking upper
# bound), and (iii) a threshold chosen from a small (20%) labelled sample of the unseen tool
# (the realistic deployment case). Needs: df_wed, LABEL_KEEP, present_features, cfg,
# group_indices, set_seed, FeatureGroupedAttentionNet, train_model, predict_torch, SEED,
# RATIO_BENIGN_TO_ATTACK, StandardScaler.
import os, json, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

try:
    from google.colab import drive; drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/fgsan_ckpt"
except Exception:
    OUT = "./fgsan_ckpt"
RES = f"{OUT}/m4_recalib.json"
rec = json.load(open(RES)) if os.path.exists(RES) else {}

# rebuild the same deduplicated, balanced task set (deterministic)
d = df_wed.copy(); d["Label"] = d["Label"].astype(str).str.strip()
d = d[d["Label"].isin(LABEL_KEEP.keys())].copy(); d["y"] = d["Label"].map(LABEL_KEEP)
d[present_features] = d[present_features].apply(pd.to_numeric, errors="coerce")
d = d.replace([np.inf,-np.inf], np.nan).dropna(subset=present_features)
d = d[["y","Label"] + present_features].reset_index(drop=True)
d = d[~d.duplicated(subset=present_features, keep="first")].reset_index(drop=True)
da, db = d[d.y==1], d[d.y==0]
nb = min(len(db), len(da)*RATIO_BENIGN_TO_ATTACK)
task = pd.concat([da, db.sample(nb, random_state=SEED)]).sample(frac=1, random_state=SEED).reset_index(drop=True)
VS = cfg.val_split; TSZ = cfg.test_split
tools = list(task.loc[task.y==1,"Label"].unique())

def best_threshold(scores, yv):
    grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 200)))
    f1s = [f1_score(yv, (scores > t).astype(int), zero_division=0) for t in grid]
    i = int(np.argmax(f1s)); return float(grid[i]), float(f1s[i])

print(f"{'held-out tool':22}{'AUC':>7}{'F1@0.5':>9}{'F1 oracle':>11}{'F1 recalib':>12}")
for tool in tools:
    tag = tool.replace(" ", "_")
    if tag not in rec:
        ben = task[task.y==0].sample(frac=1, random_state=SEED); nb_=len(ben)
        nbte, nbva = int(nb_*TSZ), int(nb_*VS)
        bte, bva, btr = ben.iloc[:nbte], ben.iloc[nbte:nbte+nbva], ben.iloc[nbte+nbva:]
        atest = task[(task.y==1)&(task.Label==tool)]
        atr_all = task[(task.y==1)&(task.Label!=tool)].sample(frac=1, random_state=SEED)
        nava = int(len(atr_all)*VS); ava, atr = atr_all.iloc[:nava], atr_all.iloc[nava:]
        tr, va, te = pd.concat([btr,atr]), pd.concat([bva,ava]), pd.concat([bte,atest])
        Xtr = tr[present_features].to_numpy(np.float32); ytr = tr.y.to_numpy(np.int64)
        Xva = va[present_features].to_numpy(np.float32); yva = va.y.to_numpy(np.int64)
        Xte = te[present_features].to_numpy(np.float32); yte = te.y.to_numpy(np.int64)
        sc = StandardScaler().fit(Xtr); Xtr,Xva,Xte = sc.transform(Xtr),sc.transform(Xva),sc.transform(Xte)
        set_seed(SEED)
        m = FeatureGroupedAttentionNet(group_indices, emb_dim=cfg.emb_dim, n_heads=cfg.n_heads,
                                       dropout=cfg.dropout, use_attention=True)
        m,_ = train_model(m, Xtr, ytr, Xva, yva, cfg, use_adv=True, use_focal=True)
        _, scores = predict_torch(m, Xte)                    # scores = P(attack)
        scores = np.asarray(scores, dtype=np.float64)
        auc = roc_auc_score(yte, scores)
        f1_def = f1_score(yte, (scores > 0.5).astype(int), zero_division=0)
        _, f1_oracle = best_threshold(scores, yte)
        # realistic: pick threshold on a 20% labelled sample of the TEST env, eval on 80%
        g = np.random.default_rng(1); cal = np.zeros(len(yte), bool)
        for cls in [0,1]:
            ix = np.where(yte==cls)[0]; k = max(1,int(len(ix)*0.2))
            cal[g.choice(ix, k, replace=False)] = True
        thr_c, _ = best_threshold(scores[cal], yte[cal])
        f1_recal = f1_score(yte[~cal], (scores[~cal] > thr_c).astype(int), zero_division=0)
        rec[tag] = {"auc": float(auc), "f1_default": float(f1_def),
                    "f1_oracle": float(f1_oracle), "f1_recalib": float(f1_recal),
                    "thr_recalib": float(thr_c)}
        json.dump(rec, open(RES, "w"), indent=1)
    r = rec[tag]
    print(f"{tool:22}{r['auc']:7.3f}{r['f1_default']:9.3f}{r['f1_oracle']:11.3f}{r['f1_recalib']:12.3f}")
print("saved to", RES)
print("\nInterpretation: if F1 recovers sharply from F1@0.5 to F1 recalib, the cross-tool")
print("weakness is a threshold-calibration effect, not a failure to separate the classes.")

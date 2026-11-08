# ===================== CELL 35 (NEW) — M6(a): external validation on CSE-CIC-IDS2018 =====================
# Applies the CICIDS2017-trained FGSAN DIRECTLY to CSE-CIC-IDS2018 (2018 AWS testbed,
# different clients/servers/days) slow-rate DoS flows. Same CICFlowMeter feature set,
# different column names -> a deterministic rename, validated against 2017 value ranges
# before any metric is trusted. No retraining: uses main_model + the 2017 training scaler.
# Needs: main_model, scaler, present_features, predict_torch, to_t, DEVICE, X_test_raw.
import os, glob, numpy as np, pandas as pd, subprocess
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

# ---- 1. download the two IDS2018 DoS days from the public AWS bucket (no credentials) ----
DST="/content/ids2018"; os.makedirs(DST, exist_ok=True)
FILES={"Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv":"Slowloris day (also GoldenEye)",
       "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv":"SlowHTTPTest day (also Hulk)"}
have=[p for p in glob.glob(f"{DST}/*.csv")]
if len(have)<2:
    subprocess.call(["pip","-q","install","awscli"])
    base="s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"
    print(subprocess.run(["aws","s3","ls",base,"--no-sign-request"],capture_output=True,text=True).stdout)
    for f in FILES:
        print("downloading",f,"...")
        subprocess.call(["aws","s3","cp",base+f,f"{DST}/{f}","--no-sign-request"])
csvs=[p for p in glob.glob(f"{DST}/*.csv")]
assert csvs, "download failed; see the aws s3 ls output above and adjust filenames"

# ---- 2. resolve the 27 features across the two naming conventions ----
ALT={ "Total Fwd Packets":"Tot Fwd Pkts","Total Backward Packets":"Tot Bwd Pkts",
      "Total Length of Fwd Packets":"TotLen Fwd Pkts","Total Length of Bwd Packets":"TotLen Bwd Pkts",
      "Fwd Packet Length Mean":"Fwd Pkt Len Mean","Bwd Packet Length Mean":"Bwd Pkt Len Mean",
      "Flow Bytes/s":"Flow Byts/s","Flow Packets/s":"Flow Pkts/s",
      "Fwd IAT Total":"Fwd IAT Tot","Bwd IAT Total":"Bwd IAT Tot" }
def load_ids(p):
    df=pd.read_csv(p, low_memory=False); df.columns=[c.strip() for c in df.columns]
    return df
frames=[]
for p in csvs:
    df=load_ids(p); cols=set(df.columns)
    resolve={}
    missing=[]
    for feat in present_features:
        if feat in cols: resolve[feat]=feat
        elif ALT.get(feat) in cols: resolve[feat]=ALT[feat]
        else: missing.append(feat)
    if missing:
        print(f"!! {os.path.basename(p)} missing features:", missing); continue
    lab_col="Label" if "Label" in cols else [c for c in df.columns if c.lower()=="label"][0]
    sub=df[[resolve[f] for f in present_features]+[lab_col]].copy()
    sub.columns=present_features+["Label"]
    frames.append(sub)
assert frames, "no IDS2018 file had all 27 features; check the aws ls output"
ids=pd.concat(frames, ignore_index=True)
ids["Label"]=ids["Label"].astype(str).str.strip()
print("\nIDS2018 label counts:", ids["Label"].value_counts().to_dict())

# keep benign + the two slow tools
LMAP={"Benign":0,"BENIGN":0,
      "DoS attacks-Slowloris":1,"DoS attacks-SlowHTTPTest":1,
      "DoS attacks-Slowhttptest":1}
ids=ids[ids["Label"].isin(LMAP)].copy()
ids["y"]=ids["Label"].map(LMAP)
ids[present_features]=ids[present_features].apply(pd.to_numeric, errors="coerce")
ids=ids.replace([np.inf,-np.inf],np.nan).dropna(subset=present_features).reset_index(drop=True)
ids=ids[~ids.duplicated(subset=present_features, keep="first")].reset_index(drop=True)
print("after filter/clean/dedup:", len(ids), "| prevalence", f"{ids.y.mean():.3%}",
      "|", ids.groupby('Label').size().to_dict())

# ---- 3. VALIDATE the rename/units against 2017 ranges before trusting results ----
Xids=ids[present_features].to_numpy(np.float64)
ref=np.asarray(X_test_raw, dtype=np.float64)           # 2017 raw test features
print("\nfeature-range sanity (median 2018 vs 2017; ratio should be within ~1-2 orders):")
bad=0
for j,f in enumerate(present_features):
    m18=np.median(Xids[:,j]); m17=np.median(ref[:,j])
    ratio=(m18+1e-9)/(m17+1e-9)
    flag="" 
    if not (0.01 <= abs(ratio) <= 100) and not (abs(m18)<1e-6 and abs(m17)<1e-6):
        flag=" <-- CHECK"; bad+=1
    if j<6 or flag:
        print(f"   {f:30} 2018={m18:14.3f}  2017={m17:14.3f}  ratio={ratio:9.3f}{flag}")
print(f"features with suspicious range mismatch: {bad}/27",
      "(if many, the rename/units are wrong -- STOP and finalize M6 instead)" if bad>5 else "(ranges consistent)")

# ---- 4. apply the 2017-trained model (2017 scaler) ----
Xz=scaler.transform(ids[present_features].to_numpy(np.float32))
preds, scores = predict_torch(main_model, Xz)
preds=np.asarray(preds); scores=np.asarray(scores, dtype=np.float64); y=ids.y.to_numpy(int)

def metrics_at(thr):
    p=(scores>thr).astype(int); tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    fpr=fp/(fp+tn) if fp+tn else 0; f1=2*prec*rec/(prec+rec) if prec+rec else 0
    return dict(thr=thr,precision=prec,recall=rec,f1=f1,fpr=fpr,alerts_per_M=fpr*1e6)

prauc=average_precision_score(y,scores); rocauc=roc_auc_score(y,scores)
print(f"\n=== EXTERNAL VALIDATION on CSE-CIC-IDS2018 (n={len(y)}, prevalence {y.mean():.2%}) ===")
print(f"PR-AUC={prauc:.4f}  ROC-AUC={rocauc:.4f}")
m=metrics_at(0.5)
print(f"default 0.5: precision={m['precision']:.4f} recall={m['recall']:.4f} "
      f"F1={m['f1']:.4f} FPR={m['fpr']:.4f} alerts/1e6={m['alerts_per_M']:.0f}")
# per-tool recall
for t in sorted(ids["Label"].unique()):
    if LMAP.get(t)==1:
        mk=(ids["Label"]==t).to_numpy()
        print(f"   recall[{t}] = {(preds[mk]==1).mean():.4f}  (n={int(mk.sum())})")
# recalibrated threshold on a 20% labelled sample of IDS2018
g=np.random.default_rng(1); cal=np.zeros(len(y),bool)
for cls in [0,1]:
    ix=np.where(y==cls)[0]; cal[g.choice(ix,max(1,int(len(ix)*0.2)),replace=False)]=True
grid=np.unique(np.quantile(scores[cal],np.linspace(0.5,0.999,300)))
from sklearn.metrics import f1_score
best=max(grid, key=lambda t: f1_score(y[cal],(scores[cal]>t).astype(int),zero_division=0))
mr=metrics_at(best)
print(f"recalibrated (thr={best:.3f} from 20% sample): precision={mr['precision']:.4f} "
      f"recall={mr['recall']:.4f} F1={mr['f1']:.4f} alerts/1e6={mr['alerts_per_M']:.0f}")
try:
    import json; json.dump({"pr_auc":prauc,"roc_auc":rocauc,"default":m,"recalibrated":mr,
        "prevalence":float(y.mean()),"n":int(len(y))},
        open("/content/drive/MyDrive/fgsan_ckpt/m6_external.json","w"),indent=1)
except Exception: pass

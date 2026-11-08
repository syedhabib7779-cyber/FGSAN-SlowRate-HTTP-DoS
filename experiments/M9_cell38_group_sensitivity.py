# ===================== CELL 38 (NEW) — M9: feature-group design sensitivity =====================
# Isolates the benefit of the SEMANTIC grouping from parameter count by training the identical
# architecture under: (1) the semantic 8/14/5 grouping, (2) 10 RANDOM groupings of the same
# sizes (identical parameter count -> a group-permutation null), and (3) an ALTERNATIVE semantic
# grouping (by direction). Matched-parameter no-attention pooling is the existing Table 7
# ablation. Reports where the semantic grouping falls in the random null (permutation p-values).
# Needs: df_wed, LABEL_KEEP, present_features, cfg, set_seed, FeatureGroupedAttentionNet,
#   train_model, evaluate_torch, predict_torch, pgd_attack, evasion_success_rate, to_t, SEED,
#   DEVICE, RATIO_BENIGN_TO_ATTACK.
import os, json, numpy as np, pandas as pd, torch
import torch.nn.functional as F   # restore F if an earlier run clobbered it
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from google.colab import drive; drive.mount("/content/drive")
    OUT="/content/drive/MyDrive/fgsan_ckpt"
except Exception:
    OUT="./fgsan_ckpt"
os.makedirs(OUT, exist_ok=True)
RES=f"{OUT}/m9_results.json"
R=json.load(open(RES)) if os.path.exists(RES) else {}
def save(): json.dump(R,open(RES,"w"),indent=1)
N_RANDOM=10

# fixed split
d=df_wed.copy(); d["Label"]=d["Label"].astype(str).str.strip()
d=d[d["Label"].isin(LABEL_KEEP.keys())].copy(); d["y"]=d["Label"].map(LABEL_KEEP)
d[present_features]=d[present_features].apply(pd.to_numeric,errors="coerce")
d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=present_features)
d=d[["y"]+present_features].reset_index(drop=True)
d=d[~d.duplicated(subset=present_features,keep="first")].reset_index(drop=True)
da,db=d[d.y==1],d[d.y==0]; nb=min(len(db),len(da)*RATIO_BENIGN_TO_ATTACK)
task=pd.concat([da,db.sample(nb,random_state=SEED)]).sample(frac=1,random_state=SEED).reset_index(drop=True)
X=task[present_features].to_numpy(np.float32); y=task.y.to_numpy(np.int64)
itr,itmp=train_test_split(np.arange(len(y)),test_size=cfg.val_split+cfg.test_split,stratify=y,random_state=SEED)
iva,ite=train_test_split(itmp,test_size=cfg.test_split/(cfg.val_split+cfg.test_split),stratify=y[itmp],random_state=SEED)
sc=StandardScaler().fit(X[itr]); Xtr,Xva,Xte=sc.transform(X[itr]),sc.transform(X[iva]),sc.transform(X[ite])
ytr,yva,yte=y[itr],y[iva],y[ite]
am=yte==1; Xatk,yatk=Xte[am],yte[am]; yatk_t=torch.tensor(yatk,dtype=torch.long,device=DEVICE)

FEATS=present_features
def idx(names): return [FEATS.index(n) for n in names]
# (1) semantic 8/14/5
G_SEM={"volume":idx(["Total Fwd Packets","Total Backward Packets","Total Length of Fwd Packets",
        "Total Length of Bwd Packets","Fwd Packet Length Mean","Bwd Packet Length Mean","Flow Bytes/s","Flow Packets/s"]),
       "timing":idx(["Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Total","Fwd IAT Mean",
        "Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min"]),
       "behavior":idx(["Flow Duration","Active Mean","Active Std","Idle Mean","Idle Std"])}
# (3) alternative semantic: by direction (sizes 8/8/11)
G_ALT={"dir_fwd":idx(["Total Fwd Packets","Total Length of Fwd Packets","Fwd Packet Length Mean",
        "Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min"]),
       "dir_bwd":idx(["Total Backward Packets","Total Length of Bwd Packets","Bwd Packet Length Mean",
        "Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min"]),
       "dir_flow":idx(["Flow Duration","Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max",
        "Flow IAT Min","Active Mean","Active Std","Idle Mean","Idle Std"])}
def random_grouping(seed, sizes=(8,14,5)):
    p=np.random.default_rng(1000+seed).permutation(27).tolist()
    g={}; s=0
    for i,sz in enumerate(sizes): g[f"g{i}"]=p[s:s+sz]; s+=sz
    return g

def run(gi, tag):
    if tag in R: return R[tag]
    set_seed(SEED)
    m=FeatureGroupedAttentionNet(gi,emb_dim=cfg.emb_dim,n_heads=cfg.n_heads,dropout=cfg.dropout,use_attention=True)
    m,_=train_model(m,Xtr,ytr,Xva,yva,cfg,use_adv=True,use_focal=True)
    met=evaluate_torch(m,Xte,yte)
    cp,_=predict_torch(m,Xatk)
    adv=pgd_attack(m,to_t(Xatk),yatk_t,cfg.adv_epsilon,cfg.adv_alpha,cfg.adv_steps)
    ap,_=predict_torch(m,adv.cpu().numpy())
    esr=evasion_success_rate(yatk,np.asarray(cp),np.asarray(ap))
    R[tag]={"f1":float(met["f1"]),"auc":float(met.get("auc",float("nan"))),"esr":float(esr)}; save()
    print(f"  {tag:16} F1={met['f1']:.4f} ESR={esr:.4f}")
    return R[tag]

print("(1) semantic grouping"); sem=run(G_SEM,"semantic")
print("(3) alternative semantic (by direction)"); alt=run(G_ALT,"alt_semantic")
print(f"(2) {N_RANDOM} random groupings (same sizes, matched params)")
for s in range(N_RANDOM): run(random_grouping(s),f"random_{s}")

rand=[R[f"random_{s}"] for s in range(N_RANDOM)]
rf1=np.array([r["f1"] for r in rand]); re_=np.array([r["esr"] for r in rand])
# permutation p-values: P(random F1 >= semantic F1) and P(random ESR <= semantic ESR)
p_f1=(1+int((rf1>=sem["f1"]).sum()))/(N_RANDOM+1)
p_esr=(1+int((re_<=sem["esr"]).sum()))/(N_RANDOM+1)
print("\n=== M9 feature-group sensitivity ===")
print(f"semantic:        F1={sem['f1']:.4f}  ESR={sem['esr']:.4f}")
print(f"alt-semantic:    F1={alt['f1']:.4f}  ESR={alt['esr']:.4f}")
print(f"random null F1:  mean={rf1.mean():.4f}  sd={rf1.std(ddof=1):.4f}  range=[{rf1.min():.4f},{rf1.max():.4f}]")
print(f"random null ESR: mean={re_.mean():.4f}  sd={re_.std(ddof=1):.4f}  range=[{re_.min():.4f},{re_.max():.4f}]")
print(f"permutation p (random F1 >= semantic):  {p_f1:.3f}")
print(f"permutation p (random ESR <= semantic): {p_esr:.3f}")
R["summary"]={"sem":sem,"alt":alt,"rand_f1_mean":float(rf1.mean()),"rand_f1_sd":float(rf1.std(ddof=1)),
  "rand_f1_range":[float(rf1.min()),float(rf1.max())],"rand_esr_mean":float(re_.mean()),
  "rand_esr_range":[float(re_.min()),float(re_.max())],"p_f1":p_f1,"p_esr":p_esr}; save()
print("\nReading: small p_f1 => semantic grouping better than most random same-size groupings;")
print("p_f1 near 0.5 => the architecture, not the semantic assignment, drives the benefit.")
print("saved to", RES)

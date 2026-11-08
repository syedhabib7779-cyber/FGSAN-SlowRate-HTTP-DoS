# ===================== CELL 31 (v2) — M4: leakage & split-independence =====================
# Rebuilds the dataset WITH metadata (Timestamp, Source IP, attack subtype) that Cell 4
# dropped, then (1) deduplicates and reports exact/near/cross-class duplicates, and
# (2) retrains FGSAN under leakage-controlled splits on the SAME deduplicated data:
#     random stratified | per-class temporal holdout | cross-campaign (by attack tool).
# Host/IP grouping is reported but, for CICIDS2017 Wednesday (one attacker host), it is
# degenerate, so a cross-campaign split is used instead — exactly the "campaign" control
# the reviewer lists. Any gap vs random quantifies campaign / near-duplicate leakage.
import os, json, numpy as np, pandas as pd, torch, glob
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    from google.colab import drive; drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/fgsan_ckpt"
except Exception:
    OUT = "./fgsan_ckpt"
os.makedirs(OUT, exist_ok=True)
RES = f"{OUT}/m4_results.json"
results = json.load(open(RES)) if os.path.exists(RES) else {}
def save(): json.dump(results, open(RES, "w"), indent=1)

assert "df_wed" in globals(), "run the download cell first so df_wed is the raw Wednesday file"
SRCIP = next((c for c in ["Source IP","Src IP"] if c in df_wed.columns), None)
TS    = "Timestamp" if "Timestamp" in df_wed.columns else None
print("metadata:", "Timestamp" if TS else "(no Timestamp)", "|", SRCIP or "(no Source IP)")

d = df_wed.copy(); d["Label"] = d["Label"].astype(str).str.strip()
d = d[d["Label"].isin(LABEL_KEEP.keys())].copy()
d["y"] = d["Label"].map(LABEL_KEEP)
d[present_features] = d[present_features].apply(pd.to_numeric, errors="coerce")
d = d.replace([np.inf,-np.inf], np.nan).dropna(subset=present_features)
keep = ["y","Label"] + present_features + [c for c in [TS,SRCIP] if c]
d = d[keep].reset_index(drop=True)

# ---------- dedup ----------
dup = d.duplicated(subset=present_features, keep="first")
near = d[present_features].round(3).duplicated(keep="first")
key = d[present_features].round(3).astype(str).agg("|".join, axis=1)
nxc = int((d.assign(_k=key).groupby("_k")["y"].nunique() > 1).sum())
print(f"\nDEDUP: exact {dup.sum()} ({dup.mean():.2%}) | near(3dp) {near.sum()} ({near.mean():.2%}) | cross-class keys {nxc}")
results["dedup"] = {"n_total": int(len(d)), "exact": int(dup.sum()), "near": int(near.sum()),
                    "cross_class_keys": nxc}; save()
dd = d[~dup].reset_index(drop=True)
da, db = dd[dd.y==1], dd[dd.y==0]
nb = min(len(db), len(da)*RATIO_BENIGN_TO_ATTACK)
task = pd.concat([da, db.sample(nb, random_state=SEED)]).sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"       after dedup + {RATIO_BENIGN_TO_ATTACK}:1 balance: {len(task)} "
      f"(benign {int((task.y==0).sum())}, attack {int((task.y==1).sum())})")
print("       attack subtypes:", task[task.y==1]["Label"].value_counts().to_dict())

VS, TSZ = cfg.val_split, cfg.test_split

def run_fgsan(tr, va, te, tag):
    if tag in results and tag != "dedup":
        print(f"  [{tag}] cached: F1={results[tag]['f1']:.4f} evasion={results[tag]['evasion']:.4f}"); return
    Xtr=tr[present_features].to_numpy(np.float32); ytr=tr.y.to_numpy(np.int64)
    Xva=va[present_features].to_numpy(np.float32); yva=va.y.to_numpy(np.int64)
    Xte=te[present_features].to_numpy(np.float32); yte=te.y.to_numpy(np.int64)
    sc=StandardScaler().fit(Xtr); Xtr,Xva,Xte=sc.transform(Xtr),sc.transform(Xva),sc.transform(Xte)
    set_seed(SEED)
    m=FeatureGroupedAttentionNet(group_indices, emb_dim=cfg.emb_dim, n_heads=cfg.n_heads,
                                 dropout=cfg.dropout, use_attention=True)
    m,_=train_model(m, Xtr, ytr, Xva, yva, cfg, use_adv=True, use_focal=True)
    met=evaluate_torch(m, Xte, yte); am=yte==1
    xa,ya=Xte[am],yte[am]; cp,_=predict_torch(m,xa)
    xadv=pgd_attack(m, to_t(xa), torch.tensor(ya,dtype=torch.long,device=DEVICE),
                    cfg.adv_epsilon, cfg.adv_alpha, cfg.adv_steps)
    ap,_=predict_torch(m, xadv.cpu().numpy()); esr=evasion_success_rate(ya,cp,ap)
    results[tag]={"f1":float(met["f1"]),"auc":float(met.get("auc",float("nan"))),
                  "fpr":float(met["fpr"]),"fnr":float(met["fnr"]),"evasion":float(esr),
                  "n_test":int(len(yte)),"test_attack":int(am.sum())}; save()
    print(f"  [{tag}] F1={met['f1']:.4f} AUC={met.get('auc',float('nan')):.4f} "
          f"FPR={met['fpr']:.4f} FNR={met['fnr']:.4f} evasion={esr:.4f} (test n={len(yte)}, atk={int(am.sum())})")

# (a) random stratified
print("\n(a) RANDOM stratified split")
itr,itmp=train_test_split(np.arange(len(task)), test_size=VS+TSZ, stratify=task.y, random_state=SEED)
iva,ite=train_test_split(itmp, test_size=TSZ/(VS+TSZ), stratify=task.y.values[itmp], random_state=SEED)
run_fgsan(task.iloc[itr], task.iloc[iva], task.iloc[ite], "random")

# (b) per-class temporal holdout
if TS:
    print("\n(b) TEMPORAL per-class holdout (latest flows of each class held out)")
    t=task.copy(); t["_ts"]=pd.to_datetime(t[TS], errors="coerce", dayfirst=True)
    if t["_ts"].isna().mean()>0.5: t["_ts"]=pd.to_datetime(t[TS], errors="coerce")
    t=t.dropna(subset=["_ts"]); tr,va,te=[],[],[]
    for cls in [0,1]:
        g=t[t.y==cls].sort_values("_ts"); n=len(g); nte=int(n*TSZ); nva=int(n*VS)
        te.append(g.iloc[n-nte:]); va.append(g.iloc[n-nte-nva:n-nte]); tr.append(g.iloc[:n-nte-nva])
    TE=pd.concat(te); print("   test attack subtypes:", TE[TE.y==1]["Label"].value_counts().to_dict())
    run_fgsan(pd.concat(tr), pd.concat(va), TE, "temporal")

# (c) campaign control
atk_ips = task[SRCIP].nunique() if SRCIP else 0
if SRCIP:
    print(f"\nattacker Source IPs in attack class: {task.loc[task.y==1, SRCIP].nunique()}")
tools=[t for t in task.loc[task.y==1,"Label"].unique()]
if SRCIP and task.loc[task.y==1, SRCIP].nunique() >= 3:
    print("\n(c) HOST-GROUPED split by Source IP")
    gss=GroupShuffleSplit(n_splits=1, test_size=TSZ, random_state=SEED)
    i1,ite2=next(gss.split(task, task.y, task[SRCIP].values)); sub=task.iloc[i1]
    g2=GroupShuffleSplit(n_splits=1, test_size=VS/(1-TSZ), random_state=SEED)
    i2,iv2=next(g2.split(sub, sub.y, sub[SRCIP].values))
    run_fgsan(sub.iloc[i2], sub.iloc[iv2], task.iloc[ite2], "grouped")
else:
    print(f"\n(c) CROSS-CAMPAIGN split (host grouping degenerate: attacks from a single host). "
          f"Tools: {tools}")
    for tool in tools:
        rng=np.random.default_rng(SEED)
        ben=task[task.y==0].sample(frac=1, random_state=SEED); nb=len(ben)
        nbte,nbva=int(nb*TSZ),int(nb*VS)
        bte,bva,btr=ben.iloc[:nbte],ben.iloc[nbte:nbte+nbva],ben.iloc[nbte+nbva:]
        atest=task[(task.y==1)&(task.Label==tool)]
        atr_all=task[(task.y==1)&(task.Label!=tool)].sample(frac=1, random_state=SEED)
        nava=int(len(atr_all)*VS); ava,atr=atr_all.iloc[:nava],atr_all.iloc[nava:]
        tag=f"campaign_test={tool.replace(' ','_')}"
        print(f"   [{tool}] held out as test (train on the other tool)")
        run_fgsan(pd.concat([btr,atr]), pd.concat([bva,ava]), pd.concat([bte,atest]), tag)

# ---------- comparison ----------
print("\n=== M4 split-independence comparison ===")
print(f"{'protocol':34}{'F1':>8}{'AUC':>8}{'FPR':>8}{'FNR':>8}{'evasion':>9}{'testATK':>9}")
for tag,r in results.items():
    if tag=="dedup": continue
    print(f"{tag:34}{r['f1']:8.4f}{r['auc']:8.4f}{r['fpr']:8.4f}{r['fnr']:8.4f}{r['evasion']:9.4f}{r['test_attack']:9d}")
print("saved to", RES)

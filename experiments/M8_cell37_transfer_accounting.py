# ===================== CELL 37 (NEW) — M8: transfer eligibility accounting =====================
# Publishes the exact bookkeeping behind the transfer evasion rates: source (FGSAN) and
# target (each baseline) clean-correct counts, the eligible subsets and their overlap,
# transferred-evasion counts, and the clean-vs-adversarial confusion for the attack class,
# under two denominators: target-clean-correct (the paper's convention) and both-correct.
# Self-contained: trains the main model + the 5 baselines on the standard split, then
# crafts FGSAN PGD adversarial examples once.
# Needs: df_wed, LABEL_KEEP, present_features, cfg, group_indices, set_seed,
#   FeatureGroupedAttentionNet, train_model, predict_torch, pgd_attack, to_t, SEED, DEVICE,
#   RATIO_BENIGN_TO_ATTACK.
import os, json, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from google.colab import drive; drive.mount("/content/drive")
    OUT="/content/drive/MyDrive/fgsan_ckpt"
except Exception:
    OUT="./fgsan_ckpt"
os.makedirs(OUT, exist_ok=True)

# ---- standard balanced split ----
d=df_wed.copy(); d["Label"]=d["Label"].astype(str).str.strip()
d=d[d["Label"].isin(LABEL_KEEP.keys())].copy(); d["y"]=d["Label"].map(LABEL_KEEP)
d[present_features]=d[present_features].apply(pd.to_numeric, errors="coerce")
d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=present_features)
d=d[["y"]+present_features].reset_index(drop=True)
d=d[~d.duplicated(subset=present_features,keep="first")].reset_index(drop=True)
da,db=d[d.y==1],d[d.y==0]; nb=min(len(db),len(da)*RATIO_BENIGN_TO_ATTACK)
task=pd.concat([da,db.sample(nb,random_state=SEED)]).sample(frac=1,random_state=SEED).reset_index(drop=True)
X=task[present_features].to_numpy(np.float32); y=task.y.to_numpy(np.int64)
itr,itmp=train_test_split(np.arange(len(y)),test_size=cfg.val_split+cfg.test_split,stratify=y,random_state=SEED)
iva,ite=train_test_split(itmp,test_size=cfg.test_split/(cfg.val_split+cfg.test_split),stratify=y[itmp],random_state=SEED)
sc=StandardScaler().fit(X[itr])
Xtr,Xva,Xte=sc.transform(X[itr]),sc.transform(X[iva]),sc.transform(X[ite]); ytr,yva,yte=y[itr],y[iva],y[ite]

# ---- train source (FGSAN) + target baselines ----
set_seed(SEED)
fg=FeatureGroupedAttentionNet(group_indices,emb_dim=cfg.emb_dim,n_heads=cfg.n_heads,dropout=cfg.dropout,use_attention=True)
fg,_=train_model(fg,Xtr,ytr,Xva,yva,cfg,use_adv=True,use_focal=True)
SVM_MAX=int(globals().get("SVM_MAX_TRAIN",20000))
def make_baselines():
    return {"Random Forest":RandomForestClassifier(n_estimators=300,random_state=SEED,n_jobs=-1),
            "XGBoost":XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.1,subsample=0.9,
                                    colsample_bytree=0.9,eval_metric="logloss",random_state=SEED,n_jobs=-1),
            "LightGBM":LGBMClassifier(n_estimators=300,max_depth=-1,learning_rate=0.1,random_state=SEED,n_jobs=-1,verbose=-1),
            "SVM (RBF)":SVC(kernel="rbf",probability=True,random_state=SEED),
            "MLP (sklearn)":MLPClassifier(hidden_layer_sizes=(128,64),max_iter=300,random_state=SEED)}
base={}
for nm,clf in make_baselines().items():
    if nm=="SVM (RBF)" and len(ytr)>SVM_MAX:
        s=np.random.default_rng(SEED).choice(len(ytr),SVM_MAX,replace=False); clf.fit(Xtr[s],ytr[s])
    else: clf.fit(Xtr,ytr)
    base[nm]=clf

# ---- craft FGSAN adversarial examples on the test ATTACK flows ----
am=yte==1; Xatk=Xte[am]; natk=int(am.sum())
yatk_t=torch.tensor(np.ones(natk,dtype=np.int64),device=DEVICE)
Xadv=pgd_attack(fg,to_t(Xatk),yatk_t,cfg.adv_epsilon,cfg.adv_alpha,cfg.adv_steps).cpu().numpy()

# ---- source (FGSAN) eligibility ----
src_clean,_=predict_torch(fg,Xatk); src_clean=np.asarray(src_clean)   # 1 = detected
src_cc = src_clean==1                                                 # FGSAN clean-correct mask
src_adv,_=predict_torch(fg,Xadv); src_adv=np.asarray(src_adv)
print(f"Attack flows in test set: {natk}")
print(f"SOURCE FGSAN clean-correct: {int(src_cc.sum())}/{natk}")
print(f"SOURCE FGSAN evaded (of its clean-correct): {int(((src_adv==0)&src_cc).sum())}\n")

# ---- per-target accounting ----
print(f"{'target':15}{'clean-correct':>14}{'adv caught':>11}{'adv evaded':>11}"
      f"{'ESR(tgt)':>10}{'overlap':>9}{'ESR(both)':>10}")
rows={}
for nm,clf in base.items():
    tc=clf.predict(Xatk)==1          # target clean-correct mask
    ta=clf.predict(Xadv)             # target prediction on adversarial examples
    caught=int((ta[tc]==1).sum()); evaded=int((ta[tc]==0).sum())
    esr_tgt=evaded/max(tc.sum(),1)
    both=tc & src_cc                 # eligible under BOTH source and target correct
    evaded_both=int((ta[both]==0).sum()); esr_both=evaded_both/max(both.sum(),1)
    rows[nm]={"target_clean_correct":int(tc.sum()),"adv_caught":caught,"adv_evaded":evaded,
              "esr_target_denom":esr_tgt,"overlap_both_correct":int(both.sum()),
              "evaded_both":evaded_both,"esr_both_denom":esr_both}
    print(f"{nm:15}{int(tc.sum()):14d}{caught:11d}{evaded:11d}{esr_tgt:10.4f}"
          f"{int(both.sum()):9d}{esr_both:10.4f}")

out={"n_attack":natk,"source_clean_correct":int(src_cc.sum()),"per_target":rows}
json.dump(out,open(f"{OUT}/m8_transfer_accounting.json","w"),indent=2)
print("\nDenominators: ESR(tgt) uses the target's clean-correct flows (paper convention);")
print("ESR(both) uses flows BOTH FGSAN and the target classify correctly on clean input.")
print("adv caught + adv evaded = target clean-correct (this is the per-model confusion on")
print("adversarial examples for the attack class: caught=still attack, evaded=flipped to benign).")
print("saved to", f"{OUT}/m8_transfer_accounting.json")

# ===================== CELL (NEW) — M21: build the reproducibility package =====================
import os, json, hashlib, numpy as np
from datetime import datetime, timezone
try:
    from google.colab import drive; drive.mount("/content/drive"); OUT="/content/drive/MyDrive/fgsan_ckpt"
except Exception: OUT="./fgsan_ckpt"
PKG=f"{OUT}/reproducibility_package"; os.makedirs(PKG, exist_ok=True)

SEED=globals().get("SEED",42)
cfg_dict={a: getattr(cfg,a) for a in dir(cfg) if not a.startswith("_") and not callable(getattr(cfg,a))}
seeds={"primary":SEED,
       "m7_evasion_uncertainty":"multi-seed set used for the evasion confidence interval",
       "m9_random_groupings":[1000+s for s in range(10)],
       "m11_stress_multiseed":[100+s for s in range(5)],
       "m15_operating_point":SEED,
       "m16_attack_sensitivity":[0,1,2,3,4]}

pkg={"generated_utc":datetime.now(timezone.utc).isoformat(),
     "dataset":"CICIDS2017 (Wednesday capture), CIC — binary task: BENIGN vs slow-rate HTTP DoS",
     "feature_columns_27":list(present_features),
     "label_map":dict(LABEL_KEEP),
     "config":cfg_dict,
     "seeds":seeds,
     "benign_to_attack_ratio":globals().get("RATIO_BENIGN_TO_ATTACK",5)}

try:
    from sklearn.model_selection import train_test_split
    y_all=df_task["y"].to_numpy(np.int64); idx=np.arange(len(y_all))
    itr,itmp=train_test_split(idx,test_size=cfg.val_split+cfg.test_split,stratify=y_all,random_state=SEED)
    rel=cfg.test_split/(cfg.val_split+cfg.test_split)
    iva,ite=train_test_split(itmp,test_size=rel,stratify=y_all[itmp],random_state=SEED)
    np.savez_compressed(f"{PKG}/split_indices.npz", idx_train=itr, idx_val=iva, idx_test=ite)
    yte=y_all[ite]
    np.savez_compressed(f"{PKG}/attack_masks.npz",
                        test_attack_idx=ite[yte==1], test_benign_idx=ite[yte==0])
    pkg["split_sizes"]={"train":len(itr),"val":len(iva),"test":len(ite),
                        "test_attack":int((yte==1).sum()),"test_benign":int((yte==0).sum())}
    pkg["split_indices_file"]="split_indices.npz"; pkg["attack_masks_file"]="attack_masks.npz"
    print("exact split indices written:", pkg["split_sizes"])
except Exception as e:
    pkg["split_note"]=("Exact indices are reproducible from the primary seed (42) and the split recipe: "
                       "sklearn train_test_split on arange(N), stratified, test_size=val_split+test_split, "
                       "then split the remainder. Re-run the data cell to regenerate split_indices.npz.")
    print("df_task not present; recorded the deterministic split recipe instead:", repr(e)[:80])

cks={}
for f in sorted(os.listdir(OUT)):
    p=os.path.join(OUT,f)
    if os.path.isfile(p) and f.endswith((".pt",".joblib",".npz",".json")):
        cks[f]=hashlib.md5(open(p,"rb").read()).hexdigest()
pkg["artifact_md5"]=cks
json.dump(pkg, open(f"{PKG}/reproducibility_package.json","w"), indent=2)

md=f"""# Reproducibility Package - FGSAN slow-rate HTTP DoS detector

Generated {pkg['generated_utc']}

- Dataset: {pkg['dataset']}
- Feature columns (27): reproducibility_package.json -> feature_columns_27
- Config / hyperparameters: -> config (seed {SEED}, {cfg_dict.get('epochs','?')} epochs, eps={cfg_dict.get('adv_epsilon','?')}, alpha={cfg_dict.get('adv_alpha','?')}, steps={cfg_dict.get('adv_steps','?')}, focal alpha={cfg_dict.get('focal_alpha','?')}, gamma={cfg_dict.get('focal_gamma','?')})
- Seeds: -> seeds
- Split indices: split_indices.npz (idx_train / idx_val / idx_test)
- Attack masks: attack_masks.npz (test attack / benign indices)
- Checkpoints + checksums: weights in parent folder; MD5s under artifact_md5
- Code / adversarial scripts: training notebook, functionality-preserving PGD, attack-strength diagnostics (M16)
"""
open(f"{PKG}/REPRODUCIBILITY.md","w").write(md)
print("\nwrote package to", PKG)
for f in sorted(os.listdir(PKG)): print("  -", f)

# ============================ CHECKPOINT — SAVE ============================
# Run this ONCE after your models are trained and the adversarial arrays exist.
# It writes everything expensive to Google Drive, which survives a Colab
# disconnect. Re-running it later overwrites the checkpoint with the latest state.

import os, json, numpy as np, torch
try:
    import joblib
except Exception:
    import pickle as _pk
    class joblib:                      # minimal shim if joblib missing
        @staticmethod
        def dump(o, p):  _pk.dump(o, open(p, "wb"))
        @staticmethod
        def load(p):     return _pk.load(open(p, "rb"))

# --- choose a location that survives disconnects ---
try:
    from google.colab import drive
    drive.mount("/content/drive")
    CKPT = "/content/drive/MyDrive/fgsan_ckpt"
    ON_DRIVE = True
except Exception:
    CKPT = "./fgsan_ckpt"             # local fallback (does NOT survive disconnect)
    ON_DRIVE = False
os.makedirs(CKPT, exist_ok=True)
print("checkpoint dir:", CKPT, "(survives disconnect)" if ON_DRIVE else "(LOCAL — will NOT survive)")

G = globals()
manifest = {"torch_models": {}, "sklearn_models": [], "arrays": [], "meta": [], "model_dicts": {}}

# --- 1. torch models: main_model + every ablation variant ---
def _save_torch(name, model, kwargs):
    torch.save(model.state_dict(), f"{CKPT}/{name}.pt")
    manifest["torch_models"][name] = kwargs

if "main_model" in G:
    _save_torch("main_model", G["main_model"],
                {"use_attention": True, "which": "main"})

if "ablation_models" in G and isinstance(G["ablation_models"], dict):
    for i, (nm, mdl) in enumerate(G["ablation_models"].items()):
        ua = getattr(mdl, "use_attention", None)
        if ua is None:  # infer from the config table if attribute absent
            ua = "No attention" not in nm and "no attention" not in nm
        _save_torch(f"abl_{i}", mdl, {"use_attention": bool(ua), "label": nm})

# --- 2. sklearn / xgboost / lightgbm baselines: auto-discover by .predict ---
for nm, obj in list(G.items()):
    if nm.startswith("_"):
        continue
    mod = type(obj).__module__ if hasattr(obj, "__class__") else ""
    if hasattr(obj, "predict") and any(k in mod for k in
                                       ("sklearn", "xgboost", "lightgbm")):
        try:
            joblib.dump(obj, f"{CKPT}/skl_{nm}.joblib")
            manifest["sklearn_models"].append(nm)
        except Exception as e:
            print("  skip", nm, "->", e)

# --- 2b. dicts that hold trained models (e.g. a 'baselines' dict) ---
for dname in ("baselines", "models", "baseline_models", "ablation_models"):
    dv = G.get(dname)
    if isinstance(dv, dict):
        keys = []
        for k, mdl in dv.items():
            dmod = type(mdl).__module__ if hasattr(mdl, "__class__") else ""
            if hasattr(mdl, "predict") and any(j in dmod for j in ("sklearn","xgboost","lightgbm")):
                safe = "".join(ch if ch.isalnum() else "_" for ch in str(k))
                try:
                    joblib.dump(mdl, f"{CKPT}/dict_{dname}__{safe}.joblib")
                    keys.append([str(k), safe])
                except Exception as e:
                    print("  skip", dname, k, "->", e)
        if keys:
            manifest["model_dicts"][dname] = keys

# --- 3. scaler + metadata ---
for nm in ["scaler"]:
    if nm in G:
        joblib.dump(G[nm], f"{CKPT}/{nm}.joblib"); manifest["meta"].append(nm)

meta = {}
for nm in ["present_features", "group_indices", "direction_mask_np"]:
    if nm in G:
        v = G[nm]
        meta[nm] = v.tolist() if isinstance(v, np.ndarray) else v
if "cfg" in G:
    c = G["cfg"]
    meta["cfg"] = {k: v for k, v in vars(c).items()
                   if isinstance(v, (int, float, str, bool))} if hasattr(c, "__dict__") else {}
json.dump(meta, open(f"{CKPT}/meta.json", "w"), default=lambda o: None)

# --- 4. arrays (data splits + adversarial outputs). Save whatever exists. ---
array_names = ["X_train","y_train","X_val","y_val","X_test","y_test","X_test_raw",
               "X_atk","y_atk","X_atk_adv_np","raw_adv_unc","raw_adv_eps","raw_f",
               "raw_eps","X_eps","X_f"]
arrs = {}
for nm in array_names:
    if nm in G and isinstance(G[nm], np.ndarray):
        arrs[nm] = G[nm]; manifest["arrays"].append(nm)
if arrs:
    np.savez_compressed(f"{CKPT}/arrays.npz", **arrs)

json.dump(manifest, open(f"{CKPT}/manifest.json", "w"), indent=1)
print("\nsaved:")
print("  torch models :", list(manifest["torch_models"]))
print("  baselines    :", manifest["sklearn_models"])
print("  arrays       :", manifest["arrays"])
print("  meta         :", manifest["meta"], "+ meta.json")
print("DONE. After a disconnect, re-run the DEFINITION cells, then the RESTORE cell.")

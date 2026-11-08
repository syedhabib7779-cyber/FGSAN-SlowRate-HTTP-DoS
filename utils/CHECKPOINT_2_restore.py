# ============================ CHECKPOINT — RESTORE (fixed) ============================
# Run AFTER re-running the fast definition cells (imports, cfg, the
# FeatureGroupedAttentionNet / FocalLoss class cell, and helpers: to_t,
# predict_torch, evasion_success_rate, pgd_attack). Reloads trained models, scaler,
# data splits and adversarial arrays from Drive -- skip ALL training cells.
import os, json, numpy as np, torch
try:
    import joblib
except Exception:
    import pickle as _pk
    class joblib:
        @staticmethod
        def load(p): return _pk.load(open(p, "rb"))

try:
    from google.colab import drive
    drive.mount("/content/drive")
    CKPT = "/content/drive/MyDrive/fgsan_ckpt"
except Exception:
    CKPT = "./fgsan_ckpt"

assert os.path.isdir(CKPT), f"no checkpoint at {CKPT} — run the SAVE cell first"
manifest = json.load(open(f"{CKPT}/manifest.json"))
meta = json.load(open(f"{CKPT}/meta.json"))
G = globals()

if meta.get("present_features"):
    G["present_features"] = meta["present_features"]
if meta.get("direction_mask_np") is not None:
    G["direction_mask_np"] = np.array(meta["direction_mask_np"], dtype=np.float32)

# group_indices: keep what your notebook cells already defined; else rebuild from meta
if "group_indices" not in G and meta.get("group_indices") is not None:
    gi = meta["group_indices"]
    if isinstance(gi, dict):
        G["group_indices"] = {k: np.array(v) for k, v in gi.items()}
    elif isinstance(gi, list) and gi and isinstance(gi[0], list):
        G["group_indices"] = [np.array(g) for g in gi]
    else:
        G["group_indices"] = gi
print("  group_indices:", "already defined" if "group_indices" in G else "MISSING",
      "->", type(G.get("group_indices")).__name__)

if os.path.exists(f"{CKPT}/scaler.joblib"):
    G["scaler"] = joblib.load(f"{CKPT}/scaler.joblib")

if "cfg" not in G and meta.get("cfg"):
    from types import SimpleNamespace
    G["cfg"] = SimpleNamespace(**meta["cfg"])
    print("  cfg restored from checkpoint (as SimpleNamespace)")

DEV = G.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

def _build(kwargs):
    return FeatureGroupedAttentionNet(            # noqa: F821
        G["group_indices"], emb_dim=cfg.emb_dim, n_heads=cfg.n_heads,
        dropout=cfg.dropout, use_attention=kwargs["use_attention"]).to(DEV)

restored_abl = {}
for name, kw in manifest["torch_models"].items():
    try:
        mdl = _build(kw)
        mdl.load_state_dict(torch.load(f"{CKPT}/{name}.pt", map_location=DEV))
        mdl.eval()
        if name == "main_model":
            G["main_model"] = mdl
        else:
            restored_abl[kw.get("label", name)] = mdl
    except NameError:
        raise RuntimeError("Run the FeatureGroupedAttentionNet class cell and the cfg "
                           "cell BEFORE this restore cell.")
if restored_abl:
    G["ablation_models"] = restored_abl

for nm in manifest["sklearn_models"]:
    G[nm] = joblib.load(f"{CKPT}/skl_{nm}.joblib")

for dname, keys in manifest.get("model_dicts", {}).items():
    d = {}
    for orig, safe in keys:
        d[orig] = joblib.load(f"{CKPT}/dict_{dname}__{safe}.joblib")
    G[dname] = d
    print(f"  restored dict '{dname}' with {len(d)} models:", list(d))

if os.path.exists(f"{CKPT}/arrays.npz"):
    z = np.load(f"{CKPT}/arrays.npz", allow_pickle=False)
    for nm in z.files:
        G[nm] = z[nm]
    if "y_atk" in G and "to_t" in G:
        G["y_atk_t"] = torch.tensor(G["y_atk"], dtype=torch.long, device=DEV)

print("RESTORED from", CKPT)
print("  main_model :", "main_model" in G)
print("  ablation   :", list(G.get("ablation_models", {})))
print("  baselines  :", manifest["sklearn_models"])
print("  arrays     :", manifest["arrays"])
print("You can skip the training cells and run the experiment cells directly.")

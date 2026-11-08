# ============ RESTORE main_model DIRECTLY (manifest lost its model list) ============
# Run after definition cells 1,2,3,4,6,8,9,10,11 and CHECKPOINT_2_restore.py.
# 1) lists what is on Drive, 2) loads main_model.pt directly, 3) checks it reproduces
# Table 4 (F1 = 0.9612, 20 FP, 108 FN), 4) rebuilds X_atk / X_atk_adv_np for R5 part D.
import os, glob, numpy as np, torch
from sklearn.metrics import f1_score, confusion_matrix
CKPT = "/content/drive/MyDrive/fgsan_ckpt"
DEV = DEVICE if "DEVICE" in globals() else ("cuda" if torch.cuda.is_available() else "cpu")

print("Model files on Drive:")
for f in sorted(glob.glob(f"{CKPT}/*.pt")):
    print(f"   {os.path.basename(f):28} {os.path.getsize(f)/1024:8.1f} KB")

path = f"{CKPT}/main_model.pt"
if os.path.exists(path):
    main_model = FeatureGroupedAttentionNet(group_indices, emb_dim=cfg.emb_dim, n_heads=cfg.n_heads,
                                            dropout=cfg.dropout, use_attention=True).to(DEV)
    main_model.load_state_dict(torch.load(path, map_location=DEV))
    main_model.eval()
    print("\nmain_model loaded from", path)
else:
    print("\nmain_model.pt NOT found -> retraining (deterministic, same seed as the paper, ~10-20 min)")
    set_seed(SEED)
    main_model = FeatureGroupedAttentionNet(group_indices, emb_dim=cfg.emb_dim, n_heads=cfg.n_heads,
                                            dropout=cfg.dropout, use_attention=True).to(DEV)
    main_model, _ = train_model(main_model, X_train, y_train, X_val, y_val, cfg, use_adv=True, use_focal=True)
    main_model.eval()
    torch.save(main_model.state_dict(), path); print("saved ->", path)

# sanity check against Table 4 of the paper
with torch.no_grad():
    p = torch.cat([torch.softmax(main_model(to_t(X_test[i:i+4096]))[0], 1)[:, 1].cpu()
                   for i in range(0, len(X_test), 4096)]).numpy()
pred = (p >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
print(f"check: F1={f1_score(y_test, pred):.4f}  FP={fp}  FN={fn}   (paper: F1=0.9612, FP=20, FN=108)")

# arrays needed by R5 part D
atk = np.asarray(y_test) == 1
X_atk, y_atk = np.asarray(X_test)[atk], np.asarray(y_test)[atk]
y_atk_t = torch.tensor(y_atk, dtype=torch.long, device=DEV)
X_atk_adv_np = pgd_attack(main_model, to_t(X_atk), y_atk_t, cfg.adv_epsilon, cfg.adv_alpha,
                          cfg.adv_steps).detach().cpu().numpy()
with torch.no_grad():
    pa = torch.softmax(main_model(to_t(X_atk_adv_np))[0], 1)[:, 1].cpu().numpy() >= 0.5
    pc = torch.softmax(main_model(to_t(X_atk))[0], 1)[:, 1].cpu().numpy() >= 0.5
print(f"check: white-box evasions {int((pc & ~pa).sum())} of {int(pc.sum())} clean-correct   (paper: 3 of 1,586)")
print("\nIf both checks match, run R5_fixable_limitations.py now.")

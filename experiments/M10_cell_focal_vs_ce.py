# ===================== CELL (NEW) — M10: focal-loss vs cross-entropy, at fixed =====================
# architecture / prevalence / adversarial settings. Trains the full FGSAN (+ adversarial training)
# under standard cross-entropy and a focal-loss gamma/alpha grid, varying ONLY the loss. Reports
# clean F1/precision/recall/PR-AUC and evasion success rate (ESR). Per-config results are saved so
# a disconnect resumes. Needs restored: X_train,y_train,X_val,y_val,X_test,y_test,X_atk,y_atk,
# group_indices, cfg, and defined: FeatureGroupedAttentionNet, train_model, pgd_attack,
# predict_torch, evasion_success_rate, to_t, set_seed.
import os, json, numpy as np, torch
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

try:
    from google.colab import drive; drive.mount("/content/drive")
    OUT="/content/drive/MyDrive/fgsan_ckpt"
except Exception:
    OUT="./fgsan_ckpt"
os.makedirs(OUT, exist_ok=True)
RES=f"{OUT}/m10_results.json"
R=json.load(open(RES)) if os.path.exists(RES) else {}
def save(): json.dump(R,open(RES,"w"),indent=1)

SEED=42
DEV=globals().get("DEVICE","cuda" if torch.cuda.is_available() else "cpu")
yatk_t=torch.tensor(np.asarray(y_atk),dtype=torch.long,device=DEV)
_g0,_a0=cfg.focal_gamma,cfg.focal_alpha          # remember paper defaults to restore at the end

# (label, use_focal, gamma, alpha)
CONFIGS=[
 ("CE (standard)",            False, None, None),
 ("Focal g=0 a=0.25",         True,  0.0, 0.25),
 ("Focal g=1 a=0.25",         True,  1.0, 0.25),
 ("Focal g=2 a=0.25 [paper]", True,  2.0, 0.25),
 ("Focal g=5 a=0.25",         True,  5.0, 0.25),
 ("Focal g=2 a=0.10",         True,  2.0, 0.10),
 ("Focal g=2 a=0.50",         True,  2.0, 0.50),
 ("Focal g=2 a=1.00",         True,  2.0, 1.00),
]

def run(label, use_focal, g, a):
    if label in R: return R[label]
    if use_focal: cfg.focal_gamma, cfg.focal_alpha = g, a
    set_seed(SEED)
    m=FeatureGroupedAttentionNet(group_indices, emb_dim=cfg.emb_dim, n_heads=cfg.n_heads,
                                 dropout=cfg.dropout, use_attention=True)
    m,_=train_model(m, X_train, y_train, X_val, y_val, cfg, use_adv=True, use_focal=use_focal)
    # clean metrics on the test set (attack = positive class)
    preds, probs = predict_torch(m, X_test)
    pr,rc,f1,_ = precision_recall_fscore_support(y_test, preds, average="binary", pos_label=1, zero_division=0)
    prauc = float(average_precision_score(y_test, probs))
    # robustness: transfer-free white-box PGD on the attack flows
    cp,_ = predict_torch(m, X_atk)
    adv  = pgd_attack(m, to_t(X_atk), yatk_t, cfg.adv_epsilon, cfg.adv_alpha, cfg.adv_steps)
    ap,_ = predict_torch(m, adv.cpu().numpy())
    esr  = float(evasion_success_rate(y_atk, np.asarray(cp), np.asarray(ap)))
    R[label]={"f1":float(f1),"precision":float(pr),"recall":float(rc),"pr_auc":prauc,"esr":esr,
              "use_focal":use_focal,"gamma":g,"alpha":a}; save()
    print(f"  {label:26} F1={f1:.4f} P={pr:.4f} R={rc:.4f} PR-AUC={prauc:.4f} ESR={esr:.4f}")
    return R[label]

print("M10: loss-function ablation (full FGSAN + adversarial training, fixed otherwise)")
for lab,uf,g,a in CONFIGS: run(lab,uf,g,a)
cfg.focal_gamma, cfg.focal_alpha = _g0,_a0        # restore paper defaults

print("\n=== M10 summary ===")
print(f"{'config':26} {'F1':>7} {'Prec':>7} {'Rec':>7} {'PR-AUC':>7} {'ESR':>7}")
for lab,_,_,_ in CONFIGS:
    r=R[lab]; print(f"{lab:26} {r['f1']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f} {r['pr_auc']:7.4f} {r['esr']:7.4f}")
ce=R["CE (standard)"]; fp=R["Focal g=2 a=0.25 [paper]"]
print(f"\nfocal(paper) - CE:  dF1={fp['f1']-ce['f1']:+.4f}  dPR-AUC={fp['pr_auc']-ce['pr_auc']:+.4f}  dESR={fp['esr']-ce['esr']:+.4f}")
print("Reading: if CE ~= focal, focal is NOT the driver (architecture+adv are); if focal >> CE, it")
print("matters and the gamma row justifies gamma=2. alpha here is a global scalar, so expect it near-flat.")
print("saved to", RES)

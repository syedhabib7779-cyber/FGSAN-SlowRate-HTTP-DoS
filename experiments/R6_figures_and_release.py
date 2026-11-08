# ============ R6 — publication figures (vector, grayscale-safe) + reproducibility release bundle ============
# Run AFTER: definition cells 1,2,3,4,6,8,9,10,11 -> CHECKPOINT_2_restore.py -> RESTORE_main_model_direct.py
# Part F (~5 min): regenerates the data figures that need the model/data, from the SAME code paths as the
#   original cells, with larger fonts, colour-blind-safe palette, hatching/line styles that survive grayscale,
#   and saves each as PDF + SVG + 600-dpi PNG to  MyDrive/fgsan_ckpt/figs_final/
#   Prints the underlying numbers so you can confirm they match the paper.
# Part R (~1 min): writes  MyDrive/fgsan_ckpt/FGSAN_release.zip  (environment, package versions, checkpoints,
#   scaler, split indices, result JSONs, all experiment cells you upload into /content) for GitHub/Zenodo.
import os, json, glob, shutil, platform, zipfile, numpy as np, torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, f1_score
CK = "/content/drive/MyDrive/fgsan_ckpt"; FD = f"{CK}/figs_final"; os.makedirs(FD, exist_ok=True)
SEED = 42
DEV = DEVICE if "DEVICE" in globals() else ("cuda" if torch.cuda.is_available() else "cpu")
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#CC79A7",
          sky="#56B4E9", yellow="#F0E442", black="#000000", grey="#7F7F7F")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10.5,
                     "axes.titleweight": "bold", "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":", "axes.axisbelow": True,
                     "savefig.dpi": 600, "pdf.fonttype": 42, "svg.fonttype": "none"})
def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{FD}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig); print("   saved", name, "(pdf/svg/png)")

Xte, yte = np.asarray(X_test, np.float32), np.asarray(y_test)
def proba(model, X, bs=4096):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.softmax(model(to_t(X[i:i + bs]))[0], 1)[:, 1].float().cpu().numpy())
    return np.concatenate(out)
p_main = proba(main_model, Xte); pred_main = (p_main >= .5).astype(int)

# ---------------- baselines exactly as in Cell 7 (SVM on the 20,000-row subsample) ----------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
if "BL" not in globals():
    BL = {"Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
          "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
                                   colsample_bytree=0.9, eval_metric="logloss", random_state=SEED, n_jobs=-1),
          "LightGBM": LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.1, random_state=SEED, n_jobs=-1, verbose=-1),
          "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=SEED),
          "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=SEED)}
    for nm, clf in BL.items():
        if nm == "SVM (RBF)":
            sub = np.random.default_rng(SEED).choice(len(X_train), 20000, replace=False)
            clf.fit(np.asarray(X_train)[sub], np.asarray(y_train)[sub])
        else:
            clf.fit(X_train, y_train)
        print(f"   {nm:14} F1={f1_score(yte, clf.predict(Xte)):.4f}")
BS = {nm: clf.predict_proba(Xte)[:, 1] for nm, clf in BL.items()}
LS = {"Random Forest": "--", "XGBoost": "-.", "LightGBM": ":", "SVM (RBF)": (0, (5, 1, 1, 1)), "MLP": (0, (3, 1, 1, 1, 1, 1))}
CL = {"Random Forest": OI["orange"], "XGBoost": OI["green"], "LightGBM": OI["purple"], "SVM (RBF)": OI["grey"], "MLP": OI["sky"]}

print("\n[F1] Figure S1 — exploratory data analysis")
d = df_task
fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.2))
cnt = d["y"].value_counts().sort_index()
b = ax[0, 0].bar(["Benign", "Slow-rate DoS"], cnt.values, color=[OI["blue"], OI["red"]], edgecolor="black", linewidth=.6)
b[1].set_hatch("///")
for r, v in zip(b, cnt.values): ax[0, 0].text(r.get_x() + r.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=8.5)
ax[0, 0].set_ylabel("Flows"); ax[0, 0].set_title("(a) Class balance (5:1)")
for lab, c, ls in [(0, OI["blue"], "-"), (1, OI["red"], "--")]:
    ax[0, 1].hist(np.log1p(d.loc[d.y == lab, "Flow IAT Mean"].clip(lower=0)), bins=50, density=True, histtype="step",
                  linewidth=1.4, color=c, linestyle=ls, label="Benign" if lab == 0 else "Slow-rate DoS")
ax[0, 1].set_xlabel("log(1 + Flow IAT Mean, µs)"); ax[0, 1].set_ylabel("Density"); ax[0, 1].legend()
ax[0, 1].set_title("(b) Flow IAT Mean by class")
corr = d[present_features].corr().values
im = ax[1, 0].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1); ax[1, 0].grid(False)
ax[1, 0].set_xticks([]); ax[1, 0].set_yticks([]); ax[1, 0].set_title("(c) Feature correlation (27 features)")
fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.03)
data = [np.log1p(d.loc[d.y == k, "Idle Mean"].clip(lower=0)) for k in (0, 1)]
bp = ax[1, 1].boxplot(data, patch_artist=True, widths=.5, showfliers=False); ax[1, 1].set_xticks([1, 2], ["Benign", "Slow-rate DoS"])
for patch, c, h in zip(bp["boxes"], [OI["blue"], OI["red"]], [None, "///"]):
    patch.set_facecolor(c); patch.set_alpha(.75); patch.set_hatch(h)
ax[1, 1].set_ylabel("log(1 + Idle Mean, µs)"); ax[1, 1].set_title("(d) Idle time by class")
fig.tight_layout(); save(fig, "Figure_S1_EDA")

print("\n[F2] Figure S2 — error analysis")
cm = confusion_matrix(yte, pred_main, labels=[0, 1]); print("   confusion matrix", cm.tolist(), "(paper: FP=20, FN=108)")
fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.4))
ax[0, 0].imshow(cm, cmap="Greys"); ax[0, 0].grid(False)
for i in range(2):
    for j in range(2):
        ax[0, 0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=11, color="white" if cm[i, j] > cm.max() / 2 else "black")
ax[0, 0].set_xticks([0, 1], ["Benign", "Attack"]); ax[0, 0].set_yticks([0, 1], ["Benign", "Attack"])
ax[0, 0].set_xlabel("Predicted"); ax[0, 0].set_ylabel("True"); ax[0, 0].set_title("(a) Confusion matrix, FGSAN")
f, t, _ = roc_curve(yte, p_main); ax[0, 1].plot(f, t, color=OI["blue"], lw=2, label=f"FGSAN ({roc_auc_score(yte, p_main):.4f})")
for nm, s in BS.items():
    f, t, _ = roc_curve(yte, s); ax[0, 1].plot(f, t, color=CL[nm], ls=LS[nm], lw=1.3, label=f"{nm} ({roc_auc_score(yte, s):.4f})")
ax[0, 1].set_xscale("log"); ax[0, 1].set_xlim(1e-4, 1); ax[0, 1].set_xlabel("False-positive rate (log)")
ax[0, 1].set_ylabel("True-positive rate"); ax[0, 1].legend(fontsize=7, title="ROC-AUC", title_fontsize=7.5)
ax[0, 1].set_title("(b) ROC curves")
best = max(BS, key=lambda k: f1_score(yte, (BS[k] >= .5).astype(int)))
for nm, s, c, ls in [("FGSAN", p_main, OI["blue"], "-"), (best, BS[best], CL[best], LS[best])]:
    pr, rc, _ = precision_recall_curve(yte, s); ax[1, 0].plot(rc, pr, color=c, ls=ls, lw=1.6, label=nm)
ax[1, 0].set_xlabel("Recall"); ax[1, 0].set_ylabel("Precision"); ax[1, 0].legend(); ax[1, 0].set_title("(c) Precision–recall")
idle = present_features.index("Idle Mean"); tp = (yte == 1) & (pred_main == 1); fn = (yte == 1) & (pred_main == 0)
Xr = np.asarray(X_test_raw)
for m_, c, ls, lab in [(tp, OI["blue"], "-", f"Detected (n={tp.sum():,})"), (fn, OI["red"], "--", f"Missed (n={fn.sum()})")]:
    ax[1, 1].hist(np.log1p(np.clip(Xr[m_, idle], 0, None)), bins=30, density=True, histtype="step", lw=1.5, color=c, ls=ls, label=lab)
ax[1, 1].set_xlabel("log(1 + Idle Mean, µs)"); ax[1, 1].set_ylabel("Density"); ax[1, 1].legend()
ax[1, 1].set_title("(d) Idle time of detected vs missed attacks")
fig.tight_layout(); save(fig, "Figure_S2_error_analysis")

print("\n[F3] Figure S5 — benign-slow stress test (identical procedure and seed to Cell 14)")
fi = {n: i for i, n in enumerate(present_features)}
IAT = [c for c in present_features if "IAT" in c or c == "Flow Duration"]; IDL = [c for c in present_features if "Idle" in c]
RAT = [c for c in present_features if "/s" in c]; LEN = [c for c in present_features if "Length" in c or "Packet Length" in c]
def stretch_row(row, a, b_, l):
    row = row.copy()
    for c in IAT: row[fi[c]] *= a
    for c in IDL: row[fi[c]] *= b_
    for c in RAT: row[fi[c]] /= a
    for c in LEN: row[fi[c]] *= l
    return row
ben = yte == 0; Xb = Xr[ben]; rng2 = np.random.default_rng(SEED); n = len(Xb)
ns = int(n * stress_cfg.benign_slow_fraction); sidx = rng2.choice(n, size=ns, replace=False)
smask = np.zeros(n, bool); smask[sidx] = True; Xs = Xb.copy()
for i in sidx:
    a = rng2.uniform(*stress_cfg.iat_stretch_range); b_ = rng2.uniform(*stress_cfg.idle_stretch_range); l = rng2.uniform(0.8, 1.3)
    Xs[i] = stretch_row(Xb[i], a, b_, l)
Xs_std = scaler.transform(Xs)
models = {"FGSAN": (proba(main_model, Xs_std) >= .5).astype(int)}
for nm, clf in BL.items(): models[nm] = clf.predict(Xs_std)
names, f_norm, f_str, fold = [], [], [], []
for nm, pr in models.items():
    a_, b2 = pr[~smask].mean(), pr[smask].mean()
    names.append(nm); f_norm.append(100 * a_); f_str.append(100 * b2); fold.append(b2 / a_ if a_ > 0 else np.nan)
    print(f"   {nm:14} FPR normal={100*a_:.3f}%  stressed={100*b2:.3f}%  fold={b2/a_ if a_>0 else float('nan'):.1f}")
print("   (paper: fold 1.4 FGSAN, 20.0 RF, 6.0 XGB, 12.0 LGBM and MLP, 1.3 SVM)")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0)); xx = np.arange(len(names))
ax[0].bar(xx - .2, f_norm, .4, color=OI["blue"], edgecolor="black", lw=.5, label="Normal benign")
ax[0].bar(xx + .2, f_str, .4, color=OI["orange"], edgecolor="black", lw=.5, hatch="///", label="Stressed benign")
ax[0].set_xticks(xx, names, rotation=30, ha="right"); ax[0].set_ylabel("False-positive rate (%)"); ax[0].legend()
ax[0].set_title(f"(a) FPR, normal (n={(~smask).sum():,}) vs stressed (n={smask.sum():,})")
ax[1].bar(xx, fold, color=[OI["blue"]] + [CL[k] for k in names[1:]], edgecolor="black", lw=.5)
for x_, v in zip(xx, fold): ax[1].text(x_, v, f"{v:.1f}×", ha="center", va="bottom", fontsize=8)
ax[1].set_xticks(xx, names, rotation=30, ha="right"); ax[1].set_ylabel("Fold increase in FPR")
ax[1].set_title("(b) Fold increase (small denominators)")
fig.tight_layout(); save(fig, "Figure_S5_stress_test")

print("\n[F4] Figure S6 — raw attention weights and Figure 6 — occlusion importance")
gnames = list(group_indices.keys())
@torch.no_grad()
def attn_imp(model, X, bs=512):
    out = np.empty((len(X), 3), np.float32)
    for s in range(0, len(X), bs):
        _, w = model(to_t(X[s:s + bs])); out[s:s + bs] = w.mean(dim=1).cpu().numpy()
    return out
A = attn_imp(main_model, Xte)
masks = {"TP": (yte == 1) & (pred_main == 1), "FN": (yte == 1) & (pred_main == 0),
         "TN": (yte == 0) & (pred_main == 0), "FP": (yte == 0) & (pred_main == 1)}
OC = [OI["blue"], OI["red"], OI["green"], OI["orange"]]; HT = [None, "///", "..", "xx"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0)); x = np.arange(3); wd = .2
for i, (k, m_) in enumerate(masks.items()):
    v = A[m_].mean(0); print(f"   attention {k} (n={m_.sum()}):", np.round(v, 3).tolist())
    ax[0].bar(x + (i - 1.5) * wd, v, wd, color=OC[i], hatch=HT[i], edgecolor="black", lw=.4, label=f"{k} (n={m_.sum():,})")
ax[0].set_xticks(x, gnames); ax[0].set_ylim(0.30, 0.36); ax[0].set_ylabel("Mean attention weight"); ax[0].legend(fontsize=7, ncol=2)
ax[0].set_title("(a) Attention by detection outcome")
for i, k in enumerate(["TP", "FN"]):
    ax[1].bar(x + (i - .5) * .35, A[masks[k]].mean(0), .35, color=OC[i], hatch=HT[i], edgecolor="black", lw=.4, label=k)
ax[1].set_xticks(x, gnames); ax[1].set_ylim(0.30, 0.36); ax[1].legend(); ax[1].set_title("(b) Detected (TP) vs missed (FN)")
fig.tight_layout(); save(fig, "Figure_S6_attention_weights")

def fwd_occ(model, x, zero=None):
    embs = []
    for g, idx in model.group_indices.items():
        e = model.encoders[g](x[:, idx]); embs.append(torch.zeros_like(e) if zero == g else e)
    tok = torch.stack(embs, 1); ao, _ = model.attn(tok, tok, tok)
    return model.classifier(model.norm(tok + ao).mean(1))
@torch.no_grad()
def occ(model, X, bs=512):
    model.eval(); eff = {g: np.empty(len(X), np.float32) for g in gnames}
    for s in range(0, len(X), bs):
        xb = to_t(X[s:s + bs]); pf = torch.softmax(fwd_occ(model, xb), 1)[:, 1]
        for g in gnames: eff[g][s:s + bs] = (pf - torch.softmax(fwd_occ(model, xb, g), 1)[:, 1]).abs().cpu().numpy()
    return eff
O = occ(main_model, Xte)
ov = [O[g].mean() for g in gnames]; print("   occlusion overall:", dict(zip(gnames, np.round(ov, 3))), "(paper: volume 0.114, timing 0.081, behavior 0.025)")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
bb = ax[0].bar(gnames, ov, color=[OI["blue"], OI["orange"], OI["green"]], edgecolor="black", lw=.5)
for r_, h in zip(bb, [None, "///", ".."]): r_.set_hatch(h)
ax[0].set_ylabel("Mean |ΔP(attack)|"); ax[0].set_title(f"(a) Overall (n={len(yte):,} test flows)")
for i, (k, m_) in enumerate(masks.items()):
    v = [O[g][m_].mean() for g in gnames]; print(f"   occlusion {k}:", np.round(v, 3).tolist())
    ax[1].bar(x + (i - 1.5) * wd, v, wd, color=OC[i], hatch=HT[i], edgecolor="black", lw=.4, label=f"{k} (n={m_.sum():,})")
ax[1].set_xticks(x, gnames); ax[1].set_ylabel("Mean |ΔP(attack)|"); ax[1].legend(fontsize=7, ncol=2)
ax[1].set_title("(b) By detection outcome")
fig.tight_layout(); save(fig, "Figure_6_occlusion_importance")

# ---------------- Part R: reproducibility release bundle ----------------
print("\n[R] Building release bundle")
REL = "/content/FGSAN_release"; shutil.rmtree(REL, ignore_errors=True)
for sub in ("checkpoints", "results", "code", "splits"): os.makedirs(f"{REL}/{sub}", exist_ok=True)
import importlib
pk = {}
for name in ["numpy", "pandas", "scikit-learn", "torch", "xgboost", "lightgbm", "scipy", "statsmodels", "matplotlib", "seaborn", "joblib"]:
    try:
        from importlib.metadata import version; pk[name] = version(name)
    except Exception: pass
open(f"{REL}/requirements.txt", "w").write("\n".join(f"{k}=={v}" for k, v in pk.items()) + "\n")
env = dict(python=platform.python_version(), platform=platform.platform(), processor=platform.processor(),
           cpu_count=os.cpu_count(), torch=torch.__version__, cuda=torch.version.cuda,
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, seed=SEED,
           features_in_input_order=list(present_features), groups={k: [present_features[i] for i in v] for k, v in group_indices.items()},
           config={k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_") and not callable(getattr(cfg, k))})
json.dump(env, open(f"{REL}/environment.json", "w"), indent=1, default=str)
for f in glob.glob(f"{CK}/*.pt") + glob.glob(f"{CK}/scaler.joblib"): shutil.copy(f, f"{REL}/checkpoints/")
for f in glob.glob(f"{CK}/*.json"): shutil.copy(f, f"{REL}/results/")
for f in glob.glob(f"{CK}/reproducibility_package/*") + glob.glob("/content/reproducibility_package/*"):
    if os.path.isfile(f): shutil.copy(f, f"{REL}/splits/")
for f in glob.glob("/content/*.py") + glob.glob("/content/*.ipynb"): shutil.copy(f, f"{REL}/code/")
open(f"{REL}/README.md", "w").write(
    "# FGSAN: Feature-Grouped Self-Attention with Adversarial Training for Slow-Rate HTTP DoS Detection\n\n"
    "Data: CICIDS2017 Wednesday (Canadian Institute for Cybersecurity); not redistributed.\n"
    "- environment.json: Python/PyTorch/CUDA/GPU, seed, feature order, groups, training config\n"
    "- requirements.txt: exact package versions\n- checkpoints/: trained FGSAN (main_model.pt), ablation models, scaler\n"
    "- splits/: split indices and attack masks (seed 42) with MD5 checksums\n- results/: JSON outputs behind every table\n"
    "- code/: Colab notebook and experiment cells (run order in the notebook header)\n")
zp = f"{CK}/FGSAN_release.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(REL):
        for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), REL))
print("   environment:", {k: env[k] for k in ("python", "torch", "cuda", "gpu")})
print("   packages:", pk)
print("   release bundle ->", zp, f"({os.path.getsize(zp)/1e6:.1f} MB)")
print("   splits copied:", len(os.listdir(f'{REL}/splits')), "| code files copied:", len(os.listdir(f'{REL}/code')))
print("\nDone. Upload the notebook + .py cells into /content first (Files pane) if you want them inside the bundle.")

# ===================== CELL 30 (NEW, self-contained) — M3: fair robustness comparison =====================
# Gives every baseline a robustness mechanism, then attacks ALL models with ONE
# comparable gradient-free attack (SPSA) at the same epsilon, so architecture is
# no longer confounded with training method. Builds the clean/robust Pareto data.
#
# SELF-CONTAINED: it rebuilds the five baselines from the paper's exact Cell-7
# configs (so their clean F1 matches Table 7) and does not depend on any
# 'baselines' object in memory. Needs only: X_train,y_train,X_test,y_test,
# present_features, direction_mask_np, cfg, main_model, to_t, DEVICE.
#
# RUNTIME: the long one (SPSA + SVM). Saves per model to Drive and SKIPS finished
# ones, so a disconnect costs at most one model. Re-run to resume.

import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from google.colab import drive; drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/fgsan_ckpt"
except Exception:
    OUT = "./fgsan_ckpt"
os.makedirs(OUT, exist_ok=True)
RES = f"{OUT}/m3_results.json"
results = json.load(open(RES)) if os.path.exists(RES) else {}
def save(): json.dump(results, open(RES, "w"), indent=1)

SEED  = 42
EPS   = cfg.adv_epsilon
N_ATK = 300
SP_STEPS, SP_BATCH = 30, 16
SVM_MAX_TRAIN = int(globals().get("SVM_MAX_TRAIN", 20000))
dmask = np.asarray(direction_mask_np, dtype=np.float64); inc = dmask == 1.0

# ---------- the paper's five baselines (Cell 7 configs, verbatim) ----------
def make_baselines():
    return {
     "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
     "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                              subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                              random_state=SEED, n_jobs=-1),
     "LightGBM": LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.1,
                                random_state=SEED, n_jobs=-1, verbose=-1),
     "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=SEED),
     "MLP (sklearn)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=SEED),
    }
def fit_baseline(name, clf, Xt, yt):
    if name == "SVM (RBF)" and len(Xt) > SVM_MAX_TRAIN:
        idx = np.random.default_rng(SEED).choice(len(Xt), SVM_MAX_TRAIN, replace=False)
        clf.fit(Xt[idx], yt[idx])
    else:
        clf.fit(Xt, yt)
    return clf

# ---------- one comparable attack for every model: numpy SPSA on P(class 1) ----------
def spsa(prob1, X0, y0, eps, steps=SP_STEPS, batch=SP_BATCH, delta=0.01, seed=0):
    g = np.random.default_rng(seed); lr = eps/8; x = X0.copy()
    def loss(Xq):
        p1 = np.clip(prob1(Xq), 1e-6, 1-1e-6)
        pc = np.where(y0 == 1, p1, 1-p1)
        return -np.log(pc)
    for _ in range(steps):
        ge = np.zeros_like(x)
        for _b in range(batch):
            v = g.integers(0, 2, x.shape)*2 - 1.0
            ge += ((loss(np.clip(x+delta*v, X0-eps, X0+eps)) -
                    loss(np.clip(x-delta*v, X0-eps, X0+eps)))/(2*delta))[:, None]*v
        step = lr*np.sign(ge/batch); step[:, inc] = np.clip(step[:, inc], 0, None)
        x = np.clip(x+step, X0-eps, X0+eps)
    return x
def sk_prob1(m):
    if hasattr(m, "predict_proba"):
        return lambda Xq: m.predict_proba(Xq)[:, 1]
    return lambda Xq: 1/(1+np.exp(-m.decision_function(Xq)))
def fgsan_prob1(Xq):
    main_model.eval()
    with torch.no_grad():
        return torch.softmax(main_model(to_t(Xq))[0], 1)[:, 1].cpu().numpy()
def evasion(prob1, X0, y0, eps, seed=0):
    ok = (prob1(X0) > 0.5).astype(int) == y0
    if ok.sum() == 0: return float("nan")
    Xadv = spsa(prob1, X0, y0, eps, seed=seed)
    return float((((prob1(Xadv) > 0.5).astype(int) != y0) & ok).sum()/ok.sum())

atk = y_test == 1
Xa = X_test[atk][:N_ATK].astype(np.float64); ya = y_test[atk][:N_ATK]

# ---------- adversarial augmentation set (surrogate-crafted) ----------
_aug = {}
def get_aug():
    if "X" not in _aug:
        idx = np.random.default_rng(1).choice(len(X_train), min(5000, len(X_train)), replace=False)
        Xs, ys = X_train[idx].astype(np.float64), y_train[idx]
        surro = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=1).fit(X_train, y_train)
        adv = spsa(sk_prob1(surro), Xs, ys, EPS, steps=15, batch=8, seed=1)
        _aug["X"] = np.vstack([X_train, adv]); _aug["y"] = np.concatenate([y_train, ys])
    return _aug["X"], _aug["y"]

# ---------- proper PGD adversarially-trained MLP (torch) ----------
class TorchMLP(nn.Module):
    def __init__(s, d):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d,128), nn.ReLU(),
                                                  nn.Dropout(.3), nn.Linear(128,2))
    def forward(s, x): return s.net(x), None
def pgd_t(model, x, y, eps, alpha, steps):
    mask = torch.tensor(dmask, dtype=torch.float32, device=DEVICE); model.eval()
    xa = x.clone().detach()
    for _ in range(steps):
        xa.requires_grad_(True); loss = F.cross_entropy(model(xa)[0], y)
        g, = torch.autograd.grad(loss, [xa])
        with torch.no_grad():
            st = alpha*g.sign()
            st = torch.where((mask==1).unsqueeze(0).expand_as(st), torch.clamp(st,min=0), st)
            xa = torch.clamp(xa+st, x-eps, x+eps).detach()
    return xa
def train_adv_mlp(seed=0):
    torch.manual_seed(seed); m = TorchMLP(X_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xt = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    for ep in range(30):
        m.train(); perm = torch.randperm(len(yt))
        for b in range(0, len(yt), 256):
            bi = perm[b:b+256]; xb, yb = Xt[bi], yt[bi]; k = len(bi)//2
            if k > 0:
                xadv = pgd_t(m, xb[:k], yb[:k], EPS, cfg.adv_alpha, cfg.adv_steps)
                m.train(); xb = torch.cat([xadv, xb[k:]], 0)
            opt.zero_grad(); F.cross_entropy(m(xb)[0], yb).backward(); opt.step()
    m.eval(); return m
def tmlp_prob1(m):
    def f(Xq):
        with torch.no_grad():
            return torch.softmax(m(to_t(Xq))[0], 1)[:, 1].cpu().numpy()
    return f

def record(key, label, f1, ev):
    results[key] = {"label": label, "clean_f1": float(f1), "evasion": float(ev)}
    save(); print(f"  {label:30} cleanF1={f1:.4f}  evasion@eps={ev:.4f}")

NAMES = list(make_baselines().keys())

# ---- FGSAN reference ----
if "FGSAN" not in results:
    pr = (fgsan_prob1(X_test) > 0.5).astype(int)
    record("FGSAN", "FGSAN (adv-trained)", f1_score(y_test, pr),
           evasion(fgsan_prob1, Xa, ya, EPS, seed=7))

# ---- five baselines: clean, then adversarially augmented ----
for name in NAMES:
    if f"{name}__clean" not in results:
        clf = fit_baseline(name, make_baselines()[name], X_train, y_train)
        record(f"{name}__clean", f"{name} (clean)", f1_score(y_test, clf.predict(X_test)),
               evasion(sk_prob1(clf), Xa, ya, EPS, seed=7))
    if f"{name}__adv" not in results:
        Xaug, yaug = get_aug()
        clf = fit_baseline(name, make_baselines()[name], Xaug, yaug)
        record(f"{name}__adv", f"{name} (adv-augmented)", f1_score(y_test, clf.predict(X_test)),
               evasion(sk_prob1(clf), Xa, ya, EPS, seed=7))

# ---- proper adversarially-trained MLP ----
if "advMLP" not in results:
    amlp = train_adv_mlp(seed=0); pr = (tmlp_prob1(amlp)(X_test) > 0.5).astype(int)
    record("advMLP", "MLP (PGD adv-trained)", f1_score(y_test, pr),
           evasion(tmlp_prob1(amlp), Xa, ya, EPS, seed=7))

print("\n=== M3 Pareto data (clean F1 vs evasion @ eps=0.05) ===")
want = {"FGSAN","advMLP"} | {f"{n}__clean" for n in NAMES} | {f"{n}__adv" for n in NAMES}
for k, v in sorted(((k,v) for k,v in results.items() if k in want),
                   key=lambda kv: -kv[1]["clean_f1"]):
    print(f"  {v['label']:30} F1={v['clean_f1']:.4f}  evasion={v['evasion']:.4f}")
print("saved to", RES)

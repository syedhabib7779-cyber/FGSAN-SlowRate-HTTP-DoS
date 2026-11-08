# ===================== R5 — close the limitations that can be closed in Colab =====================
# Run AFTER: definition cells 1,2,3,4,6,8,9,10,11  +  CHECKPOINT_2_restore.py
# Each part saves its own JSON to Drive and is SKIPPED if already done (safe to re-run after a disconnect).
#   A  near-duplicate analysis (nearest-neighbour distance test->train) ............ ~2 min
#   B  probability calibration under prevalence shift (Brier, ECE, prior correction)  ~1 min
#   C  attack-generation and training cost: wall-clock + peak GPU memory ............ ~3 min
#   D  paired bootstrap of transfer-ESR differences (FGSAN vs each baseline) ......... ~5 min
#   E  near-duplicate-GROUPED re-split + retrain FGSAN and XGBoost .................. ~15-25 min
# Paste the printed output back to Claude.
import os, json, time, copy, numpy as np, torch, torch.nn.functional as F
import torch.nn as nn   # re-bind in case an earlier run overwrote it
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
try:
    from google.colab import drive; drive.mount("/content/drive"); OUT = "/content/drive/MyDrive/fgsan_ckpt"
except Exception:
    OUT = "./fgsan_ckpt"
os.makedirs(OUT, exist_ok=True)
RES = f"{OUT}/r5_results.json"
R = json.load(open(RES)) if os.path.exists(RES) else {}
def save(): json.dump(R, open(RES, "w"), indent=1)
SEED = 42
DEV = DEVICE if "DEVICE" in globals() else ("cuda" if torch.cuda.is_available() else "cpu")
Xtr, ytr = np.asarray(X_train, np.float32), np.asarray(y_train)
Xva, yva = np.asarray(X_val, np.float32), np.asarray(y_val)
Xte, yte = np.asarray(X_test, np.float32), np.asarray(y_test)

def proba(model, X, bs=4096):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.softmax(model(to_t(X[i:i + bs]))[0], 1)[:, 1].float().cpu().numpy())
    return np.concatenate(out)

p_te = proba(main_model, Xte); p_va = proba(main_model, Xva)

# ---------------------------------------------------------------- A  near-duplicates
if "A" not in R:
    print("\n[A] Near-duplicate analysis (Euclidean distance in standardized space, test -> nearest training flow)")
    nnb = NearestNeighbors(n_neighbors=1).fit(Xtr)
    d, idx = nnb.kneighbors(Xte); d = d[:, 0]
    same_lab = ytr[idx[:, 0]] == yte
    pred = (p_te >= 0.5).astype(int)
    rows = []
    for tau in [0.0, 0.01, 0.05, 0.10, 0.25, 0.50]:
        keep = d > tau
        ya, pa = yte[keep], pred[keep]
        r = dict(tau=tau, n=int(keep.sum()), n_attack=int(ya.sum()), n_benign=int((ya == 0).sum()),
                 f1=float(f1_score(ya, pa)), recall=float(recall_score(ya, pa)),
                 fpr=float(((pa == 1) & (ya == 0)).sum() / max((ya == 0).sum(), 1)),
                 auc=float(roc_auc_score(ya, p_te[keep])))
        rows.append(r)
        print(f"  test flows with NN-distance > {tau:<4}: n={r['n']:5d} (attack {r['n_attack']:4d})  "
              f"F1={r['f1']:.4f} recall={r['recall']:.4f} FPR={r['fpr']:.4f} AUC={r['auc']:.4f}")
    q = np.percentile(d, [5, 25, 50, 75, 95])
    print(f"  NN-distance percentiles (5/25/50/75/95): {np.round(q, 4).tolist()}")
    print(f"  share of test flows with an exact duplicate in training: {np.mean(d == 0):.4f}"
          f"  (attack {np.mean(d[yte == 1] == 0):.4f}, benign {np.mean(d[yte == 0] == 0):.4f})")
    R["A"] = dict(rows=rows, pct=q.tolist(), exact_share=float(np.mean(d == 0)),
                  exact_share_attack=float(np.mean(d[yte == 1] == 0)), exact_share_benign=float(np.mean(d[yte == 0] == 0)))
    save()

# ---------------------------------------------------------------- B  calibration under prevalence shift
if "B" not in R:
    print("\n[B] Calibration under prevalence shift (test reweighted to each prevalence)")
    from sklearn.isotonic import IsotonicRegression
    pi_tr = float(ytr.mean())
    def prior_shift(p, pi_new, pi_old=pi_tr):
        odds = p / np.clip(1 - p, 1e-9, None) * (pi_new / (1 - pi_new)) / (pi_old / (1 - pi_old))
        return odds / (1 + odds)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, yva)          # fitted on VALIDATION only
    def wmetrics(p, y, w, nb=15):
        brier = float(np.sum(w * (p - y) ** 2) / w.sum())
        bins = np.minimum((p * nb).astype(int), nb - 1); ece = 0.0
        for b in range(nb):
            m = bins == b
            if m.any(): ece += w[m].sum() / w.sum() * abs(np.average(y[m], weights=w[m]) - np.average(p[m], weights=w[m]))
        return brier, float(ece)
    out = {}
    for pi in [float(yte.mean()), 0.0299, 0.01, 0.001]:
        w = np.where(yte == 1, pi / yte.mean(), (1 - pi) / (1 - yte.mean()))
        res = {}
        for name, p in [("raw", p_te), ("isotonic(val)", iso.predict(p_te)),
                        ("raw+prior-shift", prior_shift(p_te, pi)), ("isotonic+prior-shift", prior_shift(np.clip(iso.predict(p_te), 1e-6, 1 - 1e-6), pi))]:
            br, ece = wmetrics(p, yte, w)
            pr = (p >= 0.5)
            tp = np.sum(w * (pr & (yte == 1))); fp = np.sum(w * (pr & (yte == 0))); fn = np.sum(w * (~pr & (yte == 1)))
            res[name] = dict(brier=br, ece=ece, precision=float(tp / max(tp + fp, 1e-12)), recall=float(tp / max(tp + fn, 1e-12)))
            print(f"  prevalence {pi:7.4f}  {name:22} Brier={br:.5f} ECE={ece:.5f}  "
                  f"@0.5: precision={res[name]['precision']:.3f} recall={res[name]['recall']:.3f}")
        out[str(pi)] = res
    R["B"] = out; save()

# ---------------------------------------------------------------- C  attack-generation / training cost
if "C" not in R:
    print("\n[C] Attack-generation and training cost")
    atk = yte == 1; Xa = to_t(Xte[atk]); ya_t = torch.tensor(yte[atk], dtype=torch.long, device=DEV)
    res = {}
    def timed(fn, reps=5):
        if DEV == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps): fn()
        if DEV == "cuda": torch.cuda.synchronize()
        return (time.perf_counter() - t) / reps
    if DEV == "cuda": torch.cuda.reset_peak_memory_stats()
    s = timed(lambda: pgd_attack(main_model, Xa, ya_t, cfg.adv_epsilon, cfg.adv_alpha, cfg.adv_steps))
    res["gpu_pgd_1694_s"] = s; res["gpu_pgd_ms_per_flow"] = 1000 * s / len(ya_t)
    if DEV == "cuda": res["gpu_peak_mem_MB_pgd"] = torch.cuda.max_memory_allocated() / 2**20
    m_cpu = copy.deepcopy(main_model).to("cpu"); torch.set_num_threads(1)
    xs = torch.tensor(Xte[atk][:256], dtype=torch.float32); ys = torch.tensor(yte[atk][:256], dtype=torch.long)
    def cpu_pgd():
        x0 = xs.clone(); xa = xs.clone()
        mask = torch.tensor(direction_mask_np, dtype=torch.float32) == 1.0
        for _ in range(cfg.adv_steps):
            xa.requires_grad_(True); g, = torch.autograd.grad(F.cross_entropy(m_cpu(xa)[0], ys), [xa])
            with torch.no_grad():
                st = cfg.adv_alpha * g.sign(); st = torch.where(mask.expand_as(st), st.clamp(min=0), st)
                xa = torch.clamp(xa + st, x0 - cfg.adv_epsilon, x0 + cfg.adv_epsilon)
        return xa
    m_cpu.eval(); s = timed(cpu_pgd, reps=3); res["cpu1_pgd_ms_per_flow"] = 1000 * s / 256
    torch.set_num_threads(os.cpu_count() or 1)
    # one training epoch, standard vs adversarial (half of each batch replaced), same model copy
    def epoch(adv):
        m = copy.deepcopy(main_model).to(DEV); m.train(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        perm = np.random.default_rng(SEED).permutation(len(Xtr)); bs = cfg.batch_size
        if DEV == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; xb = to_t(Xtr[b]); yb = torch.tensor(ytr[b], dtype=torch.long, device=DEV)
            if adv:
                k = int(cfg.adv_ratio * len(b))
                xb = torch.cat([pgd_attack(m, xb[:k], yb[:k], cfg.adv_epsilon, cfg.adv_alpha, cfg.adv_steps, training_mode=True), xb[k:]])
            loss = F.cross_entropy(m(xb)[0], yb); opt.zero_grad(); loss.backward(); opt.step()
        if DEV == "cuda": torch.cuda.synchronize()
        return time.perf_counter() - t, (torch.cuda.max_memory_allocated() / 2**20 if DEV == "cuda" else None)
    res["epoch_standard_s"], res["epoch_standard_peakMB"] = epoch(False)
    res["epoch_adversarial_s"], res["epoch_adversarial_peakMB"] = epoch(True)
    res["adv_over_standard"] = res["epoch_adversarial_s"] / res["epoch_standard_s"]
    res["est_30_epoch_adv_training_min"] = 30 * res["epoch_adversarial_s"] / 60
    for k, v in res.items(): print(f"  {k:34} {v if v is None else round(v, 4)}")
    R["C"] = res; save()

# ---------------------------------------------------------------- D  paired bootstrap of transfer ESR
if "D" not in R:
    print("\n[D] Paired bootstrap of transfer-ESR differences (same resampled attack flows for all models)")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    defs = {"Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
                                     colsample_bytree=0.9, eval_metric="logloss", random_state=SEED, n_jobs=-1),
            "LightGBM": LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.1, random_state=SEED, n_jobs=-1, verbose=-1),
            "SVM (RBF, full)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED),
            "MLP (sklearn)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=SEED)}
    Xa = np.asarray(X_atk); Xadv = np.asarray(X_atk_adv_np); n = len(Xa)
    P = {"FGSAN": ((proba(main_model, Xa) >= .5).astype(int), (proba(main_model, Xadv) >= .5).astype(int))}
    for nm, m in defs.items():
        print("  training", nm); m.fit(Xtr, ytr); P[nm] = (m.predict(Xa), m.predict(Xadv))
    rng = np.random.default_rng(SEED); B = 2000
    def esr(c, a): cc = c == 1; return ((cc & (a == 0)).sum() / max(cc.sum(), 1))
    boots = {k: np.empty(B) for k in P}
    for b in range(B):
        i = rng.integers(0, n, n)
        for k, (c, a) in P.items(): boots[k][b] = esr(c[i], a[i])
    out = {}
    names = list(P)
    for k in names[1:]:
        dlt = boots[k] - boots["FGSAN"]
        out[f"{k} - FGSAN"] = [float(esr(*P[k]) - esr(*P["FGSAN"])), float(np.percentile(dlt, 2.5)), float(np.percentile(dlt, 97.5))]
    for a, b2 in [("XGBoost", "LightGBM"), ("LightGBM", "Random Forest"), ("XGBoost", "Random Forest")]:
        dlt = boots[a] - boots[b2]
        out[f"{a} - {b2}"] = [float(esr(*P[a]) - esr(*P[b2])), float(np.percentile(dlt, 2.5)), float(np.percentile(dlt, 97.5))]
    for k, (v, lo, hi) in out.items(): print(f"  {k:30} diff={v:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    R["D"] = out; save()

# ---------------------------------------------------------------- E  near-duplicate-grouped re-split + retrain
RUN_E = True
if RUN_E and "E" not in R:
    print("\n[E] Near-duplicate-grouped split: flows identical after rounding every standardized feature to 0.01 SD form one group;")
    print("    groups never cross partitions. Retrains FGSAN (30 epochs, adversarial) and XGBoost on this split.")
    from sklearn.model_selection import GroupShuffleSplit
    from xgboost import XGBClassifier
    Xall = np.vstack([Xtr, Xva, Xte]); yall = np.concatenate([ytr, yva, yte])
    keys = np.round(Xall, 2); _, grp = np.unique(keys, axis=0, return_inverse=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    itr, itmp = next(gss.split(Xall, yall, grp))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    iva_r, ite_r = next(gss2.split(Xall[itmp], yall[itmp], grp[itmp])); iva, ite = itmp[iva_r], itmp[ite_r]
    assert not (set(grp[itr]) & set(grp[ite])), "group leakage"
    print(f"  groups: {len(np.unique(grp))} for {len(yall)} flows; split {len(itr)}/{len(iva)}/{len(ite)}; "
          f"test attacks {int(yall[ite].sum())}")
    set_seed(SEED)
    m = FeatureGroupedAttentionNet(group_indices, emb_dim=cfg.emb_dim, n_heads=cfg.n_heads,
                                   dropout=cfg.dropout, use_attention=True).to(DEV)
    t0 = time.perf_counter()
    m, _ = train_model(m, Xall[itr], yall[itr], Xall[iva], yall[iva], cfg, use_adv=True, use_focal=True)
    tmin = (time.perf_counter() - t0) / 60
    p = proba(m, Xall[ite]); pr = (p >= .5).astype(int); yt = yall[ite]
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                        eval_metric="logloss", random_state=SEED, n_jobs=-1).fit(Xall[itr], yall[itr])
    px = xgb.predict_proba(Xall[ite])[:, 1]
    # white-box evasion of the retrained FGSAN on the grouped test split (same attack as the paper)
    ka = yt == 1; xa = to_t(Xall[ite][ka]); ya_ = torch.tensor(yt[ka], dtype=torch.long, device=DEV)
    xadv = pgd_attack(m, xa, ya_, cfg.adv_epsilon, cfg.adv_alpha, cfg.adv_steps).detach().cpu().numpy()
    cc = proba(m, Xall[ite][ka]) >= .5; ev = cc & (proba(m, xadv) < .5)
    pxa = xgb.predict(xadv); ccx = xgb.predict(Xall[ite][ka]) == 1
    res_esr = dict(fgsan_esr=float(ev.sum() / max(cc.sum(), 1)), fgsan_esr_num=int(ev.sum()), fgsan_esr_den=int(cc.sum()),
                   xgb_transfer_esr=float((ccx & (pxa == 0)).sum() / max(ccx.sum(), 1)))
    res = dict(n_groups=int(len(np.unique(grp))), n_train=len(itr), n_val=len(iva), n_test=len(ite), **res_esr,
               fgsan_f1=float(f1_score(yt, pr)), fgsan_auc=float(roc_auc_score(yt, p)),
               fgsan_fpr=float(((pr == 1) & (yt == 0)).sum() / (yt == 0).sum()), fgsan_fnr=float(((pr == 0) & (yt == 1)).sum() / (yt == 1).sum()),
               xgb_f1=float(f1_score(yt, (px >= .5).astype(int))), xgb_auc=float(roc_auc_score(yt, px)), train_minutes=tmin)
    for k, v in res.items(): print(f"  {k:12} {v}")
    R["E"] = res; save()

print("\nall parts saved ->", RES)

# ======================= CELL 27 (REPLACES CELLS 24, 25, 26) =======================
# M1, final experiment. Supersedes Cells 24-26: same purpose, correct constraints.
#
# The constraint set is now 12 algebraic relations PLUS four timestamp-
# realizability conditions, every one of which was verified to hold for
# 100% of unmodified CICIDS2017 attack flows:
#   A  Flow IAT Mean = Duration / (n_total - 1)      [pins Flow IAT Mean]
#   B  each IAT sd within the bound attainable from its own [min, max, mean]
#   C  Idle Mean is 0 or at least the 5 s activity timeout
#   E  a direction with fewer than two packets reports no IAT
#
# Condition D (active + idle vs duration) is measured but NOT enforced:
# CICFlowMeter's endActiveIdleTime appends an idle value lying outside the
# flow duration when a flow is not closed by a FIN flag, so the quantities
# genuinely fail to reconcile for 10.63% of unmodified flows.
#
# Because A pins Flow IAT Mean, it leaves the attacker's control: 20 levers,
# 7 derived features.

import numpy as np, torch


import numpy as np

ACTIVITY_TIMEOUT = 5_000_000.0
FREE_V3 = [n for n in FREE if n != "Flow IAT Mean"]        # 20 levers
DERIVED_V3 = DERIVED + ["Flow IAT Mean"]                   # 7 derived


def _sd_cap(mn, mu, mx, k):
    """Largest sample sd attainable by k values in [mn,mx] with mean mu."""
    pop = np.maximum((mu - mn) * (mx - mu), 0.0)
    infl = np.where(k >= 2, k / np.maximum(k - 1.0, 1.0), 1.0)
    return np.sqrt(pop * infl)


def _enforce(r, rc, f, increase_only):
    for n in FREE_V3:
        r[:, f[n]] = np.maximum(r[:, f[n]], 0.0)
    for n in increase_only:
        if n in f:
            r[:, f[n]] = np.maximum(r[:, f[n]], rc[:, f[n]])
    for n in INTEGER_VALUED:
        r[:, f[n]] = np.round(r[:, f[n]])

    fp = r[:, f["Total Fwd Packets"]]
    bp = r[:, f["Total Backward Packets"]]
    # E: a flow needs two packets; a direction needs two to report an IAT
    short = (fp + bp) < 2
    if short.any():
        r[short, f["Total Fwd Packets"]] = np.maximum(fp[short], 2.0)
        fp = r[:, f["Total Fwd Packets"]]
    for fam, cnt in [("Fwd IAT", fp), ("Bwd IAT", bp)]:
        z = cnt < 2
        if z.any():
            for s in ["Mean", "Std", "Max", "Min"]:
                r[z, f[f"{fam} {s}"]] = 0.0

    # per-direction order statistics
    for lo, mid, hi in ORDERINGS:
        if lo.startswith("Flow"):
            continue
        r[:, f[lo]] = np.minimum(r[:, f[lo]], r[:, f[mid]])
        r[:, f[hi]] = np.maximum(r[:, f[hi]], r[:, f[mid]])

    # A: Flow IAT Mean follows from duration and the packet count
    ntot = fp + bp
    r[:, f["Flow IAT Mean"]] = np.where(
        ntot > 1, r[:, f["Flow Duration"]] / np.maximum(ntot - 1.0, 1.0), 0.0)
    fm = r[:, f["Flow IAT Mean"]]
    r[:, f["Flow IAT Min"]] = np.minimum(r[:, f["Flow IAT Min"]], fm)
    r[:, f["Flow IAT Max"]] = np.maximum(r[:, f["Flow IAT Max"]], fm)

    # B: cap each sd at what [min,max] and the mean actually permit
    for fam, cnt in [("Flow IAT", ntot), ("Fwd IAT", fp), ("Bwd IAT", bp)]:
        k = np.maximum(cnt - 1.0, 0.0)
        cap = _sd_cap(r[:, f[f"{fam} Min"]], r[:, f[f"{fam} Mean"]],
                      r[:, f[f"{fam} Max"]], k)
        g = k >= 2
        r[g, f[f"{fam} Std"]] = np.minimum(r[g, f[f"{fam} Std"]], cap[g])
        r[~g, f[f"{fam} Std"]] = 0.0

    # C: an idle period is either absent or longer than the activity timeout
    idle = r[:, f["Idle Mean"]]
    mid = (idle > 0) & (idle < ACTIVITY_TIMEOUT)
    if mid.any():
        r[mid, f["Idle Mean"]] = np.where(
            idle[mid] < ACTIVITY_TIMEOUT / 2.0, 0.0, ACTIVITY_TIMEOUT)
        z = r[:, f["Idle Mean"]] <= 0
        r[z, f["Idle Std"]] = 0.0
    return r


def project_v3(raw, raw_clean, features, increase_only=(), budget_raw=None):
    f = _idx(features)
    r = np.asarray(raw, dtype=np.float64).copy()
    rc = np.asarray(raw_clean, dtype=np.float64)
    r = _enforce(r, rc, f, increase_only)
    if budget_raw is not None:
        b = np.asarray(budget_raw, dtype=np.float64)
        for _ in range(4):
            for n in FREE_V3:
                j = f[n]
                r[:, j] = np.clip(r[:, j], rc[:, j] - b[j], rc[:, j] + b[j])
            r = _enforce(r, rc, f, increase_only)
        for n in INTEGER_VALUED:
            j = f[n]
            lo = np.ceil(np.maximum(rc[:, j] - b[j], 0.0) - 1e-9)
            hi = np.floor(rc[:, j] + b[j] + 1e-9)
            r[:, j] = np.where(hi < lo, rc[:, j], np.clip(r[:, j], lo, hi))
            r = _enforce(r, rc, f, increase_only)
        # Conditions C and E snap values discontinuously, so clipping cannot
        # always pull them back. The clean record satisfies every condition by
        # construction, so reverting an out-of-budget feature to its clean
        # value is always both feasible and budget-respecting.
        free_j = np.array([f[n] for n in FREE_V3])
        for _ in range(8):
            over = np.abs(r[:, free_j] - rc[:, free_j]) > b[free_j] + 1e-9
            if not over.any():
                break
            idx = np.where(over)
            r[idx[0], free_j[idx[1]]] = rc[idx[0], free_j[idx[1]]]
            r = _enforce(r, rc, f, increase_only)
    return recompute_derived(r, features)

ACTIVITY_TIMEOUT = 5_000_000.0

def probe(raw, names, label, tol=0.01, verbose=True):
    f = {n: i for i, n in enumerate(names)}
    r = np.asarray(raw, dtype=np.float64)
    dur = r[:, f["Flow Duration"]]
    fp, bp = r[:, f["Total Fwd Packets"]], r[:, f["Total Backward Packets"]]
    ntot = fp + bp
    out = {}

    g = ntot > 1
    lhs = r[:, f["Flow IAT Mean"]] * np.maximum(ntot - 1, 1)
    bad = np.zeros(len(r), bool)
    bad[g] = np.abs(lhs[g] - dur[g]) / np.maximum(dur[g], 1e-9) > tol
    out["A  Flow IAT Mean x (n-1) = Duration"] = bad

    # B: Bhatia-Davis on the POPULATION variance; CICFlowMeter reports the
    # SAMPLE sd, so the cap is inflated by k/(k-1) with k = number of gaps.
    for fam, cnt in [("Flow IAT", ntot), ("Fwd IAT", fp), ("Bwd IAT", bp)]:
        mn, mu = r[:, f[f"{fam} Min"]], r[:, f[f"{fam} Mean"]]
        mx, sd = r[:, f[f"{fam} Max"]], r[:, f[f"{fam} Std"]]
        k = np.maximum(cnt - 1, 0)
        g = k >= 2
        cap = (mu - mn) * (mx - mu) * np.where(g, k / np.maximum(k - 1, 1), 1.0)
        bad = np.zeros(len(r), bool)
        bad[g] = sd[g] ** 2 > cap[g] * (1.0 + tol) + 1e-9
        out[f"B  {fam} sd within [min,max] bound"] = bad

    idle = r[:, f["Idle Mean"]]
    out["C  Idle Mean >= 5 s when non-zero"] = (idle > 0) & (idle < ACTIVITY_TIMEOUT)

    # D: n_a x ActiveMean + n_i x IdleMean = Duration for SOME small integers.
    # CICFlowMeter drops zero-length active periods, so n_a is not n_i + 1;
    # it is only loosely coupled to n_i.
    act = r[:, f["Active Mean"]]
    best = np.full(len(r), np.inf)
    maxi = 64
    for ni in range(0, maxi):
        rem = dur - ni * idle
        with np.errstate(divide="ignore", invalid="ignore"):
            na = np.where(act > 0, rem / np.maximum(act, 1e-9), 0.0)
        na_r = np.clip(np.round(na), 0, ni + 1)
        resid = np.abs(na_r * act + ni * idle - dur) / np.maximum(dur, 1e-9)
        ok_shape = (rem >= -1e-9)
        best = np.minimum(best, np.where(ok_shape, resid, np.inf))
    nothing = (act <= 0) & (idle <= 0)
    bad = (best > tol) & ~nothing
    out["D  Active/Idle reconcile with Duration"] = bad

    out["E  counts admit the reported IATs"] = (
        ((fp < 2) & (r[:, f["Fwd IAT Mean"]] > 0)) |
        ((bp < 2) & (r[:, f["Bwd IAT Mean"]] > 0)) | (ntot < 2))

    anyb = np.zeros(len(r), bool)
    for v in out.values():
        anyb |= v
    out["   ANY condition unsatisfiable"] = anyb
    if verbose:
        print(f"\n{label}  (n = {len(r)})")
        for kk, v in out.items():
            print(f"   {kk:44} {v.mean():7.2%}")
    return out


# --------------------------------------------------------------------------
# PGD restricted to the 20 controlled variables, projected with project_v3
# --------------------------------------------------------------------------
mean64 = scaler.mean_.astype(np.float64)
scale64 = scaler.scale_.astype(np.float64)
fidx = {n: i for i, n in enumerate(present_features)}
free_idx_v3 = np.array([fidx[n] for n in FREE_V3])
budget_raw = cfg.adv_epsilon * scale64


def to_std(raw):
    return ((np.asarray(raw, np.float64) - mean64) / scale64).astype(np.float32)


def pgd_v3(model, x_std, y, rc, epsilon, alpha, steps):
    was = model.training
    model.train(False)
    x0 = x_std.clone().detach()
    x_adv = x0.clone().detach()
    raw = np.asarray(rc, np.float64).copy()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits, _ = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad, = torch.autograd.grad(loss, [x_adv])
        with torch.no_grad():
            step = alpha * grad.sign()
            m = torch.zeros_like(step); m[:, free_idx_v3] = 1.0
            x_adv = (x_adv + step * m).detach()
        raw = project_v3(x_adv.cpu().numpy().astype(np.float64) * scale64 + mean64,
                         rc, present_features, inc_only, budget_raw)
        x_adv = torch.tensor(to_std(raw), dtype=torch.float32, device=DEVICE).detach()
    model.train(was)
    return x_adv, raw


X_f_t, raw_f = pgd_v3(main_model, to_t(X_atk), y_atk_t, raw_clean,
                      cfg.adv_epsilon, cfg.adv_alpha, cfg.adv_steps)
X_f = X_f_t.detach().cpu().numpy()


def disp(raw):
    return (np.abs(raw - raw_clean) / scale64).max(axis=1)


# bisect to the attacker that also respects epsilon on every feature
lo = np.zeros(len(raw_clean)); hi = np.ones(len(raw_clean))
d_f = raw_f - raw_clean
for _ in range(24):
    mid = 0.5 * (lo + hi)
    cand = project_v3(raw_clean + mid[:, None] * d_f, raw_clean,
                      present_features, inc_only, budget_raw)
    ok = disp(cand) <= cfg.adv_epsilon + 1e-9
    lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
raw_eps = project_v3(raw_clean + lo[:, None] * d_f, raw_clean,
                     present_features, inc_only, budget_raw)
X_eps = to_std(raw_eps)

# --------------------------------------------------------------------------
# audit: algebraic relations + realizability conditions
# --------------------------------------------------------------------------
def full_audit(raw, label):
    alg = audit_feasibility(raw, present_features, atol)
    rz = probe(raw, present_features, "", verbose=False)
    keys = [k for k in rz if k[0] in "ABCE"]
    anyrz = np.logical_or.reduce([rz[k] for k in keys])
    out = {k: v for k, v in alg.items() if k != "ANY constraint violated"}
    for k in keys:
        out[k.strip()] = rz[k].mean()
    out["D  (measured, not enforced)"] = rz["D  Active/Idle reconcile with Duration"].mean()
    algbool = alg["ANY constraint violated"]
    out["ANY enforced constraint violated"] = max(algbool, float(anyrz.mean())) \
        if algbool or anyrz.any() else 0.0
    out["_anyrz"] = anyrz
    return out


SIG = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean"]
env = {n: (np.percentile(raw_clean[:, fidx[n]], 1),
           np.percentile(raw_clean[:, fidx[n]], 99)) for n in SIG}


def in_env(raw):
    k = np.ones(len(raw), bool)
    for n in SIG:
        a, b = env[n]; j = fidx[n]
        k &= (raw[:, j] >= a) & (raw[:, j] <= b)
    return k


clean_preds, _ = predict_torch(main_model, X_atk)
print(f"{'':36}{'ESR':>8}{'slow':>9}{'ESR|slow':>10}{'infeas.':>9}{'in budget':>11}")
print("-" * 83)
for lab, Xa, Ra in [("unconstrained feature-space", X_atk_adv_np, raw_adv_unc),
                    ("realizable, 20 levers at eps", X_f,   raw_f),
                    ("realizable, all-27 within eps", X_eps, raw_eps)]:
    pr, _ = predict_torch(main_model, Xa)
    e = evasion_success_rate(y_atk, clean_preds, pr)
    k = in_env(Ra)
    es = evasion_success_rate(y_atk[k], clean_preds[k], pr[k]) if k.sum() else float("nan")
    a = full_audit(Ra, lab)
    inb = (disp(Ra) <= cfg.adv_epsilon + 1e-9).mean()
    print(f"{lab:36}{e:7.2%}{k.mean():9.2%}{es:9.2%}{a['ANY enforced constraint violated']:9.2%}{inb:11.2%}")

print("\n--- per-constraint audit, for the manuscript table ---")
ac = full_audit(raw_clean, "clean"); au = full_audit(raw_adv_unc, "unc")
print(f"{'constraint':48}{'clean':>9}{'attacked':>10}")
for k in ac:
    if k.startswith("_"):
        continue
    print(f"{k:48}{ac[k]:8.2%}{au[k]:9.2%}")

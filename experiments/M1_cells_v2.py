# ============================ CELL 22 (REPLACE) ============================
"""
M1 -- Feature-space attack feasibility (v2, calibrated against CICIDS2017).

CHANGES FROM v1 (all forced by measurement, not assumption):
  * Rate features carry a 1e6 factor: CICFlowMeter reports Flow Duration and all
    IAT fields in MICROSECONDS, while Flow Bytes/s and Flow Packets/s are per
    SECOND. Measured median ratio with the factor: 1.0000 (100% of clean flows
    within 1%).
  * "Active Mean + Idle Mean <= Flow Duration" is NOT enforced. It is violated by
    5.49% of unmodified CICIDS2017 flows, so CICFlowMeter does not maintain it and
    neither should a feasibility projection.
  * Non-negativity is tested with a PER-COLUMN tolerance (1e-4 x the column's
    standard deviation) rather than an absolute one. Standardising and then
    inverting in float32 perturbs a true zero by an amount proportional to the
    column's scale; with an absolute -1e-6 threshold that artefact was
    misread as 42-82% of flows carrying negative byte counts.

Calibration on the 1,694 attack flows of the CICIDS2017 test split, read from
X_test_raw (no round-trip): every identity and ordering below holds for 100% of
clean records at a 1% relative tolerance. The constraint set is therefore a
property of the data, not of the author.

Two additions to the notebook, neither of which retracts an existing result:

  (1) audit_feasibility()  -- measures how often the EXISTING adversarial examples
      violate these relations. Costs nothing, changes no reported number.

  (2) pgd_attack_feasible() -- PGD restricted to the attacker's genuinely free
      variables, with derived features recomputed after every step, so each
      iterate corresponds to a self-consistent flow record.

Reported side by side, these give an upper bound (unconstrained feature-space
attacker) and a realizable lower bound (consistency-preserving attacker).
"""
import numpy as np

# --------------------------------------------------------------------------
# A NetFlow record is not 27 free numbers. Six of the 27 are deterministic
# functions of the others; perturbing them independently produces a vector that
# no packet capture could have generated.
# --------------------------------------------------------------------------
FREE = [                       # attacker levers
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean", "Active Std", "Idle Mean", "Idle Std",
]
DERIVED = [                    # recomputed, never perturbed directly
    "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s", "Fwd IAT Total", "Bwd IAT Total",
]
INTEGER_VALUED = {"Total Fwd Packets", "Total Backward Packets"}

ORDERINGS = [("Flow IAT Min", "Flow IAT Mean", "Flow IAT Max"),
             ("Fwd IAT Min",  "Fwd IAT Mean",  "Fwd IAT Max"),
             ("Bwd IAT Min",  "Bwd IAT Mean",  "Bwd IAT Max")]

US_PER_S = 1.0e6               # CICFlowMeter: durations/IATs in us, rates per s
EPS_DIV = 1e-9


def _idx(features):
    return {n: i for i, n in enumerate(features)}


def column_atol(scaler, factor=1e-4):
    """Per-column absolute tolerance at the float32 round-trip noise floor.

    A value standardised and inverted in float32 carries an error proportional
    to the column's standard deviation, so a single absolute threshold is wrong
    by orders of magnitude across a feature set spanning 1 to 1e8.
    """
    return factor * np.asarray(scaler.scale_, dtype=np.float64)


# --------------------------------------------------------------------------
def recompute_derived(raw, features):
    """Overwrite the six derived columns so the record is internally consistent."""
    f = _idx(features)
    r = np.asarray(raw, dtype=np.float64).copy()

    dur = r[:, f["Flow Duration"]]
    fp  = r[:, f["Total Fwd Packets"]]
    bp  = r[:, f["Total Backward Packets"]]
    fb  = r[:, f["Total Length of Fwd Packets"]]
    bb  = r[:, f["Total Length of Bwd Packets"]]

    r[:, f["Fwd Packet Length Mean"]] = np.where(fp > 0, fb / np.maximum(fp, EPS_DIV), 0.0)
    r[:, f["Bwd Packet Length Mean"]] = np.where(bp > 0, bb / np.maximum(bp, EPS_DIV), 0.0)
    r[:, f["Flow Bytes/s"]]   = np.where(dur > 0, (fb + bb) * US_PER_S / np.maximum(dur, EPS_DIV), 0.0)
    r[:, f["Flow Packets/s"]] = np.where(dur > 0, (fp + bp) * US_PER_S / np.maximum(dur, EPS_DIV), 0.0)
    r[:, f["Fwd IAT Total"]]  = r[:, f["Fwd IAT Mean"]] * np.maximum(fp - 1.0, 0.0)
    r[:, f["Bwd IAT Total"]]  = r[:, f["Bwd IAT Mean"]] * np.maximum(bp - 1.0, 0.0)
    return r


def _enforce_free(r, rc, f, increase_only):
    """Constraints that involve only the attacker-controlled variables."""
    # 1. domain: counts, durations, byte totals and times are non-negative
    for n in FREE:
        r[:, f[n]] = np.maximum(r[:, f[n]], 0.0)
    # 2. direction constraint (an exploit cannot send fewer bytes than it needs)
    for n in increase_only:
        if n in f:
            r[:, f[n]] = np.maximum(r[:, f[n]], rc[:, f[n]])
    # 3. order statistics within each IAT family
    for lo, mid, hi in ORDERINGS:
        r[:, f[lo]] = np.minimum(r[:, f[lo]], r[:, f[mid]])
        r[:, f[hi]] = np.maximum(r[:, f[hi]], r[:, f[mid]])
    # 4. integrality of packet counts -- applied LAST so nothing re-breaks it
    for n in INTEGER_VALUED:
        r[:, f[n]] = np.round(r[:, f[n]])
    return r

    # NOTE: no Active+Idle <= Duration constraint. 5.49% of UNMODIFIED CICIDS2017
    # flows violate it, so it is not a property CICFlowMeter maintains.


def project_to_feasible(raw, raw_clean, features, increase_only=(), budget_raw=None):
    """Project a perturbed RAW record onto the set of realizable flow records.

    budget_raw : optional (27,) per-column L_inf budget in RAW units. When given,
        the ATTACKER-CONTROLLED features are confined to raw_clean +/- budget_raw.
        The derived features are deliberately NOT budgeted: the attacker does not
        set them, they follow from the free variables by definition, so holding
        them inside an epsilon-ball would impose a constraint no attacker faces.
    """
    f = _idx(features)
    r = np.asarray(raw, dtype=np.float64).copy()
    rc = np.asarray(raw_clean, dtype=np.float64)

    r = _enforce_free(r, rc, f, increase_only)

    if budget_raw is not None:
        b = np.asarray(budget_raw, dtype=np.float64)
        # The budget box and the ordering constraints are both convex, so
        # alternating between them converges; three passes suffice here because
        # the only interaction is min/max against a neighbouring column.
        # Feasibility is the claim being made, the epsilon-ball only a modelling
        # convention, so the loop ENDS on the feasibility pass: where the two
        # cannot both hold, the record stays realizable and the residual
        # displacement is reported rather than hidden.
        for _ in range(4):
            for n in FREE:
                j = f[n]
                r[:, j] = np.clip(r[:, j], rc[:, j] - b[j], rc[:, j] + b[j])
            r = _enforce_free(r, rc, f, increase_only)
        # Rounding to the NEAREST integer can jump a whole packet and so leave
        # the budget. Round to the nearest integer INSIDE it; where the budget
        # admits no integer other than the original, the count cannot move at
        # all -- which is itself a finding about this threat model.
        for n in INTEGER_VALUED:
            j = f[n]
            lo = np.ceil(np.maximum(rc[:, j] - b[j], 0.0) - 1e-9)
            hi = np.floor(rc[:, j] + b[j] + 1e-9)
            r[:, j] = np.where(hi < lo, rc[:, j], np.clip(r[:, j], lo, hi))

    # derived columns follow from the free ones
    return recompute_derived(r, features)


# --------------------------------------------------------------------------
def audit_feasibility(raw_adv, features, atol, tol=0.01):
    """Fraction of records violating each relation.

    raw_adv : (n, 27) RAW-unit records, float64
    atol    : (27,) per-column absolute tolerance, from column_atol(scaler)
    tol     : relative tolerance for the algebraic identities
    """
    f = _idx(features)
    r = np.asarray(raw_adv, dtype=np.float64)
    atol = np.asarray(atol, dtype=np.float64)

    dur = r[:, f["Flow Duration"]]
    fp  = r[:, f["Total Fwd Packets"]]
    bp  = r[:, f["Total Backward Packets"]]
    fb  = r[:, f["Total Length of Fwd Packets"]]
    bb  = r[:, f["Total Length of Bwd Packets"]]

    def off(lhs_name, rhs, guard):
        """Relative deviation > tol, evaluated only where the identity is defined."""
        lhs = r[:, f[lhs_name]]
        bad = np.zeros(len(r), dtype=bool)
        g = guard & np.isfinite(rhs) & (np.abs(rhs) > atol[f[lhs_name]])
        bad[g] = np.abs(lhs[g] - rhs[g]) / np.abs(rhs[g]) > tol
        # where the identity says the value must be ~0, test absolutely
        z = guard & ~g
        bad[z] = np.abs(lhs[z]) > atol[f[lhs_name]]
        return bad

    checks = {
        "Fwd Pkt Len Mean = FwdBytes/FwdPkts":
            off("Fwd Packet Length Mean", fb / np.maximum(fp, EPS_DIV), fp > 0),
        "Bwd Pkt Len Mean = BwdBytes/BwdPkts":
            off("Bwd Packet Length Mean", bb / np.maximum(bp, EPS_DIV), bp > 0),
        "Flow Bytes/s = bytes x 1e6 / duration":
            off("Flow Bytes/s", (fb + bb) * US_PER_S / np.maximum(dur, EPS_DIV), dur > 0),
        "Flow Packets/s = packets x 1e6 / duration":
            off("Flow Packets/s", (fp + bp) * US_PER_S / np.maximum(dur, EPS_DIV), dur > 0),
        "Fwd IAT Total = Mean x (FwdPkts-1)":
            off("Fwd IAT Total", r[:, f["Fwd IAT Mean"]] * np.maximum(fp - 1, 0.0), fp > 1),
        "Bwd IAT Total = Mean x (BwdPkts-1)":
            off("Bwd IAT Total", r[:, f["Bwd IAT Mean"]] * np.maximum(bp - 1, 0.0), bp > 1),
    }
    for lo, mid, hi in ORDERINGS:
        checks[f"{lo} <= Mean <= {hi}"] = (
            (r[:, f[lo]] > r[:, f[mid]] + atol[f[mid]]) |
            (r[:, f[mid]] > r[:, f[hi]] + atol[f[hi]]))

    checks["No negative-valued field"] = (r < -atol[None, :]).any(axis=1)
    checks["Packet counts integral"] = (
        (np.abs(fp - np.round(fp)) > 1e-6) | (np.abs(bp - np.round(bp)) > 1e-6))

    rows = {k: float(v.mean()) for k, v in checks.items()}
    any_viol = np.zeros(len(r), dtype=bool)
    for v in checks.values():
        any_viol |= v
    rows["ANY constraint violated"] = float(any_viol.mean())
    return rows


# --------------------------------------------------------------------------
def make_feasible_pgd(torch, F, scaler, features, device, increase_only=()):
    """Return pgd_attack_feasible(model, x_std, y, raw_clean, eps, alpha, steps).

    Operates in standardised space like the original attack, but after every
    step maps to raw units, projects onto the feasible set and maps back, so
    every iterate is a self-consistent flow record. Returns BOTH the tensor the
    model sees and the raw record it corresponds to, so the audit can be run on
    the record itself rather than on a float32 round-trip of it.
    """
    f = _idx(features)
    free_idx = np.array([f[n] for n in FREE])
    mean_  = np.asarray(scaler.mean_,  dtype=np.float64)
    scale_ = np.asarray(scaler.scale_, dtype=np.float64)

    def to_raw(x_std):
        return np.asarray(x_std, dtype=np.float64) * scale_ + mean_

    def to_std(raw):
        return ((np.asarray(raw, dtype=np.float64) - mean_) / scale_).astype(np.float32)

    def pgd_attack_feasible(model, x_std, y, raw_clean, epsilon, alpha, steps,
                            training_mode=False):
        was_training = model.training
        model.train(training_mode)

        x0 = x_std.clone().detach()
        raw_clean = np.asarray(raw_clean, dtype=np.float64)
        budget_raw = epsilon * scale_          # epsilon-ball expressed in raw units
        x_adv = x0.clone().detach()
        raw = raw_clean.copy()

        for _ in range(steps):
            x_adv.requires_grad_(True)
            logits, _ = model(x_adv)
            loss = F.cross_entropy(logits, y)
            grad, = torch.autograd.grad(loss, [x_adv])
            with torch.no_grad():
                step = alpha * grad.sign()
                mask = torch.zeros_like(step)
                mask[:, free_idx] = 1.0          # only the free levers move
                x_adv = (x_adv + step * mask).detach()

            # one projection that enforces feasibility AND the budget on the
            # free variables, then lets the derived features follow
            raw = project_to_feasible(to_raw(x_adv.cpu().numpy()), raw_clean,
                                      features, increase_only, budget_raw)
            x_adv = torch.tensor(to_std(raw), dtype=torch.float32,
                                 device=device).detach()

        model.train(was_training)
        return x_adv, raw

    return pgd_attack_feasible


# ============================ CELL 23 (REPLACE) ============================
# M1 audit: how often do the EXISTING adversarial examples correspond to a flow
# record that CICFlowMeter could have produced?
#
# The adversarial RAW record is reconstructed by applying the standardised DELTA
# to the exact clean raw values:
#       raw_adv = X_test_raw[atk] + (X_atk_adv - X_atk) * scaler.scale_
# NOT by scaler.inverse_transform(X_atk_adv). Inverting a full standardised
# vector in float32 carries an error proportional to each column's standard
# deviation, which for Flow Duration (sd ~1e7 us, minimum value 2.56 us) is
# larger than the short-duration values themselves. The delta is O(epsilon), so
# the same relative precision costs ~1e-7 * epsilon * sd -- negligible.

import numpy as np

atk_mask   = (y_test == 1)
raw_clean  = X_test_raw[atk_mask].astype(np.float64)        # exact, no round trip
atol       = column_atol(scaler)
inc_only   = [n for n, m in zip(present_features, direction_mask_np)
              if m == 1.0 and n in FREE]
print("increase-only features under the paper's direction constraint:", inc_only)

delta_std  = (X_atk_adv_np.astype(np.float64) - X_atk.astype(np.float64))
raw_adv_unc = raw_clean + delta_std * scaler.scale_.astype(np.float64)

ctrl = audit_feasibility(raw_clean,   present_features, atol)   # must be ~0
unc  = audit_feasibility(raw_adv_unc, present_features, atol)

print(f"\n{'constraint':46} {'clean':>9} {'attacked':>10}")
print("-" * 68)
for k in unc:
    print(f"{k:46} {ctrl[k]:8.2%} {unc[k]:9.2%}")

assert ctrl["ANY constraint violated"] < 0.01, (
    "unmodified CICIDS2017 flows must satisfy the constraint set; if they do "
    "not, the constraint set is wrong and nothing below is interpretable")
print("\nControl passed: the constraint set is a property of the data.")


# ============================ CELL 24 (REPLACE) ============================
# M1 experiment: consistency-preserving attack.
# Same model, same epsilon, same number of steps -- the only difference is that
# the attacker may move only the 21 variables it controls, and the 6 derived
# features are recomputed from them after every step.

import torch, numpy as np

pgd_feasible = make_feasible_pgd(torch, F, scaler, present_features, DEVICE,
                                 increase_only=inc_only)

X_atk_adv_f_t, raw_adv_f = pgd_feasible(
    main_model, to_t(X_atk), y_atk_t, raw_clean,
    cfg.adv_epsilon, cfg.adv_alpha, cfg.adv_steps)
X_atk_adv_f = X_atk_adv_f_t.detach().cpu().numpy()

feas = audit_feasibility(raw_adv_f, present_features, atol)
print(f"{'constraint':46} {'unconstr.':>10} {'feasible':>10}")
print("-" * 68)
for k in feas:
    print(f"{k:46} {unc[k]:9.2%} {feas[k]:9.2%}")

# --- evasion success rate, both attackers -------------------------------
clean_preds, _ = predict_torch(main_model, X_atk)
adv_u, _ = predict_torch(main_model, X_atk_adv_np)
adv_f, _ = predict_torch(main_model, X_atk_adv_f)
esr_u = evasion_success_rate(y_atk, clean_preds, adv_u)
esr_f = evasion_success_rate(y_atk, clean_preds, adv_f)

print(f"\nEvasion success rate")
print(f"  unconstrained feature-space attacker : {esr_u:.4f}")
print(f"  consistency-preserving attacker      : {esr_f:.4f}")

# --- how much attack surface the constraints remove ---------------------
free_j = [present_features.index(n) for n in FREE]
der_j  = [present_features.index(n) for n in DERIVED]
dev    = np.abs(raw_adv_f - raw_clean) / scaler.scale_.astype(np.float64)
ov     = dev[:, free_j].max(axis=1)
print(f"\ncontrolled-feature L_inf displacement: max {ov.max():.4f} "
      f"(budget {cfg.adv_epsilon})")
print(f"  records within budget: {(ov <= cfg.adv_epsilon + 1e-6).mean():.2%}")
print(f"  derived-feature displacement, median / 95th pct: "
      f"{np.median(dev[:, der_j].max(axis=1)):.3f} / "
      f"{np.percentile(dev[:, der_j].max(axis=1), 95):.3f}")
mv_u = (np.abs(raw_adv_unc - raw_clean) > atol[None, :]).mean()
mv_f = (np.abs(raw_adv_f  - raw_clean) > atol[None, :]).mean()
print(f"  feature cells moved: unconstrained {mv_u:.1%}, feasible {mv_f:.1%}")
for n in ["Total Fwd Packets", "Total Backward Packets"]:
    j = present_features.index(n)
    print(f"  {n:24} changed in {(np.abs(raw_adv_f[:,j]-raw_clean[:,j])>0).mean():6.2%} "
          f"of records (raw budget {cfg.adv_epsilon*scaler.scale_[j]:.3f} packets)")

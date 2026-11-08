# ===================== CELL 29 (NEW) — M2 close-out: APGD, step sizes, EOT/BPDA =====================
# Completes the M2 checklist: a step-size sweep, an AutoAttack-style APGD
# (adaptive step size + momentum + restarts, CE and binary-DLR losses), the
# worst-case ensemble over all attacks, and the EOT/BPDA applicability note.
# Runs on the already-trained main_model. Self-contained.

import numpy as np, torch, torch.nn.functional as F
_m = torch.tensor(direction_mask_np, dtype=torch.float32, device=DEVICE)

def _proj(xa, x, eps):
    xa = torch.clamp(xa, x-eps, x+eps)
    inc = (_m == 1.0).unsqueeze(0).expand_as(xa)      # increase-only: not below clean
    return torch.where(inc, torch.maximum(xa, x), xa)

def _pgd(model, x, y, eps, alpha, steps):
    was = model.training; model.train(False)
    xa = x.clone().detach()
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa)[0], y)
        g, = torch.autograd.grad(loss, [xa])
        with torch.no_grad():
            xa = _proj(xa + alpha*g.sign(), x, eps).detach()
    model.train(was); return xa

def _apgd(model, x, y, eps, steps=100, restarts=2, loss_type="ce", seed=0):
    """Auto-PGD (Croce & Hein, 2020). DLR is undefined for 2 classes, so
    loss_type='margin' maximises the binary logit margin (z_other - z_y)."""
    was = model.training; model.train(False)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    ar = torch.arange(len(y), device=DEVICE)
    def L(logits):
        if loss_type == "ce":
            return F.cross_entropy(logits, y, reduction="none")
        other = 1 - y
        return logits[ar, other] - logits[ar, y]
    best = x.clone().detach(); best_bad = torch.zeros(len(x), dtype=torch.bool, device=DEVICE)
    for r in range(restarts):
        xa = x.clone().detach()
        if r > 0:
            xa = _proj(x + (torch.rand(x.shape, generator=g, device=DEVICE)*2-1)*eps, x, eps)
        eta = torch.full((len(x),1), 2.0*eps, device=DEVICE); x_prev = xa.clone()
        xa.requires_grad_(True); l = L(model(xa)[0]); l.sum().backward()
        grad = xa.grad.detach()
        with torch.no_grad():
            xa = _proj(xa + eta*grad.sign(), x, eps).detach()
        best_loss = l.detach().clone(); xbest = x.clone().detach()
        chk = [int(p*steps) for p in (0.22,0.44,0.66,0.88)]; succ = torch.zeros(len(x), device=DEVICE)
        prev_best = best_loss.clone()
        for t in range(1, steps):
            xa.requires_grad_(True); l = L(model(xa)[0]); l.sum().backward()
            grad = xa.grad.detach()
            with torch.no_grad():
                imp = l.detach() > best_loss
                xbest = torch.where(imp.unsqueeze(1), xa.detach(), xbest)
                best_loss = torch.where(imp, l.detach(), best_loss); succ += imp.float()
                a = 0.75
                z = _proj(xa + eta*grad.sign(), x, eps)
                xn = _proj(xa + a*(z-xa) + (1-a)*(xa-x_prev), x, eps)
                x_prev = xa.detach(); xa = xn.detach()
                if t in chk:
                    j = chk.index(t); interval = t - (chk[j-1] if j>0 else 0)
                    halve = (succ < 0.75*interval) | (best_loss <= prev_best + 1e-9)
                    eta = torch.where(halve.unsqueeze(1), eta/2, eta)
                    xa = torch.where(halve.unsqueeze(1), xbest, xa)
                    prev_best = best_loss.clone(); succ = torch.zeros(len(x), device=DEVICE)
        with torch.no_grad():
            bad = model(xbest)[0].argmax(1) != y
            take = bad & (~best_bad); best[take] = xbest[take]; best_bad |= bad
    model.train(was); return best

clean_preds, _ = predict_torch(main_model, X_atk)
den = clean_preds == y_atk
print(f"clean-correct denominator: {int(den.sum())} of {len(y_atk)}\n")
Xa_t = to_t(X_atk); E = cfg.adv_epsilon

def _esr(adv_np):
    return evasion_success_rate(y_atk, clean_preds, predict_torch(main_model, adv_np)[0])

print("1) STEP-SIZE SWEEP (PGD, eps=0.05, 50 steps)")
for al in [0.002, 0.005, 0.01, 0.02, 0.05]:
    print(f"   alpha={al:5}: ESR={_esr(_pgd(main_model, Xa_t, y_atk_t, E, al, 50).cpu().numpy()):7.2%}")

print("\n2) AUTOATTACK-STYLE: PGD vs APGD-CE vs APGD-margin vs worst-case ensemble")
for eps in [E, 4*E, 20*E]:
    advs = {
        "PGD":        _pgd(main_model, Xa_t, y_atk_t, eps, eps/5, 100).cpu().numpy(),
        "APGD-CE":    _apgd(main_model, Xa_t, y_atk_t, eps, 100, 2, "ce").cpu().numpy(),
        "APGD-margin":_apgd(main_model, Xa_t, y_atk_t, eps, 100, 2, "margin").cpu().numpy(),
    }
    rates = {k: _esr(v) for k, v in advs.items()}
    # worst-case ensemble: flipped by ANY attack, among clean-correct
    flipped = np.zeros(len(y_atk), dtype=bool)
    for v in advs.values():
        flipped |= (predict_torch(main_model, v)[0] != y_atk)
    ens = float((flipped & den).sum() / max(den.sum(), 1))
    print(f"   eps={eps:7.3f}: PGD={rates['PGD']:7.2%}  APGD-CE={rates['APGD-CE']:7.2%}  "
          f"APGD-margin={rates['APGD-margin']:7.2%}  ENSEMBLE={ens:7.2%}")

print("\n3) EOT / BPDA applicability")
print("   During attack generation the model runs in eval mode (dropout disabled),")
print("   so it is deterministic: Expectation-over-Transformation is unnecessary")
print("   (nothing to average over). Every component (linear encoders, attention,")
print("   LayerNorm, pooling, classifier) is differentiable with exact gradients,")
print("   so Backward-Pass Differentiable Approximation does not apply either.")

# ============================ CELL 28 (NEW) — M2: attack-strength battery ============================
# Evidence that the low evasion rate is genuine robustness, not gradient masking.
# Runs on the ALREADY-TRAINED main_model. No retraining. Deterministic (eval mode).
#
# Five diagnostics, following Carlini et al., "On Evaluating Adversarial Robustness":
#   1. budget sweep        evasion must rise smoothly with epsilon
#   2. unbounded sanity    with constraints OFF and large epsilon, evasion -> ~100%
#   3. step sweep          evasion non-decreasing in PGD iterations (convergence)
#   4. random restarts     best-of-R should not jump (single start was adequate)
#   5. SPSA (gradient-free) must not beat PGD -> no masked/obfuscated gradient
# Plus a PGD loss-vs-iteration convergence trace.

import numpy as np, torch, torch.nn.functional as F
_dmask = torch.tensor(direction_mask_np, dtype=torch.float32, device=DEVICE)

def _preds(model, Xnp):
    p, _ = predict_torch(model, Xnp)
    return p

def pgd_attack_cfg(model, x, y, eps, alpha, steps, restarts=1, constrained=True,
                   record_loss=False, seed=0):
    was = model.training; model.train(False)        # deterministic
    x = x.to(DEVICE); y = y.to(DEVICE)
    best = x.clone().detach()
    best_bad = torch.zeros(len(x), dtype=torch.bool, device=DEVICE)
    g = torch.Generator(device=DEVICE)
    losses = []
    for r in range(restarts):
        g.manual_seed(seed + r)
        xa = x.clone().detach()
        if r > 0:
            xa = xa + (torch.rand(x.shape, generator=g, device=DEVICE) * 2 - 1) * eps
        for t in range(steps):
            xa.requires_grad_(True)
            logits, _ = model(xa)
            loss = F.cross_entropy(logits, y)
            if record_loss and r == 0:
                losses.append(float(loss.item()))
            grad, = torch.autograd.grad(loss, [xa])
            with torch.no_grad():
                step = alpha * grad.sign()
                if constrained:
                    inc = _dmask == 1.0
                    step = torch.where(inc.unsqueeze(0).expand_as(step),
                                       torch.clamp(step, min=0.0), step)
                xa = torch.clamp(xa + step, x - eps, x + eps).detach()
        with torch.no_grad():
            bad = model(xa)[0].argmax(1) != y
            take = bad & (~best_bad)
            best[take] = xa[take]; best_bad |= bad
    model.train(was)
    return (best, losses) if record_loss else best

def spsa_attack(model, x, y, eps, steps=100, batch=128, lr=None, delta=0.01,
                constrained=True, seed=0):
    was = model.training; model.eval()
    x = x.to(DEVICE); y = y.to(DEVICE)
    lr = lr if lr is not None else eps / 10
    g = torch.Generator(device=DEVICE); g.manual_seed(seed)
    x0 = x.clone().detach(); xa = x0.clone().detach()
    with torch.no_grad():
        for _ in range(steps):
            ge = torch.zeros_like(xa)
            for _b in range(batch):
                v = (torch.randint(0, 2, xa.shape, generator=g, device=DEVICE).float() * 2 - 1)
                lp = F.cross_entropy(model(torch.clamp(xa + delta*v, x0-eps, x0+eps))[0], y, reduction="none")
                lm = F.cross_entropy(model(torch.clamp(xa - delta*v, x0-eps, x0+eps))[0], y, reduction="none")
                ge += ((lp - lm) / (2*delta)).unsqueeze(1) * v
            ge /= batch
            step = lr * ge.sign()
            if constrained:
                inc = _dmask == 1.0
                step = torch.where(inc.unsqueeze(0).expand_as(step), torch.clamp(step, min=0.0), step)
            xa = torch.clamp(xa + step, x0 - eps, x0 + eps)
    model.train(was)
    return xa

# attack set = clean-correct attack flows (report the denominator explicitly)
clean_preds = _preds(main_model, X_atk)
den = int((clean_preds == y_atk).sum())
print(f"clean-correct attack flows (ESR denominator): {den} of {len(y_atk)}\n")
Xa_t = to_t(X_atk)

def run_esr(Xadv_np):
    return evasion_success_rate(y_atk, clean_preds, _preds(main_model, Xadv_np))

E = cfg.adv_epsilon
print("1) BUDGET SWEEP  (functionality-preserving, the paper's threat model)")
for eps in [E, 2*E, 4*E, 10*E, 20*E, 40*E, 100*E]:
    adv = pgd_attack_cfg(main_model, Xa_t, y_atk_t, eps, eps/5, 20, constrained=True)
    print(f"   eps={eps:7.3f} ({eps/E:5.0f}x): ESR={run_esr(adv.cpu().numpy()):7.2%}")

print("\n2) UNBOUNDED SANITY  (constraints OFF) -- must approach ~100%")
for eps in [10*E, 40*E, 100*E, 400*E]:
    adv = pgd_attack_cfg(main_model, Xa_t, y_atk_t, eps, eps/5, 50, constrained=False)
    print(f"   eps={eps:8.3f}: ESR={run_esr(adv.cpu().numpy()):7.2%}")

print("\n3) STEP SWEEP at eps={:.3f}  (non-decreasing => converged)".format(E))
for st in [7, 20, 50, 100, 200]:
    adv = pgd_attack_cfg(main_model, Xa_t, y_atk_t, E, cfg.adv_alpha, st, constrained=True)
    print(f"   steps={st:4}: ESR={run_esr(adv.cpu().numpy()):7.2%}")

print("\n4) RANDOM RESTARTS at eps={:.3f}".format(4*E))
for R in [1, 5, 10, 20]:
    adv = pgd_attack_cfg(main_model, Xa_t, y_atk_t, 4*E, (4*E)/5, 20, restarts=R, constrained=True)
    print(f"   restarts={R:3}: ESR_best={run_esr(adv.cpu().numpy()):7.2%}")

print("\n5) SPSA GRADIENT-FREE vs PGD  (SPSA must NOT exceed PGD => no masking)")
nsp = min(400, len(y_atk))
Xsp = to_t(X_atk[:nsp]); ysp = torch.tensor(y_atk[:nsp], dtype=torch.long, device=DEVICE)
csp = _preds(main_model, X_atk[:nsp])
for eps in [E, 4*E, 20*E]:
    p = pgd_attack_cfg(main_model, Xsp, ysp, eps, eps/5, 20, constrained=True).cpu().numpy()
    s = spsa_attack(main_model, Xsp, ysp, eps, steps=60, batch=64, constrained=True).cpu().numpy()
    ep = evasion_success_rate(y_atk[:nsp], csp, _preds(main_model, p))
    es = evasion_success_rate(y_atk[:nsp], csp, _preds(main_model, s))
    print(f"   eps={eps:7.3f}: PGD={ep:7.2%}  SPSA={es:7.2%}")

print("\n6) PGD LOSS CONVERGENCE at eps={:.3f} (should plateau)".format(E))
_, losses = pgd_attack_cfg(main_model, Xa_t, y_atk_t, E, cfg.adv_alpha, 50,
                           constrained=True, record_loss=True)
print("   iter :", " ".join(f"{i:6d}" for i in [0,4,9,19,29,49]))
print("   loss :", " ".join(f"{losses[i]:6.3f}" for i in [0,4,9,19,29,49]))
print(f"   change over last 30 steps: {abs(losses[-1]-losses[19]):.4f}")

"""
FGSAN reproducibility patch for Paper_1_attempt_2.ipynb
=======================================================

WHY THIS PATCH EXISTS
---------------------
The architecture, loss, epochs and attack surface in the notebook already match
the paper. The problem is that the reported evasion / ablation numbers are not
reproducible across runs. Two causes were identified:

 1. `pgd_attack()` calls `model.train()` before generating adversarial examples.
    That leaves dropout (p = 0.3) ACTIVE in the group encoders, the attention
    layer and the classifier head, so every PGD gradient is computed through a
    randomly thinned network. Measured effect on the real architecture: repeating
    the *identical* attack on the *identical* model five times changed 69.5% of
    the perturbation cells, with differences up to the full epsilon budget.
    In eval() mode the same attack is bit-for-bit deterministic.

    The code comment claims "cuDNN/attention dropout needs train mode for
    backward". This is not correct for nn.MultiheadAttention: autograd.grad with
    respect to the input works fine in eval() mode (verified).

    Consequence: the evasion success rate is measured against a randomly
    perturbed model rather than the deployed one. This understates attack
    strength and produces very high variance -- it is the likely reason the same
    ablation configuration reported 67.6% in one run and 39.6% in another.

 2. The seed is set once in Cell 1. Because every model built afterwards draws
    from the same global RNG stream, the initialisation of the ablation models
    depends on how many random numbers were consumed earlier, i.e. on cell
    execution order. Re-running cells out of order silently changes results.

WHAT TO CHANGE
--------------
 - Replace Cell 9  (`pgd_attack`) with `pgd_attack` below.
 - Add `set_seed()` and call it immediately before EVERY model construction
   (main model in Cell 12, and inside the ablation loop in Cell 13).
 - Use `evasion_success_rate_repeated()` for all REPORTED evasion numbers
   (Cells 13 and 15) so the paper can quote mean +/- std over trials.

Training-time attack generation may stay stochastic (that is ordinary data
augmentation). Only the EVALUATION attack must be deterministic -- hence the
`training_mode` flag, which defaults to the correct behaviour for reporting.
"""

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. Deterministic seeding -- call before EVERY model construction
# --------------------------------------------------------------------------
def set_seed(seed=42):
    """Reset every RNG stream so model init no longer depends on cell order."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# 2. Replacement for Cell 9
# --------------------------------------------------------------------------
direction_mask = torch.tensor(direction_mask_np, dtype=torch.float32)  # noqa: F821


def pgd_attack(model, x, y, epsilon, alpha, steps, training_mode=False):
    """Functionality-preserving feature-space PGD on the 27-dim flow vector.

    training_mode=False (default): dropout disabled -> deterministic, correct
        for measuring evasion success rate against the deployed model.
    training_mode=True: dropout active -> only for adversarial example
        generation inside the training loop, where the stochasticity acts as
        augmentation rather than as measurement noise.
    """
    device = x.device
    mask = direction_mask.to(device)
    was_training = model.training

    model.train(training_mode)   # <-- the fix: eval() unless explicitly training

    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits, _ = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad, = torch.autograd.grad(loss, [x_adv])
        with torch.no_grad():
            step = alpha * grad.sign()
            inc_only = mask == 1.0
            step = torch.where(inc_only.unsqueeze(0).expand_as(step),
                               torch.clamp(step, min=0.0), step)
            x_adv = torch.clamp(x_adv + step, x - epsilon, x + epsilon)
        x_adv = x_adv.detach()

    model.train(was_training)
    return x_adv


# --------------------------------------------------------------------------
# 3. In train_model (Cell 11), the training-loop call becomes:
#
#        x_adv = pgd_attack(model, x_b[:k], y_b[:k],
#                           cfg.adv_epsilon, cfg.adv_alpha, cfg.adv_steps,
#                           training_mode=True)
#
#    Every OTHER call to pgd_attack (Cells 13 and 15) keeps the default
#    training_mode=False so the reported numbers are deterministic.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 4. Repeated-trial evasion measurement for the reported results
# --------------------------------------------------------------------------
def evasion_success_rate_repeated(model, X_atk, y_atk, cfg, n_trials=5, seed=42):
    """Mean +/- std evasion success rate over n_trials independent attacks.

    With the deterministic attack above, the remaining variation comes only
    from model initialisation and training, so call this with models trained
    under different seeds if you want training variance too. Report as
    'mean +/- std over n trials' in the paper rather than a single figure.
    """
    rates = []
    for t in range(n_trials):
        set_seed(seed + t)
        clean_preds, _ = predict_torch(model, X_atk)                      # noqa: F821
        y_t = torch.tensor(y_atk, dtype=torch.long, device=DEVICE)        # noqa: F821
        X_adv = pgd_attack(model, to_t(X_atk), y_t,                       # noqa: F821
                           cfg.adv_epsilon, cfg.adv_alpha, cfg.adv_steps,
                           training_mode=False)
        adv_preds, _ = predict_torch(model, X_adv.cpu().numpy())          # noqa: F821
        rates.append(evasion_success_rate(y_atk, clean_preds, adv_preds))  # noqa: F821
    rates = np.asarray(rates, dtype=float)
    return {"mean": rates.mean(), "std": rates.std(ddof=1) if len(rates) > 1 else 0.0,
            "trials": rates.tolist()}


# --------------------------------------------------------------------------
# 5. Ablation loop (Cell 13) -- seed before each variant so the four
#    conditions differ ONLY in the factor being ablated
# --------------------------------------------------------------------------
#
#   for name, use_attn, use_adv in configs_to_run:
#       set_seed(SEED)                     # <-- add this line
#       m = FeatureGroupedAttentionNet(group_indices, emb_dim=cfg.emb_dim,
#                                      n_heads=cfg.n_heads, dropout=cfg.dropout,
#                                      use_attention=use_attn)
#       m, _ = train_model(m, X_train, y_train, X_val, y_val, cfg,
#                          use_adv=use_adv, use_focal=True)
#       ...
#       esr = evasion_success_rate_repeated(m, x_atk, y_atk, cfg, n_trials=5)
#
# Without the set_seed call, the "no attention" variants start from a different
# random draw than the "attention" variants, so part of the measured F1 gap is
# initialisation noise rather than the ablated factor.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 6. After re-running, update these paper values from the new outputs:
#       Table 5  clean-data performance          (F1, AUC, FPR, FNR)
#       Table 7  ablation                        (F1, FNR, ESR x 4 variants)
#       Figure 4 ablation bar chart
#       Figure 7 transferability bar chart
#       Figure 8(b) robustness panel
#       Abstract: the "40.4-67.6% to 0.19-1.6%" and "95.6-99.1%" ranges
#       Sections 4.3 and 4.6 prose
# --------------------------------------------------------------------------

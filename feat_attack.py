"""Sensitivity-guided, multi-step feature-statistic augmentation (MSSA)."""
import numpy as np
import torch

def _flatten_per_sample(x):
  return x.reshape(x.size(0), -1)


def make_topk_mask(sensitivity, ratio):
  """Binary top-k mask: top ``ratio`` fraction of features get weight 1, rest 0."""
  if sensitivity.ndim < 2:
    raise ValueError("sensitivity must include a batch dimension and at least one feature dimension")
  flat = _flatten_per_sample(sensitivity)
  total = flat.size(1)
  ratio = float(ratio)
  if ratio <= 0.0:
    return torch.zeros_like(sensitivity)
  k = int(round(total * ratio))
  k = max(1, min(total, k))
  indices = torch.topk(flat, k=k, dim=1).indices
  mask_flat = torch.zeros_like(flat)
  mask_flat.scatter_(1, indices, 1.0)
  return mask_flat.view_as(sensitivity)


# Channel statistics.

def channelwise_stat(feat):
  """feat (B, C, L) -> mu, sigma (B, C). Channel-wise statistics over spatial."""
  mu = feat.mean(dim=2)
  var = feat.var(dim=2, unbiased=False)
  sigma = var.clamp_min(1e-8).sqrt()
  return mu, sigma


def renormalize_feat(feat, mu_old, sigma_old, mu_new, sigma_new):
  """Re-apply statistics: feat_new = (feat - mu_old)/sigma_old * sigma_new + mu_new.

  Broadcast over the spatial dim. Differentiable for backprop through block34.
  """
  eps = 1e-8
  feat_norm = (feat - mu_old[:, :, None]) / (sigma_old[:, :, None] + eps)
  return feat_norm * sigma_new[:, :, None] + mu_new[:, :, None]


# Multi-step adversarial search.

def _feature_logits(model_feat, feat12):
  """Forward block34 -> logits."""
  return model_feat.forward_block34(feat12)


def _build_wmask(v_feat, mu_ref, top_ratio, full_continuous,
                 w_mode, w_alpha, w_rand_seed):
  """Build the per-channel attack mask (B, C)."""
  if v_feat is None:
    return torch.ones_like(mu_ref)
  if full_continuous:
    # Sensitivity-guided continuous modulation on all channels.
    v = v_feat.detach().to(device=mu_ref.device, dtype=mu_ref.dtype).view(-1)
    vmax = v.max().clamp_min(1e-8)
    if w_mode == "uniform":
      w = torch.full_like(v, 1.0 + float(w_alpha) * v.mean() / vmax)
    elif w_mode == "random":
      # Use an independent random generator for the permutation.
      perm = torch.from_numpy(
        np.random.RandomState(int(w_rand_seed)).permutation(int(v.numel()))
      ).to(device=v.device)
      w = 1.0 + float(w_alpha) * v[perm] / vmax
    elif w_mode == "reverse":
      order = torch.argsort(v)
      rank = torch.argsort(order)
      w = 1.0 + float(w_alpha) * v[order].flip(0)[rank] / vmax
    elif w_mode == "uniform_ratio":
      w = torch.full_like(v, float(w_alpha))
    else:  # sensitivity-guided weighting
      w = 1.0 + float(w_alpha) * v / vmax
    return w.unsqueeze(0).expand_as(mu_ref)
  # binary top-k
  v = v_feat.detach().to(device=mu_ref.device, dtype=mu_ref.dtype).view(-1)
  sensitivity = v.unsqueeze(0).expand_as(mu_ref)
  return make_topk_mask(sensitivity, top_ratio)


def attack_feat_stat(
  model_feat,
  x,
  y,
  loss_fn,
  v_feat,
  feat12_clean,
  eps=0.3,
  top_ratio=0.70,
  steps=5,
  early_stop_tau=1,
  attack_eval_mode=True,
  full_continuous=True,
  w_mode="vfeat",
  w_rand_seed=0,
  w_alpha=1.0,
  attack_geometry="per_channel_sign",
  v_mu=None,
  v_sigma=None,
  perturb_target="mu_sigma",
  **kwargs,
):
  """V_feat-guided PGD on the (mu, sigma) statistics of the block12 feature.

  Attacks the channel-wise (mu, sigma) statistics using multiplicative sign-guided
  PGD. Returns feat12_adv (gradient-preserving, so adv CE backprops to block12) +
  attack stats.

  Args:
    model_feat: DeepINClassifierFeat (exposes forward_block12 / forward_block34).
    x: clean input (B, 1, 1024). y: labels. loss_fn: classification loss.
    feat12_clean: clean block12 feat WITH gradient (computed by caller in train
      mode, outside attack). Used as the return base so adv CE backprops to
      block12. Internally detached to feat12_base for attack direction search.
    v_feat: (1, C) or (C,) offline V_feat vulnerability vector.
    eps: multiplicative step budget per run (per-step = eps/steps).
    top_ratio: fraction of channels attacked (if top-k mode).
    steps, early_stop_tau: PGD steps and early stop threshold.
    attack_eval_mode: model.eval() during attack generation (BN discipline).
    full_continuous: continuous mask on all channels (w = 1 + w_alpha * V / Vmax).
    w_mode: mask distribution mode ("vfeat", "uniform", "random", "reverse", "uniform_ratio").
    w_alpha: amplifier slope for full_continuous.
    perturb_target: statistic carrier to perturb ("mu_sigma", "mu_only", "sigma_only").
  """
  if perturb_target not in ("mu_sigma", "mu_only", "sigma_only"):
    raise ValueError(f"Unknown perturb_target: {perturb_target!r}. Must be one of ('mu_sigma', 'mu_only', 'sigma_only').")
  steps = max(1, int(steps))
  eps = float(eps)
  eps_max_val = eps
  # Keep the clean feature in the graph for gradient flow after the search.
  feat12_grad_base = feat12_clean
  if eps_max_val <= 0.0:
    stats = {
      "delta_mean_abs": 0.0, "delta_max_abs": 0.0, "delta_linf": 0.0,
      "delta_rel_mean": 0.0, "delta_rel_max": 0.0, "budget_hit_rate": 0.0,
      "mask_ratio": 0.0, "attack_steps_mean": 0.0, "early_stop_fraction": 0.0,
      "effective_step_size": 0.0, "early_stop_tau": max(0, int(early_stop_tau)),
      "perturb_target": str(perturb_target),
      "delta_mu_mean_abs": 0.0,
      "delta_sigma_mean_abs": 0.0,
    }
    return feat12_grad_base.detach(), stats
  step_size = eps / float(steps)
  early_stop_tau = max(0, int(early_stop_tau))

  # Attack base: detached view of the gradient-preserving base.
  with torch.no_grad():
    feat12_base = feat12_grad_base.detach()
    mu_ref, sigma_ref = channelwise_stat(feat12_base)  # (B, C), detached

  # Channel mask(s). Joint: single mask from v_feat. mu/sigma separated: dual masks from v_mu/v_sigma.
  if v_mu is not None and v_sigma is not None:
    mask_mu = _build_wmask(v_mu, mu_ref, top_ratio,
                           full_continuous, w_mode, w_alpha, w_rand_seed)
    mask_sigma = _build_wmask(v_sigma, mu_ref, top_ratio,
                              full_continuous, w_mode, w_alpha, w_rand_seed)
    dual_mask = True
  else:
    mask_stat = _build_wmask(v_feat, mu_ref, top_ratio,
                             full_continuous, w_mode, w_alpha, w_rand_seed)
    mask_mu = mask_stat
    mask_sigma = mask_stat
    dual_mask = False

  was_training = model_feat.training
  if attack_eval_mode:
    model_feat.eval()
  try:
    mu_adv = mu_ref.clone()
    sigma_adv = sigma_ref.clone().clamp_min(1e-8)

    attack_counts = torch.zeros(mu_ref.size(0), device=mu_ref.device, dtype=torch.float32)
    early_stop_budget = torch.full(
      (mu_ref.size(0),), early_stop_tau, device=mu_ref.device, dtype=torch.long
    )
    active = torch.ones(mu_ref.size(0), device=mu_ref.device, dtype=torch.bool)
    for _ in range(steps):
      if early_stop_tau > 0:
        with torch.no_grad():
          feat_adv_cur = renormalize_feat(feat12_base, mu_ref, sigma_ref, mu_adv, sigma_adv)
          pred = _feature_logits(model_feat, feat_adv_cur).argmax(dim=1)
          changed = pred.ne(y)
          early_stop_budget[changed & active] -= 1
          active = early_stop_budget > 0
        if not bool(active.any().item()):
          break
      mu_adv = mu_adv.detach().requires_grad_(True)
      sigma_adv = sigma_adv.detach().requires_grad_(True)
      feat_adv = renormalize_feat(feat12_base, mu_ref, sigma_ref, mu_adv, sigma_adv)
      logits = _feature_logits(model_feat, feat_adv)
      loss = loss_fn(logits, y)
      grads = torch.autograd.grad(loss, [mu_adv, sigma_adv], retain_graph=False, create_graph=False)
      grad_mu, grad_sigma = [g.detach() for g in grads]
      if early_stop_tau > 0:
        act = active.to(dtype=grad_mu.dtype).view(-1, 1)
        grad_mu = grad_mu * act
        grad_sigma = grad_sigma * act
      # Multiplicative update on (mu, sigma), guided by per-key mask (dual) or shared mask.
      if perturb_target in ("mu_sigma", "mu_only"):
        mu_adv = mu_adv.detach() * (1.0 + step_size * mask_mu * grad_mu.sign())
      else:
        mu_adv = mu_ref.clone()

      if perturb_target in ("mu_sigma", "sigma_only"):
        sigma_adv = sigma_adv.detach() * (1.0 + step_size * mask_sigma * grad_sigma.sign())
        sigma_adv = sigma_adv.clamp_min(1e-8)
      else:
        sigma_adv = sigma_ref.clone()

      attack_counts += active.to(dtype=attack_counts.dtype)
  finally:
    if attack_eval_mode and was_training:
      model_feat.train()

  # Return gradient-preserving feat12_adv: renormalize the GRADIENT base with attacked statistics.
  if perturb_target in ("mu_sigma", "mu_only"):
    mu_ret = mu_adv.detach()
  else:
    mu_ret = mu_ref.detach()

  if perturb_target in ("mu_sigma", "sigma_only"):
    sigma_ret = sigma_adv.detach()
  else:
    sigma_ret = sigma_ref.detach()

  feat12_adv = renormalize_feat(
    feat12_grad_base, mu_ref, sigma_ref, mu_ret, sigma_ret
  )
  early_stopped = (early_stop_budget <= 0) if early_stop_tau > 0 else None

  # Perturbation diagnostics.
  delta_mu = (mu_ret - mu_ref).detach()
  delta_sigma = (sigma_ret - sigma_ref).detach()
  delta_mu_mean_abs = float(delta_mu.abs().mean().item())
  delta_sigma_mean_abs = float(delta_sigma.abs().mean().item())
  delta_abs = torch.cat([delta_mu.abs().flatten(), delta_sigma.abs().flatten()])
  rel_delta = torch.cat([
    (delta_mu / mu_ref.detach().abs().clamp_min(1e-12)).flatten(),
    (delta_sigma / sigma_ref.detach().abs().clamp_min(1e-12)).flatten(),
  ])
  rel_delta_abs = rel_delta.abs()
  effective_mask = mask_sigma if perturb_target == "sigma_only" else mask_mu
  stats = {
    "delta_mean_abs": float(delta_abs.mean().item()),
    "delta_max_abs": float(delta_abs.max().item()),
    "delta_linf": float(delta_abs.max().item()),
    "delta_rel_mean": float(rel_delta.mean().item()),
    "delta_rel_max": float(rel_delta.abs().max().item()),
    "budget_hit_rate": float((rel_delta_abs >= max(0.0, eps_max_val) - 1e-7).float().mean().item()),
    "mask_ratio": float(effective_mask.detach().float().mean().item()),
    "attack_steps_mean": float(attack_counts.mean().item()),
    "attack_steps_max": float(attack_counts.max().item()),
    "early_stop_fraction": (
      float(early_stopped.float().mean().item()) if early_stopped is not None else 0.0
    ),
    "effective_step_size": float(step_size),
    "early_stop_tau": int(early_stop_tau),
    "full_continuous": bool(full_continuous),
    "w_alpha": float(w_alpha),
    "w_mode": str(w_mode),
    "w_rand_seed": int(w_rand_seed),
    "w_min": float(effective_mask.detach().min().item()),
    "w_max": float(effective_mask.detach().max().item()),
    "dual_mask": bool(dual_mask),
    "w_sigma_min": (float(mask_sigma.detach().min().item()) if (dual_mask and perturb_target != "mu_only") else None),
    "w_sigma_max": (float(mask_sigma.detach().max().item()) if (dual_mask and perturb_target != "mu_only") else None),
    "w_sigma_ratio": (float(mask_sigma.detach().float().mean().item()) if (dual_mask and perturb_target != "mu_only") else None),
    "perturb_target": str(perturb_target),
    "delta_mu_mean_abs": delta_mu_mean_abs,
    "delta_sigma_mean_abs": delta_sigma_mean_abs,
    "attack_geometry": str(attack_geometry),
  }
  return feat12_adv, stats


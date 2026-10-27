"""Offline channel-sensitivity measurement for SAReL (DRS).

The input loader must be re-iterable. Its batches may be input tensors or
sequences with the input tensor in the first position.
"""

import numpy as np
import torch

from feat_attack import channelwise_stat, renormalize_feat


def _input(batch):
  return batch[0] if isinstance(batch, (tuple, list)) else batch


@torch.no_grad()
def compute_channel_mu_stats(model_feat, loader, device, num_channels):
  """Per-channel source-domain mean and std of block12 mu AND sigma.
  (sigma stats feed the mu/sigma separated measurement D_sigma.)"""
  total = torch.zeros(num_channels, dtype=torch.float64, device=device)
  total_sq = torch.zeros(num_channels, dtype=torch.float64, device=device)
  sig_total = torch.zeros(num_channels, dtype=torch.float64, device=device)
  sig_total_sq = torch.zeros(num_channels, dtype=torch.float64, device=device)
  count = 0
  for batch in loader:
    x = _input(batch)
    x = x.to(device, non_blocking=True)
    feat12 = model_feat.forward_block12(x)
    mu, sigma = channelwise_stat(feat12)  # (B, C) both
    mu = mu.double()
    sigma = sigma.double()
    total += mu.sum(dim=0)
    total_sq += (mu * mu).sum(dim=0)
    sig_total += sigma.sum(dim=0)
    sig_total_sq += (sigma * sigma).sum(dim=0)
    count += int(x.size(0))
  count = max(1, count)
  mean = total / count
  var = (total_sq / count) - mean * mean
  std = var.clamp_min(0.0).sqrt()
  sig_mean = sig_total / count
  sig_var = (sig_total_sq / count) - sig_mean * sig_mean
  sig_std = sig_var.clamp_min(0.0).sqrt()
  return mean.float(), std.float(), sig_mean.float(), sig_std.float()


@torch.no_grad()
def mean_feat_shift_channel(model_feat, loader, device, channels, perturb_scale):
  """Mean normalized deep feature shift per perturbed channel (GAP 128-dim, SAReL)."""
  model_feat.eval()
  num_c = int(channels.numel())
  total_shift = np.zeros(num_c, dtype=np.float64)
  total_count = 0
  eps = 1e-8
  for batch in loader:
    x = _input(batch)
    x = x.to(device, non_blocking=True)
    batch_size = int(x.size(0))
    total_count += batch_size
    feat12 = model_feat.forward_block12(x)
    mu, sigma = channelwise_stat(feat12)  # (B, C)
    base_feat = model_feat.forward_block34_feat(feat12)
    base_repr = base_feat.view(batch_size, 128, -1).mean(dim=2)
    base_norm = base_repr.norm(dim=1, keepdim=True).clamp_min(eps)
    for start in range(0, num_c):
      c = int(channels[start].item())
      scale = float(perturb_scale[start].item())
      best_shift = 0.0
      for d_mu in (+1.0, -1.0):
        for d_sig in (+1.0, -1.0):
          mu_pert = mu.clone()
          mu_pert[:, c] = mu[:, c] * (1.0 + d_mu * scale)
          sigma_pert = sigma.clone()
          sigma_pert[:, c] = sigma[:, c] * (1.0 + d_sig * scale)
          sigma_pert = sigma_pert.clamp_min(1e-8)
          feat_pert = renormalize_feat(feat12, mu, sigma, mu_pert, sigma_pert)
          pert_feat = model_feat.forward_block34_feat(feat_pert)
          pert_repr = pert_feat.view(batch_size, 128, -1).mean(dim=2)
          shift = (pert_repr - base_repr).norm(dim=1) / base_norm.squeeze(1)
          s = float(shift.sum().item())
          if s > best_shift:
            best_shift = s
      total_shift[start] += best_shift
  return total_shift / max(1, total_count)


@torch.no_grad()
def mean_feat_shift_channel_mu(model_feat, loader, device, channels, perturb_scale):
  """V_mu: mu-only perturbation on GAP-128 deep feature space (sigma held fixed)."""
  model_feat.eval()
  num_c = int(channels.numel())
  total_shift = np.zeros(num_c, dtype=np.float64)
  total_count = 0
  eps = 1e-8
  for batch in loader:
    x = _input(batch)
    x = x.to(device, non_blocking=True)
    batch_size = int(x.size(0))
    total_count += batch_size
    feat12 = model_feat.forward_block12(x)
    mu, sigma = channelwise_stat(feat12)  # (B, C)
    base_feat = model_feat.forward_block34_feat(feat12)
    base_repr = base_feat.view(batch_size, 128, -1).mean(dim=2)
    base_norm = base_repr.norm(dim=1, keepdim=True).clamp_min(eps)
    for start in range(0, num_c):
      c = int(channels[start].item())
      scale = float(perturb_scale[start].item())
      best_shift = 0.0
      for d_mu in (+1.0, -1.0):
        mu_pert = mu.clone()
        mu_pert[:, c] = mu[:, c] * (1.0 + d_mu * scale)
        feat_pert = renormalize_feat(feat12, mu, sigma, mu_pert, sigma)
        pert_feat = model_feat.forward_block34_feat(feat_pert)
        pert_repr = pert_feat.view(batch_size, 128, -1).mean(dim=2)
        shift = (pert_repr - base_repr).norm(dim=1) / base_norm.squeeze(1)
        s = float(shift.sum().item())
        if s > best_shift:
          best_shift = s
      total_shift[start] += best_shift
  return total_shift / max(1, total_count)


@torch.no_grad()
def mean_feat_shift_channel_sigma(model_feat, loader, device, channels, perturb_scale):
  """V_sigma: sigma-only perturbation on GAP-128 deep feature space (mu held fixed)."""
  model_feat.eval()
  num_c = int(channels.numel())
  total_shift = np.zeros(num_c, dtype=np.float64)
  total_count = 0
  eps = 1e-8
  for batch in loader:
    x = _input(batch)
    x = x.to(device, non_blocking=True)
    batch_size = int(x.size(0))
    total_count += batch_size
    feat12 = model_feat.forward_block12(x)
    mu, sigma = channelwise_stat(feat12)  # (B, C)
    base_feat = model_feat.forward_block34_feat(feat12)
    base_repr = base_feat.view(batch_size, 128, -1).mean(dim=2)
    base_norm = base_repr.norm(dim=1, keepdim=True).clamp_min(eps)
    for start in range(0, num_c):
      c = int(channels[start].item())
      scale = float(perturb_scale[start].item())
      best_shift = 0.0
      for d_sig in (+1.0, -1.0):
        sigma_pert = sigma.clone()
        sigma_pert[:, c] = sigma[:, c] * (1.0 + d_sig * scale)
        sigma_pert = sigma_pert.clamp_min(1e-8)
        feat_pert = renormalize_feat(feat12, mu, sigma, mu, sigma_pert)
        pert_feat = model_feat.forward_block34_feat(feat_pert)
        pert_repr = pert_feat.view(batch_size, 128, -1).mean(dim=2)
        shift = (pert_repr - base_repr).norm(dim=1) / base_norm.squeeze(1)
        s = float(shift.sum().item())
        if s > best_shift:
          best_shift = s
      total_shift[start] += best_shift
  return total_shift / max(1, total_count)


def measure_sarel_sensitivity(model, loader, device, sensitivity_eps=0.10, max_channels=0):
  """Measure joint, mean-only and std-only sensitivity spectra.

  Mirrors the original GAP-128 DRS measurement. Each perturbed channel is
  evaluated in both directions; joint probes evaluate all four combinations.
  The returned dictionary is compatible with the SAReL core loss interface.
  A pre-trained, weight-loaded feature model is required.
  """
  last_conv = [m for m in model.block12.modules() if isinstance(m, torch.nn.Conv1d)][-1]
  full_channels = int(last_conv.out_channels)
  channels_count = full_channels if max_channels <= 0 else min(int(max_channels), full_channels)
  was_training = model.training
  model.eval()
  try:
    mu_mean, mu_std, sigma_mean, sigma_std = compute_channel_mu_stats(
      model, loader, device, full_channels
    )
    D = mu_mean[:channels_count]
    D_mean = float(mu_mean.mean().item())
    rel_strength = (float(sensitivity_eps) * D / max(D_mean, 1e-12)).clamp_min(0.0).float()
    channel_indices = torch.arange(channels_count, device=device)
    V = mean_feat_shift_channel(model, loader, device, channel_indices, rel_strength)

    D_mu = mu_mean[:channels_count].abs()
    D_mu_mean = float(D_mu.mean().item())
    rel_mu = (float(sensitivity_eps) * D_mu / max(D_mu_mean, 1e-12)).float()
    D_sigma = sigma_mean[:channels_count]
    D_sigma_mean = float(D_sigma.mean().item())
    rel_sigma = (float(sensitivity_eps) * D_sigma / max(D_sigma_mean, 1e-12)).float()
    V_mu = mean_feat_shift_channel_mu(model, loader, device, channel_indices, rel_mu)
    V_sigma = mean_feat_shift_channel_sigma(model, loader, device, channel_indices, rel_sigma)
  finally:
    model.train(was_training)

  return {
    "V_feat": V,
    "V_mu_feat": V_mu,
    "V_sigma_feat": V_sigma,
    "mu_mean": mu_mean.detach().cpu().numpy(),
    "mu_std": mu_std.detach().cpu().numpy(),
    "sigma_mean": sigma_mean.detach().cpu().numpy(),
    "sigma_std": sigma_std.detach().cpu().numpy(),
    "rel_strength": rel_strength.detach().cpu().numpy(),
  }

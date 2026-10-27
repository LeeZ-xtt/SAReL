"""Classification and cross-covariance decorrelation objectives for SAReL."""
import torch
import torch.nn as nn
import torch.nn.functional as F
# ---------------------------------------------------------------------------
# Classification Losses
# ---------------------------------------------------------------------------

class LabelSmoothingCrossEntropyLoss(torch.nn.Module):
  """Label smoothing cross-entropy loss.

  Softens one-hot target distribution by alpha, preventing softmax saturation
  and extreme logit magnitude divergence.
  """
  def __init__(self, label_smoothing=0.05):
    super().__init__()
    self.label_smoothing = float(label_smoothing)

  def forward(self, logits, targets):
    return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)


def build_classification_loss(cls_loss_type="label_smoothing", label_smoothing=0.05):
  """Build a classification loss instance by type.

  Supported types:
    - 'ce': Standard torch.nn.CrossEntropyLoss
    - 'label_smoothing': LabelSmoothingCrossEntropyLoss (alpha)
  """
  lowered = str(cls_loss_type).strip().lower()
  if lowered in {"ce", "cross_entropy", "standard"}:
    return torch.nn.CrossEntropyLoss()
  if lowered in {"label_smoothing", "ls"}:
    return LabelSmoothingCrossEntropyLoss(label_smoothing=label_smoothing)
  raise ValueError(f"Unknown classification loss type: {cls_loss_type!r}. Supported: ['ce', 'label_smoothing']")


# ---------------------------------------------------------------------------
# Feature Representation Space Regularization
# ---------------------------------------------------------------------------

class PairwiseCrossCovarianceLoss(nn.Module):
  """Pairwise Cross-Covariance and Decorrelation Loss (CCDL).

  Aligns feature channels between Clean and Adv while suppressing cross-channel
  spurious correlations induced by domain shift.
  """
  def __init__(self, off_diag_weight=0.005):
    super().__init__()
    self.off_diag_weight = float(off_diag_weight)

  def forward(self, features_clean, features_adv):
    """
    Args:
      features_clean: (B, D)
      features_adv: (B, D)
    """
    device = features_clean.device
    B, D = features_clean.size()
    if B <= 1:
      return torch.tensor(0.0, device=device, requires_grad=True)

    # Channel-standardize features across batch: (B, D)
    mean_c = features_clean.mean(dim=0, keepdim=True)
    std_c = features_clean.std(dim=0, keepdim=True) + 1e-6
    h_c = (features_clean - mean_c) / std_c

    mean_a = features_adv.mean(dim=0, keepdim=True)
    std_a = features_adv.std(dim=0, keepdim=True) + 1e-6
    h_a = (features_adv - mean_a) / std_a

    # Cross-correlation matrix: (D, D)
    cross_cov = torch.matmul(h_c.T, h_a) / float(B - 1)

    # Diagonal term: diag(cross_cov) -> 1
    diag = torch.diag(cross_cov)
    loss_diag = ((diag - 1.0) ** 2).mean()

    # Off-diagonal term: offdiag(cross_cov) -> 0
    if self.off_diag_weight > 0.0:
      off_diag = cross_cov - torch.diag(diag)
      loss_off = (off_diag ** 2).mean()
    else:
      loss_off = torch.tensor(0.0, device=device)

    return loss_diag + self.off_diag_weight * loss_off


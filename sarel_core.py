"""SAReL objective and configuration.

Dataset loading, optimizers, checkpointing and evaluation are intentionally
outside this module. ``compute_sarel_loss`` performs a single differentiable
forward computation; the caller is responsible for parameter updates.
"""

import argparse
from dataclasses import asdict, dataclass
import json

import torch
from torch import nn

from feat_attack import attack_feat_stat
from loss import LabelSmoothingCrossEntropyLoss, PairwiseCrossCovarianceLoss


@dataclass(frozen=True)
class SAReLConfig:
    # Published experimental configuration.
    sarel_eps: float = 0.3
    sarel_steps: int = 5
    sarel_early_stop_tau: int = 1
    sarel_attack_eval_mode: bool = True
    sarel_attack_geometry: str = "per_channel_sign"
    sarel_mask_source: str = "vfeat"
    sarel_mask_mode: str = "mu_sigma"
    sarel_perturb_target: str = "mu_sigma"
    sarel_top_ratio: float = 0.7
    sarel_full_continuous: bool = True
    sarel_w_alpha: float = 1.0
    sarel_w_mode: str = "vfeat"
    sarel_w_rand_seed: int = 0
    sarel_lambda_adv: float = 1.0
    sarel_lambda_cov: float = 0.5
    cov_off_diag_weight: float = 0.005
    cls_loss_type: str = "label_smoothing"
    cls_label_smoothing: float = 0.05
    cls_loss_scope: str = "adv_only"

    # Reference experiment settings (not consumed by the loss calculation).
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 5e-4
    grad_clip_norm: float = 5.0
    start_epoch: int = 0
    stop_epoch: int = 50
    seed: int = 0
    head_type: str = "linear"
    head_proj_dim: int = 0


def _bool_arg(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "yes", "true", "on", "y"}:
        return True
    if normalized in {"0", "no", "false", "off", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value: {value!r}")


def make_cli_parser():
    """CLI for inspecting and overriding the standalone core configuration."""
    defaults = SAReLConfig()
    parser = argparse.ArgumentParser(
        description="SAReL core parameter configuration (no training entry point)."
    )
    boolean_fields = {"sarel_attack_eval_mode", "sarel_full_continuous"}
    for name, default in asdict(defaults).items():
        kind = _bool_arg if name in boolean_fields else type(default)
        parser.add_argument(f"--{name}", type=kind, default=default)
    return parser


def config_from_cli(argv=None):
    return SAReLConfig(**vars(make_cli_parser().parse_args(argv)))


def compute_sarel_loss(model, x, y, v_mu, v_sigma, config=None):
    """Compute the two-branch SAReL objective for a preprocessed batch.

    Args:
        model: Network with ``forward_block12``, ``forward_block34_feat`` and
            ``forward_classifier`` as provided by ``DeepINClassifierFeat``.
        x: Preprocessed model input, with shape (B, 1, 1024) for the CNN4 model.
        y: Integer class labels, with shape (B,).
        v_mu, v_sigma: Fixed offline DRS sensitivity spectra, one per channel.
        config: SAReLConfig; the default corresponds to the published setting.

    Returns:
        (total_loss, components) where ``components`` includes tensors for the
        individual losses and logits, as well as attack diagnostics.

    This operation does NOT perform ``backward()``, optimizer updates, or
    checkpoint management. The model should already be in training mode when
    called, as in the original training loop.
    """
    cfg = SAReLConfig() if config is None else config
    if cfg.sarel_mask_source != "vfeat" or cfg.sarel_mask_mode != "mu_sigma":
        raise ValueError("The released core requires separate DRS mean/std spectra.")
    if cfg.sarel_attack_geometry != "per_channel_sign":
        raise ValueError("Only the published per-channel sign update is supported.")
    if cfg.cls_loss_type != "label_smoothing" or cfg.cls_loss_scope != "adv_only":
        raise ValueError("The released core uses label smoothing on the augmented branch only.")
    if v_mu is None or v_sigma is None:
        raise ValueError("Pass both V_mu_feat and V_sigma_feat from DRS.")

    clean_loss_fn = nn.CrossEntropyLoss()
    adv_loss_fn = LabelSmoothingCrossEntropyLoss(
        label_smoothing=cfg.cls_label_smoothing
    )
    cov_loss_fn = PairwiseCrossCovarianceLoss(
        off_diag_weight=cfg.cov_off_diag_weight
    )

    feat12_clean = model.forward_block12(x)
    feat34_clean = model.forward_block34_feat(feat12_clean)
    logits_clean = model.forward_classifier(feat34_clean, y=y)
    clean_loss = clean_loss_fn(logits_clean, y)

    feat12_adv, attack_stats = attack_feat_stat(
        model_feat=model,
        x=x,
        y=y,
        loss_fn=adv_loss_fn,
        feat12_clean=feat12_clean,
        v_feat=v_mu,
        v_mu=v_mu,
        v_sigma=v_sigma,
        eps=cfg.sarel_eps,
        top_ratio=cfg.sarel_top_ratio,
        steps=cfg.sarel_steps,
        early_stop_tau=cfg.sarel_early_stop_tau,
        attack_eval_mode=cfg.sarel_attack_eval_mode,
        attack_geometry=cfg.sarel_attack_geometry,
        full_continuous=cfg.sarel_full_continuous,
        w_alpha=cfg.sarel_w_alpha,
        w_mode=cfg.sarel_w_mode,
        w_rand_seed=cfg.sarel_w_rand_seed,
        perturb_target=cfg.sarel_perturb_target,
    )
    feat34_adv = model.forward_block34_feat(feat12_adv)
    logits_adv = model.forward_classifier(feat34_adv, y=y)
    adv_loss = adv_loss_fn(logits_adv, y)

    total_loss = clean_loss + float(cfg.sarel_lambda_adv) * adv_loss
    cov_loss = None
    if float(cfg.sarel_lambda_cov) > 0.0:
        cov_loss = cov_loss_fn(feat34_clean, feat34_adv)
        total_loss = total_loss + float(cfg.sarel_lambda_cov) * cov_loss

    return total_loss, {
        "clean_loss": clean_loss,
        "adv_loss": adv_loss,
        "cov_loss": cov_loss,
        "logits_clean": logits_clean,
        "logits_adv": logits_adv,
        "attack_stats": attack_stats,
    }


if __name__ == "__main__":
    print(json.dumps(asdict(config_from_cli()), indent=2))

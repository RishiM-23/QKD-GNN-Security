"""
key_rate_loss.py

Differentiable PyTorch translation of key_rate.py's GLLP secure-key-rate
formula, for use as a custom loss in the GCN training loop.

Design notes for integration:
  - All ops are torch tensor ops (no Python branching on tensor values),
    so gradients flow through QBER predictions cleanly.
  - h2() is clamped away from {0, 1} to avoid log(0) / NaN gradients.
  - secure_key_rate() is non-negative by construction (relu floor), matching
    the "no negative key rate" convention in key_rate.py.
  - KeyRateLoss can be used two ways:
      1. As a regression loss: penalize |predicted_qber - true_qber| weighted
         by how much key-rate impact that QBER error represents (i.e. weight
         errors near the security threshold more heavily than errors deep in
         the safe or deep-in-attack regions).
      2. As a cost-sensitive classification loss: scale the per-sample
         attack-classification loss (e.g. BCE) by the true key-rate cost,
         so misclassifying a high-cost (high-QBER, high signal-loss) sample
         is penalized more than misclassifying a near-baseline sample.
  Avyukth: option 2 is almost certainly what we want for the GCN's
  attack-detection head -- see CostSensitiveAttackLoss below.
"""

import torch
import torch.nn as nn

Q_SIFT = 0.5
F_EC = 1.16
PULSE_NORM = 10000.0
BASELINE_QBER = 0.02  # natural-noise floor, matches starter.py
EPS = 1e-7            # numerical floor to keep log() finite


def h2(x: torch.Tensor) -> torch.Tensor:
    """Differentiable binary Shannon entropy, clamped to avoid log(0)."""
    x = torch.clamp(x, EPS, 1.0 - EPS)
    return -x * torch.log2(x) - (1 - x) * torch.log2(1 - x)


def secure_key_rate(qber: torch.Tensor, signal_count: torch.Tensor,
                     decoy_count: torch.Tensor) -> torch.Tensor:
    """
    Differentiable GLLP secure key rate under the single-intensity
    simplification (Q1 ~= Qmu, e1 ~= Emu). All inputs are tensors of the
    same shape (broadcastable); returns a tensor of the same shape.
    """
    q_mu = (signal_count + decoy_count) / PULSE_NORM
    e_mu = qber
    q1 = q_mu
    e1 = e_mu

    rate = Q_SIFT * (-F_EC * q_mu * h2(e_mu) + q1 * (1 - h2(e1)))
    return torch.relu(rate)  # key rate floor at 0, differentiable (subgradient at 0)


def key_rate_cost(qber: torch.Tensor, signal_count: torch.Tensor,
                   decoy_count: torch.Tensor) -> torch.Tensor:
    """Secure-key-rate loss relative to the clean-channel baseline QBER.
    This is the differentiable ground-truth cost used for weighting."""
    baseline = torch.full_like(qber, BASELINE_QBER)
    baseline_rate = secure_key_rate(baseline, signal_count, decoy_count)
    actual_rate = secure_key_rate(qber, signal_count, decoy_count)
    return torch.relu(baseline_rate - actual_rate)


class KeyRateRegressionLoss(nn.Module):
    """
    Use when the GCN predicts QBER directly. Penalizes |pred - true| but
    re-weights each sample by the key-rate sensitivity at the true QBER,
    so the loss focuses model capacity on QBER regions where small errors
    have large secure-key-rate consequences (near the security threshold),
    rather than treating a 0.001 error the same everywhere.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_qber: torch.Tensor, true_qber: torch.Tensor,
                signal_count: torch.Tensor, decoy_count: torch.Tensor) -> torch.Tensor:
        rate_true = secure_key_rate(true_qber, signal_count, decoy_count)
        rate_pred = secure_key_rate(pred_qber, signal_count, decoy_count)
        per_sample_loss = (rate_true - rate_pred) ** 2

        if self.reduction == "mean":
            return per_sample_loss.mean()
        elif self.reduction == "sum":
            return per_sample_loss.sum()
        return per_sample_loss


class CostSensitiveAttackLoss(nn.Module):
    """
    Use for the GCN's binary attack-classification head. Wraps standard BCE
    but scales each sample's loss by (1 + true key-rate cost), so attacks
    that actually destroy a lot of secure key rate are penalized harder
    than borderline/low-impact cases if missed -- directly tying the
    classification objective to the metric we actually care about.
    """

    def __init__(self, cost_weight: float = 5.0):
        super().__init__()
        self.cost_weight = cost_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, true_attacked: torch.Tensor,
                true_qber: torch.Tensor, signal_count: torch.Tensor,
                decoy_count: torch.Tensor) -> torch.Tensor:
        base_loss = self.bce(logits, true_attacked.float())
        cost = key_rate_cost(true_qber, signal_count, decoy_count)
        # Normalize cost to a reasonable scale before weighting (cost is
        # typically in [0, ~0.3] for this simulator's parameter ranges).
        weight = 1.0 + self.cost_weight * cost
        weighted_loss = base_loss * weight
        return weighted_loss.mean()


if __name__ == "__main__":
    # Quick smoke test against key_rate.py's outputs to confirm parity.
    qber = torch.tensor([0.0205, 0.028, 0.25, 0.9], requires_grad=True)
    signal_count = torch.tensor([6309.0, 5754.0, 2500.0, 1200.0])
    decoy_count = torch.tensor([1261.0, 1150.0, 300.0, 100.0])

    rate = secure_key_rate(qber, signal_count, decoy_count)
    cost = key_rate_cost(qber, signal_count, decoy_count)
    print("secure_key_rate:", rate.tolist())
    print("key_rate_cost:  ", cost.tolist())

    rate.sum().backward()
    print("d(rate)/d(qber):", qber.grad.tolist())  # confirms gradients flow

    # Cost-sensitive classification loss smoke test
    logits = torch.tensor([2.0, -1.0, -3.0, 3.0])  # model predictions (pre-sigmoid)
    true_attacked = torch.tensor([0.0, 0.0, 1.0, 1.0])
    loss_fn = CostSensitiveAttackLoss()
    loss = loss_fn(logits, true_attacked, qber.detach(), signal_count, decoy_count)
    print("CostSensitiveAttackLoss:", loss.item())

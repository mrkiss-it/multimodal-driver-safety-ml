"""
Mathematical & Numerical Unit Tests for Custom Imbalanced and Cost-Sensitive Losses.

Course: Máy học nâng cao (Advanced Machine Learning) - UIT, VNU-HCM
Author: Vũ Văn Duy (MSSV: 26410031) - ML Modeling & Evaluation Lead
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.losses import (
    DEFAULT_ASYMMETRIC_COST_MATRIX,
    ClassBalancedFocalLoss,
    CombinedFocalCostLoss,
    CostSensitiveBayesLoss,
    MultiClassFocalLoss,
    bayes_optimal_predict,
)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_focal_loss_gamma_zero_matches_cross_entropy(reduction: str) -> None:
    """When gamma == 0 and alpha is None, Focal Loss must equal standard CrossEntropyLoss."""
    torch.manual_seed(42)
    logits = torch.randn(12, 3, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 0, 0, 1, 2, 0, 1, 0, 2, 0], dtype=torch.long)

    fl = MultiClassFocalLoss(alpha=None, gamma=0.0, reduction=reduction)(logits, targets)
    ce = nn.CrossEntropyLoss(reduction=reduction)(logits, targets)

    assert fl.shape == ce.shape
    assert torch.allclose(fl, ce, atol=1e-10, rtol=1e-10)


def test_focal_loss_alpha_weights_and_batch_mean_semantics() -> None:
    """Verify per-class alpha scaling and non-canceling batch mean on single-class batches."""
    torch.manual_seed(7)
    alpha = [0.15, 0.35, 0.50]
    alpha_t = torch.tensor(alpha, dtype=torch.float64)
    logits = torch.randn(6, 3, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)

    fl_none = MultiClassFocalLoss(alpha=alpha, gamma=0.0, reduction="none")(logits, targets)
    ce_none = F.cross_entropy(logits, targets, weight=alpha_t, reduction="none")
    assert torch.allclose(fl_none, ce_none, atol=1e-10)

    fl_sum = MultiClassFocalLoss(alpha=alpha, gamma=0.0, reduction="sum")(logits, targets)
    ce_sum = F.cross_entropy(logits, targets, weight=alpha_t, reduction="sum")
    assert torch.allclose(fl_sum, ce_sum, atol=1e-10)

    # Critical single-class minority batch check: mean must scale by alpha[2] = 0.50
    minority_targets = torch.tensor([2, 2, 2, 2], dtype=torch.long)
    logits_min = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    unweighted = MultiClassFocalLoss(alpha=None, gamma=2.0, reduction="mean")(
        logits_min, minority_targets
    )
    weighted = MultiClassFocalLoss(alpha=alpha, gamma=2.0, reduction="mean")(
        logits_min, minority_targets
    )
    assert torch.allclose(weighted, 0.50 * unweighted, atol=1e-10)


def test_focal_loss_modulating_factor_suppresses_easy_samples() -> None:
    """Higher gamma must progressively increase the hard-to-easy loss ratio."""
    # Sample 0: easy well-classified Normal (p_0 ~ 0.95)
    # Sample 1: hard misclassified Assault (p_2 ~ 0.08)
    logits = torch.tensor(
        [
            [3.2, -0.5, -0.8],
            [2.1, 0.4, -0.3],
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 2], dtype=torch.long)

    ratios = []
    for gamma in (0.0, 1.0, 2.0, 3.0):
        per_sample = MultiClassFocalLoss(gamma=gamma, reduction="none")(logits, targets)
        easy_loss = float(per_sample[0].item())
        hard_loss = float(per_sample[1].item())
        ratios.append(hard_loss / easy_loss)

    assert ratios[0] < ratios[1] < ratios[2] < ratios[3]
    # At gamma=2.0, closed-form verification: -(1 - p_t)^2 * log(p_t)
    probs = F.softmax(logits, dim=-1)
    p_easy = probs[0, 0]
    expected_easy = -((1.0 - p_easy) ** 2.0) * torch.log(p_easy)
    actual_easy = MultiClassFocalLoss(gamma=2.0, reduction="none")(logits, targets)[0]
    assert torch.allclose(actual_easy, expected_easy, atol=1e-12)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 2.0, 3.0])
def test_focal_loss_numerical_stability_under_extreme_logits(gamma: float) -> None:
    """Extreme saturated logits (p_t -> 1 or p_t -> 0) must not produce NaN/Inf gradients."""
    logits = torch.tensor(
        [
            [100.0, -100.0, -100.0],   # Saturated correct (p_0 == 1.0 in float32)
            [-100.0, -100.0, 100.0],   # Saturated wrong for target=0 (p_0 -> 0)
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([0, 0], dtype=torch.long)

    loss = MultiClassFocalLoss(gamma=gamma, reduction="mean")(logits, targets)
    loss.backward()

    assert torch.isfinite(loss).item()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all().item()


def test_class_balanced_focal_loss_weights_70_20_10() -> None:
    """Verify Cui et al. (2019) effective-number weights on project 70:20:10 distribution."""
    # Project dataset specification (Chapter 5.2): 4200 Normal, 1200 Drowsy, 600 Assault
    cb = ClassBalancedFocalLoss(
        samples_per_class=[4200, 1200, 600], beta=0.999, gamma=2.0
    )
    w = cb.alpha
    assert w.shape == (3,)
    assert float(w[0].item()) < float(w[1].item()) < float(w[2].item())
    assert float(w.sum().item()) == pytest.approx(3.0, abs=1e-5)

    # Limiting case beta == 0.0 -> uniform weights [1.0, 1.0, 1.0]
    cb_zero = ClassBalancedFocalLoss(
        samples_per_class=[4200, 1200, 600], beta=0.0, gamma=2.0
    )
    assert torch.allclose(cb_zero.alpha, torch.ones(3), atol=1e-7)

    logits = torch.randn(5, 3)
    targets = torch.tensor([0, 1, 2, 0, 2])
    assert torch.allclose(
        cb_zero(logits, targets),
        MultiClassFocalLoss(gamma=2.0)(logits, targets),
        atol=1e-6,
    )


def test_cost_sensitive_bayes_loss_asymmetry_and_reductions() -> None:
    """Verify asymmetric penalty C_20=200 >> C_10=50 >> C_01=1 and all reduction modes."""
    criterion_none = CostSensitiveBayesLoss(reduction="none")
    criterion_mean = CostSensitiveBayesLoss(reduction="mean")
    criterion_sum = CostSensitiveBayesLoss(reduction="sum")

    expected_c = torch.tensor(DEFAULT_ASYMMETRIC_COST_MATRIX, dtype=torch.float32)
    assert torch.equal(criterion_none.C, expected_c)
    assert float(criterion_none.C[1, 0].item()) == 50.0
    assert float(criterion_none.C[2, 0].item()) == 200.0

    # Near one-hot predictions:
    # Sample 0: true=0 (Normal), pred~0 (Normal)  -> cost ~ 0
    # Sample 1: true=0 (Normal), pred~1 (Drowsy)  -> False Alarm cost ~ 1
    # Sample 2: true=1 (Drowsy), pred~0 (Normal)  -> Missed Drowsy cost ~ 50
    # Sample 3: true=2 (Assault), pred~0 (Normal) -> Missed Assault cost ~ 200
    logits = torch.tensor(
        [
            [20.0, -20.0, -20.0],
            [-20.0, 20.0, -20.0],
            [20.0, -20.0, -20.0],
            [20.0, -20.0, -20.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1, 2], dtype=torch.long)

    per_sample = criterion_none(logits, targets)
    assert per_sample.shape == (4,)
    assert float(per_sample[0].item()) == pytest.approx(0.0, abs=1e-5)
    assert float(per_sample[1].item()) == pytest.approx(1.0, abs=1e-5)
    assert float(per_sample[2].item()) == pytest.approx(50.0, abs=1e-4)
    assert float(per_sample[3].item()) == pytest.approx(200.0, abs=1e-4)

    assert float(criterion_sum(logits, targets).item()) == pytest.approx(251.0, abs=1e-3)
    assert float(criterion_mean(logits, targets).item()) == pytest.approx(
        251.0 / 4.0, abs=1e-3
    )


def test_combined_focal_cost_loss_non_vanishing_gradient() -> None:
    """Combined L_FL + mu * R_cost prevents Softmax gradient saturation on wrong predictions."""
    # Severely misclassified Assault sample (true=2, predicted strongly as 0)
    logits_pure = torch.tensor([[20.0, -20.0, -20.0]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([2], dtype=torch.long)

    pure_cost = CostSensitiveBayesLoss()(logits_pure, targets)
    pure_cost.backward()
    assert logits_pure.grad is not None
    # Linear expected Softmax cost saturates when p_0 ~ 1, p_2 ~ 0
    assert float(logits_pure.grad.norm().item()) < 1e-10

    logits_comb = torch.tensor([[20.0, -20.0, -20.0]], dtype=torch.float64, requires_grad=True)
    combined = CombinedFocalCostLoss(mu=0.05, gamma=2.0)(logits_comb, targets)
    combined.backward()
    assert logits_comb.grad is not None
    # Logarithmic Focal Loss term preserves strong O(1) corrective gradient!
    assert float(logits_comb.grad.norm().item()) > 1.0

    # Verify mu=0.0 matches pure MultiClassFocalLoss
    s_logits = torch.randn(4, 3, dtype=torch.float64)
    s_targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    assert torch.allclose(
        CombinedFocalCostLoss(mu=0.0, gamma=2.0)(s_logits, s_targets),
        MultiClassFocalLoss(gamma=2.0)(s_logits, s_targets),
        atol=1e-12,
    )


def test_autograd_gradcheck_double_precision() -> None:
    """Analytical PyTorch gradients must match finite-difference Jacobians in float64."""
    torch.manual_seed(99)
    logits = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    losses = [
        MultiClassFocalLoss(alpha=[0.2, 0.3, 0.5], gamma=2.0, reduction="mean"),
        ClassBalancedFocalLoss(samples_per_class=[420, 120, 60], beta=0.99, gamma=1.5),
        CostSensitiveBayesLoss(reduction="mean"),
        CombinedFocalCostLoss(mu=0.05, gamma=2.0, reduction="mean"),
    ]
    for criterion in losses:
        criterion_f64 = criterion.to(dtype=torch.float64)
        assert torch.autograd.gradcheck(
            lambda x: criterion_f64(x, targets),
            (logits,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )


def test_bayes_optimal_predict_decision_shifts() -> None:
    """Verify Bayes optimal inference rule argmin_k sum_j P(y=j|x) C[j, k] (Eq. 4.3)."""
    posteriors = torch.tensor(
        [
            [0.850, 0.050, 0.100],  # 10% Assault -> risk=[22.5, 2.85, 1.95] -> class 2
            [0.900, 0.090, 0.010],  # 9% Drowsy   -> risk=[6.50, 1.10, 2.25] -> class 1
            [0.995, 0.004, 0.001],  # Safe Normal -> risk=[0.40, 1.015, 2.01] -> class 0
        ],
        dtype=torch.float64,
    )

    preds, costs = bayes_optimal_predict(
        posteriors, from_logits=False, return_expected_costs=True
    )
    assert preds.tolist() == [2, 1, 0]

    expected_costs = torch.tensor(
        [
            [22.500, 2.850, 1.950],
            [6.500, 1.100, 2.250],
            [0.400, 1.015, 2.010],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(costs, expected_costs, atol=1e-10)

    # Consistency with from_logits=True via log-probabilities
    log_posteriors = torch.log(posteriors)
    preds_from_logits = bayes_optimal_predict(log_posteriors, from_logits=True)
    assert torch.equal(preds_from_logits, preds)

    # Single-sample 1D input support
    single_pred, single_cost = bayes_optimal_predict(
        posteriors[0], from_logits=False, return_expected_costs=True
    )
    assert single_pred.ndim == 0 and int(single_pred.item()) == 2
    assert single_cost.shape == (3,)


def test_invalid_inputs_raise_clear_exceptions() -> None:
    """Verify defensive input validation across all custom loss classes and helpers."""
    with pytest.raises(ValueError, match="gamma"):
        MultiClassFocalLoss(gamma=-0.5)

    with pytest.raises(ValueError, match="reduction"):
        MultiClassFocalLoss(reduction="invalid")

    with pytest.raises(ValueError, match="alpha"):
        MultiClassFocalLoss(alpha=[0.5, -0.1, 0.6])

    with pytest.raises(ValueError, match="at least 2"):
        ClassBalancedFocalLoss(samples_per_class=[100])

    with pytest.raises(ValueError, match="strictly positive"):
        ClassBalancedFocalLoss(samples_per_class=[4200, 0, 600])

    with pytest.raises(ValueError, match="beta"):
        ClassBalancedFocalLoss(samples_per_class=[4200, 1200, 600], beta=1.0)

    with pytest.raises(ValueError, match="square"):
        CostSensitiveBayesLoss(cost_matrix=[[0.0, 1.0, 2.0], [50.0, 0.0, 5.0]])

    with pytest.raises(ValueError, match="non-negative"):
        CostSensitiveBayesLoss(cost_matrix=[[0.0, -1.0], [10.0, 0.0]])

    with pytest.raises(ValueError, match="mu"):
        CombinedFocalCostLoss(mu=-0.1)

    fl = MultiClassFocalLoss()
    with pytest.raises(ValueError, match="out of bounds"):
        fl(torch.randn(2, 3), torch.tensor([0, 3]))

    with pytest.raises(TypeError, match="integer"):
        fl(torch.randn(2, 3), torch.tensor([0.0, 1.0]))

    with pytest.raises(ValueError, match="sum to 1.0"):
        bayes_optimal_predict(
            torch.tensor([[0.5, 0.5, 0.5]]), from_logits=False
        )

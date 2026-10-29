"""
Custom Loss Functions and Bayes Decision Rules for Extreme Imbalanced
and Cost-Sensitive Multimodal Driver Safety Classification.

Course: Máy học nâng cao (Advanced Machine Learning) - UIT, VNU-HCM
Lecturer: ThS./TS. Đặng Việt Dũng
Author / Maintainer: Vũ Văn Duy (MSSV: 26410031) - ML Modeling & Evaluation Lead
Team: Trần Hoàng Hôn, Nguyễn Duy Khang, Phạm Thành Trung, Vũ Văn Duy, Hoàng Cao Sơn

Mathematical Reference:
    Chapter 4 of `Dinh_Nghia_Bai_Toan_De_Tai_Machine_Learning.md`:
    - Section 4.1: Multi-Class Focal Loss (Lin et al., ICCV 2017)
    - Section 4.2: Class-Balanced Focal Loss (Cui et al., CVPR 2019)
    - Section 4.3: Asymmetric Cost-Sensitive Risk & Bayes Optimal Decision Rule
"""

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_REDUCTIONS: Tuple[str, ...] = ("mean", "sum", "none")

# Asymmetric misclassification cost matrix C[true_class, predicted_class]
# Classes: 0 = Normal, 1 = Drowsiness, 2 = Assault / Extreme Hazard
# Missing Drowsiness (1 -> 0) costs 50; Missing Assault (2 -> 0) costs 200.
DEFAULT_ASYMMETRIC_COST_MATRIX: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 1.0, 2.0),
    (50.0, 0.0, 5.0),
    (200.0, 20.0, 0.0),
)

_PROB_EPS: float = 1e-12


def _validate_reduction(reduction: str) -> None:
    """Validate that a reduction mode is one of 'mean', 'sum', or 'none'."""
    if reduction not in VALID_REDUCTIONS:
        raise ValueError(
            f"Invalid reduction '{reduction}'. Expected one of {VALID_REDUCTIONS}."
        )


def _apply_reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """Apply 'mean', 'sum', or 'none' reduction to a per-sample loss tensor."""
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def _validate_logits_and_targets(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    expected_num_classes: Optional[int] = None,
) -> None:
    """Validate shapes, dtypes, and index bounds for classification losses."""
    if inputs.ndim != 2:
        raise ValueError(
            f"Expected 2D logits tensor of shape (B, C), got shape {tuple(inputs.shape)}."
        )
    if targets.ndim != 1:
        raise ValueError(
            f"Expected 1D targets tensor of shape (B,), got shape {tuple(targets.shape)}."
        )
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Batch size mismatch: inputs has {inputs.shape[0]} samples, "
            f"targets has {targets.shape[0]} samples."
        )
    if targets.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8):
        raise TypeError(
            f"Expected integer class indices for targets, got dtype {targets.dtype}."
        )
    num_classes = inputs.shape[1]
    if expected_num_classes is not None and num_classes != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes, got logits with C={num_classes}."
        )
    if targets.numel() > 0:
        min_t = int(targets.min().item())
        max_t = int(targets.max().item())
        if min_t < 0 or max_t >= num_classes:
            raise ValueError(
                f"Target class indices out of bounds [0, {num_classes - 1}]: "
                f"found range [{min_t}, {max_t}]."
            )


class MultiClassFocalLoss(nn.Module):
    """Multi-Class Focal Loss for addressing extreme class imbalance.

    Implements Section 4.1 (Lin et al., ICCV 2017):
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Note on `'mean'` reduction with `alpha`:
        Following Lin et al. (2017) and Cui et al. (2019), `'mean'` divides the
        batch sum by batch size ``B`` (not by ``sum(alpha_y)`` as in PyTorch's
        ``nn.CrossEntropyLoss(weight=alpha)``). Dividing by ``B`` ensures that
        minority-class gradient scaling does not cancel out when a mini-batch
        contains predominantly one class.

    Args:
        alpha: Optional 1D class weights of length ``C`` (sequence or Tensor).
            Must contain finite non-negative values.
        gamma: Non-negative focusing parameter ``gamma >= 0`` (default ``2.0``).
            When ``gamma == 0.0`` and ``alpha is None``, reduces identically to
            standard ``nn.CrossEntropyLoss``.
        reduction: Reduction mode over the batch: ``'mean'``, ``'sum'``, or ``'none'``.

    Raises:
        ValueError: If ``gamma < 0``, ``reduction`` is invalid, or ``alpha`` is
            not a 1D non-negative tensor/sequence.
    """

    alpha: Optional[torch.Tensor]

    def __init__(
        self,
        alpha: Optional[Union[Sequence[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0.0:
            raise ValueError(f"Focusing parameter gamma must be >= 0.0, got {gamma}.")
        _validate_reduction(reduction)

        self.gamma: float = float(gamma)
        self.reduction: str = reduction

        if alpha is not None:
            alpha_tensor = (
                alpha.detach().clone().to(dtype=torch.float32)
                if isinstance(alpha, torch.Tensor)
                else torch.tensor(list(alpha), dtype=torch.float32)
            )
            if alpha_tensor.ndim != 1 or alpha_tensor.numel() == 0:
                raise ValueError(
                    f"alpha must be a non-empty 1D sequence/tensor, got shape {tuple(alpha_tensor.shape)}."
                )
            if not torch.isfinite(alpha_tensor).all() or (alpha_tensor < 0.0).any():
                raise ValueError("alpha weights must be finite and non-negative.")
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.register_buffer("alpha", None)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multi-class focal loss from raw unnormalized logits.

        Args:
            inputs: Raw logits tensor of shape ``(B, C)``.
            targets: Ground-truth integer class indices of shape ``(B,)``.

        Returns:
            Scalar loss tensor (if ``reduction`` is ``'mean'`` or ``'sum'``) or
            per-sample loss tensor of shape ``(B,)`` (if ``reduction == 'none'``).
        """
        expected_c = int(self.alpha.numel()) if self.alpha is not None else None
        _validate_logits_and_targets(inputs, targets, expected_num_classes=expected_c)
        targets_long = targets.long()

        log_prob = F.log_softmax(inputs, dim=-1)
        target_log_prob = log_prob.gather(
            dim=-1, index=targets_long.unsqueeze(1)
        ).squeeze(1)

        if self.gamma == 0.0:
            loss = -target_log_prob
        else:
            # -expm1(log(p_t)) = 1 - p_t, avoiding catastrophic cancellation near p_t -> 1
            one_minus_pt = (-torch.expm1(target_log_prob)).clamp(min=0.0, max=1.0)
            if self.gamma < 1.0:
                # Avoid infinite derivative d/dx (x^gamma) at x = 0 when 0 < gamma < 1
                one_minus_pt = one_minus_pt.clamp(min=_PROB_EPS)
            modulating_factor = one_minus_pt.pow(self.gamma)
            loss = -modulating_factor * target_log_prob

        if self.alpha is not None:
            alpha_cast = self.alpha.to(device=inputs.device, dtype=inputs.dtype)
            alpha_factor = alpha_cast.gather(dim=0, index=targets_long)
            loss = alpha_factor * loss

        return _apply_reduction(loss, self.reduction)


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Loss Based on Effective Number of Samples (Cui et al., CVPR 2019).

    Implements Section 4.2:
        E_{n_c} = (1 - beta^{n_c}) / (1 - beta)
        w_c     = 1 / E_{n_c}, normalized such that sum_{c=0}^{C-1} w_c = C
        CB_FL   = -w_y * (1 - p_y)^gamma * log(p_y)

    When ``beta == 0.0``, effective sample counts satisfy ``E_{n_c} = 1`` for all
    classes, so normalized weights become ``[1.0, ..., 1.0]`` (unweighted Focal Loss).

    Args:
        samples_per_class: Positive sample counts ``[n_0, n_1, ..., n_{C-1}]`` in
            the training split (e.g., ``[4200, 1200, 600]`` for 70:20:10).
        beta: Hyperparameter ``beta in [0.0, 1.0)`` controlling effective volume
            saturation (default ``0.999``).
        gamma: Focusing parameter ``gamma >= 0.0`` (default ``2.0``).
        reduction: Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.

    Raises:
        ValueError: If ``samples_per_class`` has fewer than 2 classes, contains
            non-positive counts, or ``beta`` is outside ``[0.0, 1.0)``.
    """

    alpha: torch.Tensor

    def __init__(
        self,
        samples_per_class: Sequence[Union[int, float]],
        beta: float = 0.999,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        counts = [float(n) for n in samples_per_class]
        if len(counts) < 2:
            raise ValueError(
                f"samples_per_class must contain at least 2 classes, got {len(counts)}."
            )
        if any(n <= 0.0 for n in counts):
            raise ValueError(
                f"All class sample counts must be strictly positive, got {counts}."
            )
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be in [0.0, 1.0), got {beta}.")

        self.beta: float = float(beta)
        self.gamma: float = float(gamma)
        self.reduction: str = reduction

        num_classes = len(counts)
        if self.beta == 0.0:
            norm_weights = [1.0] * num_classes
        else:
            effective_num = [1.0 - (self.beta ** n) for n in counts]
            raw_weights = [(1.0 - self.beta) / en for en in effective_num]
            total_w = sum(raw_weights)
            norm_weights = [w / total_w * num_classes for w in raw_weights]

        alpha_tensor = torch.tensor(norm_weights, dtype=torch.float32)
        self.register_buffer("alpha", alpha_tensor)
        self.focal_loss = MultiClassFocalLoss(
            alpha=alpha_tensor, gamma=self.gamma, reduction=self.reduction
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute class-balanced focal loss from raw logits.

        Args:
            inputs: Raw logits tensor of shape ``(B, C)``.
            targets: Ground-truth integer class indices of shape ``(B,)``.

        Returns:
            Reduced scalar loss or per-sample loss tensor of shape ``(B,)``.
        """
        return self.focal_loss(inputs, targets)


def _prepare_cost_matrix(
    cost_matrix: Optional[Union[Sequence[Sequence[float]], torch.Tensor]] = None,
) -> torch.Tensor:
    """Convert and validate an asymmetric misclassification cost matrix C."""
    if cost_matrix is None:
        mat = torch.tensor(DEFAULT_ASYMMETRIC_COST_MATRIX, dtype=torch.float32)
    elif isinstance(cost_matrix, torch.Tensor):
        mat = cost_matrix.detach().clone().to(dtype=torch.float32)
    else:
        mat = torch.tensor(cost_matrix, dtype=torch.float32)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] < 2:
        raise ValueError(
            f"cost_matrix must be a square 2D matrix of shape (C, C) with C >= 2, "
            f"got shape {tuple(mat.shape)}."
        )
    if not torch.isfinite(mat).all() or (mat < 0.0).any():
        raise ValueError("cost_matrix entries must be finite and non-negative.")
    return mat


class CostSensitiveBayesLoss(nn.Module):
    """Differentiable Expected Bayes Misclassification Risk Loss.

    Implements Section 4.3 expected cost over predicted Softmax probabilities:
        R_cost(p, y) = sum_{c=0}^{C-1} p_c * C[y, c]

    where ``C[i, j]`` is the penalty incurred when true class is ``i`` and the
    model predicts class ``j`` (default: ``C[1, 0] = 50.0`` for missed Drowsiness,
    ``C[2, 0] = 200.0`` for missed Assault).

    Optimization Note:
        Because ``R_cost`` is linear in Softmax probabilities ``p``, its gradient
        with respect to logits can saturate near zero if a sample is strongly
        misclassified (``p_y -> 0``). For end-to-end gradient training, combine
        this risk term with Focal Loss via :class:`CombinedFocalCostLoss`.

    Args:
        cost_matrix: Optional ``(C, C)`` non-negative cost matrix. Defaults to
            :data:`DEFAULT_ASYMMETRIC_COST_MATRIX`.
        reduction: Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    C: torch.Tensor

    def __init__(
        self,
        cost_matrix: Optional[Union[Sequence[Sequence[float]], torch.Tensor]] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        _validate_reduction(reduction)
        mat = _prepare_cost_matrix(cost_matrix)
        self.register_buffer("C", mat)
        self.reduction: str = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute expected misclassification cost from raw logits.

        Args:
            logits: Raw logits tensor of shape ``(B, C)``.
            targets: Ground-truth integer class indices of shape ``(B,)``.

        Returns:
            Reduced scalar expected cost or per-sample expected cost ``(B,)``.
        """
        _validate_logits_and_targets(
            logits, targets, expected_num_classes=int(self.C.shape[0])
        )
        probs = F.softmax(logits, dim=-1)
        cost_mat = self.C.to(device=logits.device, dtype=logits.dtype)
        target_costs = cost_mat[targets.long()]
        loss = torch.sum(probs * target_costs, dim=-1)
        return _apply_reduction(loss, self.reduction)


class CombinedFocalCostLoss(nn.Module):
    """Composite Objective Combining Focal Loss and Asymmetric Bayes Cost Risk.

    Implements the formal project training objective:
        L_total(logits, y) = L_FL(logits, y) + mu * R_cost(logits, y)

    Combining logarithmic Focal Loss with expected Bayes cost preserves strong
    ``O(1)`` logarithmic gradients when ``p_y -> 0`` while explicitly steering
    decision boundaries away from catastrophic False Negatives (Assault/Drowsy).

    Args:
        mu: Non-negative trade-off weight ``mu >= 0.0`` for expected Bayes cost
            (default ``0.05``).
        alpha: Optional 1D class weights for Focal Loss.
        gamma: Focusing parameter ``gamma >= 0.0`` for Focal Loss (default ``2.0``).
        cost_matrix: Optional ``(C, C)`` cost matrix (defaults to 3x3 project matrix).
        reduction: Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        mu: float = 0.05,
        alpha: Optional[Union[Sequence[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        cost_matrix: Optional[Union[Sequence[Sequence[float]], torch.Tensor]] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if mu < 0.0:
            raise ValueError(f"Trade-off weight mu must be >= 0.0, got {mu}.")
        _validate_reduction(reduction)

        self.mu: float = float(mu)
        self.reduction: str = reduction
        self.focal_loss = MultiClassFocalLoss(
            alpha=alpha, gamma=gamma, reduction=reduction
        )
        self.cost_loss = CostSensitiveBayesLoss(
            cost_matrix=cost_matrix, reduction=reduction
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ``L_FL + mu * R_cost`` on a batch of logits and targets."""
        fl = self.focal_loss(inputs, targets)
        if self.mu == 0.0:
            return fl
        rc = self.cost_loss(inputs, targets)
        return fl + self.mu * rc


def bayes_optimal_predict(
    inputs: torch.Tensor,
    cost_matrix: Optional[Union[Sequence[Sequence[float]], torch.Tensor]] = None,
    from_logits: bool = True,
    return_expected_costs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Predict minimum-risk class labels via the Bayes Optimal Decision Rule.

    Implements Section 4.3 inference decision rule:
        y_hat* = argmin_{k in {0..C-1}} sum_{j=0}^{C-1} C[j, k] * P(y = j | x)

    where ``C[j, k]`` is the cost of true class ``j`` being predicted as ``k``.
    In matrix form for posterior batch ``P`` of shape ``(B, C)``:
        expected_costs = P @ C   # shape (B, C), column k is risk of predicting k
        y_hat*         = argmin(expected_costs, dim=-1)

    Args:
        inputs: Tensor of shape ``(B, C)`` (or ``(C,)`` for a single sample)
            containing raw logits (if ``from_logits=True``) or posterior class
            probabilities ``P(y | x)`` (if ``from_logits=False``).
        cost_matrix: Optional ``(C, C)`` cost matrix ``C[true, pred]``. Defaults
            to :data:`DEFAULT_ASYMMETRIC_COST_MATRIX`.
        from_logits: If ``True``, applies ``softmax`` along the last dimension.
            If ``False``, validates that ``inputs`` are valid probabilities.
        return_expected_costs: If ``True``, returns ``(predictions, expected_costs)``.

    Returns:
        Predicted integer class tensor of shape ``(B,)`` (or scalar ``()`` if 1D
        input was provided), and optionally ``expected_costs`` of matching shape.

    Raises:
        ValueError: If input dimensions/probabilities or ``cost_matrix`` are invalid.
    """
    squeeze_output = False
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(0)
        squeeze_output = True
    elif inputs.ndim != 2:
        raise ValueError(
            f"Expected 1D (C,) or 2D (B, C) inputs, got shape {tuple(inputs.shape)}."
        )

    mat = _prepare_cost_matrix(cost_matrix).to(
        device=inputs.device, dtype=inputs.dtype
    )
    if inputs.shape[1] != mat.shape[0]:
        raise ValueError(
            f"Number of classes in inputs ({inputs.shape[1]}) does not match "
            f"cost_matrix dimension ({mat.shape[0]})."
        )

    if from_logits:
        probs = F.softmax(inputs, dim=-1)
    else:
        if (inputs < -1e-6).any():
            raise ValueError("Probability inputs must be non-negative.")
        row_sums = inputs.sum(dim=-1)
        if not torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3
        ):
            raise ValueError(
                "Probability rows must sum to 1.0 within tolerance 1e-3 when from_logits=False."
            )
        probs = inputs

    expected_costs = probs @ mat  # (B, C): column k = sum_j P(y=j|x) * C[j, k]
    predictions = torch.argmin(expected_costs, dim=-1)

    if squeeze_output:
        predictions = predictions.squeeze(0)
        expected_costs = expected_costs.squeeze(0)

    if return_expected_costs:
        return predictions, expected_costs
    return predictions

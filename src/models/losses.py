"""
Custom Loss Functions for Extreme Imbalanced and Cost-Sensitive Classification
Course: Máy học nâng cao (Advanced Machine Learning)
Lecturer: ThS./TS. Đặng Việt Dũng
Team: Trần Hoàng Hôn, Nguyễn Duy Khang, Phạm Thành Trung, Vũ Văn Duy, Hoàng Cao Sơn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for addressing extreme class imbalance:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        inputs: (B, C) logits
        targets: (B,) class indices
        """
        log_prob = F.log_softmax(inputs, dim=-1)
        prob = torch.exp(log_prob)

        # Gather target prob
        target_log_prob = log_prob.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        target_prob = prob.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        # Modulating factor
        modulating_factor = (1.0 - target_prob) ** self.gamma
        loss = -modulating_factor * target_log_prob

        # Class weights alpha
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_factor = self.alpha.gather(dim=0, index=targets)
            loss = alpha_factor * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Loss Based on Effective Number of Samples (Cui et al., CVPR 2019)
    Combined with Focal Loss
    """
    def __init__(self, samples_per_class, beta=0.999, gamma=2.0, reduction='mean'):
        super(ClassBalancedFocalLoss, self).__init__()
        effective_num = [1.0 - (beta ** n) for n in samples_per_class]
        weights = [(1.0 - beta) / en for en in effective_num]
        total_w = sum(weights)
        norm_weights = [w / total_w * len(samples_per_class) for w in weights]
        
        self.alpha = torch.tensor(norm_weights, dtype=torch.float32)
        self.focal_loss = MultiClassFocalLoss(alpha=self.alpha, gamma=gamma, reduction=reduction)

    def forward(self, inputs, targets):
        return self.focal_loss(inputs, targets)


class CostSensitiveBayesLoss(nn.Module):
    """
    Cost-Sensitive Loss with custom asymmetric misclassification cost matrix C:
    Penalty matrix C where C[i, j] is the cost of predicting class j when actual is i.
    """
    def __init__(self, cost_matrix=None, reduction='mean'):
        super(CostSensitiveBayesLoss, self).__init__()
        if cost_matrix is None:
            # Default asymmetric matrix: Normal=0, Drowsy=1, Assault=2
            self.C = torch.tensor([
                [0.0, 1.0, 2.0],
                [50.0, 0.0, 5.0],
                [200.0, 20.0, 0.0]
            ], dtype=torch.float32)
        else:
            self.C = torch.tensor(cost_matrix, dtype=torch.float32)
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (B, C)
        targets: (B,)
        """
        probs = F.softmax(logits, dim=-1) # (B, C)
        B, num_classes = logits.shape
        if self.C.device != logits.device:
            self.C = self.C.to(logits.device)
            
        target_costs = self.C[targets] # (B, C)
        # Expected risk
        loss = torch.sum(probs * target_costs, dim=-1) # (B,)
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

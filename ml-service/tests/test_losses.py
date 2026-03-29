from __future__ import annotations

import math

import pytest
import torch

from ml_service.models.losses import bpr_loss


class TestBPRLoss:
    def test_pos_greater_than_neg_small_loss(self):
        pos = torch.tensor([5.0, 4.0])
        neg = torch.tensor([1.0, 0.5])
        loss = bpr_loss(pos, neg)
        assert loss.item() < 0.1

    def test_pos_less_than_neg_large_loss(self):
        pos = torch.tensor([0.5, 1.0])
        neg = torch.tensor([5.0, 4.0])
        loss = bpr_loss(pos, neg)
        assert loss.item() > 2.0

    def test_equal_scores(self):
        pos = torch.tensor([2.0, 2.0])
        neg = torch.tensor([2.0, 2.0])
        loss = bpr_loss(pos, neg)
        assert loss.item() == pytest.approx(math.log(2), abs=1e-5)

    def test_gradient_flows(self):
        pos = torch.tensor([2.0], requires_grad=True)
        neg = torch.tensor([1.0], requires_grad=True)
        loss = bpr_loss(pos, neg)
        loss.backward()
        assert pos.grad is not None
        assert neg.grad is not None

    def test_returns_scalar(self):
        pos = torch.tensor([3.0, 2.0, 1.0])
        neg = torch.tensor([1.0, 1.0, 1.0])
        loss = bpr_loss(pos, neg)
        assert loss.dim() == 0

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPFuserTorch(nn.Module):
    def __init__(self, input_dim: int, num_models: int, hidden_dims: Tuple[int, int] = (256, 128)) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)


def train_fuser(model: MLPFuserTorch, x: torch.Tensor, y_soft: torch.Tensor, epochs: int = 200, lr: float = 1e-2, batch_size: int = 64) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = x.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = x[idx]
            yb = y_soft[idx]
            optimizer.zero_grad()
            logp = model(xb)
            loss = F.kl_div(logp, yb, reduction="batchmean")
            loss.backward()
            optimizer.step()


class AttentionFuserTorch(nn.Module):
    def __init__(self, limbo_dim: int, bert_dim: int, num_models: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.limbo_proj = nn.Linear(limbo_dim, hidden_dim)
        self.bert_proj = nn.Linear(bert_dim, hidden_dim)
        self.attn_vec = nn.Parameter(torch.randn(hidden_dim))
        self.head = nn.Linear(hidden_dim, num_models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        limbo_x, bert_x = torch.split(x, [self.limbo_proj.in_features, self.bert_proj.in_features], dim=-1)
        l = torch.tanh(self.limbo_proj(limbo_x))
        b = torch.tanh(self.bert_proj(bert_x))
        w_l = torch.softmax((l * self.attn_vec).sum(dim=-1, keepdim=True), dim=0)
        w_b = torch.softmax((b * self.attn_vec).sum(dim=-1, keepdim=True), dim=0)
        fused = w_l * l + w_b * b
        logits = self.head(fused)
        return F.log_softmax(logits, dim=-1)


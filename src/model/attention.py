"""
Multi-head causal self-attention implemented from scratch using PyTorch primitives.

Each component is kept in its own method so the notebook can inspect
intermediate tensors step by step.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention with a causal (upper-triangular) mask.

    Shapes throughout (using B = batch, T = sequence length, E = embed_dim):
        Q, K, V:  (B, n_heads, T, head_dim)
        scores:   (B, n_heads, T, T)
        weights:  (B, n_heads, T, T)   — after softmax
        context:  (B, n_heads, T, head_dim)
        output:   (B, T, E)
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % n_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # Single projection matrices for Q, K, V and output
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Step-by-step helpers (used by the notebook)
    # ------------------------------------------------------------------

    def make_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Return upper-triangular mask of shape (T, T).
        True where attention is *blocked* (future positions)."""
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, E) → (B, n_heads, T, head_dim)."""
        B, T, E = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def scaled_dot_product(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and context vectors.
        Returns (context, weights) — weights kept for visualisation."""
        scale = math.sqrt(self.head_dim)
        # (B, n_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights) if training else weights
        context = torch.matmul(weights, v)
        return context, weights

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_heads, T, head_dim) → (B, T, E)."""
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, E = x.shape

        # 1. Project to Q, K, V
        qkv = self.qkv_proj(x)                        # (B, T, 3E)
        q, k, v = qkv.split(self.embed_dim, dim=-1)   # each (B, T, E)

        # 2. Split into heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 3. Build causal mask and compute attention
        mask = self.make_causal_mask(T, x.device)
        context, weights = self.scaled_dot_product(q, k, v, mask, training=self.training)

        # 4. Merge heads and project
        out = self.merge_heads(context)           # (B, T, E)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        if return_weights:
            return out, weights
        return out

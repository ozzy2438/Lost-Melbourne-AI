"""
Decoder-only Transformer language model built from scratch with PyTorch primitives.

Architecture (GPT-style):
    Input ids
      → Token embeddings  (vocab_size, embed_dim)
      → Positional embeddings  (context_len, embed_dim)
      → N × TransformerBlock
            LayerNorm
            CausalSelfAttention
            Residual
            LayerNorm
            FeedForward (Linear → GELU → Linear)
            Residual
      → Final LayerNorm
      → Language model head  (embed_dim → vocab_size)
      → logits

No ready-made transformer module is used.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention
from .config import TransformerConfig


class FeedForward(nn.Module):
    """Two-layer feed-forward with GELU activation and dropout."""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU is smoother than ReLU and common in GPT-style models
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """One Transformer decoder block: LayerNorm + Attention + Residual + LN + FF + Residual."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = CausalSelfAttention(cfg.embed_dim, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.embed_dim)
        self.ff = FeedForward(cfg.embed_dim, cfg.ff_dim, cfg.dropout)

    def forward(
        self, x: torch.Tensor, return_attn_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Pre-LayerNorm style (more stable than post-LN)
        if return_attn_weights:
            attn_out, weights = self.attn(self.ln1(x), return_weights=True)
            x = x + attn_out
            x = x + self.ff(self.ln2(x))
            return x, weights
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    """
    Decoder-only Transformer language model.

    Configured entirely through TransformerConfig so experiments can compare
    different sizes without editing source code.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # 1. Token embeddings
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)

        # 2. Positional embeddings (learned, not sinusoidal)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.embed_dim)

        self.emb_dropout = nn.Dropout(cfg.dropout)

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        # 4. Final layer norm
        self.ln_final = nn.LayerNorm(cfg.embed_dim)

        # 5. Language model head (no bias, weight shared with token_emb)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        """Small Gaussian init as in GPT-2."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attn_weights: bool = False,
    ) -> dict:
        """
        Args:
            input_ids: (B, T) long tensor of token ids
            targets:   (B, T) long tensor — if given, loss is computed
            return_attn_weights: collect attention weights from all blocks

        Returns:
            dict with keys: logits, loss (optional), attn_weights (optional)
        """
        B, T = input_ids.shape
        assert T <= self.cfg.context_len, (
            f"Sequence length {T} exceeds context_len {self.cfg.context_len}"
        )

        # Token + positional embeddings
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        tok = self.token_emb(input_ids)   # (B, T, E)
        pos = self.pos_emb(positions)      # (1, T, E)
        x = self.emb_dropout(tok + pos)   # (B, T, E)

        # Transformer blocks
        all_weights = []
        for block in self.blocks:
            if return_attn_weights:
                x, w = block(x, return_attn_weights=True)
                all_weights.append(w)
            else:
                x = block(x)

        # Final norm and projection
        x = self.ln_final(x)             # (B, T, E)
        logits = self.lm_head(x)         # (B, T, vocab_size)

        result: dict = {"logits": logits}

        # Cross-entropy loss: shift so prediction at t predicts token at t+1
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=0,  # PAD_ID
            )
            result["loss"] = loss

        if return_attn_weights:
            result["attn_weights"] = all_weights

        return result

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
        eos_id: int | None = None,
        device: str = "cpu",
    ) -> list[int]:
        """
        Autoregressive generation from a prompt.

        temperature=0 or very low → greedy
        top_k > 0  → sample from top-k logits only
        """
        if seed is not None:
            torch.manual_seed(seed)

        self.eval()
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            # Truncate to context window
            context = ids[-self.cfg.context_len:]
            inp = torch.tensor([context], dtype=torch.long, device=device)
            out = self.forward(inp)
            logits = out["logits"][0, -1, :]  # (vocab_size,)

            if temperature <= 0:
                next_id = int(logits.argmax())
            else:
                logits = logits / temperature
                if top_k > 0:
                    top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_vals[-1]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1))

            ids.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break

        return ids

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.cfg.__dict__,
            "state_dict": self.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "cpu") -> tuple["TinyTransformer", dict]:
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = TransformerConfig(**payload["config"])
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"])
        return model, payload

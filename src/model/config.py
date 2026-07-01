"""Transformer configuration dataclass."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TransformerConfig:
    # Vocabulary
    vocab_size: int = 256           # set after tokenizer is trained
    # Architecture
    n_layers: int = 2
    n_heads: int = 4
    embed_dim: int = 128
    ff_dim: int = 512               # feed-forward hidden dim (usually 4 * embed_dim)
    context_len: int = 128
    dropout: float = 0.1
    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 5000
    warmup_steps: int = 200
    grad_clip: float = 1.0
    val_interval: int = 200
    checkpoint_dir: str = "artifacts/checkpoints"
    # Experiment
    seed: int = 42
    tokenizer_type: str = "bpe"     # "char" or "bpe"
    run_id: str = ""

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TransformerConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def param_summary(self) -> str:
        return (
            f"layers={self.n_layers}  heads={self.n_heads}  "
            f"embed={self.embed_dim}  ff={self.ff_dim}  "
            f"ctx={self.context_len}  vocab={self.vocab_size}"
        )

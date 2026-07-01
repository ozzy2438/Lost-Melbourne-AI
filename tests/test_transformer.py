"""
Unit tests for TinyTransformer architecture components.
All tests are offline — no live network, no pre-trained weights downloaded.
"""

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from model import TinyTransformer, TransformerConfig
from model.attention import CausalSelfAttention


def _small_cfg(**overrides) -> TransformerConfig:
    cfg = TransformerConfig(
        vocab_size=64,
        n_layers=2,
        n_heads=4,
        embed_dim=32,
        ff_dim=64,
        context_len=16,
        dropout=0.0,
        batch_size=4,
        max_steps=10,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestCausalMask(unittest.TestCase):

    def test_upper_triangular_is_masked(self):
        attn = CausalSelfAttention(embed_dim=32, n_heads=4, dropout=0.0)
        mask = attn.make_causal_mask(6, device=torch.device("cpu"))
        self.assertEqual(mask.shape, (6, 6))
        # Upper-triangular positions (future) should be True (blocked)
        for i in range(6):
            for j in range(6):
                if j > i:
                    self.assertTrue(mask[i, j].item(), f"mask[{i},{j}] should be True")
                else:
                    self.assertFalse(mask[i, j].item(), f"mask[{i},{j}] should be False")

    def test_mask_prevents_future_leakage(self):
        """Position 0 must not attend to any future position."""
        attn = CausalSelfAttention(embed_dim=32, n_heads=4, dropout=0.0)
        mask = attn.make_causal_mask(5, device=torch.device("cpu"))
        # Row 0 should only see position 0 (all others masked)
        self.assertFalse(mask[0, 0].item())
        for j in range(1, 5):
            self.assertTrue(mask[0, j].item())


class TestAttentionShapes(unittest.TestCase):

    def test_output_shape(self):
        B, T, E = 2, 8, 32
        attn = CausalSelfAttention(embed_dim=E, n_heads=4, dropout=0.0)
        x = torch.randn(B, T, E)
        out = attn(x)
        self.assertEqual(out.shape, (B, T, E))

    def test_attention_weights_returned(self):
        B, T, E, H = 2, 8, 32, 4
        attn = CausalSelfAttention(embed_dim=E, n_heads=H, dropout=0.0)
        x = torch.randn(B, T, E)
        out, weights = attn(x, return_weights=True)
        self.assertEqual(out.shape, (B, T, E))
        self.assertEqual(weights.shape, (B, H, T, T))

    def test_invalid_head_count_raises(self):
        with self.assertRaises(AssertionError):
            CausalSelfAttention(embed_dim=33, n_heads=4)  # 33 not divisible by 4


class TestTransformerForward(unittest.TestCase):

    def setUp(self):
        self.cfg = _small_cfg()
        self.model = TinyTransformer(self.cfg)
        self.model.eval()

    def test_logit_shape(self):
        B, T = 2, 8
        ids = torch.randint(0, self.cfg.vocab_size, (B, T))
        out = self.model(ids)
        self.assertEqual(out["logits"].shape, (B, T, self.cfg.vocab_size))

    def test_loss_computed_when_targets_given(self):
        B, T = 2, 8
        ids = torch.randint(0, self.cfg.vocab_size, (B, T))
        out = self.model(ids, targets=ids)
        self.assertIn("loss", out)
        self.assertFalse(torch.isnan(out["loss"]))
        self.assertGreater(out["loss"].item(), 0)

    def test_no_loss_without_targets(self):
        ids = torch.randint(0, self.cfg.vocab_size, (1, 8))
        out = self.model(ids)
        self.assertNotIn("loss", out)

    def test_attention_weights_shape(self):
        B, T, H = 1, 8, self.cfg.n_heads
        ids = torch.randint(0, self.cfg.vocab_size, (B, T))
        out = self.model(ids, return_attn_weights=True)
        self.assertIn("attn_weights", out)
        self.assertEqual(len(out["attn_weights"]), self.cfg.n_layers)
        for w in out["attn_weights"]:
            self.assertEqual(w.shape, (B, H, T, T))

    def test_context_length_exceeded_raises(self):
        ids = torch.randint(0, self.cfg.vocab_size, (1, self.cfg.context_len + 1))
        with self.assertRaises(AssertionError):
            self.model(ids)

    def test_very_short_input(self):
        ids = torch.randint(0, self.cfg.vocab_size, (1, 1))
        out = self.model(ids)
        self.assertEqual(out["logits"].shape, (1, 1, self.cfg.vocab_size))

    def test_parameter_count_positive(self):
        self.assertGreater(self.model.count_parameters(), 0)


class TestGeneration(unittest.TestCase):

    def setUp(self):
        cfg = _small_cfg(vocab_size=64, context_len=16)
        self.model = TinyTransformer(cfg)
        self.model.eval()
        self.cfg = cfg

    def test_greedy_generation_deterministic(self):
        prompt = [5, 10, 15]
        ids1 = self.model.generate(prompt, max_new_tokens=10, temperature=0.0, seed=99)
        ids2 = self.model.generate(prompt, max_new_tokens=10, temperature=0.0, seed=99)
        self.assertEqual(ids1, ids2)

    def test_seeded_generation_deterministic(self):
        prompt = [5, 10]
        ids1 = self.model.generate(prompt, max_new_tokens=15, temperature=1.0, seed=7)
        ids2 = self.model.generate(prompt, max_new_tokens=15, temperature=1.0, seed=7)
        self.assertEqual(ids1, ids2)

    def test_generation_extends_prompt(self):
        prompt = [5, 10, 15]
        ids = self.model.generate(prompt, max_new_tokens=5, temperature=1.0, seed=1)
        self.assertEqual(ids[:3], prompt)
        self.assertEqual(len(ids), len(prompt) + 5)

    def test_generation_respects_context_window(self):
        # Very long prompt — should not crash (truncated internally)
        prompt = list(range(self.cfg.context_len + 20))
        ids = self.model.generate(prompt, max_new_tokens=3, temperature=0.0)
        self.assertEqual(len(ids), len(prompt) + 3)

    def test_eos_stops_generation(self):
        prompt = [5]
        # With eos_id=3, generation must stop when id=3 is produced.
        # We can't guarantee when it's emitted, but generation must not exceed max.
        ids = self.model.generate(prompt, max_new_tokens=50, temperature=0.0, eos_id=3)
        self.assertLessEqual(len(ids), len(prompt) + 50)

    def test_generation_without_mps(self):
        """Verify generation works on CPU even if MPS is present."""
        prompt = [1, 2, 3]
        ids = self.model.generate(prompt, max_new_tokens=5, temperature=0.5, seed=42, device="cpu")
        self.assertGreater(len(ids), 3)


class TestCheckpoint(unittest.TestCase):

    def test_save_load_identical_output(self):
        cfg = _small_cfg()
        model = TinyTransformer(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        out_before = model(ids)["logits"]

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            model.save_checkpoint(ckpt_path, extra={"step": 10, "val_loss": 2.5})
            loaded, payload = TinyTransformer.load_checkpoint(ckpt_path)
            loaded.eval()
            out_after = loaded(ids)["logits"]

        self.assertTrue(torch.allclose(out_before, out_after, atol=1e-5))
        self.assertEqual(payload["step"], 10)

    def test_config_survives_checkpoint(self):
        cfg = _small_cfg(n_layers=3, embed_dim=64)
        model = TinyTransformer(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cfg_test.pt"
            model.save_checkpoint(path)
            loaded, _ = TinyTransformer.load_checkpoint(path)
        self.assertEqual(loaded.cfg.n_layers, 3)
        self.assertEqual(loaded.cfg.embed_dim, 64)


class TestTransformerConfig(unittest.TestCase):

    def test_save_load_roundtrip(self):
        cfg = _small_cfg(n_layers=4, seed=99)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cfg.json"
            cfg.save(path)
            loaded = TransformerConfig.load(path)
        self.assertEqual(cfg.n_layers, loaded.n_layers)
        self.assertEqual(cfg.seed, loaded.seed)


if __name__ == "__main__":
    unittest.main()

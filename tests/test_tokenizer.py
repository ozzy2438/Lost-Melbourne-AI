"""
Unit tests for CharTokenizer and BPETokenizer.
All tests are offline — no live network or pre-trained model download.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tokenization import BPETokenizer, CharTokenizer


SAMPLE_TEXT = (
    "The Eastern Market in Melbourne was demolished in 1960. "
    "Flinders Street Station opened in 1905. "
    "Carlton and Fitzroy are inner Melbourne suburbs."
)


class TestCharTokenizer(unittest.TestCase):

    def setUp(self):
        self.tok = CharTokenizer().fit(SAMPLE_TEXT)

    def test_vocab_contains_special_tokens(self):
        for sp in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            self.assertIn(sp, self.tok._char_to_id)

    def test_vocab_size_positive(self):
        self.assertGreater(self.tok.vocab_size, 4)

    def test_encode_decode_round_trip(self):
        ids = self.tok.encode(SAMPLE_TEXT)
        decoded = self.tok.decode(ids)
        # Round trip should recover original text exactly
        self.assertEqual(decoded, SAMPLE_TEXT)

    def test_add_special_tokens(self):
        ids = self.tok.encode("hello", add_special=True)
        self.assertEqual(ids[0], self.tok.bos_id)
        self.assertEqual(ids[-1], self.tok.eos_id)

    def test_unknown_character_becomes_unk_id(self):
        ids = self.tok.encode("τελεία")  # Greek, not in corpus
        self.assertTrue(all(i == self.tok.unk_id for i in ids))

    def test_empty_input(self):
        ids = self.tok.encode("")
        self.assertEqual(ids, [])

    def test_deterministic_vocabulary(self):
        tok2 = CharTokenizer().fit(SAMPLE_TEXT)
        self.assertEqual(self.tok._char_to_id, tok2._char_to_id)

    def test_save_load_consistency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "char_tok.json"
            self.tok.save(path)
            loaded = CharTokenizer.load(path)
        self.assertEqual(self.tok._char_to_id, loaded._char_to_id)
        self.assertEqual(loaded.encode(SAMPLE_TEXT), self.tok.encode(SAMPLE_TEXT))

    def test_tokens_per_word_positive(self):
        tpw = self.tok.tokens_per_word(SAMPLE_TEXT)
        self.assertGreater(tpw, 0)

    def test_compression_ratio_is_one(self):
        self.assertAlmostEqual(self.tok.compression_ratio(SAMPLE_TEXT), 1.0)


class TestBPETokenizer(unittest.TestCase):

    def setUp(self):
        # Use a small number of merges for fast tests
        self.tok = BPETokenizer(num_merges=50).fit(SAMPLE_TEXT)

    def test_vocab_contains_special_tokens(self):
        for sp in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            self.assertIn(sp, self.tok._token_to_id)

    def test_vocab_size_greater_than_char(self):
        # BPE should have at least as many tokens as unique chars + specials
        self.assertGreater(self.tok.vocab_size, 4)

    def test_encode_nonempty(self):
        ids = self.tok.encode(SAMPLE_TEXT)
        self.assertGreater(len(ids), 0)

    def test_encode_fewer_tokens_than_chars(self):
        # BPE compression: fewer tokens than characters
        ids = self.tok.encode(SAMPLE_TEXT)
        self.assertLess(len(ids), len(SAMPLE_TEXT))

    def test_add_special_tokens(self):
        ids = self.tok.encode("test", add_special=True)
        self.assertEqual(ids[0], self.tok.bos_id)
        self.assertEqual(ids[-1], self.tok.eos_id)

    def test_unknown_character_becomes_unk(self):
        ids = self.tok.encode("τελεία日本語")
        self.assertTrue(any(i == self.tok.unk_id for i in ids))

    def test_empty_input(self):
        ids = self.tok.encode("")
        self.assertEqual(ids, [])

    def test_deterministic_vocabulary(self):
        tok2 = BPETokenizer(num_merges=50).fit(SAMPLE_TEXT)
        self.assertEqual(self.tok._token_to_id, tok2._token_to_id)
        self.assertEqual(self.tok._merges, tok2._merges)

    def test_save_load_consistency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bpe_tok.json"
            self.tok.save(path)
            loaded = BPETokenizer.load(path)
        self.assertEqual(self.tok._token_to_id, loaded._token_to_id)
        self.assertEqual(self.tok.encode(SAMPLE_TEXT), loaded.encode(SAMPLE_TEXT))

    def test_decode_produces_text(self):
        ids = self.tok.encode("Melbourne")
        decoded = self.tok.decode(ids)
        self.assertIn("Melbourne", decoded)

    def test_compression_ratio_below_one(self):
        ratio = self.tok.compression_ratio(SAMPLE_TEXT)
        self.assertGreater(ratio, 0)
        self.assertLess(ratio, 1.0)

    def test_tokens_per_word_positive(self):
        tpw = self.tok.tokens_per_word(SAMPLE_TEXT)
        self.assertGreater(tpw, 0)

    def test_load_type_mismatch_raises(self):
        # A char tokenizer file should be rejected by BPETokenizer.load
        char_tok = CharTokenizer().fit(SAMPLE_TEXT)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "char.json"
            char_tok.save(path)
            with self.assertRaises(AssertionError):
                BPETokenizer.load(path)


if __name__ == "__main__":
    unittest.main()

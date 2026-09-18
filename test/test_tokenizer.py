"""Software tests for the WordPiece tokenizer.

Cover the two responsibilities: **learning** a subword vocabulary from a corpus
(the WordPiece training loop) and **applying** it (greedy segmentation, special
tokens, padding/truncation, round-trip decode), plus the hand-off of encoded ids
straight into ``nn.Embedding``.

Run them with::

    pytest test/test_tokenizer.py
"""

import numpy as np

from bert_cpu import engine as cpu
from bert_cpu import nn
from bert_cpu.tokenizer import Tokenizer


# A small corpus where several words share the stem "play" and the suffix "ing",
# so WordPiece training has a reason to learn those as reusable subword units.
CORPUS = [
    "playing played plays player",
    "the player is playing",
    "raining trained running",
    "she is running and playing",
    "a rainy day of playing",
]


def _trained(max_size: int = 40) -> Tokenizer:
    tok = Tokenizer()
    tok.build_vocab(CORPUS, max_size=max_size)
    return tok


# ====================================================================== #
# Special tokens
# ====================================================================== #
def test_special_tokens_have_reserved_ids():
    """The five special tokens map to fixed ids 0-4 (PAD=0 for the Embedding)."""
    tok = Tokenizer()
    assert tok.vocab["[PAD]"] == 0
    assert tok.vocab["[UNK]"] == 1
    assert tok.vocab["[CLS]"] == 2
    assert tok.vocab["[SEP]"] == 3
    assert tok.vocab["[MASK]"] == 4
    # [MASK] must exist for masked-language-model pretraining.
    assert tok.mask_id == 4
    assert tok.pad_id == 0


def test_special_ids_survive_build_vocab():
    """Building a vocab keeps the specials pinned to ids 0-4."""
    tok = _trained()
    for i, name in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]):
        assert tok.vocab[name] == i


# ====================================================================== #
# WordPiece training — subwords are actually learned
# ====================================================================== #
def test_build_vocab_learns_multichar_subwords():
    """Training discovers reusable multi-character stems and ## continuations.

    With a small corpus, WordPiece merges up to whole frequent words; capped
    lower (max_size=40) it stops at genuine subword units. Either way, training
    must produce multi-character pieces of *both* kinds: a bare stem (word-initial)
    and a ``##`` continuation.
    """
    tok = _trained(max_size=40)

    stems = [t for t in tok.vocab if len(t) > 1 and not t.startswith("##")
             and not t.startswith("[")]
    conts = [t for t in tok.vocab if t.startswith("##") and len(t) > 3]
    assert stems, "expected at least one learned multi-char stem (e.g. 'pla')"
    assert conts, "expected at least one learned multi-char ## continuation"

    # The payoff: a word splits into a learned stem + a ## continuation.
    assert tok.tokenize("player") == ["pla", "##yer"]


def test_build_vocab_respects_max_size_and_keeps_base_symbols():
    """Vocab size is capped; every seen character stays representable."""
    tok = _trained(max_size=60)
    assert tok.vocab_size <= 60

    # Every character seen in the corpus is representable — as a bare (word-initial)
    # symbol or a ## continuation — so any in-alphabet word always segments (worst
    # case: single-character pieces), never a whole-word [UNK].
    seen_chars = set("".join(CORPUS).replace(" ", ""))
    for ch in seen_chars:
        assert ch in tok.vocab or "##" + ch in tok.vocab


def test_greedy_match_prefers_longer_unit():
    """The scan takes the longest vocab piece, not a shorter prefix."""
    tok = _trained(max_size=40)
    pieces = tok.tokenize("player")
    # "pla" is a learned unit, so the segmentation starts with it rather than
    # breaking "p", "##l", ... one character at a time.
    assert pieces[0] == "pla"
    assert len(pieces[0]) > 1
    assert "".join(p.replace("##", "") for p in pieces) == "player"


def test_unknown_character_becomes_unk():
    """A character never seen during training makes the whole word [UNK]."""
    tok = _trained()
    assert "ß" not in tok.vocab
    assert tok.tokenize("straße") == ["[UNK]"]


# ====================================================================== #
# Encoding: special tokens, padding, truncation
# ====================================================================== #
def test_encode_wraps_and_pads():
    """encode wraps in [CLS]..[SEP] and right-pads with [PAD]=0 to max_len."""
    tok = _trained()
    ids = tok.encode("the player", max_len=12)

    assert len(ids) == 12
    assert ids[0] == tok.cls_id
    assert tok.sep_id in ids
    # Everything after the [SEP] is padding (id 0).
    sep_pos = ids.index(tok.sep_id)
    assert all(i == tok.pad_id for i in ids[sep_pos + 1:])


def test_encode_without_special_tokens():
    """add_special_tokens=False omits the [CLS]/[SEP] wrappers."""
    tok = _trained()
    ids = tok.encode("the player", add_special_tokens=False)
    assert tok.cls_id not in ids
    assert tok.sep_id not in ids


def test_encode_truncates_to_max_len():
    """A long input is cut to exactly max_len (specials included)."""
    tok = _trained()
    long_text = "playing " * 50
    ids = tok.encode(long_text, max_len=10)
    assert len(ids) == 10
    assert ids[0] == tok.cls_id


# ====================================================================== #
# Decoding
# ====================================================================== #
def test_decode_merges_wordpiece_and_strips_specials():
    """Continuation (##) pieces re-attach; special tokens are dropped."""
    tok = _trained(max_size=40)
    stem_id = tok.vocab["pla"]
    cont_id = tok.vocab["##yer"]
    ids = [tok.cls_id, stem_id, cont_id, tok.sep_id, tok.pad_id]
    assert tok.decode(ids) == "player"


def test_encode_decode_round_trip():
    """decode(encode(text)) reproduces the lowercased, subword-merged text."""
    tok = _trained(max_size=40)
    text = "the player is playing"
    assert tok.decode(tok.encode(text)) == text


# ====================================================================== #
# Hand-off to nn.Embedding
# ====================================================================== #
def test_encoded_ids_feed_embedding():
    """Encoded ids are valid indices into an Embedding sized to the vocab."""
    tok = _trained()
    cpu.set_seed(0)
    emb = nn.Embedding(tok.vocab_size, 8, padding_idx=tok.pad_id)

    ids = tok.encode("the player is playing", max_len=16)
    out = emb(np.array([ids]))                 # (batch=1, seq=16, dim=8)
    assert out.shape == (1, 16, 8)

    # The [PAD] row is the zero vector, so padded positions embed to zeros.
    pad_positions = [k for k, i in enumerate(ids) if i == tok.pad_id]
    assert pad_positions, "expected some padding in a length-16 encode"
    for k in pad_positions:
        assert np.allclose(out.data[0, k], 0.0)

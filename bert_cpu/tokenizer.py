"""Minimal WordPiece tokenizer for feeding text into the encoder.

The tokenizer is the model's **front door**: it turns raw text into the integer
id sequences that ``nn.Embedding`` looks up. It has two jobs that mirror real
BERT:

1. **Learn a subword vocabulary** from a corpus (``build_vocab``). This is a
   compact but genuine implementation of WordPiece training (Schuster &
   Nakajima, 2012): starting from single characters, it repeatedly merges the
   symbol pair with the best *likelihood* score until the vocabulary is full. The
   result is a vocabulary of subword units — a frequent word like ``play`` becomes
   its own token, while a rarer inflection is split into pieces such as
   ``play`` + ``##ing``. The ``##`` prefix marks a *continuation* piece (one that
   attaches to the previous piece with no space).

2. **Apply that vocabulary** to new text (``tokenize`` / ``encode``) with a greedy
   longest-match scan, wrapping the sequence in BERT's special tokens and
   padding/truncating to a fixed length.

Why subwords at all? A fixed word-level vocabulary drowns in out-of-vocabulary
words. WordPiece keeps the vocabulary small *and* can represent any word made of
seen characters, because the worst case is falling back to single-character
pieces rather than a bare ``[UNK]``.

Special tokens (fixed ids, so a saved id sequence is unambiguous)::

    [PAD]  = 0   padding filler (id 0 matches nn.Embedding(padding_idx=0))
    [UNK]  = 1   an unrepresentable token
    [CLS]  = 2   prepended; its final hidden state is the sequence summary
    [SEP]  = 3   appended; marks the end (and separates sentence pairs)
    [MASK] = 4   the masked-language-model target token
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional


class Tokenizer:
    """A small WordPiece tokenizer with BERT's special tokens.

    Handles the vocabulary, the special tokens (``[PAD]``, ``[UNK]``, ``[CLS]``,
    ``[SEP]``, ``[MASK]``), and conversion between text and id sequences (with
    padding/truncation). Call :meth:`build_vocab` on a corpus first (or pass a
    ready-made ``vocab``), then :meth:`encode` / :meth:`decode`.
    """

    # Fixed special tokens, in id order. [PAD] is deliberately id 0 so it lines up
    # with ``nn.Embedding(padding_idx=0)`` (that row stays the zero vector).
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    MASK_TOKEN = "[MASK]"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]

    def __init__(self, vocab: Optional[Dict[str, int]] = None) -> None:
        if vocab is None:
            # Start with only the special tokens; build_vocab extends this.
            self.vocab: Dict[str, int] = {t: i for i, t in enumerate(self.SPECIAL_TOKENS)}
        else:
            # Adopt the given vocab, but guarantee the specials are present so the
            # tokenizer is always usable (missing ones are appended).
            self.vocab = dict(vocab)
            for tok in self.SPECIAL_TOKENS:
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
        self._sync()

    # ------------------------------------------------------------------ #
    # Internal bookkeeping
    # ------------------------------------------------------------------ #
    def _sync(self) -> None:
        """Rebuild the id->token reverse map and cache special-token ids."""
        self.ids_to_tokens: Dict[int, str] = {i: t for t, i in self.vocab.items()}
        self.pad_token = self.PAD_TOKEN
        self.unk_token = self.UNK_TOKEN
        self.cls_token = self.CLS_TOKEN
        self.sep_token = self.SEP_TOKEN
        self.mask_token = self.MASK_TOKEN
        self.pad_id = self.vocab[self.PAD_TOKEN]
        self.unk_id = self.vocab[self.UNK_TOKEN]
        self.cls_id = self.vocab[self.CLS_TOKEN]
        self.sep_id = self.vocab[self.SEP_TOKEN]
        self.mask_id = self.vocab[self.MASK_TOKEN]
        self._special_ids = {
            self.pad_id, self.unk_id, self.cls_id, self.sep_id, self.mask_id
        }

    @staticmethod
    def _basic_tokenize(text: str) -> List[str]:
        """Lowercase and split into words and standalone punctuation.

        A run of alphanumeric characters is one token; every other visible
        character (punctuation) becomes its own token. Whitespace is a separator
        and is dropped. Regex-free so the rule is easy to read and step through.
        """
        tokens: List[str] = []
        buf: List[str] = []
        for ch in text.lower():
            if ch.isalnum():
                buf.append(ch)
            else:
                if buf:
                    tokens.append("".join(buf))
                    buf = []
                if not ch.isspace():
                    tokens.append(ch)      # punctuation is its own token
        if buf:
            tokens.append("".join(buf))
        return tokens

    @staticmethod
    def _word_to_pieces(word: str) -> List[str]:
        """Explode a word into single-symbol WordPiece pieces.

        The first character is bare, every following character carries the ``##``
        continuation marker: ``"cat" -> ['c', '##a', '##t']``. This is the initial
        state each word starts in before any merges are learned.
        """
        return [c if i == 0 else "##" + c for i, c in enumerate(word)]

    @staticmethod
    def _merge_symbols(a: str, b: str) -> str:
        """Combine two adjacent pieces into one.

        ``b`` is always a continuation here, so its ``##`` is stripped before
        joining; ``a`` keeps its own form, so continuation-ness is preserved::

            ('p',   '##l') -> 'pl'     ('##i', '##n') -> '##in'
        """
        return a + (b[2:] if b.startswith("##") else b)

    # ------------------------------------------------------------------ #
    # Vocabulary construction (WordPiece training)
    # ------------------------------------------------------------------ #
    def build_vocab(self, corpus: List[str], max_size: int = 30000) -> None:
        """Learn a WordPiece vocabulary from a corpus of raw text.

        This is the actual WordPiece training loop. Starting from single
        characters, it repeatedly merges the symbol pair with the highest
        *likelihood* score until the vocabulary reaches ``max_size``.

        The score for a candidate pair ``(a, b)`` is::

            score(a, b) = freq(a, b) / (freq(a) * freq(b))

        i.e. how much more often the two pieces occur *together* than their
        individual frequencies would predict. This is WordPiece's criterion and
        it differs from plain BPE (which merges the most *frequent* pair): it
        prefers pairs that are distinctively glued together, which is what turns
        common morphemes like ``##ing`` into their own unit.

        Layout of the final vocabulary: the 5 special tokens (ids 0–4) come first,
        then every base single symbol (so any word of seen characters is always
        representable), then the learned merges in the order they were created,
        until ``max_size`` is hit.

        Parameters
        ----------
        corpus : list of str
            The raw texts to learn from.
        max_size : int
            Maximum total vocabulary size (specials + base symbols + merges).
        """
        # 1. Word frequencies from the pre-tokenized corpus.
        word_freqs: Counter = Counter()
        for text in corpus:
            word_freqs.update(self._basic_tokenize(text))

        # 2. Each word starts as its sequence of single-symbol pieces.
        splits: Dict[str, List[str]] = {
            w: self._word_to_pieces(w) for w in word_freqs
        }

        # Base vocabulary: specials first, then every distinct single symbol.
        vocab: Dict[str, int] = {t: i for i, t in enumerate(self.SPECIAL_TOKENS)}
        for pieces in splits.values():
            for sym in pieces:
                if sym not in vocab:
                    vocab[sym] = len(vocab)

        # 3. Merge loop: recount pairs, merge the best-scoring one, repeat.
        while len(vocab) < max_size:
            sym_freq: Counter = Counter()
            pair_freq: Counter = Counter()
            for word, pieces in splits.items():
                wf = word_freqs[word]
                for i, sym in enumerate(pieces):
                    sym_freq[sym] += wf
                    if i > 0:
                        pair_freq[(pieces[i - 1], sym)] += wf

            if not pair_freq:
                break                      # every word is a single piece already

            # Best pair by the WordPiece likelihood score.
            best_pair = max(
                pair_freq,
                key=lambda p: pair_freq[p] / (sym_freq[p[0]] * sym_freq[p[1]]),
            )
            merged = self._merge_symbols(*best_pair)
            if merged in vocab:
                # Already known (can happen with ties); drop this pair to avoid
                # looping forever without growing the vocab.
                del pair_freq[best_pair]
                if not pair_freq:
                    break
            else:
                vocab[merged] = len(vocab)

            # Rewrite every split so the merged symbol replaces the pair.
            a, b = best_pair
            for word, pieces in splits.items():
                merged_pieces: List[str] = []
                i = 0
                while i < len(pieces):
                    if i < len(pieces) - 1 and pieces[i] == a and pieces[i + 1] == b:
                        merged_pieces.append(merged)
                        i += 2
                    else:
                        merged_pieces.append(pieces[i])
                        i += 1
                splits[word] = merged_pieces

        self.vocab = vocab
        self._sync()

    # ------------------------------------------------------------------ #
    # Applying the vocabulary
    # ------------------------------------------------------------------ #
    def _wordpiece(self, word: str) -> List[str]:
        """Greedy longest-match-first WordPiece segmentation of one word.

        Scans left to right, at each position taking the longest substring that
        is in the vocabulary (with a ``##`` prefix for non-initial pieces). If no
        prefix from some position is in the vocabulary, the whole word is
        unrepresentable and collapses to a single ``[UNK]``.
        """
        pieces: List[str] = []
        start = 0
        n = len(word)
        while start < n:
            end = n
            cur: Optional[str] = None
            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in self.vocab:
                    cur = sub
                    break
                end -= 1
            if cur is None:
                return [self.UNK_TOKEN]    # dead end -> the word is [UNK]
            pieces.append(cur)
            start = end
        return pieces

    def tokenize(self, text: str) -> List[str]:
        """Split text into subword tokens using the learned vocabulary."""
        tokens: List[str] = []
        for word in self._basic_tokenize(text):
            tokens.extend(self._wordpiece(word))
        return tokens

    def encode(
        self, text: str, max_len: Optional[int] = None, add_special_tokens: bool = True
    ) -> List[int]:
        """Convert text into a list of token ids.

        With ``add_special_tokens`` the sequence is wrapped as
        ``[CLS] … [SEP]`` (content truncated to leave room for the two specials).
        With ``max_len`` set, the result is truncated to ``max_len`` and then
        right-padded with ``[PAD]`` (id 0) so every sequence has the same length —
        ready to stack into a batch for ``nn.Embedding``.
        """
        tokens = self.tokenize(text)

        if add_special_tokens:
            if max_len is not None:
                tokens = tokens[: max_len - 2]      # leave room for [CLS], [SEP]
            tokens = [self.CLS_TOKEN] + tokens + [self.SEP_TOKEN]
        elif max_len is not None:
            tokens = tokens[:max_len]

        ids = [self.vocab.get(t, self.unk_id) for t in tokens]

        if max_len is not None:
            ids = ids[:max_len] + [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert a list of token ids back into text.

        Special tokens are dropped, then WordPiece pieces are merged: a ``##``
        piece attaches to the previous token with no space, other tokens are
        space-separated. The result is the (lowercased, subword-merged) text.
        """
        words: List[str] = []
        for i in ids:
            if i in self._special_ids:
                continue
            tok = self.ids_to_tokens.get(i, self.UNK_TOKEN)
            if tok.startswith("##") and words:
                words[-1] += tok[2:]
            else:
                words.append(tok)
        return " ".join(words)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

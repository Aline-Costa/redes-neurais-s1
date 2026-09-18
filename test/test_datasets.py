"""Software tests for the bundled dataset loaders."""

import numpy as np
import pytest

import datasets


def test_load_adult_train_shapes_and_labels():
    """The train split loads with the documented size and clean {0,1} labels."""
    ds = datasets.load_adult("train")

    # 32561 training instances (datasets/adult/adult.names), column-oriented X.
    assert ds.n_samples == 32561
    assert ds.X.shape == (ds.n_features, ds.n_samples)
    assert ds.y.shape == (ds.n_samples,)
    assert len(ds) == ds.n_samples

    # Labels are integers in {0, 1}; the positive rate matches the known ~24%.
    assert set(np.unique(ds.y).tolist()) == {0, 1}
    assert 0.23 < ds.y.mean() < 0.25

    # Every row of X is named, and X is finite (no NaNs from bad parsing).
    assert len(ds.feature_names) == ds.n_features
    assert np.isfinite(ds.X).all()


def test_continuous_features_are_standardised_on_train():
    """The 6 continuous features are z-scored: ~0 mean, ~1 std on train."""
    ds = datasets.load_adult("train")
    continuous = ds.X[:6]                       # continuous block comes first
    assert np.allclose(continuous.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(continuous.std(axis=1), 1.0, atol=1e-9)


def test_test_split_matches_train_feature_space():
    """Test uses the *same* encoder as train, so feature space is identical."""
    train = datasets.load_adult("train")
    test = datasets.load_adult("test")

    assert test.n_samples == 16281            # documented test size
    assert test.n_features == train.n_features
    assert test.feature_names == train.feature_names
    assert set(np.unique(test.y).tolist()) == {0, 1}
    # Test standardisation uses train stats, so its mean is near — but not
    # exactly — zero (that's the point: no leakage from fitting on test).
    assert np.isfinite(test.X).all()


def test_missing_value_is_its_own_category():
    """The '?' missing token is kept as an explicit one-hot column."""
    ds = datasets.load_adult("train")
    assert "?" in ds.categories["workclass"]
    assert "workclass=?" in ds.feature_names


def test_one_hot_blocks_sum_to_one_per_sample():
    """Each categorical field contributes exactly one active one-hot column."""
    ds = datasets.load_adult("train")
    offset = 6                                  # skip the 6 continuous rows
    # categories is ordered the same way the one-hot blocks are laid out in X.
    for field, vocab in ds.categories.items():
        width = len(vocab)
        block = ds.X[offset:offset + width]     # (width, n_samples)
        assert np.allclose(block.sum(axis=0), 1.0), field
        offset += width
    # The continuous block + every one-hot block together fill all of X.
    assert offset == ds.n_features


def test_invalid_split_raises():
    with pytest.raises(ValueError):
        datasets.load_adult("validation")


# ====================================================================== #
# Flatland — the text corpus
# ====================================================================== #
def test_flatland_loads_text_paragraphs_sentences():
    """The corpus loads with non-empty text, paragraphs, and sentences."""
    flat = datasets.load_flatland()

    assert isinstance(flat.text, str) and len(flat.text) > 10000
    assert flat.n_paragraphs > 100
    assert flat.n_sentences > flat.n_paragraphs      # paragraphs split into more
    assert len(flat) == flat.n_sentences             # len() counts sentences
    assert all(isinstance(s, str) and s for s in flat.sentences)


def test_flatland_strips_gutenberg_boilerplate():
    """The Gutenberg header/footer is removed from the cleaned text."""
    flat = datasets.load_flatland()

    # The marker lines and licensing/catalogue text must be gone...
    assert "START OF THE PROJECT GUTENBERG" not in flat.text
    assert "END OF THE PROJECT GUTENBERG" not in flat.text
    assert "www.gutenberg.org" not in flat.text
    # ...while the actual novella is present.
    assert "Flatland" in flat.text


def test_flatland_paragraphs_have_no_illustration_markers():
    """Whitespace is collapsed and pure ``[Illustration]`` markers are dropped."""
    flat = datasets.load_flatland()

    for para in flat.paragraphs:
        assert "\n" not in para                      # internal newlines collapsed
        assert para == para.strip()
        assert para != "[Illustration]"


def test_flatland_feeds_tokenizer():
    """The sentences train the WordPiece tokenizer end to end.

    Uses a slice + small vocab so the (deliberately un-optimised, clarity-first)
    WordPiece training loop stays fast in the test suite.
    """
    from bert_cpu import Tokenizer

    flat = datasets.load_flatland()
    tok = Tokenizer()
    tok.build_vocab(flat.sentences[:150], max_size=300)

    assert tok.vocab_size <= 300
    # A common word from the book round-trips through encode/decode (decode
    # re-merges any ## pieces, so this holds even if "square" is split).
    ids = tok.encode("the square", max_len=8)
    assert ids[0] == tok.cls_id
    assert tok.decode(ids) == "the square"

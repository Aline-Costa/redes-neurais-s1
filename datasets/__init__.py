"""Example datasets bundled with BERT-cpu.

Import this module and call a loader. There are two, one per training task:

**Adult** (tabular classification) — model-ready NumPy arrays::

    import datasets

    train = datasets.load_adult("train")
    test  = datasets.load_adult("test")

    train.X            # (n_features, n_samples) float array — column-oriented
    train.y            # (n_samples,) int labels in {0, 1}  (1 == income >50K)
    train.feature_names
    train.categories

The arrays are plain NumPy, so wrap them in the engine when you want to train::

    from bert_cpu import engine as cpu
    X = cpu.Tensor(train.X)            # already (features, batch)

**Flatland** (text corpus) — cleaned prose for the tokenizer / masked-LM::

    flat = datasets.load_flatland()

    flat.text          # the whole book (str), Gutenberg boilerplate removed
    flat.paragraphs    # list[str], blank-line-separated
    flat.sentences     # list[str], naive .!? split

    from bert_cpu import Tokenizer
    tok = Tokenizer()
    tok.build_vocab(flat.sentences, max_size=2000)   # learn WordPiece from it
"""

from datasets.loaders import AdultDataset, FlatlandDataset, load_adult, load_flatland

__all__ = ["load_adult", "AdultDataset", "load_flatland", "FlatlandDataset"]

from setuptools import setup, find_packages

setup(
    name="bert-cpu",
    version="0.1.0",
    author="Luis",
    description=(
        "A tiny NumPy-backed BERT-style Transformer encoder, "
        "a tensor-valued successor to micrograd."
    ),
    packages=find_packages(),
    install_requires=["numpy"],
    python_requires=">=3.8",
)

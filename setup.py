from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nested-learning",
    version="0.1.0",
    author="Implementation of Behrouz et al. NeurIPS 2025",
    description="Partial implementation of Nested Learning (NeurIPS 2025) - API structure with documented gaps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/erikl2/nested-learning",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "wandb": ["wandb>=0.15.0"],
    },
)
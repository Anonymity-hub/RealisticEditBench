#!/usr/bin/env python3
"""
Setup script for EditBench
"""

from setuptools import setup, find_packages
from pathlib import Path

readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="RealisticEditBench",
    version="1.0.0",
    description="Realistic Code Edit Benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    packages=find_packages(exclude=["tests", "tmp", "logs", "crawled_data", "experiment_results", "patch_histories"]),
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.0.0",
        "tenacity>=8.0.0",
        "python-dotenv>=1.0.0",
        "requests>=2.28.0",
        
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "tiktoken>=0.5.0",
        
        "scikit-learn>=1.2.0",
        "rank-bm25>=0.2.2",
        "editdistance>=0.6.0",  # Note: requires C++ compiler to build
        "nltk>=3.8.0",
        "codebleu>=0.1.0",
        "tree-sitter>=0.22.0,<0.23.0",  # Required by codebleu, must match codebleu's version constraint
        "tree-sitter-python==0.21.0",  # Required by codebleu for Python code parsing, must be compatible with tree-sitter 0.22.x
        
        "docker>=6.0.0",
        "psutil>=5.9.0",
        
        "tqdm>=4.64.0",
        
        "huggingface-hub>=0.16.0",
        
        "pyyaml>=6.0",
    ],
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
        ],
    },
    include_package_data=True,
    zip_safe=False,
)


#!/usr/bin/env python3
"""
GRAFT: Gradient-Aware Fast MaxVol Technique for Dynamic Data Sampling
A PyTorch implementation of smart sampling for efficient deep learning training.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="graft-pytorch",
    version="0.1.7",
    author="Ashish Jha",
    author_email="Ashish.Jha@skoltech.ru",  # Update with your actual email
    description="Gradient-Aware Fast MaxVol Technique for Dynamic Data Sampling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ashishjv1/GRAFT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
        "tracking": [
            "wandb>=0.12.0",
            "eco2ai>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "graft-train=graft.cli:main",
        ],
    },
    keywords="machine-learning, deep-learning, pytorch, data-sampling, gradient-based-sampling",
    project_urls={
        "Bug Reports": "https://github.com/ashishjv1/GRAFT/issues",
        "Source": "https://github.com/ashishjv1/GRAFT",
        "Documentation": "https://github.com/ashishjv1/GRAFT/blob/main/README.md",
    },
    include_package_data=True,
    zip_safe=False,
)
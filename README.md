# GRAFT: Gradient-Aware Fast MaxVol Technique for Dynamic Data Sampling

[![PyPI version](https://badge.fury.io/py/graft-pytorch.svg)](https://badge.fury.io/py/graft-pytorch)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of smart sampling for efficient deep learning training.

## Overview
GRAFT uses gradient information and feature decomposition to select the most informative samples during training, reducing computation time while maintaining model performance.

## Features
- **Smart sample selection** using gradient-based importance scoring
- **Multi-architecture support** (ResNet, ResNeXT, EfficientNet, BERT)
- **Dataset compatibility** (CIFAR10/100, TinyImageNet, Caltech256, Medical datasets)
- **Experiment tracking** with Weights & Biases integration
- **Carbon footprint tracking** with eco2AI
- **Efficient training** with reduced computational overhead

## Installation

### From PyPI (Recommended)
```bash
pip install graft-pytorch
```

### With optional dependencies
```bash
# For experiment tracking
pip install graft-pytorch[tracking]

# For development
pip install graft-pytorch[dev]

# Everything
pip install graft-pytorch[all]
```

### From Source
```bash
git clone https://github.com/ashishjv1/GRAFT.git
cd GRAFT
pip install -e .
```

## Quick Start

### Command Line Interface
```bash
# Install and train with smart sampling
pip install graft-pytorch

# Basic training with GRAFT sampling on CIFAR-10
graft-train \
    --numEpochs=200 \
    --batch_size=128 \
    --device="cuda" \
    --optimizer="sgd" \
    --lr=0.1 \
    --numClasses=10 \
    --dataset="cifar10" \
    --model="resnet18" \
    --fraction=0.5 \
    --select_iter=25 \
    --warm_start
```

### Python API

For end-to-end training, the `graft-train` CLI above is the supported entry point
(`graft.cli:main` builds the loaders, model, config, and trainer for you).

To use the selection primitives directly:
```python
from graft import feature_sel, sample_selection

# Precompute per-batch low-rank features once (SVD or torch backend)
data3 = feature_sel(dataloader, batch_size=128, device="cuda", decomp_type="numpy")

# Select a representative subset for the current model state
selected_indices = sample_selection(
    dataloader, data3, model, model.state_dict(),
    128,            # batch_size
    0.3,            # fraction
    10,             # sel_iter
    200,            # numEpochs
    "cuda",         # device
    "cifar10",      # dataset_name
)
```

## Functionality Overview

### Core Components

#### 1. Smart Sample Selection
- **`sample_selection()`**: Selects most informative samples using gradient-based importance
- **`feature_sel()`**: Performs feature decomposition for efficient sampling
- Reduces training time by 30-50% while maintaining model performance

#### 2. Supported Models
- **Vision Models**: ResNet, ResNeXt, EfficientNet, MobileNet, FashionCNN
- **Language Models**: BERT for sequence classification
- **Custom Models**: Easy integration with any PyTorch model

#### 3. Dataset Support
- **Computer Vision**: CIFAR-10/100, TinyImageNet, Caltech256
- **Medical Imaging**: Integration with MedMNIST datasets  
- **Custom Datasets**: Support for any PyTorch DataLoader

#### 4. Training Features
- **Dynamic Sampling**: Adaptive sample selection during training
- **Warm Starting**: Begin with full dataset, then switch to sampling
- **Experiment Tracking**: Built-in WandB integration
- **Carbon Tracking**: Monitor environmental impact with eco2AI

### Configuration Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `numEpochs` | Training epochs | 200 | Any integer |
| `batch_size` | Batch size | 128 | 32, 64, 128, 256+ |
| `device` | Computing device | "cuda" | "cpu", "cuda" |
| `model` | Model architecture | "resnet18" | "resnet18/50", "resnext", "efficientnet" |
| `fraction` | Data sampling ratio | 0.5 | 0.1 - 1.0 |
| `select_iter` | Reselection frequency | 25 | Any integer |
| `optimizer` | Optimization algorithm | "sgd" | "sgd", "adam" |
| `lr` | Learning rate | 0.1 | 0.001 - 0.1 |
| `warm_start` | Use full data initially | False | True/False |
| `decomp` | Decomposition backend | "numpy" | "numpy", "torch" |

### Performance Benefits

- **Speed**: 30-50% faster training time
- **Memory**: Reduced memory usage through smart sampling
- **Accuracy**: Maintains or improves model performance
- **Efficiency**: Lower carbon footprint and energy consumption

## Package Structure
```
graft-pytorch/
├── graft/
│   ├── __init__.py          # Main package exports
│   ├── cli.py               # graft-train command-line entry point
│   ├── trainer.py           # Training orchestration (ModelTrainer, TrainingConfig)
│   ├── genindices.py        # Sample selection algorithms
│   ├── decompositions.py    # Feature decomposition (SVD + Fast MaxVol)
│   ├── grad_dist.py         # Gradient-distance criterion
│   ├── scheduler.py         # Learning-rate scheduling
│   ├── models/              # Supported architectures
│   │   ├── resnet.py        # ResNet implementations
│   │   ├── ResNeXt.py       # ResNeXt implementations
│   │   ├── efficientnet.py  # EfficientNet models
│   │   └── BERT_model.py    # BERT for classification
│   └── utils/               # Utility functions
│       ├── loader.py        # Dataset loaders
│       └── model_mapper.py  # Model selection
└── examples/                # Usage examples
```

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/ashishjv1/GRAFT.git
cd GRAFT

# Install in development mode
pip install -e .[dev]

# Run linting
flake8 graft/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use GRAFT in your research, please cite our paper:

```bibtex
@misc{jha2025graftgradientawarefastmaxvol,
  title         = {GRAFT: Gradient-Aware Fast MaxVol Technique for Dynamic Data Sampling},
  author        = {Ashish Jha and Anh Huy Phan and Razan Dibo and Valentin Leplat},
  year          = {2025},
  eprint        = {2508.13653},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2508.13653}
}
```

## Acknowledgments

- Built using PyTorch
- Inspired by MaxVol techniques for data sampling
- Special thanks to the open-source community

---

**PyPI Package**: [graft-pytorch](https://pypi.org/project/graft-pytorch/)  
**Paper**: [arXiv:2508.13653](https://arxiv.org/abs/2508.13653)  
**Issues**: [GitHub Issues](https://github.com/ashishjv1/GRAFT/issues)  
**Contact**: [Ashish Jha](mailto:Ashish.Jha@skoltech.ru)



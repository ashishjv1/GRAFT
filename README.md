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
```python
import torch
from graft import ModelTrainer, TrainingConfig
from graft.utils.loader import loader

# Load your dataset
trainloader, valloader, trainset, valset = loader(
    dataset="cifar10",
    trn_batch_size=128,
    val_batch_size=128
)

# Configure training with GRAFT
config = TrainingConfig(
    numEpochs=100,
    batch_size=128,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_name="resnet18",
    dataset_name="cifar10",
    trainloader=trainloader,
    valloader=valloader,
    trainset=trainset,
    optimizer_name="sgd",
    lr=0.1,
    fraction=0.5,         # Use 50% of data per epoch
    selection_iter=25,    # Reselect samples every 25 epochs
    warm_start=True       # Train on full data initially
)

# Train with smart sampling
trainer = ModelTrainer(config, trainloader, valloader, trainset)
train_stats, val_stats = trainer.train()

print(f"Best validation accuracy: {val_stats['best_acc']:.2%}")
```

### Advanced Usage
```python
from graft import feature_sel, sample_selection
import torch.nn as nn

# Custom model and data selection
model = MyCustomModel()
data3 = feature_sel(dataloader, batch_size=128, device="cuda")

# Manual sample selection
selected_indices = sample_selection(
    dataloader, data3, model, model.state_dict(),
    batch_size=128, fraction=0.3, select_iter=10,
    numEpochs=200, device="cuda", dataset="custom"
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
│   ├── trainer.py           # Training orchestration
│   ├── genindices.py        # Sample selection algorithms
│   ├── decompositions.py    # Feature decomposition
│   ├── models/              # Supported architectures
│   │   ├── resnet.py        # ResNet implementations  
│   │   ├── efficientnet.py  # EfficientNet models
│   │   └── BERT_model.py    # BERT for classification
│   └── utils/               # Utility functions
│       ├── loader.py        # Dataset loaders
│       └── model_mapper.py  # Model selection
├── tests/                   # Comprehensive test suite
├── examples/                # Usage examples
└── OIDC_SETUP.md           # Deployment configuration
```

## Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/ashishjv1/GRAFT.git
cd GRAFT

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Run linting
flake8 graft/ tests/
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



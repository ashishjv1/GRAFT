# GRAFT: Gradient-Aware Fast MaxVol Technique for Dynamic Data Sampling

A PyTorch implementation of smart sampling for efficient deep learning training.

## Overview
GRAFT uses gradient information and feature decomposition to select the most informative samples during training, reducing computation time while maintaining model performance.

## Features
- Smart sample selection using gradient-based importance scoring
- Support for multiple architectures (ResNet, ResNeXT, EfficientNet)
- Compatible with major datasets (CIFAR10, CIFAR100, TinyImageNet, Caltech256)
- Experiment tracking with WandB
- Carbon footprint tracking with eco2AI

## Installation
```bash
git clone https://github.com/ashishjv1/GRAFT.git
cd GRAFT
pip install -r requirements.txt
```

## Quick Start
```bash
# Basic usage with CPU
python GRAFT.py \
    --numEpochs=200 \
    --batch_size=200 \
    --device="cpu" \
    --optimizer="sgd" \
    --lr=0.1 \
    --weight_decay=4e-5 \
    --numClasses=10 \
    --dataset="cifar10" \
    --model="resnext" \
    --fraction=0.25 \
    --select_iter=25 \
    --save_pickle \
    --dataset_dir="data10" \
    --decomp="torch"

# Usage with GPU
python GRAFT.py \
    --numEpochs=200 \
    --batch_size=200 \
    --device="cuda" \
    --optimizer="sgd" \
    --lr=0.1 \
    --weight_decay=4e-5 \
    --numClasses=10 \
    --dataset="cifar10" \
    --model="resnext" \
    --fraction=0.25 \
    --select_iter=25 \
    --save_pickle \
    --dataset_dir="data10" \
    --decomp="torch" \

# Usage with GPU and warm-starting  
python GRAFT.py \
    --numEpochs=200 \
    --batch_size=200 \
    --device="cuda" \
    --optimizer="sgd" \
    --lr=0.1 \
    --weight_decay=4e-5 \
    --numClasses=10 \
    --dataset="cifar10" \
    --model="resnext" \
    --fraction=0.25 \
    --select_iter=25 \
    --save_pickle \
    --dataset_dir="data10" \
    --decomp="torch" \
    --warm_start
```

### Key Arguments
- `numEpochs`: Number of training epochs
- `batch_size`: Batch size for training
- `device`: Training device ("cpu" or "cuda")
- `optimizer`: Optimization algorithm ("sgd" or "adam")
- `lr`: Learning rate
- `weight_decay`: Weight decay for regularization
- `model`: Model architecture ("resnet18", "resnext", "efficientnet")
- `fraction`: Fraction of data to select (0-1)
- `select_iter`: Selection interval in epochs
- `decomp`: Decomposition backend ("numpy" or "torch")
- `save_pickle`: Save decomposition results for reuse
- `warm_start`: To warm start for the first few epochs (Normally until the first selection Iteration select_iter) 

## Project Structure
```
GRAFT/
├── models/          # Model architectures
├── utils/           # Utility functions
├── data/           # Data loading and processing
├── configs/        # Configuration files
├── tests/          # Unit tests
└── examples/       # Usage examples
```

## Citation
```bibtex
@article{graft2025,
  title={GRAFT: Gradient-Aware Fast MaxVol Technique for Dynamic Data Sampling},
  author={Ashish Jha},
  year={2025}
}
```

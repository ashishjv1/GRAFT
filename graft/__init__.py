"""
GRAFT: Gradient-Aware Fast MaxVol Technique for Dynamic Data Sampling

A PyTorch implementation of smart sampling for efficient deep learning training.
"""

__version__ = "0.1.7"
__author__ = "Ashish Jha"
__email__ = "ashish.jha@skoltech.ru"

from .trainer import ModelTrainer, TrainingConfig
from .decompositions import feature_sel
from .genindices import sample_selection

__all__ = [
    "ModelTrainer",
    "TrainingConfig", 
    "feature_sel",
    "sample_selection",
]
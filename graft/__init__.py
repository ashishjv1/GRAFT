"""
GRAFT: Gradient-Aware Fast MaxVol Technique for Dynamic Data Sampling

A PyTorch implementation of smart sampling for efficient deep learning training.
"""

__version__ = "1.2.1"
__author__ = "Ashish Jha"
__email__ = "ashish.jha@skoltech.ru"

# Core functionality imports with error handling
try:
    from .decompositions import feature_sel
    from .genindices import sample_selection
    _CORE_AVAILABLE = True
except ImportError as e:
    _CORE_AVAILABLE = False
    import warnings
    warnings.warn(f"Core GRAFT functionality not available: {e}")

# Trainer imports with error handling
try:
    from .trainer import ModelTrainer, TrainingConfig
    _TRAINER_AVAILABLE = True
except ImportError as e:
    _TRAINER_AVAILABLE = False
    import warnings
    warnings.warn(f"GRAFT trainer not available: {e}")
    # Create dummy classes to prevent import errors
    class ModelTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("ModelTrainer not available due to missing dependencies")
    class TrainingConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("TrainingConfig not available due to missing dependencies")

# Build __all__ list conditionally
__all__ = []

if _CORE_AVAILABLE:
    __all__.extend(["feature_sel", "sample_selection"])

if _TRAINER_AVAILABLE:
    __all__.extend(["ModelTrainer", "TrainingConfig"])

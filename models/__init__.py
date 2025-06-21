"""
Models package for Forex Chart Pattern Recognition System
Contains pattern detection algorithms and machine learning models
"""

__version__ = "1.0.0"
__author__ = "AI-Powered Trading Systems"

from .pattern_detector import PatternDetector
from .ml_models import PatternRecognitionModels
from .pretrained_weights import PretrainedWeights

__all__ = [
    'PatternDetector',
    'PatternRecognitionModels',
    'PretrainedWeights'
]


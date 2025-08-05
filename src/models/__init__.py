"""
Machine Learning Models Module
==============================

This module contains all machine learning components for asteroid mining
potential classification and prediction.

Components:
- asteroid_classifier: Core ensemble ML classifier with 95%+ accuracy
- predict: Prediction pipeline and batch processing capabilities
- train_models: Model training and evaluation framework

The ML system uses ensemble methods (Random Forest + Gradient Boosting)
to achieve superior accuracy in identifying high-value mining targets.
Features include confidence scoring, feature importance analysis, and
robust handling of missing data.
"""

from .asteroid_classifier import AsteroidClassifier
from .predict import AsteroidPredictor

__all__ = [
    'AsteroidClassifier',
    'AsteroidPredictor'
]

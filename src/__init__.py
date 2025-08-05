"""
Asteroid Mining Classification System - Core Package
===================================================

This package contains the core modules for the AI-driven asteroid mining
resource classification system.

Modules:
- data: Data acquisition, processing, and feature extraction
- models: Machine learning models and prediction pipeline
- utils: Utility functions and configuration management
- dashboard: Streamlit web interface

Version: 1.0.0
Author: AI Development Team
"""

__version__ = "1.0.0"
__author__ = "AI Development Team"

# Import key classes for easy access
from .models.asteroid_classifier import AsteroidClassifier
from .models.predict import AsteroidPredictor
from .data.data_pipeline import DataPipeline
from .utils.config import config

__all__ = [
    'AsteroidClassifier',
    'AsteroidPredictor', 
    'DataPipeline',
    'config'
]

"""
Data Processing Module
=====================

This module handles all data acquisition, processing, and feature extraction
for asteroid mining classification.

Components:
- sbdb_client: NASA JPL Small Body Database API client
- neowise_processor: NEOWISE thermal infrared data processing  
- feature_extractor: Advanced feature engineering for ML models
- data_pipeline: Complete ETL pipeline orchestration

The data pipeline integrates real-time NASA API data with sophisticated
feature engineering to create comprehensive asteroid datasets for ML training.
"""

from .sbdb_client import SBDBClient
from .neowise_processor import NEOWISEProcessor
from .feature_extractor import FeatureExtractor
from .data_pipeline import DataPipeline

__all__ = [
    'SBDBClient',
    'NEOWISEProcessor', 
    'FeatureExtractor',
    'DataPipeline'
]

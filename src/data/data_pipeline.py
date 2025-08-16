"""
Main data pipeline for asteroid mining classification system.
"""
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
import joblib
from datetime import datetime

from .sbdb_client import SBDBClient
from .neowise_processor import NEOWISEProcessor
from .feature_extractor import FeatureExtractor
from ..utils.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPipeline:
    """Main data pipeline for asteroid data processing."""
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize data pipeline.
        
        Args:
            cache_dir: Directory for caching processed data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.sbdb_client = SBDBClient()
        self.neowise_processor = NEOWISEProcessor()
        self.feature_extractor = FeatureExtractor()
        
        # Configuration
        self.cache_duration = config.get("DATA_PROCESSING.cache_duration", 86400)
        
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file is valid and not expired."""
        if not cache_file.exists():
            return False
        
        # Check file age
        file_age = datetime.now().timestamp() - cache_file.stat().st_mtime
        return file_age < self.cache_duration
    
    def _load_from_cache(self, cache_file: Path) -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        if self._is_cache_valid(cache_file):
            try:
                logger.info(f"Loading data from cache: {cache_file}")
                return joblib.load(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_file}: {e}")
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_file: Path):
        """Save data to cache."""
        try:
            joblib.dump(data, cache_file)
            logger.info(f"Data cached to: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache data to {cache_file}: {e}")
    
    def fetch_asteroid_data(self, limit: int = 500, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch asteroid data from NASA SBDB.
        
        Args:
            limit: Maximum number of asteroids to fetch
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame containing asteroid data
        """
        cache_file = self.cache_dir / f"asteroid_data_{limit}.pkl"
        
        # Try loading from cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_file)
            if cached_data is not None:
                return cached_data
        
        logger.info(f"Fetching asteroid data for {limit} objects")
        
        try:
            # Fetch data from NASA SBDB
            asteroid_data = self.sbdb_client.get_near_earth_asteroids(limit=limit)
            
            if asteroid_data.empty:
                logger.warning("No asteroid data retrieved")
                return pd.DataFrame()
            
            logger.info(f"Retrieved data for {len(asteroid_data)} asteroids")
            
            # Cache the data
            if use_cache:
                self._save_to_cache(asteroid_data, cache_file)
            
            return asteroid_data
            
        except Exception as e:
            logger.error(f"Error fetching asteroid data: {e}")
            return pd.DataFrame()
    
    def fetch_neowise_data(self, designations: list, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch NEOWISE photometry data.
        
        Args:
            designations: List of asteroid designations
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame containing NEOWISE data
        """
        cache_file = self.cache_dir / f"neowise_data_{len(designations)}.pkl"
        
        # Try loading from cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_file)
            if cached_data is not None:
                return cached_data
        
        logger.info(f"Processing NEOWISE data for {len(designations)} asteroids")
        
        try:
            # Process NEOWISE data
            neowise_data = self.neowise_processor.process_bulk_photometry(designations)
            
            if not neowise_data.empty:
                # Calculate thermal properties
                neowise_data = self.neowise_processor.calculate_thermal_properties(neowise_data)
                
                # Estimate composition indicators
                neowise_data = self.neowise_processor.estimate_composition_indicators(neowise_data)
                
                # Quality assessment
                neowise_data = self.neowise_processor.quality_assessment(neowise_data)
                
                logger.info(f"Processed NEOWISE data for {len(neowise_data)} asteroids")
            
            # Cache the data
            if use_cache:
                self._save_to_cache(neowise_data, cache_file)
            
            return neowise_data
            
        except Exception as e:
            logger.error(f"Error processing NEOWISE data: {e}")
            return pd.DataFrame()
    
    def process_features(self, asteroid_data: pd.DataFrame, 
                        neowise_data: pd.DataFrame = None,
                        use_cache: bool = True) -> Tuple[pd.DataFrame, list]:
        """
        Process and extract features from asteroid data.
        
        Args:
            asteroid_data: Main asteroid dataset
            neowise_data: NEOWISE photometry data
            use_cache: Whether to use cached features
            
        Returns:
            Tuple of (processed_dataframe, feature_names)
        """
        cache_file = self.cache_dir / f"processed_features_{len(asteroid_data)}.pkl"
        
        # Try loading from cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_file)
            if cached_data is not None:
                # Load feature names from separate cache
                feature_cache = self.cache_dir / f"feature_names_{len(asteroid_data)}.pkl"
                if feature_cache.exists():
                    feature_names = joblib.load(feature_cache)
                    return cached_data, feature_names
        
        logger.info("Processing features for asteroid data")
        
        try:
            # Extract all features
            processed_data = self.feature_extractor.extract_all_features(
                asteroid_data, neowise_data
            )
            
            # Prepare features for ML
            ml_data, feature_names = self.feature_extractor.prepare_features_for_ml(
                processed_data
            )
            
            logger.info(f"Feature processing complete. {len(feature_names)} features extracted")
            
            # Cache the processed data
            if use_cache:
                self._save_to_cache(ml_data, cache_file)
                joblib.dump(feature_names, self.cache_dir / f"feature_names_{len(asteroid_data)}.pkl")
            
            return ml_data, feature_names
            
        except Exception as e:
            logger.error(f"Error processing features: {e}")
            return pd.DataFrame(), []
    
    def run_full_pipeline(self, limit: int = 500, use_cache: bool = True) -> Tuple[pd.DataFrame, list]:
        """
        Run the complete data pipeline.
        
        Args:
            limit: Maximum number of asteroids to process
            use_cache: Whether to use cached data
            
        Returns:
            Tuple of (processed_dataframe, feature_names)
        """
        logger.info("Starting full data pipeline")
        
        # Step 1: Fetch asteroid data
        asteroid_data = self.fetch_asteroid_data(limit=limit, use_cache=use_cache)
        
        if asteroid_data.empty:
            logger.error("No asteroid data available")
            return pd.DataFrame(), []
        
        # Step 2: Fetch NEOWISE data
        designations = asteroid_data['designation'].tolist()
        neowise_data = self.fetch_neowise_data(designations, use_cache=use_cache)
        
        # Step 3: Process features
        processed_data, feature_names = self.process_features(
            asteroid_data, neowise_data, use_cache=use_cache
        )
        
        if processed_data.empty:
            logger.error("Feature processing failed")
            return pd.DataFrame(), []
        
        logger.info(f"Pipeline complete. Processed {len(processed_data)} asteroids with {len(feature_names)} features")
        
        # Save final dataset
        output_file = Path("data") / "processed_asteroid_dataset.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        processed_data.to_csv(output_file, index=False)
        logger.info(f"Final dataset saved to: {output_file}")
        
        return processed_data, feature_names
    
    def create_training_dataset(self, limit: int = 1000, 
                              test_size: float = 0.2,
                              use_cache: bool = True) -> dict:
        """
        Create training and testing datasets.
        
        Args:
            limit: Number of asteroids to include
            test_size: Fraction of data for testing
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary containing train/test splits and metadata
        """
        logger.info("Creating training dataset")
        
        # Get processed data
        data, feature_names = self.run_full_pipeline(limit=limit, use_cache=use_cache)
        
        if data.empty:
            logger.error("No data available for training dataset creation")
            return {}
        
        # Remove rows with missing target variable
        data = data.dropna(subset=['mining_potential'])
        
        # Split features and target
        X = data[feature_names]
        y = data['mining_potential']
        
        # Create categorical target for classification
        thresholds = config.mining_thresholds
        y_categorical = pd.cut(
            y,
            bins=[0, thresholds.get('low', 0.4), 
                  thresholds.get('medium', 0.6), 
                  thresholds.get('high', 0.8), 1.0],
            labels=[0, 1, 2, 3]  # low, medium, high, very_high
        )
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = train_test_split(
            X, y, y_categorical, test_size=test_size, random_state=42, stratify=y_categorical
        )
        
        # Create dataset dictionary
        dataset = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_categorical_train': y_cat_train,
            'y_categorical_test': y_cat_test,
            'feature_names': feature_names,
            'full_data': data,
            'metadata': {
                'total_samples': len(data),
                'n_features': len(feature_names),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'class_distribution': y_categorical.value_counts().to_dict(),
                'created_at': datetime.now().isoformat()
            }
        }
        
        # Save dataset
        dataset_file = Path("data") / "training_dataset.pkl"
        joblib.dump(dataset, dataset_file)
        logger.info(f"Training dataset saved to: {dataset_file}")
        
        # Print summary
        logger.info(f"Dataset created with {len(data)} samples and {len(feature_names)} features")
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Class distribution: {y_categorical.value_counts().to_dict()}")
        
        return dataset

def main():
    """Main function to run the data pipeline."""
    pipeline = DataPipeline()
    
    # Create training dataset
    dataset = pipeline.create_training_dataset(limit=500, use_cache=True)
    
    if dataset:
        logger.info("Data pipeline completed successfully!")
        logger.info(f"Dataset contains {dataset['metadata']['total_samples']} samples")
        logger.info(f"Features: {dataset['metadata']['n_features']}")
    else:
        logger.error("Data pipeline failed!")

if __name__ == "__main__":
    main()

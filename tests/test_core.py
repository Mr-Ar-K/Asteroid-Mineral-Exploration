"""
Test suite for the asteroid mining classification system.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.data.feature_extractor import FeatureExtractor
from src.utils.config import config
from src.utils.delta_v_calculator import DeltaVCalculator

class TestFeatureExtractor:
    """Test feature extraction functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = FeatureExtractor()
        
        # Sample asteroid data
        self.sample_data = pd.DataFrame({
            'designation': ['2000 SG344', '2019 GT3'],
            'semi_major_axis': [1.0, 1.5],
            'eccentricity': [0.1, 0.3],
            'inclination': [5.0, 10.0],
            'diameter': [0.5, 1.2],
            'absolute_magnitude': [20.0, 18.5],
            'albedo': [0.15, 0.25]
        })
    
    def test_orbital_features(self):
        """Test orbital feature extraction."""
        result = self.extractor.extract_orbital_features(self.sample_data)
        
        assert 'sma_log' in result.columns
        assert 'ecc_squared' in result.columns
        assert len(result) == len(self.sample_data)
    
    def test_physical_features(self):
        """Test physical feature extraction."""
        result = self.extractor.extract_physical_features(self.sample_data)
        
        assert 'diameter_log' in result.columns
        assert 'albedo_log' in result.columns
        assert len(result) == len(self.sample_data)
    
    def test_mining_potential_features(self):
        """Test mining potential feature creation."""
        result = self.extractor.create_mining_potential_features(self.sample_data)
        
        assert 'mining_potential' in result.columns
        assert 'economic_value' in result.columns
        assert all(0 <= score <= 1 for score in result['mining_potential'])

class TestDeltaVCalculator:
    """Test delta-v calculation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = DeltaVCalculator()
    
    def test_orbital_velocity(self):
        """Test orbital velocity calculation."""
        # Earth's orbital velocity should be ~29.78 km/s
        velocity = self.calculator.orbital_velocity(1.0, 1.0)
        assert 29.0 < velocity < 30.5
    
    def test_asteroid_delta_v(self):
        """Test asteroid delta-v calculation."""
        result = self.calculator.calculate_asteroid_delta_v(
            semi_major_axis=1.2,
            eccentricity=0.1,
            inclination=5.0
        )
        
        assert 'total' in result
        assert 'accessibility_score' in result
        assert result['total'] > 0
        assert 0 <= result['accessibility_score'] <= 1
    
    def test_launch_window_analysis(self):
        """Test launch window analysis."""
        result = self.calculator.launch_window_analysis(1.5, 0.2)
        
        assert 'synodic_period_years' in result
        assert 'window_duration_days' in result
        assert result['synodic_period_years'] > 0

class TestConfig:
    """Test configuration management."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        assert config.get('NASA_SBDB_API') is not None
        assert config.get('MODEL_CONFIG') is not None
    
    def test_model_config(self):
        """Test model configuration access."""
        rf_config = config.get_model_config('random_forest')
        assert 'n_estimators' in rf_config
    
    def test_mining_thresholds(self):
        """Test mining potential thresholds."""
        thresholds = config.mining_thresholds
        assert 'high' in thresholds
        assert 'medium' in thresholds
        assert 'low' in thresholds

@pytest.fixture
def sample_asteroid_data():
    """Sample asteroid data for testing."""
    return {
        'designation': '2000 SG344',
        'name': 'Test Asteroid',
        'neo': True,
        'pha': False,
        'diameter': 0.5,
        'absolute_magnitude': 20.0,
        'semi_major_axis': 1.0,
        'eccentricity': 0.1,
        'inclination': 5.0
    }

def test_data_pipeline_integration(sample_asteroid_data):
    """Test data pipeline integration."""
    # This would test the full pipeline with mock data
    df = pd.DataFrame([sample_asteroid_data])
    
    extractor = FeatureExtractor()
    processed = extractor.extract_all_features(df)
    
    assert len(processed) == 1
    assert 'mining_potential' in processed.columns

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

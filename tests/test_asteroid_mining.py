"""
Test suite for asteroid mining classification system.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.sbdb_client import SBDBClient
from src.data.neowise_processor import NEOWISEProcessor
from src.data.feature_extractor import FeatureExtractor
from src.models.asteroid_classifier import AsteroidClassifier
from src.utils.config import config

class TestSBDBClient(unittest.TestCase):
    """Test NASA SBDB client."""
    
    def setUp(self):
        """Set up test client."""
        self.client = SBDBClient()
    
    def test_client_initialization(self):
        """Test client initializes correctly."""
        self.assertIsNotNone(self.client.base_url)
        self.assertIsNotNone(self.client.session)
    
    def test_asteroid_data_parsing(self):
        """Test asteroid data parsing."""
        # Mock data structure similar to SBDB API
        mock_data = {
            'des': '2000 SG344',
            'fullname': '2000 SG344',
            'neo': True,
            'pha': False,
            'H': 24.7,
            'diameter': 0.037,
            'albedo': 0.154,
            'orbit': {
                'a': 0.977,
                'e': 0.067,
                'i': 0.111,
                'q': 0.911,
                'Q': 1.043
            }
        }
        
        parsed = self.client._parse_asteroid_data(mock_data)
        
        self.assertEqual(parsed['designation'], '2000 SG344')
        self.assertTrue(parsed['neo'])
        self.assertFalse(parsed['pha'])
        self.assertEqual(parsed['absolute_magnitude'], 24.7)

class TestNEOWISEProcessor(unittest.TestCase):
    """Test NEOWISE data processor."""
    
    def setUp(self):
        """Set up test processor."""
        self.processor = NEOWISEProcessor()
    
    def test_synthetic_data_generation(self):
        """Test synthetic NEOWISE data generation."""
        data = self.processor._generate_synthetic_neowise_data("2000 SG344")
        
        self.assertIn('designation', data)
        self.assertIn('w1_mag', data)
        self.assertIn('diameter_km', data)
        self.assertIn('albedo', data)
        self.assertGreater(data['diameter_km'], 0)
        self.assertGreater(data['albedo'], 0)
    
    def test_thermal_properties_calculation(self):
        """Test thermal properties calculation."""
        # Create sample photometry data
        data = pd.DataFrame([{
            'designation': '2000 SG344',
            'w1_mag': 15.0,
            'w2_mag': 14.8,
            'w3_mag': 12.5,
            'w4_mag': 9.8
        }])
        
        result = self.processor.calculate_thermal_properties(data)
        
        self.assertIn('w1_w2_color', result.columns)
        self.assertIn('thermal_flux_w3', result.columns)

class TestFeatureExtractor(unittest.TestCase):
    """Test feature extraction."""
    
    def setUp(self):
        """Set up test extractor."""
        self.extractor = FeatureExtractor()
    
    def test_orbital_feature_extraction(self):
        """Test orbital feature extraction."""
        # Create sample data
        data = pd.DataFrame([{
            'designation': '2000 SG344',
            'semi_major_axis': 0.977,
            'eccentricity': 0.067,
            'inclination': 0.111,
            'perihelion_distance': 0.911,
            'aphelion_distance': 1.043
        }])
        
        result = self.extractor.extract_orbital_features(data)
        
        self.assertIn('sma_log', result.columns)
        self.assertIn('ecc_squared', result.columns)
        self.assertIn('orbital_range', result.columns)
    
    def test_physical_feature_extraction(self):
        """Test physical feature extraction."""
        data = pd.DataFrame([{
            'designation': '2000 SG344',
            'absolute_magnitude': 24.7,
            'diameter': 0.037,
            'albedo': 0.154
        }])
        
        result = self.extractor.extract_physical_features(data)
        
        self.assertIn('h_mag_norm', result.columns)
        self.assertIn('diameter_log', result.columns)
        self.assertIn('albedo_log', result.columns)

class TestAsteroidClassifier(unittest.TestCase):
    """Test asteroid classifier."""
    
    def setUp(self):
        """Set up test classifier."""
        self.classifier = AsteroidClassifier()
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIn('random_forest_classifier', self.classifier.models)
        self.assertIn('gradient_boosting_classifier', self.classifier.models)
    
    def test_prediction_format(self):
        """Test prediction output format."""
        # Create sample feature data
        X = pd.DataFrame(np.random.rand(10, 20))
        
        # Mock trained models for testing
        self.classifier.is_trained = True
        
        # This would require actual trained models, so we'll skip for now
        # predictions = self.classifier.predict_mining_potential(X)
        # self.assertIn('categorical_predictions', predictions)
        # self.assertIn('continuous_predictions', predictions)

class TestConfiguration(unittest.TestCase):
    """Test configuration management."""
    
    def test_config_loading(self):
        """Test configuration loads correctly."""
        self.assertIsNotNone(config.get("NASA_SBDB_API"))
        self.assertIsNotNone(config.get("NASA_API_KEY"))
    
    def test_model_config(self):
        """Test model configuration."""
        rf_config = config.get_model_config("random_forest")
        self.assertIsInstance(rf_config, dict)
    
    def test_delta_v_config(self):
        """Test delta-v configuration."""
        delta_v_config = config.get_delta_v_config()
        self.assertIn('earth_departure', delta_v_config)

class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_data_pipeline_integration(self):
        """Test data pipeline components work together."""
        # This would test the full pipeline
        # Skipped for now due to API dependencies
        pass
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline."""
        # This would test from raw data to final prediction
        # Skipped for now due to model training requirements
        pass

def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)

if __name__ == "__main__":
    run_tests()

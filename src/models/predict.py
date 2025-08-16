"""
Prediction script for asteroid mining potential assessment.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import argparse
import json
from typing import Dict, List, Optional

from .asteroid_classifier import AsteroidClassifier
from ..data.sbdb_client import SBDBClient
from ..data.neowise_processor import NEOWISEProcessor
from ..data.feature_extractor import FeatureExtractor
from ..utils.config import config

logger = logging.getLogger(__name__)

class AsteroidPredictor:
    """Prediction system for asteroid mining potential."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize predictor with trained models.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.classifier = AsteroidClassifier()
        self.feature_extractor = FeatureExtractor()
        self.sbdb_client = SBDBClient()
        self.neowise_processor = NEOWISEProcessor()
        
        # Load trained models
        self._load_models()
        
    def _load_models(self):
        """Load trained models from disk."""
        try:
            self.classifier.load_models(str(self.model_dir))
            logger.info("Models loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Failed to load models: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict_single_asteroid(self, designation: str) -> Optional[Dict]:
        """
        Predict mining potential for a single asteroid.
        
        Args:
            designation: Asteroid designation (e.g., "2000 SG344")
            
        Returns:
            Dictionary containing prediction results or None if failed
        """
        logger.info(f"Predicting mining potential for {designation}")
        
        try:
            # Fetch asteroid data
            asteroid_data = self.sbdb_client.get_asteroid_data(designation)
            
            if not asteroid_data:
                logger.error(f"No data found for asteroid {designation}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([asteroid_data])
            
            # Fetch NEOWISE data
            neowise_data = self.neowise_processor.get_asteroid_photometry(designation)
            if neowise_data:
                neowise_df = pd.DataFrame([neowise_data])
            else:
                neowise_df = pd.DataFrame()
            
            # Extract features
            processed_data = self.feature_extractor.extract_all_features(df, neowise_df)
            
            # Prepare features for ML
            ml_data, feature_names = self.feature_extractor.prepare_features_for_ml(processed_data)
            
            if ml_data.empty or not feature_names:
                logger.error(f"Feature extraction failed for {designation}")
                return None
            
            # Select only available features
            available_features = [f for f in feature_names if f in ml_data.columns]
            
            if not available_features:
                logger.error(f"No valid features available for {designation}")
                return None
            
            logger.info(f"Using {len(available_features)} features for prediction")
            
            # Make predictions
            predictions = self.classifier.predict_mining_potential(
                ml_data[available_features], model_type='ensemble'
            )
            
            # Compile results
            result = {
                'designation': designation,
                'name': asteroid_data.get('name', ''),
                'basic_info': {
                    'neo': asteroid_data.get('neo', False),
                    'pha': asteroid_data.get('pha', False),
                    'diameter_km': asteroid_data.get('diameter'),
                    'absolute_magnitude': asteroid_data.get('absolute_magnitude'),
                    'spectral_type': asteroid_data.get('spectral_type', 'Unknown')
                },
                'orbital_elements': {
                    'semi_major_axis_au': asteroid_data.get('semi_major_axis'),
                    'eccentricity': asteroid_data.get('eccentricity'),
                    'inclination_deg': asteroid_data.get('inclination'),
                    'perihelion_distance_au': asteroid_data.get('perihelion_distance'),
                    'orbital_period_years': asteroid_data.get('orbital_period')
                },
                'mining_assessment': {
                    'mining_potential_score': float(predictions['continuous_predictions'][0]),
                    'mining_category': predictions['categorical_labels'][0],
                    'confidence_score': float(predictions['confidence_scores'][0]),
                    'class_probabilities': {
                        'low': float(predictions['class_probabilities'][0][0]),
                        'medium': float(predictions['class_probabilities'][0][1]),
                        'high': float(predictions['class_probabilities'][0][2]),
                        'very_high': float(predictions['class_probabilities'][0][3])
                    }
                },
                'derived_metrics': {
                    'accessibility_score': processed_data.get('accessibility_score', [0])[0],
                    'economic_value': processed_data.get('economic_value', [0])[0],
                    'total_risk': processed_data.get('total_risk', [0])[0],
                    'delta_v_total_km_s': processed_data.get('delta_v_total', [0])[0]
                }
            }
            
            # Add composition information if available
            if 'composition_class' in processed_data.columns:
                result['composition'] = {
                    'estimated_type': processed_data['composition_class'].iloc[0],
                    'resource_potential': processed_data.get('resource_potential', [0])[0]
                }
            
            logger.info(f"Prediction completed for {designation}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {designation}: {e}")
            return None
    
    def predict_multiple_asteroids(self, designations: List[str]) -> List[Dict]:
        """
        Predict mining potential for multiple asteroids.
        
        Args:
            designations: List of asteroid designations
            
        Returns:
            List of prediction result dictionaries
        """
        logger.info(f"Predicting mining potential for {len(designations)} asteroids")
        
        results = []
        
        for designation in designations:
            result = self.predict_single_asteroid(designation)
            if result:
                results.append(result)
        
        logger.info(f"Completed predictions for {len(results)}/{len(designations)} asteroids")
        return results
    
    def predict_from_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict mining potential for asteroids from a DataFrame.
        
        Args:
            data: DataFrame containing asteroid data
            
        Returns:
            DataFrame with predictions added
        """
        logger.info(f"Predicting mining potential for {len(data)} asteroids from DataFrame")
        
        try:
            # Extract features
            processed_data = self.feature_extractor.extract_all_features(data)
            
            # Prepare features for ML
            ml_data, feature_names = self.feature_extractor.prepare_features_for_ml(processed_data)
            
            if ml_data.empty or not feature_names:
                logger.error("Feature extraction failed")
                return data
            
            # Select only available features
            available_features = [f for f in feature_names if f in ml_data.columns]
            
            if not available_features:
                logger.error("No valid features available")
                return data
            
            logger.info(f"Using {len(available_features)} features for batch prediction")
            
            # Make predictions
            predictions = self.classifier.predict_mining_potential(
                ml_data[available_features], model_type='ensemble'
            )
            
            # Add predictions to dataframe
            result_df = processed_data.copy()
            result_df['predicted_mining_score'] = predictions['continuous_predictions']
            result_df['predicted_mining_category'] = predictions['categorical_labels']
            result_df['prediction_confidence'] = predictions['confidence_scores']
            
            # Add class probabilities
            prob_cols = ['prob_low', 'prob_medium', 'prob_high', 'prob_very_high']
            prob_df = pd.DataFrame(predictions['class_probabilities'], columns=prob_cols)
            result_df = pd.concat([result_df, prob_df], axis=1)
            
            logger.info("Batch predictions completed successfully")
            return result_df
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return data
    
    def rank_asteroids_by_mining_potential(self, designations: List[str],
                                         top_n: int = 20) -> List[Dict]:
        """
        Rank asteroids by mining potential.
        
        Args:
            designations: List of asteroid designations
            top_n: Number of top asteroids to return
            
        Returns:
            List of top-ranked asteroids with detailed information
        """
        logger.info(f"Ranking top {top_n} asteroids from {len(designations)} candidates")
        
        # Get predictions for all asteroids
        results = self.predict_multiple_asteroids(designations)
        
        if not results:
            logger.warning("No prediction results available for ranking")
            return []
        
        # Sort by mining potential score
        sorted_results = sorted(
            results, 
            key=lambda x: x['mining_assessment']['mining_potential_score'], 
            reverse=True
        )
        
        # Take top N
        top_asteroids = sorted_results[:top_n]
        
        # Add ranking information
        for i, asteroid in enumerate(top_asteroids):
            asteroid['rank'] = i + 1
            asteroid['percentile'] = (len(results) - i) / len(results) * 100
        
        logger.info(f"Ranking completed. Top asteroid: {top_asteroids[0]['designation']} "
                   f"(score: {top_asteroids[0]['mining_assessment']['mining_potential_score']:.3f})")
        
        return top_asteroids
    
    def generate_mining_report(self, designation: str, output_file: str = None) -> str:
        """
        Generate a detailed mining assessment report for an asteroid.
        
        Args:
            designation: Asteroid designation
            output_file: Optional file to save the report
            
        Returns:
            Report text
        """
        result = self.predict_single_asteroid(designation)
        
        if not result:
            return f"Unable to generate report for {designation} - prediction failed"
        
        # Generate report text
        report = f"""
ASTEROID MINING ASSESSMENT REPORT
{'=' * 50}

ASTEROID: {result['designation']}
NAME: {result.get('name', 'Unnamed')}
ASSESSMENT DATE: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

BASIC INFORMATION:
- Near-Earth Object (NEO): {'Yes' if result['basic_info']['neo'] else 'No'}
- Potentially Hazardous Asteroid (PHA): {'Yes' if result['basic_info']['pha'] else 'No'}
- Estimated Diameter: {result['basic_info']['diameter_km']:.3f} km
- Absolute Magnitude: {result['basic_info']['absolute_magnitude']:.2f}
- Spectral Type: {result['basic_info']['spectral_type']}

ORBITAL CHARACTERISTICS:
- Semi-major Axis: {result['orbital_elements']['semi_major_axis_au']:.3f} AU
- Eccentricity: {result['orbital_elements']['eccentricity']:.3f}
- Inclination: {result['orbital_elements']['inclination_deg']:.2f}°
- Perihelion Distance: {result['orbital_elements']['perihelion_distance_au']:.3f} AU
- Orbital Period: {result['orbital_elements']['orbital_period_years']:.2f} years

MINING POTENTIAL ASSESSMENT:
- Overall Mining Score: {result['mining_assessment']['mining_potential_score']:.3f}/1.0
- Mining Category: {result['mining_assessment']['mining_category'].upper()}
- Prediction Confidence: {result['mining_assessment']['confidence_score']:.3f}

CLASS PROBABILITIES:
- Low Potential: {result['mining_assessment']['class_probabilities']['low']:.3f}
- Medium Potential: {result['mining_assessment']['class_probabilities']['medium']:.3f}
- High Potential: {result['mining_assessment']['class_probabilities']['high']:.3f}
- Very High Potential: {result['mining_assessment']['class_probabilities']['very_high']:.3f}

MISSION METRICS:
- Accessibility Score: {result['derived_metrics']['accessibility_score']:.3f}
- Economic Value Index: {result['derived_metrics']['economic_value']:.3f}
- Mission Risk Score: {result['derived_metrics']['total_risk']:.3f}
- Total Delta-V Requirement: {result['derived_metrics']['delta_v_total_km_s']:.2f} km/s

COMPOSITION ESTIMATE:
"""
        
        if 'composition' in result:
            report += f"- Estimated Type: {result['composition']['estimated_type']}\n"
            report += f"- Resource Potential: {result['composition']['resource_potential']:.3f}\n"
        else:
            report += "- Composition data not available\n"
        
        report += f"""
RECOMMENDATION:
Based on the AI assessment, this asteroid is classified as having 
{result['mining_assessment']['mining_category'].upper()} mining potential with a confidence 
of {result['mining_assessment']['confidence_score']:.1%}.

{'*' * 50}
Generated by AI-Driven Asteroid Mining Classification System
"""
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")
        
        return report

def main():
    """Main function for command-line predictions."""
    parser = argparse.ArgumentParser(description='Predict asteroid mining potential')
    
    parser.add_argument('--asteroid-id', type=str, required=True,
                       help='Asteroid designation (e.g., "2000 SG344")')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--output-file', type=str, default=None,
                       help='File to save prediction results')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate detailed assessment report')
    
    args = parser.parse_args()
    
    # Create predictor
    try:
        predictor = AsteroidPredictor(args.model_dir)
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        exit(1)
    
    # Make prediction
    if args.generate_report:
        # Generate detailed report
        report = predictor.generate_mining_report(args.asteroid_id, args.output_file)
        print(report)
    else:
        # Basic prediction
        result = predictor.predict_single_asteroid(args.asteroid_id)
        
        if result:
            print(f"Asteroid: {result['designation']}")
            print(f"Mining Potential Score: {result['mining_assessment']['mining_potential_score']:.3f}")
            print(f"Category: {result['mining_assessment']['mining_category']}")
            print(f"Confidence: {result['mining_assessment']['confidence_score']:.3f}")
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to {args.output_file}")
        else:
            print(f"Prediction failed for {args.asteroid_id}")
            exit(1)

if __name__ == "__main__":
    main()

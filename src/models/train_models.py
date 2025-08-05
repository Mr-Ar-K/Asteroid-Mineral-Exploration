"""
Model training script for asteroid mining classification.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime
import argparse

from .asteroid_classifier import AsteroidClassifier
from ..data.data_pipeline import DataPipeline
from ..utils.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Comprehensive model training and evaluation system."""
    
    def __init__(self, output_dir: str = "models"):
        """
        Initialize model trainer.
        
        Args:
            output_dir: Directory to save trained models and results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.classifier = AsteroidClassifier()
        self.training_results = {}
        self.evaluation_results = {}
        
    def load_training_data(self, data_file: str = None) -> dict:
        """
        Load training data from file or create new dataset.
        
        Args:
            data_file: Path to existing training dataset file
            
        Returns:
            Dictionary containing training data
        """
        if data_file and Path(data_file).exists():
            logger.info(f"Loading existing training data from {data_file}")
            return joblib.load(data_file)
        else:
            logger.info("Creating new training dataset")
            pipeline = DataPipeline()
            return pipeline.create_training_dataset(limit=1000, use_cache=True)
    
    def train_models(self, dataset: dict, perform_tuning: bool = False) -> dict:
        """
        Train all models on the dataset.
        
        Args:
            dataset: Training dataset dictionary
            perform_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary containing training results
        """
        logger.info("Starting model training")
        
        # Extract data
        X_train = dataset['X_train']
        X_test = dataset['X_test']
        y_train_continuous = dataset['y_train']
        y_test_continuous = dataset['y_test']
        y_train_categorical = dataset['y_categorical_train']
        y_test_categorical = dataset['y_categorical_test']
        feature_names = dataset['feature_names']
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Features: {len(feature_names)}")
        
        # Hyperparameter tuning (optional)
        if perform_tuning:
            logger.info("Performing hyperparameter tuning")
            
            # Tune Random Forest
            rf_tuning = self.classifier.hyperparameter_tuning(
                X_train, y_train_categorical, 'random_forest', cv_folds=3
            )
            
            # Tune Gradient Boosting
            gb_tuning = self.classifier.hyperparameter_tuning(
                X_train, y_train_categorical, 'gradient_boosting', cv_folds=3
            )
            
            # Update model configuration with best parameters
            self.classifier.model_config['random_forest'].update(rf_tuning['best_params'])
            self.classifier.model_config['gradient_boosting'].update(gb_tuning['best_params'])
            
            # Reinitialize models with tuned parameters
            self.classifier._initialize_models()
            
            # Save tuning results
            tuning_results = {
                'random_forest': rf_tuning,
                'gradient_boosting': gb_tuning
            }
            
            tuning_file = self.output_dir / "hyperparameter_tuning_results.json"
            with open(tuning_file, 'w') as f:
                json.dump(tuning_results, f, indent=2, default=str)
            
            logger.info(f"Hyperparameter tuning results saved to {tuning_file}")
        
        # Train all models
        training_results = self.classifier.train_all_models(
            X_train, y_train_continuous, y_train_categorical,
            X_test, y_test_continuous, y_test_categorical
        )
        
        # Evaluate models
        evaluation_results = self.classifier.evaluate_models(
            X_test, y_test_continuous, y_test_categorical
        )
        
        # Get feature importance
        feature_importance = self.classifier.get_feature_importance(feature_names)
        
        # Compile results
        all_results = {
            'training_metrics': training_results,
            'evaluation_metrics': evaluation_results,
            'feature_importance': feature_importance,
            'dataset_info': dataset['metadata'],
            'training_timestamp': datetime.now().isoformat()
        }
        
        self.training_results = all_results
        
        logger.info("Model training completed successfully")
        return all_results
    
    def save_results(self, results: dict):
        """
        Save training results and models.
        
        Args:
            results: Training results dictionary
        """
        # Save models
        self.classifier.save_models(str(self.output_dir))
        
        # Save training results
        results_file = self.output_dir / "training_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._convert_for_json(results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"Training results saved to {results_file}")
        
        # Save feature importance as CSV for easy viewing
        self._save_feature_importance_csv(results['feature_importance'])
        
        # Generate summary report
        self._generate_summary_report(results)
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _save_feature_importance_csv(self, feature_importance: dict):
        """Save feature importance rankings as CSV files."""
        for model_name, importance_list in feature_importance.items():
            df = pd.DataFrame(importance_list, columns=['feature', 'importance'])
            csv_file = self.output_dir / f"feature_importance_{model_name}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Feature importance for {model_name} saved to {csv_file}")
    
    def _generate_summary_report(self, results: dict):
        """Generate a summary report of training results."""
        report_file = self.output_dir / "training_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("ASTEROID MINING CLASSIFICATION - TRAINING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Dataset information
            dataset_info = results['dataset_info']
            f.write("DATASET INFORMATION:\n")
            f.write(f"Total samples: {dataset_info['total_samples']}\n")
            f.write(f"Number of features: {dataset_info['n_features']}\n")
            f.write(f"Training samples: {dataset_info['train_samples']}\n")
            f.write(f"Test samples: {dataset_info['test_samples']}\n")
            f.write(f"Class distribution: {dataset_info['class_distribution']}\n\n")
            
            # Classification results
            f.write("CLASSIFICATION RESULTS:\n")
            classification_metrics = results['evaluation_metrics']
            
            models = ['random_forest_classifier', 'gradient_boosting_classifier', 'ensemble_classifier']
            for model in models:
                accuracy_key = f'{model}_accuracy'
                if accuracy_key in classification_metrics:
                    accuracy = classification_metrics[accuracy_key]
                    f.write(f"{model.replace('_', ' ').title()}: {accuracy:.4f}\n")
            
            f.write("\n")
            
            # Regression results
            f.write("REGRESSION RESULTS (R² Score):\n")
            regressors = ['random_forest_regressor', 'gradient_boosting_regressor']
            for model in regressors:
                r2_key = f'{model}_r2'
                if r2_key in classification_metrics:
                    r2 = classification_metrics[r2_key]
                    f.write(f"{model.replace('_', ' ').title()}: {r2:.4f}\n")
            
            f.write("\n")
            
            # Top features
            f.write("TOP 10 MOST IMPORTANT FEATURES:\n")
            if 'random_forest' in results['feature_importance']:
                top_features = results['feature_importance']['random_forest'][:10]
                for i, (feature, importance) in enumerate(top_features, 1):
                    f.write(f"{i:2d}. {feature:30s} {importance:.4f}\n")
            
            f.write(f"\nTraining completed: {results['training_timestamp']}\n")
        
        logger.info(f"Summary report saved to {report_file}")
    
    def validate_model_performance(self, results: dict) -> bool:
        """
        Validate that model performance meets requirements.
        
        Args:
            results: Training results dictionary
            
        Returns:
            True if performance meets requirements, False otherwise
        """
        evaluation_metrics = results['evaluation_metrics']
        
        # Check ensemble classifier accuracy (target: >95%)
        ensemble_accuracy = evaluation_metrics.get('ensemble_classifier_accuracy', 0)
        accuracy_threshold = 0.95
        
        if ensemble_accuracy < accuracy_threshold:
            logger.warning(f"Ensemble accuracy {ensemble_accuracy:.4f} below threshold {accuracy_threshold}")
            return False
        
        # Check regression R² score (target: >0.8)
        rf_r2 = evaluation_metrics.get('random_forest_regressor_r2', 0)
        r2_threshold = 0.8
        
        if rf_r2 < r2_threshold:
            logger.warning(f"Random Forest R² {rf_r2:.4f} below threshold {r2_threshold}")
            return False
        
        logger.info("Model performance validation passed")
        return True
    
    def run_complete_training(self, data_file: str = None, 
                            perform_tuning: bool = False,
                            validate_performance: bool = True) -> bool:
        """
        Run the complete training pipeline.
        
        Args:
            data_file: Path to existing training data file
            perform_tuning: Whether to perform hyperparameter tuning
            validate_performance: Whether to validate model performance
            
        Returns:
            True if training successful and meets requirements
        """
        try:
            # Load data
            dataset = self.load_training_data(data_file)
            
            if not dataset:
                logger.error("Failed to load training data")
                return False
            
            # Train models
            results = self.train_models(dataset, perform_tuning)
            
            # Validate performance
            if validate_performance:
                if not self.validate_model_performance(results):
                    logger.error("Model performance validation failed")
                    return False
            
            # Save results
            self.save_results(results)
            
            logger.info("Complete training pipeline executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return False

def main():
    """Main function for command-line training."""
    parser = argparse.ArgumentParser(description='Train asteroid mining classification models')
    
    parser.add_argument('--data-file', type=str, default=None,
                       help='Path to existing training data file')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip performance validation')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ModelTrainer(args.output_dir)
    
    # Run training
    success = trainer.run_complete_training(
        data_file=args.data_file,
        perform_tuning=args.tune_hyperparameters,
        validate_performance=not args.skip_validation
    )
    
    if success:
        print("Training completed successfully!")
        print(f"Models saved to: {args.output_dir}")
    else:
        print("Training failed!")
        exit(1)

if __name__ == "__main__":
    main()

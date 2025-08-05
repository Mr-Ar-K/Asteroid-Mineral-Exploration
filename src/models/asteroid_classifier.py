"""
Advanced asteroid classification models using ensemble learning.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import warnings

from ..utils.config import config

warnings.filterwarnings('ignore', category=FutureWarning)
logger = logging.getLogger(__name__)

class AsteroidClassifier:
    """Advanced machine learning classifier for asteroid mining potential."""
    
    def __init__(self, model_config: Optional[Dict] = None):
        """
        Initialize asteroid classifier.
        
        Args:
            model_config: Configuration for models. If None, uses default config.
        """
        self.model_config = model_config or config.get("MODEL_CONFIG", {})
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.is_trained = False
        
        # Initialize individual models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize individual ML models."""
        # Random Forest Classifier
        rf_config = self.model_config.get('random_forest', {})
        self.models['random_forest_classifier'] = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 200),
            max_depth=rf_config.get('max_depth', 15),
            min_samples_split=rf_config.get('min_samples_split', 5),
            min_samples_leaf=rf_config.get('min_samples_leaf', 2),
            random_state=rf_config.get('random_state', 42),
            n_jobs=-1
        )
        
        # Random Forest Regressor
        self.models['random_forest_regressor'] = RandomForestRegressor(
            n_estimators=rf_config.get('n_estimators', 200),
            max_depth=rf_config.get('max_depth', 15),
            min_samples_split=rf_config.get('min_samples_split', 5),
            min_samples_leaf=rf_config.get('min_samples_leaf', 2),
            random_state=rf_config.get('random_state', 42),
            n_jobs=-1
        )
        
        # Gradient Boosting Classifier
        gb_config = self.model_config.get('gradient_boosting', {})
        self.models['gradient_boosting_classifier'] = GradientBoostingClassifier(
            n_estimators=gb_config.get('n_estimators', 150),
            learning_rate=gb_config.get('learning_rate', 0.1),
            max_depth=gb_config.get('max_depth', 10),
            subsample=gb_config.get('subsample', 0.8),
            random_state=gb_config.get('random_state', 42)
        )
        
        # Gradient Boosting Regressor
        self.models['gradient_boosting_regressor'] = GradientBoostingRegressor(
            n_estimators=gb_config.get('n_estimators', 150),
            learning_rate=gb_config.get('learning_rate', 0.1),
            max_depth=gb_config.get('max_depth', 10),
            subsample=gb_config.get('subsample', 0.8),
            random_state=gb_config.get('random_state', 42)
        )
        
        logger.info("Initialized ML models")
    
    def train_classification_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, float]:
        """
        Train classification models for mining potential categories.
        
        Args:
            X_train: Training features
            y_train: Training categorical labels (0: low, 1: medium, 2: high, 3: very_high)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary of model performance metrics
        """
        logger.info("Training classification models")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        results = {}
        
        # Train Random Forest Classifier
        logger.info("Training Random Forest Classifier")
        self.models['random_forest_classifier'].fit(X_train_scaled, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.models['random_forest_classifier'], X_train_scaled, y_train, cv=5
        )
        results['rf_classifier_cv_mean'] = cv_scores.mean()
        results['rf_classifier_cv_std'] = cv_scores.std()
        
        # Validation score
        if X_val is not None:
            val_pred = self.models['random_forest_classifier'].predict(X_val_scaled)
            results['rf_classifier_val_accuracy'] = accuracy_score(y_val, val_pred)
        
        # Train Gradient Boosting Classifier
        logger.info("Training Gradient Boosting Classifier")
        self.models['gradient_boosting_classifier'].fit(X_train_scaled, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.models['gradient_boosting_classifier'], X_train_scaled, y_train, cv=5
        )
        results['gb_classifier_cv_mean'] = cv_scores.mean()
        results['gb_classifier_cv_std'] = cv_scores.std()
        
        # Validation score
        if X_val is not None:
            val_pred = self.models['gradient_boosting_classifier'].predict(X_val_scaled)
            results['gb_classifier_val_accuracy'] = accuracy_score(y_val, val_pred)
        
        # Create ensemble classifier
        ensemble_config = self.model_config.get('ensemble', {})
        voting = ensemble_config.get('voting', 'soft')
        weights = ensemble_config.get('weights', [0.6, 0.4])
        
        self.models['ensemble_classifier'] = VotingClassifier(
            estimators=[
                ('rf', self.models['random_forest_classifier']),
                ('gb', self.models['gradient_boosting_classifier'])
            ],
            voting=voting,
            weights=weights
        )
        
        logger.info("Training Ensemble Classifier")
        self.models['ensemble_classifier'].fit(X_train_scaled, y_train)
        
        # Ensemble cross-validation
        cv_scores = cross_val_score(
            self.models['ensemble_classifier'], X_train_scaled, y_train, cv=5
        )
        results['ensemble_classifier_cv_mean'] = cv_scores.mean()
        results['ensemble_classifier_cv_std'] = cv_scores.std()
        
        # Ensemble validation score
        if X_val is not None:
            val_pred = self.models['ensemble_classifier'].predict(X_val_scaled)
            results['ensemble_classifier_val_accuracy'] = accuracy_score(y_val, val_pred)
        
        # Extract feature importance
        self.feature_importance['random_forest'] = self.models['random_forest_classifier'].feature_importances_
        self.feature_importance['gradient_boosting'] = self.models['gradient_boosting_classifier'].feature_importances_
        
        logger.info("Classification model training completed")
        return results
    
    def train_regression_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, float]:
        """
        Train regression models for continuous mining potential scores.
        
        Args:
            X_train: Training features
            y_train: Training continuous scores (0.0 - 1.0)
            X_val: Validation features (optional)
            y_val: Validation scores (optional)
            
        Returns:
            Dictionary of model performance metrics
        """
        logger.info("Training regression models")
        
        # Use the already fitted scaler
        X_train_scaled = self.scaler.transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        results = {}
        
        # Train Random Forest Regressor
        logger.info("Training Random Forest Regressor")
        self.models['random_forest_regressor'].fit(X_train_scaled, y_train)
        
        # Cross-validation score (negative MSE)
        cv_scores = cross_val_score(
            self.models['random_forest_regressor'], X_train_scaled, y_train, 
            cv=5, scoring='neg_mean_squared_error'
        )
        results['rf_regressor_cv_mse'] = -cv_scores.mean()
        results['rf_regressor_cv_std'] = cv_scores.std()
        
        # Validation score
        if X_val is not None:
            val_pred = self.models['random_forest_regressor'].predict(X_val_scaled)
            results['rf_regressor_val_mse'] = mean_squared_error(y_val, val_pred)
            results['rf_regressor_val_r2'] = r2_score(y_val, val_pred)
            results['rf_regressor_val_mae'] = mean_absolute_error(y_val, val_pred)
        
        # Train Gradient Boosting Regressor
        logger.info("Training Gradient Boosting Regressor")
        self.models['gradient_boosting_regressor'].fit(X_train_scaled, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.models['gradient_boosting_regressor'], X_train_scaled, y_train, 
            cv=5, scoring='neg_mean_squared_error'
        )
        results['gb_regressor_cv_mse'] = -cv_scores.mean()
        results['gb_regressor_cv_std'] = cv_scores.std()
        
        # Validation score
        if X_val is not None:
            val_pred = self.models['gradient_boosting_regressor'].predict(X_val_scaled)
            results['gb_regressor_val_mse'] = mean_squared_error(y_val, val_pred)
            results['gb_regressor_val_r2'] = r2_score(y_val, val_pred)
            results['gb_regressor_val_mae'] = mean_absolute_error(y_val, val_pred)
        
        logger.info("Regression model training completed")
        return results
    
    def train_all_models(self, X_train: pd.DataFrame, y_continuous: pd.Series, 
                        y_categorical: pd.Series, X_val: pd.DataFrame = None,
                        y_val_continuous: pd.Series = None, y_val_categorical: pd.Series = None) -> Dict[str, Any]:
        """
        Train all models (classification and regression).
        
        Args:
            X_train: Training features
            y_continuous: Continuous mining potential scores
            y_categorical: Categorical mining potential labels
            X_val: Validation features
            y_val_continuous: Validation continuous scores
            y_val_categorical: Validation categorical labels
            
        Returns:
            Dictionary containing all training results
        """
        logger.info("Training all models")
        
        # Train classification models
        classification_results = self.train_classification_models(
            X_train, y_categorical, X_val, y_val_categorical
        )
        
        # Train regression models
        regression_results = self.train_regression_models(
            X_train, y_continuous, X_val, y_val_continuous
        )
        
        # Combine results
        all_results = {
            'classification': classification_results,
            'regression': regression_results,
            'feature_importance': self.feature_importance
        }
        
        self.is_trained = True
        logger.info("All model training completed")
        
        return all_results
    
    def predict_mining_potential(self, X: pd.DataFrame, 
                                model_type: str = 'ensemble') -> Dict[str, np.ndarray]:
        """
        Predict mining potential for asteroids.
        
        Args:
            X: Feature matrix
            model_type: Type of model to use ('ensemble', 'random_forest', 'gradient_boosting')
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        results = {}
        
        # Classification predictions
        if model_type == 'ensemble':
            classifier = self.models['ensemble_classifier']
            regressor = None  # Use average of regressors
        elif model_type == 'random_forest':
            classifier = self.models['random_forest_classifier']
            regressor = self.models['random_forest_regressor']
        elif model_type == 'gradient_boosting':
            classifier = self.models['gradient_boosting_classifier']
            regressor = self.models['gradient_boosting_regressor']
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Get classification predictions
        results['categorical_predictions'] = classifier.predict(X_scaled)
        results['class_probabilities'] = classifier.predict_proba(X_scaled)
        
        # Get regression predictions
        if regressor is not None:
            results['continuous_predictions'] = regressor.predict(X_scaled)
        else:
            # Use ensemble of regressors
            rf_pred = self.models['random_forest_regressor'].predict(X_scaled)
            gb_pred = self.models['gradient_boosting_regressor'].predict(X_scaled)
            results['continuous_predictions'] = (rf_pred + gb_pred) / 2
        
        # Calculate confidence scores
        max_probs = np.max(results['class_probabilities'], axis=1)
        results['confidence_scores'] = max_probs
        
        # Create prediction summary
        class_labels = ['low', 'medium', 'high', 'very_high']
        results['categorical_labels'] = [class_labels[i] for i in results['categorical_predictions']]
        
        return results
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test_continuous: pd.Series,
                       y_test_categorical: pd.Series) -> Dict[str, Any]:
        """
        Evaluate all trained models on test set.
        
        Args:
            X_test: Test features
            y_test_continuous: Test continuous scores
            y_test_categorical: Test categorical labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation")
        
        logger.info("Evaluating models on test set")
        
        X_test_scaled = self.scaler.transform(X_test)
        results = {}
        
        # Evaluate classifiers
        classifiers = ['random_forest_classifier', 'gradient_boosting_classifier', 'ensemble_classifier']
        
        for clf_name in classifiers:
            clf = self.models[clf_name]
            y_pred = clf.predict(X_test_scaled)
            y_proba = clf.predict_proba(X_test_scaled)
            
            results[f'{clf_name}_accuracy'] = accuracy_score(y_test_categorical, y_pred)
            results[f'{clf_name}_report'] = classification_report(
                y_test_categorical, y_pred, output_dict=True
            )
            results[f'{clf_name}_confusion_matrix'] = confusion_matrix(
                y_test_categorical, y_pred
            ).tolist()
        
        # Evaluate regressors
        regressors = ['random_forest_regressor', 'gradient_boosting_regressor']
        
        for reg_name in regressors:
            reg = self.models[reg_name]
            y_pred = reg.predict(X_test_scaled)
            
            results[f'{reg_name}_mse'] = mean_squared_error(y_test_continuous, y_pred)
            results[f'{reg_name}_rmse'] = np.sqrt(mean_squared_error(y_test_continuous, y_pred))
            results[f'{reg_name}_r2'] = r2_score(y_test_continuous, y_pred)
            results[f'{reg_name}_mae'] = mean_absolute_error(y_test_continuous, y_pred)
        
        logger.info("Model evaluation completed")
        return results
    
    def get_feature_importance(self, feature_names: List[str], top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance from trained models.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature importance rankings
        """
        if not self.feature_importance:
            raise ValueError("Models must be trained to get feature importance")
        
        importance_rankings = {}
        
        for model_name, importance in self.feature_importance.items():
            # Create feature-importance pairs
            feature_imp_pairs = list(zip(feature_names, importance))
            
            # Sort by importance (descending)
            feature_imp_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Take top N
            importance_rankings[model_name] = feature_imp_pairs[:top_n]
        
        return importance_rankings
    
    def save_models(self, save_dir: str = "models"):
        """
        Save trained models to disk.
        
        Args:
            save_dir: Directory to save models
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            model_file = save_path / f"{name}.pkl"
            joblib.dump(model, model_file)
            logger.info(f"Saved {name} to {model_file}")
        
        # Save scaler
        scaler_file = save_path / "scaler.pkl"
        joblib.dump(self.scaler, scaler_file)
        
        # Save feature importance
        importance_file = save_path / "feature_importance.pkl"
        joblib.dump(self.feature_importance, importance_file)
        
        # Save metadata
        metadata = {
            'model_config': self.model_config,
            'is_trained': self.is_trained,
            'models_list': list(self.models.keys())
        }
        metadata_file = save_path / "metadata.pkl"
        joblib.dump(metadata, metadata_file)
        
        logger.info(f"All models saved to {save_path}")
    
    def load_models(self, load_dir: str = "models"):
        """
        Load trained models from disk.
        
        Args:
            load_dir: Directory containing saved models
        """
        load_path = Path(load_dir)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model directory not found: {load_path}")
        
        # Load metadata
        metadata_file = load_path / "metadata.pkl"
        if metadata_file.exists():
            metadata = joblib.load(metadata_file)
            self.model_config = metadata.get('model_config', {})
            models_list = metadata.get('models_list', [])
        else:
            models_list = ['random_forest_classifier', 'gradient_boosting_classifier',
                          'ensemble_classifier', 'random_forest_regressor', 'gradient_boosting_regressor']
        
        # Load individual models
        for name in models_list:
            model_file = load_path / f"{name}.pkl"
            if model_file.exists():
                self.models[name] = joblib.load(model_file)
                logger.info(f"Loaded {name} from {model_file}")
        
        # Load scaler
        scaler_file = load_path / "scaler.pkl"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
        
        # Load feature importance
        importance_file = load_path / "feature_importance.pkl"
        if importance_file.exists():
            self.feature_importance = joblib.load(importance_file)
        
        self.is_trained = True
        logger.info(f"Models loaded from {load_path}")
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                            model_name: str = 'random_forest', cv_folds: int = 3) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for specified model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Model to tune ('random_forest' or 'gradient_boosting')
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing best parameters and scores
        """
        logger.info(f"Starting hyperparameter tuning for {model_name}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if model_name == 'random_forest':
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [5, 10, 15],
                'subsample': [0.8, 0.9, 1.0]
            }
        else:
            raise ValueError(f"Unsupported model for tuning: {model_name}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_folds, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Hyperparameter tuning completed for {model_name}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best params: {grid_search.best_params_}")
        
        return results

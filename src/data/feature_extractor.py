"""
Feature extraction and engineering for asteroid classification.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

from ..utils.config import config

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class FeatureExtractor:
    """Advanced feature extraction for asteroid classification."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.feature_config = config.get_features()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        
    def extract_orbital_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and engineer orbital mechanics features.
        
        Args:
            data: Raw asteroid data
            
        Returns:
            DataFrame with orbital features
        """
        df = data.copy()
        
        # Handle None values and convert to numeric where possible
        numeric_columns = ['semi_major_axis', 'eccentricity', 'inclination', 
                          'perihelion_distance', 'aphelion_distance', 'orbital_period',
                          'diameter', 'albedo', 'absolute_magnitude', 'rotation_period']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Basic orbital elements
        orbital_features = []
        
        # Semi-major axis features
        if 'semi_major_axis' in df.columns:
            df['sma_log'] = np.log(df['semi_major_axis'].clip(lower=0.1))
            orbital_features.extend(['semi_major_axis', 'sma_log'])
        
        # Eccentricity features
        if 'eccentricity' in df.columns:
            df['ecc_squared'] = df['eccentricity'] ** 2
            df['ecc_category'] = pd.cut(df['eccentricity'], 
                                      bins=[0, 0.1, 0.3, 0.7, 1.0],
                                      labels=['circular', 'low', 'moderate', 'high'])
            orbital_features.extend(['eccentricity', 'ecc_squared'])
        
        # Inclination features
        if 'inclination' in df.columns:
            df['inc_sin'] = np.sin(np.radians(df['inclination']))
            df['inc_cos'] = np.cos(np.radians(df['inclination']))
            orbital_features.extend(['inclination', 'inc_sin', 'inc_cos'])
        
        # Perihelion and aphelion distances
        if 'perihelion_distance' in df.columns and 'aphelion_distance' in df.columns:
            df['orbital_range'] = df['aphelion_distance'] - df['perihelion_distance']
            df['orbital_ratio'] = df['aphelion_distance'] / df['perihelion_distance'].clip(lower=0.1)
            orbital_features.extend(['perihelion_distance', 'aphelion_distance', 
                                   'orbital_range', 'orbital_ratio'])
        
        # Earth proximity indicators
        if 'perihelion_distance' in df.columns:
            df['earth_proximity'] = 1.0 / (df['perihelion_distance'].clip(lower=0.1))
            
            # Handle NaN values in the comparison
            perihelion_mask = df['perihelion_distance'].notna() & (df['perihelion_distance'] < 1.02)
            aphelion_mask = df.get('aphelion_distance', pd.Series([999]*len(df), index=df.index)).notna() & \
                           (df.get('aphelion_distance', pd.Series([999]*len(df), index=df.index)) > 0.98)
            
            df['crosses_earth_orbit'] = perihelion_mask & aphelion_mask
            orbital_features.extend(['earth_proximity'])
        
        # Orbital period features
        if 'orbital_period' in df.columns:
            df['period_log'] = np.log(df['orbital_period'].clip(lower=0.1))
            df['period_ratio_earth'] = df['orbital_period'] / 365.25
            orbital_features.extend(['orbital_period', 'period_log', 'period_ratio_earth'])
        
        # Tisserand parameter (Jupiter)
        if all(col in df.columns for col in ['semi_major_axis', 'eccentricity', 'inclination']):
            a_jupiter = 5.2  # AU
            df['tisserand_jupiter'] = (a_jupiter / df['semi_major_axis']) + \
                                    2 * np.sqrt((df['semi_major_axis'] / a_jupiter) * \
                                              (1 - df['eccentricity']**2)) * \
                                    np.cos(np.radians(df['inclination']))
            orbital_features.append('tisserand_jupiter')
        
        logger.info(f"Extracted {len(orbital_features)} orbital features")
        return df
    
    def extract_physical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and engineer physical characteristics features.
        
        Args:
            data: Asteroid data with physical properties
            
        Returns:
            DataFrame with physical features
        """
        df = data.copy()
        physical_features = []
        
        # Absolute magnitude features
        if 'absolute_magnitude' in df.columns:
            df['h_mag_norm'] = (df['absolute_magnitude'] - 15.0) / 5.0  # Normalized around typical value
            df['brightness_category'] = pd.cut(df['absolute_magnitude'],
                                             bins=[0, 15, 20, 25, 35],
                                             labels=['bright', 'moderate', 'dim', 'very_dim'])
            physical_features.extend(['absolute_magnitude', 'h_mag_norm'])
        
        # Diameter features
        if 'diameter' in df.columns:
            df['diameter_log'] = np.log(df['diameter'].clip(lower=0.001))
            df['size_category'] = pd.cut(df['diameter'],
                                       bins=[0, 0.1, 1.0, 10.0, 100.0],
                                       labels=['small', 'medium', 'large', 'very_large'])
            physical_features.extend(['diameter', 'diameter_log'])
        
        # Albedo features
        if 'albedo' in df.columns:
            df['albedo_log'] = np.log(df['albedo'].clip(lower=0.001))
            df['albedo_category'] = pd.cut(df['albedo'],
                                         bins=[0, 0.05, 0.15, 0.3, 1.0],
                                         labels=['very_dark', 'dark', 'moderate', 'bright'])
            physical_features.extend(['albedo', 'albedo_log'])
        
        # Size-albedo relationship
        if 'diameter' in df.columns and 'albedo' in df.columns:
            df['size_albedo_product'] = df['diameter'] * df['albedo']
            df['albedo_size_ratio'] = df['albedo'] / df['diameter'].clip(lower=0.001)
            physical_features.extend(['size_albedo_product', 'albedo_size_ratio'])
        
        # Rotation period features
        if 'rotation_period' in df.columns:
            # Remove invalid values (NaN, None, <= 0)
            valid_rotation = df['rotation_period'].notna() & (df['rotation_period'] > 0)
            df = df[valid_rotation] if valid_rotation.any() else df
            
            if len(df) > 0 and 'rotation_period' in df.columns:
                df['rotation_log'] = np.log(df['rotation_period'].clip(lower=0.1))
                df['rotation_category'] = pd.cut(df['rotation_period'],
                                               bins=[0, 2.2, 12, 100, 1000],
                                               labels=['fast', 'moderate', 'slow', 'very_slow'])
                
                # Fast rotators might indicate structural strength
                df['fast_rotator'] = df['rotation_period'] < 2.2
                physical_features.extend(['rotation_period', 'rotation_log'])
        
        # Density estimation (simplified)
        if 'diameter' in df.columns and 'absolute_magnitude' in df.columns:
            # Rough density estimate based on size-magnitude relationship
            df['estimated_density'] = np.exp(-0.2 * df['absolute_magnitude']) / \
                                     (df['diameter']**3).clip(lower=0.001)
            physical_features.append('estimated_density')
        
        logger.info(f"Extracted {len(physical_features)} physical features")
        return df
    
    def extract_spectral_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract spectral and compositional features.
        
        Args:
            data: Asteroid data with spectral information
            
        Returns:
            DataFrame with spectral features
        """
        df = data.copy()
        spectral_features = []
        
        # Spectral type encoding
        if 'spectral_type' in df.columns:
            # Group similar spectral types
            spectral_groups = {
                'C': ['C', 'B', 'F', 'G'], 'S': ['S', 'Q', 'R', 'A', 'V'],
                'M': ['M', 'E', 'P'], 'D': ['D', 'T'], 'X': ['X', 'L', 'K']
            }
            
            def group_spectral_type(spec_type):
                if pd.isna(spec_type) or spec_type == '':
                    return 'Unknown'
                spec_type = str(spec_type).upper().strip()
                for group, types in spectral_groups.items():
                    if any(t in spec_type for t in types):
                        return group
                return 'Other'
            
            df['spectral_group'] = df['spectral_type'].apply(group_spectral_type)
            
            # Create binary features for major groups
            for group in ['C', 'S', 'M', 'D', 'X']:
                df[f'is_{group}_type'] = (df['spectral_group'] == group).astype(int)
                spectral_features.append(f'is_{group}_type')
        
        # Taxonomic class features
        if 'taxonomic_class' in df.columns:
            # Similar grouping for taxonomic classes
            df['taxonomic_group'] = df['taxonomic_class'].fillna('Unknown')
            
        # Composition indicators from NEOWISE data
        if 'composition_class' in df.columns:
            for comp_type in ['S-type', 'C-type', 'M-type', 'X-type']:
                df[f'is_{comp_type.replace("-", "_")}'] = \
                    (df['composition_class'] == comp_type).astype(int)
                spectral_features.append(f'is_{comp_type.replace("-", "_")}')
        
        # Thermal features from NEOWISE
        thermal_features = ['w1_w2_color', 'w3_w4_color', 'thermal_spectral_index']
        for feature in thermal_features:
            if feature in df.columns:
                spectral_features.append(feature)
        
        # Resource potential indicators
        if 'resource_potential' in df.columns:
            df['high_resource_potential'] = (df['resource_potential'] > 0.7).astype(int)
            spectral_features.extend(['resource_potential', 'high_resource_potential'])
        
        logger.info(f"Extracted {len(spectral_features)} spectral features")
        return df
    
    def calculate_delta_v_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate delta-v and accessibility features.
        
        Args:
            data: Asteroid data with orbital elements
            
        Returns:
            DataFrame with delta-v features
        """
        df = data.copy()
        delta_v_config = config.get_delta_v_config()
        
        # Simplified delta-v calculation
        # In practice, this would use more sophisticated orbital mechanics
        
        if 'semi_major_axis' in df.columns and 'eccentricity' in df.columns:
            # Characteristic velocity (simplified)
            mu_sun = 1.327e11  # km³/s² (standard gravitational parameter of Sun)
            
            # Convert AU to km
            a_km = df['semi_major_axis'] * 1.496e8
            
            # Orbital velocity at perihelion
            v_perihelion = np.sqrt(mu_sun * (2 / (df['perihelion_distance'] * 1.496e8) - 1 / a_km))
            
            # Earth's orbital velocity
            v_earth = 29.78  # km/s
            
            # Simplified delta-v estimate
            df['delta_v_rendezvous'] = np.abs(v_perihelion - v_earth) + \
                                     delta_v_config.get('asteroid_rendezvous', 1.5)
            
            # Total mission delta-v
            df['delta_v_total'] = df['delta_v_rendezvous'] + \
                                delta_v_config.get('earth_departure', 3.2) + \
                                delta_v_config.get('return_trajectory', 2.8)
            
            # Accessibility score (inverse of delta-v)
            df['accessibility_score'] = 1.0 / (1.0 + df['delta_v_total'] / 10.0)
            
            # Mission difficulty categories
            df['mission_difficulty'] = pd.cut(df['delta_v_total'],
                                            bins=[0, 8, 12, 16, 50],
                                            labels=['easy', 'moderate', 'difficult', 'extreme'])
        
        return df
    
    def create_mining_potential_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite features for mining potential assessment.
        
        Args:
            data: Asteroid data with all features
            
        Returns:
            DataFrame with mining potential features
        """
        df = data.copy()
        
        # Size-based value (larger asteroids have more material)
        if 'diameter' in df.columns:
            df['size_value'] = np.log(1 + df['diameter'])
        
        # Composition value based on spectral type
        composition_values = {
            'M': 0.9,  # Metallic - high value
            'S': 0.7,  # Stony - moderate value
            'C': 0.6,  # Carbonaceous - water/organics
            'X': 0.5,  # Unknown
            'D': 0.3   # Organic-rich but distant
        }
        
        if 'spectral_group' in df.columns:
            df['composition_value'] = df['spectral_group'].map(composition_values).fillna(0.4)
        
        # Accessibility value (inverse of mission difficulty)
        if 'accessibility_score' in df.columns:
            df['accessibility_value'] = df['accessibility_score']
        else:
            df['accessibility_value'] = 0.5  # Default
        
        # Economic viability composite score
        value_weights = {
            'size': 0.3,
            'composition': 0.4,
            'accessibility': 0.3
        }
        
        df['economic_value'] = (
            value_weights['size'] * df.get('size_value', 0.5) +
            value_weights['composition'] * df.get('composition_value', 0.5) +
            value_weights['accessibility'] * df.get('accessibility_value', 0.5)
        )
        
        # Risk factors
        risk_factors = []
        
        # Rotation risk (very fast rotators are structurally risky)
        if 'rotation_period' in df.columns:
            df['rotation_risk'] = (df['rotation_period'] < 2.2).astype(float) * 0.3
            risk_factors.append('rotation_risk')
        
        # Orbital risk (highly eccentric orbits are less predictable)
        if 'eccentricity' in df.columns:
            df['orbital_risk'] = (df['eccentricity'] > 0.5).astype(float) * 0.2
            risk_factors.append('orbital_risk')
        
        # Size risk (very small asteroids are harder to work with)
        if 'diameter' in df.columns:
            df['size_risk'] = (df['diameter'] < 0.1).astype(float) * 0.3
            risk_factors.append('size_risk')
        
        # Total risk score
        if risk_factors:
            df['total_risk'] = df[risk_factors].sum(axis=1)
        else:
            df['total_risk'] = 0.0
        
        # Final mining potential score
        df['mining_potential_raw'] = df['economic_value'] * (1.0 - df['total_risk'])
        df['mining_potential'] = np.clip(df['mining_potential_raw'], 0.0, 1.0)
        
        # Mining potential categories
        thresholds = config.mining_thresholds
        df['mining_category'] = pd.cut(
            df['mining_potential'],
            bins=[0, thresholds.get('low', 0.4), 
                  thresholds.get('medium', 0.6), 
                  thresholds.get('high', 0.8), 1.0],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        return df
    
    def prepare_features_for_ml(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for machine learning models.
        
        Args:
            data: DataFrame with all extracted features
            
        Returns:
            Tuple of (processed_dataframe, feature_names)
        """
        df = data.copy()
        
        # Select numerical features for ML
        numerical_features = []
        categorical_features = []
        
        # Get feature lists from config
        feature_config = self.feature_config
        
        for category, features in feature_config.items():
            for feature in features:
                if feature in df.columns:
                    if df[feature].dtype in ['object', 'category']:
                        categorical_features.append(feature)
                    else:
                        numerical_features.append(feature)
        
        # Add engineered features
        engineered_features = [
            'sma_log', 'ecc_squared', 'inc_sin', 'inc_cos',
            'orbital_range', 'orbital_ratio', 'earth_proximity',
            'period_log', 'period_ratio_earth', 'tisserand_jupiter',
            'h_mag_norm', 'diameter_log', 'albedo_log',
            'size_albedo_product', 'albedo_size_ratio',
            'rotation_log', 'estimated_density',
            'w1_w2_color', 'w3_w4_color', 'thermal_spectral_index',
            'resource_potential', 'delta_v_total', 'accessibility_score',
            'economic_value', 'total_risk'
        ]
        
        for feature in engineered_features:
            if feature in df.columns and feature not in numerical_features:
                numerical_features.append(feature)
        
        # Add binary spectral features
        spectral_binary = [col for col in df.columns if col.startswith('is_')]
        numerical_features.extend(spectral_binary)
        
        # Handle categorical features
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(
                    df[feature].fillna('Unknown').astype(str)
                )
            else:
                # Handle unseen categories
                unique_values = set(df[feature].fillna('Unknown').astype(str))
                known_values = set(self.label_encoders[feature].classes_)
                
                if unique_values - known_values:
                    # Refit encoder with new categories
                    all_values = list(known_values.union(unique_values))
                    self.label_encoders[feature].fit(all_values)
                
                df[f'{feature}_encoded'] = self.label_encoders[feature].transform(
                    df[feature].fillna('Unknown').astype(str)
                )
            
            numerical_features.append(f'{feature}_encoded')
        
        # Select final feature set
        available_features = [f for f in numerical_features if f in df.columns]
        
        # Handle missing values
        if available_features:
            # Check for features with all NaN values and remove them
            valid_features = []
            for feature in available_features:
                if not df[feature].isna().all():
                    valid_features.append(feature)
            
            if valid_features:
                # Fit and transform the valid features
                feature_data = df[valid_features].copy()
                imputed_values = self.imputer.fit_transform(feature_data)
                
                # Update the original DataFrame with imputed values
                for i, col in enumerate(valid_features):
                    df[col] = imputed_values[:, i]
                
                # Remove infinite values and handle them
                for col in valid_features:
                    # Replace infinite values with NaN
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                # Apply imputation again if we introduced new NaNs
                if df[valid_features].isna().any().any():
                    feature_data = df[valid_features].copy()
                    imputed_values = self.imputer.fit_transform(feature_data)
                    for i, col in enumerate(valid_features):
                        df[col] = imputed_values[:, i]
                
                available_features = valid_features
            else:
                available_features = []
        
        logger.info(f"Prepared {len(available_features)} features for ML")
        return df, available_features
    
    def extract_all_features(self, asteroid_data: pd.DataFrame, 
                           neowise_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract all features from asteroid and NEOWISE data.
        
        Args:
            asteroid_data: Main asteroid dataset
            neowise_data: NEOWISE photometry data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Starting comprehensive feature extraction")
        
        # Merge datasets if NEOWISE data provided
        if neowise_data is not None and not neowise_data.empty:
            df = asteroid_data.merge(neowise_data, on='designation', how='left')
        else:
            df = asteroid_data.copy()
        
        # Extract features in sequence
        df = self.extract_orbital_features(df)
        df = self.extract_physical_features(df)
        df = self.extract_spectral_features(df)
        df = self.calculate_delta_v_features(df)
        df = self.create_mining_potential_features(df)
        
        logger.info(f"Feature extraction complete. Dataset shape: {df.shape}")
        return df

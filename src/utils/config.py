"""
Configuration manager for the Asteroid Mining Classification Dashboard.
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Try to load python-dotenv if available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        # Load .env file if available
        if DOTENV_AVAILABLE:
            env_file = Path(__file__).parent.parent.parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
        
        if config_path is None:
            # Try to find config.yaml, fallback to template
            config_dir = Path(__file__).parent.parent.parent / "config"
            config_file = config_dir / "config.yaml"
            template_file = config_dir / "config.template.yaml"
            
            if config_file.exists():
                config_path = config_file
            elif template_file.exists():
                config_path = template_file
            else:
                # Create minimal config if none exists
                config_path = self._create_minimal_config(config_dir)
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._load_env_variables()
    
    def _create_minimal_config(self, config_dir: Path) -> Path:
        """Create a minimal configuration file if none exists."""
        config_dir.mkdir(parents=True, exist_ok=True)
        minimal_config = config_dir / "config.yaml"
        
        minimal_content = {
            'NASA_SBDB_API': 'https://ssd-api.jpl.nasa.gov/sbdb.api',
            'NASA_API_KEY': '',
            'NEOWISE_DATA_URL': 'https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query',
            'MODEL_CONFIG': {
                'random_forest': {'n_estimators': 200, 'max_depth': 15, 'random_state': 42},
                'gradient_boosting': {'n_estimators': 150, 'learning_rate': 0.1, 'random_state': 42}
            },
            'DATA_PROCESSING': {'max_api_calls_per_minute': 100, 'cache_expiry_days': 7}
        }
        
        with open(minimal_config, 'w') as f:
            yaml.dump(minimal_content, f, default_flow_style=False)
        
        return minimal_config
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing configuration file: {e}")
            return {}
    
    def _load_env_variables(self):
        """Load environment variables to override config values."""
        # Map of environment variable names to config keys
        env_mappings = {
            'NASA_API_KEY': 'NASA_API_KEY',
            'NASA_EMAIL': 'NASA_ACCOUNT.email',
            'NASA_ACCOUNT_ID': 'NASA_ACCOUNT.account_id',
            'LOG_LEVEL': 'LOGGING.level',
            'DATA_CACHE_DIR': 'DATA_PROCESSING.cache_dir',
            'MODELS_DIR': 'MODEL_CONFIG.models_dir'
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                self._set_nested_value(config_key, env_value)
    
    def _set_nested_value(self, key_path: str, value: str):
        """Set a nested configuration value using dot notation."""
        keys = key_path.split('.')
        config = self._config
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the final value
        config[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key, with environment variable override support.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'MODEL_CONFIG.random_forest')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Check for direct environment variable first
        if '.' not in key:
            env_value = os.getenv(key)
            if env_value is not None:
                return env_value
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_api_key(self) -> str:
        """Get NASA API key with environment variable priority."""
        # First check environment variable
        api_key = os.getenv('NASA_API_KEY')
        if api_key:
            return api_key
        
        # Fallback to config file
        return self.get('NASA_API_KEY', '')
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model."""
        return self.get(f"MODEL_CONFIG.{model_name}", {})
    
    def get_features(self) -> Dict[str, list]:
        """Get feature configuration."""
        return self.get("FEATURES", {})
    
    def get_api_config(self) -> Dict[str, str]:
        """Get API configuration."""
        return {
            'nasa_sbdb': self.get("NASA_SBDB_API"),
            'neowise': self.get("NEOWISE_DATA_URL")
        }
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration."""
        return self.get("DASHBOARD", {})
    
    def get_delta_v_config(self) -> Dict[str, float]:
        """Get delta-v calculation parameters."""
        return self.get("DELTA_V_CONFIG", {})
    
    @property
    def mining_thresholds(self) -> Dict[str, float]:
        """Get mining potential thresholds."""
        return self.get("MINING_POTENTIAL_THRESHOLDS", {})

# Global configuration instance
config = Config()

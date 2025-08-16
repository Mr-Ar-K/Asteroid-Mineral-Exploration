"""
NASA JPL Small Body Database API client for asteroid data retrieval.
"""
import requests
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode
import json

from ..utils.config import config

logger = logging.getLogger(__name__)

class SBDBClient:
    """Client for NASA JPL Small Body Database API."""
    
    def __init__(self):
        """Initialize the SBDB client."""
        self.logger = logging.getLogger(__name__)
        self.base_url = config.get("NASA_SBDB_API")
        self.api_key = config.get_api_key()  # Use the new method
        
        # Set up session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Asteroid-Mining-Classifier/1.0',
            'X-API-Key': self.api_key if self.api_key else ''
        })
        
        self.api_calls_count = 0
        self.last_call_time = time.time()  # Initialize last_call_time
        
    def _rate_limit(self):
        """Implement rate limiting for API calls."""
        max_calls_per_minute = config.get("DATA_PROCESSING.max_api_calls_per_minute", 100)
        current_time = time.time()
        
        if self.api_calls_count >= max_calls_per_minute:
            if current_time - self.last_call_time < 60:
                sleep_time = 60 - (current_time - self.last_call_time)
                self.logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.api_calls_count = 0
        
        if current_time - self.last_call_time >= 60:
            self.api_calls_count = 0
            
        self.api_calls_count += 1
        self.last_call_time = current_time  # Update last call time
        self.last_call_time = current_time
    
    def get_asteroid_data(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed data for a specific asteroid.
        
        Args:
            designation: Asteroid designation (e.g., "2000 SG344")
            
        Returns:
            Dictionary containing asteroid data or None if not found
        """
        self._rate_limit()
        
        params = {
            'sstr': designation,
            'full-prec': 'true'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'object' in data:
                return self._parse_asteroid_data(data['object'])
            else:
                logger.warning(f"No data found for asteroid: {designation}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error fetching data for {designation}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response for {designation}: {e}")
            return None
    
    def _parse_asteroid_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw SBDB data into structured format."""
        parsed = {
            'designation': data.get('des', ''),
            'name': data.get('fullname', ''),
            'neo': data.get('neo', False),
            'pha': data.get('pha', False),
            'absolute_magnitude': self._safe_float(data.get('H')),
            'diameter': self._safe_float(data.get('diameter')),
            'albedo': self._safe_float(data.get('albedo')),
            'rotation_period': self._safe_float(data.get('rot_per')),
            'spectral_type': data.get('spec_T', ''),
            'taxonomic_class': data.get('class', ''),
        }
        
        # Orbital elements
        if 'orbit' in data:
            orbit = data['orbit']
            parsed.update({
                'semi_major_axis': self._safe_float(orbit.get('a')),
                'eccentricity': self._safe_float(orbit.get('e')),
                'inclination': self._safe_float(orbit.get('i')),
                'perihelion_distance': self._safe_float(orbit.get('q')),
                'aphelion_distance': self._safe_float(orbit.get('Q')),
                'orbital_period': self._safe_float(orbit.get('per')),
                'mean_anomaly': self._safe_float(orbit.get('ma')),
                'argument_perihelion': self._safe_float(orbit.get('w')),
                'longitude_ascending_node': self._safe_float(orbit.get('om')),
                'epoch': orbit.get('epoch', ''),
            })
        
        return parsed
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def search_asteroids(self, 
                        neo_only: bool = True,
                        pha_only: bool = False,
                        limit: int = 1000) -> List[str]:
        """
        Search for asteroids matching criteria.
        
        Args:
            neo_only: Only return Near-Earth Objects
            pha_only: Only return Potentially Hazardous Asteroids
            limit: Maximum number of results
            
        Returns:
            List of asteroid designations
        """
        # This is a simplified implementation
        # In practice, you'd use the SBDB search API or pre-compiled lists
        
        # Sample NEO designations for demonstration
        sample_neos = [
            "2000 SG344", "2019 GT3", "2020 BX12", "2021 AC",
            "2022 AP7", "2019 FU", "2020 NK1", "2021 AN5",
            "2022 EB5", "2019 UA4", "2020 CD3", "2021 DW1",
            "2022 WJ1", "2019 BE5", "2020 QG", "2021 PT",
            "2022 YG", "2019 OK", "2020 SO", "2021 YQ",
            "2022 ES3", "2019 AQ3", "2020 JJ", "2021 PH27",
            "2022 AP7", "2019 LF6", "2020 QU6", "2021 LJ4"
        ]
        
        return sample_neos[:limit]
    
    def get_bulk_data(self, designations: List[str]) -> pd.DataFrame:
        """
        Get data for multiple asteroids.
        
        Args:
            designations: List of asteroid designations
            
        Returns:
            DataFrame containing asteroid data
        """
        data_list = []
        
        for designation in designations:
            logger.info(f"Fetching data for {designation}")
            data = self.get_asteroid_data(designation)
            if data:
                data_list.append(data)
            
            # Add small delay between requests
            time.sleep(0.1)
        
        if data_list:
            df = pd.DataFrame(data_list)
            return df
        else:
            return pd.DataFrame()
    
    def get_near_earth_asteroids(self, limit: int = 500) -> pd.DataFrame:
        """
        Get data for Near-Earth Asteroids.
        
        Args:
            limit: Maximum number of asteroids to fetch
            
        Returns:
            DataFrame containing NEO data
        """
        designations = self.search_asteroids(neo_only=True, limit=limit)
        return self.get_bulk_data(designations)

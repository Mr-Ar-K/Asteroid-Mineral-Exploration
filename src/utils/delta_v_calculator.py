"""
Utility functions for delta-v calculations and orbital mechanics.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import math
import warnings

from .config import config

warnings.filterwarnings('ignore')

class DeltaVCalculator:
    """Advanced delta-v and mission trajectory calculator."""
    
    def __init__(self):
        """Initialize calculator with physical constants."""
        # Gravitational parameters (km³/s²)
        self.mu_sun = 1.327124400e11
        self.mu_earth = 3.986004418e5
        
        # Earth orbital parameters
        self.earth_sma = 1.0  # AU
        self.earth_velocity = 29.78  # km/s
        
        # Mission parameters from config
        self.delta_v_config = config.get_delta_v_config()
    
    def orbital_velocity(self, semi_major_axis: float, distance: float) -> float:
        """
        Calculate orbital velocity at given distance.
        
        Args:
            semi_major_axis: Semi-major axis in AU
            distance: Current distance from Sun in AU
            
        Returns:
            Orbital velocity in km/s
        """
        # Convert AU to km
        a_km = semi_major_axis * 1.496e8
        r_km = distance * 1.496e8
        
        # Vis-viva equation
        velocity = math.sqrt(self.mu_sun * (2/r_km - 1/a_km))
        return velocity
    
    def hohmann_transfer_delta_v(self, r1: float, r2: float) -> Tuple[float, float]:
        """
        Calculate delta-v for Hohmann transfer between two circular orbits.
        
        Args:
            r1: Initial orbital radius (AU)
            r2: Final orbital radius (AU)
            
        Returns:
            Tuple of (delta_v1, delta_v2) in km/s
        """
        # Convert to km
        r1_km = r1 * 1.496e8
        r2_km = r2 * 1.496e8
        
        # Transfer orbit semi-major axis
        a_transfer = (r1_km + r2_km) / 2
        
        # Initial and final circular velocities
        v1_circular = math.sqrt(self.mu_sun / r1_km)
        v2_circular = math.sqrt(self.mu_sun / r2_km)
        
        # Transfer orbit velocities
        v1_transfer = math.sqrt(self.mu_sun * (2/r1_km - 1/a_transfer))
        v2_transfer = math.sqrt(self.mu_sun * (2/r2_km - 1/a_transfer))
        
        # Delta-v calculations
        delta_v1 = abs(v1_transfer - v1_circular)
        delta_v2 = abs(v2_circular - v2_transfer)
        
        return delta_v1, delta_v2
    
    def calculate_asteroid_delta_v(self, 
                                  semi_major_axis: float,
                                  eccentricity: float,
                                  inclination: float = 0.0,
                                  perihelion_distance: float = None) -> Dict[str, float]:
        """
        Calculate comprehensive delta-v requirements for asteroid mission.
        
        Args:
            semi_major_axis: Asteroid's semi-major axis in AU
            eccentricity: Orbital eccentricity
            inclination: Orbital inclination in degrees
            perihelion_distance: Perihelion distance in AU (calculated if None)
            
        Returns:
            Dictionary containing various delta-v components
        """
        if perihelion_distance is None:
            perihelion_distance = semi_major_axis * (1 - eccentricity)
        
        aphelion_distance = semi_major_axis * (1 + eccentricity)
        
        # Earth departure delta-v (from config)
        earth_departure = self.delta_v_config.get('earth_departure', 3.2)
        
        # Interplanetary transfer
        # Use perihelion distance for most efficient transfer
        target_distance = perihelion_distance
        
        if target_distance < 1.0:  # Inside Earth's orbit
            # Transfer from Earth to asteroid perihelion
            dv_transfer_1, dv_transfer_2 = self.hohmann_transfer_delta_v(1.0, target_distance)
        else:  # Outside Earth's orbit
            # Transfer from Earth to asteroid
            dv_transfer_1, dv_transfer_2 = self.hohmann_transfer_delta_v(1.0, target_distance)
        
        # Rendezvous delta-v (depends on asteroid's orbital velocity)
        asteroid_velocity = self.orbital_velocity(semi_major_axis, target_distance)
        transfer_velocity = self.orbital_velocity((1.0 + target_distance)/2, target_distance)
        
        # Account for eccentricity - higher eccentricity means more variable velocity
        eccentricity_factor = 1.0 + eccentricity * 0.5
        rendezvous_delta_v = abs(asteroid_velocity - transfer_velocity) * eccentricity_factor
        
        # Inclination change penalty
        if inclination > 0:
            # Simplified inclination change cost
            inclination_penalty = 2 * asteroid_velocity * math.sin(math.radians(inclination/2))
        else:
            inclination_penalty = 0.0
        
        # Station-keeping and operations
        operations_delta_v = self.delta_v_config.get('asteroid_rendezvous', 1.5)
        
        # Return trajectory
        return_delta_v = self.delta_v_config.get('return_trajectory', 2.8)
        
        # Total mission delta-v
        total_delta_v = (earth_departure + 
                        dv_transfer_1 + 
                        rendezvous_delta_v + 
                        inclination_penalty + 
                        operations_delta_v + 
                        return_delta_v)
        
        return {
            'earth_departure': earth_departure,
            'interplanetary_transfer': dv_transfer_1,
            'rendezvous': rendezvous_delta_v,
            'inclination_change': inclination_penalty,
            'operations': operations_delta_v,
            'return_trajectory': return_delta_v,
            'total': total_delta_v,
            'accessibility_score': self._calculate_accessibility_score(total_delta_v)
        }
    
    def _calculate_accessibility_score(self, total_delta_v: float) -> float:
        """
        Calculate accessibility score based on total delta-v.
        
        Args:
            total_delta_v: Total mission delta-v in km/s
            
        Returns:
            Accessibility score (0-1, higher is better)
        """
        # Accessibility decreases exponentially with delta-v
        # Typical missions range from 8-20 km/s total delta-v
        
        if total_delta_v <= 8.0:
            return 1.0
        elif total_delta_v >= 20.0:
            return 0.1
        else:
            # Exponential decay between 8 and 20 km/s
            normalized_dv = (total_delta_v - 8.0) / 12.0
            return 0.9 * math.exp(-3 * normalized_dv) + 0.1
    
    def launch_window_analysis(self, 
                              target_sma: float,
                              target_ecc: float,
                              launch_year: int = 2025) -> Dict[str, any]:
        """
        Analyze optimal launch windows for asteroid mission.
        
        Args:
            target_sma: Target asteroid semi-major axis (AU)
            target_ecc: Target asteroid eccentricity
            launch_year: Launch year
            
        Returns:
            Dictionary containing launch window analysis
        """
        # Simplified launch window calculation
        # In reality, this would require precise ephemeris calculations
        
        target_period = target_sma ** 1.5  # Kepler's 3rd law (years)
        earth_period = 1.0  # years
        
        # Synodic period (time between optimal launch opportunities)
        if target_period > earth_period:
            synodic_period = 1 / (1/earth_period - 1/target_period)
        else:
            synodic_period = 1 / (1/target_period - 1/earth_period)
        
        # Launch window duration (simplified)
        # More eccentric orbits have shorter windows
        window_duration = 30 * (1 - target_ecc)  # days
        
        # Next optimal launch dates
        optimal_dates = []
        for i in range(3):  # Next 3 opportunities
            launch_date = launch_year + i * synodic_period
            optimal_dates.append(launch_date)
        
        return {
            'synodic_period_years': synodic_period,
            'window_duration_days': window_duration,
            'next_launch_opportunities': optimal_dates,
            'optimal_year': min(optimal_dates, key=lambda x: abs(x - launch_year))
        }
    
    def mission_duration_estimate(self, 
                                 target_sma: float,
                                 mission_type: str = "sample_return") -> Dict[str, float]:
        """
        Estimate mission duration for different mission types.
        
        Args:
            target_sma: Target semi-major axis (AU)
            mission_type: Type of mission ('sample_return', 'mining', 'survey')
            
        Returns:
            Dictionary with mission phase durations
        """
        # Transfer time (simplified as half the synodic period)
        target_period = target_sma ** 1.5
        synodic_period = 1 / abs(1/1.0 - 1/target_period) if target_period != 1.0 else 2.0
        
        transfer_time = synodic_period / 2  # years
        
        # Mission duration based on type
        duration_map = {
            'sample_return': 0.5,  # 6 months
            'mining': 2.0,         # 2 years
            'survey': 1.0          # 1 year
        }
        
        operations_time = duration_map.get(mission_type, 1.0)
        
        # Return transfer time
        return_time = transfer_time
        
        # Total mission duration
        total_duration = transfer_time + operations_time + return_time
        
        return {
            'outbound_transfer_years': transfer_time,
            'operations_years': operations_time,
            'return_transfer_years': return_time,
            'total_duration_years': total_duration,
            'total_duration_months': total_duration * 12
        }

def economic_value_calculator(diameter: float,
                            composition_type: str,
                            accessibility_score: float) -> Dict[str, float]:
    """
    Calculate economic value estimate for asteroid mining.
    
    Args:
        diameter: Asteroid diameter in km
        composition_type: Spectral/composition type
        accessibility_score: Mission accessibility (0-1)
        
    Returns:
        Dictionary with economic value estimates
    """
    # Volume and mass estimation
    volume_km3 = (4/3) * math.pi * (diameter/2)**3
    volume_m3 = volume_km3 * 1e9
    
    # Density estimation based on composition
    density_map = {
        'M-type': 7500,    # kg/m³ (metallic)
        'S-type': 3500,    # kg/m³ (stony)
        'C-type': 2000,    # kg/m³ (carbonaceous)
        'X-type': 4000,    # kg/m³ (unknown/mixed)
    }
    
    density = density_map.get(composition_type, 3000)
    mass_kg = volume_m3 * density
    mass_tons = mass_kg / 1000
    
    # Resource value estimation (USD per ton)
    value_map = {
        'M-type': 1e9,     # High platinum group metals
        'S-type': 5e7,     # Moderate metals + water
        'C-type': 1e7,     # Water + organics
        'X-type': 5e7,     # Average estimate
    }
    
    value_per_ton = value_map.get(composition_type, 1e7)
    
    # Total raw value
    total_raw_value = mass_tons * value_per_ton
    
    # Extraction efficiency (depends on technology)
    extraction_efficiency = 0.1  # 10% initially
    
    # Accessibility modifier
    accessibility_modifier = accessibility_score
    
    # Economic value estimate
    economic_value = total_raw_value * extraction_efficiency * accessibility_modifier
    
    return {
        'mass_tons': mass_tons,
        'raw_value_usd': total_raw_value,
        'extractable_value_usd': economic_value,
        'value_per_ton_usd': value_per_ton,
        'extraction_efficiency': extraction_efficiency,
        'accessibility_modifier': accessibility_modifier
    }

def mission_risk_assessment(delta_v_total: float,
                          mission_duration: float,
                          asteroid_properties: Dict) -> Dict[str, float]:
    """
    Assess mission risk factors.
    
    Args:
        delta_v_total: Total mission delta-v (km/s)
        mission_duration: Total mission duration (years)
        asteroid_properties: Dictionary of asteroid properties
        
    Returns:
        Dictionary with risk assessment scores
    """
    # Technical risk (based on delta-v complexity)
    if delta_v_total < 10:
        technical_risk = 0.2
    elif delta_v_total < 15:
        technical_risk = 0.5
    else:
        technical_risk = 0.8
    
    # Duration risk (longer missions are riskier)
    duration_risk = min(0.8, mission_duration / 10)
    
    # Asteroid-specific risks
    rotation_period = asteroid_properties.get('rotation_period', 24)
    rotation_risk = 0.3 if rotation_period < 2.2 else 0.1  # Fast rotators
    
    eccentricity = asteroid_properties.get('eccentricity', 0)
    orbital_risk = min(0.4, eccentricity)  # Highly eccentric orbits
    
    size_km = asteroid_properties.get('diameter', 1)
    size_risk = 0.3 if size_km < 0.1 else 0.1  # Very small asteroids
    
    # Overall risk score
    risk_factors = [technical_risk, duration_risk, rotation_risk, orbital_risk, size_risk]
    overall_risk = min(1.0, sum(risk_factors) / len(risk_factors))
    
    return {
        'technical_risk': technical_risk,
        'duration_risk': duration_risk,
        'rotation_risk': rotation_risk,
        'orbital_risk': orbital_risk,
        'size_risk': size_risk,
        'overall_risk': overall_risk,
        'risk_category': 'high' if overall_risk > 0.7 else 'medium' if overall_risk > 0.4 else 'low'
    }

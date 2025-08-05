"""
Dashboard Module
================

This module contains the Streamlit web interface for the asteroid mining
classification system.

Components:
- app: Main Streamlit application with multi-page interface
  * Asteroid Explorer: Search and analyze individual asteroids
  * Mining Assessment: Batch analysis and comparison tools
  * Mission Planner: Mission planning and optimization tools  
  * Analytics: Model performance and system metrics

The dashboard provides an intuitive web interface for space mission planners,
researchers, and stakeholders to interact with the AI classification system.
"""

# Dashboard application is imported as needed to avoid Streamlit import issues
# Import using: from src.dashboard.app import main

__all__ = ['app']

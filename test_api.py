#!/usr/bin/env python3
"""
Quick test to verify NASA API credentials are working.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from dotenv import load_dotenv
    import requests
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('NASA_API_KEY')
    if not api_key or api_key == 'your_nasa_api_key_here':
        print("❌ NASA_API_KEY not found or not set properly")
        sys.exit(1)
    
    # Test NASA API connection
    print("🔍 Testing NASA API connection...")
    test_url = f"https://api.nasa.gov/neo/rest/v1/stats?api_key={api_key}"
    
    response = requests.get(test_url, timeout=10)
    
    if response.status_code == 200:
        print("✅ NASA API connection successful!")
        data = response.json()
        print(f"📊 NEO Stats: {data.get('near_earth_object_count', 'N/A')} objects tracked")
    else:
        print(f"❌ NASA API error: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
        
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install python-dotenv requests")
except Exception as e:
    print(f"❌ Test failed: {e}")

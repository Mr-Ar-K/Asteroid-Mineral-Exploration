#!/usr/bin/env python3
"""
Quick start script to launch the Streamlit dashboard.
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit dashboard."""
    print("🚀 Launching AI-Driven Asteroid Mining Dashboard...")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    dashboard_path = project_dir / "src" / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        print("❌ Dashboard file not found!")
        sys.exit(1)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.address", "0.0.0.0",
            "--server.port", "8501"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

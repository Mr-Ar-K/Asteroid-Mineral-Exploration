#!/usr/bin/env python3
"""
Dashboard launcher for the AI-Driven Asteroid Mining Classification System.

This script launches the Streamlit web dashboard for interactive asteroid analysis.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit dashboard."""
    project_root = Path(__file__).parent
    dashboard_path = project_root / "src" / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        sys.exit(1)
    
    print("🚀 Launching Asteroid Mining Dashboard...")
    print("📊 Open your browser to: http://localhost:8501")
    print("⚠️  Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            str(dashboard_path),
            '--server.port', '8501',
            '--server.headless', 'false'
        ], cwd=project_root)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

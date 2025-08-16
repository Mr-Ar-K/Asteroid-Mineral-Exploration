#!/usr/bin/env python3
"""
Asteroid Mining Classification System - Main Launcher
====================================================

This script provides easy access to all system functionality:
- Dashboard: Web interface for interactive analysis
- CLI Prediction: Command-line asteroid analysis
- Data Pipeline: ETL and model training
- Quick Start: System demonstration

Usage:
    python launcher.py dashboard    # Launch web dashboard
    python launcher.py predict <asteroid_id>  # Predict single asteroid
    python launcher.py pipeline     # Run data pipeline
    python launcher.py demo         # Run demo/quickstart
    python launcher.py test         # Run test suite
"""

import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Asteroid Mining Classification System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch web dashboard')
    dashboard_parser.add_argument('--port', type=int, default=8501, help='Port number (default: 8501)')
    
    # Predict command  
    predict_parser = subparsers.add_parser('predict', help='Predict asteroid mining potential')
    predict_parser.add_argument('asteroid_id', help='Asteroid designation (e.g., "2000 SG344")')
    predict_parser.add_argument('--output', help='Output file for results')
    
    # Pipeline command
    subparsers.add_parser('pipeline', help='Run data pipeline')
    
    # Demo command
    subparsers.add_parser('demo', help='Run system demonstration')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run test suite')
    test_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Setup command
    subparsers.add_parser('setup', help='Initialize system and install dependencies')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Get project root directory
    project_root = Path(__file__).parent
    
    try:
        if args.command == 'dashboard':
            print(f"🚀 Launching Asteroid Mining Dashboard on port {args.port}...")
            subprocess.run([
                sys.executable, '-m', 'streamlit', 'run', 
                str(project_root / 'src' / 'dashboard' / 'app.py'),
                '--server.port', str(args.port),
                '--server.headless', 'true'
            ], cwd=project_root)
            
        elif args.command == 'predict':
            print(f"🔍 Analyzing asteroid: {args.asteroid_id}")
            cmd = [sys.executable, str(project_root / 'scripts' / 'predict_cli.py'), args.asteroid_id]
            if args.output:
                cmd.extend(['--output', args.output])
            subprocess.run(cmd, cwd=project_root)
            
        elif args.command == 'pipeline':
            print("📊 Running data pipeline...")
            subprocess.run([
                sys.executable, '-c', 
                'from src.data.data_pipeline import DataPipeline; DataPipeline().run_full_pipeline()'
            ], cwd=project_root)
            
        elif args.command == 'demo':
            print("🎯 Running system demonstration...")
            subprocess.run([sys.executable, str(project_root / 'scripts' / 'quickstart.py')], cwd=project_root)
            
        elif args.command == 'test':
            print("🧪 Running test suite...")
            cmd = [sys.executable, '-m', 'pytest', 'tests/']
            if args.verbose:
                cmd.append('-v')
            subprocess.run(cmd, cwd=project_root)
            
        elif args.command == 'setup':
            print("⚙️ Setting up system...")
            subprocess.run([sys.executable, 'setup.py'], cwd=project_root)
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

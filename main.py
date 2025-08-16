#!/usr/bin/env python3
"""
Main CLI entry point for the AI-Driven Asteroid Mining Classification System.

This script provides command-line access to core functionality:
- Asteroid prediction and analysis
- Data pipeline operations
- Model training
- System setup

Usage:
    python main.py predict <asteroid_id>     # Analyze specific asteroid
    python main.py pipeline                  # Run data pipeline
    python main.py train                     # Train models
    python main.py setup                     # Initialize system
"""

import sys
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Asteroid Mining Classification System - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict asteroid mining potential')
    predict_parser.add_argument('asteroid_id', help='Asteroid designation (e.g., "2000 SG344")')
    predict_parser.add_argument('--output', '-o', help='Output file for results')
    predict_parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run data pipeline')
    pipeline_parser.add_argument('--limit', type=int, help='Limit number of asteroids to process')
    pipeline_parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument('--models', nargs='+', help='Specific models to train')
    train_parser.add_argument('--validate', action='store_true', help='Run validation after training')
    
    # Setup command
    subparsers.add_parser('setup', help='Initialize system and install dependencies')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'predict':
            from src.models.predict import main as predict_main
            sys.argv = ['predict.py', args.asteroid_id]
            if args.output:
                sys.argv.extend(['--output', args.output])
            predict_main()
            
        elif args.command == 'pipeline':
            from src.data.data_pipeline import DataPipeline
            pipeline = DataPipeline()
            limit = args.limit if hasattr(args, 'limit') else None
            use_cache = not (hasattr(args, 'no_cache') and args.no_cache)
            dataset = pipeline.create_training_dataset(limit=limit, use_cache=use_cache)
            print(f"✅ Pipeline completed. Dataset has {dataset['metadata']['total_samples']} samples")
            
        elif args.command == 'train':
            from src.models.train_models import ModelTrainer
            trainer = ModelTrainer()
            validate = hasattr(args, 'validate') and args.validate
            success = trainer.run_complete_training(validate_performance=validate)
            if success:
                print("✅ Model training completed successfully")
            else:
                print("❌ Model training failed")
                sys.exit(1)
                
        elif args.command == 'setup':
            import setup
            setup.main()
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

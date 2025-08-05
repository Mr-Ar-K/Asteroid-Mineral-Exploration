"""
Main setup and execution script for the Asteroid Mining Classification Dashboard.
"""
import sys
import os
from pathlib import Path
import argparse
import logging

# Add src to Python path
project_root = Path(__file__).parent.parent  # Go up one level from scripts/
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.utils.logging_config import setup_logging
from src.utils.config import config

def setup_project():
    """Setup project directories and configurations."""
    # Create necessary directories
    directories = [
        "data/cache",
        "models",
        "logs",
        "outputs"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Initialize logging
    logger = setup_logging("INFO", "logs")
    logger.info("Project setup completed successfully")
    
    print("✓ Project setup completed!")
    return True

def run_data_pipeline():
    """Run the data pipeline to fetch and process asteroid data."""
    print("🚀 Running data pipeline...")
    
    try:
        from src.data.data_pipeline import DataPipeline
        
        pipeline = DataPipeline()
        dataset = pipeline.create_training_dataset(limit=500, use_cache=True)
        
        if dataset:
            print(f"✓ Data pipeline completed successfully!")
            print(f"  - Total samples: {dataset['metadata']['total_samples']}")
            print(f"  - Features: {dataset['metadata']['n_features']}")
            print(f"  - Training samples: {dataset['metadata']['train_samples']}")
            return True
        else:
            print("❌ Data pipeline failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in data pipeline: {e}")
        return False

def train_models():
    """Train the machine learning models."""
    print("🤖 Training machine learning models...")
    
    try:
        from src.models.train_models import ModelTrainer
        
        trainer = ModelTrainer()
        success = trainer.run_complete_training(
            data_file=None,
            perform_tuning=False,  # Set to True for hyperparameter tuning
            validate_performance=True
        )
        
        if success:
            print("✓ Model training completed successfully!")
            return True
        else:
            print("❌ Model training failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return False

def run_dashboard():
    """Run the Streamlit dashboard."""
    print("🌐 Starting Streamlit dashboard...")
    
    try:
        import subprocess
        
        dashboard_path = project_root / "src" / "dashboard" / "app.py"
        
        # Run Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ]
        
        print("Dashboard will be available at: http://localhost:8501")
        subprocess.run(cmd)
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return False

def test_prediction():
    """Test the prediction system with a sample asteroid."""
    print("🔍 Testing prediction system...")
    
    try:
        from src.models.predict import AsteroidPredictor
        
        predictor = AsteroidPredictor()
        result = predictor.predict_single_asteroid("2000 SG344")
        
        if result:
            print("✓ Prediction test successful!")
            print(f"  - Asteroid: {result['designation']}")
            print(f"  - Mining Score: {result['mining_assessment']['mining_potential_score']:.3f}")
            print(f"  - Category: {result['mining_assessment']['mining_category']}")
            return True
        else:
            print("❌ Prediction test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in prediction test: {e}")
        return False

def run_full_pipeline():
    """Run the complete pipeline from setup to trained models."""
    print("🚀 Running complete AI-Driven Asteroid Mining Classification pipeline...")
    print("=" * 80)
    
    steps = [
        ("Project Setup", setup_project),
        ("Data Pipeline", run_data_pipeline),
        ("Model Training", train_models),
        ("Prediction Test", test_prediction)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 Step: {step_name}")
        print("-" * 40)
        
        success = step_func()
        
        if not success:
            print(f"\n❌ Pipeline failed at step: {step_name}")
            return False
        
        print(f"✅ {step_name} completed successfully!\n")
    
    print("=" * 80)
    print("🎉 Complete pipeline executed successfully!")
    print("\nNext steps:")
    print("1. Run 'python main.py --dashboard' to start the web interface")
    print("2. Navigate to http://localhost:8501 to access the dashboard")
    print("3. Explore asteroid mining potential analysis and mission planning tools")
    
    return True

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="AI-Driven Asteroid Mining Resource Classification Dashboard"
    )
    
    parser.add_argument(
        "--setup", 
        action="store_true", 
        help="Setup project directories and configuration"
    )
    
    parser.add_argument(
        "--data-pipeline", 
        action="store_true", 
        help="Run data pipeline to fetch and process asteroid data"
    )
    
    parser.add_argument(
        "--train", 
        action="store_true", 
        help="Train machine learning models"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Test prediction system"
    )
    
    parser.add_argument(
        "--dashboard", 
        action="store_true", 
        help="Run Streamlit dashboard"
    )
    
    parser.add_argument(
        "--full-pipeline", 
        action="store_true", 
        help="Run complete pipeline (setup, data, training, testing)"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help and run full pipeline
    if not any(vars(args).values()):
        print("🌌 AI-Driven Asteroid Mining Resource Classification Dashboard")
        print("=" * 80)
        print("No specific action specified. Running full pipeline...")
        print("Use --help to see available options.")
        print()
        return run_full_pipeline()
    
    # Execute requested actions
    if args.setup:
        return setup_project()
    
    if args.data_pipeline:
        return run_data_pipeline()
    
    if args.train:
        return train_models()
    
    if args.test:
        return test_prediction()
    
    if args.dashboard:
        return run_dashboard()
    
    if args.full_pipeline:
        return run_full_pipeline()

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

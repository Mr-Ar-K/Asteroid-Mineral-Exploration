#!/usr/bin/env python3
"""
Main setup script for the AI-Driven Asteroid Mining Classification Dashboard.
"""
import os
import sys
import subprocess
from pathlib import Path
import logging

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logging_config import setup_logging
from src.data.data_pipeline import DataPipeline
from src.models.train_models import ModelTrainer

def main():
    """Main setup function."""
    print("🚀 AI-Driven Asteroid Mining Classification Dashboard Setup")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging("INFO", "logs")
    logger.info("Starting project setup")
    
    try:
        # Step 1: Install dependencies
        print("\n📦 Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed successfully")
        
        # Step 2: Create directories
        print("\n📁 Creating project directories...")
        directories = ["data", "models", "logs", "data/cache"]
        for dir_name in directories:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
        print("✅ Directories created successfully")
        
        # Step 3: Generate sample data
        print("\n📊 Generating sample asteroid dataset...")
        pipeline = DataPipeline()
        dataset = pipeline.create_training_dataset(limit=200, use_cache=True)
        
        if dataset:
            print(f"✅ Dataset created with {dataset['metadata']['total_samples']} samples")
        else:
            print("⚠️ Dataset creation failed, but setup will continue")
        
        # Step 4: Train initial models
        print("\n🤖 Training initial ML models...")
        trainer = ModelTrainer()
        
        if dataset:
            success = trainer.run_complete_training(validate_performance=False)
            if success:
                print("✅ Models trained successfully")
            else:
                print("⚠️ Model training failed, but setup will continue")
        else:
            print("⚠️ Skipping model training due to dataset issues")
        
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the dashboard: streamlit run src/dashboard/app.py")
        print("2. Or use command line tools:")
        print("   - python src/models/predict.py --asteroid-id '2000 SG344'")
        print("   - python src/data/data_pipeline.py")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
Quick start script for development and testing.
"""
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def quick_setup():
    """Quick setup for development."""
    print("🚀 Quick Setup for Asteroid Mining Dashboard")
    print("=" * 50)
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Run main setup
    try:
        subprocess.check_call([sys.executable, "main.py", "--setup"])
        print("✅ Project setup completed!")
    except subprocess.CalledProcessError:
        print("❌ Project setup failed")
        return False
    
    print("\n🎉 Quick setup completed!")
    print("\nNext steps:")
    print("1. Run: python main.py --full-pipeline")
    print("2. Or run: python main.py --dashboard")
    
    return True

def run_demo():
    """Run a quick demo of the system."""
    print("🎯 Running quick demo...")
    
    try:
        # Run setup
        subprocess.check_call([sys.executable, "main.py", "--setup"])
        
        # Run data pipeline with small dataset
        subprocess.check_call([sys.executable, "main.py", "--data-pipeline"])
        
        # Train models
        subprocess.check_call([sys.executable, "main.py", "--train"])
        
        # Test prediction
        subprocess.check_call([sys.executable, "main.py", "--test"])
        
        print("✅ Demo completed successfully!")
        print("Run 'python main.py --dashboard' to see the web interface")
        
    except subprocess.CalledProcessError:
        print("❌ Demo failed")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick start for Asteroid Mining Dashboard")
    parser.add_argument("--setup", action="store_true", help="Quick setup")
    parser.add_argument("--demo", action="store_true", help="Run quick demo")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    else:
        quick_setup()

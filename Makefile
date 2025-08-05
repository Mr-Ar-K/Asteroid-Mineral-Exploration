# Asteroid Mining Classification System - Makefile
# ================================================

.PHONY: help install dashboard predict pipeline demo test clean lint format docker

# Default target
help: ## Show this help message
	@echo "Asteroid Mining Classification System"
	@echo "===================================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and setup environment
	@echo "🔧 Installing dependencies..."
	pip install -r requirements.txt
	@echo "⚙️ Running initial setup..."
	python setup.py

dashboard: ## Launch the Streamlit dashboard
	@echo "🚀 Launching dashboard..."
	python launcher.py dashboard

predict: ## Predict asteroid mining potential (usage: make predict ASTEROID="2000 SG344")
	@echo "🔍 Analyzing asteroid: $(ASTEROID)"
	python launcher.py predict "$(ASTEROID)"

pipeline: ## Run the complete data pipeline
	@echo "📊 Running data pipeline..."
	python launcher.py pipeline

demo: ## Run system demonstration
	@echo "🎯 Running demo..."
	python launcher.py demo

test: ## Run test suite
	@echo "🧪 Running tests..."
	python launcher.py test --verbose

lint: ## Run code linting
	@echo "🔍 Running linters..."
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	@echo "✅ Linting complete"

format: ## Format code with black
	@echo "🎨 Formatting code..."
	black src/ tests/ scripts/ --line-length=100
	@echo "✅ Formatting complete"

clean: ## Clean cache and temporary files
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/ 2>/dev/null || true
	@echo "✅ Cleanup complete"

docker: ## Build and run Docker container
	@echo "🐳 Building Docker image..."
	docker-compose up --build

docker-down: ## Stop Docker containers
	@echo "🐳 Stopping containers..."
	docker-compose down

logs: ## View application logs
	@echo "📋 Recent logs:"
	@tail -n 50 logs/asteroid_mining_*.log 2>/dev/null || echo "No logs found"

status: ## Show system status
	@echo "📊 System Status:"
	@echo "  Python: $(shell python --version)"
	@echo "  Dependencies: $(shell pip list | wc -l) packages installed"
	@echo "  Data cache: $(shell ls -la data/cache/ 2>/dev/null | wc -l) files"
	@echo "  Models: $(shell ls -la models/ 2>/dev/null | wc -l) files"
	@echo "  Logs: $(shell ls -la logs/ 2>/dev/null | wc -l) files"

# Development targets
dev-install: ## Install development dependencies
	@echo "🔧 Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest black flake8 jupyter

jupyter: ## Launch Jupyter notebook for development
	@echo "📓 Launching Jupyter..."
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Quick commands
quick-test: ## Quick test run (core functionality only)
	@echo "⚡ Quick test..."
	python -m pytest tests/test_core.py -v

quick-predict: ## Quick prediction demo (uses 2000 SG344)
	@echo "⚡ Quick prediction demo..."
	python launcher.py predict "2000 SG344"

# Help with specific commands
usage: ## Show detailed usage examples
	@echo "Usage Examples:"
	@echo "==============="
	@echo ""
	@echo "1. First time setup:"
	@echo "   make install"
	@echo ""
	@echo "2. Launch dashboard:"
	@echo "   make dashboard"
	@echo ""
	@echo "3. Analyze specific asteroid:"
	@echo "   make predict ASTEROID=\"2000 SG344\""
	@echo ""
	@echo "4. Run full data pipeline:"
	@echo "   make pipeline"
	@echo ""
	@echo "5. Run tests:"
	@echo "   make test"
	@echo ""
	@echo "6. Development setup:"
	@echo "   make dev-install"
	@echo "   make jupyter"

.PHONY: help train-all clean
.DEFAULT_GOAL := help

# Configuration
PYTHON := python
TRAIN_SCRIPT := train.py

# Get environment variables
include .env
export

help: ## Show this help message
	@echo "Trash Optimizer - Training Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

# Individual model training targets
train-efficientnet-b0: ## Train EfficientNet-B0 model
	$(PYTHON) $(TRAIN_SCRIPT) --model efficientnet_b0

train-efficientnet-b2: ## Train EfficientNet-B2 model
	$(PYTHON) $(TRAIN_SCRIPT) --model efficientnet_b2

train-efficientnet-v2-s: ## Train EfficientNetV2-S model
	$(PYTHON) $(TRAIN_SCRIPT) --model efficientnet_v2_s

train-efficientnet-v2-m: ## Train EfficientNetV2-M model
	$(PYTHON) $(TRAIN_SCRIPT) --model efficientnet_v2_m

train-convnext: ## Train ConvNeXt-Tiny model
	$(PYTHON) $(TRAIN_SCRIPT) --model convnext_tiny

train-resnet50: ## Train ResNet50 model
	$(PYTHON) $(TRAIN_SCRIPT) --model resnet50

# Training groups
train-efficientnets: train-efficientnet-b0 train-efficientnet-b2 train-efficientnet-v2-s ## Train all EfficientNet variants

train-all: train-efficientnet-b0 train-efficientnet-b2 train-efficientnet-v2-s train-convnext ## Train all models sequentially

# Model comparison
compare-all: train-all ## Train all models and generate comparison report
	@echo ""
	@echo "🔬 Generating model comparison report..."
	$(PYTHON) compare_models.py
	@echo ""
	@echo "✓ Comparison complete! Check model_comparison.json and the report above."

compare-models: ## Generate comparison report from existing results (no training)
	@echo "🔬 Generating model comparison report from existing results..."
	$(PYTHON) compare_models.py

# Quick experiments with reduced epochs
quick-test: ## Quick test run with 5 epochs
	$(PYTHON) $(TRAIN_SCRIPT) --model efficientnet_b0 --epochs 5

# Custom training with parameters
train-custom: ## Train with custom parameters (use MODEL, EPOCHS, BATCH_SIZE variables)
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL variable not set. Usage: make train-custom MODEL=efficientnet_b0"; \
		exit 1; \
	fi
	$(PYTHON) $(TRAIN_SCRIPT) --model $(MODEL) \
		$(if $(EPOCHS),--epochs $(EPOCHS)) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE)) \
		$(if $(LR),--learning-rate $(LR))

# Utility targets
list-models: ## List all available models
	@echo "Available models:"
	@$(PYTHON) -c "from config.model_configs import MODEL_CONFIGS; [print(f'  - {name}: {cfg[\"description\"]}') for name, cfg in MODEL_CONFIGS.items()]"

check-env: ## Check environment variables
	@echo "Checking environment configuration..."
	@if [ -z "$(DATASET_ROOT_DIR)" ]; then \
		echo "❌ DATASET_ROOT_DIR not set"; \
		exit 1; \
	else \
		echo "✓ DATASET_ROOT_DIR: $(DATASET_ROOT_DIR)"; \
	fi
	@if [ -z "$(RESULTS_ROOT_DIR)" ]; then \
		echo "❌ RESULTS_ROOT_DIR not set"; \
		exit 1; \
	else \
		echo "✓ RESULTS_ROOT_DIR: $(RESULTS_ROOT_DIR)"; \
	fi
	@echo "✓ Environment configured correctly"

clean: ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

.PHONY: train-efficientnet-b0 train-efficientnet-b2 train-efficientnet-v2-s train-efficientnet-v2-m
.PHONY: train-convnext train-resnet50 train-efficientnets train-custom
.PHONY: quick-test list-models check-env compare-all compare-models

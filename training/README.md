# Training Module

This module provides a flexible and extensible system for training image classification models on any classification dataset. It supports multiple architectures, configurable training parameters, and easy experiment management.

## 📁 Project Structure

```
training/
├── train.py                    # Main CLI entry point
├── Makefile                    # Training orchestration
├── config/
│   ├── base_config.py         # Base training configuration
│   └── model_configs.py       # Model-specific configs
├── models/
│   └── model_factory.py       # Model creation for all architectures
├── data/
│   ├── dataset.py             # TrashDataset class
│   ├── transforms.py          # Data augmentation strategies
│   └── loader.py              # DataLoader creation
├── training_lib/
│   ├── trainer.py             # Training loop logic
│   └── evaluator.py           # Evaluation & metrics
└── utils/
    ├── device.py              # Device detection
    └── visualization.py       # Plotting functions
```

## 🚀 Quick Start

### Using Makefile (Recommended)

```bash
# Show all available commands
make help

# Check environment setup
make check-env

# Train specific models
make train-efficientnet-b0
make train-efficientnet-v2-s
make train-convnext

# Train all EfficientNet variants
make train-efficientnets

# Train all models sequentially
make train-all

# Quick test (5 epochs)
make quick-test

# Custom training
make train-custom MODEL=efficientnet_b2 EPOCHS=100 BATCH_SIZE=64
```

### Using CLI Directly

```bash
# Basic training
python train.py --model efficientnet_b0

# Custom parameters
python train.py --model efficientnet_v2_s --epochs 100 --batch-size 64 --learning-rate 0.0002

# Custom output directory
python train.py --model convnext_tiny --output-dir ./my_experiment

# List available models
make list-models
```

## 🤖 Supported Models

| Model | Description | Default Batch Size |
|-------|-------------|-------------------|
| `efficientnet_b0` | Lightweight baseline | 32 |
| `efficientnet_b2` | More capacity | 32 |
| `efficientnet_v2_s` | Improved efficiency | 32 |
| `efficientnet_v2_m` | Larger capacity | 24 |
| `convnext_tiny` | Modern architecture | 32 |
| `resnet50` | Classic architecture | 32 |

## ⚙️ Configuration

### Environment Variables (.env)

```bash
DATASET_ROOT_DIR=/path/to/dataset
RESULTS_ROOT_DIR=/path/to/results
```

### Default Training Parameters

Edit `config/base_config.py` for global defaults:
- Batch size: 32
- Learning rate: 0.0001
- Epochs: 50
- Early stopping patience: 10
- Validation split: 25%

### Per-Model Overrides

Edit `config/model_configs.py` to customize individual models:
```python
MODEL_CONFIGS = {
    'efficientnet_v2_s': {
        'learning_rate': 0.0002,  # Custom LR
        'batch_size': 64,         # Larger batch
        'freeze_layers': -2,      # Different freezing
    }
}
```

## 📊 Output Structure

Each training run creates:

```
RESULTS_ROOT_DIR/
└── YYYYMMDD_HHMMSS_modelname/
    ├── best_model.pth              # Best checkpoint
    ├── final_model.pth             # Final model
    ├── class_mapping.txt           # Class index mapping
    ├── training_history.txt        # Epoch-by-epoch metrics
    ├── training_curves.png         # Loss/accuracy plots
    ├── confusion_matrix.png        # Confusion matrix
    └── classification_report.txt   # Per-class metrics
```

## 🔧 Advanced Usage

### Training with Custom Dataset

```bash
python train.py --model efficientnet_b0 --dataset-dir /path/to/custom/dataset
```

### Hyperparameter Tuning

```bash
# Try different learning rates
for lr in 0.0001 0.0005 0.001; do
    python train.py --model efficientnet_v2_s --learning-rate $lr --output-dir ./lr_$lr
done
```

### Batch Training Script

```bash
#!/bin/bash
# train_all_experiments.sh

models=("efficientnet_b0" "efficientnet_b2" "efficientnet_v2_s" "convnext_tiny")

for model in "${models[@]}"; do
    echo "Training $model..."
    python train.py --model $model
    if [ $? -ne 0 ]; then
        echo "Error training $model"
        exit 1
    fi
done

echo "All models trained successfully!"
```

## 🧩 Extending the System

### Adding a New Model

1. Add to `models/model_factory.py`:
```python
@staticmethod
def _create_my_model(num_classes, freeze_layers):
    model = models.my_model(weights='IMAGENET1K_V1')
    # Configure model...
    return model
```

2. Register in `SUPPORTED_MODELS` list

3. Add config to `config/model_configs.py`:
```python
MODEL_CONFIGS = {
    'my_model': {
        'learning_rate': 0.0001,
        'batch_size': 32,
        'description': 'My custom model'
    }
}
```

4. Add Makefile target (optional):
```makefile
train-my-model: ## Train my custom model
	$(PYTHON) $(TRAIN_SCRIPT) --model my_model
```

### Custom Data Augmentation

Edit `data/transforms.py`:
```python
def get_train_transforms():
    return transforms.Compose([
        # Add your custom transforms
        transforms.RandomPerspective(p=0.2),
        # ...
    ])
```

### Custom Loss Functions

Edit `training_lib/trainer.py`:
```python
# Replace CrossEntropyLoss with custom loss
self.criterion = FocalLoss(alpha=1, gamma=2)
```

## 📈 Monitoring Training

View training progress:
```bash
# Watch training history
tail -f <output_dir>/training_history.txt

# Monitor GPU usage (if using CUDA)
watch -n 1 nvidia-smi
```

## 🐛 Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 16`
- Use smaller model: `--model efficientnet_b0`

### Slow Training
- Increase batch size: `--batch-size 64`
- Increase num_workers in `config/base_config.py`

### Poor Performance
- Increase epochs: `--epochs 100`
- Adjust learning rate: `--learning-rate 0.0002`
- Check data augmentation in `data/transforms.py`

## 🧹 Maintenance

```bash
# Clean Python cache
make clean

# List all training runs
ls -lht $RESULTS_ROOT_DIR

# Remove old results (be careful!)
rm -rf $RESULTS_ROOT_DIR/YYYYMMDD_*
```

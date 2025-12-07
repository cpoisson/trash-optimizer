#!/usr/bin/env python3
"""Main training script for trash classification models."""
import argparse
import sys
from datetime import datetime
from pathlib import Path

from config.base_config import BaseConfig
from config.model_configs import get_model_config, MODEL_CONFIGS
from models.model_factory import ModelFactory
from data.loader import create_data_loaders
from data.dataset import TrashDataset
from training_lib.trainer import Trainer
from training_lib.evaluator import evaluate_per_class_metrics, save_classification_report
from utils.device import get_device
from utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    save_training_history,
    save_class_mapping
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train trash classification models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
{chr(10).join(f'  - {name}: {cfg["description"]}' for name, cfg in MODEL_CONFIGS.items())}

Examples:
  # Train a single model
  python train.py --model efficientnet_b0

  # Train with custom parameters
  python train.py --model efficientnet_v2_s --epochs 100 --batch-size 64

  # Train with specific output directory
  python train.py --model convnext_tiny --output-dir ./my_results
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help='Model architecture to train'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: from config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for training (default: from config)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (default: from config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: RESULTS_ROOT_DIR/<timestamp>_<model>)'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=None,
        help='Dataset root directory (default: from DATASET_ROOT_DIR env var)'
    )

    return parser.parse_args()


def train_model(args):
    """Train a single model.

    Args:
        args: Command line arguments.
    """
    # Validate base configuration
    BaseConfig.validate()

    # Get model configuration
    config = get_model_config(args.model, BaseConfig)

    # Override with command line arguments
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.dataset_dir is not None:
        config['dataset_root_dir'] = args.dataset_dir

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config['results_root_dir']) / f"{timestamp}_{config['model_name']}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("\n" + "=" * 80)
    print(f"Training Configuration: {config['model_name'].upper()}")
    print("=" * 80)
    print(f"Description: {config['description']}")
    print(f"Dataset: {config['dataset_root_dir']}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Early stopping patience: {config['early_stopping_patience']}")
    print("=" * 80)

    # Get device
    device = get_device()
    print(f"\n🖥️  Using device: {device}")

    # Load dataset info
    temp_dataset = TrashDataset(config['dataset_root_dir'])
    num_classes = temp_dataset.get_class_num()
    class_to_idx = temp_dataset.get_class_to_idx()
    class_names = temp_dataset.get_classes()

    print(f"\n📊 Dataset info:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {', '.join(class_names)}")

    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        root_dir=config['dataset_root_dir'],
        batch_size=config['batch_size'],
        test_split=config['test_split'],
        num_workers=config['num_workers'],
        seed=config['random_seed']
    )

    # Create model
    print(f"\n🤖 Creating model: {config['model_name']}")
    model = ModelFactory.create(
        model_name=config['model_name'],
        num_classes=num_classes,
        freeze_layers=config['freeze_layers']
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience']
    )

    # Train
    history = trainer.train()

    # Save final model
    trainer.save_final_model()

    # Evaluate per-class metrics
    class_accs, all_labels, all_preds = evaluate_per_class_metrics(
        model=trainer.model,
        val_loader=val_loader,
        class_names=class_names,
        device=device
    )

    # Save all outputs
    print("\n💾 Saving results...")
    save_class_mapping(class_to_idx, output_dir / 'categories.txt')
    save_training_history(history, output_dir / 'training_history.txt')
    plot_training_curves(history, config['model_name'], output_dir / 'training_curves.png')
    plot_confusion_matrix(all_labels, all_preds, class_names, config['model_name'],
                         output_dir / 'confusion_matrix.png')
    save_classification_report(all_labels, all_preds, class_names,
                              output_dir / 'classification_report.txt')

    print(f"\n✅ Training complete! Results saved to: {output_dir}")
    print(f"🏆 Best validation accuracy: {trainer.best_val_accuracy:.4f}\n")

    return trainer.best_val_accuracy


def main():
    """Main entry point."""
    args = parse_args()

    try:
        best_acc = train_model(args)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

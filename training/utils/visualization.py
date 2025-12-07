"""Visualization utilities for training results."""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_training_curves(history, model_name, output_path):
    """Plot training curves.

    Args:
        history: Training history dictionary.
        model_name: Name of the model.
        output_path: Path to save the plot.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Accuracy
    axes[1].plot(epochs, history['train_accuracy'], 'g-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Learning Rate
    axes[2].plot(epochs, history['learning_rates'], 'purple', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title(f'{model_name} - Learning Rate', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Training curves saved to {output_path}")


def plot_confusion_matrix(all_labels, all_preds, class_names, model_name, output_path):
    """Generate and save a confusion matrix.

    Args:
        all_labels: Ground truth labels.
        all_preds: Predicted labels.
        class_names: List of class names.
        model_name: Name of the model.
        output_path: Path to save the plot.
    """
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix saved to {output_path}")


def save_training_history(history, output_path):
    """Save training history to a text file.

    Args:
        history: Training history dictionary.
        output_path: Path to save the file.
    """
    with open(output_path, 'w') as f:
        f.write("Training History\n")
        f.write("=" * 80 + "\n\n")
        for epoch in range(len(history['train_loss'])):
            f.write(f"Epoch {epoch+1}:\n")
            f.write(f"  Train Loss: {history['train_loss'][epoch]:.6f}\n")
            f.write(f"  Train Accuracy: {history['train_accuracy'][epoch]:.6f}\n")
            f.write(f"  Val Accuracy: {history['val_accuracy'][epoch]:.6f}\n")
            f.write(f"  Learning Rate: {history['learning_rates'][epoch]:.8f}\n")
            f.write(f"  Epoch Time: {history['epoch_times'][epoch]:.2f}s\n\n")
    print(f"💾 Training history saved to {output_path}")


def save_class_mapping(class_to_idx, output_path):
    """Save the class to index mapping.

    Args:
        class_to_idx: Dictionary mapping class names to indices.
        output_path: Path to save the mapping.
    """
    with open(output_path, 'w') as f:
        for class_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
            f.write(f'{class_name}:{idx}\n')
    print(f"💾 Class mapping saved to {output_path}")

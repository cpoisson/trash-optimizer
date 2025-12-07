"""
Multi-Model Fine-tuning Script v4
==================================
This script combines the best practices from the simple approach with multi-model comparison.
Key features:
- Moderate data augmentation (preserves natural textures)
- Conservative learning rates for stable training
- Support for multiple architectures (EfficientNet-B0/B2, ConvNeXt-Tiny, ResNet50)
- Per-class performance metrics
- Comprehensive comparison reports
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import time
import os
from datetime import datetime
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import Counter
import json

# Load environment variables
load_dotenv()
DATASET_ROOT_DIR = os.getenv('DATASET_ROOT_DIR')
RESULTS_ROOT_DIR = os.getenv('RESULTS_ROOT_DIR')


def get_device():
    """Get the available device (CUDA, MPS, or CPU) for PyTorch operations"""
    return torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")


class TrashDataset(Dataset):
    """Custom Dataset for Trash Classification"""
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.targets = [label for _, label in self.dataset.imgs]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_classes(self):
        return self.dataset.classes

    def get_class_to_idx(self):
        return self.dataset.class_to_idx

    def get_class_num(self):
        return len(self.dataset.classes)


def analyze_class_distribution(dataset):
    """Analyze and print class distribution in the dataset"""
    targets = dataset.targets
    class_counts = Counter(targets)
    class_names = dataset.get_classes()

    print("\n📊 Class Distribution Analysis:")
    print("-" * 60)
    total_samples = len(targets)
    for idx, class_name in enumerate(class_names):
        count = class_counts[idx]
        percentage = (count / total_samples) * 100
        print(f"{class_name:30s}: {count:5d} samples ({percentage:5.2f}%)")
    print("-" * 60)
    print(f"{'Total':30s}: {total_samples:5d} samples\n")

    return class_counts


def get_data_transforms():
    """Get moderate data augmentation transforms (preserves natural textures)"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def get_data_loaders(root_dir, batch_size=32, test_split=0.25):
    """Create data loaders for training and validation

    Args:
        root_dir: Root directory containing the dataset
        batch_size: Batch size for training
        test_split: Fraction of data to use for validation (increased to 25% for better minority class eval)
    """
    train_transform, val_transform = get_data_transforms()

    # Create separate dataset instances
    train_dataset = TrashDataset(root_dir=root_dir, transform=train_transform)
    val_dataset = TrashDataset(root_dir=root_dir, transform=val_transform)

    # Analyze class distribution
    print("\n🔍 Analyzing dataset...")
    analyze_class_distribution(train_dataset)

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))

    # Shuffle before splitting
    import random
    random.seed(42)
    random.shuffle(indices)

    split = int(test_split * dataset_size)
    train_indices, val_indices = indices[split:], indices[:split]

    print(f"📦 Dataset split: {len(train_indices)} training, {len(val_indices)} validation samples")

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.get_classes()


def get_model(model_name, num_classes):
    """Get a pre-trained model and modify it for custom number of classes

    Args:
        model_name: One of 'efficientnet_b0', 'efficientnet_b2', 'convnext_tiny', 'resnet50'
        num_classes: Number of output classes

    Returns:
        model: Modified PyTorch model
    """
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        # Freeze most layers initially
        for param in model.features[:-3].parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(weights='IMAGENET1K_V1')
        for param in model.features[:-3].parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights='IMAGENET1K_V1')
        for param in model.features[:-2].parameters():
            param.requires_grad = False
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V2')
        # Freeze early layers
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def evaluate_per_class_metrics(model, val_loader, class_names, device):
    """Evaluate and display per-class performance metrics"""
    model.eval()

    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for label, pred in zip(labels, predicted):
                class_name = class_names[label]
                class_total[class_name] += 1
                if label == pred:
                    class_correct[class_name] += 1

    print("\n📊 Per-Class Performance Metrics:")
    print("-" * 80)
    print(f"{'Class Name':30s} {'Accuracy':>10s} {'Correct/Total':>15s}")
    print("-" * 80)

    class_accuracies = {}
    for name in sorted(class_names):
        if class_total[name] > 0:
            acc = 100 * class_correct[name] / class_total[name]
            class_accuracies[name] = acc
            print(f"{name:30s} {acc:9.2f}% {class_correct[name]:5d}/{class_total[name]:5d}")
        else:
            class_accuracies[name] = 0.0
            print(f"{name:30s} {'N/A':>10s} {'0/0':>15s}")

    print("-" * 80)
    overall_acc = 100 * sum(class_correct.values()) / sum(class_total.values())
    print(f"{'Overall Accuracy':30s} {overall_acc:9.2f}%")
    print("-" * 80 + "\n")

    return class_accuracies, all_labels, all_preds


def fine_tune_model(
    model_name,
    num_classes,
    train_loader,
    val_loader,
    class_names,
    num_epochs=50,
    learning_rate=0.0001,
    early_stopping_patience=10,
    output_dir='.'
):
    """Fine-tune a model on the custom dataset

    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        train_loader: Training data loader
        val_loader: Validation data loader
        class_names: List of class names
        num_epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
        early_stopping_patience: Epochs to wait before early stopping
        output_dir: Directory to save outputs
    """
    device = get_device()
    print(f"\n🤖 Training {model_name.upper()}")
    print(f"🖥️  Using device: {device}")

    model = get_model(model_name, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'epoch_times': [],
        'learning_rates': []
    }

    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    training_start = time.time()

    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"⏱️  Early stopping patience: {early_stopping_patience} epochs\n")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.sampler)
        train_accuracy = train_correct / train_total

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        current_lr = optimizer.param_groups[0]['lr']

        # Learning rate scheduling
        scheduler.step(val_accuracy)

        # Record history
        epoch_time = time.time() - epoch_start
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['epoch_times'].append(epoch_time)
        history['learning_rates'].append(current_lr)

        # Print epoch summary
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"| Loss: {epoch_loss:.4f} "
              f"| Train Acc: {train_accuracy:.4f} "
              f"| Val Acc: {val_accuracy:.4f} "
              f"| LR: {current_lr:.6f} "
              f"| Time: {epoch_time:.1f}s")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ New best model! Val accuracy: {val_accuracy:.4f}")
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break

    training_time = time.time() - training_start

    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))

    print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
    print(f"🏆 Best validation accuracy: {best_val_accuracy:.4f}")

    # Evaluate per-class metrics
    class_accuracies, all_labels, all_preds = evaluate_per_class_metrics(
        model, val_loader, class_names, device
    )

    return model, history, best_val_accuracy, class_accuracies, (all_labels, all_preds)


def save_model(model, path='model.pth'):
    """Save the fine-tuned model"""
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to {path}")


def save_class_mapping(class_to_idx, path='class_mapping.txt'):
    """Save the class to index mapping"""
    with open(path, 'w') as f:
        for class_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
            f.write(f'{class_name}:{idx}\n')
    print(f"💾 Class mapping saved to {path}")


def save_history(history, path='training_history.txt'):
    """Save the training history to a text file"""
    with open(path, 'w') as f:
        f.write("Training History\n")
        f.write("=" * 80 + "\n\n")
        for epoch in range(len(history['train_loss'])):
            f.write(f"Epoch {epoch+1}:\n")
            f.write(f"  Train Loss: {history['train_loss'][epoch]:.6f}\n")
            f.write(f"  Train Accuracy: {history['train_accuracy'][epoch]:.6f}\n")
            f.write(f"  Val Accuracy: {history['val_accuracy'][epoch]:.6f}\n")
            f.write(f"  Learning Rate: {history['learning_rates'][epoch]:.8f}\n")
            f.write(f"  Epoch Time: {history['epoch_times'][epoch]:.2f}s\n\n")
    print(f"💾 Training history saved to {path}")


def plot_training_curves(history, model_name, path='training_curves.png'):
    """Plot training curves"""
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
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Training curves saved to {path}")


def generate_confusion_matrix(all_labels, all_preds, class_names, model_name, path='confusion_matrix.png'):
    """Generate and save a confusion matrix"""
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
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix saved to {path}")


def save_classification_report(all_labels, all_preds, class_names, path='classification_report.txt'):
    """Save detailed classification report"""
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    with open(path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
    print(f"📊 Classification report saved to {path}")


def plot_model_comparison(results, path='model_comparison.png'):
    """Plot comparison of all models"""
    models = list(results.keys())
    accuracies = [results[m]['best_accuracy'] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1: Overall accuracy comparison
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax1.bar(models, accuracies, color=colors[:len(models)])
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_title('Model Comparison - Overall Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: Per-class accuracy heatmap for problematic classes
    problematic_classes = ['wood', 'vegetation', 'mirror', 'textile_trash']
    available_classes = [c for c in problematic_classes if c in results[models[0]]['class_accuracies']]

    if available_classes:
        heatmap_data = []
        for model in models:
            row = [results[model]['class_accuracies'].get(c, 0) for c in available_classes]
            heatmap_data.append(row)

        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn',
                   xticklabels=available_classes, yticklabels=models,
                   vmin=0, vmax=100, ax=ax2)
        ax2.set_title('Per-Class Accuracy (%) - Focus Classes', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylabel('Model', fontsize=12)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Model comparison saved to {path}")


def save_comparison_report(results, path='comparison_report.txt'):
    """Save comprehensive comparison report"""
    with open(path, 'w') as f:
        f.write("Multi-Model Comparison Report\n")
        f.write("=" * 80 + "\n\n")

        # Overall rankings
        f.write("Overall Accuracy Rankings:\n")
        f.write("-" * 80 + "\n")
        sorted_models = sorted(results.items(), key=lambda x: x[1]['best_accuracy'], reverse=True)
        for rank, (model, data) in enumerate(sorted_models, 1):
            f.write(f"{rank}. {model:20s}: {data['best_accuracy']:.4f}\n")

        f.write("\n" + "=" * 80 + "\n\n")

        # Per-class analysis
        f.write("Per-Class Accuracy Analysis:\n")
        f.write("-" * 80 + "\n\n")

        all_classes = sorted(list(sorted_models[0][1]['class_accuracies'].keys()))

        for class_name in all_classes:
            f.write(f"\n{class_name}:\n")
            class_results = [(m, data['class_accuracies'][class_name])
                           for m, data in sorted_models]
            class_results.sort(key=lambda x: x[1], reverse=True)

            for model, acc in class_results:
                f.write(f"  {model:20s}: {acc:6.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"📊 Comparison report saved to {path}")


if __name__ == '__main__':
    if DATASET_ROOT_DIR is None:
        raise ValueError("DATASET_ROOT_DIR environment variable is not set")
    if RESULTS_ROOT_DIR is None:
        raise ValueError("RESULTS_ROOT_DIR environment variable is not set")

    print("\n" + "=" * 80)
    print("Multi-Model Fine-tuning Script v4")
    print("=" * 80)

    # Configuration
    MODELS_TO_TRAIN = ['efficientnet_b0', 'efficientnet_b2', 'convnext_tiny']
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    TEST_SPLIT = 0.25

    print(f"\n📁 Dataset directory: {DATASET_ROOT_DIR}")
    print(f"💾 Results directory: {RESULTS_ROOT_DIR}")
    print(f"\n⚙️  Configuration:")
    print(f"  Models: {', '.join(MODELS_TO_TRAIN)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max epochs: {NUM_EPOCHS}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Validation split: {TEST_SPLIT:.1%}")

    # Load dataset info
    temp_dataset = TrashDataset(DATASET_ROOT_DIR)
    num_classes = temp_dataset.get_class_num()
    class_to_idx = temp_dataset.get_class_to_idx()

    print(f"\n📊 Dataset info:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {', '.join(temp_dataset.get_classes())}")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = os.path.join(RESULTS_ROOT_DIR, f"{timestamp}_multimodel_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    print(f"\n📂 Comparison output directory: {comparison_dir}")

    # Prepare data loaders (shared across all models)
    train_loader, val_loader, class_names = get_data_loaders(
        DATASET_ROOT_DIR,
        batch_size=BATCH_SIZE,
        test_split=TEST_SPLIT
    )

    # Train all models
    results = {}

    for model_name in MODELS_TO_TRAIN:
        print("\n" + "=" * 80)

        # Create model-specific output directory
        model_dir = os.path.join(comparison_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Train model
        model, history, best_acc, class_accs, (labels, preds) = fine_tune_model(
            model_name=model_name,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            class_names=class_names,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            output_dir=model_dir
        )

        # Save model outputs
        save_model(model, path=os.path.join(model_dir, 'final_model.pth'))
        save_class_mapping(class_to_idx, path=os.path.join(model_dir, 'class_mapping.txt'))
        save_history(history, path=os.path.join(model_dir, 'training_history.txt'))
        plot_training_curves(history, model_name, path=os.path.join(model_dir, 'training_curves.png'))
        generate_confusion_matrix(labels, preds, class_names, model_name,
                                 path=os.path.join(model_dir, 'confusion_matrix.png'))
        save_classification_report(labels, preds, class_names,
                                  path=os.path.join(model_dir, 'classification_report.txt'))

        # Store results
        results[model_name] = {
            'best_accuracy': best_acc,
            'class_accuracies': class_accs,
            'history': history
        }

        print(f"\n✅ {model_name} completed and saved to {model_dir}")

    # Generate comparison visualizations
    print("\n" + "=" * 80)
    print("Generating comparison reports...")
    print("=" * 80 + "\n")

    plot_model_comparison(results, path=os.path.join(comparison_dir, 'model_comparison.png'))
    save_comparison_report(results, path=os.path.join(comparison_dir, 'comparison_report.txt'))

    # Save results as JSON
    results_json = {}
    for model, data in results.items():
        results_json[model] = {
            'best_accuracy': float(data['best_accuracy']),
            'class_accuracies': {k: float(v) for k, v in data['class_accuracies'].items()}
        }

    with open(os.path.join(comparison_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    print("\n" + "=" * 80)
    print("🎉 All models trained successfully!")
    print("=" * 80)
    print(f"\n📂 All results saved to: {comparison_dir}")
    print("\n🏆 Final Rankings:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_accuracy'], reverse=True)
    for rank, (model, data) in enumerate(sorted_results, 1):
        print(f"  {rank}. {model:20s}: {data['best_accuracy']:.4f}")
    print("\n")

"""
Multi-Model Fine-tuning Script v4
==================================
This script trains and compares multiple modern architectures:
- EfficientNet-B2 (baseline)
- EfficientNetV2-S (faster training)
- ConvNeXt-Tiny (best accuracy)

Features:
- Trains multiple models in sequence
- Generates comparative analysis
- All v3 enhancements included
- Weighted sampling, progressive unfreezing, enhanced augmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
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
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CustomDataset(Dataset):
    """Custom Dataset wrapper for trash classification"""
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


def get_enhanced_transforms():
    """Get enhanced data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def get_model(model_name, num_classes):
    """Load model based on name

    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes

    Returns:
        Initialized model with replaced classifier
    """
    print(f"\n🤖 Loading {model_name}...")

    if model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        print(f"   Parameters: 9.2M | Input: 260x260")

    elif model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        print(f"   Parameters: 21M | Input: 384x384 | Faster training!")

    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights='IMAGENET1K_V1')
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        print(f"   Parameters: 28M | Input: 224x224 | Best accuracy!")

    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V2')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        print(f"   Parameters: 25M | Classic architecture")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def get_model_info(model_name):
    """Get model-specific configuration"""
    configs = {
        'efficientnet_b2': {
            'params': '9.2M',
            'input_size': 260,
            'expected_acc': '89-92%',
            'speed': 'Fast'
        },
        'efficientnet_v2_s': {
            'params': '21M',
            'input_size': 384,
            'expected_acc': '90-93%',
            'speed': 'Very Fast'
        },
        'convnext_tiny': {
            'params': '28M',
            'input_size': 224,
            'expected_acc': '91-94%',
            'speed': 'Fast'
        },
        'resnet50': {
            'params': '25M',
            'input_size': 224,
            'expected_acc': '88-91%',
            'speed': 'Fast'
        }
    }
    return configs.get(model_name, {})


def analyze_class_distribution(dataset):
    """Analyze and print class distribution in the dataset"""
    targets = [label for _, label in dataset.dataset.imgs]
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


def get_balanced_sampler(dataset, indices):
    """Create a weighted sampler for balanced training"""
    targets = [dataset.targets[i] for i in indices]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[dataset.targets[i]] for i in indices]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(indices),
        replacement=True
    )

    return sampler


def get_class_weights(dataset, indices):
    """Calculate class weights for weighted loss function"""
    targets = [dataset.targets[i] for i in indices]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    return torch.FloatTensor(class_weights)


def get_data_loaders(root_dir, batch_size=64, test_split=0.15, use_balanced_sampling=True):
    """Create enhanced data loaders with balanced sampling"""
    train_transform, val_transform = get_enhanced_transforms()

    train_dataset = CustomDataset(root_dir=root_dir, transform=train_transform)
    val_dataset = CustomDataset(root_dir=root_dir, transform=val_transform)

    print("\n🔍 Analyzing dataset...")
    analyze_class_distribution(train_dataset)

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))

    import random
    random.seed(42)
    random.shuffle(indices)

    split = int(test_split * dataset_size)
    train_indices, val_indices = indices[split:], indices[:split]

    print(f"📦 Dataset split: {len(train_indices)} training, {len(val_indices)} validation samples")

    if use_balanced_sampling:
        print("⚖️  Using weighted sampling for balanced training")
        train_sampler = get_balanced_sampler(train_dataset, train_indices)
    else:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    class_weights = get_class_weights(train_dataset, train_indices)

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

    return train_loader, val_loader, class_weights


def progressive_unfreeze(model, model_name, epoch, strategy='gradual'):
    """Progressively unfreeze layers during training"""
    if strategy == 'gradual':
        if epoch == 0:
            # Phase 1: Train only classifier
            if 'efficientnet' in model_name:
                for param in model.features.parameters():
                    param.requires_grad = False
                for param in model.classifier.parameters():
                    param.requires_grad = True
            elif 'convnext' in model_name:
                for param in model.features.parameters():
                    param.requires_grad = False
                for param in model.classifier.parameters():
                    param.requires_grad = True
            elif 'resnet' in model_name:
                for name, param in model.named_parameters():
                    if 'fc' not in name:
                        param.requires_grad = False
            print("🔒 Phase 1: Training classifier only")

        elif epoch == 10:
            # Phase 2: Unfreeze last layers
            if 'efficientnet' in model_name:
                for param in model.features[-3:].parameters():
                    param.requires_grad = True
            elif 'convnext' in model_name:
                for param in model.features[-2:].parameters():
                    param.requires_grad = True
            elif 'resnet' in model_name:
                for param in model.layer4.parameters():
                    param.requires_grad = True
            print("🔓 Phase 2: Unfreezing last blocks")

        elif epoch == 20:
            # Phase 3: Unfreeze all
            for param in model.parameters():
                param.requires_grad = True
            print("🔓 Phase 3: Unfreezing all layers")


def evaluate_per_class_metrics(model, val_loader, class_names, device):
    """Evaluate and display per-class performance metrics"""
    model.eval()

    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

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

    print("-" * 80)
    overall_acc = 100 * sum(class_correct.values()) / sum(class_total.values())
    print(f"{'Overall Accuracy':30s} {overall_acc:9.2f}%")
    print("-" * 80 + "\n")

    return class_accuracies


def train_model(
    model_name,
    model,
    num_classes,
    train_loader,
    val_loader,
    class_weights,
    class_names,
    num_epochs=100,
    learning_rate=0.001,
    early_stopping_patience=15,
    output_dir='.',
    use_focal_loss=False
):
    """Train a single model with all enhancements"""
    device = get_device()
    print(f"\n🖥️  Using device: {device}")

    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None

    model = model.to(device)

    class_weights = class_weights.to(device)
    if use_focal_loss:
        criterion = FocalLoss(alpha=1, gamma=2)
        print("📉 Using Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("📉 Using Weighted Cross Entropy Loss")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'val_loss': [],
        'learning_rate': [],
        'epoch_times': [],
        'per_class_accuracy': []
    }

    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    total_training_time = 0

    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"⏱️  Early stopping patience: {early_stopping_patience} epochs\n")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        progressive_unfreeze(model, model_name, epoch, strategy='gradual')

        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.sampler)
        val_accuracy = correct / total
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step()

        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)

        epoch_time = time.time() - epoch_start_time
        history['epoch_times'].append(epoch_time)
        total_training_time += epoch_time

        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"| Train Loss: {epoch_loss:.4f} "
              f"| Train Acc: {train_accuracy:.4f} "
              f"| Val Loss: {val_loss:.4f} "
              f"| Val Acc: {val_accuracy:.4f} "
              f"| LR: {current_lr:.6f} "
              f"| Time: {epoch_time:.1f}s")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best model saved! Validation accuracy: {val_accuracy:.4f}")

            class_accuracies = evaluate_per_class_metrics(model, val_loader, class_names, device)
            history['per_class_accuracy'].append(class_accuracies)
        else:
            epochs_without_improvement += 1
            history['per_class_accuracy'].append({})

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            print(f"🏆 Best validation accuracy: {best_val_accuracy:.4f}")
            break

    print(f"\n✅ Training completed!")
    print(f"🏆 Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"⏱️  Total training time: {total_training_time/3600:.2f} hours")

    # Add summary statistics
    history['best_val_accuracy'] = best_val_accuracy
    history['total_training_time'] = total_training_time
    history['total_epochs'] = len(history['train_loss'])

    return model, history, best_val_accuracy


def plot_model_comparison(all_results, save_path):
    """Plot comparison of all trained models"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # Plot 1: Best Validation Accuracy
    models = list(all_results.keys())
    best_accs = [all_results[m]['best_val_accuracy'] * 100 for m in models]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    axes[0, 0].bar(models, best_accs, color=colors[:len(models)])
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Best Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, acc in enumerate(best_accs):
        axes[0, 0].text(i, acc + 0.5, f'{acc:.2f}%', ha='center', fontweight='bold')

    # Plot 2: Training Time
    training_times = [all_results[m]['total_training_time'] / 3600 for m in models]
    axes[0, 1].bar(models, training_times, color=colors[:len(models)])
    axes[0, 1].set_ylabel('Time (hours)', fontsize=12)
    axes[0, 1].set_title('Total Training Time', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, t in enumerate(training_times):
        axes[0, 1].text(i, t + 0.1, f'{t:.2f}h', ha='center', fontweight='bold')

    # Plot 3: Validation Accuracy over Epochs
    for model_name in models:
        history = all_results[model_name]['history']
        epochs = range(1, len(history['val_accuracy']) + 1)
        axes[1, 0].plot(epochs, [acc * 100 for acc in history['val_accuracy']],
                       label=model_name, linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[1, 0].set_title('Validation Accuracy over Epochs', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Parameter Count vs Accuracy
    param_counts = [float(get_model_info(m)['params'].replace('M', '')) for m in models]
    axes[1, 1].scatter(param_counts, best_accs, s=200, c=colors[:len(models)], alpha=0.6)
    for i, model_name in enumerate(models):
        axes[1, 1].annotate(model_name, (param_counts[i], best_accs[i]),
                          fontsize=10, ha='center', va='bottom')
    axes[1, 1].set_xlabel('Parameters (Millions)', fontsize=12)
    axes[1, 1].set_ylabel('Best Accuracy (%)', fontsize=12)
    axes[1, 1].set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Model comparison saved to {save_path}")


def save_comparison_report(all_results, class_names, save_path):
    """Save detailed comparison report"""
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Multi-Model Training Comparison Report\n")
        f.write("=" * 80 + "\n\n")

        f.write("Summary\n")
        f.write("-" * 80 + "\n")
        for model_name, results in all_results.items():
            info = get_model_info(model_name)
            f.write(f"\n{model_name.upper()}\n")
            f.write(f"  Parameters: {info['params']}\n")
            f.write(f"  Best Val Accuracy: {results['best_val_accuracy']*100:.2f}%\n")
            f.write(f"  Training Time: {results['total_training_time']/3600:.2f} hours\n")
            f.write(f"  Total Epochs: {results['total_epochs']}\n")
            f.write(f"  Expected Range: {info['expected_acc']}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Recommendation\n")
        f.write("-" * 80 + "\n")

        # Find best model
        best_model = max(all_results.items(), key=lambda x: x[1]['best_val_accuracy'])
        f.write(f"\n🏆 Best Model: {best_model[0]}\n")
        f.write(f"   Accuracy: {best_model[1]['best_val_accuracy']*100:.2f}%\n")
        f.write(f"   Training Time: {best_model[1]['total_training_time']/3600:.2f} hours\n")

        # Find fastest model
        fastest_model = min(all_results.items(), key=lambda x: x[1]['total_training_time'])
        f.write(f"\n⚡ Fastest Training: {fastest_model[0]}\n")
        f.write(f"   Training Time: {fastest_model[1]['total_training_time']/3600:.2f} hours\n")
        f.write(f"   Accuracy: {fastest_model[1]['best_val_accuracy']*100:.2f}%\n")

    print(f"📄 Comparison report saved to {save_path}")


if __name__ == '__main__':

    if DATASET_ROOT_DIR is None:
        raise ValueError("DATASET_ROOT_DIR environment variable is not set")
    if RESULTS_ROOT_DIR is None:
        raise ValueError("RESULTS_ROOT_DIR environment variable is not set")

    print("=" * 80)
    print("🗑️  Multi-Model Trash Classification Training - Version 4")
    print("=" * 80)

    # Configuration
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    TEST_SPLIT = 0.15
    USE_FOCAL_LOSS = False
    USE_BALANCED_SAMPLING = True

    # Models to train and compare
    MODELS_TO_TRAIN = [
        'efficientnet_b2',      # Baseline
        'efficientnet_v2_s',    # Faster
        'convnext_tiny',        # Best accuracy
    ]

    print(f"\n📋 Models to train: {', '.join(MODELS_TO_TRAIN)}")
    print(f"📂 Loading dataset from: {DATASET_ROOT_DIR}")

    # Create comparison output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = os.path.join(RESULTS_ROOT_DIR, f"{timestamp}_multimodel_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    print(f"📁 Comparison directory: {comparison_dir}")

    # Get data loaders (shared across all models)
    train_loader, val_loader, class_weights = get_data_loaders(
        DATASET_ROOT_DIR,
        batch_size=BATCH_SIZE,
        test_split=TEST_SPLIT,
        use_balanced_sampling=USE_BALANCED_SAMPLING
    )

    # Get dataset info
    temp_dataset = CustomDataset(DATASET_ROOT_DIR)
    num_classes = temp_dataset.get_class_num()
    class_names = temp_dataset.get_classes()

    print(f"\n🏷️  Training on {num_classes} classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    # Train all models
    all_results = {}

    for model_name in MODELS_TO_TRAIN:
        print("\n" + "=" * 80)
        print(f"🚀 Training {model_name}")
        print("=" * 80)

        # Create model-specific output directory
        model_output_dir = os.path.join(comparison_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # Load model
        model = get_model(model_name, num_classes)

        # Train model
        trained_model, history, best_acc = train_model(
            model_name=model_name,
            model=model,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            class_names=class_names,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            output_dir=model_output_dir,
            use_focal_loss=USE_FOCAL_LOSS
        )

        # Save model
        torch.save(trained_model.state_dict(),
                  os.path.join(model_output_dir, 'final_model.pth'))

        # Store results
        all_results[model_name] = {
            'history': history,
            'best_val_accuracy': best_acc,
            'total_training_time': history['total_training_time'],
            'total_epochs': history['total_epochs']
        }

        print(f"\n✅ {model_name} training completed!")
        print(f"   Best accuracy: {best_acc*100:.2f}%")
        print(f"   Training time: {history['total_training_time']/3600:.2f} hours")

    # Generate comparison visualizations
    print("\n" + "=" * 80)
    print("📊 Generating Model Comparison")
    print("=" * 80)

    plot_model_comparison(all_results,
                         os.path.join(comparison_dir, 'model_comparison.png'))

    save_comparison_report(all_results, class_names,
                          os.path.join(comparison_dir, 'comparison_report.txt'))

    # Save results as JSON
    results_json = {
        model: {
            'best_val_accuracy': float(results['best_val_accuracy']),
            'total_training_time': float(results['total_training_time']),
            'total_epochs': int(results['total_epochs']),
            'model_info': get_model_info(model)
        }
        for model, results in all_results.items()
    }

    with open(os.path.join(comparison_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Print final summary
    print("\n" + "=" * 80)
    print("🏁 Multi-Model Training Summary")
    print("=" * 80)

    best_model = max(all_results.items(), key=lambda x: x[1]['best_val_accuracy'])
    fastest_model = min(all_results.items(), key=lambda x: x[1]['total_training_time'])

    print(f"\n🏆 Best Model: {best_model[0]}")
    print(f"   Accuracy: {best_model[1]['best_val_accuracy']*100:.2f}%")
    print(f"   Time: {best_model[1]['total_training_time']/3600:.2f} hours")

    print(f"\n⚡ Fastest Training: {fastest_model[0]}")
    print(f"   Time: {fastest_model[1]['total_training_time']/3600:.2f} hours")
    print(f"   Accuracy: {fastest_model[1]['best_val_accuracy']*100:.2f}%")

    print(f"\n📁 All results saved to: {comparison_dir}")
    print("\n" + "=" * 80)
    print("✅ Multi-model training completed successfully!")
    print("=" * 80)

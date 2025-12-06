"""
Enhanced EfficientNet Fine-tuning Script v3
============================================
This script implements advanced techniques for training on imbalanced datasets:
- Weighted sampling for balanced training
- Class-weighted loss function
- Progressive unfreezing strategy
- Enhanced data augmentation
- Mixed precision training
- Per-class performance metrics
- Early stopping with model checkpointing
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

# Load environment variables
load_dotenv()
DATASET_ROOT_DIR = os.getenv('DATASET_ROOT_DIR')
RESULTS_ROOT_DIR = os.getenv('RESULTS_ROOT_DIR')


def get_device():
    """Get the available device (CUDA, MPS, or CPU) for PyTorch operations"""
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance

    Focal loss applies a modulating term to the cross entropy loss to focus
    learning on hard misclassified examples.
    """
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
    """Create a weighted sampler for balanced training

    Args:
        dataset: The dataset containing targets
        indices: Indices of samples to include in the sampler

    Returns:
        WeightedRandomSampler configured for balanced sampling
    """
    targets = [dataset.targets[i] for i in indices]
    class_counts = np.bincount(targets)

    # Calculate weights: inverse of class frequency
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[dataset.targets[i]] for i in indices]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(indices),
        replacement=True
    )

    return sampler


def get_class_weights(dataset, indices):
    """Calculate class weights for weighted loss function

    Args:
        dataset: The dataset containing targets
        indices: Indices of samples to include

    Returns:
        Tensor of class weights
    """
    targets = [dataset.targets[i] for i in indices]
    class_counts = np.bincount(targets)

    # Calculate weights: inverse frequency normalized
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    return torch.FloatTensor(class_weights)


def get_data_loaders(root_dir, batch_size=64, test_split=0.15, use_balanced_sampling=True):
    """Create enhanced data loaders with balanced sampling

    Args:
        root_dir: Root directory containing the dataset
        batch_size: Batch size for training
        test_split: Fraction of data to use for validation
        use_balanced_sampling: Whether to use weighted sampling for balanced training

    Returns:
        train_loader, val_loader, class_weights
    """
    train_transform, val_transform = get_enhanced_transforms()

    # Create separate dataset instances
    train_dataset = CustomDataset(root_dir=root_dir, transform=train_transform)
    val_dataset = CustomDataset(root_dir=root_dir, transform=val_transform)

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

    # Create samplers
    if use_balanced_sampling:
        print("⚖️  Using weighted sampling for balanced training")
        train_sampler = get_balanced_sampler(train_dataset, train_indices)
    else:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    # Calculate class weights for loss function
    class_weights = get_class_weights(train_dataset, train_indices)

    # Create data loaders
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


def progressive_unfreeze(model, epoch, strategy='gradual'):
    """Progressively unfreeze layers during training

    Args:
        model: The model to unfreeze
        epoch: Current epoch number
        strategy: Unfreezing strategy ('gradual' or 'aggressive')
    """
    if strategy == 'gradual':
        if epoch == 0:
            # Phase 1: Train only classifier
            for param in model.features.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
            print("🔒 Phase 1: Training classifier only")

        elif epoch == 10:
            # Phase 2: Unfreeze last 3 blocks
            for param in model.features[-3:].parameters():
                param.requires_grad = True
            print("🔓 Phase 2: Unfreezing last 3 blocks")

        elif epoch == 20:
            # Phase 3: Unfreeze all
            for param in model.parameters():
                param.requires_grad = True
            print("🔓 Phase 3: Unfreezing all layers")


def evaluate_per_class_metrics(model, val_loader, class_names, device):
    """Evaluate and display per-class performance metrics

    Args:
        model: The trained model
        val_loader: Validation data loader
        class_names: List of class names
        device: Device to run evaluation on

    Returns:
        Dictionary containing per-class accuracies
    """
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

    return class_accuracies


def fine_tune_efficientnet_v3(
    num_classes,
    train_loader,
    val_loader,
    class_weights,
    class_names,
    num_epochs=100,
    learning_rate=0.001,
    early_stopping_patience=15,
    output_dir='.',
    use_focal_loss=False,
    model_variant='b2'
):
    """Enhanced fine-tuning with advanced techniques

    Args:
        num_classes: Number of output classes
        train_loader: Training data loader
        val_loader: Validation data loader
        class_weights: Weights for each class
        class_names: List of class names
        num_epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
        early_stopping_patience: Epochs to wait before early stopping
        output_dir: Directory to save outputs
        use_focal_loss: Whether to use focal loss instead of cross entropy
        model_variant: EfficientNet variant ('b0', 'b1', 'b2', 'b3')
    """
    device = get_device()
    print(f"\n🖥️  Using device: {device}")

    # Support for mixed precision training
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None

    # Load appropriate model variant
    if model_variant == 'b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    elif model_variant == 'b1':
        model = models.efficientnet_b1(weights='IMAGENET1K_V1')
    elif model_variant == 'b2':
        model = models.efficientnet_b2(weights='IMAGENET1K_V1')
    elif model_variant == 'b3':
        model = models.efficientnet_b3(weights='IMAGENET1K_V1')
    else:
        raise ValueError(f"Unsupported model variant: {model_variant}")

    print(f"🤖 Using EfficientNet-{model_variant.upper()}")

    # Replace classifier
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    # Choose loss function
    class_weights = class_weights.to(device)
    if use_focal_loss:
        criterion = FocalLoss(alpha=1, gamma=2)
        print("📉 Using Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("📉 Using Weighted Cross Entropy Loss")

    # Optimizer with weight decay
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Training history
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

    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"⏱️  Early stopping patience: {early_stopping_patience} epochs\n")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Progressive unfreezing
        progressive_unfreeze(model, epoch, strategy='gradual')

        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision training
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

        # Update learning rate
        scheduler.step()

        # Record history
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)

        epoch_time = time.time() - epoch_start_time
        history['epoch_times'].append(epoch_time)

        # Print epoch summary
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"| Train Loss: {epoch_loss:.4f} "
              f"| Train Acc: {train_accuracy:.4f} "
              f"| Val Loss: {val_loss:.4f} "
              f"| Val Acc: {val_accuracy:.4f} "
              f"| LR: {current_lr:.6f} "
              f"| Time: {epoch_time:.1f}s")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best model saved! Validation accuracy: {val_accuracy:.4f}")

            # Evaluate per-class metrics for best model
            class_accuracies = evaluate_per_class_metrics(model, val_loader, class_names, device)
            history['per_class_accuracy'].append(class_accuracies)
        else:
            epochs_without_improvement += 1
            history['per_class_accuracy'].append({})

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            print(f"🏆 Best validation accuracy: {best_val_accuracy:.4f}")
            break

    print(f"\n✅ Training completed!")
    print(f"🏆 Best validation accuracy: {best_val_accuracy:.4f}")

    return model, history


def save_model(model, path='fine_tuned_efficientnet.pth'):
    """Save the fine-tuned model"""
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to {path}")


def save_class_mapping(class_to_idx, path='class_mapping.txt'):
    """Save the class to index mapping"""
    with open(path, 'w') as f:
        for class_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
            f.write(f'{class_name}:{idx}\n')
    print(f"💾 Class mapping saved to {path}")


def save_enhanced_history(history, path='training_history.txt'):
    """Save comprehensive training history"""
    with open(path, 'w') as f:
        f.write("Enhanced Training History\n")
        f.write("=" * 80 + "\n\n")

        for epoch in range(len(history['train_loss'])):
            f.write(f"Epoch {epoch+1}:\n")
            f.write(f"  Train Loss: {history['train_loss'][epoch]:.6f}\n")
            f.write(f"  Train Accuracy: {history['train_accuracy'][epoch]:.6f}\n")
            f.write(f"  Val Loss: {history['val_loss'][epoch]:.6f}\n")
            f.write(f"  Val Accuracy: {history['val_accuracy'][epoch]:.6f}\n")
            f.write(f"  Learning Rate: {history['learning_rate'][epoch]:.8f}\n")
            f.write(f"  Epoch Time: {history['epoch_times'][epoch]:.2f}s\n")

            if history['per_class_accuracy'][epoch]:
                f.write("  Per-Class Accuracy:\n")
                for class_name, acc in sorted(history['per_class_accuracy'][epoch].items()):
                    f.write(f"    {class_name}: {acc:.2f}%\n")

            f.write("\n")

    print(f"💾 Training history saved to {path}")


def plot_enhanced_training_curves(history, path='training_curves.png'):
    """Plot comprehensive training curves"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Plot 1: Training and Validation Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss over Epochs', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Training and Validation Accuracy
    axes[0, 1].plot(epochs, history['train_accuracy'], 'g-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Accuracy over Epochs', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Learning Rate Schedule
    axes[1, 0].plot(epochs, history['learning_rate'], 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Epoch Time
    axes[1, 1].plot(epochs, history['epoch_times'], 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 1].set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Training curves saved to {path}")


def generate_enhanced_confusion_matrix(model, val_loader, class_names, device, path='confusion_matrix.png'):
    """Generate and save an enhanced confusion matrix"""
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('True', fontsize=12)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)

    # Plot 2: Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('True', fontsize=12)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix saved to {path}")


def generate_classification_report(model, val_loader, class_names, device, path='classification_report.txt'):
    """Generate and save a detailed classification report"""
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    with open(path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)

    print(f"📊 Classification report saved to {path}")
    print(f"\n{report}")


if __name__ == '__main__':

    if DATASET_ROOT_DIR is None:
        raise ValueError("DATASET_ROOT_DIR environment variable is not set")
    if RESULTS_ROOT_DIR is None:
        raise ValueError("RESULTS_ROOT_DIR environment variable is not set")

    print("=" * 80)
    print("🗑️  Enhanced Trash Classification Training - Version 3")
    print("=" * 80)

    print(f"\n📂 Loading dataset from: {DATASET_ROOT_DIR}")

    # Configuration
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    TEST_SPLIT = 0.15
    MODEL_VARIANT = 'b2'  # Options: 'b0', 'b1', 'b2', 'b3'
    USE_FOCAL_LOSS = False  # Set to True to use Focal Loss
    USE_BALANCED_SAMPLING = True

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULTS_ROOT_DIR, f"{timestamp}_efficientnet_{MODEL_VARIANT}_enhanced")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Get data loaders with balanced sampling
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

    # Train the model
    fine_tuned_model, history = fine_tune_efficientnet_v3(
        num_classes=num_classes,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        class_names=class_names,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        output_dir=output_dir,
        use_focal_loss=USE_FOCAL_LOSS,
        model_variant=MODEL_VARIANT
    )

    # Save final model and artifacts
    print("\n💾 Saving model and artifacts...")
    save_model(fine_tuned_model, path=os.path.join(output_dir, 'final_model.pth'))
    save_class_mapping(temp_dataset.get_class_to_idx(), path=os.path.join(output_dir, 'class_mapping.txt'))
    save_enhanced_history(history, path=os.path.join(output_dir, 'training_history.txt'))

    # Generate visualizations and reports
    print("\n📊 Generating visualizations and reports...")
    plot_enhanced_training_curves(history, path=os.path.join(output_dir, 'training_curves.png'))

    device = get_device()
    generate_enhanced_confusion_matrix(
        fine_tuned_model, val_loader, class_names, device,
        path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    generate_classification_report(
        fine_tuned_model, val_loader, class_names, device,
        path=os.path.join(output_dir, 'classification_report.txt')
    )

    # Final evaluation
    print("\n📈 Final Model Evaluation:")
    evaluate_per_class_metrics(fine_tuned_model, val_loader, class_names, device)

    print("\n" + "=" * 80)
    print("✅ Training completed successfully!")
    print(f"📁 All results saved to: {output_dir}")
    print("=" * 80)

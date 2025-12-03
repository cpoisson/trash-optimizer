# finetune_efficientnet.py
# This script fine-tunes a pre-trained EfficientNet model on a custom dataset.

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
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Fine-tuning will based on https://www.kaggle.com/datasets/joebeachcapital/realwaste dataset
# Classifying images into the following waste categories:
# "Cardboard", "Food Organics", "Glass", "Metal",
# "Miscellaneous Trash", "Paper", "Plastic",
# "Textile Trash", "Vegetation"

# Get DATASET_ROOT_DIR from environment variable DATASET_ROOT_DIR
load_dotenv()  # Load environment variables from .env file if present
DATASET_ROOT_DIR = os.getenv('DATASET_ROOT_DIR')
RESULTS_ROOT_DIR = os.getenv('RESULTS_ROOT_DIR')

# The dataset is not organized into train/val/test splits, so we will create our own splits.
# The current is organized as follows:
# root
# ├── Cardboard
# ├── Food Organics
# ├── Glass
# ├── Metal
# ├── Miscellaneous Trash
# ├── Paper
# ├── Plastic
# ├── Textile Trash
# └── Vegetation

def get_device():
    '''Get the available device (CUDA, MPS, or CPU) for PyTorch operations'''
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class RealWasteDataset(Dataset):
    '''Custom Dataset for Real Waste Images'''
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)

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



def get_data_loaders(root_dir, batch_size=32, test_split=0.2):
    '''Create Data Loaders for Training and Validation'''
    # Separate transforms for train/val
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

    # Create separate dataset instances
    train_dataset = RealWasteDataset(root_dir=root_dir, transform=train_transform)
    val_dataset = RealWasteDataset(root_dir=root_dir, transform=val_transform)

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))

    # CRITICAL: Shuffle before splitting
    import random
    random.seed(42)
    random.shuffle(indices)

    split = int(test_split * dataset_size)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2)

    return train_loader, val_loader


def fine_tune_efficientnet(
    num_classes, train_loader, val_loader,
    num_epochs=10, learning_rate=0.0001, early_stopping_patience=10,
    output_dir='.'
):
    '''Fine-tune EfficientNet on the custom dataset. Train history is returned for further analysis if needed.'''
    device = get_device()
    print(f"Using device: {device}")

    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    # Freeze most layers initially
    for param in model.features[:-3].parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    history = {'train_loss': [], 'train_accuracy': [], 'val_accuracy': [], 'epoch_times': []}
    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f'\nStarting epoch {epoch+1}/{num_epochs}')
        time_start = time.time()

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
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

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
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}')

        # Learning rate scheduling
        scheduler.step(val_accuracy)

        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)

        # Save best model only
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_path = os.path.join(output_dir, 'best_model.pth')
            save_model(model, path=save_path)
            print(f'New best model saved with validation accuracy: {val_accuracy:.4f}')
        else:
            epochs_without_improvement += 1
            print(f'No improvement for {epochs_without_improvement} epoch(s)')

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            print(f'Best validation accuracy: {best_val_accuracy:.4f}')
            break

        time_end = time.time()
        epoch_time = time_end - time_start
        history['epoch_times'].append(epoch_time)
        print(f'Epoch {epoch+1} completed in {epoch_time:.2f} seconds.')

    return model, history

def save_model(model, path='fine_tuned_efficientnet.pth'):
    '''Save the fine-tuned model'''
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def save_class_mapping(class_to_idx, path='class_mapping.txt'):
    '''Save the class to index mapping to be opened as a dictionary later'''
    with open(path, 'w') as f:
        for class_name, idx in class_to_idx.items():
            f.write(f'{class_name}:{idx}\n')
    print(f'Class mapping saved to {path}')

def save_history(history, path='training_history.txt'):
    '''Save the training history to a text file'''
    with open(path, 'w') as f:
        for epoch in range(len(history['train_loss'])):
            f.write(f'Epoch {epoch+1}: Train Loss: {history["train_loss"][epoch]}, Train Accuracy: {history["train_accuracy"][epoch]}, Val Accuracy: {history["val_accuracy"][epoch]}\n')
    print(f'Training history saved to {path}')

def plot_training_curves(history, path='training_curves.png'):
    '''Plot training loss, training accuracy, and validation accuracy over epochs'''
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot training and validation accuracy
    ax2.plot(epochs, history['train_accuracy'], 'g-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to {path}')

def generate_confusion_matrix(model, val_loader, class_names, path='confusion_matrix.png'):
    '''Generate and save a confusion matrix for the validation set.

    Args:
        model: The trained PyTorch model to evaluate.
        val_loader: DataLoader for the validation dataset.
        class_names: List of class names for labeling the matrix axes.
        path: File path to save the confusion matrix image. Defaults to 'confusion_matrix.png'.
    '''
    device = get_device()
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

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to {path}')

if __name__ == '__main__':

    if DATASET_ROOT_DIR is None:
        raise ValueError("DATASET_ROOT_DIR environment variable is not set")
    if RESULTS_ROOT_DIR is None:
        raise ValueError("RESULTS_ROOT_DIR environment variable is not set")

    print("Loading dataset... at ", DATASET_ROOT_DIR)
    dataset = RealWasteDataset(DATASET_ROOT_DIR)

    print("Classes: ", dataset.get_classes())
    print("Class to Index Mapping: ", dataset.get_class_to_idx())
    print("Number of Classes: ", dataset.get_class_num())

    num_classes = dataset.get_class_num()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    model_name = "efficientnet_b0"
    dataset_name = "realwaste"

    output_dir = os.path.join(RESULTS_ROOT_DIR, f"{timestamp}_{model_name}_{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    train_loader, val_loader = get_data_loaders(DATASET_ROOT_DIR, batch_size=32, test_split=0.2)
    fine_tuned_model, history = fine_tune_efficientnet(num_classes, train_loader, val_loader, num_epochs=50, learning_rate=0.0001, output_dir=output_dir, early_stopping_patience=10)

    save_model(fine_tuned_model, path=os.path.join(output_dir, 'final_model.pth'))
    save_class_mapping(dataset.get_class_to_idx(), path=os.path.join(output_dir, 'class_mapping.txt'))
    save_history(history, path=os.path.join(output_dir, 'training_history.txt'))
    plot_training_curves(history, path=os.path.join(output_dir, 'training_curves.png'))
    generate_confusion_matrix(fine_tuned_model, val_loader, dataset.get_classes(), path=os.path.join(output_dir, 'confusion_matrix.png'))

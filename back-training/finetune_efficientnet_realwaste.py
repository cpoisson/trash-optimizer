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
from dotenv import load_dotenv

# Fine-tuning will based on https://www.kaggle.com/datasets/joebeachcapital/realwaste dataset
# Classifying images into the following waste categories:
# "Cardboard", "Food Organics", "Glass", "Metal",
# "Miscellaneous Trash", "Paper", "Plastic",
# "Textile Trash", "Vegetation"

# Get ROOT_DIR from environment variable DATASET_ROOT_DIR
load_dotenv()  # Load environment variables from .env file if present
ROOT_DIR = os.getenv('DATASET_ROOT_DIR')
RESULTS_DIR = os.getenv('RESULTS_ROOT_DIR')

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


def fine_tune_efficientnet(num_classes, train_loader, val_loader, num_epochs=30, learning_rate=0.0001):
    '''Fine-tune EfficientNet on the custom dataset. Train history is returned for further analysis if needed.'''
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
            save_path = 'best_efficientnet_model.pth'
            save_model(model, path=save_path)
            print(f'New best model saved with validation accuracy: {val_accuracy:.4f}')

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

if __name__ == '__main__':

    print("Loading dataset... at ", ROOT_DIR)
    dataset = RealWasteDataset(ROOT_DIR)

    print("Classes: ", dataset.get_classes())
    print("Class to Index Mapping: ", dataset.get_class_to_idx())
    print("Number of Classes: ", dataset.get_class_num())

    num_classes = dataset.get_class_num()

    train_loader, val_loader = get_data_loaders(ROOT_DIR, batch_size=32, test_split=0.2)
    fine_tuned_model, history = fine_tune_efficientnet(num_classes, train_loader, val_loader, num_epochs=50, learning_rate=0.0001)

    save_model(fine_tuned_model, path='fine_tuned_efficientnet.pth')
    save_class_mapping(dataset.get_class_to_idx(), path='class_mapping.txt')
    save_history(history, path='training_history.txt')

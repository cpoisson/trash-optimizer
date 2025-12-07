"""DataLoader creation utilities."""
import random
from torch.utils.data import DataLoader
from collections import Counter

from .dataset import TrashDataset
from .transforms import get_train_transforms, get_val_transforms


def analyze_class_distribution(dataset):
    """Analyze and print class distribution in the dataset.

    Args:
        dataset: TrashDataset instance.

    Returns:
        Counter: Class counts by index.
    """
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


def create_data_loaders(root_dir, batch_size=32, test_split=0.25, num_workers=4, seed=42):
    """Create training and validation data loaders.

    Args:
        root_dir: Root directory containing the dataset.
        batch_size: Batch size for training.
        test_split: Fraction of data to use for validation.
        num_workers: Number of worker processes for data loading.
        seed: Random seed for reproducible splits.

    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    # Create datasets with appropriate transforms
    train_dataset = TrashDataset(root_dir=root_dir, transform=get_train_transforms())
    val_dataset = TrashDataset(root_dir=root_dir, transform=get_val_transforms())

    # Analyze class distribution
    print("\n🔍 Analyzing dataset...")
    analyze_class_distribution(train_dataset)

    # Split dataset
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))

    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(indices)

    split = int(test_split * dataset_size)
    train_indices, val_indices = indices[split:], indices[:split]

    print(f"📦 Dataset split: {len(train_indices)} training, {len(val_indices)} validation samples")

    # Create samplers
    from torch.utils.data import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.get_classes()

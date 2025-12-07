"""Custom dataset for trash classification."""
from torch.utils.data import Dataset
from torchvision import datasets


class TrashDataset(Dataset):
    """Custom Dataset for Trash Classification.

    This wraps ImageFolder and provides additional utilities for class information.
    """

    def __init__(self, root_dir, transform=None):
        """Initialize the dataset.

        Args:
            root_dir: Root directory containing class subdirectories with images.
            transform: Optional transform to be applied on images.
        """
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.targets = [label for _, label in self.dataset.imgs]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_classes(self):
        """Get list of class names."""
        return self.dataset.classes

    def get_class_to_idx(self):
        """Get mapping from class name to index."""
        return self.dataset.class_to_idx

    def get_class_num(self):
        """Get number of classes."""
        return len(self.dataset.classes)

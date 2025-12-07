"""Data augmentation and transformation strategies."""
from torchvision import transforms


def get_train_transforms():
    """Get training data transforms with moderate augmentation.

    Preserves natural textures which is important for materials like wood and vegetation.

    Returns:
        transforms.Compose: Composed training transforms.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    """Get validation/test data transforms.

    Returns:
        transforms.Compose: Composed validation transforms.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

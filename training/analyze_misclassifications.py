"""
Analyze misclassifications for vegetation and wood classes
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter, defaultdict
import sys

class SimpleDataset(Dataset):
    def __init__(self, root_dir, class_name, transform=None):
        self.root_dir = Path(root_dir)
        self.class_name = class_name
        self.transform = transform

        # Get all images
        class_path = self.root_dir / class_name
        self.images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.images.extend(list(class_path.glob(ext)))

        print(f"Found {len(self.images)} images in {class_name}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, str(img_path)


def load_model_and_classes(model_path, categories_path):
    """Load trained model and class names"""
    # Load class names
    with open(categories_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    num_classes = len(class_names)

    # Load model architecture (assuming EfficientNet-B2)
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, class_names, device


def analyze_class(model, dataset, class_names, true_class_name, device):
    """Analyze predictions for a specific class"""
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_predictions = []
    all_confidences = []
    misclassified_samples = []

    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

            # Track misclassified samples
            for i, (pred, conf, path) in enumerate(zip(predictions, confidences, paths)):
                pred_class = class_names[pred.item()]
                if pred_class != true_class_name:
                    misclassified_samples.append({
                        'path': path,
                        'predicted': pred_class,
                        'confidence': conf.item()
                    })

    # Count predictions
    pred_counter = Counter([class_names[p] for p in all_predictions])

    print(f"\n{'='*80}")
    print(f"Analysis for: {true_class_name.upper()}")
    print(f"{'='*80}")
    print(f"Total samples: {len(all_predictions)}")
    print(f"Correctly classified: {pred_counter.get(true_class_name, 0)} ({pred_counter.get(true_class_name, 0)/len(all_predictions)*100:.2f}%)")
    print(f"Misclassified: {len(misclassified_samples)} ({len(misclassified_samples)/len(all_predictions)*100:.2f}%)")

    print(f"\nPrediction Distribution:")
    print(f"{'Class':<25s} {'Count':>8s} {'Percentage':>12s}")
    print(f"{'-'*50}")
    for pred_class, count in pred_counter.most_common():
        percentage = count / len(all_predictions) * 100
        marker = "✓" if pred_class == true_class_name else "✗"
        print(f"{marker} {pred_class:<23s} {count:>8d} {percentage:>11.2f}%")

    # Show top confusions
    if misclassified_samples:
        print(f"\nTop Misclassifications (by confidence):")
        sorted_misclass = sorted(misclassified_samples, key=lambda x: x['confidence'], reverse=True)
        for i, sample in enumerate(sorted_misclass[:10], 1):
            print(f"  {i}. Predicted as {sample['predicted']:20s} (conf: {sample['confidence']:.3f})")
            print(f"     {Path(sample['path']).name}")

    return pred_counter, misclassified_samples


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python analyze_misclassifications.py <model_path> <categories_path> <dataset_root>")
        print("Example: python analyze_misclassifications.py model.pth categories.txt ~/Data/trash-optimizer/datasets_processed/trash_optimizer_dataset_20251206-231833")
        sys.exit(1)

    MODEL_PATH = sys.argv[1]
    CATEGORIES_PATH = sys.argv[2]
    DATASET_ROOT = sys.argv[3]

    print(f"Loading model from: {MODEL_PATH}")
    print(f"Loading categories from: {CATEGORIES_PATH}")
    print(f"Dataset root: {DATASET_ROOT}")

    # Load model
    model, class_names, device = load_model_and_classes(MODEL_PATH, CATEGORIES_PATH)
    print(f"\nModel loaded successfully on {device}")
    print(f"Number of classes: {len(class_names)}")

    # Define transform (same as validation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Analyze vegetation
    vegetation_dataset = SimpleDataset(DATASET_ROOT, 'vegetation', transform=transform)
    veg_predictions, veg_misclass = analyze_class(model, vegetation_dataset, class_names, 'vegetation', device)

    # Analyze wood
    wood_dataset = SimpleDataset(DATASET_ROOT, 'wood', transform=transform)
    wood_predictions, wood_misclass = analyze_class(model, wood_dataset, class_names, 'wood', device)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\nTop 3 confusions for VEGETATION:")
    for pred_class, count in veg_predictions.most_common(4):
        if pred_class != 'vegetation':
            print(f"  → {pred_class}: {count} samples")

    print("\nTop 3 confusions for WOOD:")
    for pred_class, count in wood_predictions.most_common(4):
        if pred_class != 'wood':
            print(f"  → {pred_class}: {count} samples")

    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    print("1. Check if confused classes have visual similarities")
    print("2. Consider collecting more diverse images for vegetation and wood")
    print("3. Increase data augmentation for these specific classes")
    print("4. Add class-specific preprocessing or augmentation")
    print("5. Consider using focal loss or adjusting class weights significantly")

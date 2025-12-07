"""Evaluation and metrics computation."""
import torch
from sklearn.metrics import classification_report


def evaluate_per_class_metrics(model, val_loader, class_names, device):
    """Evaluate and display per-class performance metrics.

    Args:
        model: Trained PyTorch model.
        val_loader: Validation data loader.
        class_names: List of class names.
        device: Device to run evaluation on.

    Returns:
        tuple: (class_accuracies dict, all_labels list, all_preds list)
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

    return class_accuracies, all_labels, all_preds


def save_classification_report(all_labels, all_preds, class_names, output_path):
    """Save detailed classification report.

    Args:
        all_labels: Ground truth labels.
        all_preds: Predicted labels.
        class_names: List of class names.
        output_path: Path to save the report.
    """
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    with open(output_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
    print(f"📊 Classification report saved to {output_path}")

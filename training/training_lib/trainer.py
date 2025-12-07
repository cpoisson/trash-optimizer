"""Training logic and Trainer class."""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


class Trainer:
    """Trainer class for model training."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        output_dir,
        learning_rate=0.0001,
        weight_decay=1e-4,
        num_epochs=50,
        early_stopping_patience=10
    ):
        """Initialize the trainer.

        Args:
            model: PyTorch model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            device: Device to train on (cuda/mps/cpu).
            output_dir: Directory to save checkpoints and results.
            learning_rate: Initial learning rate.
            weight_decay: Weight decay for optimizer.
            num_epochs: Maximum number of training epochs.
            early_stopping_patience: Epochs to wait before early stopping.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience

        # Setup loss, optimizer, scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epoch_times': [],
            'learning_rates': []
        }

        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0

    def train_epoch(self):
        """Train for one epoch.

        Returns:
            tuple: (epoch_loss, train_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.train_loader.sampler)
        train_accuracy = train_correct / train_total

        return epoch_loss, train_accuracy

    def validate(self):
        """Validate the model.

        Returns:
            float: Validation accuracy.
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        return val_accuracy

    def train(self):
        """Run the full training loop.

        Returns:
            dict: Training history.
        """
        print(f"\n🚀 Starting training for {self.num_epochs} epochs...")
        print(f"⏱️  Early stopping patience: {self.early_stopping_patience} epochs\n")

        training_start = time.time()

        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            # Train
            epoch_loss, train_accuracy = self.train_epoch()

            # Validate
            val_accuracy = self.validate()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update scheduler
            self.scheduler.step(val_accuracy)

            # Record history
            epoch_time = time.time() - epoch_start
            self.history['train_loss'].append(epoch_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['epoch_times'].append(epoch_time)
            self.history['learning_rates'].append(current_lr)

            # Print epoch summary
            print(f"Epoch [{epoch+1:3d}/{self.num_epochs}] "
                  f"| Loss: {epoch_loss:.4f} "
                  f"| Train Acc: {train_accuracy:.4f} "
                  f"| Val Acc: {val_accuracy:.4f} "
                  f"| LR: {current_lr:.6f} "
                  f"| Time: {epoch_time:.1f}s")

            # Save best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.epochs_without_improvement = 0
                save_path = self.output_dir / 'best_model.pth'
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✅ New best model! Val accuracy: {val_accuracy:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Early stopping check
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
                break

        training_time = time.time() - training_start

        # Load best model
        best_model_path = self.output_dir / 'best_model.pth'
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path))

        print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
        print(f"🏆 Best validation accuracy: {self.best_val_accuracy:.4f}")

        return self.history

    def save_final_model(self, filename='final_model.pth'):
        """Save the final model state.

        Args:
            filename: Name of the file to save.
        """
        save_path = self.output_dir / filename
        torch.save(self.model.state_dict(), save_path)
        print(f"💾 Final model saved to {save_path}")

# Trash Optimizer - Training Backend

This module handles the training and deployment of deep learning models for waste classification using transfer learning.

## Base Models

This module uses the following base models:

## 3rd Party Services

This module relies on the following external services:

- **Hugging Face Hub**: For model hosting and version control. [https://huggingface.co/](https://huggingface.co/)
- **PyTorch**: Deep learning framework
- **torchvision**: Pre-trained models and computer vision utilities

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the RealWaste dataset from Kaggle:
```bash
# Install kaggle CLI
pip install kaggle

# Download dataset (requires Kaggle API credentials)
kaggle datasets download -d joebeachcapital/realwaste
unzip realwaste.zip -d /path/to/your/data/directory
```

### 3. Configuration

1. Copy the `.env.template` file to a new file named `.env`:
   ```bash
   cp .env.template .env
   ```

2. Fill in the required environment variables in the `.env` file:
   ```bash
   DATASET_ROOT_DIR=/path/to/realwaste/RealWaste
   RESULTS_ROOT_DIR=/path/to/save/results
   HF_TOKEN=your_huggingface_token_here
   HF_REPO_ID=your-username/trash-optimizer-models
   ```

3. Get your Hugging Face token:
   - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create a new token with write permissions
   - Copy it to your `.env` file

## Training EfficientNet on RealWaste

### Run Training

Train the EfficientNet-B0 model on the RealWaste dataset:

```bash
python finetune_efficientnet_realwaste.py
```

### Training Features

The training script includes:

- **Transfer Learning**: Fine-tunes EfficientNet-B0 pretrained on ImageNet
- **Data Augmentation**: Random crops, flips, rotations, and color jitter
- **Early Stopping**: Stops training when validation accuracy plateaus
- **Learning Rate Scheduling**: Automatically reduces learning rate when needed
- **Regularization**: Weight decay and layer freezing to prevent overfitting
- **Comprehensive Logging**: Saves training history, plots, and model checkpoints

### Training Output

All training artifacts are saved to a timestamped directory in your `RESULTS_ROOT_DIR`:

```
results/
└── 202412031530_efficientnet_b0_realwaste/
    ├── best_model.pth           # Best model checkpoint (highest val accuracy)
    ├── final_model.pth          # Final model after all epochs
    ├── class_mapping.txt        # Class name to index mapping
    ├── training_history.txt     # Per-epoch metrics
    └── training_curves.png      # Visualization of training progress
```

## Model Deployment

### Deploy to Hugging Face Hub

After training, deploy your model to Hugging Face Hub for easy access:

```bash
python deploy_model.py results/202412031530_efficientnet_b0_realwaste
```

The script will:
- Create a repository on Hugging Face (if it doesn't exist)
- Upload all training artifacts
- Generate a README with usage instructions
- Provide a link to view your model online

## Training Parameters

Key hyperparameters (can be adjusted in the script):

- **Batch Size**: 32
- **Learning Rate**: 0.0001
- **Epochs**: 50 (with early stopping)
- **Train/Val Split**: 80/20
- **Early Stopping Patience**: 10 epochs
- **Image Size**: 224x224
- **Optimizer**: Adam with weight decay (1e-4)

## Model Architecture

- **Base Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Modifications**:
  - Last 3 feature blocks unfrozen for fine-tuning
  - Custom classifier head for 9 waste categories
  - Batch normalization and dropout included

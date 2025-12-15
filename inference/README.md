# Inference Backend

FastAPI service providing waste classification using fine-tuned EfficientNet-B0 models from Hugging Face Hub.

## Overview

The inference server dynamically loads trained models and provides RESTful API endpoints for waste image classification. Models are automatically downloaded from Hugging Face Hub on startup.

## Features

- **Dynamic Model Loading**: Fetches latest model version from HF Hub using `latest` pointer
- **Top-5 Predictions**: Returns top 5 waste categories with confidence scores
- **Auto-scaling Categories**: Adapts to model's category count via `class_mapping.txt`
- **Health Monitoring**: Built-in health check and model info endpoints

## Setup

### Requirements
- Python 3.12+
- Hugging Face account with read access to model repository

### Installation
```bash
cd inference
pip install -r requirements.txt
```

### Configuration
Copy `.env.template` to `.env` and configure:
```bash
HF_TOKEN=your_hf_token_here
HF_MODEL_REPO_ID=your_org/model_repo_name
```

## Running the Server

```bash
./run.sh
# Or manually: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server starts on `http://localhost:8000`

## API Endpoints

### `GET /`
Health check - returns API status

### `GET /health`
Service health verification

### `GET /categories`
Returns list of all waste categories the model can classify

### `GET /model-info`
Returns model metadata (architecture, number of categories, pretrained base)

### `POST /predict`
**Main classification endpoint**
- **Input**: Multipart form-data with image file
- **Output**: Top 5 predictions with confidence scores
- **Example**:
```json
[
  {"class": "plastic", "confidence": 0.92},
  {"class": "metal", "confidence": 0.05},
  {"class": "glass", "confidence": 0.02},
  {"class": "cardboard", "confidence": 0.01},
  {"class": "paper", "confidence": 0.00}
]
```

## Model Loading Process

1. Authenticates with Hugging Face Hub using `HF_TOKEN`
2. Downloads `latest` file to determine current model version
3. Downloads `model.pth` and `class_mapping.txt` from version folder
4. Initializes EfficientNet-B0 with correct output dimensions
5. Loads trained weights and sets to evaluation mode

## Image Preprocessing

- Resize to 256×256
- Center crop to 224×224
- Normalize using ImageNet statistics
- Convert to RGB tensor

## Notes

- Model runs on CPU by default (suitable for containerized deployment)
- For GPU inference, modify device in `main.py`
- Supports any EfficientNet-B0 model trained with project's training pipeline

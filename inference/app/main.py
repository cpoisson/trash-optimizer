from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import io
from typing import List, Dict
from dotenv import load_dotenv
import os
import huggingface_hub as hf

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID")

app = FastAPI(title="Inference Backend", version="1.0.0")

if HF_TOKEN and HF_MODEL_REPO_ID:
    # Load model from Hugging Face Hub
    print("Loading model from Hugging Face Hub...")
    hf.login(token=HF_TOKEN)

    # Download and read the latest file to get the model version folder name
    latest_file_path = hf.hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename="latest")
    with open(latest_file_path, "r") as f:
        latest_model_folder = f.read().strip()

    print(f"Latest model folder: {latest_model_folder}")

    # Construct remote filenames (these are paths within the repo, not local paths)
    hf_model_filename = f"{latest_model_folder}/model.pth"
    hf_model_categories_filename = f"{latest_model_folder}/class_mapping.txt"

    # Download model file
    model_path = hf.hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=hf_model_filename)
    print(f"Model downloaded to: {model_path}")

    # Download and load class mapping first to get num_classes
    class_mapping_path = hf.hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=hf_model_categories_filename)
    print(f"Categories downloaded to: {class_mapping_path}")

    # Load class labels
    with open(class_mapping_path, "r") as f:
        lines = f.readlines()
        class_to_idx = {}
        for line in lines:
            class_name, idx = line.strip().split(":")
            class_to_idx[class_name] = int(idx)

    categories = list(class_to_idx.keys())
    num_classes = len(categories)

    # Initialize model with correct number of classes
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

else:
    raise EnvironmentError("HF_TOKEN and HF_MODEL_REPO_ID must be set in environment variables.")


@app.get("/")
def read_root():
    return {"message": "Inference Backend API", "status": "running"}

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "healthy"}

@app.get("/categories")
def get_categories():
    """
    Get the list of categories the model can predict.
    """
    try:
        return JSONResponse(content={"categories": categories})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve categories: {str(e)}")

@app.get("/model-info")
def model_info():
    """
    Get information about the loaded model.
    """
    try:
        info = {
            "model_name": "EfficientNet_B0",
            "num_categories": len(categories),
            "pretrained_on": "ImageNet1K",
        }
        return JSONResponse(content=info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model info: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> List[Dict]:
    """
    Predict top 5 classifications for an uploaded image.
    """
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        input_tensor = preprocess(image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)

        # Get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top 5
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        # Format results
        results = [
            {
                "class": categories[top5_catid[i]],
                "confidence": float(top5_prob[i])
            }
            for i in range(5)
        ]

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

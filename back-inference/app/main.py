from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import io
from typing import List, Dict

app = FastAPI(title="Inference Backend", version="1.0.0")

# Load model and weights
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)
model.eval()

# Get preprocessing transforms
preprocess = weights.transforms()

# Get class labels
categories = weights.meta["categories"]


@app.get("/")
def read_root():
    return {"message": "Inference Backend API", "status": "running"}


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

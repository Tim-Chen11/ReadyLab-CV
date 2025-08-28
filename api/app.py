"""
FastAPI server for model inference
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import io
import base64
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch
import uvicorn

from scripts.inference import ModelInference


# Pydantic models for API
class PredictionResponse(BaseModel):
    status: str
    predictions: Dict
    timestamp: str
    model_info: Dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Optional[Dict] = None


# Global model instance
model_inference = None


# Create FastAPI app
app = FastAPI(
    title="Decade Classifier API",
    description="API for classifying images by decade",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model(checkpoint_path: str, multi_task: bool = False):
    """Load model globally"""
    global model_inference
    try:
        model_inference = ModelInference(
            checkpoint_path=checkpoint_path,
            multi_task=multi_task
        )
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    # Get checkpoint path from environment or use default
    checkpoint_path = os.getenv(
        "MODEL_CHECKPOINT", 
        "experiments/best_checkpoint.pth"
    )
    multi_task = os.getenv("MULTI_TASK", "false").lower() == "true"
    
    print(f"Starting API server...")
    print(f"Multi-task mode: {multi_task}")
    
    if Path(checkpoint_path).exists():
        success = load_model(checkpoint_path, multi_task)
        if success:
            print(f"✅ Model loaded successfully from {checkpoint_path}")
        else:
            print("⚠️ Failed to load model - API will run without model")
    else:
        print(f"⚠️ Checkpoint not found at {checkpoint_path}")
        print("Set MODEL_CHECKPOINT environment variable to point to your model")
        print("API will run without model (training data endpoint will still work)")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve simple UI"""
    return """
    <html>
        <head>
            <title>Decade Classifier</title>
        </head>
        <body style="font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px;">
            <h1>Decade Classifier API</h1>
            <p>Upload an image to classify which decade it's from!</p>
            
            <div style="margin: 20px 0;">
                <h3>API Endpoints:</h3>
                <ul>
                    <li><code>GET /health</code> - Check API health</li>
                    <li><code>POST /predict</code> - Predict decade from image</li>
                    <li><code>GET /docs</code> - Interactive API documentation</li>
                </ul>
            </div>
            
            <div style="margin: 20px 0;">
                <h3>Quick Test:</h3>
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" accept="image/*" required>
                    <button type="submit">Predict</button>
                </form>
            </div>
        </body>
    </html>
    """


@app.get("/training_data")
async def get_training_data():
    """Get training data for similar images display"""
    import json
    import math
    
    def clean_nan_values(obj):
        """Recursively clean NaN values from data structure"""
        if isinstance(obj, dict):
            return {k: clean_nan_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan_values(item) for item in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        return obj
    
    try:
        train_file = Path("data/splits/train.json")
        if train_file.exists():
            with open(train_file, 'r') as f:
                data = json.load(f)
            
            # Clean any NaN or Inf values
            cleaned_data = clean_nan_values(data)
            
            return {"status": "success", "data": cleaned_data}
        else:
            return {"status": "error", "message": "Training data not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""
    if model_inference is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_info={
            "model_name": model_inference.config.get("model_name", "unknown"),
            "multi_task": model_inference.multi_task,
            "device": str(model_inference.device)
        }
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict decade from uploaded image
    
    Args:
        file: Image file (JPEG, PNG)
        
    Returns:
        Prediction results with confidence scores
    """
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Save temporarily
        temp_path = Path(f"temp_{datetime.now().timestamp()}.jpg")
        image.save(temp_path)
        
        # Get prediction
        try:
            result = model_inference.predict_image(temp_path)
            
            # Clean up
            temp_path.unlink()
            
            return PredictionResponse(
                status="success",
                predictions=result,
                timestamp=datetime.now().isoformat(),
                model_info={
                    "model_name": model_inference.config.get("model_name", "unknown"),
                    "multi_task": model_inference.multi_task
                }
            )
        finally:
            # Ensure cleanup
            if temp_path.exists():
                temp_path.unlink()
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_base64")
async def predict_base64(image_data: str = Form(...)):
    """
    Predict decade from base64 encoded image
    
    Args:
        image_data: Base64 encoded image string
        
    Returns:
        Prediction results
    """
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Save temporarily
        temp_path = Path(f"temp_{datetime.now().timestamp()}.jpg")
        image.save(temp_path)
        
        # Get prediction
        try:
            result = model_inference.predict_image(temp_path)
            
            # Clean up
            temp_path.unlink()
            
            return {
                "status": "success",
                "predictions": result,
                "timestamp": datetime.now().isoformat()
            }
        finally:
            # Ensure cleanup
            if temp_path.exists():
                temp_path.unlink()
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/load_model")
async def load_model_endpoint(
    checkpoint_path: str = Form(...),
    multi_task: bool = Form(False)
):
    """
    Load a different model checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        multi_task: Whether model is multi-task
        
    Returns:
        Status message
    """
    if not Path(checkpoint_path).exists():
        raise HTTPException(status_code=404, detail="Checkpoint file not found")
    
    success = load_model(checkpoint_path, multi_task)
    
    if success:
        return {"status": "success", "message": f"Model loaded from {checkpoint_path}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")


if __name__ == "__main__":
    # Run with: python api/app.py
    # Or: uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
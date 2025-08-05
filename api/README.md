# Decade Classifier Inference API

This directory contains the inference system for the trained decade classifier model, including:
- Command-line inference script
- FastAPI server
- Web UI

## Setup

1. Install API dependencies:
```bash
pip install -r api/requirements.txt
```

2. Set your model checkpoint path:
```bash
# Windows
set MODEL_CHECKPOINT=experiments/your_experiment/checkpoints/best_checkpoint.pth

# Linux/Mac
export MODEL_CHECKPOINT=experiments/your_experiment/checkpoints/best_checkpoint.pth
```

3. For multi-task models, also set:
```bash
# Windows
set MULTI_TASK=true

# Linux/Mac
export MULTI_TASK=true
```

## Usage

### 1. Command Line Inference

```bash
# Single image
python scripts/inference.py path/to/checkpoint.pth path/to/image.jpg

# Multiple images
python scripts/inference.py path/to/checkpoint.pth image1.jpg image2.jpg image3.jpg

# Directory of images
python scripts/inference.py path/to/checkpoint.pth path/to/image/directory/

# Save results to JSON
python scripts/inference.py path/to/checkpoint.pth images/*.jpg --output results.json

# Multi-task model
python scripts/inference.py path/to/checkpoint.pth image.jpg --multi-task
```

### 2. API Server

Start the server:
```bash
# From project root
python api/app.py

# Or with uvicorn (auto-reload)
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /` - Simple upload interface
- `GET /health` - Check API and model status
- `POST /predict` - Upload image for prediction
- `POST /predict_base64` - Send base64 encoded image
- `POST /load_model` - Load a different model
- `GET /docs` - Interactive API documentation

Example API usage with curl:
```bash
# Check health
curl http://localhost:8000/health

# Predict from image file
curl -X POST -F "file=@image.jpg" http://localhost:8000/predict

# Load different model
curl -X POST -F "checkpoint_path=path/to/model.pth" -F "multi_task=true" http://localhost:8000/load_model
```

### 3. Web UI

1. Make sure the API server is running
2. Open `ui/index.html` in your web browser
3. Drag and drop or click to upload images
4. View predictions with confidence scores

Features:
- Drag & drop image upload
- Real-time API status indicator
- Confidence visualization
- Top-3 predictions display
- Support for both single-task and multi-task models

## Python SDK Usage

```python
from scripts.inference import ModelInference

# Initialize
model = ModelInference(
    checkpoint_path="path/to/checkpoint.pth",
    multi_task=False  # Set True for multi-task models
)

# Single image prediction
result = model.predict_image("path/to/image.jpg")
print(f"Prediction: {result['decade']['prediction']}")
print(f"Confidence: {result['decade']['confidence']:.2%}")

# Batch prediction
results = model.predict_batch(["image1.jpg", "image2.jpg", "image3.jpg"])
for res in results:
    if res['status'] == 'success':
        print(f"{res['image_path']}: {res['decade']['prediction']}")
```

## Model Output Format

### Single-task Model
```json
{
    "decade": {
        "prediction": "1980s",
        "confidence": 0.876,
        "top3": [
            {"class": "1980s", "confidence": 0.876},
            {"class": "1990s", "confidence": 0.089},
            {"class": "1970s", "confidence": 0.021}
        ],
        "all_probabilities": {
            "1950s": 0.005,
            "1960s": 0.009,
            "1970s": 0.021,
            "1980s": 0.876,
            "1990s": 0.089
        }
    }
}
```

### Multi-task Model
```json
{
    "decade": {
        "prediction": "1980s",
        "confidence": 0.876,
        "top3": [...]
    },
    "cluster": {
        "prediction": "Cluster_2",
        "confidence": 0.654,
        "top3": [...]
    }
}
```

## Deployment

### Using Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV MODEL_CHECKPOINT=/app/models/best_checkpoint.pth
ENV MULTI_TASK=false

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t decade-classifier .
docker run -p 8000:8000 -v /path/to/models:/app/models decade-classifier
```

### Production Considerations

1. **Model Loading**: The model is loaded once on startup for efficiency
2. **Caching**: Consider adding Redis for caching predictions
3. **Authentication**: Add API keys or JWT tokens for security
4. **Monitoring**: Add logging and metrics collection
5. **Scaling**: Use multiple workers with gunicorn:
   ```bash
   gunicorn api.app:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

## Troubleshooting

1. **Model not loading**: Check MODEL_CHECKPOINT path and file exists
2. **Out of memory**: Reduce batch size or use CPU inference
3. **Slow predictions**: Ensure GPU is available and being used
4. **API connection refused**: Check firewall and port availability

## Performance Tips

1. Use GPU for faster inference:
   ```python
   model = ModelInference(checkpoint_path, device="cuda:0")
   ```

2. Batch predictions for efficiency when processing multiple images

3. Pre-resize images to model's expected size before uploading

4. Enable response caching for frequently accessed images
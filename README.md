# ReadyLab-ML: Design Object Decade Classification

A deep learning project for classifying design objects by decade using computer vision techniques. This system analyzes images of design objects from various museums and collections to predict their decade of creation.

## Overview

ReadyLab-ML is a machine learning pipeline that:
- Collects design object data from multiple museum sources (MoMA, Cooper Hewitt, etc.)
- Processes and analyzes images of design objects
- Trains deep learning models to classify objects by decade (1960s-2000s)
- Provides inference capabilities through a REST API and web interface

## Features

- **Multi-source Data Collection**: Automated scrapers for various design databases
- **Robust Data Pipeline**: Comprehensive data processing and augmentation
- **Multiple Model Support**: ResNet50, EfficientNet, Vision Transformer architectures
- **Multi-task Learning**: Simultaneous decade classification and feature clustering
- **Production Ready**: FastAPI backend with web interface for real-time inference
- **Extensive Testing**: Comprehensive test suite for all components

## Project Structure

```
ReadyLab-ML/
├── api/                      # FastAPI backend
│   ├── app.py               # Main API application
│   └── requirements.txt     # API-specific dependencies
├── data/                    # Data directory
│   ├── raw/                # Raw unprocessed data
│   ├── metadata/           # Structured metadata (XLSX, JSON)
│   ├── cache/              # Cached images
│   ├── splits/             # Train/val/test splits
│   └── analysis/           # Dataset analysis results
├── src/                    # Source code
│   ├── data/              # Data loading and transforms
│   │   ├── data_utils.py  # Business logic for data handling
│   │   ├── transforms.py  # Image preprocessing
│   │   └── url_dataset.py # Dataset implementation
│   ├── datacollection/    # Data collection scripts
│   │   ├── fetch_MoMA.py
│   │   ├── fetch_cooper_hewitt.py
│   │   └── ...
│   ├── models/            # Model architectures
│   │   ├── base_model.py
│   │   ├── model_configs.py
│   │   └── model_factory.py
│   ├── training/          # Training utilities
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   └── utils/             # Helper utilities
├── scripts/               # Executable scripts
│   ├── train.py          # Main training script
│   ├── inference.py      # Inference utilities
│   ├── analyze_dataset.py # Dataset analysis
│   ├── create_splits.py  # Data splitting
│   └── setup_pipeline.py # Pipeline setup
├── experiments/          # Training experiment logs
├── notebooks/           # Jupyter notebooks
└── ui/                  # Web interface

```

## Dataset Statistics

- **Total Products**: 5,478 design objects
- **Total Images**: 9,863 images
- **Valid Years**: 5,201 products with valid year information
- **Decade Distribution**:
  - 1960s: 916 objects
  - 1970s: 1,395 objects
  - 1980s: 1,874 objects
  - 1990s: 2,213 objects
  - 2000s: 3,465 objects

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ReadyLab-ML.git
cd ReadyLab-ML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up data pipeline:
```bash
python scripts/setup_pipeline.py
```

## Usage

### Training

Train a model with default settings:
```bash
python -m scripts.train --model_name resnet50 --epochs 30 --batch_size 32
```

Advanced training with all options:
```bash
python -m scripts.train \
    --model_name resnet50 \
    --use_cached \
    --epochs 30 \
    --batch_size 32 \
    --num_workers 8 \
    --learning_rate 0.001 \
    --weight_decay 0.01 \
    --loss label_smoothing \
    --label_smoothing 0.2 \
    --early_stopping 10 \
    --use_amp \
    --multi_task \
    --decade_weight 1.0 \
    --cluster_weight 1.0
```

#### Training Parameters

- `--model_name`: Model architecture (resnet50, efficientnet_b0, vit_base_patch16_224)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Initial learning rate
- `--weight_decay`: L2 regularization weight
- `--loss`: Loss function (ce, focal, label_smoothing)
- `--label_smoothing`: Label smoothing factor (0-1)
- `--early_stopping`: Patience for early stopping
- `--use_amp`: Enable automatic mixed precision training
- `--multi_task`: Enable multi-task learning
- `--use_cached`: Use cached images instead of downloading

More can be find in train.py

### Inference

#### Start the API Server
```bash
python start_api.py
```

The API will be available at `http://localhost:8000`

#### Start the Web Interface
```bash
python start_frontend.py
```

Access the web interface at `http://localhost:8001`

### API Endpoints

- `GET /health` - Check API health status
- `POST /predict` - Upload image for decade prediction
- `GET /docs` - Interactive API documentation

Example API usage:
```python
import requests

# Upload and predict
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    print(response.json())
```

## Data Collection

The project includes scrapers for multiple design databases:

- **MoMA**: Museum of Modern Art design collection
- **Cooper Hewitt**: Smithsonian Design Museum
- **DataMath Calculator Museum**: Historical calculator designs
- **Mobile Phone Museum**: Mobile phone design evolution
- **1stdibs**: Contemporary and vintage design marketplace

Run data collection:
```bash
python src/datacollection/fetch_all_execution.py
```

## Testing

Run the complete test suite:
```bash
pytest scripts/test/
```

Individual test modules:
```bash
pytest scripts/test/test_data_loading.py
pytest scripts/test/test_transforms_comprehensive.py
pytest scripts/test/test_url_dataset_comprehensive.py
```

## Model Performance

Current best results (ResNet50):
- **Validation Accuracy**: ~75%
- **Test Accuracy**: ~73%
- **Per-decade F1 Scores**:
  - 1960s: 0.68
  - 1970s: 0.71
  - 1980s: 0.74
  - 1990s: 0.76
  - 2000s: 0.78

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MoMA for providing access to their design collection data
- Cooper Hewitt Smithsonian Design Museum for their open API
- The PyTorch team for the excellent deep learning framework
- All contributors to the open-source libraries used in this project

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

## Citation

If you use this project in your research, please cite:
```bibtex
@software{readylab_ml,
  title={ReadyLab-ML: Design Object Decade Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ReadyLab-ML}
}
```
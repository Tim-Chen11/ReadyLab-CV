# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReadyLab-ML is a deep learning project for Design Object Decade Classification (1960s-2000s) using PyTorch and modern computer vision models. It analyzes design objects from museum collections to predict their decade of creation.

## Common Development Commands

### Training Models
```bash
# Basic training with default settings
python -m scripts.train --model_name resnet50 --epochs 30 --batch_size 32

# Advanced multi-task training with all optimizations
python -m scripts.train --model_name convnext_v2_tiny_384 --use_cached --epochs 30 --batch_size 16 --learning_rate 0.0001 --weight_decay 0.01 --loss label_smoothing --label_smoothing 0.2 --early_stopping 10 --use_amp --multi_task --decade_weight 1.0 --cluster_weight 0.5

# Resume training from checkpoint
python -m scripts.train --model_name resnet50 --resume checkpoints/model_best.pth
```

### Testing and Validation
```bash
# Run all tests
pytest scripts/test/

# Run specific test modules
pytest scripts/test/test_data_loading.py
pytest scripts/test/test_transforms_comprehensive.py
pytest scripts/test/test_url_dataset_comprehensive.py
pytest scripts/test/test_model_factory.py

# Analyze dataset statistics
python scripts/analyze_dataset.py

# Create train/val/test splits
python scripts/create_splits.py
```

### Production Deployment
```bash
# Start API server (automatically finds best model)
python start_api.py --port 8000

# Start web interface
python start_frontend.py

# API documentation available at http://localhost:8000/docs
```

## High-Level Architecture

### Core Components

1. **Model Factory Pattern** (`src/models/model_factory.py`)
   - Centralized model creation supporting ResNet, EfficientNet, ConvNeXt, ViT, MobileNet families
   - Configuration-driven instantiation via `model_configs.py`
   - Support for both single-task (decade only) and multi-task learning (decade + clustering + device type)

2. **Data Pipeline** (`src/data/`)
   - `URLDataset`: Downloads images on-demand with intelligent caching
   - `CachedDataset`: Uses pre-downloaded images for faster training
   - Smart retry mechanisms and fallback strategies for robust data loading
   - Model-specific augmentation strategies (light/medium/heavy)

3. **Training Infrastructure** (`src/training/`)
   - Mixed precision training support (AMP)
   - Early stopping with configurable patience
   - Experiment logging with automatic checkpointing
   - Multi-GPU ready architecture

4. **Multi-Task Learning Architecture**
   - Primary task: Decade classification (5 classes: 1960s-2000s)
   - Secondary tasks: Feature clustering (10 classes), Device classification (2 classes)
   - Shared backbone with task-specific heads
   - Configurable loss weighting and task balancing

### Key Design Patterns

- **Factory Pattern**: Model creation through `ModelFactory.create_model()`
- **Configuration-Driven**: All hyperparameters in `model_configs.py`
- **Strategy Pattern**: Different augmentation and fine-tuning strategies per model
- **Repository Pattern**: Data access through standardized dataset interfaces

### Directory Structure
```
src/
├── data/           # Dataset classes and data loading utilities
├── datacollection/ # Museum API scrapers (MoMA, Cooper Hewitt, etc.)
├── models/         # Model architectures and factory
├── training/       # Training loop, losses, and utilities
└── utils/          # Logging, visualization, and helper functions

scripts/
├── train.py        # Main training script
├── test/           # Comprehensive test suite
└── *.py            # Analysis and utility scripts

api/
└── main.py         # FastAPI production server
```

## Important Considerations

### Model-Specific Configurations
Each model family has optimized settings in `model_configs.py`:
- Batch sizes adjusted for GPU memory requirements
- Model-specific learning rates and weight decay
- Tailored augmentation strategies
- Fine-tuning layer specifications

### Dataset Characteristics
- 5,478 design objects with 9,863 images total
- 5,201 products with valid decade information
- Imbalanced distribution: fewer 1960s (916) vs 2000s (3,465) samples
- Multi-source data from MoMA, Cooper Hewitt, Mobile Phone Museum, etc.

### Performance Optimization
- Use `--use_cached` flag after first run to speed up data loading
- Enable `--use_amp` for mixed precision training (2x faster, less memory)
- Model-specific batch sizes prevent OOM errors
- Automatic best checkpoint selection for inference

### Testing Strategy
- Always run tests after modifying core components
- Use `pytest scripts/test/test_data_loading.py` for data pipeline changes
- Use `pytest scripts/test/test_model_factory.py` for model architecture changes
- Comprehensive transform testing ensures augmentation pipeline integrity
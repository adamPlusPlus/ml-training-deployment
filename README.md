# ML Training & Deployment Pipeline

A complete machine learning training and deployment system demonstrating production-ready MLOps practices for model training, versioning, and API deployment.

## Overview

This project provides a complete pipeline for training machine learning models and deploying them as production-ready inference APIs. The system demonstrates best practices for model training, experiment tracking, versioning, and containerized deployment.

## Features

- **Model Training Pipeline**
  - PyTorch/TensorFlow model training
  - Data preprocessing and augmentation
  - Experiment tracking with MLflow
  - Model versioning and checkpointing
  - Hyperparameter tuning support

- **Model Serving API**
  - FastAPI-based inference endpoint
  - Batch prediction support
  - Model loading and caching
  - Health checks and monitoring endpoints
  - Request/response validation

- **Containerization & Deployment**
  - Docker containerization with multi-stage builds
  - Cloud deployment configurations
  - Environment management
  - Resource optimization

## Project Structure

```
.
├── training/
│   ├── models/          # Model architectures
│   ├── data/            # Training data and preprocessing
│   ├── scripts/         # Training scripts
│   └── configs/         # Training configurations
├── api/
│   ├── app.py          # FastAPI application
│   ├── models/         # API models and schemas
│   └── inference/      # Inference logic
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   ├── test_training.py
│   └── test_api.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- MLflow (for experiment tracking)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ml-training-deployment

# Install dependencies
pip install -r requirements.txt

# Set up MLflow tracking
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Training a Model

```bash
# Train model with default configuration
python training/scripts/train.py --config training/configs/default.yaml

# Train with custom parameters
python training/scripts/train.py \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001
```

### Running the API

```bash
# Start the FastAPI server
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Or using Docker
docker-compose up
```

### API Usage

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/api/predict",
    json={
        "input": "your_input_data_here"
    }
)

# Batch prediction
response = requests.post(
    "http://localhost:8000/api/predict/batch",
    json={
        "inputs": ["input1", "input2", "input3"]
    }
)
```

## Model Architecture

The system supports flexible model architectures:

- **Input Processing**: Data preprocessing and feature engineering
- **Model**: Configurable neural network architectures
- **Output**: Validated prediction results

## Training Data

Training data structure:
- Raw data files
- Preprocessed datasets
- Validation metadata
- Quality scores

## Model Versioning

Models are versioned using MLflow:
- Automatic versioning on training completion
- Metadata tracking (hyperparameters, metrics, artifacts)
- Model registry for production models
- Easy rollback to previous versions

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t ml-model:latest .

# Run container
docker run -p 8000:8000 ml-model:latest
```

### Cloud Deployment

The project includes configurations for:
- AWS (EC2, ECS, Lambda)
- Azure (Container Instances, App Service)
- GCP (Cloud Run, Compute Engine)

See deployment documentation in `docs/deployment/` for platform-specific instructions.

## Monitoring

The API includes built-in monitoring endpoints:
- `/health` - Health check
- `/metrics` - Performance metrics
- `/api/model/info` - Model metadata

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_training.py
pytest tests/test_api.py
```

## Documentation

- [Training Guide](docs/training.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Model Architecture](docs/architecture.md)

## License

MIT License

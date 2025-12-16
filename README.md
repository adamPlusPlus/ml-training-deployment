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
  - Cloud deployment configurations (AWS, Azure, GCP)
  - Environment management
  - Resource optimization

- **Security**
  - Secrets management (AWS Secrets Manager)
  - Data encryption (at rest and in transit)
  - Network security configurations

- **Monitoring & Cost Tracking**
  - AWS cost monitoring
  - Custom metrics
  - Cost optimization strategies

- **LLM Integration**
  - Hugging Face model deployment
  - LLM inference endpoints
  - Token usage and cost tracking

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- MLflow (for experiment tracking)
- AWS CLI (for cloud deployment)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ml-training-deployment

# Install dependencies
pip install -r requirements.txt
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

## Project Structure

```
ml-training-deployment/
├── training/          # Model training scripts and configurations
├── api/              # FastAPI inference API
├── infrastructure/   # Terraform and Docker configurations
├── security/         # Security implementations
├── monitoring/       # Cost tracking and metrics
├── tests/            # Unit and integration tests
└── docs/             # Documentation (including learning guide)
```

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t ml-model:latest .

# Run container
docker run -p 8000:8000 ml-model:latest
```

### Cloud Deployment

The project includes Terraform configurations for:
- AWS (ECS, SageMaker)
- Azure (Container Instances)
- GCP (Cloud Run)

See `docs/learn.md` for detailed deployment instructions.

## Documentation

- **[Learning Guide](docs/learn.md)** - Complete step-by-step learning path
- [AWS Deployment Guide](docs/aws_deployment.md)
- [Terraform Guide](docs/terraform_guide.md)
- [Security Guide](docs/security_guide.md)
- [LLM Integration Guide](docs/llm_integration.md)

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_training.py
pytest tests/test_api.py
```

## License

MIT License

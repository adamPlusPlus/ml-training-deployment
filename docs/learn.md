# ML Training & Deployment Pipeline - Complete Learning Guide

**Goal:** Build a production-ready ML training and deployment system that demonstrates ALL Tier 1, Tier 2, and Tier 3 skills.

**What You'll Learn:**
- Docker containerization
- FastAPI ML serving
- MLflow model versioning
- AWS deployment (SageMaker/ECS)
- Terraform Infrastructure as Code
- Security best practices
- LLM deployment
- Cost optimization

**What You'll Demonstrate:**
- Complete ML training pipeline
- Production API deployment
- Cloud infrastructure
- Security implementation
- Multi-cloud capability
- LLM integration

---

## 📋 Learning Checklist

### Tier 1: Must Have (90% Job Guarantee)

- [ ] **Docker** - Multi-stage builds, optimization
- [ ] **FastAPI** - ML serving API with validation
- [ ] **MLflow** - Model versioning and tracking
- [ ] **AWS Deployment** - SageMaker or ECS
- [ ] **Terraform** - Infrastructure as Code
- [ ] **Testing** - Unit and integration tests

### Tier 2: High Value (95% Job Guarantee)

- [ ] **Security** - Secrets management, encryption
- [ ] **Cost Tracking** - AWS cost monitoring
- [ ] **Multi-Cloud** - AWS + one other (Azure/GCP)

### Tier 3: Differentiators (99% Top Offers)

- [ ] **LLM Integration** - Hugging Face deployment
- [ ] **Cost Optimization** - Detailed analysis and strategies

---

## 🎯 Project Structure

```
ml-training-deployment/
├── training/
│   ├── models/              # Model architectures
│   │   ├── pytorch_model.py
│   │   ├── tensorflow_model.py
│   │   └── llm_model.py     # LLM integration (Tier 3)
│   ├── data/                # Training data
│   ├── scripts/
│   │   ├── train.py         # Training script
│   │   ├── train_aws.py     # AWS SageMaker training (Tier 1)
│   │   └── validate_data.py # Data validation
│   └── configs/             # Training configurations
├── api/
│   ├── app.py              # FastAPI application
│   ├── models/             # Pydantic schemas
│   ├── inference/          # Inference logic
│   │   ├── pytorch_inference.py
│   │   ├── tensorflow_inference.py
│   │   └── llm_inference.py # LLM inference (Tier 3)
│   └── security/           # Security middleware (Tier 2)
│       ├── auth.py
│       └── encryption.py
├── infrastructure/
│   ├── terraform/          # Infrastructure as Code (Tier 1)
│   │   ├── aws/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   ├── outputs.tf
│   │   │   └── ecs.tf      # ECS deployment
│   │   ├── azure/          # Multi-cloud (Tier 2)
│   │   └── gcp/            # Multi-cloud (Tier 2)
│   ├── docker/
│   │   ├── Dockerfile      # Multi-stage build
│   │   ├── Dockerfile.llm  # LLM-specific (Tier 3)
│   │   └── docker-compose.yml
│   └── kubernetes/         # K8s manifests (for Project 2)
│       └── deployment.yaml
├── security/               # Security implementation (Tier 2)
│   ├── secrets/           # Secrets management
│   │   ├── vault_config.hcl
│   │   └── aws_secrets.py
│   └── encryption/        # Data encryption
│       ├── at_rest.py
│       └── in_transit.py
├── monitoring/
│   ├── cost_tracking.py   # Cost monitoring (Tier 2/3)
│   └── metrics.py         # Custom metrics
├── tests/
│   ├── test_training.py
│   ├── test_api.py
│   ├── test_security.py   # Security tests (Tier 2)
│   └── test_llm.py        # LLM tests (Tier 3)
├── docs/
│   ├── LEARNING_GUIDE.md  # This file
│   ├── aws_deployment.md  # AWS deployment guide
│   ├── terraform_guide.md # Terraform learning
│   ├── security_guide.md  # Security implementation
│   └── llm_integration.md # LLM deployment guide
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 📚 Step-by-Step Learning Path

### Phase 1: Foundation (Week 1-2)

#### Step 1.1: Docker Basics (2-3 days)

**Learning Resources:**
- Docker official tutorial: https://docs.docker.com/get-started/
- Time: 2-3 days

**What to Build:**
```dockerfile
# docker/Dockerfile
# Multi-stage build for optimization
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Demonstration:**
- [ ] Build Docker image
- [ ] Run container locally
- [ ] Optimize image size (multi-stage build)
- [ ] Document build process

**Checkpoint:** Can you build and run your API in Docker?

---

#### Step 1.2: FastAPI ML Serving (2-3 days)

**Learning Resources:**
- FastAPI tutorial: https://fastapi.tiangolo.com/tutorial/
- Time: 1-2 days

**What to Build:**
```python
# api/app.py
from fastapi import FastAPI, HTTPException
from api.models import PredictionRequest, PredictionResponse
from api.inference import ModelInference
import mlflow

app = FastAPI(title="ML Inference API")

# Load model using MLflow
model = mlflow.pyfunc.load_model("models:/production-model/1")

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        prediction = model.predict(request.input)
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    # Return custom metrics
    return {"requests_per_second": 100, "avg_latency_ms": 50}
```

**Demonstration:**
- [ ] Single prediction endpoint
- [ ] Batch prediction endpoint
- [ ] Health check endpoint
- [ ] Metrics endpoint
- [ ] Request/response validation
- [ ] Error handling

**Checkpoint:** Can you serve predictions via API?

---

#### Step 1.3: MLflow Model Versioning (1-2 days)

**Learning Resources:**
- MLflow docs: https://mlflow.org/docs/latest/index.html
- Time: 1-2 days

**What to Build:**
```python
# training/scripts/train.py
import mlflow
import mlflow.pytorch
from training.models.pytorch_model import MyModel

def train_model():
    mlflow.set_experiment("ml-training-deployment")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("epochs", 50)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("learning_rate", 0.001)
        
        # Train model
        model = MyModel()
        # ... training code ...
        
        # Log metrics
        mlflow.log_metric("accuracy", 0.92)
        mlflow.log_metric("loss", 0.15)
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        # Register model
        mlflow.register_model(
            "runs:/{}/model".format(mlflow.active_run().info.run_id),
            "production-model"
        )
```

**Demonstration:**
- [ ] Track training experiments
- [ ] Log hyperparameters and metrics
- [ ] Register models in MLflow registry
- [ ] Version models
- [ ] Load models from registry

**Checkpoint:** Can you track experiments and version models?

---

### Phase 2: Cloud Deployment (Week 3-4)

#### Step 2.1: AWS Fundamentals (1 week)

**Learning Resources:**
- AWS Free Tier: https://aws.amazon.com/free/
- AWS ECS tutorial: https://docs.aws.amazon.com/ecs/
- Time: 1 week

**Prerequisites:**
- [ ] Create AWS account (free tier)
- [ ] Set up AWS CLI
- [ ] Configure credentials

**What to Learn:**
- ECS (Elastic Container Service) basics
- ECR (Elastic Container Registry) for images
- IAM roles and policies
- VPC and networking basics

**Checkpoint:** Can you navigate AWS console and understand basic services?

---

#### Step 2.2: Terraform Basics (3-5 days)

**Learning Resources:**
- HashiCorp Learn: https://learn.hashicorp.com/terraform
- Time: 3-5 days

**What to Build:**
```hcl
# infrastructure/terraform/aws/main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ECR Repository for Docker images
resource "aws_ecr_repository" "ml_model" {
  name                 = "ml-training-deployment"
  image_tag_mutability = "MUTABLE"
}

# ECS Cluster
resource "aws_ecs_cluster" "ml_cluster" {
  name = "ml-training-cluster"
}

# ECS Task Definition
resource "aws_ecs_task_definition" "ml_api" {
  family                   = "ml-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  
  container_definitions = jsonencode([{
    name  = "ml-api"
    image = "${aws_ecr_repository.ml_model.repository_url}:latest"
    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]
  }])
}

# ECS Service
resource "aws_ecs_service" "ml_service" {
  name            = "ml-api-service"
  cluster         = aws_ecs_cluster.ml_cluster.id
  task_definition = aws_ecs_task_definition.ml_api.arn
  desired_count   = 2
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets = var.subnet_ids
    security_groups = [aws_security_group.ml_api.id]
  }
}
```

**Demonstration:**
- [ ] Write Terraform configuration
- [ ] Initialize Terraform
- [ ] Plan infrastructure changes
- [ ] Apply infrastructure
- [ ] Destroy infrastructure (cleanup)
- [ ] Document infrastructure

**Checkpoint:** Can you deploy infrastructure with Terraform?

---

#### Step 2.3: AWS Deployment (1 week)

**What to Build:**

1. **Build and Push Docker Image:**
```bash
# Build image
docker build -t ml-training-deployment:latest .

# Tag for ECR
docker tag ml-training-deployment:latest \
  <account-id>.dkr.ecr.<region>.amazonaws.com/ml-training-deployment:latest

# Push to ECR
aws ecr get-login-password --region <region> | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/ml-training-deployment:latest
```

2. **Deploy with Terraform:**
```bash
cd infrastructure/terraform/aws
terraform init
terraform plan
terraform apply
```

3. **Verify Deployment:**
```bash
# Get service URL
aws ecs describe-services \
  --cluster ml-training-cluster \
  --services ml-api-service

# Test API
curl https://your-service-url/api/health
```

**Demonstration:**
- [ ] Build Docker image
- [ ] Push to ECR
- [ ] Deploy with Terraform
- [ ] Access deployed API
- [ ] Test endpoints
- [ ] Monitor in AWS console

**Checkpoint:** Is your API running on AWS?

---

### Phase 3: Security (Week 5)

#### Step 3.1: Secrets Management (2-3 days)

**Learning Resources:**
- AWS Secrets Manager: https://docs.aws.amazon.com/secretsmanager/
- Time: 2-3 days

**What to Build:**
```python
# security/secrets/aws_secrets.py
import boto3
import json
from botocore.exceptions import ClientError

def get_secret(secret_name, region_name="us-east-1"):
    """Retrieve secret from AWS Secrets Manager"""
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        secret = get_secret_value_response['SecretString']
        return json.loads(secret)
    except ClientError as e:
        raise e

# Usage in API
# api/app.py
from security.secrets.aws_secrets import get_secret

secrets = get_secret("ml-api-secrets")
DATABASE_URL = secrets["database_url"]
API_KEY = secrets["api_key"]
```

**Terraform Configuration:**
```hcl
# infrastructure/terraform/aws/secrets.tf
resource "aws_secretsmanager_secret" "ml_api_secrets" {
  name = "ml-api-secrets"
}

resource "aws_secretsmanager_secret_version" "ml_api_secrets" {
  secret_id = aws_secretsmanager_secret.ml_api_secrets.id
  secret_string = jsonencode({
    database_url = var.database_url
    api_key      = var.api_key
  })
}

# IAM policy for ECS task to access secrets
resource "aws_iam_role_policy" "ecs_secrets_access" {
  name = "ecs-secrets-access"
  role = aws_iam_role.ecs_task_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "secretsmanager:GetSecretValue"
      ]
      Resource = aws_secretsmanager_secret.ml_api_secrets.arn
    }]
  })
}
```

**Demonstration:**
- [ ] Store secrets in AWS Secrets Manager
- [ ] Retrieve secrets in application
- [ ] Configure IAM permissions
- [ ] Never hardcode secrets
- [ ] Document secret management

**Checkpoint:** Are all secrets managed securely?

---

#### Step 3.2: Data Encryption (2-3 days)

**What to Build:**
```python
# security/encryption/at_rest.py
from cryptography.fernet import Fernet
import os

class DataEncryption:
    def __init__(self):
        # In production, get key from secrets manager
        key = os.getenv("ENCRYPTION_KEY")
        self.cipher = Fernet(key)
    
    def encrypt(self, data: bytes) -> bytes:
        return self.cipher.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        return self.cipher.decrypt(encrypted_data)

# security/encryption/in_transit.py
# Use HTTPS/TLS (configured in infrastructure)
# api/app.py - Add HTTPS middleware
```

**Terraform Configuration:**
```hcl
# infrastructure/terraform/aws/security.tf
# Security group for ECS service
resource "aws_security_group" "ml_api" {
  name        = "ml-api-sg"
  description = "Security group for ML API"
  vpc_id      = var.vpc_id
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Enable encryption at rest for ECR
resource "aws_ecr_repository" "ml_model" {
  name                 = "ml-training-deployment"
  image_tag_mutability = "MUTABLE"
  
  encryption_configuration {
    encryption_type = "AES256"
  }
}
```

**Demonstration:**
- [ ] Encrypt data at rest
- [ ] Use HTTPS for data in transit
- [ ] Configure security groups
- [ ] Enable ECR encryption
- [ ] Document encryption strategy

**Checkpoint:** Is data encrypted at rest and in transit?

---

### Phase 4: Advanced Features (Week 6-7)

#### Step 4.1: LLM Integration (1 week) - Tier 3

**Learning Resources:**
- Hugging Face: https://huggingface.co/docs/transformers/
- Time: 1 week

**What to Build:**
```python
# training/models/llm_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMModel:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def generate(self, prompt: str, max_length: int = 100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# api/inference/llm_inference.py
from training.models.llm_model import LLMModel

llm_model = LLMModel()

@app.post("/api/llm/generate")
async def generate_text(request: LLMRequest):
    result = llm_model.generate(request.prompt, request.max_length)
    return {"generated_text": result}
```

**Dockerfile for LLM:**
```dockerfile
# infrastructure/docker/Dockerfile.llm
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app
COPY requirements-llm.txt .
RUN pip install --no-cache-dir -r requirements-llm.txt

COPY . .

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Demonstration:**
- [ ] Deploy LLM model (Hugging Face)
- [ ] Create LLM inference endpoint
- [ ] Monitor token usage
- [ ] Track LLM costs
- [ ] Version prompts
- [ ] Document LLM integration

**Checkpoint:** Can you serve LLM predictions?

---

#### Step 4.2: Cost Tracking (2-3 days) - Tier 2/3

**What to Build:**
```python
# monitoring/cost_tracking.py
import boto3
from datetime import datetime, timedelta

class CostTracker:
    def __init__(self):
        self.client = boto3.client('ce')  # Cost Explorer
    
    def get_daily_costs(self, days=7):
        end = datetime.now()
        start = end - timedelta(days=days)
        
        response = self.client.get_cost_and_usage(
            TimePeriod={
                'Start': start.strftime('%Y-%m-%d'),
                'End': end.strftime('%Y-%m-%d')
            },
            Granularity='DAILY',
            Metrics=['BlendedCost']
        )
        return response
    
    def get_service_costs(self):
        # Get costs by AWS service
        response = self.client.get_cost_and_usage(
            TimePeriod={
                'Start': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'End': datetime.now().strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['BlendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
        )
        return response

# API endpoint
@app.get("/api/costs")
async def get_costs():
    tracker = CostTracker()
    return tracker.get_daily_costs()
```

**Demonstration:**
- [ ] Track daily costs
- [ ] Track costs by service
- [ ] Create cost dashboard
- [ ] Set cost alerts
- [ ] Document cost optimization strategies

**Checkpoint:** Can you monitor and report costs?

---

#### Step 4.3: Multi-Cloud Support (3-5 days) - Tier 2

**What to Build:**

**Azure Deployment:**
```hcl
# infrastructure/terraform/azure/main.tf
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_container_group" "ml_api" {
  name                = "ml-api"
  location            = var.location
  resource_group_name = var.resource_group_name
  os_type             = "Linux"
  
  container {
    name   = "ml-api"
    image  = var.container_image
    cpu    = "0.5"
    memory = "1.5"
    
    ports {
      port     = 8000
      protocol = "TCP"
    }
  }
}
```

**Demonstration:**
- [ ] Deploy to Azure Container Instances
- [ ] Use same Docker image
- [ ] Document multi-cloud setup
- [ ] Compare costs
- [ ] Show flexibility

**Checkpoint:** Can you deploy to multiple clouds?

---

## ✅ Final Demonstration Checklist

### Must Demonstrate (Tier 1):

- [ ] **Docker:** Multi-stage build, optimized image
- [ ] **FastAPI:** Working ML inference API
- [ ] **MLflow:** Model versioning and tracking
- [ ] **AWS:** Deployed and accessible
- [ ] **Terraform:** Infrastructure as Code
- [ ] **Testing:** Unit and integration tests

### Should Demonstrate (Tier 2):

- [ ] **Security:** Secrets management, encryption
- [ ] **Cost Tracking:** Monitoring and reporting
- [ ] **Multi-Cloud:** AWS + one other

### Nice to Demonstrate (Tier 3):

- [ ] **LLM:** Hugging Face deployment
- [ ] **Cost Optimization:** Detailed analysis

---

## 📖 Documentation Requirements

Create these documents:

1. **docs/aws_deployment.md** - Step-by-step AWS deployment guide
2. **docs/terraform_guide.md** - Terraform learning and usage
3. **docs/security_guide.md** - Security implementation details
4. **docs/llm_integration.md** - LLM deployment guide
5. **docs/cost_optimization.md** - Cost analysis and strategies

---

## 🎓 Learning Resources Summary

| Topic | Resource | Time |
|-------|----------|------|
| Docker | https://docs.docker.com/get-started/ | 2-3 days |
| FastAPI | https://fastapi.tiangolo.com/tutorial/ | 1-2 days |
| MLflow | https://mlflow.org/docs/latest/ | 1-2 days |
| AWS | https://aws.amazon.com/free/ | 1 week |
| Terraform | https://learn.hashicorp.com/terraform | 3-5 days |
| Security | AWS Secrets Manager docs | 2-3 days |
| LLM | https://huggingface.co/docs/transformers/ | 1 week |

---

## 🚀 Getting Started

1. **Start with Phase 1** - Build foundation
2. **Use Cursor AI** - Generate code, ask questions
3. **Learn as you build** - Don't wait to learn everything
4. **Check off items** - Track your progress
5. **Document everything** - Show what you learned

**Remember:** You're learning by building. Start now, learn as you go!

---

## 📝 Next Steps

After completing this project:
1. Move to **mlops-infrastructure** (Project 2)
2. Add Kubernetes, CI/CD, monitoring
3. Integrate with this project
4. Build complete MLOps platform

**You've got this!** 🎯

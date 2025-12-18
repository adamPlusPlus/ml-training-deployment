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

#### Step 2.4: AWS Lambda Serverless Deployment (3-5 days) - Tier 1.5

**Learning Resources:**
- AWS Lambda: https://docs.aws.amazon.com/lambda/
- AWS Lambda Python: https://docs.aws.amazon.com/lambda/latest/dg/lambda-python.html
- Time: 3-5 days

**Why This Matters:**
- Many job listings require Lambda experience (Cognizant, Modernizing Medicine)
- Serverless is cost-effective for low-traffic or bursty workloads
- Demonstrates understanding of different deployment patterns

**What to Build:**

**1. Lambda Function for Model Inference:**
```python
# lambda/inference_handler.py
import json
import boto3
import mlflow
import os
from typing import Dict, Any

# Initialize MLflow client
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

# Load model once (Lambda container reuse)
model = None

def load_model():
    """Load model from MLflow registry (cached for warm starts)"""
    global model
    if model is None:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{os.environ['MODEL_NAME']}/{os.environ['MODEL_STAGE']}"
        )
    return model

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for ML inference
    
    Expected event format:
    {
        "body": {
            "input": [1.0, 2.0, 3.0],
            "model_version": "optional"
        }
    }
    """
    try:
        # Parse request
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        input_data = body.get('input')
        if not input_data:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing input data'})
            }
        
        # Load model (cached on warm starts)
        model = load_model()
        
        # Run inference
        prediction = model.predict([input_data])
        
        # Log to CloudWatch
        print(f"Inference completed: input_shape={len(input_data)}, prediction={prediction[0]}")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'prediction': float(prediction[0]),
                'model_version': os.environ.get('MODEL_VERSION', 'unknown'),
                'timestamp': context.aws_request_id
            })
        }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

**2. Lambda Deployment Package:**
```python
# lambda/requirements.txt
mlflow>=2.0.0
boto3>=1.28.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

**3. Terraform Lambda Configuration:**
```hcl
# infrastructure/terraform/aws/lambda.tf
resource "aws_lambda_function" "ml_inference" {
  filename         = "lambda_deployment.zip"
  function_name    = "ml-inference-lambda"
  role            = aws_iam_role.lambda_role.arn
  handler         = "inference_handler.lambda_handler"
  runtime         = "python3.11"
  timeout         = 30
  memory_size     = 512
  
  environment {
    variables = {
      MLFLOW_TRACKING_URI = var.mlflow_tracking_uri
      MODEL_NAME          = "production-model"
      MODEL_STAGE         = "Production"
      MODEL_VERSION       = "1"
    }
  }
  
  # Enable X-Ray tracing
  tracing_config {
    mode = "Active"
  }
  
  # Dead letter queue for failed invocations
  dead_letter_config {
    target_arn = aws_sqs_queue.dlq.arn
  }
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda_role" {
  name = "ml-lambda-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

# IAM Policy for Lambda
resource "aws_iam_role_policy" "lambda_policy" {
  name = "ml-lambda-policy"
  role = aws_iam_role.lambda_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject"
        ]
        Resource = "${aws_s3_bucket.ml_models.arn}/*"
      }
    ]
  })
}

# API Gateway integration
resource "aws_api_gateway_rest_api" "ml_api" {
  name        = "ml-inference-api"
  description = "ML Inference API via Lambda"
}

resource "aws_api_gateway_resource" "predict" {
  rest_api_id = aws_api_gateway_rest_api.ml_api.id
  parent_id   = aws_api_gateway_rest_api.ml_api.root_resource_id
  path_part   = "predict"
}

resource "aws_api_gateway_method" "predict" {
  rest_api_id   = aws_api_gateway_rest_api.ml_api.id
  resource_id   = aws_api_gateway_resource.predict.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda" {
  rest_api_id = aws_api_gateway_rest_api.ml_api.id
  resource_id = aws_api_gateway_resource.predict.id
  http_method = aws_api_gateway_method.predict.http_method
  
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.ml_inference.invoke_arn
}

# Lambda permission for API Gateway
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ml_inference.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.ml_api.execution_arn}/*/*"
}

# Dead Letter Queue
resource "aws_sqs_queue" "dlq" {
  name = "ml-lambda-dlq"
  
  message_retention_seconds = 1209600  # 14 days
}
```

**4. Lambda Deployment Script:**
```bash
#!/bin/bash
# scripts/deploy_lambda.sh

# Create deployment package
cd lambda
zip -r ../lambda_deployment.zip .
cd ..

# Upload to S3 (or deploy directly)
aws lambda update-function-code \
  --function-name ml-inference-lambda \
  --zip-file fileb://lambda_deployment.zip

# Update environment variables
aws lambda update-function-configuration \
  --function-name ml-inference-lambda \
  --environment Variables="{
    MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI,
    MODEL_NAME=production-model,
    MODEL_STAGE=Production
  }"
```

**5. Lambda Testing:**
```python
# tests/test_lambda.py
import json
from lambda.inference_handler import lambda_handler

def test_lambda_handler():
    """Test Lambda function locally"""
    event = {
        'body': json.dumps({
            'input': [1.0, 2.0, 3.0, 4.0]
        })
    }
    
    class MockContext:
        aws_request_id = 'test-request-id'
    
    response = lambda_handler(event, MockContext())
    
    assert response['statusCode'] == 200
    body = json.loads(response['body'])
    assert 'prediction' in body
    print(f"Prediction: {body['prediction']}")
```

**Demonstration:**
- [ ] Create Lambda function for model inference
- [ ] Set up API Gateway integration
- [ ] Configure IAM roles and permissions
- [ ] Implement dead letter queue
- [ ] Test Lambda function (local and deployed)
- [ ] Monitor Lambda metrics (duration, memory, errors)
- [ ] Compare costs: Lambda vs ECS
- [ ] Document serverless architecture decisions

**Checkpoint:** Can you serve predictions via Lambda?

---

#### Step 2.5: AWS Step Functions for ML Workflows (3-5 days) - Tier 1.5

**Learning Resources:**
- AWS Step Functions: https://docs.aws.amazon.com/step-functions/
- Step Functions with Lambda: https://docs.aws.amazon.com/step-functions/latest/dg/concepts-lambda.html
- Time: 3-5 days

**Why This Matters:**
- Cognizant job listing explicitly requires Step Functions
- Orchestrates complex ML workflows (training, validation, deployment)
- Demonstrates workflow automation skills

**What to Build:**

**1. Step Functions State Machine for ML Pipeline:**
```json
{
  "Comment": "ML Training and Deployment Pipeline",
  "StartAt": "ValidateData",
  "States": {
    "ValidateData": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:validate-data",
      "Next": "TrainModel",
      "Retry": [
        {
          "ErrorEquals": ["States.TaskFailed"],
          "IntervalSeconds": 2,
          "MaxAttempts": 3,
          "BackoffRate": 2.0
        }
      ],
      "Catch": [
        {
          "ErrorEquals": ["States.ALL"],
          "Next": "NotifyFailure",
          "ResultPath": "$.error"
        }
      ]
    },
    "TrainModel": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:train-model",
      "Next": "ValidateModel",
      "ResultPath": "$.training"
    },
    "ValidateModel": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:validate-model",
      "Next": "CheckPerformance",
      "ResultPath": "$.validation"
    },
    "CheckPerformance": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.validation.accuracy",
          "NumericGreaterThan": 0.85,
          "Next": "RegisterModel"
        }
      ],
      "Default": "NotifyFailure"
    },
    "RegisterModel": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:register-model",
      "Next": "DeployModel",
      "ResultPath": "$.registration"
    },
    "DeployModel": {
      "Type": "Parallel",
      "Branches": [
        {
          "StartAt": "DeployToStaging",
          "States": {
            "DeployToStaging": {
              "Type": "Task",
              "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:deploy-staging",
              "End": true
            }
          }
        },
        {
          "StartAt": "UpdateFeatureStore",
          "States": {
            "UpdateFeatureStore": {
              "Type": "Task",
              "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:update-features",
              "End": true
            }
          }
        }
      ],
      "Next": "NotifySuccess"
    },
    "NotifySuccess": {
      "Type": "Task",
      "Resource": "arn:aws:sns:REGION:ACCOUNT:topic:ml-notifications",
      "End": true
    },
    "NotifyFailure": {
      "Type": "Task",
      "Resource": "arn:aws:sns:REGION:ACCOUNT:topic:ml-notifications",
      "End": true
    }
  }
}
```

**2. Terraform Step Functions Configuration:**
```hcl
# infrastructure/terraform/aws/step_functions.tf
resource "aws_sfn_state_machine" "ml_pipeline" {
  name     = "ml-training-pipeline"
  role_arn = aws_iam_role.step_functions.arn
  
  definition = jsonencode({
    Comment = "ML Training and Deployment Pipeline"
    StartAt = "ValidateData"
    States = {
      ValidateData = {
        Type = "Task"
        Resource = aws_lambda_function.validate_data.arn
        Next = "TrainModel"
        Retry = [{
          ErrorEquals = ["States.TaskFailed"]
          IntervalSeconds = 2
          MaxAttempts = 3
          BackoffRate = 2.0
        }]
        Catch = [{
          ErrorEquals = ["States.ALL"]
          Next = "NotifyFailure"
          ResultPath = "$.error"
        }]
      }
      TrainModel = {
        Type = "Task"
        Resource = aws_lambda_function.train_model.arn
        Next = "ValidateModel"
        ResultPath = "$.training"
      }
      ValidateModel = {
        Type = "Task"
        Resource = aws_lambda_function.validate_model.arn
        Next = "CheckPerformance"
        ResultPath = "$.validation"
      }
      CheckPerformance = {
        Type = "Choice"
        Choices = [{
          Variable = "$.validation.accuracy"
          NumericGreaterThan = 0.85
          Next = "RegisterModel"
        }]
        Default = "NotifyFailure"
      }
      RegisterModel = {
        Type = "Task"
        Resource = aws_lambda_function.register_model.arn
        Next = "DeployModel"
        ResultPath = "$.registration"
      }
      DeployModel = {
        Type = "Parallel"
        Branches = [
          {
            StartAt = "DeployToStaging"
            States = {
              DeployToStaging = {
                Type = "Task"
                Resource = aws_lambda_function.deploy_staging.arn
                End = true
              }
            }
          }
        ]
        Next = "NotifySuccess"
      }
      NotifySuccess = {
        Type = "Task"
        Resource = aws_sns_topic.ml_notifications.arn
        End = true
      }
      NotifyFailure = {
        Type = "Task"
        Resource = aws_sns_topic.ml_notifications.arn
        End = true
      }
    }
  })
  
  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.step_functions.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }
  
  tracing_configuration {
    enabled = true
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "step_functions" {
  name              = "/aws/stepfunctions/ml-pipeline"
  retention_in_days = 14
}

# SNS Topic for notifications
resource "aws_sns_topic" "ml_notifications" {
  name = "ml-pipeline-notifications"
}
```

**3. Lambda Functions for Step Functions:**
```python
# lambda/step_functions/validate_data.py
import json
import boto3
import pandas as pd
from typing import Dict, Any

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Validate training data before model training"""
    s3 = boto3.client('s3')
    
    # Get data from S3
    bucket = event['data_bucket']
    key = event['data_key']
    
    # Download and validate
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])
    
    # Validation checks
    checks = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'null_percentage': df.isnull().sum().sum() / (len(df) * len(df.columns)),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Fail if validation fails
    if checks['null_percentage'] > 0.1:
        raise ValueError(f"Too many nulls: {checks['null_percentage']}")
    
    if checks['row_count'] < 100:
        raise ValueError(f"Insufficient data: {checks['row_count']} rows")
    
    return {
        'statusCode': 200,
        'validation': checks,
        'data_bucket': bucket,
        'data_key': key
    }
```

**4. Trigger Step Functions from Event:**
```python
# lambda/trigger_pipeline.py
import boto3
import json

def lambda_handler(event, context):
    """Trigger ML pipeline when new data arrives"""
    sfn = boto3.client('stepfunctions')
    
    # Extract S3 event details
    s3_event = event['Records'][0]['s3']
    bucket = s3_event['bucket']['name']
    key = s3_event['object']['key']
    
    # Start Step Functions execution
    execution_input = {
        'data_bucket': bucket,
        'data_key': key,
        'timestamp': event['Records'][0]['eventTime']
    }
    
    response = sfn.start_execution(
        stateMachineArn=os.environ['STEP_FUNCTIONS_ARN'],
        name=f"ml-pipeline-{context.aws_request_id}",
        input=json.dumps(execution_input)
    )
    
    return {
        'statusCode': 200,
        'executionArn': response['executionArn']
    }
```

**Demonstration:**
- [ ] Create Step Functions state machine
- [ ] Implement Lambda functions for each step
- [ ] Add error handling and retries
- [ ] Set up CloudWatch logging
- [ ] Configure SNS notifications
- [ ] Test complete pipeline end-to-end
- [ ] Monitor Step Functions executions
- [ ] Document workflow architecture

**Checkpoint:** Can you orchestrate ML workflows with Step Functions?

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

#### Step 4.1: LLM Integration - Hugging Face (1 week) - Tier 3

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

**Checkpoint:** Can you serve LLM predictions with Hugging Face?

---

#### Step 4.1b: AWS Bedrock LLM Integration (3-5 days) - Tier 3

**Learning Resources:**
- AWS Bedrock: https://docs.aws.amazon.com/bedrock/
- Bedrock Python SDK: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime.html
- Time: 3-5 days

**Why This Matters:**
- Cognizant job listing explicitly requires Bedrock
- Managed LLM service (no infrastructure to manage)
- Production-ready alternative to self-hosted models

**What to Build:**

**1. Bedrock Client and Model Invocation:**
```python
# training/models/bedrock_llm.py
import boto3
import json
from typing import Dict, List, Optional
from botocore.exceptions import ClientError

class BedrockLLM:
    def __init__(self, region_name: str = "us-east-1"):
        """Initialize Bedrock runtime client"""
        self.client = boto3.client('bedrock-runtime', region_name=region_name)
        self.region = region_name
    
    def invoke_model(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict:
        """
        Invoke Bedrock model
        
        Supported models:
        - anthropic.claude-v2 (Claude)
        - amazon.titan-text-lite-v1 (Titan)
        - ai21.j2-ultra-v1 (Jurassic)
        """
        # Prepare request body based on model
        if "claude" in model_id.lower():
            body = {
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
        elif "titan" in model_id.lower():
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": top_p
                }
            }
        else:
            body = {
                "prompt": prompt,
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p
            }
        
        try:
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response['body'].read())
            
            # Parse response based on model
            if "claude" in model_id.lower():
                return {
                    "text": response_body.get("completion", ""),
                    "stop_reason": response_body.get("stop_reason", ""),
                    "tokens_used": response_body.get("usage", {}).get("total_tokens", 0)
                }
            elif "titan" in model_id.lower():
                return {
                    "text": response_body.get("results", [{}])[0].get("outputText", ""),
                    "tokens_used": response_body.get("results", [{}])[0].get("tokenCount", 0)
                }
            else:
                return {
                    "text": response_body.get("completions", [{}])[0].get("data", {}).get("text", ""),
                    "tokens_used": response_body.get("completions", [{}])[0].get("data", {}).get("tokenCount", 0)
                }
        
        except ClientError as e:
            raise Exception(f"Bedrock invocation failed: {str(e)}")
    
    def stream_invoke(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 500
    ):
        """Stream responses from Bedrock"""
        body = {
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": max_tokens
        }
        
        response = self.client.invoke_model_with_response_stream(
            modelId=model_id,
            body=json.dumps(body)
        )
        
        for event in response['body']:
            chunk = json.loads(event['chunk']['bytes'])
            if 'completion' in chunk:
                yield chunk['completion']
    
    def estimate_cost(self, model_id: str, tokens: int) -> float:
        """Estimate cost based on model and tokens"""
        # Pricing (as of 2024, check AWS for current)
        pricing = {
            "anthropic.claude-v2": {"input": 0.008, "output": 0.024},  # per 1K tokens
            "amazon.titan-text-lite-v1": {"input": 0.0008, "output": 0.0016},
            "ai21.j2-ultra-v1": {"input": 0.0125, "output": 0.0125}
        }
        
        if model_id not in pricing:
            return 0.0
        
        # Rough estimate (assuming 50/50 input/output split)
        cost = (tokens / 1000) * (
            pricing[model_id]["input"] * 0.5 + 
            pricing[model_id]["output"] * 0.5
        )
        return cost
```

**2. Bedrock API Endpoint:**
```python
# api/inference/bedrock_inference.py
from fastapi import APIRouter, HTTPException
from training.models.bedrock_llm import BedrockLLM
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
bedrock_llm = BedrockLLM()

class BedrockRequest(BaseModel):
    prompt: str
    model_id: str = "anthropic.claude-v2"
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9

class BedrockResponse(BaseModel):
    text: str
    tokens_used: int
    estimated_cost: float
    model_id: str

@router.post("/api/bedrock/generate", response_model=BedrockResponse)
async def generate_with_bedrock(request: BedrockRequest):
    """Generate text using AWS Bedrock"""
    try:
        result = bedrock_llm.invoke_model(
            model_id=request.model_id,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        cost = bedrock_llm.estimate_cost(request.model_id, result["tokens_used"])
        
        return BedrockResponse(
            text=result["text"],
            tokens_used=result["tokens_used"],
            estimated_cost=cost,
            model_id=request.model_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/bedrock/stream")
async def stream_with_bedrock(request: BedrockRequest):
    """Stream responses from Bedrock"""
    from fastapi.responses import StreamingResponse
    
    def generate():
        for chunk in bedrock_llm.stream_invoke(
            model_id=request.model_id,
            prompt=request.prompt,
            max_tokens=request.max_tokens
        ):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

**3. Terraform Bedrock Configuration:**
```hcl
# infrastructure/terraform/aws/bedrock.tf
# IAM policy for Bedrock access
resource "aws_iam_role_policy" "bedrock_access" {
  name = "bedrock-access-policy"
  role = aws_iam_role.ecs_task_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = [
          "arn:aws:bedrock:*::foundation-model/anthropic.claude-v2",
          "arn:aws:bedrock:*::foundation-model/amazon.titan-text-lite-v1",
          "arn:aws:bedrock:*::foundation-model/ai21.j2-ultra-v1"
        ]
      }
    ]
  })
}
```

**4. Bedrock Cost Tracking:**
```python
# monitoring/bedrock_cost_tracking.py
import boto3
from datetime import datetime, timedelta
from collections import defaultdict

class BedrockCostTracker:
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.cost_data = defaultdict(lambda: {'tokens': 0, 'cost': 0.0})
    
    def track_invocation(
        self,
        model_id: str,
        tokens_used: int,
        cost: float
    ):
        """Track Bedrock invocation costs"""
        self.cost_data[model_id]['tokens'] += tokens_used
        self.cost_data[model_id]['cost'] += cost
        
        # Send to CloudWatch
        self.cloudwatch.put_metric_data(
            Namespace='ML/Bedrock',
            MetricData=[
                {
                    'MetricName': 'TokensUsed',
                    'Dimensions': [
                        {'Name': 'ModelId', 'Value': model_id}
                    ],
                    'Value': tokens_used,
                    'Unit': 'Count',
                    'Timestamp': datetime.utcnow()
                },
                {
                    'MetricName': 'Cost',
                    'Dimensions': [
                        {'Name': 'ModelId', 'Value': model_id}
                    ],
                    'Value': cost,
                    'Unit': 'None',
                    'Timestamp': datetime.utcnow()
                }
            ]
        )
    
    def get_daily_costs(self, days: int = 7) -> Dict:
        """Get daily costs from CloudWatch"""
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        
        response = self.cloudwatch.get_metric_statistics(
            Namespace='ML/Bedrock',
            MetricName='Cost',
            StartTime=start,
            EndTime=end,
            Period=86400,  # 1 day
            Statistics=['Sum']
        )
        
        return response
```

**Demonstration:**
- [ ] Set up Bedrock access (request model access in AWS console)
- [ ] Implement Bedrock client with multiple models
- [ ] Create API endpoints for Bedrock inference
- [ ] Implement streaming responses
- [ ] Track costs per model
- [ ] Compare Bedrock vs Hugging Face (cost, latency, quality)
- [ ] Document when to use Bedrock vs self-hosted

**Checkpoint:** Can you use AWS Bedrock for LLM inference?

---

#### Step 4.1c: Apache Spark for Data Processing (Optional, 3-5 days) - Tier 3

**Learning Resources:**
- Spark Python API: https://spark.apache.org/docs/latest/api/python/
- PySpark Tutorial: https://spark.apache.org/docs/latest/api/python/getting_started/index.html
- Time: 3-5 days (optional)

**Why This Matters:**
- Modernizing Medicine job listing requires Spark
- Handles large-scale data processing
- Useful for feature engineering at scale

**What to Build:**

**1. Spark Session Setup:**
```python
# data_processing/spark_setup.py
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import col, avg, stddev, count

class SparkDataProcessor:
    def __init__(self, app_name: str = "MLDataProcessing"):
        """Initialize Spark session"""
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
    
    def read_csv(self, path: str, schema: StructType = None):
        """Read CSV file into Spark DataFrame"""
        if schema:
            return self.spark.read.schema(schema).csv(path, header=True)
        return self.spark.read.csv(path, header=True, inferSchema=True)
    
    def process_features(self, df):
        """Process features for ML training"""
        # Example: Feature engineering
        processed_df = df \
            .withColumn("feature_1_normalized", 
                       (col("feature_1") - avg("feature_1").over()) / 
                       stddev("feature_1").over()) \
            .withColumn("feature_2_binned",
                       col("feature_2") // 10)
        
        return processed_df
    
    def write_parquet(self, df, path: str):
        """Write DataFrame to Parquet format"""
        df.write.mode("overwrite").parquet(path)
    
    def stop(self):
        """Stop Spark session"""
        self.spark.stop()
```

**2. Spark Feature Engineering Pipeline:**
```python
# data_processing/spark_features.py
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, 
    StringIndexer, OneHotEncoder
)
from pyspark.ml import Pipeline

class SparkFeaturePipeline:
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def create_pipeline(self, categorical_cols: list, numeric_cols: list):
        """Create ML feature pipeline"""
        stages = []
        
        # String indexing for categorical
        for col_name in categorical_cols:
            indexer = StringIndexer(
                inputCol=col_name,
                outputCol=f"{col_name}_indexed"
            )
            encoder = OneHotEncoder(
                inputCol=f"{col_name}_indexed",
                outputCol=f"{col_name}_encoded"
            )
            stages.extend([indexer, encoder])
        
        # Assemble features
        feature_cols = [f"{c}_encoded" for c in categorical_cols] + numeric_cols
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features"
        )
        stages.append(assembler)
        
        # Scale features
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        stages.append(scaler)
        
        return Pipeline(stages=stages)
```

**3. Spark Integration with MLflow:**
```python
# training/scripts/train_spark.py
import mlflow
import mlflow.spark
from pyspark.ml.regression import LinearRegression
from data_processing.spark_features import SparkFeaturePipeline

def train_spark_model(spark_session, data_path: str):
    """Train model using Spark MLlib"""
    mlflow.set_experiment("spark-ml-training")
    
    with mlflow.start_run():
        # Load data
        df = spark_session.read.parquet(data_path)
        
        # Create feature pipeline
        pipeline = SparkFeaturePipeline(spark_session)
        feature_pipeline = pipeline.create_pipeline(
            categorical_cols=["category"],
            numeric_cols=["feature_1", "feature_2"]
        )
        
        # Fit pipeline
        model_pipeline = feature_pipeline.fit(df)
        processed_df = model_pipeline.transform(df)
        
        # Train model
        lr = LinearRegression(
            featuresCol="scaled_features",
            labelCol="target"
        )
        model = lr.fit(processed_df)
        
        # Log to MLflow
        mlflow.spark.log_model(model, "spark-model")
        mlflow.log_param("algorithm", "LinearRegression")
        mlflow.log_metric("rmse", model.summary.rootMeanSquaredError)
```

**Demonstration:**
- [ ] Set up Spark session (local or EMR)
- [ ] Process large datasets with Spark
- [ ] Create feature engineering pipeline
- [ ] Train model with Spark MLlib
- [ ] Integrate with MLflow
- [ ] Compare Spark vs Pandas (performance, scale)
- [ ] Document when to use Spark

**Checkpoint:** Can you process large datasets with Spark?

---

#### Step 4.1d: Model Optimization (ONNX/TensorRT/vLLM) (1 week) - Tier 2

**Learning Resources:**
- ONNX: https://onnx.ai/
- TensorRT: https://developer.nvidia.com/tensorrt
- vLLM: https://docs.vllm.ai/
- Time: 1 week

**Why This Matters:**
- Coactive AI and performance-focused roles require optimization
- Demonstrates production optimization skills
- Critical for high-throughput inference
- Shows understanding of model serving performance

**What to Build:**

**1. ONNX Model Conversion and Serving:**
```python
# training/models/onnx_converter.py
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np

class ONNXConverter:
    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape
    
    def convert_to_onnx(self, output_path: str):
        """Convert PyTorch model to ONNX"""
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *self.input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        return output_path
    
    def quantize_model(self, model_path: str, quantized_path: str):
        """Quantize ONNX model for faster inference"""
        quantize_dynamic(
            model_input=model_path,
            model_output=quantized_path,
            weight_type=QuantType.QUInt8
        )
        return quantized_path

class ONNXRuntimeInference:
    def __init__(self, model_path: str, providers=None):
        if providers is None:
            providers = ['CPUExecutionProvider', 'CUDAExecutionProvider']
        
        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
    
    def predict(self, input_data: np.ndarray):
        """Run inference with ONNX Runtime"""
        outputs = self.session.run(
            None,
            {self.input_name: input_data.astype(np.float32)}
        )
        return outputs[0]
    
    def benchmark(self, input_data: np.ndarray, iterations: int = 100):
        """Benchmark inference performance"""
        import time
        
        # Warmup
        for _ in range(10):
            self.predict(input_data)
        
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            self.predict(input_data)
        end = time.time()
        
        avg_time = (end - start) / iterations
        throughput = 1.0 / avg_time
        
        return {
            'avg_latency_ms': avg_time * 1000,
            'throughput_per_sec': throughput,
            'total_time_sec': end - start
        }
```

**2. TensorRT Optimization:**
```python
# training/models/tensorrt_converter.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTConverter:
    def __init__(self, onnx_model_path: str):
        self.onnx_model_path = onnx_model_path
        self.logger = trt.Logger(trt.Logger.WARNING)
    
    def build_engine(self, max_batch_size: int = 1, fp16_mode: bool = True):
        """Build TensorRT engine from ONNX model"""
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX model
        with open(self.onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        if fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        return engine
    
    def save_engine(self, engine, engine_path: str):
        """Save TensorRT engine to file"""
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
    
    def load_engine(self, engine_path: str):
        """Load TensorRT engine from file"""
        runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

class TensorRTInference:
    def __init__(self, engine):
        self.engine = engine
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
    
    def allocate_buffers(self):
        """Allocate GPU buffers"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream
    
    def infer(self, input_data: np.ndarray):
        """Run inference with TensorRT"""
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )
        
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        
        self.stream.synchronize()
        
        return self.outputs[0]['host']
```

**3. vLLM for LLM Optimization:**
```python
# training/models/vllm_serving.py
from vllm import LLM, SamplingParams
from typing import List, Dict

class vLLMServer:
    def __init__(self, model_name: str, tensor_parallel_size: int = 1):
        """
        Initialize vLLM server
        
        Args:
            model_name: Hugging Face model name
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=4096
        )
    
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[Dict]:
        """Generate text with vLLM"""
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            results.append({
                'prompt': output.prompt,
                'generated_text': output.outputs[0].text,
                'tokens': len(output.outputs[0].token_ids),
                'finish_reason': output.outputs[0].finish_reason
            })
        
        return results
    
    def benchmark(self, prompts: List[str], iterations: int = 10):
        """Benchmark vLLM performance"""
        import time
        
        # Warmup
        self.generate(prompts[:1], max_tokens=10)
        
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            self.generate(prompts, max_tokens=100)
        end = time.time()
        
        total_tokens = sum(len(p) for p in prompts) * iterations
        tokens_per_sec = total_tokens / (end - start)
        
        return {
            'total_time_sec': end - start,
            'tokens_per_second': tokens_per_sec,
            'avg_latency_per_request_ms': ((end - start) / iterations) * 1000
        }
```

**4. FastAPI Endpoint for Optimized Models:**
```python
# api/inference/optimized_inference.py
from fastapi import APIRouter, HTTPException
from training.models.onnx_converter import ONNXRuntimeInference
from training.models.tensorrt_converter import TensorRTInference, TensorRTConverter
from training.models.vllm_serving import vLLMServer
import numpy as np

router = APIRouter()

# Initialize optimized models
onnx_inference = None
tensorrt_inference = None
vllm_server = None

def initialize_optimized_models():
    """Initialize optimized models"""
    global onnx_inference, tensorrt_inference, vllm_server
    
    # ONNX Runtime
    onnx_inference = ONNXRuntimeInference("models/model.onnx")
    
    # TensorRT (if available)
    try:
        converter = TensorRTConverter("models/model.onnx")
        engine = converter.load_engine("models/model.trt")
        tensorrt_inference = TensorRTInference(engine)
    except Exception as e:
        print(f"TensorRT not available: {e}")
    
    # vLLM (if LLM model)
    try:
        vllm_server = vLLMServer("gpt2", tensor_parallel_size=1)
    except Exception as e:
        print(f"vLLM not available: {e}")

@router.post("/api/inference/onnx")
async def onnx_predict(input_data: List[List[float]]):
    """Predict using ONNX Runtime"""
    if onnx_inference is None:
        raise HTTPException(status_code=503, detail="ONNX model not loaded")
    
    input_array = np.array(input_data, dtype=np.float32)
    prediction = onnx_inference.predict(input_array)
    
    return {"prediction": prediction.tolist()}

@router.post("/api/inference/tensorrt")
async def tensorrt_predict(input_data: List[List[float]]):
    """Predict using TensorRT"""
    if tensorrt_inference is None:
        raise HTTPException(status_code=503, detail="TensorRT engine not loaded")
    
    input_array = np.array(input_data, dtype=np.float32)
    prediction = tensorrt_inference.infer(input_array)
    
    return {"prediction": prediction.tolist()}

@router.post("/api/inference/vllm")
async def vllm_generate(prompts: List[str], max_tokens: int = 100):
    """Generate text using vLLM"""
    if vllm_server is None:
        raise HTTPException(status_code=503, detail="vLLM server not available")
    
    results = vllm_server.generate(prompts, max_tokens=max_tokens)
    return {"results": results}

@router.get("/api/inference/benchmark")
async def benchmark_models():
    """Benchmark all optimized models"""
    dummy_input = np.random.randn(1, 10).astype(np.float32)
    
    results = {}
    
    if onnx_inference:
        results['onnx'] = onnx_inference.benchmark(dummy_input)
    
    if tensorrt_inference:
        # TensorRT benchmark
        import time
        start = time.time()
        for _ in range(100):
            tensorrt_inference.infer(dummy_input)
        end = time.time()
        results['tensorrt'] = {
            'avg_latency_ms': ((end - start) / 100) * 1000,
            'throughput_per_sec': 100 / (end - start)
        }
    
    if vllm_server:
        results['vllm'] = vllm_server.benchmark(["Test prompt"], iterations=10)
    
    return results
```

**Demonstration:**
- [ ] Convert PyTorch model to ONNX
- [ ] Quantize ONNX model
- [ ] Serve ONNX model with ONNX Runtime
- [ ] Convert ONNX to TensorRT
- [ ] Serve TensorRT model
- [ ] Set up vLLM for LLM serving
- [ ] Benchmark all optimization methods
- [ ] Compare performance (latency, throughput)
- [ ] Document optimization strategies

**Checkpoint:** Can you optimize models for production performance?

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
- [ ] **AWS:** Deployed and accessible (ECS)
- [ ] **Terraform:** Infrastructure as Code
- [ ] **Testing:** Unit and integration tests

### Should Demonstrate (Tier 1.5 - Critical for Jobs):

- [ ] **AWS Lambda:** Serverless model inference
- [ ] **AWS Step Functions:** ML workflow orchestration
- [ ] **API Gateway:** Lambda integration

### Should Demonstrate (Tier 2):

- [ ] **Security:** Secrets management, encryption
- [ ] **Cost Tracking:** Monitoring and reporting
- [ ] **Multi-Cloud:** AWS + one other (Azure/GCP)

### Nice to Demonstrate (Tier 3):

- [ ] **LLM:** Hugging Face deployment
- [ ] **AWS Bedrock:** Managed LLM service
- [ ] **Spark:** Large-scale data processing
- [ ] **Model Optimization:** ONNX, TensorRT, vLLM
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

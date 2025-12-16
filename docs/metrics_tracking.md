# Metrics Tracking Guide

This guide explains how to implement comprehensive metrics tracking for the ML Training & Deployment Pipeline to demonstrate production readiness and real-world usage.

## Overview

Tracking metrics is crucial for:
- **Job Applications:** Show real production experience with concrete numbers
- **Portfolio:** Demonstrate impact and optimization
- **Interviews:** Provide specific examples of your work
- **Learning:** Understand system performance and costs

## Metrics to Track

### 1. API Performance Metrics

#### Request Metrics

**What to Track:**
- Total requests served
- Requests per second (RPS)
- Request latency (p50, p95, p99)
- Error rate (4xx, 5xx)
- Success rate

**Implementation:**

```python
# api/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Request counters
request_count = Counter(
    'ml_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

# Latency histogram
request_latency = Histogram(
    'ml_api_request_duration_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# Active requests
active_requests = Gauge(
    'ml_api_active_requests',
    'Number of active requests'
)

# Usage in FastAPI
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        active_requests.inc()
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status = response.status_code
            request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status=status
            ).inc()
            return response
        finally:
            latency = time.time() - start_time
            request_latency.labels(endpoint=request.url.path).observe(latency)
            active_requests.dec()
```

**Key Metrics to Report:**
- Total predictions served: `sum(ml_api_requests_total{endpoint="/api/predict"})`
- Average latency: `rate(ml_api_request_duration_seconds_sum[5m]) / rate(ml_api_request_duration_seconds_count[5m])`
- Error rate: `rate(ml_api_requests_total{status=~"5.."}[5m]) / rate(ml_api_requests_total[5m])`

---

### 2. Model Performance Metrics

#### Prediction Metrics

**What to Track:**
- Total predictions made
- Prediction latency (model inference time)
- Model accuracy (if ground truth available)
- Prediction confidence scores
- Model version in use

**Implementation:**

```python
# api/inference/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Prediction counters
predictions_total = Counter(
    'ml_model_predictions_total',
    'Total number of predictions',
    ['model_name', 'model_version']
)

# Inference latency
inference_latency = Histogram(
    'ml_model_inference_seconds',
    'Model inference time in seconds',
    ['model_name'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

# Model accuracy (if ground truth available)
model_accuracy = Gauge(
    'ml_model_accuracy',
    'Model accuracy score',
    ['model_name', 'model_version']
)

# Usage in inference code
import time

def predict_with_metrics(model, input_data, model_name, model_version):
    start_time = time.time()
    
    try:
        prediction = model.predict(input_data)
        latency = time.time() - start_time
        
        predictions_total.labels(
            model_name=model_name,
            model_version=model_version
        ).inc()
        
        inference_latency.labels(model_name=model_name).observe(latency)
        
        return prediction
    except Exception as e:
        # Track errors
        predictions_total.labels(
            model_name=model_name,
            model_version=model_version
        ).inc()
        raise
```

**Key Metrics to Report:**
- Total predictions: `sum(ml_model_predictions_total)`
- Average inference time: `rate(ml_model_inference_seconds_sum[5m]) / rate(ml_model_inference_seconds_count[5m])`
- Predictions per model version: `sum by (model_version) (ml_model_predictions_total)`

---

### 3. Cost Tracking Metrics

#### AWS Cost Metrics

**What to Track:**
- Daily AWS costs
- Cost by service (ECS, ECR, CloudWatch, etc.)
- Cost per prediction
- Cost trends over time

**Implementation:**

```python
# monitoring/cost_tracking.py
import boto3
from datetime import datetime, timedelta
from prometheus_client import Gauge
import time

# Cost metrics
daily_cost = Gauge(
    'aws_daily_cost_usd',
    'Daily AWS cost in USD'
)

cost_per_prediction = Gauge(
    'aws_cost_per_prediction_usd',
    'Cost per prediction in USD'
)

cost_by_service = Gauge(
    'aws_cost_by_service_usd',
    'AWS cost by service',
    ['service']
)

class CostTracker:
    def __init__(self):
        self.ce_client = boto3.client('ce')  # Cost Explorer
    
    def update_metrics(self):
        """Update Prometheus metrics with current costs"""
        # Get daily costs
        end = datetime.now()
        start = end - timedelta(days=1)
        
        response = self.ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start.strftime('%Y-%m-%d'),
                'End': end.strftime('%Y-%m-%d')
            },
            Granularity='DAILY',
            Metrics=['BlendedCost']
        )
        
        if response['ResultsByTime']:
            cost = float(response['ResultsByTime'][0]['Total']['BlendedCost']['Amount'])
            daily_cost.set(cost)
        
        # Get costs by service
        response = self.ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'End': datetime.now().strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['BlendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
        )
        
        for group in response['ResultsByTime'][0]['Groups']:
            service = group['Keys'][0]
            cost = float(group['Metrics']['BlendedCost']['Amount'])
            cost_by_service.labels(service=service).set(cost)
        
        # Calculate cost per prediction
        total_predictions = sum(predictions_total._value.get((('model_name', 'default'), ('model_version', '1')), 0))
        if total_predictions > 0:
            monthly_cost = sum([cost_by_service._value.get((('service', s),), 0) for s in ['Amazon Elastic Container Service', 'Amazon EC2 Container Registry']])
            cost_per_pred = monthly_cost / total_predictions
            cost_per_prediction.set(cost_per_pred)

# Run cost tracking periodically
import threading

def run_cost_tracker():
    tracker = CostTracker()
    while True:
        tracker.update_metrics()
        time.sleep(3600)  # Update every hour

# Start in background thread
cost_tracker_thread = threading.Thread(target=run_cost_tracker, daemon=True)
cost_tracker_thread.start()
```

**Key Metrics to Report:**
- Daily cost: `aws_daily_cost_usd`
- Cost per prediction: `aws_cost_per_prediction_usd`
- Cost by service: `aws_cost_by_service_usd{service="Amazon Elastic Container Service"}`

---

### 4. LLM-Specific Metrics

#### Token Usage and Costs

**What to Track:**
- Total tokens used (input + output)
- LLM API calls
- LLM inference latency
- LLM costs per request
- Token usage by model

**Implementation:**

```python
# api/inference/llm_metrics.py
from prometheus_client import Counter, Histogram, Gauge

# LLM metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['model_name', 'provider']
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens used',
    ['model_name', 'token_type']  # token_type: input, output
)

llm_inference_latency = Histogram(
    'llm_inference_seconds',
    'LLM inference time',
    ['model_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

llm_cost_total = Counter(
    'llm_cost_usd',
    'Total LLM costs in USD',
    ['model_name', 'provider']
)

# Usage in LLM inference
def generate_with_metrics(model_name, provider, prompt, max_tokens):
    start_time = time.time()
    
    try:
        response = llm_client.generate(
            model=model_name,
            prompt=prompt,
            max_tokens=max_tokens
        )
        
        latency = time.time() - start_time
        input_tokens = len(prompt.split())
        output_tokens = len(response.text.split())
        
        # Track metrics
        llm_requests_total.labels(model_name=model_name, provider=provider).inc()
        llm_tokens_total.labels(model_name=model_name, token_type='input').inc(input_tokens)
        llm_tokens_total.labels(model_name=model_name, token_type='output').inc(output_tokens)
        llm_inference_latency.labels(model_name=model_name).observe(latency)
        
        # Calculate and track cost (example rates)
        cost = calculate_llm_cost(model_name, input_tokens, output_tokens)
        llm_cost_total.labels(model_name=model_name, provider=provider).inc(cost)
        
        return response
    except Exception as e:
        llm_requests_total.labels(model_name=model_name, provider=provider).inc()
        raise

def calculate_llm_cost(model_name, input_tokens, output_tokens):
    """Calculate cost based on model pricing"""
    # Example: GPT-4 pricing (update with actual rates)
    pricing = {
        'gpt-4': {'input': 0.03 / 1000, 'output': 0.06 / 1000},
        'gpt-3.5-turbo': {'input': 0.0015 / 1000, 'output': 0.002 / 1000}
    }
    
    if model_name in pricing:
        cost = (input_tokens * pricing[model_name]['input'] + 
                output_tokens * pricing[model_name]['output'])
        return cost
    return 0
```

**Key Metrics to Report:**
- Total LLM requests: `sum(llm_requests_total)`
- Total tokens used: `sum(llm_tokens_total)`
- Average LLM latency: `rate(llm_inference_seconds_sum[5m]) / rate(llm_inference_seconds_count[5m])`
- Total LLM costs: `sum(llm_cost_usd)`

---

### 5. Resource Utilization Metrics

#### Infrastructure Metrics

**What to Track:**
- CPU utilization
- Memory usage
- Container/pod metrics
- Network I/O

**Implementation:**

```python
# monitoring/resource_metrics.py
import psutil
from prometheus_client import Gauge

# Resource metrics
cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'CPU usage percentage'
)

memory_usage = Gauge(
    'system_memory_usage_bytes',
    'Memory usage in bytes'
)

memory_available = Gauge(
    'system_memory_available_bytes',
    'Available memory in bytes'
)

def update_resource_metrics():
    """Update system resource metrics"""
    cpu_usage.set(psutil.cpu_percent(interval=1))
    memory = psutil.virtual_memory()
    memory_usage.set(memory.used)
    memory_available.set(memory.available)

# Update every 10 seconds
import threading
import time

def run_resource_tracker():
    while True:
        update_resource_metrics()
        time.sleep(10)

resource_tracker_thread = threading.Thread(target=run_resource_tracker, daemon=True)
resource_tracker_thread.start()
```

---

## Metrics Dashboard

### Prometheus + Grafana Setup

**1. Expose Metrics Endpoint:**

```python
# api/app.py
from prometheus_client import make_asgi_app
from fastapi import FastAPI

app = FastAPI()

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

**2. Grafana Dashboard Queries:**

```promql
# Total Predictions Served
sum(increase(ml_model_predictions_total[1d]))

# Average Latency (p95)
histogram_quantile(0.95, rate(ml_api_request_duration_seconds_bucket[5m]))

# Error Rate
rate(ml_api_requests_total{status=~"5.."}[5m]) / rate(ml_api_requests_total[5m])

# Cost per Prediction
aws_cost_per_prediction_usd

# LLM Token Usage
sum(increase(llm_tokens_total[1d]))
```

---

## Metrics to Report for Job Applications

### Key Numbers to Track

**API Metrics:**
- Total requests served: `X,XXX requests`
- Average latency: `XX ms (p95)`
- Uptime: `99.X%`
- Error rate: `X.X%`

**Model Metrics:**
- Total predictions: `X,XXX predictions`
- Average inference time: `XX ms`
- Model accuracy: `XX%` (if applicable)

**Cost Metrics:**
- Daily AWS cost: `$X.XX`
- Cost per prediction: `$0.00XX`
- Monthly cost: `$XXX.XX`
- Cost optimization savings: `XX%`

**LLM Metrics:**
- Total LLM requests: `X,XXX`
- Total tokens used: `X,XXX,XXX`
- LLM costs: `$XX.XX`
- Average LLM latency: `X.X seconds`

**Infrastructure:**
- Average CPU usage: `XX%`
- Average memory usage: `XX%`
- Container uptime: `XX days`

---

## Implementation Checklist

- [ ] Add Prometheus client to requirements.txt
- [ ] Implement request metrics middleware
- [ ] Add model inference metrics
- [ ] Set up cost tracking (AWS Cost Explorer)
- [ ] Add LLM metrics (if using LLMs)
- [ ] Expose /metrics endpoint
- [ ] Set up Grafana dashboard
- [ ] Document key metrics in README
- [ ] Create metrics summary for portfolio

---

## Example Metrics Summary for Portfolio

```markdown
## Production Metrics (Last 30 Days)

- **Total Predictions Served:** 15,234
- **Average Latency:** 45ms (p95: 120ms)
- **Uptime:** 99.7%
- **Error Rate:** 0.3%
- **Cost per Prediction:** $0.0023
- **Monthly AWS Cost:** $35.04
- **LLM Requests:** 2,456 (if applicable)
- **Total Tokens:** 1,234,567 (if applicable)
```

---

## Next Steps

1. **Implement metrics tracking** in your API
2. **Deploy with metrics endpoint** exposed
3. **Set up Grafana** to visualize metrics
4. **Track for 1-2 weeks** to get real numbers
5. **Document metrics** in your portfolio
6. **Use in interviews** as concrete examples

**Remember:** Real metrics, even if small, are more valuable than no metrics. Start tracking now!


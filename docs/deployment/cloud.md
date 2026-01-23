# Cloud Deployment (GCP)

Deploy MetaPathPredict on Google Cloud Platform.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Google Cloud Platform                    │
├──────────────────┬──────────────────┬───────────────────────┤
│   Cloud Storage  │  Artifact Reg.   │   Secret Manager      │
│   - Input data   │  - Docker images │   - API keys          │
│   - Model weights│                  │   - Credentials       │
│   - Results      │                  │                       │
├──────────────────┴──────────────────┴───────────────────────┤
│                    Google Kubernetes Engine                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Inference Pods  │  │ Training Jobs   │  │ MLflow       │ │
│  │ (GPU nodes)     │  │ (Preemptible)   │  │ Tracking     │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     Cloud Functions                          │
│              - Trigger pipeline on upload                    │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
gcloud init

# Authenticate
gcloud auth login
gcloud auth configure-docker

# Set project
export PROJECT_ID=your-project-id
gcloud config set project $PROJECT_ID
```

## Cloud Storage Setup

### Create Buckets

```bash
# Create buckets
gsutil mb -l us-central1 gs://${PROJECT_ID}-metapathpredict-data
gsutil mb -l us-central1 gs://${PROJECT_ID}-metapathpredict-models
gsutil mb -l us-central1 gs://${PROJECT_ID}-metapathpredict-results

# Upload model weights
gsutil -m cp -r data/weights/* gs://${PROJECT_ID}-metapathpredict-models/

# Set lifecycle for results (delete after 30 days)
cat > lifecycle.json << EOF
{
  "rule": [{
    "action": {"type": "Delete"},
    "condition": {"age": 30}
  }]
}
EOF
gsutil lifecycle set lifecycle.json gs://${PROJECT_ID}-metapathpredict-results
```

## Artifact Registry

### Push Docker Images

```bash
# Create repository
gcloud artifacts repositories create metapathpredict \
    --repository-format=docker \
    --location=us-central1

# Tag and push
docker tag metapathpredict:gpu \
    us-central1-docker.pkg.dev/${PROJECT_ID}/metapathpredict/inference:latest

docker push us-central1-docker.pkg.dev/${PROJECT_ID}/metapathpredict/inference:latest
```

## GKE Deployment

### Create Cluster

```bash
# Create GKE cluster with GPU node pool
gcloud container clusters create metapathpredict-cluster \
    --zone=us-central1-a \
    --num-nodes=1 \
    --machine-type=n1-standard-4

# Add GPU node pool
gcloud container node-pools create gpu-pool \
    --cluster=metapathpredict-cluster \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --num-nodes=1 \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=5

# Get credentials
gcloud container clusters get-credentials metapathpredict-cluster --zone=us-central1-a

# Install NVIDIA driver
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### Kubernetes Manifests

#### Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: metapathpredict
```

#### ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: metapathpredict-config
  namespace: metapathpredict
data:
  MODEL_BUCKET: "gs://your-project-metapathpredict-models"
  RESULTS_BUCKET: "gs://your-project-metapathpredict-results"
  BATCH_SIZE: "64"
  NUM_WORKERS: "4"
```

#### Inference Deployment

```yaml
# k8s/inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference
  namespace: metapathpredict
spec:
  replicas: 2
  selector:
    matchLabels:
      app: inference
  template:
    metadata:
      labels:
        app: inference
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
      containers:
      - name: inference
        image: us-central1-docker.pkg.dev/PROJECT_ID/metapathpredict/inference:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2"
        envFrom:
        - configMapRef:
            name: metapathpredict-config
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        volumeMounts:
        - name: model-cache
          mountPath: /app/weights
      initContainers:
      - name: download-model
        image: google/cloud-sdk:slim
        command: ['sh', '-c', 'gsutil -m cp -r gs://PROJECT_ID-metapathpredict-models/* /app/weights/']
        volumeMounts:
        - name: model-cache
          mountPath: /app/weights
      volumes:
      - name: model-cache
        emptyDir: {}
```

#### Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: inference-service
  namespace: metapathpredict
spec:
  selector:
    app: inference
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### HorizontalPodAutoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
  namespace: metapathpredict
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy

```bash
# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -n metapathpredict
kubectl get services -n metapathpredict
```

## Training Jobs

### GKE Job

```yaml
# k8s/training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: training-job
  namespace: metapathpredict
spec:
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
      containers:
      - name: training
        image: us-central1-docker.pkg.dev/PROJECT_ID/metapathpredict/training:latest
        command: ["python", "-m", "metapathpredict.train"]
        args:
        - "--data-dir=/data/unified"
        - "--output-dir=/output"
        - "--epochs=100"
        - "--approach=cnn"
        - "--kernel-size=7"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
        volumeMounts:
        - name: data-volume
          mountPath: /data
        - name: output-volume
          mountPath: /output
      restartPolicy: Never
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: training-data-pvc
      - name: output-volume
        persistentVolumeClaim:
          claimName: training-output-pvc
  backoffLimit: 2
```

### Vertex AI Training

```python
# vertex_training.py
from google.cloud import aiplatform

aiplatform.init(project="your-project-id", location="us-central1")

job = aiplatform.CustomContainerTrainingJob(
    display_name="metapathpredict-training",
    container_uri="us-central1-docker.pkg.dev/PROJECT_ID/metapathpredict/training:latest",
    model_serving_container_image_uri="us-central1-docker.pkg.dev/PROJECT_ID/metapathpredict/inference:latest",
)

model = job.run(
    replica_count=1,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    args=[
        "--data-dir=gs://PROJECT_ID-metapathpredict-data/unified",
        "--output-dir=gs://PROJECT_ID-metapathpredict-models/",
        "--epochs=100",
    ],
)
```

## Cloud Functions

### Trigger on Upload

```python
# cloud_function/main.py
import functions_framework
from google.cloud import pubsub_v1, storage

@functions_framework.cloud_event
def trigger_inference(cloud_event):
    """Triggered by file upload to input bucket."""
    data = cloud_event.data
    bucket = data["bucket"]
    name = data["name"]
    
    if name.endswith(".fasta"):
        # Publish to Pub/Sub for processing
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path("PROJECT_ID", "inference-requests")
        
        message = {
            "bucket": bucket,
            "file": name,
        }
        
        publisher.publish(topic_path, json.dumps(message).encode())
        print(f"Triggered inference for {name}")
```

### Deploy Function

```bash
gcloud functions deploy trigger-inference \
    --gen2 \
    --runtime=python311 \
    --region=us-central1 \
    --source=cloud_function/ \
    --entry-point=trigger_inference \
    --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
    --trigger-event-filters="bucket=PROJECT_ID-metapathpredict-data"
```

## Cost Optimization

### Preemptible VMs for Training

```bash
gcloud container node-pools create preemptible-gpu-pool \
    --cluster=metapathpredict-cluster \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --preemptible \
    --num-nodes=0 \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=10
```

### Committed Use Discounts

```bash
# Purchase 1-year commitment for inference
gcloud compute commitments create metapathpredict-commitment \
    --region=us-central1 \
    --resources=vcpu=8,memory=32GB \
    --plan=12-month
```

### Budget Alerts

```bash
gcloud billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="MetaPathPredict Budget" \
    --budget-amount=1000USD \
    --threshold-rule=percent=50,basis=current-spend \
    --threshold-rule=percent=90,basis=current-spend
```

## Monitoring

### Cloud Monitoring Dashboard

```bash
# Create custom dashboard
gcloud monitoring dashboards create --config-from-file=dashboard.json
```

### Alerts

```yaml
# alert-policy.yaml
displayName: "High GPU Utilization"
conditions:
- displayName: "GPU > 90%"
  conditionThreshold:
    filter: metric.type="kubernetes.io/container/accelerator/duty_cycle"
    comparison: COMPARISON_GT
    thresholdValue: 90
    duration: "300s"
notificationChannels:
- projects/PROJECT_ID/notificationChannels/CHANNEL_ID
```

## Security

### Workload Identity

```bash
# Create service account
gcloud iam service-accounts create metapathpredict-sa \
    --display-name="MetaPathPredict Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:metapathpredict-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

# Bind to Kubernetes service account
gcloud iam service-accounts add-iam-policy-binding \
    metapathpredict-sa@${PROJECT_ID}.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:${PROJECT_ID}.svc.id.goog[metapathpredict/default]"
```

See [Terraform](terraform.md) for Infrastructure as Code deployment.

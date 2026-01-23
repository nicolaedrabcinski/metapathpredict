# MetaPathPredict GCP Terraform

Terraform configuration for deploying MetaPathPredict ML infrastructure on Google Cloud Platform.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         GCP Project                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │   Orchestration VM  │    │   GPU Training VM   │            │
│  │     (e2-micro)      │    │  (n1-std-4 + T4)    │            │
│  │      [FREE]         │    │   [Preemptible]     │            │
│  │                     │    │                     │            │
│  │  - Dagster UI       │    │  - PyTorch 2.0      │            │
│  │  - DuckDB           │    │  - Model Training   │            │
│  │  - Monitoring       │    │  - Ray Workers      │            │
│  └─────────┬───────────┘    └──────────┬──────────┘            │
│            │                           │                        │
│            └─────────┬─────────────────┘                        │
│                      │                                          │
│  ┌───────────────────▼───────────────────┐                     │
│  │           Cloud Storage               │                     │
│  │  ┌──────────────┐ ┌──────────────┐   │                     │
│  │  │  Data Bucket │ │ Models Bucket│   │                     │
│  │  │  (FASTA/HDF5)│ │  (Weights)   │   │                     │
│  │  └──────────────┘ └──────────────┘   │                     │
│  └───────────────────────────────────────┘                     │
│                                                                  │
│  ┌───────────────────┐    ┌────────────────────┐               │
│  │  Artifact Registry│    │  Cloud Run         │               │
│  │  (Docker Images)  │    │  (Inference API)   │               │
│  └───────────────────┘    │   [Serverless]     │               │
│                           └────────────────────┘               │
│                                                                  │
│  [Optional] Cloud SQL PostgreSQL (db-f1-micro)                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Cost Estimation

| Component | Type | Monthly Cost |
|-----------|------|--------------|
| Orchestration VM | e2-micro | **FREE** (Always Free) |
| GPU Training | n1-standard-4 + T4 (Preemptible) | ~$6-10 (60 hrs) |
| Cloud Storage | 10 GB | ~$0.20 |
| Artifact Registry | 5 GB | ~$0.50 |
| Data Transfer | 50 GB out | ~$4.00 |
| **Total** | | **~$15-20/month** |

With Cloud SQL add ~$8/month.

## Prerequisites

1. [GCP Account](https://cloud.google.com/) with billing enabled
2. [Terraform](https://www.terraform.io/downloads) >= 1.5.0
3. [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and configured

## Quick Start

### 1. Authenticate with GCP

```bash
gcloud auth login
gcloud auth application-default login
```

### 2. Create/Select Project

```bash
# Create new project
gcloud projects create metapathpredict-dev --name="MetaPathPredict Dev"

# Or use existing
gcloud config set project YOUR_PROJECT_ID
```

### 3. Enable Billing

Enable billing for your project in the [GCP Console](https://console.cloud.google.com/billing).

### 4. Configure Terraform

```bash
cd terraform/gcp

# Copy example config
cp terraform.tfvars.example terraform.tfvars

# Edit with your values
nano terraform.tfvars
```

**Important:** Change at minimum:
- `project_id` - Your GCP project ID
- `allowed_ssh_ips` - Your IP address for SSH access

### 5. Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Apply changes
terraform apply
```

### 6. Access Resources

```bash
# SSH to orchestration VM
gcloud compute ssh metapathpredict-orchestration --zone=us-central1-a

# View outputs
terraform output

# Get Dagster UI URL
terraform output dagster_ui_url
```

## Usage Scenarios

### Scenario 1: Development (Cheapest)

```hcl
# terraform.tfvars
create_gpu_vm    = false  # Use Kaggle/Colab for training
create_cloud_sql = false  # Use DuckDB
create_cloud_run = false  # Test locally
```

Cost: **~$0-5/month** (mostly free tier)

### Scenario 2: Training Phase

```hcl
# terraform.tfvars
create_gpu_vm   = true
use_preemptible = true  # 91% cheaper!
```

```bash
# Deploy GPU VM
terraform apply

# SSH and train
gcloud compute ssh metapathpredict-gpu-training

# After training, destroy GPU VM to save costs
terraform apply -var="create_gpu_vm=false"
```

Cost: **~$20-30/month** during training

### Scenario 3: Production

```hcl
# terraform.tfvars
environment      = "production"
create_cloud_sql = true
create_cloud_run = true
use_preemptible  = false  # More stable
```

Cost: **~$50-100/month**

## Managing GPU VM Lifecycle

The GPU VM is the most expensive component. Start it only when training:

```bash
# Start GPU VM for training
terraform apply -var="create_gpu_vm=true"

# After training, stop to save costs
terraform apply -var="create_gpu_vm=false"
```

Or use gcloud directly:

```bash
# Stop VM (keeps disk)
gcloud compute instances stop metapathpredict-gpu-training --zone=us-central1-a

# Start VM
gcloud compute instances start metapathpredict-gpu-training --zone=us-central1-a
```

## Uploading Data

```bash
# Upload training data
gsutil -m cp -r ./data/datasets/* gs://YOUR_PROJECT-metapathpredict-data/datasets/

# Upload models
gsutil -m cp -r ./data/weights/* gs://YOUR_PROJECT-metapathpredict-models/
```

## Docker Images

```bash
# Configure Docker for Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build and push
docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT/metapathpredict-docker/inference:latest -f docker/Dockerfile.app .
docker push us-central1-docker.pkg.dev/YOUR_PROJECT/metapathpredict-docker/inference:latest
```

## Cleanup

```bash
# Destroy all resources
terraform destroy

# Or just stop expensive resources
terraform apply -var="create_gpu_vm=false" -var="create_cloud_sql=false"
```

## Troubleshooting

### GPU Quota

If you get quota errors for GPUs:
1. Go to [IAM & Admin > Quotas](https://console.cloud.google.com/iam-admin/quotas)
2. Filter for "NVIDIA T4 GPUs"
3. Request quota increase (usually approved within minutes)

### Preemptible VM Termination

Preemptible VMs can be terminated anytime. For long training:
- Use checkpointing in your training code
- Consider on-demand VMs for critical runs

### SSH Connection Issues

```bash
# Reset SSH keys
gcloud compute os-login ssh-keys remove --all

# Use IAP tunnel if direct SSH blocked
gcloud compute ssh INSTANCE_NAME --tunnel-through-iap
```

## File Structure

```
terraform/gcp/
├── main.tf                    # Main infrastructure
├── variables.tf               # Input variables
├── outputs.tf                 # Output values
├── terraform.tfvars.example   # Example configuration
└── README.md                  # This file
```

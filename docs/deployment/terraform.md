# Terraform Deployment

Infrastructure as Code for MetaPathPredict on GCP.

## Project Structure

```
terraform/
├── main.tf
├── variables.tf
├── outputs.tf
├── versions.tf
├── modules/
│   ├── gke/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── storage/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── networking/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   │   └── ...
│   └── prod/
│       └── ...
└── scripts/
    └── init.sh
```

## Quick Start

```bash
cd terraform/environments/dev
terraform init
terraform plan
terraform apply
```

## Root Configuration

### main.tf

```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.5.0"
  
  backend "gcs" {
    bucket = "your-project-terraform-state"
    prefix = "metapathpredict"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "services" {
  for_each = toset([
    "container.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudfunctions.googleapis.com",
    "secretmanager.googleapis.com",
    "monitoring.googleapis.com",
  ])
  
  project = var.project_id
  service = each.value
  
  disable_on_destroy = false
}

# Networking
module "networking" {
  source = "./modules/networking"
  
  project_id   = var.project_id
  region       = var.region
  network_name = "${var.project_name}-vpc"
}

# Storage
module "storage" {
  source = "./modules/storage"
  
  project_id   = var.project_id
  region       = var.region
  project_name = var.project_name
}

# GKE Cluster
module "gke" {
  source = "./modules/gke"
  
  project_id     = var.project_id
  region         = var.region
  zone           = var.zone
  network        = module.networking.network_name
  subnetwork     = module.networking.subnetwork_name
  cluster_name   = "${var.project_name}-cluster"
  
  depends_on = [google_project_service.services]
}

# Artifact Registry
resource "google_artifact_registry_repository" "metapathpredict" {
  location      = var.region
  repository_id = "metapathpredict"
  format        = "DOCKER"
  
  labels = var.labels
}
```

### variables.tf

```hcl
# terraform/variables.tf
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "metapathpredict"
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "dev"
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default = {
    project     = "metapathpredict"
    managed_by  = "terraform"
  }
}
```

### outputs.tf

```hcl
# terraform/outputs.tf
output "gke_cluster_name" {
  description = "GKE cluster name"
  value       = module.gke.cluster_name
}

output "gke_cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = module.gke.cluster_endpoint
  sensitive   = true
}

output "artifact_registry_url" {
  description = "Artifact Registry URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.metapathpredict.repository_id}"
}

output "data_bucket" {
  description = "Data storage bucket"
  value       = module.storage.data_bucket
}

output "models_bucket" {
  description = "Models storage bucket"
  value       = module.storage.models_bucket
}
```

## GKE Module

### modules/gke/main.tf

```hcl
# terraform/modules/gke/main.tf
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.zone
  
  # We'll manage node pools separately
  remove_default_node_pool = true
  initial_node_count       = 1
  
  network    = var.network
  subnetwork = var.subnetwork
  
  # Enable Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  # Addons
  addons_config {
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
    http_load_balancing {
      disabled = false
    }
  }
  
  # Logging
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }
  
  # Monitoring
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = true
    }
  }
}

# CPU Node Pool
resource "google_container_node_pool" "cpu_nodes" {
  name       = "cpu-pool"
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = var.cpu_node_count
  
  autoscaling {
    min_node_count = var.cpu_min_nodes
    max_node_count = var.cpu_max_nodes
  }
  
  node_config {
    machine_type = var.cpu_machine_type
    disk_size_gb = 100
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    labels = {
      pool = "cpu"
    }
    
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# GPU Node Pool
resource "google_container_node_pool" "gpu_nodes" {
  name     = "gpu-pool"
  location = var.zone
  cluster  = google_container_cluster.primary.name
  
  autoscaling {
    min_node_count = var.gpu_min_nodes
    max_node_count = var.gpu_max_nodes
  }
  
  node_config {
    machine_type = var.gpu_machine_type
    disk_size_gb = 200
    
    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    labels = {
      pool = "gpu"
    }
    
    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }
    
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# Preemptible GPU Node Pool for Training
resource "google_container_node_pool" "preemptible_gpu" {
  name     = "preemptible-gpu-pool"
  location = var.zone
  cluster  = google_container_cluster.primary.name
  
  autoscaling {
    min_node_count = 0
    max_node_count = var.preemptible_gpu_max_nodes
  }
  
  node_config {
    machine_type = var.gpu_machine_type
    preemptible  = true
    disk_size_gb = 200
    
    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    labels = {
      pool        = "preemptible-gpu"
      preemptible = "true"
    }
    
    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }
    
    taint {
      key    = "cloud.google.com/gke-preemptible"
      value  = "true"
      effect = "NO_SCHEDULE"
    }
  }
}
```

### modules/gke/variables.tf

```hcl
# terraform/modules/gke/variables.tf
variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "zone" {
  type = string
}

variable "network" {
  type = string
}

variable "subnetwork" {
  type = string
}

variable "cluster_name" {
  type = string
}

variable "cpu_machine_type" {
  type    = string
  default = "n1-standard-4"
}

variable "cpu_node_count" {
  type    = number
  default = 1
}

variable "cpu_min_nodes" {
  type    = number
  default = 1
}

variable "cpu_max_nodes" {
  type    = number
  default = 5
}

variable "gpu_machine_type" {
  type    = string
  default = "n1-standard-8"
}

variable "gpu_type" {
  type    = string
  default = "nvidia-tesla-t4"
}

variable "gpu_count" {
  type    = number
  default = 1
}

variable "gpu_min_nodes" {
  type    = number
  default = 0
}

variable "gpu_max_nodes" {
  type    = number
  default = 5
}

variable "preemptible_gpu_max_nodes" {
  type    = number
  default = 10
}
```

## Storage Module

### modules/storage/main.tf

```hcl
# terraform/modules/storage/main.tf
resource "google_storage_bucket" "data" {
  name     = "${var.project_id}-${var.project_name}-data"
  location = var.region
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  labels = var.labels
}

resource "google_storage_bucket" "models" {
  name     = "${var.project_id}-${var.project_name}-models"
  location = var.region
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  labels = var.labels
}

resource "google_storage_bucket" "results" {
  name     = "${var.project_id}-${var.project_name}-results"
  location = var.region
  
  uniform_bucket_level_access = true
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
  
  labels = var.labels
}

resource "google_storage_bucket" "mlflow" {
  name     = "${var.project_id}-${var.project_name}-mlflow"
  location = var.region
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  labels = var.labels
}
```

## Environment Configurations

### environments/dev/terraform.tfvars

```hcl
# terraform/environments/dev/terraform.tfvars
project_id   = "your-dev-project"
environment  = "dev"
region       = "us-central1"
zone         = "us-central1-a"

cpu_node_count = 1
cpu_max_nodes  = 3

gpu_min_nodes = 0
gpu_max_nodes = 2

preemptible_gpu_max_nodes = 5

labels = {
  project     = "metapathpredict"
  environment = "dev"
  managed_by  = "terraform"
}
```

### environments/prod/terraform.tfvars

```hcl
# terraform/environments/prod/terraform.tfvars
project_id   = "your-prod-project"
environment  = "prod"
region       = "us-central1"
zone         = "us-central1-a"

cpu_node_count = 2
cpu_max_nodes  = 10

gpu_min_nodes = 1
gpu_max_nodes = 10

preemptible_gpu_max_nodes = 20

labels = {
  project     = "metapathpredict"
  environment = "prod"
  managed_by  = "terraform"
}
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/terraform.yml
name: Terraform

on:
  push:
    branches: [main]
    paths: ['terraform/**']
  pull_request:
    paths: ['terraform/**']

jobs:
  terraform:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - uses: hashicorp/setup-terraform@v3
      with:
        terraform_version: 1.5.0
    
    - name: Authenticate to GCP
      uses: google-github-actions/auth@v2
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}
    
    - name: Terraform Init
      working-directory: terraform/environments/dev
      run: terraform init
    
    - name: Terraform Plan
      working-directory: terraform/environments/dev
      run: terraform plan -out=tfplan
    
    - name: Terraform Apply
      if: github.ref == 'refs/heads/main' && github.event_name == 'push'
      working-directory: terraform/environments/dev
      run: terraform apply -auto-approve tfplan
```

## Commands Reference

```bash
# Initialize Terraform
terraform init

# Validate configuration
terraform validate

# Format code
terraform fmt -recursive

# Plan changes
terraform plan -out=tfplan

# Apply changes
terraform apply tfplan

# Destroy infrastructure
terraform destroy

# Import existing resource
terraform import google_container_cluster.primary projects/PROJECT/locations/ZONE/clusters/CLUSTER

# State management
terraform state list
terraform state show google_container_cluster.primary
```

## Best Practices

1. **State Management**: Use remote state with GCS backend
2. **Workspaces**: Use separate workspaces or directories for environments
3. **Modules**: Reuse modules across environments
4. **Variables**: Use `.tfvars` files for environment-specific values
5. **Secrets**: Use Secret Manager, never commit secrets
6. **Locking**: Enable state locking with GCS
7. **Versioning**: Pin provider and Terraform versions

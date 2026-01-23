# =============================================================================
# Variables for MetaPathPredict GCP Infrastructure
# =============================================================================

# -----------------------------------------------------------------------------
# Project Configuration
# -----------------------------------------------------------------------------

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "metapathpredict"
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for compute resources"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment (development, staging, production)"
  type        = string
  default     = "development"

  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be one of: development, staging, production."
  }
}

variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default = {
    project     = "metapathpredict"
    managed_by  = "terraform"
    environment = "development"
  }
}

# -----------------------------------------------------------------------------
# Networking
# -----------------------------------------------------------------------------

variable "allowed_ssh_ips" {
  description = "List of IP addresses allowed to SSH (CIDR format)"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict in production!
}

# -----------------------------------------------------------------------------
# Orchestration VM (Dagster, DuckDB, Monitoring)
# -----------------------------------------------------------------------------

variable "orchestration_machine_type" {
  description = "Machine type for orchestration VM (e2-micro is Always Free)"
  type        = string
  default     = "e2-micro"
}

# -----------------------------------------------------------------------------
# GPU Training VM
# -----------------------------------------------------------------------------

variable "create_gpu_vm" {
  description = "Whether to create GPU VM for training"
  type        = bool
  default     = false  # Set to true when needed
}

variable "gpu_machine_type" {
  description = "Machine type for GPU training VM"
  type        = string
  default     = "n1-standard-4"
}

variable "gpu_type" {
  description = "GPU type (nvidia-tesla-t4 is cheapest)"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "use_preemptible" {
  description = "Use preemptible/spot VMs (up to 91% cheaper)"
  type        = bool
  default     = true
}

# -----------------------------------------------------------------------------
# Cloud SQL PostgreSQL
# -----------------------------------------------------------------------------

variable "create_cloud_sql" {
  description = "Whether to create Cloud SQL instance"
  type        = bool
  default     = false  # Use DuckDB on VM for cheaper option
}

variable "sql_tier" {
  description = "Cloud SQL machine tier"
  type        = string
  default     = "db-f1-micro"  # Cheapest option
}

variable "db_password" {
  description = "PostgreSQL database password"
  type        = string
  sensitive   = true
  default     = ""
}

# -----------------------------------------------------------------------------
# Cloud Run (Inference API)
# -----------------------------------------------------------------------------

variable "create_cloud_run" {
  description = "Whether to create Cloud Run service for inference"
  type        = bool
  default     = false
}

variable "inference_public" {
  description = "Allow unauthenticated access to inference API"
  type        = bool
  default     = false
}

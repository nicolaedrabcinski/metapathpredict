# =============================================================================
# MetaPathPredict - GCP Infrastructure
# Terraform configuration for deploying ML training and inference stack
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }

  # Uncomment for remote state storage
  # backend "gcs" {
  #   bucket = "metapathpredict-terraform-state"
  #   prefix = "terraform/state"
  # }
}

# =============================================================================
# Provider Configuration
# =============================================================================

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# =============================================================================
# Enable Required APIs
# =============================================================================

resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "sqladmin.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "secretmanager.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

# =============================================================================
# Networking
# =============================================================================

resource "google_compute_network" "vpc" {
  name                    = "${var.project_name}-vpc"
  auto_create_subnetworks = false

  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "main" {
  name          = "${var.project_name}-subnet"
  ip_cidr_range = "10.0.0.0/24"
  network       = google_compute_network.vpc.id
  region        = var.region

  private_ip_google_access = true
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "${var.project_name}-allow-ssh"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = var.allowed_ssh_ips
  target_tags   = ["ssh-enabled"]
}

resource "google_compute_firewall" "allow_internal" {
  name    = "${var.project_name}-allow-internal"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = ["10.0.0.0/24"]
}

resource "google_compute_firewall" "allow_dagster" {
  name    = "${var.project_name}-allow-dagster"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["3000"]
  }

  source_ranges = var.allowed_ssh_ips
  target_tags   = ["dagster"]
}

# =============================================================================
# Cloud Storage - Data & Models
# =============================================================================

resource "google_storage_bucket" "data" {
  name          = "${var.project_id}-${var.project_name}-data"
  location      = var.region
  force_destroy = var.environment != "production"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  labels = var.labels

  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket" "models" {
  name          = "${var.project_id}-${var.project_name}-models"
  location      = var.region
  force_destroy = var.environment != "production"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  labels = var.labels

  depends_on = [google_project_service.apis]
}

# =============================================================================
# Artifact Registry - Docker Images
# =============================================================================

resource "google_artifact_registry_repository" "docker" {
  location      = var.region
  repository_id = "${var.project_name}-docker"
  format        = "DOCKER"

  labels = var.labels

  depends_on = [google_project_service.apis]
}

# =============================================================================
# Service Account
# =============================================================================

resource "google_service_account" "ml_training" {
  account_id   = "${var.project_name}-ml-training"
  display_name = "MetaPathPredict ML Training Service Account"

  depends_on = [google_project_service.apis]
}

resource "google_project_iam_member" "ml_training_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.ml_training.email}"
}

resource "google_project_iam_member" "ml_training_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.ml_training.email}"
}

resource "google_project_iam_member" "ml_training_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.ml_training.email}"
}

# =============================================================================
# Orchestration VM (Always Free e2-micro)
# =============================================================================

resource "google_compute_instance" "orchestration" {
  name         = "${var.project_name}-orchestration"
  machine_type = var.orchestration_machine_type
  zone         = var.zone

  tags = ["ssh-enabled", "dagster"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 30
      type  = "pd-standard"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.main.id

    access_config {
      # Ephemeral public IP
    }
  }

  service_account {
    email  = google_service_account.ml_training.email
    scopes = ["cloud-platform"]
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    set -e

    # Update system
    apt-get update && apt-get upgrade -y

    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker ubuntu

    # Install Docker Compose
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose

    # Install Python
    apt-get install -y python3-pip python3-venv

    # Install gcloud CLI
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
    apt-get update && apt-get install -y google-cloud-cli

    # Create working directory
    mkdir -p /opt/metapathpredict
    chown ubuntu:ubuntu /opt/metapathpredict

    echo "Orchestration VM setup complete!"
  EOF

  labels = var.labels

  # Allow stopping for updates
  allow_stopping_for_update = true

  depends_on = [
    google_compute_subnetwork.main,
    google_service_account.ml_training,
  ]
}

# =============================================================================
# GPU Training VM (Preemptible for cost savings)
# =============================================================================

resource "google_compute_instance" "gpu_training" {
  count = var.create_gpu_vm ? 1 : 0

  name         = "${var.project_name}-gpu-training"
  machine_type = var.gpu_machine_type
  zone         = var.zone

  tags = ["ssh-enabled"]

  scheduling {
    preemptible                 = var.use_preemptible
    automatic_restart           = false
    on_host_maintenance         = "TERMINATE"
    provisioning_model          = var.use_preemptible ? "SPOT" : "STANDARD"
    instance_termination_action = var.use_preemptible ? "STOP" : null
  }

  guest_accelerator {
    type  = var.gpu_type
    count = 1
  }

  boot_disk {
    initialize_params {
      image = "deeplearning-platform-release/pytorch-latest-gpu"
      size  = 100
      type  = "pd-ssd"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.main.id

    access_config {
      # Ephemeral public IP
    }
  }

  service_account {
    email  = google_service_account.ml_training.email
    scopes = ["cloud-platform"]
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    set -e

    # GPU drivers are pre-installed on Deep Learning VM
    nvidia-smi

    # Install additional dependencies
    pip install --upgrade pip
    pip install duckdb ray[default] dagster

    # Pull latest code from GCS
    gsutil -m cp -r gs://${google_storage_bucket.data.name}/code/* /opt/ml/ || true

    echo "GPU Training VM setup complete!"
  EOF

  labels = var.labels

  depends_on = [
    google_compute_subnetwork.main,
    google_service_account.ml_training,
    google_storage_bucket.data,
  ]
}

# =============================================================================
# Cloud SQL PostgreSQL (Optional - for DuckLake metadata)
# =============================================================================

resource "google_sql_database_instance" "postgres" {
  count = var.create_cloud_sql ? 1 : 0

  name             = "${var.project_name}-postgres"
  database_version = "POSTGRES_15"
  region           = var.region

  deletion_protection = var.environment == "production"

  settings {
    tier              = var.sql_tier
    availability_type = var.environment == "production" ? "REGIONAL" : "ZONAL"
    disk_size         = 10
    disk_type         = "PD_SSD"

    ip_configuration {
      ipv4_enabled    = true
      private_network = google_compute_network.vpc.id

      authorized_networks {
        name  = "orchestration-vm"
        value = google_compute_instance.orchestration.network_interface[0].access_config[0].nat_ip
      }
    }

    backup_configuration {
      enabled            = var.environment == "production"
      start_time         = "03:00"
      binary_log_enabled = false

      backup_retention_settings {
        retained_backups = 7
      }
    }

    maintenance_window {
      day  = 7
      hour = 3
    }

    user_labels = var.labels
  }

  depends_on = [
    google_project_service.apis,
    google_compute_network.vpc,
  ]
}

resource "google_sql_database" "metapathpredict" {
  count = var.create_cloud_sql ? 1 : 0

  name     = "metapathpredict"
  instance = google_sql_database_instance.postgres[0].name
}

resource "google_sql_user" "app" {
  count = var.create_cloud_sql ? 1 : 0

  name     = "metapathpredict"
  instance = google_sql_database_instance.postgres[0].name
  password = var.db_password
}

# =============================================================================
# Cloud Run - Inference API (Serverless)
# =============================================================================

resource "google_cloud_run_v2_service" "inference" {
  count = var.create_cloud_run ? 1 : 0

  name     = "${var.project_name}-inference"
  location = var.region

  template {
    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}/inference:latest"

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
      }

      env {
        name  = "MODEL_BUCKET"
        value = google_storage_bucket.models.name
      }

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 10
        period_seconds        = 3
        failure_threshold     = 3
      }
    }

    service_account = google_service_account.ml_training.email
  }

  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }

  labels = var.labels

  depends_on = [
    google_artifact_registry_repository.docker,
    google_storage_bucket.models,
  ]
}

# Allow unauthenticated access to inference API (optional)
resource "google_cloud_run_v2_service_iam_member" "inference_public" {
  count = var.create_cloud_run && var.inference_public ? 1 : 0

  location = var.region
  name     = google_cloud_run_v2_service.inference[0].name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# =============================================================================
# Secret Manager (for sensitive configuration)
# =============================================================================

resource "google_secret_manager_secret" "db_password" {
  count = var.create_cloud_sql ? 1 : 0

  secret_id = "${var.project_name}-db-password"

  replication {
    auto {}
  }

  labels = var.labels

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "db_password" {
  count = var.create_cloud_sql ? 1 : 0

  secret      = google_secret_manager_secret.db_password[0].id
  secret_data = var.db_password
}

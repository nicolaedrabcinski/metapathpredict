# =============================================================================
# Outputs for MetaPathPredict GCP Infrastructure
# =============================================================================

# -----------------------------------------------------------------------------
# Networking
# -----------------------------------------------------------------------------

output "vpc_id" {
  description = "VPC Network ID"
  value       = google_compute_network.vpc.id
}

output "subnet_id" {
  description = "Subnet ID"
  value       = google_compute_subnetwork.main.id
}

# -----------------------------------------------------------------------------
# Storage
# -----------------------------------------------------------------------------

output "data_bucket_name" {
  description = "Cloud Storage bucket for data"
  value       = google_storage_bucket.data.name
}

output "data_bucket_url" {
  description = "Cloud Storage bucket URL for data"
  value       = google_storage_bucket.data.url
}

output "models_bucket_name" {
  description = "Cloud Storage bucket for models"
  value       = google_storage_bucket.models.name
}

output "models_bucket_url" {
  description = "Cloud Storage bucket URL for models"
  value       = google_storage_bucket.models.url
}

# -----------------------------------------------------------------------------
# Artifact Registry
# -----------------------------------------------------------------------------

output "docker_registry" {
  description = "Docker registry URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}

# -----------------------------------------------------------------------------
# Orchestration VM
# -----------------------------------------------------------------------------

output "orchestration_vm_name" {
  description = "Orchestration VM name"
  value       = google_compute_instance.orchestration.name
}

output "orchestration_vm_ip" {
  description = "Orchestration VM external IP"
  value       = google_compute_instance.orchestration.network_interface[0].access_config[0].nat_ip
}

output "orchestration_vm_internal_ip" {
  description = "Orchestration VM internal IP"
  value       = google_compute_instance.orchestration.network_interface[0].network_ip
}

output "dagster_ui_url" {
  description = "Dagster UI URL"
  value       = "http://${google_compute_instance.orchestration.network_interface[0].access_config[0].nat_ip}:3000"
}

# -----------------------------------------------------------------------------
# GPU Training VM
# -----------------------------------------------------------------------------

output "gpu_vm_name" {
  description = "GPU Training VM name"
  value       = var.create_gpu_vm ? google_compute_instance.gpu_training[0].name : null
}

output "gpu_vm_ip" {
  description = "GPU Training VM external IP"
  value       = var.create_gpu_vm ? google_compute_instance.gpu_training[0].network_interface[0].access_config[0].nat_ip : null
}

# -----------------------------------------------------------------------------
# Cloud SQL
# -----------------------------------------------------------------------------

output "postgres_connection_name" {
  description = "Cloud SQL connection name"
  value       = var.create_cloud_sql ? google_sql_database_instance.postgres[0].connection_name : null
}

output "postgres_ip" {
  description = "Cloud SQL public IP"
  value       = var.create_cloud_sql ? google_sql_database_instance.postgres[0].public_ip_address : null
}

output "postgres_connection_string" {
  description = "PostgreSQL connection string"
  value       = var.create_cloud_sql ? "postgresql://metapathpredict:****@${google_sql_database_instance.postgres[0].public_ip_address}:5432/metapathpredict" : null
  sensitive   = true
}

# -----------------------------------------------------------------------------
# Cloud Run
# -----------------------------------------------------------------------------

output "inference_url" {
  description = "Cloud Run inference service URL"
  value       = var.create_cloud_run ? google_cloud_run_v2_service.inference[0].uri : null
}

# -----------------------------------------------------------------------------
# Service Account
# -----------------------------------------------------------------------------

output "service_account_email" {
  description = "ML Training service account email"
  value       = google_service_account.ml_training.email
}

# -----------------------------------------------------------------------------
# SSH Commands
# -----------------------------------------------------------------------------

output "ssh_orchestration" {
  description = "SSH command for orchestration VM"
  value       = "gcloud compute ssh ${google_compute_instance.orchestration.name} --zone=${var.zone}"
}

output "ssh_gpu" {
  description = "SSH command for GPU VM"
  value       = var.create_gpu_vm ? "gcloud compute ssh ${google_compute_instance.gpu_training[0].name} --zone=${var.zone}" : null
}

# -----------------------------------------------------------------------------
# Useful Commands
# -----------------------------------------------------------------------------

output "upload_data_command" {
  description = "Command to upload data to GCS"
  value       = "gsutil -m cp -r ./data/* gs://${google_storage_bucket.data.name}/"
}

output "upload_models_command" {
  description = "Command to upload models to GCS"
  value       = "gsutil -m cp -r ./models/* gs://${google_storage_bucket.models.name}/"
}

output "docker_push_command" {
  description = "Command to push Docker image"
  value       = "docker push ${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}/IMAGE:TAG"
}

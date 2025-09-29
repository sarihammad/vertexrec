# VertexRec Terraform Variables
# Configuration variables for infrastructure deployment

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone for resources"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "bucket_name" {
  description = "Name of the GCS bucket for data storage"
  type        = string
}

variable "dataset_id" {
  description = "BigQuery dataset ID"
  type        = string
  default     = "vertexrec_dataset"
}

variable "feature_store_id" {
  description = "Vertex AI Feature Store ID"
  type        = string
  default     = "vertexrec-feature-store"
}

variable "model_name" {
  description = "Name for the ML model"
  type        = string
  default     = "vertexrec-model"
}

variable "endpoint_name" {
  description = "Name for the Vertex AI endpoint"
  type        = string
  default     = "vertexrec-endpoint"
}

variable "cloud_run_service_name" {
  description = "Name for the Cloud Run service"
  type        = string
  default     = "vertexrec-api"
}

variable "cloud_run_image" {
  description = "Container image for Cloud Run service"
  type        = string
  default     = "gcr.io/PROJECT_ID/vertexrec-api:latest"
}

variable "machine_type" {
  description = "Machine type for compute resources"
  type        = string
  default     = "n1-standard-4"
}

variable "min_instances" {
  description = "Minimum number of Cloud Run instances"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of Cloud Run instances"
  type        = number
  default     = 10
}

variable "cpu_limit" {
  description = "CPU limit for Cloud Run service"
  type        = string
  default     = "2"
}

variable "memory_limit" {
  description = "Memory limit for Cloud Run service"
  type        = string
  default     = "2Gi"
}

variable "enable_monitoring" {
  description = "Enable monitoring and alerting"
  type        = bool
  default     = true
}

variable "alert_email" {
  description = "Email for monitoring alerts"
  type        = string
  default     = ""
}

variable "enable_cdn" {
  description = "Enable Cloud CDN for API"
  type        = bool
  default     = false
}

variable "enable_ssl" {
  description = "Enable SSL for custom domain"
  type        = bool
  default     = false
}

variable "custom_domain" {
  description = "Custom domain for the API"
  type        = string
  default     = ""
}

variable "vpc_name" {
  description = "Name of the VPC network"
  type        = string
  default     = "vertexrec-vpc"
}

variable "subnet_name" {
  description = "Name of the subnet"
  type        = string
  default     = "vertexrec-subnet"
}

variable "ip_cidr_range" {
  description = "CIDR range for the subnet"
  type        = string
  default     = "10.0.0.0/24"
}

variable "enable_private_ip" {
  description = "Enable private IP for Cloud Run"
  type        = bool
  default     = false
}

variable "enable_vpc_connector" {
  description = "Enable VPC connector for Cloud Run"
  type        = bool
  default     = false
}

variable "service_account_email" {
  description = "Email of the service account for Cloud Run"
  type        = string
  default     = ""
}

variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default = {
    project     = "vertexrec"
    managed_by  = "terraform"
    environment = "dev"
  }
}

variable "retention_days" {
  description = "Data retention period in days"
  type        = number
  default     = 365
}

variable "backup_enabled" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "encryption_key" {
  description = "Customer-managed encryption key"
  type        = string
  default     = ""
}

variable "network_tags" {
  description = "Network tags for firewall rules"
  type        = list(string)
  default     = ["vertexrec", "ml-pipeline"]
}

variable "enable_apis" {
  description = "List of GCP APIs to enable"
  type        = list(string)
  default = [
    "aiplatform.googleapis.com",
    "bigquery.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudfunctions.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "cloudrun.googleapis.com",
    "compute.googleapis.com",
    "container.googleapis.com",
    "dataflow.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "pubsub.googleapis.com",
    "storage.googleapis.com",
    "vpcaccess.googleapis.com"
  ]
}

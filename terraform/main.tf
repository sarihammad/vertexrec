# VertexRec Infrastructure as Code
# Complete Google Cloud Platform setup for ML recommendation system

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
}

# Configure providers
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

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset(var.enable_apis)
  
  service                    = each.value
  disable_dependent_services = false
  disable_on_destroy         = false
}

# Create service account for Cloud Run
resource "google_service_account" "cloud_run_sa" {
  account_id   = "vertexrec-cloud-run"
  display_name = "VertexRec Cloud Run Service Account"
  description  = "Service account for VertexRec Cloud Run service"
}

# Create service account for Vertex AI
resource "google_service_account" "vertex_ai_sa" {
  account_id   = "vertexrec-vertex-ai"
  display_name = "VertexRec Vertex AI Service Account"
  description  = "Service account for VertexRec ML pipeline"
}

# Create service account for Dataflow
resource "google_service_account" "dataflow_sa" {
  account_id   = "vertexrec-dataflow"
  display_name = "VertexRec Dataflow Service Account"
  description  = "Service account for VertexRec data processing"
}

# IAM bindings for Cloud Run service account
resource "google_project_iam_member" "cloud_run_permissions" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/bigquery.dataViewer",
    "roles/bigquery.jobUser",
    "roles/storage.objectViewer",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/cloudtrace.agent"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# IAM bindings for Vertex AI service account
resource "google_project_iam_member" "vertex_ai_permissions" {
  for_each = toset([
    "roles/aiplatform.admin",
    "roles/bigquery.admin",
    "roles/storage.admin",
    "roles/dataflow.admin",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.vertex_ai_sa.email}"
}

# IAM bindings for Dataflow service account
resource "google_project_iam_member" "dataflow_permissions" {
  for_each = toset([
    "roles/dataflow.worker",
    "roles/storage.admin",
    "roles/bigquery.admin",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

# Create VPC network
resource "google_compute_network" "vpc_network" {
  name                    = var.vpc_name
  auto_create_subnetworks = false
  mtu                     = 1460
}

# Create subnet
resource "google_compute_subnetwork" "vpc_subnet" {
  name          = var.subnet_name
  ip_cidr_range = var.ip_cidr_range
  region        = var.region
  network       = google_compute_network.vpc_network.id
  
  private_ip_google_access = true
}

# Create VPC connector for Cloud Run (if enabled)
resource "google_vpc_access_connector" "connector" {
  count = var.enable_vpc_connector ? 1 : 0
  
  name          = "vertexrec-vpc-connector"
  ip_cidr_range = "10.8.0.0/28"
  network       = google_compute_network.vpc_network.name
  region        = var.region
  
  min_instances = 2
  max_instances = 3
  
  machine_type = "e2-micro"
}

# Create Cloud Storage bucket
resource "google_storage_bucket" "data_bucket" {
  name          = var.bucket_name
  location      = "US"
  force_destroy = true
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = var.retention_days
    }
    action {
      type = "Delete"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  labels = var.labels
}

# Create BigQuery dataset
resource "google_bigquery_dataset" "dataset" {
  dataset_id  = var.dataset_id
  location    = var.region
  description = "Dataset for VertexRec recommendation system"
  
  labels = var.labels
  
  # Configure table expiration
  default_table_expiration_ms = var.retention_days * 24 * 60 * 60 * 1000
  
  access {
    role          = "OWNER"
    user_by_email = google_service_account.vertex_ai_sa.email
  }
  
  access {
    role          = "READER"
    user_by_email = google_service_account.cloud_run_sa.email
  }
}

# Create BigQuery tables
resource "google_bigquery_table" "users_table" {
  dataset_id = google_bigquery_dataset.dataset.dataset_id
  table_id   = "users"
  
  description = "User profiles and demographics"
  
  schema = jsonencode([
    {
      name = "user_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "age_group"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "gender"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "location"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "signup_date"
      type = "DATE"
      mode = "NULLABLE"
    },
    {
      name = "preferred_genres"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "activity_level"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "subscription_type"
      type = "STRING"
      mode = "NULLABLE"
    }
  ])
  
  labels = var.labels
}

resource "google_bigquery_table" "items_table" {
  dataset_id = google_bigquery_dataset.dataset.dataset_id
  table_id   = "items"
  
  description = "Item catalog and metadata"
  
  schema = jsonencode([
    {
      name = "item_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "title"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "genre"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "release_date"
      type = "DATE"
      mode = "NULLABLE"
    },
    {
      name = "popularity_score"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "quality_score"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "price"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "duration_minutes"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "language"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "rating_avg"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "rating_count"
      type = "INTEGER"
      mode = "NULLABLE"
    }
  ])
  
  labels = var.labels
}

resource "google_bigquery_table" "interactions_table" {
  dataset_id = google_bigquery_dataset.dataset.dataset_id
  table_id   = "interactions"
  
  description = "User-item interaction history"
  
  schema = jsonencode([
    {
      name = "user_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "item_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "interaction_type"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "rating"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "NULLABLE"
    },
    {
      name = "session_id"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "device_type"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "duration_seconds"
      type = "INTEGER"
      mode = "NULLABLE"
    }
  ])
  
  labels = var.labels
}

# Create Vertex AI Feature Store
resource "google_vertex_ai_feature_store" "feature_store" {
  name   = var.feature_store_id
  region = var.region
  
  labels = var.labels
  
  depends_on = [google_project_service.apis]
}

# Create Feature Store entity types
resource "google_vertex_ai_feature_store_entity_type" "user_entity_type" {
  name     = "users"
  feature_store = google_vertex_ai_feature_store.feature_store.id
  
  description = "User entity type for recommendation features"
  
  labels = var.labels
}

resource "google_vertex_ai_feature_store_entity_type" "item_entity_type" {
  name     = "items"
  feature_store = google_vertex_ai_feature_store.feature_store.id
  
  description = "Item entity type for recommendation features"
  
  labels = var.labels
}

# Create Vertex AI endpoint
resource "google_vertex_ai_endpoint" "endpoint" {
  name         = var.endpoint_name
  display_name = "VertexRec Recommendation Endpoint"
  description  = "Endpoint for serving recommendation models"
  location     = var.region
  
  labels = var.labels
  
  depends_on = [google_project_service.apis]
}

# Create Cloud Run service
resource "google_cloud_run_v2_service" "api_service" {
  name     = var.cloud_run_service_name
  location = var.region
  
  template {
    labels = var.labels
    
    service_account = google_service_account.cloud_run_sa.email
    
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }
    
    containers {
      image = var.cloud_run_image
      
      resources {
        limits = {
          cpu    = var.cpu_limit
          memory = var.memory_limit
        }
      }
      
      ports {
        container_port = 8080
      }
      
      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }
      
      env {
        name  = "REGION"
        value = var.region
      }
      
      env {
        name  = "DATASET_ID"
        value = google_bigquery_dataset.dataset.dataset_id
      }
      
      env {
        name  = "ENDPOINT_NAME"
        value = google_vertex_ai_endpoint.endpoint.name
      }
      
      env {
        name  = "FEATURE_STORE_ID"
        value = google_vertex_ai_feature_store.feature_store.name
      }
      
      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 10
        timeout_seconds      = 5
        period_seconds       = 10
        failure_threshold    = 3
      }
      
      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 30
        timeout_seconds      = 5
        period_seconds       = 30
        failure_threshold    = 3
      }
    }
    
    vpc_access {
      connector = var.enable_vpc_connector ? google_vpc_access_connector.connector[0].id : null
      egress    = "PRIVATE_RANGES_ONLY"
    }
  }
  
  depends_on = [google_project_service.apis]
}

# Allow unauthenticated access to Cloud Run (for public API)
resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_v2_service.api_service.name
  location = google_cloud_run_v2_service.api_service.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Create monitoring notification channel (if email provided)
resource "google_monitoring_notification_channel" "email" {
  count = var.enable_monitoring && var.alert_email != "" ? 1 : 0
  
  display_name = "VertexRec Email Alerts"
  type         = "email"
  
  labels = {
    email_address = var.alert_email
  }
}

# Create alerting policy for API errors
resource "google_monitoring_alert_policy" "api_errors" {
  count = var.enable_monitoring ? 1 : 0
  
  display_name = "VertexRec API Error Rate"
  combiner     = "OR"
  
  conditions {
    display_name = "High error rate"
    
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${var.cloud_run_service_name}\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 0.05
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields = ["resource.label.service_name"]
      }
    }
  }
  
  notification_channels = var.alert_email != "" ? [google_monitoring_notification_channel.email[0].id] : []
  
  alert_strategy {
    auto_close = "1800s"
  }
  
  depends_on = [google_cloud_run_v2_service.api_service]
}

# Create alerting policy for high latency
resource "google_monitoring_alert_policy" "high_latency" {
  count = var.enable_monitoring ? 1 : 0
  
  display_name = "VertexRec High Latency"
  combiner     = "OR"
  
  conditions {
    display_name = "High latency"
    
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${var.cloud_run_service_name}\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 1000
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields = ["resource.label.service_name"]
      }
    }
  }
  
  notification_channels = var.alert_email != "" ? [google_monitoring_notification_channel.email[0].id] : []
  
  alert_strategy {
    auto_close = "1800s"
  }
  
  depends_on = [google_cloud_run_v2_service.api_service]
}

# VertexRec Terraform Outputs
# Output values for infrastructure resources

output "project_id" {
  description = "The GCP project ID"
  value       = var.project_id
}

output "region" {
  description = "The GCP region"
  value       = var.region
}

output "bucket_name" {
  description = "Name of the GCS bucket for data storage"
  value       = google_storage_bucket.data_bucket.name
}

output "bucket_url" {
  description = "URL of the GCS bucket"
  value       = google_storage_bucket.data_bucket.url
}

output "dataset_id" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.dataset.dataset_id
}

output "dataset_location" {
  description = "BigQuery dataset location"
  value       = google_bigquery_dataset.dataset.location
}

output "users_table_id" {
  description = "BigQuery users table ID"
  value       = google_bigquery_table.users_table.table_id
}

output "items_table_id" {
  description = "BigQuery items table ID"
  value       = google_bigquery_table.items_table.table_id
}

output "interactions_table_id" {
  description = "BigQuery interactions table ID"
  value       = google_bigquery_table.interactions_table.table_id
}

output "feature_store_id" {
  description = "Vertex AI Feature Store ID"
  value       = google_vertex_ai_feature_store.feature_store.name
}

output "feature_store_location" {
  description = "Vertex AI Feature Store location"
  value       = google_vertex_ai_feature_store.feature_store.region
}

output "user_entity_type_id" {
  description = "User entity type ID in Feature Store"
  value       = google_vertex_ai_feature_store_entity_type.user_entity_type.name
}

output "item_entity_type_id" {
  description = "Item entity type ID in Feature Store"
  value       = google_vertex_ai_feature_store_entity_type.item_entity_type.name
}

output "endpoint_id" {
  description = "Vertex AI endpoint ID"
  value       = google_vertex_ai_endpoint.endpoint.name
}

output "endpoint_location" {
  description = "Vertex AI endpoint location"
  value       = google_vertex_ai_endpoint.endpoint.location
}

output "cloud_run_service_name" {
  description = "Cloud Run service name"
  value       = google_cloud_run_v2_service.api_service.name
}

output "cloud_run_service_url" {
  description = "Cloud Run service URL"
  value       = google_cloud_run_v2_service.api_service.uri
}

output "cloud_run_service_location" {
  description = "Cloud Run service location"
  value       = google_cloud_run_v2_service.api_service.location
}

output "cloud_run_sa_email" {
  description = "Cloud Run service account email"
  value       = google_service_account.cloud_run_sa.email
}

output "vertex_ai_sa_email" {
  description = "Vertex AI service account email"
  value       = google_service_account.vertex_ai_sa.email
}

output "dataflow_sa_email" {
  description = "Dataflow service account email"
  value       = google_service_account.dataflow_sa.email
}

output "vpc_network_name" {
  description = "VPC network name"
  value       = google_compute_network.vpc_network.name
}

output "vpc_network_id" {
  description = "VPC network ID"
  value       = google_compute_network.vpc_network.id
}

output "subnet_name" {
  description = "Subnet name"
  value       = google_compute_subnetwork.vpc_subnet.name
}

output "subnet_id" {
  description = "Subnet ID"
  value       = google_compute_subnetwork.vpc_subnet.id
}

output "vpc_connector_id" {
  description = "VPC connector ID (if enabled)"
  value       = var.enable_vpc_connector ? google_vpc_access_connector.connector[0].id : null
}

output "monitoring_enabled" {
  description = "Whether monitoring is enabled"
  value       = var.enable_monitoring
}

output "alert_email" {
  description = "Email for monitoring alerts"
  value       = var.alert_email
}

output "api_endpoints" {
  description = "API endpoint URLs"
  value = {
    recommend = "${google_cloud_run_v2_service.api_service.uri}/recommend"
    health    = "${google_cloud_run_v2_service.api_service.uri}/health"
    docs      = "${google_cloud_run_v2_service.api_service.uri}/docs"
  }
}

output "bigquery_tables" {
  description = "BigQuery table references"
  value = {
    users        = "${var.project_id}.${google_bigquery_dataset.dataset.dataset_id}.${google_bigquery_table.users_table.table_id}"
    items        = "${var.project_id}.${google_bigquery_dataset.dataset.dataset_id}.${google_bigquery_table.items_table.table_id}"
    interactions = "${var.project_id}.${google_bigquery_dataset.dataset.dataset_id}.${google_bigquery_table.interactions_table.table_id}"
  }
}

output "feature_store_entity_types" {
  description = "Feature Store entity type references"
  value = {
    users = "${google_vertex_ai_feature_store.feature_store.name}/entityTypes/${google_vertex_ai_feature_store_entity_type.user_entity_type.name}"
    items = "${google_vertex_ai_feature_store.feature_store.name}/entityTypes/${google_vertex_ai_feature_store_entity_type.item_entity_type.name}"
  }
}

output "deployment_info" {
  description = "Deployment information"
  value = {
    environment = var.environment
    region      = var.region
    project_id  = var.project_id
    timestamp   = timestamp()
  }
}

# Output for easy copy-paste commands
output "quick_start_commands" {
  description = "Quick start commands for the deployment"
  value = <<-EOT
    # Test the API
    curl -X POST "${google_cloud_run_v2_service.api_service.uri}/recommend" \
      -H "Content-Type: application/json" \
      -d '{"user_id": "user_000001", "k": 10}'
    
    # Check API health
    curl "${google_cloud_run_v2_service.api_service.uri}/health"
    
    # View API documentation
    open "${google_cloud_run_v2_service.api_service.uri}/docs"
    
    # Query BigQuery
    bq query --use_legacy_sql=false "SELECT COUNT(*) FROM \`${var.project_id}.${google_bigquery_dataset.dataset.dataset_id}.${google_bigquery_table.users_table.table_id}\`"
  EOT
}

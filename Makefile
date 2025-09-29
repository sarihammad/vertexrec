# VertexRec Makefile
# Production-ready build automation for ML pipeline

.PHONY: help setup install clean test lint format deploy-infra deploy-pipeline deploy-api run-pipeline generate-data upload-data

# Default target
help: ## Show this help message
	@echo "VertexRec - End-to-End Recommendation System"
	@echo "============================================="
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment setup
setup: ## Set up development environment
	@echo "Setting up VertexRec development environment..."
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pre-commit install
	@echo "Environment setup complete!"

install: ## Install dependencies
	pip install -r requirements.txt

# Code quality
clean: ## Clean up temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

test: ## Run tests
	python -m pytest tests/ -v --cov=pipelines --cov=serving --cov-report=html

lint: ## Run linting
	flake8 pipelines/ serving/ scripts/ --max-line-length=88
	mypy pipelines/ serving/ scripts/ --ignore-missing-imports

format: ## Format code
	black pipelines/ serving/ scripts/ notebooks/ --line-length=88
	isort pipelines/ serving/ scripts/ notebooks/ --profile black

# Infrastructure
deploy-infra: ## Deploy infrastructure with Terraform
	@echo "Deploying infrastructure..."
	cd terraform && terraform init
	cd terraform && terraform plan -var="project_id=$(PROJECT_ID)" -var="region=$(REGION)"
	cd terraform && terraform apply -var="project_id=$(PROJECT_ID)" -var="region=$(REGION)" -auto-approve
	@echo "Infrastructure deployment complete!"

destroy-infra: ## Destroy infrastructure
	cd terraform && terraform destroy -var="project_id=$(PROJECT_ID)" -var="region=$(REGION)" -auto-approve

# Data pipeline
generate-data: ## Generate synthetic dataset
	@echo "Generating synthetic data..."
	python scripts/generate_synthetic_data.py \
		--output-dir data/ \
		--num-users 10000 \
		--num-items 5000 \
		--num-interactions 100000
	@echo "Data generation complete!"

upload-data: ## Upload data to GCS and BigQuery
	@echo "Uploading data to cloud..."
	gsutil -m cp data/*.csv gs://$(BUCKET_NAME)/data/
	python scripts/upload_to_bigquery.py \
		--project-id $(PROJECT_ID) \
		--bucket-name $(BUCKET_NAME)
	@echo "Data upload complete!"

# ML Pipeline
run-pipeline: ## Execute ML training pipeline
	@echo "Running ML pipeline..."
	python scripts/deploy_pipeline.py \
		--project-id $(PROJECT_ID) \
		--region $(REGION) \
		--pipeline-file pipelines/pipeline.py
	@echo "Pipeline execution complete!"

deploy-pipeline: ## Deploy pipeline to Vertex AI
	@echo "Deploying ML pipeline..."
	python scripts/deploy_pipeline.py \
		--project-id $(PROJECT_ID) \
		--region $(REGION) \
		--deploy-only

# Serving API
deploy-api: ## Deploy Cloud Run API
	@echo "Deploying Cloud Run API..."
	python scripts/deploy_cloud_run.py \
		--project-id $(PROJECT_ID) \
		--region $(REGION) \
		--service-name vertexrec-api
	@echo "API deployment complete!"

# Monitoring
setup-monitoring: ## Set up monitoring and alerting
	@echo "Setting up monitoring..."
	python serving/monitoring/setup_monitoring.py \
		--project-id $(PROJECT_ID) \
		--region $(REGION)

# Development
dev-setup: ## Set up development environment
	@echo "Setting up development environment..."
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "Development setup complete!"

# Docker
build-image: ## Build Docker image for serving
	docker build -t gcr.io/$(PROJECT_ID)/vertexrec-api:latest .

push-image: ## Push Docker image to GCR
	docker push gcr.io/$(PROJECT_ID)/vertexrec-api:latest

# End-to-end deployment
deploy-all: deploy-infra generate-data upload-data deploy-pipeline deploy-api setup-monitoring ## Deploy entire system

# Local testing
test-local: ## Test API locally
	uvicorn serving.cloud_run_api.main:app --host 0.0.0.0 --port 8080 --reload

# Data validation
validate-data: ## Validate data with TFDV
	python pipelines/data_validation/validate_data.py \
		--input-data gs://$(BUCKET_NAME)/data/interactions.csv \
		--schema-file data/schema.pbtxt

# Feature store
setup-feature-store: ## Set up Vertex AI Feature Store
	python serving/feature_store/setup_feature_store.py \
		--project-id $(PROJECT_ID) \
		--region $(REGION)

# Documentation
docs: ## Generate documentation
	mkdir -p docs/_build
	sphinx-build -b html docs/ docs/_build/html

# CI/CD helpers
ci-test: ## Run tests for CI
	python -m pytest tests/ -v --junitxml=test-results.xml

ci-lint: ## Run linting for CI
	flake8 pipelines/ serving/ scripts/ --max-line-length=88 --format=json > lint-results.json || true

# Environment variables
.env: .env.example
	cp .env.example .env
	@echo "Please edit .env file with your configuration"

# Helpers
check-env: ## Check environment variables
	@echo "Checking environment variables..."
	@test -n "$(PROJECT_ID)" || (echo "PROJECT_ID not set" && exit 1)
	@test -n "$(REGION)" || (echo "REGION not set" && exit 1)
	@test -n "$(BUCKET_NAME)" || (echo "BUCKET_NAME not set" && exit 1)
	@echo "Environment variables OK"

# Default values (can be overridden)
PROJECT_ID ?= $(shell gcloud config get-value project)
REGION ?= us-central1
BUCKET_NAME ?= vertexrec-data-$(PROJECT_ID)

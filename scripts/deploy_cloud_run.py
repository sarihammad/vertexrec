#!/usr/bin/env python3
"""
Cloud Run Deployment Script

Helper script to deploy the VertexRec API service to Cloud Run.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_and_push_image(project_id: str, service_name: str, region: str):
    """Build and push Docker image to Google Container Registry.
    
    Args:
        project_id: GCP project ID
        service_name: Name of the Cloud Run service
        region: GCP region
    """
    logger.info(f"Building and pushing Docker image for {service_name}")
    
    # Image configuration
    image_name = f"gcr.io/{project_id}/{service_name}"
    image_tag = "latest"
    full_image_name = f"{image_name}:{image_tag}"
    
    try:
        # Build Docker image
        logger.info("Building Docker image...")
        subprocess.run([
            "docker", "build", 
            "-t", full_image_name,
            "-f", "Dockerfile",
            "."
        ], check=True)
        
        # Push image to GCR
        logger.info("Pushing image to Google Container Registry...")
        subprocess.run([
            "docker", "push", full_image_name
        ], check=True)
        
        logger.info(f"Image pushed successfully: {full_image_name}")
        return full_image_name
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build/push image: {e}")
        sys.exit(1)


def deploy_to_cloud_run(project_id: str, service_name: str, region: str, image_name: str):
    """Deploy service to Cloud Run.
    
    Args:
        project_id: GCP project ID
        service_name: Name of the Cloud Run service
        region: GCP region
        image_name: Full image name with tag
    """
    logger.info(f"Deploying {service_name} to Cloud Run")
    
    try:
        # Deploy to Cloud Run
        subprocess.run([
            "gcloud", "run", "deploy", service_name,
            "--image", image_name,
            "--platform", "managed",
            "--region", region,
            "--project", project_id,
            "--allow-unauthenticated",
            "--memory", "2Gi",
            "--cpu", "2",
            "--min-instances", "0",
            "--max-instances", "10",
            "--port", "8080",
            "--set-env-vars", f"PROJECT_ID={project_id},REGION={region}",
            "--timeout", "300"
        ], check=True)
        
        logger.info(f"Service {service_name} deployed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to deploy to Cloud Run: {e}")
        sys.exit(1)


def get_service_url(project_id: str, service_name: str, region: str) -> str:
    """Get the URL of the deployed Cloud Run service.
    
    Args:
        project_id: GCP project ID
        service_name: Name of the Cloud Run service
        region: GCP region
        
    Returns:
        Service URL
    """
    try:
        result = subprocess.run([
            "gcloud", "run", "services", "describe", service_name,
            "--platform", "managed",
            "--region", region,
            "--project", project_id,
            "--format", "value(status.url)"
        ], capture_output=True, text=True, check=True)
        
        service_url = result.stdout.strip()
        logger.info(f"Service URL: {service_url}")
        return service_url
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get service URL: {e}")
        sys.exit(1)


def test_service(service_url: str):
    """Test the deployed service.
    
    Args:
        service_url: URL of the Cloud Run service
    """
    logger.info("Testing deployed service")
    
    try:
        # Test health endpoint
        import requests
        
        health_url = f"{service_url}/health"
        response = requests.get(health_url, timeout=30)
        
        if response.status_code == 200:
            logger.info("Health check passed")
            logger.info(f"Response: {response.json()}")
        else:
            logger.error(f"Health check failed: {response.status_code}")
            
        # Test recommendation endpoint
        recommend_url = f"{service_url}/recommend"
        test_request = {
            "user_id": "user_000001",
            "k": 5
        }
        
        response = requests.post(recommend_url, json=test_request, timeout=30)
        
        if response.status_code == 200:
            logger.info("Recommendation endpoint test passed")
            logger.info(f"Response: {response.json()}")
        else:
            logger.error(f"Recommendation endpoint test failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Service test failed: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Deploy VertexRec API to Cloud Run")
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--service-name", default="vertexrec-api", help="Cloud Run service name")
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker build/push")
    parser.add_argument("--skip-test", action="store_true", help="Skip service testing")
    parser.add_argument("--image-name", help="Custom image name (if skipping build)")
    
    args = parser.parse_args()
    
    try:
        # Build and push image (unless skipped)
        if args.skip_build:
            if args.image_name:
                image_name = args.image_name
            else:
                image_name = f"gcr.io/{args.project_id}/{args.service_name}:latest"
        else:
            image_name = build_and_push_image(args.project_id, args.service_name, args.region)
        
        # Deploy to Cloud Run
        deploy_to_cloud_run(args.project_id, args.service_name, args.region, image_name)
        
        # Get service URL
        service_url = get_service_url(args.project_id, args.service_name, args.region)
        
        # Test service (unless skipped)
        if not args.skip_test:
            test_service(service_url)
        
        logger.info("Deployment completed successfully!")
        logger.info(f"Service URL: {service_url}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

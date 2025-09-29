#!/usr/bin/env python3
"""
Pipeline Deployment Script

Helper script to deploy the VertexRec ML pipeline to Vertex AI Pipelines.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from google.cloud import aiplatform
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def deploy_pipeline(project_id: str, region: str, bucket_name: str, pipeline_file: str):
    """Deploy the ML pipeline to Vertex AI Pipelines.
    
    Args:
        project_id: GCP project ID
        region: GCP region
        bucket_name: GCS bucket name
        pipeline_file: Path to pipeline YAML file
    """
    logger.info(f"Deploying pipeline to project {project_id} in region {region}")
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Check if pipeline file exists
    if not Path(pipeline_file).exists():
        logger.error(f"Pipeline file not found: {pipeline_file}")
        sys.exit(1)
    
    try:
        # Create pipeline run
        pipeline_run = aiplatform.PipelineJob(
            display_name="vertexrec-pipeline-run",
            template_path=pipeline_file,
            pipeline_root=f"gs://{bucket_name}/pipeline-runs",
            parameter_values={
                "users_data_path": f"gs://{bucket_name}/data/users.csv",
                "items_data_path": f"gs://{bucket_name}/data/items.csv",
                "interactions_data_path": f"gs://{bucket_name}/data/interactions.csv",
                "epochs": 10,
                "embedding_dim": 64,
                "learning_rate": 0.01,
                "n_estimators": 100,
                "max_depth": 6,
                "xgb_learning_rate": 0.1
            }
        )
        
        # Submit pipeline
        pipeline_run.submit()
        
        logger.info("Pipeline submitted successfully")
        logger.info(f"Pipeline run: {pipeline_run.resource_name}")
        
    except Exception as e:
        logger.error(f"Failed to deploy pipeline: {e}")
        sys.exit(1)


def compile_pipeline(project_id: str, region: str, bucket_name: str, pipeline_file: str):
    """Compile the ML pipeline.
    
    Args:
        project_id: GCP project ID
        region: GCP region
        bucket_name: GCS bucket name
        pipeline_file: Path to pipeline YAML file
    """
    logger.info("Compiling pipeline")
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    try:
        # Import pipeline module
        sys.path.append(str(Path(__file__).parent.parent / "pipelines"))
        from pipeline import vertexrec_pipeline
        
        # Compile pipeline
        import kfp
        kfp.compiler.Compiler().compile(
            pipeline_func=vertexrec_pipeline,
            package_path=pipeline_file
        )
        
        logger.info(f"Pipeline compiled to {pipeline_file}")
        
    except Exception as e:
        logger.error(f"Failed to compile pipeline: {e}")
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Deploy VertexRec ML pipeline")
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--bucket-name", required=True, help="GCS bucket name")
    parser.add_argument("--pipeline-file", default="vertexrec_pipeline.yaml",
                       help="Path to pipeline YAML file")
    parser.add_argument("--action", choices=["compile", "deploy"], default="deploy",
                       help="Action to perform")
    
    args = parser.parse_args()
    
    try:
        if args.action == "compile":
            compile_pipeline(args.project_id, args.region, args.bucket_name, args.pipeline_file)
        elif args.action == "deploy":
            deploy_pipeline(args.project_id, args.region, args.bucket_name, args.pipeline_file)
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
VertexRec ML Pipeline Orchestrator

This module orchestrates the complete ML pipeline for the recommendation system,
including data validation, feature engineering, model training, evaluation, and deployment.
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import kfp
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Pipeline configuration
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
BUCKET_NAME = os.getenv("BUCKET_NAME")
PIPELINE_NAME = "vertexrec-ml-pipeline"
PIPELINE_VERSION = "1.0.0"

# Initialize clients
aiplatform.init(project=PROJECT_ID, location=REGION)
bq_client = bigquery.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)


# KFP Components
@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.1.4",
        "numpy==1.24.4",
        "tensorflow-data-validation==1.14.0",
        "google-cloud-storage==2.10.0",
        "google-cloud-bigquery==3.13.0"
    ]
)
def data_validation_component(
    input_data_path: str,
    output_dir: str,
    data_type: str = "interactions"
) -> str:
    """Validate input data using TFDV."""
    import sys
    sys.path.append("/pipelines/component/src")
    
    from data_validation.validate_data import DataValidator
    
    # Initialize validator
    validator = DataValidator(output_dir)
    
    # Validate data based on type
    if data_type == "interactions":
        is_valid, anomalies_path = validator.validate_interactions_data(input_data_path)
    elif data_type == "users":
        is_valid, anomalies_path = validator.validate_users_data(input_data_path)
    elif data_type == "items":
        is_valid, anomalies_path = validator.validate_items_data(input_data_path)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    if not is_valid:
        raise ValueError(f"Data validation failed. See anomalies: {anomalies_path}")
    
    return output_dir


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.1.4",
        "numpy==1.24.4",
        "scikit-learn==1.3.2",
        "google-cloud-storage==2.10.0",
        "google-cloud-bigquery==3.13.0"
    ]
)
def feature_engineering_component(
    users_data_path: str,
    items_data_path: str,
    interactions_data_path: str,
    output_dir: str
) -> str:
    """Engineer features for the recommendation system."""
    import sys
    sys.path.append("/pipelines/component/src")
    
    from feature_engineering.feature_engineering import FeatureEngineer
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(output_dir)
    
    # Load data
    users_df, items_df, interactions_df = feature_engineer.load_data(
        users_data_path, items_data_path, interactions_data_path
    )
    
    # Engineer features
    user_features = feature_engineer.engineer_user_features(users_df, interactions_df)
    item_features = feature_engineer.engineer_item_features(items_df, interactions_df)
    interaction_features = feature_engineer.engineer_interaction_features(
        interactions_df, users_df, items_df
    )
    
    # Save features
    feature_engineer.save_features(user_features, item_features, interaction_features)
    
    return output_dir


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.1.4",
        "numpy==1.24.4",
        "tensorflow==2.15.0",
        "tensorflow-recommenders==0.7.3",
        "google-cloud-storage==2.10.0"
    ]
)
def tf_recommenders_training_component(
    interactions_data_path: str,
    users_data_path: str,
    items_data_path: str,
    output_dir: str,
    epochs: int = 10,
    embedding_dim: int = 64,
    learning_rate: float = 0.01
) -> str:
    """Train TF Recommenders model."""
    import sys
    sys.path.append("/pipelines/component/src")
    
    from training.tf_recommenders_trainer import TFRecommendersTrainer
    
    # Initialize trainer
    trainer = TFRecommendersTrainer(
        output_dir=output_dir,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate
    )
    
    # Train model
    model = trainer.train(
        interactions_path=interactions_data_path,
        users_path=users_data_path,
        items_path=items_data_path,
        epochs=epochs
    )
    
    return output_dir


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.1.4",
        "numpy==1.24.4",
        "xgboost==2.0.2",
        "scikit-learn==1.3.2",
        "google-cloud-storage==2.10.0"
    ]
)
def xgboost_training_component(
    interactions_data_path: str,
    users_data_path: str,
    items_data_path: str,
    output_dir: str,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1
) -> str:
    """Train XGBoost ranking model."""
    import sys
    sys.path.append("/pipelines/component/src")
    
    from training.xgboost_trainer import XGBoostRankingTrainer
    
    # Initialize trainer
    trainer = XGBoostRankingTrainer(output_dir=output_dir)
    
    # Model parameters
    model_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate
    }
    
    # Train model
    model = trainer.train(
        interactions_path=interactions_data_path,
        users_path=users_data_path,
        items_path=items_path,
        model_params=model_params
    )
    
    return output_dir


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.1.4",
        "numpy==1.24.4",
        "scikit-learn==1.3.2",
        "google-cloud-storage==2.10.0"
    ]
)
def evaluation_component(
    interactions_data_path: str,
    users_data_path: str,
    items_data_path: str,
    tf_recommenders_model_path: str,
    xgboost_model_path: str,
    output_dir: str
) -> str:
    """Evaluate trained models."""
    import sys
    sys.path.append("/pipelines/component/src")
    
    from evaluation.evaluation_metrics import RecommendationEvaluator
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator(output_dir)
    
    # Load data
    interactions_df, users_df, items_df = evaluator.load_data(
        interactions_data_path, users_data_path, items_data_path
    )
    
    # Split data
    train_df, test_df = evaluator.split_data(interactions_df)
    
    # TODO: Generate recommendations using trained models
    # For now, create dummy recommendations
    recommendations = {}
    for user_id in test_df['user_id'].unique()[:100]:  # Limit for demo
        recommendations[user_id] = test_df[
            test_df['user_id'] != user_id
        ]['item_id'].unique()[:10].tolist()
    
    # Evaluate recommendations
    results = evaluator.evaluate_recommendations(
        recommendations, test_df, items_df, len(items_df)
    )
    
    # Save results
    evaluator.save_results(results)
    
    return output_dir


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-aiplatform==1.38.1",
        "google-cloud-storage==2.10.0"
    ]
)
def model_registry_component(
    tf_recommenders_model_path: str,
    xgboost_model_path: str,
    evaluation_results_path: str
) -> str:
    """Register models in Vertex AI Model Registry."""
    from google.cloud import aiplatform
    
    # Register TF Recommenders model
    tf_model = aiplatform.Model.upload(
        display_name="tf-recommenders-model",
        artifact_uri=tf_recommenders_model_path,
        serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-11:latest",
        description="TensorFlow Recommenders model for collaborative filtering"
    )
    
    # Register XGBoost model
    xgb_model = aiplatform.Model.upload(
        display_name="xgboost-model",
        artifact_uri=xgboost_model_path,
        serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/xgboost-cpu.1-7:latest",
        description="XGBoost model for ranking and regression"
    )
    
    return f"TF Model: {tf_model.resource_name}, XGB Model: {xgb_model.resource_name}"


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-aiplatform==1.38.1"
    ]
)
def model_deployment_component(
    model_resource_name: str,
    endpoint_name: str = "vertexrec-endpoint"
) -> str:
    """Deploy model to Vertex AI endpoint."""
    from google.cloud import aiplatform
    
    # Get model
    model = aiplatform.Model(model_resource_name)
    
    # Create endpoint if it doesn't exist
    try:
        endpoint = aiplatform.Endpoint(endpoint_name)
    except:
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_name,
            description="VertexRec recommendation endpoint"
        )
    
    # Deploy model
    deployed_model = endpoint.deploy(
        model=model,
        deployed_model_display_name=f"{model.display_name}-deployment",
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3
    )
    
    return endpoint.resource_name


# Pipeline definition
@dsl.pipeline(
    name=PIPELINE_NAME,
    description="End-to-end ML pipeline for VertexRec recommendation system"
)
def vertexrec_pipeline(
    users_data_path: str,
    items_data_path: str,
    interactions_data_path: str,
    epochs: int = 10,
    embedding_dim: int = 64,
    learning_rate: float = 0.01,
    n_estimators: int = 100,
    max_depth: int = 6,
    xgb_learning_rate: float = 0.1
):
    """Main pipeline for VertexRec recommendation system."""
    
    # Data validation
    validate_users = data_validation_component(
        input_data_path=users_data_path,
        output_dir="/tmp/validation/users",
        data_type="users"
    )
    
    validate_items = data_validation_component(
        input_data_path=items_data_path,
        output_dir="/tmp/validation/items",
        data_type="items"
    )
    
    validate_interactions = data_validation_component(
        input_data_path=interactions_data_path,
        output_dir="/tmp/validation/interactions",
        data_type="interactions"
    )
    
    # Feature engineering
    engineer_features = feature_engineering_component(
        users_data_path=users_data_path,
        items_data_path=items_data_path,
        interactions_data_path=interactions_data_path,
        output_dir="/tmp/features"
    ).after(validate_users, validate_items, validate_interactions)
    
    # Model training
    train_tf_recommenders = tf_recommenders_training_component(
        interactions_data_path=interactions_data_path,
        users_data_path=users_data_path,
        items_data_path=items_data_path,
        output_dir="/tmp/models/tf_recommenders",
        epochs=epochs,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate
    ).after(engineer_features)
    
    train_xgboost = xgboost_training_component(
        interactions_data_path=interactions_data_path,
        users_data_path=users_data_path,
        items_data_path=items_data_path,
        output_dir="/tmp/models/xgboost",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=xgb_learning_rate
    ).after(engineer_features)
    
    # Model evaluation
    evaluate_models = evaluation_component(
        interactions_data_path=interactions_data_path,
        users_data_path=users_data_path,
        items_data_path=items_data_path,
        tf_recommenders_model_path=train_tf_recommenders.output,
        xgboost_model_path=train_xgboost.output,
        output_dir="/tmp/evaluation"
    ).after(train_tf_recommenders, train_xgboost)
    
    # Model registry
    register_models = model_registry_component(
        tf_recommenders_model_path=train_tf_recommenders.output,
        xgboost_model_path=train_xgboost.output,
        evaluation_results_path=evaluate_models.output
    ).after(evaluate_models)
    
    # Model deployment
    deploy_models = model_deployment_component(
        model_resource_name=register_models.output,
        endpoint_name="vertexrec-endpoint"
    ).after(register_models)


def compile_pipeline():
    """Compile the pipeline."""
    logger.info("Compiling VertexRec pipeline")
    
    # Compile pipeline
    pipeline_compiler = kfp.dsl.Compiler()
    pipeline_compiler.compile(
        pipeline_func=vertexrec_pipeline,
        package_path="vertexrec_pipeline.yaml"
    )
    
    logger.info("Pipeline compiled successfully")


def run_pipeline():
    """Run the pipeline."""
    logger.info("Running VertexRec pipeline")
    
    # Compile pipeline
    compile_pipeline()
    
    # Create pipeline run
    pipeline_run = aiplatform.PipelineJob(
        display_name="vertexrec-pipeline-run",
        template_path="vertexrec_pipeline.yaml",
        pipeline_root=f"gs://{BUCKET_NAME}/pipeline-runs",
        parameter_values={
            "users_data_path": f"gs://{BUCKET_NAME}/data/users.csv",
            "items_data_path": f"gs://{BUCKET_NAME}/data/items.csv",
            "interactions_data_path": f"gs://{BUCKET_NAME}/data/interactions.csv",
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


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="VertexRec ML Pipeline")
    parser.add_argument("--action", choices=["compile", "run"], required=True,
                       help="Action to perform")
    parser.add_argument("--project-id", default=PROJECT_ID,
                       help="GCP project ID")
    parser.add_argument("--region", default=REGION,
                       help="GCP region")
    parser.add_argument("--bucket-name", default=BUCKET_NAME,
                       help="GCS bucket name")
    
    args = parser.parse_args()
    
    # Update global variables
    global PROJECT_ID, REGION, BUCKET_NAME
    PROJECT_ID = args.project_id
    REGION = args.region
    BUCKET_NAME = args.bucket_name
    
    # Initialize clients
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    try:
        if args.action == "compile":
            compile_pipeline()
        elif args.action == "run":
            run_pipeline()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()

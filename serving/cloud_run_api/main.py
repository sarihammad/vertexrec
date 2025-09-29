"""
VertexRec FastAPI Service

Production-ready FastAPI service for serving recommendations via Cloud Run.
Integrates with Vertex AI Endpoints, Feature Store, and BigQuery for real-time
recommendation serving.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud import storage
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="VertexRec Recommendation API",
    description="Production-ready recommendation system powered by Vertex AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Configuration
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
DATASET_ID = os.getenv("DATASET_ID", "vertexrec_dataset")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "vertexrec-endpoint")
FEATURE_STORE_ID = os.getenv("FEATURE_STORE_ID", "vertexrec-feature-store")

# Initialize Google Cloud clients
aiplatform.init(project=PROJECT_ID, location=REGION)
bq_client = bigquery.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)

# Global variables for caching
model_cache = {}
feature_cache = {}


# Pydantic models
class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    user_id: str = Field(..., description="User ID for recommendations")
    k: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    
    @validator('k')
    def validate_k(cls, v):
        if v < 1 or v > 100:
            raise ValueError('k must be between 1 and 100')
        return v


class SimilarItemsRequest(BaseModel):
    """Request model for similar items."""
    item_id: str = Field(..., description="Item ID to find similar items for")
    k: int = Field(default=10, ge=1, le=100, description="Number of similar items")
    
    @validator('k')
    def validate_k(cls, v):
        if v < 1 or v > 100:
            raise ValueError('k must be between 1 and 100')
        return v


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    user_id: str
    recommendations: List[Dict[str, Any]]
    model_version: str
    timestamp: datetime
    processing_time_ms: float


class SimilarItemsResponse(BaseModel):
    """Response model for similar items."""
    item_id: str
    similar_items: List[Dict[str, Any]]
    model_version: str
    timestamp: datetime
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    dependencies: Dict[str, str]


# Dependency injection
async def get_model_cache():
    """Get or create model cache."""
    if not model_cache:
        await load_models()
    return model_cache


async def get_feature_cache():
    """Get or create feature cache."""
    if not feature_cache:
        await load_features()
    return feature_cache


# Model loading functions
async def load_models():
    """Load trained models from Vertex AI."""
    try:
        logger.info("Loading models from Vertex AI")
        
        # Load TF Recommenders model
        tf_model = aiplatform.Model.list(filter=f'display_name="tf-recommenders-model"')[-1]
        tf_model_cache = {
            'model': tf_model,
            'loaded_at': datetime.now()
        }
        model_cache['tf_recommenders'] = tf_model_cache
        
        # Load XGBoost model
        xgb_model = aiplatform.Model.list(filter=f'display_name="xgboost-model"')[-1]
        xgb_model_cache = {
            'model': xgb_model,
            'loaded_at': datetime.now()
        }
        model_cache['xgboost'] = xgb_model_cache
        
        logger.info("Models loaded successfully")
        
    except Exception as e:
        logger.error("Failed to load models", error=str(e))
        raise


async def load_features():
    """Load feature data from BigQuery."""
    try:
        logger.info("Loading features from BigQuery")
        
        # Load user features
        user_query = f"""
        SELECT user_id, age_group, gender, location, subscription_type, activity_level
        FROM `{PROJECT_ID}.{DATASET_ID}.users`
        LIMIT 1000
        """
        user_features = bq_client.query(user_query).to_dataframe()
        feature_cache['users'] = user_features.set_index('user_id').to_dict('index')
        
        # Load item features
        item_query = f"""
        SELECT item_id, genre, popularity_score, quality_score, price, rating_avg
        FROM `{PROJECT_ID}.{DATASET_ID}.items`
        LIMIT 1000
        """
        item_features = bq_client.query(item_query).to_dataframe()
        feature_cache['items'] = item_features.set_index('item_id').to_dict('index')
        
        logger.info("Features loaded successfully")
        
    except Exception as e:
        logger.error("Failed to load features", error=str(e))
        raise


# Recommendation functions
async def get_user_recommendations(user_id: str, k: int, context: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """Get recommendations for a user."""
    start_time = datetime.now()
    
    try:
        # Get user features
        user_features = feature_cache.get('users', {}).get(user_id, {})
        
        # Prepare input for model
        model_input = {
            'user_id': user_id,
            'user_features': user_features,
            'context': context or {}
        }
        
        # Get predictions from TF Recommenders model
        tf_model = model_cache['tf_recommenders']['model']
        predictions = tf_model.predict(model_input)
        
        # Get top-k recommendations
        recommendations = predictions[:k]
        
        # Enrich with item details
        enriched_recommendations = []
        for rec in recommendations:
            item_id = rec['item_id']
            item_features = feature_cache.get('items', {}).get(item_id, {})
            
            enriched_rec = {
                'item_id': item_id,
                'score': rec['score'],
                'title': item_features.get('title', 'Unknown'),
                'genre': item_features.get('genre', 'Unknown'),
                'rating_avg': item_features.get('rating_avg', 0),
                'price': item_features.get('price', 0)
            }
            enriched_recommendations.append(enriched_rec)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(
            "Generated recommendations",
            user_id=user_id,
            k=k,
            processing_time_ms=processing_time
        )
        
        return enriched_recommendations
        
    except Exception as e:
        logger.error("Failed to get recommendations", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


async def get_similar_items(item_id: str, k: int) -> List[Dict[str, Any]]:
    """Get similar items for a given item."""
    start_time = datetime.now()
    
    try:
        # Get item features
        item_features = feature_cache.get('items', {}).get(item_id, {})
        
        if not item_features:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
        
        # Prepare input for model
        model_input = {
            'item_id': item_id,
            'item_features': item_features
        }
        
        # Get predictions from model
        tf_model = model_cache['tf_recommenders']['model']
        predictions = tf_model.predict(model_input)
        
        # Get top-k similar items
        similar_items = predictions[:k]
        
        # Enrich with item details
        enriched_similar_items = []
        for item in similar_items:
            similar_item_id = item['item_id']
            similar_item_features = feature_cache.get('items', {}).get(similar_item_id, {})
            
            enriched_item = {
                'item_id': similar_item_id,
                'similarity_score': item['score'],
                'title': similar_item_features.get('title', 'Unknown'),
                'genre': similar_item_features.get('genre', 'Unknown'),
                'rating_avg': similar_item_features.get('rating_avg', 0),
                'price': similar_item_features.get('price', 0)
            }
            enriched_similar_items.append(enriched_item)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(
            "Generated similar items",
            item_id=item_id,
            k=k,
            processing_time_ms=processing_time
        )
        
        return enriched_similar_items
        
    except Exception as e:
        logger.error("Failed to get similar items", item_id=item_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate similar items: {str(e)}")


# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "VertexRec Recommendation API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    dependencies = {
        "vertex_ai": "ok",
        "bigquery": "ok",
        "storage": "ok"
    }
    
    # Check Vertex AI
    try:
        aiplatform.init(project=PROJECT_ID, location=REGION)
        dependencies["vertex_ai"] = "ok"
    except Exception:
        dependencies["vertex_ai"] = "error"
    
    # Check BigQuery
    try:
        bq_client.query("SELECT 1").result()
        dependencies["bigquery"] = "ok"
    except Exception:
        dependencies["bigquery"] = "error"
    
    # Check Storage
    try:
        storage_client.list_buckets()
        dependencies["storage"] = "ok"
    except Exception:
        dependencies["storage"] = "error"
    
    status = "healthy" if all(status == "ok" for status in dependencies.values()) else "unhealthy"
    
    return HealthResponse(
        status=status,
        timestamp=datetime.now(),
        version="1.0.0",
        dependencies=dependencies
    )


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    model_cache: Dict = Depends(get_model_cache),
    feature_cache: Dict = Depends(get_feature_cache)
):
    """Get recommendations for a user."""
    start_time = datetime.now()
    
    try:
        # Generate recommendations
        recommendations = await get_user_recommendations(
            request.user_id, request.k, request.context
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log recommendation request
        background_tasks.add_task(
            log_recommendation_request,
            request.user_id,
            request.k,
            processing_time
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_version="1.0.0",
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("Recommendation request failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similar", response_model=SimilarItemsResponse)
async def similar_items(
    request: SimilarItemsRequest,
    background_tasks: BackgroundTasks,
    model_cache: Dict = Depends(get_model_cache),
    feature_cache: Dict = Depends(get_feature_cache)
):
    """Get similar items for a given item."""
    start_time = datetime.now()
    
    try:
        # Generate similar items
        similar_items = await get_similar_items(request.item_id, request.k)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log similar items request
        background_tasks.add_task(
            log_similar_items_request,
            request.item_id,
            request.k,
            processing_time
        )
        
        return SimilarItemsResponse(
            item_id=request.item_id,
            similar_items=similar_items,
            model_version="1.0.0",
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("Similar items request failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/history")
async def get_user_history(user_id: str, limit: int = 20):
    """Get user interaction history."""
    try:
        query = f"""
        SELECT item_id, rating, timestamp, interaction_type
        FROM `{PROJECT_ID}.{DATASET_ID}.interactions`
        WHERE user_id = @user_id
        ORDER BY timestamp DESC
        LIMIT @limit
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
        )
        
        results = bq_client.query(query, job_config=job_config).to_dataframe()
        
        return {
            "user_id": user_id,
            "history": results.to_dict('records')
        }
        
    except Exception as e:
        logger.error("Failed to get user history", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/items/{item_id}/details")
async def get_item_details(item_id: str):
    """Get detailed information about an item."""
    try:
        query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.items`
        WHERE item_id = @item_id
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("item_id", "STRING", item_id)
            ]
        )
        
        results = bq_client.query(query, job_config=job_config).to_dataframe()
        
        if results.empty:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
        
        return results.iloc[0].to_dict()
        
    except Exception as e:
        logger.error("Failed to get item details", item_id=item_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Background tasks
async def log_recommendation_request(user_id: str, k: int, processing_time: float):
    """Log recommendation request for analytics."""
    try:
        # Log to BigQuery for analytics
        query = f"""
        INSERT INTO `{PROJECT_ID}.{DATASET_ID}.recommendation_logs`
        (user_id, k, processing_time_ms, timestamp)
        VALUES (@user_id, @k, @processing_time_ms, @timestamp)
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter("k", "INT64", k),
                bigquery.ScalarQueryParameter("processing_time_ms", "FLOAT64", processing_time),
                bigquery.ScalarQueryParameter("timestamp", "TIMESTAMP", datetime.now())
            ]
        )
        
        bq_client.query(query, job_config=job_config)
        
    except Exception as e:
        logger.error("Failed to log recommendation request", error=str(e))


async def log_similar_items_request(item_id: str, k: int, processing_time: float):
    """Log similar items request for analytics."""
    try:
        # Log to BigQuery for analytics
        query = f"""
        INSERT INTO `{PROJECT_ID}.{DATASET_ID}.similar_items_logs`
        (item_id, k, processing_time_ms, timestamp)
        VALUES (@item_id, @k, @processing_time_ms, @timestamp)
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("item_id", "STRING", item_id),
                bigquery.ScalarQueryParameter("k", "INT64", k),
                bigquery.ScalarQueryParameter("processing_time_ms", "FLOAT64", processing_time),
                bigquery.ScalarQueryParameter("timestamp", "TIMESTAMP", datetime.now())
            ]
        )
        
        bq_client.query(query, job_config=job_config)
        
    except Exception as e:
        logger.error("Failed to log similar items request", error=str(e))


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting VertexRec API")
    
    # Load models and features
    await load_models()
    await load_features()
    
    logger.info("VertexRec API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down VertexRec API")
    
    # Clear caches
    model_cache.clear()
    feature_cache.clear()
    
    logger.info("VertexRec API shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )

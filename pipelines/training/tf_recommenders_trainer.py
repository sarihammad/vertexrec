"""
TensorFlow Recommenders Training Component

This module provides training functionality for collaborative filtering models
using TensorFlow Recommenders for the VertexRec recommendation system.
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow_recommenders import metrics
from tensorflow_recommenders import layers
from tensorflow_recommenders import tasks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class UserModel(tf.keras.Model):
    """User model for collaborative filtering."""
    
    def __init__(self, unique_user_ids: List[str], embedding_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dim)
        ])
        
        # Additional user features
        self.user_features = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.1)
        ])
        
        # Combine embeddings and features
        self.combine = tf.keras.layers.Dense(embedding_dim, activation='relu')
        
    def call(self, inputs):
        user_id = inputs["user_id"]
        user_features = inputs.get("user_features", None)
        
        # Get user embedding
        user_embedding = self.user_embedding(user_id)
        
        # Process additional user features if provided
        if user_features is not None:
            processed_features = self.user_features(user_features)
            # Combine embedding and features
            combined = tf.concat([user_embedding, processed_features], axis=-1)
            return self.combine(combined)
        
        return user_embedding


class ItemModel(tf.keras.Model):
    """Item model for collaborative filtering."""
    
    def __init__(self, unique_item_ids: List[str], embedding_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        
        self.item_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_item_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_item_ids) + 1, embedding_dim)
        ])
        
        # Additional item features
        self.item_features = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.1)
        ])
        
        # Combine embeddings and features
        self.combine = tf.keras.layers.Dense(embedding_dim, activation='relu')
        
    def call(self, inputs):
        item_id = inputs["item_id"]
        item_features = inputs.get("item_features", None)
        
        # Get item embedding
        item_embedding = self.item_embedding(item_id)
        
        # Process additional item features if provided
        if item_features is not None:
            processed_features = self.item_features(item_features)
            # Combine embedding and features
            combined = tf.concat([item_embedding, processed_features], axis=-1)
            return self.combine(combined)
        
        return item_embedding


class RecommendationModel(tfrs.Model):
    """Main recommendation model combining user and item models."""
    
    def __init__(self, unique_user_ids: List[str], unique_item_ids: List[str], 
                 embedding_dim: int = 64, learning_rate: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize user and item models
        self.user_model = UserModel(unique_user_ids, embedding_dim)
        self.item_model = ItemModel(unique_item_ids, embedding_dim)
        
        # Retrieval task
        self.retrieval_task = tasks.Retrieval(
            metrics=metrics.FactorizedTopK(
                candidates=tf.data.Dataset.from_tensor_slices(unique_item_ids).batch(128).map(self.item_model)
            )
        )
        
        # Ranking task
        self.ranking_task = tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        
        # Learning rate
        self.learning_rate = learning_rate
        
    def compute_loss(self, features: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """Compute loss for both retrieval and ranking tasks."""
        
        # Get user and item representations
        user_embeddings = self.user_model(features)
        item_embeddings = self.item_model(features)
        
        # Compute retrieval loss
        retrieval_loss = self.retrieval_task(user_embeddings, item_embeddings, training=training)
        
        # Compute ranking loss if ratings are provided
        ranking_loss = 0
        if "rating" in features:
            rating_predictions = tf.reduce_sum(user_embeddings * item_embeddings, axis=1)
            rating_loss = self.ranking_task(
                rating_predictions, features["rating"], training=training
            )
            ranking_loss = rating_loss
        
        # Combine losses
        total_loss = retrieval_loss + ranking_loss
        
        return total_loss


class TFRecommendersTrainer:
    """Trainer class for TensorFlow Recommenders models."""
    
    def __init__(self, output_dir: str, embedding_dim: int = 64, learning_rate: float = 0.01):
        """Initialize the trainer.
        
        Args:
            output_dir: Directory to save trained models
            embedding_dim: Dimension of embeddings
            learning_rate: Learning rate for optimizer
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        # Training history
        self.training_history = {}
        
    def load_data(self, interactions_path: str, users_path: Optional[str] = None, 
                  items_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load training data.
        
        Args:
            interactions_path: Path to interactions data
            users_path: Path to users data (optional)
            items_path: Path to items data (optional)
            
        Returns:
            Tuple of (interactions_df, users_df, items_df)
        """
        logger.info("Loading training data")
        
        interactions_df = pd.read_csv(interactions_path)
        
        users_df = pd.DataFrame()
        items_df = pd.DataFrame()
        
        if users_path and os.path.exists(users_path):
            users_df = pd.read_csv(users_path)
        
        if items_path and os.path.exists(items_path):
            items_df = pd.read_csv(items_path)
        
        logger.info(f"Loaded {len(interactions_df)} interactions")
        if not users_df.empty:
            logger.info(f"Loaded {len(users_df)} users")
        if not items_df.empty:
            logger.info(f"Loaded {len(items_df)} items")
        
        return interactions_df, users_df, items_df
    
    def prepare_tf_dataset(self, interactions_df: pd.DataFrame, users_df: pd.DataFrame = None, 
                          items_df: pd.DataFrame = None) -> tf.data.Dataset:
        """Prepare TensorFlow dataset from pandas DataFrames.
        
        Args:
            interactions_df: Interactions DataFrame
            users_df: Users DataFrame (optional)
            items_df: Items DataFrame (optional)
            
        Returns:
            TensorFlow dataset
        """
        logger.info("Preparing TensorFlow dataset")
        
        # Merge with user and item data if available
        if not users_df.empty:
            interactions_df = interactions_df.merge(
                users_df[['user_id']], on='user_id', how='inner'
            )
        
        if not items_df.empty:
            interactions_df = interactions_df.merge(
                items_df[['item_id']], on='item_id', how='inner'
            )
        
        # Convert to dictionary format for TensorFlow
        data_dict = {
            'user_id': interactions_df['user_id'].values,
            'item_id': interactions_df['item_id'].values
        }
        
        # Add rating if available
        if 'rating' in interactions_df.columns:
            data_dict['rating'] = interactions_df['rating'].values.astype(np.float32)
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices(data_dict)
        
        # Shuffle and batch
        dataset = dataset.shuffle(10000).batch(1024)
        
        logger.info(f"Created TensorFlow dataset with {len(dataset)} batches")
        return dataset
    
    def create_model(self, unique_user_ids: List[str], unique_item_ids: List[str]) -> RecommendationModel:
        """Create the recommendation model.
        
        Args:
            unique_user_ids: List of unique user IDs
            unique_item_ids: List of unique item IDs
            
        Returns:
            RecommendationModel instance
        """
        logger.info("Creating recommendation model")
        
        model = RecommendationModel(
            unique_user_ids=unique_user_ids,
            unique_item_ids=unique_item_ids,
            embedding_dim=self.embedding_dim,
            learning_rate=self.learning_rate
        )
        
        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        
        logger.info(f"Model created with {model.count_params()} parameters")
        return model
    
    def train_model(self, model: RecommendationModel, train_dataset: tf.data.Dataset, 
                   validation_dataset: Optional[tf.data.Dataset] = None, 
                   epochs: int = 10, batch_size: int = 1024) -> RecommendationModel:
        """Train the recommendation model.
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            validation_dataset: Validation dataset (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Trained model
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        # Prepare callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_dataset else 'loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_dataset else 'loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store training history
        self.training_history = history.history
        
        logger.info("Training completed successfully")
        return model
    
    def evaluate_model(self, model: RecommendationModel, test_dataset: tf.data.Dataset) -> Dict:
        """Evaluate the trained model.
        
        Args:
            model: Trained model
            test_dataset: Test dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model")
        
        # Evaluate model
        evaluation_results = model.evaluate(test_dataset, return_dict=True)
        
        # Calculate additional metrics
        metrics_dict = {
            'loss': evaluation_results.get('loss', 0),
            'retrieval_loss': evaluation_results.get('retrieval_task_loss', 0),
            'ranking_loss': evaluation_results.get('ranking_task_loss', 0),
            'rmse': evaluation_results.get('root_mean_squared_error', 0)
        }
        
        logger.info(f"Evaluation results: {metrics_dict}")
        return metrics_dict
    
    def save_model(self, model: RecommendationModel, model_name: str = "tf_recommenders_model"):
        """Save the trained model.
        
        Args:
            model: Trained model
            model_name: Name for the saved model
        """
        logger.info(f"Saving model as {model_name}")
        
        # Save model
        model_path = self.output_dir / f"{model_name}"
        model.save(str(model_path))
        
        # Save training history
        history_path = self.output_dir / f"{model_name}_history.json"
        import json
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'embedding_dim': self.embedding_dim,
            'learning_rate': self.learning_rate,
            'total_params': model.count_params(),
            'training_history': self.training_history
        }
        
        metadata_path = self.output_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def create_index(self, model: RecommendationModel, unique_item_ids: List[str]) -> tfrs.layers.factorized_top_k.TopK:
        """Create index for efficient retrieval.
        
        Args:
            model: Trained model
            unique_item_ids: List of unique item IDs
            
        Returns:
            TopK index for retrieval
        """
        logger.info("Creating retrieval index")
        
        # Create item dataset
        item_dataset = tf.data.Dataset.from_tensor_slices(unique_item_ids)
        
        # Get item embeddings
        item_embeddings = model.item_model({
            "item_id": item_dataset.batch(1024)
        })
        
        # Create index
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        index.index_from_dataset(
            tf.data.Dataset.zip((
                item_dataset.batch(1024),
                item_embeddings
            ))
        )
        
        logger.info("Retrieval index created")
        return index
    
    def train(self, interactions_path: str, users_path: Optional[str] = None, 
              items_path: Optional[str] = None, epochs: int = 10, 
              validation_split: float = 0.2) -> RecommendationModel:
        """Complete training pipeline.
        
        Args:
            interactions_path: Path to interactions data
            users_path: Path to users data (optional)
            items_path: Path to items data (optional)
            epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            
        Returns:
            Trained model
        """
        # Load data
        interactions_df, users_df, items_df = self.load_data(
            interactions_path, users_path, items_path
        )
        
        # Get unique IDs
        unique_user_ids = interactions_df['user_id'].unique().tolist()
        unique_item_ids = interactions_df['item_id'].unique().tolist()
        
        logger.info(f"Training with {len(unique_user_ids)} users and {len(unique_item_ids)} items")
        
        # Prepare datasets
        dataset = self.prepare_tf_dataset(interactions_df, users_df, items_df)
        
        # Split data
        dataset_size = sum(1 for _ in dataset)
        train_size = int(dataset_size * (1 - validation_split))
        
        train_dataset = dataset.take(train_size)
        validation_dataset = dataset.skip(train_size)
        
        # Create model
        model = self.create_model(unique_user_ids, unique_item_ids)
        
        # Train model
        trained_model = self.train_model(model, train_dataset, validation_dataset, epochs)
        
        # Evaluate model
        evaluation_results = self.evaluate_model(trained_model, validation_dataset)
        
        # Save model
        self.save_model(trained_model)
        
        # Create and save index
        index = self.create_index(trained_model, unique_item_ids)
        index_path = self.output_dir / "retrieval_index"
        index.save(str(index_path))
        
        logger.info("Training pipeline completed successfully")
        return trained_model


def main():
    """Main function for TF Recommenders training."""
    parser = argparse.ArgumentParser(description='Train TF Recommenders model')
    parser.add_argument('--interactions-data', type=str, required=True,
                       help='Path to interactions data file')
    parser.add_argument('--users-data', type=str,
                       help='Path to users data file (optional)')
    parser.add_argument('--items-data', type=str,
                       help='Path to items data file (optional)')
    parser.add_argument('--output-dir', type=str, default='tf_recommenders_output',
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = TFRecommendersTrainer(
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate
    )
    
    try:
        # Train model
        model = trainer.train(
            interactions_path=args.interactions_data,
            users_path=args.users_data,
            items_path=args.items_data,
            epochs=args.epochs,
            validation_split=args.validation_split
        )
        
        print("TF Recommenders training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()

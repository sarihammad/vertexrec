"""
XGBoost Training Component for Ranking

This module provides training functionality for ranking models using XGBoost
as a complement to the collaborative filtering approach in VertexRec.
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XGBoostRankingTrainer:
    """XGBoost trainer for ranking and regression tasks."""
    
    def __init__(self, output_dir: str, objective: str = 'reg:squarederror'):
        """Initialize the XGBoost trainer.
        
        Args:
            output_dir: Directory to save trained models
            objective: XGBoost objective function
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.objective = objective
        
        # Model and preprocessing objects
        self.model = None
        self.label_encoders = {}
        self.scalers = {}
        self.feature_names = []
        
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
        
        if users_path:
            users_df = pd.read_csv(users_path)
        
        if items_path:
            items_df = pd.read_csv(items_path)
        
        logger.info(f"Loaded {len(interactions_df)} interactions")
        if not users_df.empty:
            logger.info(f"Loaded {len(users_df)} users")
        if not items_df.empty:
            logger.info(f"Loaded {len(items_df)} items")
        
        return interactions_df, users_df, items_df
    
    def engineer_features(self, interactions_df: pd.DataFrame, users_df: pd.DataFrame = None, 
                         items_df: pd.DataFrame = None) -> pd.DataFrame:
        """Engineer features for XGBoost training.
        
        Args:
            interactions_df: Interactions DataFrame
            users_df: Users DataFrame (optional)
            items_df: Items DataFrame (optional)
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features for XGBoost")
        
        features_df = interactions_df.copy()
        
        # Merge with user and item data if available
        if not users_df.empty:
            features_df = features_df.merge(users_df, on='user_id', how='left')
        
        if not items_df.empty:
            features_df = features_df.merge(items_df, on='item_id', how='left')
        
        # Time-based features
        if 'timestamp' in features_df.columns:
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            features_df['hour'] = features_df['timestamp'].dt.hour
            features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
            features_df['month'] = features_df['timestamp'].dt.month
            features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
            features_df['is_evening'] = features_df['hour'].between(18, 23).astype(int)
        
        # User-item interaction features
        user_item_stats = features_df.groupby(['user_id', 'item_id']).agg({
            'rating': ['count', 'mean', 'std']
        }).round(3)
        
        user_item_stats.columns = ['user_item_interactions', 'user_item_avg_rating', 'user_item_rating_std']
        user_item_stats = user_item_stats.reset_index()
        
        features_df = features_df.merge(user_item_stats, on=['user_id', 'item_id'], how='left')
        
        # User statistics
        user_stats = features_df.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std'],
            'item_id': 'nunique'
        }).round(3)
        
        user_stats.columns = ['user_total_interactions', 'user_avg_rating', 'user_rating_std', 'user_unique_items']
        user_stats = user_stats.reset_index()
        
        features_df = features_df.merge(user_stats, on='user_id', how='left')
        
        # Item statistics
        item_stats = features_df.groupby('item_id').agg({
            'rating': ['count', 'mean', 'std'],
            'user_id': 'nunique'
        }).round(3)
        
        item_stats.columns = ['item_total_interactions', 'item_avg_rating', 'item_rating_std', 'item_unique_users']
        item_stats = item_stats.reset_index()
        
        features_df = features_df.merge(item_stats, on='item_id', how='left')
        
        # Popularity features
        features_df['user_popularity'] = features_df['user_total_interactions'] / features_df['user_total_interactions'].max()
        features_df['item_popularity'] = features_df['item_total_interactions'] / features_df['item_total_interactions'].max()
        
        # Quality features
        features_df['user_quality'] = features_df['user_avg_rating'] / 5.0
        features_df['item_quality'] = features_df['item_avg_rating'] / 5.0
        
        # Diversity features
        features_df['user_diversity'] = features_df['user_unique_items'] / features_df['user_total_interactions']
        features_df['item_diversity'] = features_df['item_unique_users'] / features_df['item_total_interactions']
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        logger.info(f"Engineered {len(features_df.columns)} features")
        return features_df
    
    def prepare_features(self, features_df: pd.DataFrame, target_column: str = 'rating') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for training.
        
        Args:
            features_df: DataFrame with features
            target_column: Name of target column
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Preparing features for training")
        
        # Select feature columns (exclude target and metadata)
        exclude_columns = [
            target_column, 'user_id', 'item_id', 'timestamp', 'session_id', 
            'device_type', 'duration_seconds', 'interaction_type'
        ]
        
        feature_columns = [col for col in features_df.columns if col not in exclude_columns]
        
        # Separate features and target
        X = features_df[feature_columns].copy()
        y = features_df[target_column].values if target_column in features_df.columns else None
        
        # Encode categorical features
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Scale numerical features
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
                X[col] = self.scalers[col].fit_transform(X[[col]]).flatten()
            else:
                X[col] = self.scalers[col].transform(X[[col]]).flatten()
        
        # Convert to numpy array
        X_array = X.values.astype(np.float32)
        
        # Store feature names
        self.feature_names = feature_columns
        
        logger.info(f"Prepared {X_array.shape[1]} features for training")
        return X_array, y, feature_columns
    
    def create_model(self, params: Dict = None) -> xgb.XGBRegressor:
        """Create XGBoost model with specified parameters.
        
        Args:
            params: XGBoost parameters
            
        Returns:
            XGBoost model
        """
        logger.info("Creating XGBoost model")
        
        # Default parameters
        default_params = {
            'objective': self.objective,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'rmse'
        }
        
        # Update with provided parameters
        if params:
            default_params.update(params)
        
        # Create model
        model = xgb.XGBRegressor(**default_params)
        
        logger.info(f"Created XGBoost model with parameters: {default_params}")
        return model
    
    def train_model(self, model: xgb.XGBRegressor, X: np.ndarray, y: np.ndarray, 
                   validation_split: float = 0.2) -> xgb.XGBRegressor:
        """Train the XGBoost model.
        
        Args:
            model: XGBoost model to train
            X: Feature matrix
            y: Target values
            validation_split: Fraction of data to use for validation
            
        Returns:
            Trained model
        """
        logger.info("Training XGBoost model")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=True
        )
        
        # Store training history
        self.training_history = {
            'train_rmse': model.evals_result()['validation_0']['rmse'],
            'best_iteration': model.best_iteration,
            'best_score': model.best_score
        }
        
        logger.info(f"Training completed. Best iteration: {model.best_iteration}")
        return model
    
    def evaluate_model(self, model: xgb.XGBRegressor, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate the trained model.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model")
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        
        # Calculate NDCG if applicable
        ndcg = None
        if len(np.unique(y)) > 1:  # Only if we have multiple rating values
            try:
                # Convert to relevance scores for NDCG
                y_relevance = np.array([y]).T
                y_pred_relevance = np.array([y_pred]).T
                ndcg = ndcg_score(y_relevance, y_pred_relevance)
            except:
                ndcg = None
        
        metrics_dict = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'ndcg': ndcg
        }
        
        logger.info(f"Evaluation results: {metrics_dict}")
        return metrics_dict
    
    def get_feature_importance(self, model: xgb.XGBRegressor) -> pd.DataFrame:
        """Get feature importance from trained model.
        
        Args:
            model: Trained XGBoost model
            
        Returns:
            DataFrame with feature importance
        """
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model: xgb.XGBRegressor, model_name: str = "xgboost_model"):
        """Save the trained model and preprocessing objects.
        
        Args:
            model: Trained model
            model_name: Name for the saved model
        """
        logger.info(f"Saving model as {model_name}")
        
        # Save model
        model_path = self.output_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save preprocessing objects
        preprocessors = {
            'label_encoders': self.label_encoders,
            'scalers': self.scalers,
            'feature_names': self.feature_names
        }
        
        preprocessors_path = self.output_dir / f"{model_name}_preprocessors.pkl"
        with open(preprocessors_path, 'wb') as f:
            pickle.dump(preprocessors, f)
        
        # Save training history
        history_path = self.output_dir / f"{model_name}_history.json"
        import json
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save feature importance
        importance_df = self.get_feature_importance(model)
        importance_path = self.output_dir / f"{model_name}_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'objective': self.objective,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        
        metadata_path = self.output_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def train(self, interactions_path: str, users_path: Optional[str] = None, 
              items_path: Optional[str] = None, validation_split: float = 0.2,
              model_params: Dict = None) -> xgb.XGBRegressor:
        """Complete training pipeline.
        
        Args:
            interactions_path: Path to interactions data
            users_path: Path to users data (optional)
            items_path: Path to items data (optional)
            validation_split: Fraction of data to use for validation
            model_params: XGBoost parameters
            
        Returns:
            Trained model
        """
        # Load data
        interactions_df, users_df, items_df = self.load_data(
            interactions_path, users_path, items_path
        )
        
        # Engineer features
        features_df = self.engineer_features(interactions_df, users_df, items_df)
        
        # Prepare features
        X, y, feature_names = self.prepare_features(features_df)
        
        logger.info(f"Training with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Create model
        model = self.create_model(model_params)
        
        # Train model
        trained_model = self.train_model(model, X, y, validation_split)
        
        # Evaluate model
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        evaluation_results = self.evaluate_model(trained_model, X_val, y_val)
        
        # Save model
        self.save_model(trained_model)
        
        logger.info("XGBoost training pipeline completed successfully")
        return trained_model


def main():
    """Main function for XGBoost training."""
    parser = argparse.ArgumentParser(description='Train XGBoost ranking model')
    parser.add_argument('--interactions-data', type=str, required=True,
                       help='Path to interactions data file')
    parser.add_argument('--users-data', type=str,
                       help='Path to users data file (optional)')
    parser.add_argument('--items-data', type=str,
                       help='Path to items data file (optional)')
    parser.add_argument('--output-dir', type=str, default='xgboost_output',
                       help='Output directory for trained model')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of boosting rounds')
    parser.add_argument('--max-depth', type=int, default=6,
                       help='Maximum tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = XGBoostRankingTrainer(output_dir=args.output_dir)
    
    # Model parameters
    model_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate
    }
    
    try:
        # Train model
        model = trainer.train(
            interactions_path=args.interactions_data,
            users_path=args.users_data,
            items_path=args.items_data,
            validation_split=args.validation_split,
            model_params=model_params
        )
        
        print("XGBoost training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()

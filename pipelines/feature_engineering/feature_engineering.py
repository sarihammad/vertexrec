"""
Feature Engineering Pipeline for VertexRec

This module provides comprehensive feature engineering for the recommendation system,
including user embeddings, item features, and interaction-based features.
"""

import argparse
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering class for recommendation system."""
    
    def __init__(self, output_dir: str):
        """Initialize the feature engineer.
        
        Args:
            output_dir: Directory to save engineered features and models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoders and scalers
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.genre_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        self.age_encoder = LabelEncoder()
        
        self.rating_scaler = StandardScaler()
        self.popularity_scaler = MinMaxScaler()
        self.quality_scaler = MinMaxScaler()
        
        # Feature transformers
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.svd_transformer = TruncatedSVD(n_components=20, random_state=42)
        
        # Feature statistics
        self.feature_stats = {}
        
    def load_data(self, users_path: str, items_path: str, interactions_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and prepare data for feature engineering.
        
        Args:
            users_path: Path to users data
            items_path: Path to items data
            interactions_path: Path to interactions data
            
        Returns:
            Tuple of (users_df, items_df, interactions_df)
        """
        logger.info("Loading data for feature engineering")
        
        # Load data
        users_df = pd.read_csv(users_path)
        items_df = pd.read_csv(items_path)
        interactions_df = pd.read_csv(interactions_path)
        
        # Convert timestamp to datetime
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        
        logger.info(f"Loaded {len(users_df)} users, {len(items_df)} items, {len(interactions_df)} interactions")
        
        return users_df, items_df, interactions_df
    
    def engineer_user_features(self, users_df: pd.DataFrame, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer user-specific features.
        
        Args:
            users_df: User data
            interactions_df: Interaction data
            
        Returns:
            DataFrame with engineered user features
        """
        logger.info("Engineering user features")
        
        user_features = users_df.copy()
        
        # Encode categorical features
        user_features['user_id_encoded'] = self.user_encoder.fit_transform(user_features['user_id'])
        user_features['gender_encoded'] = self.gender_encoder.fit_transform(user_features['gender'].fillna('Unknown'))
        user_features['age_group_encoded'] = self.age_encoder.fit_transform(user_features['age_group'].fillna('Unknown'))
        user_features['location_encoded'] = self.location_encoder.fit_transform(user_features['location'].fillna('Unknown'))
        
        # Calculate interaction statistics
        user_stats = interactions_df.groupby('user_id').agg({
            'rating': ['mean', 'std', 'count', 'min', 'max'],
            'timestamp': ['min', 'max'],
            'item_id': 'nunique'
        }).round(3)
        
        # Flatten column names
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
        user_stats = user_stats.reset_index()
        
        # Calculate derived features
        user_stats['user_activity_days'] = (user_stats['timestamp_max'] - user_stats['timestamp_min']).dt.days
        user_stats['avg_rating'] = user_stats['rating_mean']
        user_stats['rating_std'] = user_stats['rating_std'].fillna(0)
        user_stats['total_interactions'] = user_stats['rating_count']
        user_stats['unique_items_interacted'] = user_stats['item_id_nunique']
        user_stats['interaction_frequency'] = user_stats['total_interactions'] / (user_stats['user_activity_days'] + 1)
        
        # Calculate genre preferences
        user_genre_prefs = self._calculate_user_genre_preferences(interactions_df, items_df)
        user_features = user_features.merge(user_genre_prefs, on='user_id', how='left')
        
        # Merge with user stats
        user_features = user_features.merge(user_stats, on='user_id', how='left')
        
        # Fill missing values
        user_features = user_features.fillna({
            'avg_rating': user_features['avg_rating'].mean(),
            'rating_std': 0,
            'total_interactions': 0,
            'unique_items_interacted': 0,
            'interaction_frequency': 0,
            'user_activity_days': 0
        })
        
        # Create user segments
        user_features['user_segment'] = self._create_user_segments(user_features)
        
        logger.info(f"Engineered {len(user_features.columns)} user features")
        return user_features
    
    def engineer_item_features(self, items_df: pd.DataFrame, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer item-specific features.
        
        Args:
            items_df: Item data
            interactions_df: Interaction data
            
        Returns:
            DataFrame with engineered item features
        """
        logger.info("Engineering item features")
        
        item_features = items_df.copy()
        
        # Encode categorical features
        item_features['item_id_encoded'] = self.item_encoder.fit_transform(item_features['item_id'])
        item_features['genre_encoded'] = self.genre_encoder.fit_transform(item_features['genre'].fillna('Unknown'))
        
        # Calculate interaction statistics
        item_stats = interactions_df.groupby('item_id').agg({
            'rating': ['mean', 'std', 'count', 'min', 'max'],
            'timestamp': ['min', 'max'],
            'user_id': 'nunique'
        }).round(3)
        
        # Flatten column names
        item_stats.columns = ['_'.join(col).strip() for col in item_stats.columns]
        item_stats = item_stats.reset_index()
        
        # Calculate derived features
        item_stats['item_activity_days'] = (item_stats['timestamp_max'] - item_stats['timestamp_min']).dt.days
        item_stats['avg_rating'] = item_stats['rating_mean']
        item_stats['rating_std'] = item_stats['rating_std'].fillna(0)
        item_stats['total_interactions'] = item_stats['rating_count']
        item_stats['unique_users_interacted'] = item_stats['user_id_nunique']
        item_stats['interaction_frequency'] = item_stats['total_interactions'] / (item_stats['item_activity_days'] + 1)
        
        # Calculate popularity metrics
        item_stats['popularity_score'] = self.popularity_scaler.fit_transform(
            item_stats[['total_interactions', 'unique_users_interacted']].values
        ).mean(axis=1)
        
        # Merge with item stats
        item_features = item_features.merge(item_stats, on='item_id', how='left')
        
        # Fill missing values
        item_features = item_features.fillna({
            'avg_rating': item_features['avg_rating'].mean(),
            'rating_std': 0,
            'total_interactions': 0,
            'unique_users_interacted': 0,
            'interaction_frequency': 0,
            'item_activity_days': 0,
            'popularity_score': 0
        })
        
        # Create item segments
        item_features['item_segment'] = self._create_item_segments(item_features)
        
        # Text features from titles
        if 'title' in item_features.columns:
            title_features = self._extract_text_features(item_features['title'].fillna(''))
            item_features = pd.concat([item_features, title_features], axis=1)
        
        logger.info(f"Engineered {len(item_features.columns)} item features")
        return item_features
    
    def engineer_interaction_features(self, interactions_df: pd.DataFrame, 
                                    users_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer interaction-specific features.
        
        Args:
            interactions_df: Interaction data
            users_df: User data
            items_df: Item data
            
        Returns:
            DataFrame with engineered interaction features
        """
        logger.info("Engineering interaction features")
        
        interaction_features = interactions_df.copy()
        
        # Time-based features
        interaction_features['hour'] = interaction_features['timestamp'].dt.hour
        interaction_features['day_of_week'] = interaction_features['timestamp'].dt.dayofweek
        interaction_features['month'] = interaction_features['timestamp'].dt.month
        interaction_features['is_weekend'] = interaction_features['day_of_week'].isin([5, 6]).astype(int)
        interaction_features['is_evening'] = interaction_features['hour'].between(18, 23).astype(int)
        
        # Calculate time since last interaction
        interaction_features = interaction_features.sort_values(['user_id', 'timestamp'])
        interaction_features['time_since_last'] = interaction_features.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        interaction_features['time_since_last'] = interaction_features['time_since_last'].fillna(0)
        
        # Merge user and item features
        user_cols = ['age_group', 'gender', 'location', 'subscription_type']
        item_cols = ['genre', 'popularity_score', 'quality_score', 'price']
        
        interaction_features = interaction_features.merge(
            users_df[['user_id'] + user_cols], on='user_id', how='left'
        )
        interaction_features = interaction_features.merge(
            items_df[['item_id'] + item_cols], on='item_id', how='left'
        )
        
        # Encode categorical features
        for col in user_cols + item_cols:
            if col in ['popularity_score', 'quality_score', 'price']:
                continue
            le = LabelEncoder()
            interaction_features[f'{col}_encoded'] = le.fit_transform(
                interaction_features[col].fillna('Unknown')
            )
        
        # Calculate user-item affinity
        interaction_features['user_item_affinity'] = self._calculate_user_item_affinity(
            interaction_features, users_df, items_df
        )
        
        # Normalize ratings
        if 'rating' in interaction_features.columns:
            interaction_features['rating_normalized'] = self.rating_scaler.fit_transform(
                interaction_features[['rating']].values
            ).flatten()
        
        logger.info(f"Engineered {len(interaction_features.columns)} interaction features")
        return interaction_features
    
    def _calculate_user_genre_preferences(self, interactions_df: pd.DataFrame, 
                                        items_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate user genre preferences from interactions."""
        # Merge interactions with items to get genres
        user_genre_interactions = interactions_df.merge(
            items_df[['item_id', 'genre']], on='item_id', how='left'
        )
        
        # Calculate genre preferences
        genre_prefs = user_genre_interactions.groupby(['user_id', 'genre']).agg({
            'rating': ['mean', 'count']
        }).round(3)
        
        genre_prefs.columns = ['genre_avg_rating', 'genre_interaction_count']
        genre_prefs = genre_prefs.reset_index()
        
        # Pivot to wide format
        genre_prefs_wide = genre_prefs.pivot(
            index='user_id', 
            columns='genre', 
            values='genre_avg_rating'
        ).fillna(0)
        
        # Add prefix to column names
        genre_prefs_wide.columns = [f'genre_pref_{col}' for col in genre_prefs_wide.columns]
        
        return genre_prefs_wide.reset_index()
    
    def _create_user_segments(self, user_features: pd.DataFrame) -> pd.Series:
        """Create user segments based on activity and preferences."""
        segments = []
        
        for _, user in user_features.iterrows():
            if user['total_interactions'] >= 50:
                if user['avg_rating'] >= 4.0:
                    segments.append('high_activity_positive')
                else:
                    segments.append('high_activity_mixed')
            elif user['total_interactions'] >= 10:
                if user['avg_rating'] >= 4.0:
                    segments.append('medium_activity_positive')
                else:
                    segments.append('medium_activity_mixed')
            else:
                segments.append('low_activity')
        
        return pd.Series(segments, index=user_features.index)
    
    def _create_item_segments(self, item_features: pd.DataFrame) -> pd.Series:
        """Create item segments based on popularity and quality."""
        segments = []
        
        for _, item in item_features.iterrows():
            if item['popularity_score'] >= 0.7:
                if item['quality_score'] >= 0.7:
                    segments.append('popular_high_quality')
                else:
                    segments.append('popular_mixed_quality')
            elif item['quality_score'] >= 0.7:
                segments.append('niche_high_quality')
            else:
                segments.append('niche_mixed_quality')
        
        return pd.Series(segments, index=item_features.index)
    
    def _extract_text_features(self, titles: pd.Series) -> pd.DataFrame:
        """Extract text features from item titles."""
        # TF-IDF features
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(titles)
        tfidf_features = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        
        # SVD features
        svd_features = self.svd_transformer.fit_transform(tfidf_matrix)
        svd_df = pd.DataFrame(
            svd_features,
            columns=[f'svd_{i}' for i in range(svd_features.shape[1])]
        )
        
        return pd.concat([tfidf_features, svd_df], axis=1)
    
    def _calculate_user_item_affinity(self, interactions_df: pd.DataFrame,
                                    users_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.Series:
        """Calculate user-item affinity scores."""
        # Simple affinity based on genre match and rating history
        affinity_scores = []
        
        for _, interaction in interactions_df.iterrows():
            user_id = interaction['user_id']
            item_id = interaction['item_id']
            
            # Get user and item info
            user_info = users_df[users_df['user_id'] == user_id].iloc[0] if len(users_df[users_df['user_id'] == user_id]) > 0 else None
            item_info = items_df[items_df['item_id'] == item_id].iloc[0] if len(items_df[items_df['item_id'] == item_id]) > 0 else None
            
            if user_info is None or item_info is None:
                affinity_scores.append(0.5)
                continue
            
            # Calculate affinity based on genre preference
            user_genres = eval(user_info.get('preferred_genres', '[]')) if isinstance(user_info.get('preferred_genres'), str) else user_info.get('preferred_genres', [])
            item_genre = item_info.get('genre', '')
            
            genre_match = 1.0 if item_genre in user_genres else 0.0
            
            # Combine with item quality and popularity
            quality_score = item_info.get('quality_score', 0.5)
            popularity_score = item_info.get('popularity_score', 0.5)
            
            affinity = (genre_match * 0.5 + quality_score * 0.3 + popularity_score * 0.2)
            affinity_scores.append(affinity)
        
        return pd.Series(affinity_scores, index=interactions_df.index)
    
    def save_features(self, user_features: pd.DataFrame, item_features: pd.DataFrame, 
                     interaction_features: pd.DataFrame):
        """Save engineered features to files.
        
        Args:
            user_features: User features DataFrame
            item_features: Item features DataFrame
            interaction_features: Interaction features DataFrame
        """
        logger.info("Saving engineered features")
        
        # Save feature DataFrames
        user_features.to_csv(self.output_dir / "user_features.csv", index=False)
        item_features.to_csv(self.output_dir / "item_features.csv", index=False)
        interaction_features.to_csv(self.output_dir / "interaction_features.csv", index=False)
        
        # Save encoders and scalers
        encoders = {
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'genre_encoder': self.genre_encoder,
            'location_encoder': self.location_encoder,
            'gender_encoder': self.gender_encoder,
            'age_encoder': self.age_encoder
        }
        
        scalers = {
            'rating_scaler': self.rating_scaler,
            'popularity_scaler': self.popularity_scaler,
            'quality_scaler': self.quality_scaler
        }
        
        transformers = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'svd_transformer': self.svd_transformer
        }
        
        # Save to pickle files
        with open(self.output_dir / "encoders.pkl", 'wb') as f:
            pickle.dump(encoders, f)
        
        with open(self.output_dir / "scalers.pkl", 'wb') as f:
            pickle.dump(scalers, f)
        
        with open(self.output_dir / "transformers.pkl", 'wb') as f:
            pickle.dump(transformers, f)
        
        # Save feature statistics
        self.feature_stats = {
            'user_features_count': len(user_features.columns),
            'item_features_count': len(item_features.columns),
            'interaction_features_count': len(interaction_features.columns),
            'total_users': len(user_features),
            'total_items': len(item_features),
            'total_interactions': len(interaction_features)
        }
        
        with open(self.output_dir / "feature_stats.json", 'w') as f:
            import json
            json.dump(self.feature_stats, f, indent=2)
        
        logger.info("Features saved successfully")
    
    def create_feature_summary(self, user_features: pd.DataFrame, item_features: pd.DataFrame, 
                             interaction_features: pd.DataFrame) -> Dict:
        """Create a summary of engineered features.
        
        Args:
            user_features: User features DataFrame
            item_features: Item features DataFrame
            interaction_features: Interaction features DataFrame
            
        Returns:
            Dictionary with feature summary
        """
        summary = {
            'user_features': {
                'count': len(user_features.columns),
                'columns': list(user_features.columns),
                'shape': user_features.shape
            },
            'item_features': {
                'count': len(item_features.columns),
                'columns': list(item_features.columns),
                'shape': item_features.shape
            },
            'interaction_features': {
                'count': len(interaction_features.columns),
                'columns': list(interaction_features.columns),
                'shape': interaction_features.shape
            }
        }
        
        return summary


def main():
    """Main function for feature engineering."""
    parser = argparse.ArgumentParser(description='Engineer features for recommendation system')
    parser.add_argument('--users-data', type=str, required=True,
                       help='Path to users data file')
    parser.add_argument('--items-data', type=str, required=True,
                       help='Path to items data file')
    parser.add_argument('--interactions-data', type=str, required=True,
                       help='Path to interactions data file')
    parser.add_argument('--output-dir', type=str, default='feature_output',
                       help='Output directory for engineered features')
    
    args = parser.parse_args()
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(args.output_dir)
    
    try:
        # Load data
        users_df, items_df, interactions_df = feature_engineer.load_data(
            args.users_data, args.items_data, args.interactions_data
        )
        
        # Engineer features
        user_features = feature_engineer.engineer_user_features(users_df, interactions_df)
        item_features = feature_engineer.engineer_item_features(items_df, interactions_df)
        interaction_features = feature_engineer.engineer_interaction_features(
            interactions_df, users_df, items_df
        )
        
        # Save features
        feature_engineer.save_features(user_features, item_features, interaction_features)
        
        # Create summary
        summary = feature_engineer.create_feature_summary(user_features, item_features, interaction_features)
        
        print("Feature engineering completed successfully!")
        print(f"User features: {summary['user_features']['count']} features")
        print(f"Item features: {summary['item_features']['count']} features")
        print(f"Interaction features: {summary['interaction_features']['count']} features")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


if __name__ == '__main__':
    main()

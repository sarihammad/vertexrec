#!/usr/bin/env python3
"""
Synthetic Data Generator for VertexRec

Generates realistic user-item interaction data for testing and demonstration
purposes. Creates users, items, and interaction data with proper distributions
and realistic patterns.
"""

import argparse
import csv
import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from faker import Faker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

fake = Faker()


class SyntheticDataGenerator:
    """Generate synthetic recommendation system data."""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
        
        # Define realistic distributions
        self.genres = [
            'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
            'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
            'Romance', 'Science Fiction', 'Thriller', 'War', 'Western'
        ]
        
        self.age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        self.genders = ['M', 'F', 'Other']
        self.locations = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'KR', 'IN', 'BR']
        
    def generate_users(self, num_users: int) -> pd.DataFrame:
        """Generate user profiles with realistic demographics.
        
        Args:
            num_users: Number of users to generate
            
        Returns:
            DataFrame with user information
        """
        logger.info(f"Generating {num_users} users...")
        
        users = []
        for i in range(num_users):
            # Create realistic user profile
            age_group = np.random.choice(self.age_groups, p=[0.15, 0.25, 0.20, 0.15, 0.15, 0.10])
            gender = np.random.choice(self.genders, p=[0.48, 0.48, 0.04])
            location = np.random.choice(self.locations, p=[0.4, 0.1, 0.05, 0.05, 0.08, 0.07, 0.08, 0.07, 0.05, 0.05])
            
            # Generate preferences based on demographics
            preferred_genres = self._generate_preferred_genres(age_group, gender)
            
            user = {
                'user_id': f'user_{i:06d}',
                'age_group': age_group,
                'gender': gender,
                'location': location,
                'signup_date': fake.date_between(start_date='-2y', end_date='today'),
                'preferred_genres': json.dumps(preferred_genres),
                'activity_level': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2]),
                'subscription_type': np.random.choice(['free', 'premium'], p=[0.7, 0.3])
            }
            users.append(user)
            
        return pd.DataFrame(users)
    
    def _generate_preferred_genres(self, age_group: str, gender: str) -> List[str]:
        """Generate genre preferences based on demographics."""
        # Define preference patterns
        genre_preferences = {
            '18-24': ['Action', 'Comedy', 'Horror', 'Science Fiction'],
            '25-34': ['Drama', 'Romance', 'Thriller', 'Comedy'],
            '35-44': ['Drama', 'Documentary', 'History', 'Family'],
            '45-54': ['Drama', 'Documentary', 'History', 'Mystery'],
            '55-64': ['Documentary', 'History', 'Drama', 'Mystery'],
            '65+': ['Documentary', 'History', 'Drama', 'Family']
        }
        
        gender_preferences = {
            'M': ['Action', 'Thriller', 'Science Fiction', 'War'],
            'F': ['Romance', 'Drama', 'Comedy', 'Family'],
            'Other': ['Drama', 'Comedy', 'Documentary', 'Animation']
        }
        
        # Combine age and gender preferences
        preferred = set(genre_preferences.get(age_group, []))
        preferred.update(gender_preferences.get(gender, []))
        
        # Add some random genres
        additional = random.sample(
            [g for g in self.genres if g not in preferred], 
            random.randint(1, 3)
        )
        preferred.update(additional)
        
        return list(preferred)
    
    def generate_items(self, num_items: int) -> pd.DataFrame:
        """Generate item catalog with metadata.
        
        Args:
            num_items: Number of items to generate
            
        Returns:
            DataFrame with item information
        """
        logger.info(f"Generating {num_items} items...")
        
        items = []
        for i in range(num_items):
            # Generate item metadata
            title = fake.catch_phrase()
            genre = random.choice(self.genres)
            
            # Generate popularity and quality scores
            popularity_score = np.random.beta(2, 5)  # Skewed towards lower popularity
            quality_score = np.random.normal(0.7, 0.2)  # Around 0.7 with some variance
            quality_score = max(0, min(1, quality_score))  # Clamp to [0, 1]
            
            # Generate release date
            release_date = fake.date_between(start_date='-10y', end_date='today')
            
            # Generate price (free or paid)
            is_free = random.choice([True, False], p=[0.3, 0.7])
            price = 0 if is_free else round(random.uniform(0.99, 29.99), 2)
            
            item = {
                'item_id': f'item_{i:06d}',
                'title': title,
                'genre': genre,
                'release_date': release_date,
                'popularity_score': round(popularity_score, 3),
                'quality_score': round(quality_score, 3),
                'price': price,
                'duration_minutes': random.randint(30, 180),
                'language': random.choice(['English', 'Spanish', 'French', 'German', 'Japanese']),
                'rating_avg': round(random.uniform(2.0, 5.0), 1),
                'rating_count': random.randint(10, 10000)
            }
            items.append(item)
            
        return pd.DataFrame(items)
    
    def generate_interactions(self, users_df: pd.DataFrame, items_df: pd.DataFrame, 
                            num_interactions: int) -> pd.DataFrame:
        """Generate user-item interactions with realistic patterns.
        
        Args:
            users_df: User data
            items_df: Item data
            num_interactions: Number of interactions to generate
            
        Returns:
            DataFrame with interaction data
        """
        logger.info(f"Generating {num_interactions} interactions...")
        
        interactions = []
        user_ids = users_df['user_id'].tolist()
        item_ids = items_df['item_id'].tolist()
        
        # Create user activity profiles
        user_activity = {}
        for _, user in users_df.iterrows():
            activity_level = user['activity_level']
            if activity_level == 'high':
                user_activity[user['user_id']] = random.randint(50, 200)
            elif activity_level == 'medium':
                user_activity[user['user_id']] = random.randint(10, 50)
            else:  # low
                user_activity[user['user_id']] = random.randint(1, 10)
        
        # Generate interactions
        interaction_count = 0
        while interaction_count < num_interactions:
            user_id = random.choice(user_ids)
            
            # Check if user has reached their activity limit
            if user_activity.get(user_id, 0) <= 0:
                continue
                
            # Select item based on user preferences and item popularity
            item_id = self._select_item_for_user(user_id, users_df, items_df)
            
            # Generate interaction timestamp
            timestamp = self._generate_interaction_timestamp()
            
            # Generate interaction type and rating
            interaction_type = self._generate_interaction_type()
            rating = self._generate_rating(user_id, item_id, users_df, items_df)
            
            interaction = {
                'user_id': user_id,
                'item_id': item_id,
                'interaction_type': interaction_type,
                'rating': rating,
                'timestamp': timestamp,
                'session_id': f'session_{random.randint(1000, 9999)}',
                'device_type': random.choice(['mobile', 'desktop', 'tablet']),
                'duration_seconds': random.randint(30, 7200) if interaction_type == 'watch' else None
            }
            
            interactions.append(interaction)
            user_activity[user_id] -= 1
            interaction_count += 1
            
            if interaction_count % 10000 == 0:
                logger.info(f"Generated {interaction_count} interactions...")
        
        return pd.DataFrame(interactions)
    
    def _select_item_for_user(self, user_id: str, users_df: pd.DataFrame, 
                            items_df: pd.DataFrame) -> str:
        """Select an item for a user based on preferences and popularity."""
        user_row = users_df[users_df['user_id'] == user_id].iloc[0]
        preferred_genres = json.loads(user_row['preferred_genres'])
        
        # Filter items by preferred genres
        preferred_items = items_df[items_df['genre'].isin(preferred_genres)]
        
        if len(preferred_items) == 0:
            preferred_items = items_df
            
        # Weight selection by popularity and quality
        weights = (preferred_items['popularity_score'] * 0.3 + 
                  preferred_items['quality_score'] * 0.7)
        
        # Add some randomness
        weights = weights * np.random.uniform(0.8, 1.2, len(weights))
        
        selected_item = np.random.choice(preferred_items['item_id'], p=weights/weights.sum())
        return selected_item
    
    def _generate_interaction_timestamp(self) -> str:
        """Generate realistic interaction timestamps."""
        # More interactions during evening hours and weekends
        base_date = fake.date_between(start_date='-1y', end_date='today')
        
        # Weekend bias
        if random.random() < 0.3:  # 30% chance of weekend
            day_offset = random.choice([5, 6])  # Saturday or Sunday
            interaction_date = base_date + timedelta(days=day_offset)
        else:
            interaction_date = base_date
            
        # Evening hours bias (6 PM - 11 PM)
        hour = np.random.choice(
            list(range(24)), 
            p=[0.01]*6 + [0.02]*6 + [0.03]*6 + [0.05]*3 + [0.08]*3 + [0.02]*6
        )
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        timestamp = datetime.combine(interaction_date, datetime.min.time().replace(hour=hour, minute=minute, second=second))
        return timestamp.isoformat()
    
    def _generate_interaction_type(self) -> str:
        """Generate interaction type with realistic distribution."""
        return np.random.choice(
            ['view', 'like', 'watch', 'purchase', 'share'],
            p=[0.4, 0.2, 0.25, 0.1, 0.05]
        )
    
    def _generate_rating(self, user_id: str, item_id: str, users_df: pd.DataFrame, 
                        items_df: pd.DataFrame) -> float:
        """Generate realistic ratings based on user and item characteristics."""
        user_row = users_df[users_df['user_id'] == user_id].iloc[0]
        item_row = items_df[items_df['item_id'] == item_id].iloc[0]
        
        # Base rating on item quality
        base_rating = item_row['quality_score'] * 5
        
        # Add user bias (some users are more generous)
        user_bias = np.random.normal(0, 0.5)
        
        # Add some randomness
        noise = np.random.normal(0, 0.3)
        
        rating = base_rating + user_bias + noise
        rating = max(1, min(5, rating))  # Clamp to [1, 5]
        
        return round(rating, 1)


def main():
    """Main function to generate synthetic data."""
    parser = argparse.ArgumentParser(description='Generate synthetic recommendation data')
    parser.add_argument('--output-dir', type=str, default='data/', 
                       help='Output directory for generated data')
    parser.add_argument('--num-users', type=int, default=10000,
                       help='Number of users to generate')
    parser.add_argument('--num-items', type=int, default=5000,
                       help='Number of items to generate')
    parser.add_argument('--num-interactions', type=int, default=100000,
                       help='Number of interactions to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = SyntheticDataGenerator(seed=args.seed)
    
    try:
        # Generate data
        logger.info("Starting data generation...")
        
        users_df = generator.generate_users(args.num_users)
        items_df = generator.generate_items(args.num_items)
        interactions_df = generator.generate_interactions(users_df, items_df, args.num_interactions)
        
        # Save data
        logger.info("Saving data to CSV files...")
        
        users_df.to_csv(output_dir / 'users.csv', index=False)
        items_df.to_csv(output_dir / 'items.csv', index=False)
        interactions_df.to_csv(output_dir / 'interactions.csv', index=False)
        
        # Generate data summary
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'parameters': {
                'num_users': args.num_users,
                'num_items': args.num_items,
                'num_interactions': args.num_interactions,
                'seed': args.seed
            },
            'data_summary': {
                'users': len(users_df),
                'items': len(items_df),
                'interactions': len(interactions_df),
                'unique_user_interactions': interactions_df['user_id'].nunique(),
                'unique_item_interactions': interactions_df['item_id'].nunique(),
                'avg_rating': round(interactions_df['rating'].mean(), 2),
                'date_range': {
                    'start': interactions_df['timestamp'].min(),
                    'end': interactions_df['timestamp'].max()
                }
            }
        }
        
        with open(output_dir / 'data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Data generation completed successfully!")
        logger.info(f"Generated {len(users_df)} users, {len(items_df)} items, {len(interactions_df)} interactions")
        logger.info(f"Data saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        raise


if __name__ == '__main__':
    main()

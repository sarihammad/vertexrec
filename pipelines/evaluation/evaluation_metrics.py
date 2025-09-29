"""
Evaluation Metrics for Recommendation Systems

This module provides comprehensive evaluation metrics for recommendation systems,
including Recall@K, NDCG@K, MRR, Coverage, and Diversity metrics.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """Comprehensive evaluator for recommendation systems."""
    
    def __init__(self, output_dir: str):
        """Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation results
        self.evaluation_results = {}
        
    def load_data(self, interactions_path: str, users_path: Optional[str] = None, 
                  items_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load data for evaluation.
        
        Args:
            interactions_path: Path to interactions data
            users_path: Path to users data (optional)
            items_path: Path to items data (optional)
            
        Returns:
            Tuple of (interactions_df, users_df, items_df)
        """
        logger.info("Loading data for evaluation")
        
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
    
    def create_user_item_matrix(self, interactions_df: pd.DataFrame, 
                               rating_threshold: float = 3.0) -> pd.DataFrame:
        """Create user-item interaction matrix.
        
        Args:
            interactions_df: Interactions DataFrame
            rating_threshold: Minimum rating to consider as positive interaction
            
        Returns:
            User-item matrix
        """
        logger.info("Creating user-item interaction matrix")
        
        # Filter positive interactions
        positive_interactions = interactions_df[
            interactions_df['rating'] >= rating_threshold
        ].copy()
        
        # Create matrix
        user_item_matrix = positive_interactions.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        
        logger.info(f"Created matrix with {user_item_matrix.shape[0]} users and {user_item_matrix.shape[1]} items")
        return user_item_matrix
    
    def split_data(self, interactions_df: pd.DataFrame, test_ratio: float = 0.2, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split interactions into train and test sets.
        
        Args:
            interactions_df: Interactions DataFrame
            test_ratio: Fraction of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Splitting data into train and test sets")
        
        # Sort by timestamp if available
        if 'timestamp' in interactions_df.columns:
            interactions_df = interactions_df.sort_values('timestamp')
        
        # Split data
        test_size = int(len(interactions_df) * test_ratio)
        train_df = interactions_df.iloc[:-test_size].copy()
        test_df = interactions_df.iloc[-test_size:].copy()
        
        logger.info(f"Split into {len(train_df)} training and {len(test_df)} test interactions")
        return train_df, test_df
    
    def calculate_recall_at_k(self, recommendations: Dict[str, List[str]], 
                             test_interactions: pd.DataFrame, k: int = 10) -> float:
        """Calculate Recall@K metric.
        
        Args:
            recommendations: Dictionary mapping user_id to list of recommended item_ids
            test_interactions: Test interactions DataFrame
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        logger.info(f"Calculating Recall@{k}")
        
        total_recall = 0
        total_users = 0
        
        for user_id in recommendations.keys():
            # Get recommended items
            recommended_items = set(recommendations[user_id][:k])
            
            # Get actual items in test set
            actual_items = set(test_interactions[
                test_interactions['user_id'] == user_id
            ]['item_id'].tolist())
            
            if len(actual_items) > 0:
                recall = len(recommended_items.intersection(actual_items)) / len(actual_items)
                total_recall += recall
                total_users += 1
        
        avg_recall = total_recall / total_users if total_users > 0 else 0
        logger.info(f"Recall@{k}: {avg_recall:.4f}")
        return avg_recall
    
    def calculate_precision_at_k(self, recommendations: Dict[str, List[str]], 
                                test_interactions: pd.DataFrame, k: int = 10) -> float:
        """Calculate Precision@K metric.
        
        Args:
            recommendations: Dictionary mapping user_id to list of recommended item_ids
            test_interactions: Test interactions DataFrame
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        logger.info(f"Calculating Precision@{k}")
        
        total_precision = 0
        total_users = 0
        
        for user_id in recommendations.keys():
            # Get recommended items
            recommended_items = set(recommendations[user_id][:k])
            
            # Get actual items in test set
            actual_items = set(test_interactions[
                test_interactions['user_id'] == user_id
            ]['item_id'].tolist())
            
            if len(recommended_items) > 0:
                precision = len(recommended_items.intersection(actual_items)) / len(recommended_items)
                total_precision += precision
                total_users += 1
        
        avg_precision = total_precision / total_users if total_users > 0 else 0
        logger.info(f"Precision@{k}: {avg_precision:.4f}")
        return avg_precision
    
    def calculate_ndcg_at_k(self, recommendations: Dict[str, List[str]], 
                           test_interactions: pd.DataFrame, k: int = 10) -> float:
        """Calculate NDCG@K metric.
        
        Args:
            recommendations: Dictionary mapping user_id to list of recommended item_ids
            test_interactions: Test interactions DataFrame
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score
        """
        logger.info(f"Calculating NDCG@{k}")
        
        total_ndcg = 0
        total_users = 0
        
        for user_id in recommendations.keys():
            # Get recommended items
            recommended_items = recommendations[user_id][:k]
            
            # Get actual items with ratings
            actual_items = test_interactions[
                test_interactions['user_id'] == user_id
            ][['item_id', 'rating']].set_index('item_id')['rating'].to_dict()
            
            if len(actual_items) > 0:
                # Create relevance scores
                y_true = []
                y_score = []
                
                for i, item_id in enumerate(recommended_items):
                    if item_id in actual_items:
                        y_true.append(actual_items[item_id])
                        y_score.append(len(recommended_items) - i)  # Higher score for higher rank
                    else:
                        y_true.append(0)
                        y_score.append(len(recommended_items) - i)
                
                # Calculate NDCG
                if len(y_true) > 0:
                    ndcg = ndcg_score([y_true], [y_score], k=k)
                    total_ndcg += ndcg
                    total_users += 1
        
        avg_ndcg = total_ndcg / total_users if total_users > 0 else 0
        logger.info(f"NDCG@{k}: {avg_ndcg:.4f}")
        return avg_ndcg
    
    def calculate_mrr(self, recommendations: Dict[str, List[str]], 
                     test_interactions: pd.DataFrame) -> float:
        """Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            recommendations: Dictionary mapping user_id to list of recommended item_ids
            test_interactions: Test interactions DataFrame
            
        Returns:
            MRR score
        """
        logger.info("Calculating MRR")
        
        total_mrr = 0
        total_users = 0
        
        for user_id in recommendations.keys():
            # Get recommended items
            recommended_items = recommendations[user_id]
            
            # Get actual items in test set
            actual_items = set(test_interactions[
                test_interactions['user_id'] == user_id
            ]['item_id'].tolist())
            
            if len(actual_items) > 0:
                # Find first relevant item
                reciprocal_rank = 0
                for rank, item_id in enumerate(recommended_items, 1):
                    if item_id in actual_items:
                        reciprocal_rank = 1.0 / rank
                        break
                
                total_mrr += reciprocal_rank
                total_users += 1
        
        avg_mrr = total_mrr / total_users if total_users > 0 else 0
        logger.info(f"MRR: {avg_mrr:.4f}")
        return avg_mrr
    
    def calculate_coverage(self, recommendations: Dict[str, List[str]], 
                          total_items: int) -> float:
        """Calculate Catalog Coverage.
        
        Args:
            recommendations: Dictionary mapping user_id to list of recommended item_ids
            total_items: Total number of items in catalog
            
        Returns:
            Coverage score
        """
        logger.info("Calculating Coverage")
        
        # Get all recommended items
        all_recommended_items = set()
        for user_recommendations in recommendations.values():
            all_recommended_items.update(user_recommendations)
        
        coverage = len(all_recommended_items) / total_items
        logger.info(f"Coverage: {coverage:.4f}")
        return coverage
    
    def calculate_diversity(self, recommendations: Dict[str, List[str]], 
                           items_df: pd.DataFrame) -> float:
        """Calculate Intra-list Diversity.
        
        Args:
            recommendations: Dictionary mapping user_id to list of recommended item_ids
            items_df: Items DataFrame with genre information
            
        Returns:
            Diversity score
        """
        logger.info("Calculating Diversity")
        
        total_diversity = 0
        total_users = 0
        
        # Create item-genre mapping
        item_genres = items_df.set_index('item_id')['genre'].to_dict()
        
        for user_id, user_recommendations in recommendations.items():
            if len(user_recommendations) > 1:
                # Get genres of recommended items
                recommended_genres = [item_genres.get(item_id, 'Unknown') 
                                    for item_id in user_recommendations]
                
                # Calculate diversity as 1 - average pairwise similarity
                diversity = 1 - (len(set(recommended_genres)) / len(recommended_genres))
                total_diversity += diversity
                total_users += 1
        
        avg_diversity = total_diversity / total_users if total_users > 0 else 0
        logger.info(f"Diversity: {avg_diversity:.4f}")
        return avg_diversity
    
    def calculate_novelty(self, recommendations: Dict[str, List[str]], 
                         interactions_df: pd.DataFrame) -> float:
        """Calculate Novelty (average popularity of recommended items).
        
        Args:
            recommendations: Dictionary mapping user_id to list of recommended item_ids
            interactions_df: Interactions DataFrame
            
        Returns:
            Novelty score
        """
        logger.info("Calculating Novelty")
        
        # Calculate item popularity
        item_popularity = interactions_df['item_id'].value_counts()
        max_popularity = item_popularity.max()
        
        total_novelty = 0
        total_items = 0
        
        for user_recommendations in recommendations.values():
            for item_id in user_recommendations:
                popularity = item_popularity.get(item_id, 0)
                novelty = 1 - (popularity / max_popularity)
                total_novelty += novelty
                total_items += 1
        
        avg_novelty = total_novelty / total_items if total_items > 0 else 0
        logger.info(f"Novelty: {avg_novelty:.4f}")
        return avg_novelty
    
    def evaluate_recommendations(self, recommendations: Dict[str, List[str]], 
                               test_interactions: pd.DataFrame, items_df: pd.DataFrame,
                               total_items: int, k_values: List[int] = [5, 10, 20]) -> Dict:
        """Comprehensive evaluation of recommendations.
        
        Args:
            recommendations: Dictionary mapping user_id to list of recommended item_ids
            test_interactions: Test interactions DataFrame
            items_df: Items DataFrame
            total_items: Total number of items in catalog
            k_values: List of K values for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting comprehensive evaluation")
        
        results = {}
        
        # Calculate metrics for different K values
        for k in k_values:
            logger.info(f"Evaluating with K={k}")
            
            results[f'recall@{k}'] = self.calculate_recall_at_k(recommendations, test_interactions, k)
            results[f'precision@{k}'] = self.calculate_precision_at_k(recommendations, test_interactions, k)
            results[f'ndcg@{k}'] = self.calculate_ndcg_at_k(recommendations, test_interactions, k)
        
        # Calculate other metrics
        results['mrr'] = self.calculate_mrr(recommendations, test_interactions)
        results['coverage'] = self.calculate_coverage(recommendations, total_items)
        results['diversity'] = self.calculate_diversity(recommendations, items_df)
        results['novelty'] = self.calculate_novelty(recommendations, test_interactions)
        
        # Store results
        self.evaluation_results = results
        
        logger.info("Evaluation completed successfully")
        return results
    
    def save_results(self, results: Dict, filename: str = "evaluation_results.json"):
        """Save evaluation results to file.
        
        Args:
            results: Evaluation results dictionary
            filename: Name of the output file
        """
        logger.info(f"Saving evaluation results to {filename}")
        
        import json
        
        results_path = self.output_dir / filename
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save as CSV for easy analysis
        csv_path = self.output_dir / "evaluation_results.csv"
        results_df = pd.DataFrame([results])
        results_df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to {results_path} and {csv_path}")
    
    def generate_report(self, results: Dict) -> str:
        """Generate a human-readable evaluation report.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 50)
        report.append("RECOMMENDATION SYSTEM EVALUATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Ranking metrics
        report.append("RANKING METRICS:")
        report.append("-" * 20)
        for k in [5, 10, 20]:
            if f'recall@{k}' in results:
                report.append(f"Recall@{k}:    {results[f'recall@{k}']:.4f}")
            if f'precision@{k}' in results:
                report.append(f"Precision@{k}: {results[f'precision@{k}']:.4f}")
            if f'ndcg@{k}' in results:
                report.append(f"NDCG@{k}:      {results[f'ndcg@{k}']:.4f}")
        report.append("")
        
        # Other metrics
        report.append("OTHER METRICS:")
        report.append("-" * 15)
        if 'mrr' in results:
            report.append(f"MRR:           {results['mrr']:.4f}")
        if 'coverage' in results:
            report.append(f"Coverage:      {results['coverage']:.4f}")
        if 'diversity' in results:
            report.append(f"Diversity:     {results['diversity']:.4f}")
        if 'novelty' in results:
            report.append(f"Novelty:       {results['novelty']:.4f}")
        report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 8)
        if 'recall@10' in results and 'ndcg@10' in results:
            report.append(f"Primary metrics (K=10):")
            report.append(f"  Recall:  {results['recall@10']:.4f}")
            report.append(f"  NDCG:    {results['ndcg@10']:.4f}")
        
        report.append("=" * 50)
        
        return "\n".join(report)


def main():
    """Main function for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate recommendation system')
    parser.add_argument('--interactions-data', type=str, required=True,
                       help='Path to interactions data file')
    parser.add_argument('--users-data', type=str,
                       help='Path to users data file (optional)')
    parser.add_argument('--items-data', type=str,
                       help='Path to items data file (optional)')
    parser.add_argument('--recommendations', type=str, required=True,
                       help='Path to recommendations file (JSON format)')
    parser.add_argument('--output-dir', type=str, default='evaluation_output',
                       help='Output directory for evaluation results')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10, 20],
                       help='K values for evaluation')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator(args.output_dir)
    
    try:
        # Load data
        interactions_df, users_df, items_df = evaluator.load_data(
            args.interactions_data, args.users_data, args.items_data
        )
        
        # Split data
        train_df, test_df = evaluator.split_data(interactions_df, args.test_ratio)
        
        # Load recommendations
        import json
        with open(args.recommendations, 'r') as f:
            recommendations = json.load(f)
        
        # Evaluate recommendations
        results = evaluator.evaluate_recommendations(
            recommendations, test_df, items_df, len(items_df), args.k_values
        )
        
        # Save results
        evaluator.save_results(results)
        
        # Generate and print report
        report = evaluator.generate_report(results)
        print(report)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()

"""
Data Validation Component for VertexRec Pipeline

This module provides comprehensive data validation using TensorFlow Data Validation (TFDV)
for the recommendation system data pipeline.
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils.display_util import display_schema
from tensorflow_data_validation.utils.display_util import display_statistics
from tensorflow_data_validation.utils.display_util import display_anomalies

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation class using TensorFlow Data Validation."""
    
    def __init__(self, output_dir: str):
        """Initialize the data validator.
        
        Args:
            output_dir: Directory to save validation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TFDV
        tfdv.init_stats()
        
    def generate_statistics(self, data_path: str, file_format: str = 'csv') -> str:
        """Generate statistics for the dataset.
        
        Args:
            data_path: Path to the data file
            file_format: Format of the data file ('csv', 'tfrecord', etc.)
            
        Returns:
            Path to the generated statistics file
        """
        logger.info(f"Generating statistics for {data_path}")
        
        if file_format == 'csv':
            stats = tfdv.generate_statistics_from_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Save statistics
        stats_path = self.output_dir / "statistics.pb"
        tfdv.write_stats_text(stats, str(stats_path))
        
        logger.info(f"Statistics saved to {stats_path}")
        return str(stats_path)
    
    def infer_schema(self, stats_path: str) -> str:
        """Infer schema from statistics.
        
        Args:
            stats_path: Path to the statistics file
            
        Returns:
            Path to the generated schema file
        """
        logger.info("Inferring schema from statistics")
        
        stats = tfdv.load_stats_text(str(stats_path))
        schema = tfdv.infer_schema(stats)
        
        # Save schema
        schema_path = self.output_dir / "schema.pbtxt"
        tfdv.write_schema_text(schema, str(schema_path))
        
        logger.info(f"Schema saved to {schema_path}")
        return str(schema_path)
    
    def validate_data(self, data_path: str, schema_path: str, 
                     file_format: str = 'csv') -> Tuple[bool, str]:
        """Validate data against the schema.
        
        Args:
            data_path: Path to the data file to validate
            schema_path: Path to the schema file
            file_format: Format of the data file
            
        Returns:
            Tuple of (is_valid, anomalies_path)
        """
        logger.info(f"Validating data in {data_path} against schema {schema_path}")
        
        # Load schema
        schema = tfdv.load_schema_text(str(schema_path))
        
        # Generate statistics for validation data
        if file_format == 'csv':
            validation_stats = tfdv.generate_statistics_from_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Validate data
        anomalies = tfdv.validate_statistics(validation_stats, schema)
        
        # Save anomalies
        anomalies_path = self.output_dir / "anomalies.pb"
        tfdv.write_anomalies_text(anomalies, str(anomalies_path))
        
        # Check if validation passed
        is_valid = len(anomalies.anomaly_info) == 0
        
        if is_valid:
            logger.info("Data validation passed - no anomalies found")
        else:
            logger.warning(f"Data validation found {len(anomalies.anomaly_info)} anomalies")
            
        return is_valid, str(anomalies_path)
    
    def update_schema(self, schema_path: str, data_path: str) -> str:
        """Update schema based on new data.
        
        Args:
            schema_path: Path to existing schema
            data_path: Path to new data
            
        Returns:
            Path to updated schema
        """
        logger.info("Updating schema based on new data")
        
        # Load existing schema
        schema = tfdv.load_schema_text(str(schema_path))
        
        # Generate statistics for new data
        new_stats = tfdv.generate_statistics_from_csv(data_path)
        
        # Update schema
        updated_schema = tfdv.update_schema(
            schema, 
            new_stats,
            infer_feature_shape=True
        )
        
        # Save updated schema
        updated_schema_path = self.output_dir / "updated_schema.pbtxt"
        tfdv.write_schema_text(updated_schema, str(updated_schema_path))
        
        logger.info(f"Updated schema saved to {updated_schema_path}")
        return str(updated_schema_path)
    
    def create_baseline_schema(self, data_path: str, 
                              custom_constraints: Optional[Dict] = None) -> str:
        """Create a baseline schema with custom constraints.
        
        Args:
            data_path: Path to the baseline data
            custom_constraints: Custom validation constraints
            
        Returns:
            Path to the baseline schema
        """
        logger.info("Creating baseline schema with custom constraints")
        
        # Generate statistics
        stats_path = self.generate_statistics(data_path)
        
        # Infer initial schema
        schema_path = self.infer_schema(stats_path)
        
        # Load schema
        schema = tfdv.load_schema_text(str(schema_path))
        
        # Apply custom constraints if provided
        if custom_constraints:
            self._apply_constraints(schema, custom_constraints)
        
        # Save baseline schema
        baseline_schema_path = self.output_dir / "baseline_schema.pbtxt"
        tfdv.write_schema_text(schema, str(baseline_schema_path))
        
        logger.info(f"Baseline schema saved to {baseline_schema_path}")
        return str(baseline_schema_path)
    
    def _apply_constraints(self, schema, constraints: Dict):
        """Apply custom constraints to the schema.
        
        Args:
            schema: TFDV schema object
            constraints: Dictionary of constraints to apply
        """
        for feature_name, constraint_config in constraints.items():
            if feature_name not in [f.name for f in schema.feature]:
                logger.warning(f"Feature {feature_name} not found in schema")
                continue
                
            # Find the feature
            feature = None
            for f in schema.feature:
                if f.name == feature_name:
                    feature = f
                    break
            
            if feature is None:
                continue
            
            # Apply constraints based on type
            if 'domain' in constraint_config:
                # Set domain constraints
                domain = constraint_config['domain']
                if domain == 'categorical':
                    feature.domain_info.ints_is_categorical = True
                elif domain == 'numeric':
                    feature.domain_info.ints_is_categorical = False
            
            if 'min_value' in constraint_config:
                feature.domain_info.min_domain = constraint_config['min_value']
            
            if 'max_value' in constraint_config:
                feature.domain_info.max_domain = constraint_config['max_value']
            
            if 'allowed_values' in constraint_config:
                feature.domain_info.string_domain.value.extend(constraint_config['allowed_values'])
            
            if 'presence_required' in constraint_config:
                feature.presence.min_fraction = 1.0 if constraint_config['presence_required'] else 0.0
    
    def validate_interactions_data(self, data_path: str) -> Tuple[bool, str]:
        """Validate interactions data with specific constraints.
        
        Args:
            data_path: Path to interactions data
            
        Returns:
            Tuple of (is_valid, anomalies_path)
        """
        logger.info("Validating interactions data with specific constraints")
        
        # Define constraints for interactions data
        constraints = {
            'user_id': {
                'presence_required': True,
                'domain': 'categorical'
            },
            'item_id': {
                'presence_required': True,
                'domain': 'categorical'
            },
            'interaction_type': {
                'presence_required': False,
                'allowed_values': ['view', 'like', 'watch', 'purchase', 'share']
            },
            'rating': {
                'presence_required': False,
                'min_value': 1.0,
                'max_value': 5.0
            },
            'timestamp': {
                'presence_required': False
            }
        }
        
        # Create baseline schema
        baseline_schema_path = self.create_baseline_schema(data_path, constraints)
        
        # Validate data
        return self.validate_data(data_path, baseline_schema_path)
    
    def validate_users_data(self, data_path: str) -> Tuple[bool, str]:
        """Validate users data with specific constraints.
        
        Args:
            data_path: Path to users data
            
        Returns:
            Tuple of (is_valid, anomalies_path)
        """
        logger.info("Validating users data with specific constraints")
        
        # Define constraints for users data
        constraints = {
            'user_id': {
                'presence_required': True,
                'domain': 'categorical'
            },
            'age_group': {
                'presence_required': False,
                'allowed_values': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            },
            'gender': {
                'presence_required': False,
                'allowed_values': ['M', 'F', 'Other']
            },
            'location': {
                'presence_required': False,
                'domain': 'categorical'
            },
            'subscription_type': {
                'presence_required': False,
                'allowed_values': ['free', 'premium']
            }
        }
        
        # Create baseline schema
        baseline_schema_path = self.create_baseline_schema(data_path, constraints)
        
        # Validate data
        return self.validate_data(data_path, baseline_schema_path)
    
    def validate_items_data(self, data_path: str) -> Tuple[bool, str]:
        """Validate items data with specific constraints.
        
        Args:
            data_path: Path to items data
            
        Returns:
            Tuple of (is_valid, anomalies_path)
        """
        logger.info("Validating items data with specific constraints")
        
        # Define constraints for items data
        constraints = {
            'item_id': {
                'presence_required': True,
                'domain': 'categorical'
            },
            'title': {
                'presence_required': False
            },
            'genre': {
                'presence_required': False,
                'domain': 'categorical'
            },
            'popularity_score': {
                'presence_required': False,
                'min_value': 0.0,
                'max_value': 1.0
            },
            'quality_score': {
                'presence_required': False,
                'min_value': 0.0,
                'max_value': 1.0
            },
            'price': {
                'presence_required': False,
                'min_value': 0.0
            },
            'rating_avg': {
                'presence_required': False,
                'min_value': 0.0,
                'max_value': 5.0
            }
        }
        
        # Create baseline schema
        baseline_schema_path = self.create_baseline_schema(data_path, constraints)
        
        # Validate data
        return self.validate_data(data_path, baseline_schema_path)
    
    def generate_validation_report(self, anomalies_paths: List[str]) -> str:
        """Generate a comprehensive validation report.
        
        Args:
            anomalies_paths: List of paths to anomaly files
            
        Returns:
            Path to the generated report
        """
        logger.info("Generating validation report")
        
        report_path = self.output_dir / "validation_report.html"
        
        with open(report_path, 'w') as f:
            f.write("<html><head><title>VertexRec Data Validation Report</title></head><body>")
            f.write("<h1>VertexRec Data Validation Report</h1>")
            
            for i, anomalies_path in enumerate(anomalies_paths):
                if Path(anomalies_path).exists():
                    anomalies = tfdv.load_anomalies_text(str(anomalies_path))
                    
                    f.write(f"<h2>Dataset {i+1} Anomalies</h2>")
                    
                    if len(anomalies.anomaly_info) == 0:
                        f.write("<p>✅ No anomalies found</p>")
                    else:
                        f.write(f"<p>❌ Found {len(anomalies.anomaly_info)} anomalies:</p>")
                        f.write("<ul>")
                        
                        for feature_name, anomaly_info in anomalies.anomaly_info.items():
                            f.write(f"<li><strong>{feature_name}:</strong> {anomaly_info.description}</li>")
                        
                        f.write("</ul>")
            
            f.write("</body></html>")
        
        logger.info(f"Validation report saved to {report_path}")
        return str(report_path)


def main():
    """Main function for data validation."""
    parser = argparse.ArgumentParser(description='Validate recommendation system data')
    parser.add_argument('--input-data', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--schema-file', type=str,
                       help='Path to schema file (optional)')
    parser.add_argument('--output-dir', type=str, default='validation_output',
                       help='Output directory for validation results')
    parser.add_argument('--data-type', type=str, choices=['users', 'items', 'interactions'],
                       help='Type of data being validated')
    parser.add_argument('--create-baseline', action='store_true',
                       help='Create baseline schema from data')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = DataValidator(args.output_dir)
    
    try:
        if args.create_baseline:
            # Create baseline schema
            if args.data_type == 'interactions':
                baseline_schema_path = validator.create_baseline_schema(args.input_data)
            elif args.data_type == 'users':
                baseline_schema_path = validator.create_baseline_schema(args.input_data)
            elif args.data_type == 'items':
                baseline_schema_path = validator.create_baseline_schema(args.input_data)
            else:
                baseline_schema_path = validator.create_baseline_schema(args.input_data)
            
            print(f"Baseline schema created: {baseline_schema_path}")
        
        elif args.schema_file:
            # Validate against existing schema
            is_valid, anomalies_path = validator.validate_data(args.input_data, args.schema_file)
            
            if is_valid:
                print("✅ Data validation passed")
            else:
                print(f"❌ Data validation failed - see anomalies: {anomalies_path}")
        
        else:
            # Validate with data-specific constraints
            if args.data_type == 'interactions':
                is_valid, anomalies_path = validator.validate_interactions_data(args.input_data)
            elif args.data_type == 'users':
                is_valid, anomalies_path = validator.validate_users_data(args.input_data)
            elif args.data_type == 'items':
                is_valid, anomalies_path = validator.validate_items_data(args.input_data)
            else:
                raise ValueError("Data type must be specified for validation")
            
            if is_valid:
                print("✅ Data validation passed")
            else:
                print(f"❌ Data validation failed - see anomalies: {anomalies_path}")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


if __name__ == '__main__':
    main()

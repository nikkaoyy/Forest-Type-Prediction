"""
Preprocessing Utilities
Helper functions for loading and using preprocessor artifacts
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessorLoader:
    """
    Utility class for loading and using preprocessor artifacts
    """
    
    def __init__(self, preprocessor_path: str = 'data/artifacts/preprocessor.pkl'):
        """
        Initialize preprocessor loader
        
        Args:
            preprocessor_path: Path to preprocessor.pkl file
        """
        self.preprocessor_path = preprocessor_path
        self.preprocessor = None
        self.metadata = None
        
    def load(self) -> Dict[str, Any]:
        """
        Load preprocessor from disk
        
        Returns:
            Dictionary containing pipeline and metadata
        """
        try:
            with open(self.preprocessor_path, 'rb') as f:
                data = pickle.load(f)
            
            self.preprocessor = data['pipeline']
            self.metadata = data.get('metadata', {})
            
            logger.info(f"✓ Preprocessor loaded from: {self.preprocessor_path}")
            logger.info(f"  Features: {data['n_features_in']} → {data['n_features_out']}")
            
            return data
            
        except FileNotFoundError:
            logger.error(f" Preprocessor not found at: {self.preprocessor_path}")
            logger.error("   Run 'python generate_preprocessor.py' to create it")
            raise
        except Exception as e:
            logger.error(f" Error loading preprocessor: {e}")
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data using loaded preprocessor
        
        Args:
            X: Input DataFrame with raw features
            
        Returns:
            Transformed DataFrame
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded. Call load() first.")
        
        if not self.preprocessor.is_fitted:
            raise ValueError("Preprocessor is not fitted. Generate a new artifact.")
        
        # Store 'Id' column if present (will be excluded from scaling)
        has_id = 'Id' in X.columns
        if has_id:
            id_col = X['Id'].copy()
        
        # Apply transformation
        X_transformed = self.preprocessor.elevation_processor.transform(X)
        X_transformed = self.preprocessor.aspect_transformer.transform(X_transformed)
        X_transformed = self.preprocessor.soil_consolidator.transform(X_transformed)
        
        # Apply normalization if scaler exists
        if self.preprocessor.scaler is not None:
            # Get numerical columns excluding 'Id'
            numerical_cols = [col for col in X_transformed.select_dtypes(include=[np.number]).columns 
                            if col != 'Id']
            
            if len(numerical_cols) > 0:
                # Convert to numpy to avoid sklearn's feature name validation
                numerical_data = X_transformed[numerical_cols].values
                scaled_data = self.preprocessor.scaler.transform(numerical_data)
                X_transformed[numerical_cols] = scaled_data
        
        # Restore 'Id' column if it was present
        if has_id:
            X_transformed['Id'] = id_col
        
        return X_transformed
    
    def validate_input(self, X: pd.DataFrame, ignore_id: bool = True) -> pd.DataFrame:
        """
        Validate that input data has correct schema
        
        Args:
            X: Input DataFrame
            ignore_id: If True, 'Id' column is optional (for inference)
            
        Returns:
            DataFrame with validated and reordered columns
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded. Call load() first.")
        
        # Get expected features from loaded metadata
        with open(self.preprocessor_path, 'rb') as f:
            data = pickle.load(f)
        
        expected_features = data['feature_names_in']
        actual_features = X.columns.tolist()
        
        # Filter out 'Id' if requested (common for inference)
        if ignore_id and 'Id' in expected_features:
            expected_features = [f for f in expected_features if f != 'Id']
        
        # Check if all expected features are present
        missing = set(expected_features) - set(actual_features)
        extra = set(actual_features) - set(expected_features)
        
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")
        
        if extra:
            logger.warning(f"Extra features will be ignored: {sorted(extra)}")
        
        # Reorder columns to match expected order
        X_validated = X[expected_features].copy()
        
        logger.info("✓ Input validation passed")
        return X_validated
    
    def get_feature_names(self) -> Dict[str, list]:
        """
        Get input and output feature names
        
        Returns:
            Dictionary with 'input' and 'output' feature lists
        """
        if self.preprocessor is None:
            self.load()
        
        with open(self.preprocessor_path, 'rb') as f:
            data = pickle.load(f)
        
        return {
            'input': data['feature_names_in'],
            'output': data['feature_names_out']
        }
    
    def get_chaos_metadata(self) -> Dict:
        """
        Get chaos detection metadata
        
        Returns:
            Dictionary with threshold information
        """
        if self.metadata is None:
            self.load()
        
        return self.metadata


def preprocess_for_inference(X: pd.DataFrame, 
                            preprocessor_path: str = 'data/artifacts/preprocessor.pkl',
                            ignore_id: bool = True) -> pd.DataFrame:
    """
    Convenience function to preprocess data for inference
    
    Args:
        X: Raw input DataFrame
        preprocessor_path: Path to preprocessor artifact
        ignore_id: If True, 'Id' column is optional
        
    Returns:
        Transformed DataFrame ready for model inference
    """
    loader = PreprocessorLoader(preprocessor_path)
    loader.load()
    X_validated = loader.validate_input(X, ignore_id=ignore_id)
    X_transformed = loader.transform(X_validated)
    
    return X_transformed


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("PREPROCESSOR LOADER DEMO")
    print("=" * 70)
    
    # Check if preprocessor exists
    import os
    preprocessor_path = 'data/artifacts/preprocessor.pkl'
    
    if not os.path.exists(preprocessor_path):
        print(f" Preprocessor not found at: {preprocessor_path}")
        print("   Run: python generate_preprocessor.py")
        exit(1)
    
    # Load preprocessor
    loader = PreprocessorLoader(preprocessor_path)
    data = loader.load()
    
    print("\n" + "=" * 70)
    print("PREPROCESSOR INFO")
    print("=" * 70)
    print(f"Input features: {data['n_features_in']}")
    print(f"Output features: {data['n_features_out']}")
    print(f"Is fitted: {data['is_fitted']}")
    
    # Get feature names
    features = loader.get_feature_names()
    print(f"\nFirst 5 input features: {features['input'][:5]}")
    print(f"First 5 output features: {features['output'][:5]}")
    
    # Get chaos metadata
    chaos_meta = loader.get_chaos_metadata()
    print(f"\nChaos detection settings:")
    print(f"  Thresholds: {chaos_meta['elevation_thresholds']}")
    print(f"  Proximity window: ±{chaos_meta['proximity_window']}m")
    print(f"  Amplification factor: {chaos_meta['uncertainty_amplification']}x")
    
    # Test transformation with sample data
    print("\n" + "=" * 70)
    print("TESTING TRANSFORMATION")
    print("=" * 70)
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Elevation': [2500, 2750, 3100],
        'Aspect': [90, 180, 270],
        'Slope': [15, 20, 25],
        'Horizontal_Distance_To_Hydrology': [100, 200, 300],
        'Vertical_Distance_To_Hydrology': [-50, 0, 50],
        'Horizontal_Distance_To_Roadways': [500, 1000, 1500],
        'Horizontal_Distance_To_Fire_Points': [600, 1200, 1800],
        'Hillshade_9am': [100, 150, 200],
        'Hillshade_Noon': [200, 220, 240],
        'Hillshade_3pm': [100, 120, 140],
        **{f'Wilderness_Area{i}': [1, 0, 0] for i in range(1, 5)},
        **{f'Soil_Type{i}': [0, 1, 0] for i in range(1, 41)}
    })
    
    print(f"Sample input shape: {sample_data.shape}")
    
    # Validate and transform (ignore 'Id' for this demo)
    sample_data = loader.validate_input(sample_data, ignore_id=True)
    transformed = loader.transform(sample_data)
    
    print(f"Transformed shape: {transformed.shape}")
    print(f"\nNew features created:")
    new_features = set(transformed.columns) - set(sample_data.columns)
    for feat in sorted(list(new_features)[:10]):  # Show first 10
        print(f"  - {feat}")
    
    print("\n Demo complete!")
"""
Generate Preprocessor Pipeline
Creates and saves preprocessor.pkl artifact with complete feature engineering
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.featureEngineer import FeatureEngineeringPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Generate and save preprocessing pipeline"""
    
    print("=" * 70)
    print("PREPROCESSING PIPELINE GENERATOR")
    print("=" * 70)
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "src" / "data"
    raw_dir = data_dir / "raw"
    artifacts_dir = data_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    train_file = raw_dir / "train.csv"
    
    if not train_file.exists():
        logger.error(f" Training data not found: {train_file}")
        logger.info(f"   Current directory: {Path.cwd()}")
        logger.info(f"   Looking for: {train_file.absolute()}")
        logger.info("   Please ensure train.csv is in: src/data/raw/")
        return
    
    logger.info(f"\n Loading data from: {train_file}")
    df = pd.read_csv(train_file)
    logger.info(f"✓ Loaded {len(df)} samples")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['Id', 'Cover_Type']]
    X_full = df[feature_cols].copy()
    
    logger.info(f"✓ Features: {X_full.shape[1]} columns")
    
    # Use subset for fitting (faster, but fit on full dataset is better)
    use_full_data = True  # Set to False for faster testing
    
    if use_full_data:
        X_sample = X_full
        logger.info("  Using FULL dataset for fitting")
    else:
        X_sample = X_full.head(5000).copy()
        logger.info(f"  Using sample: {X_sample.shape} for fitting")
    
    # Initialize pipeline with normalization
    logger.info("\n Initializing feature engineering pipeline...")
    pipeline = FeatureEngineeringPipeline(normalize=True)
    
    # Fit and transform
    logger.info("\n Fitting pipeline on training data...")
    X_transformed = pipeline.fit_transform(X_sample)
    
    logger.info(f"\n✓ Pipeline fitted successfully!")
    logger.info(f"  Input features: {X_sample.shape[1]}")
    logger.info(f"  Output features: {X_transformed.shape[1]}")
    logger.info(f"  Net change: {X_transformed.shape[1] - X_sample.shape[1]:+d} features")
    
    # Get chaos statistics
    chaos_stats = pipeline.get_chaos_statistics(X_transformed)
    logger.info(f"\n Chaos Zone Statistics:")
    logger.info(f"  Total observations: {chaos_stats['total_observations']}")
    logger.info(f"  In chaos zones: {chaos_stats['chaos_zone_count']} ({chaos_stats['chaos_zone_percentage']:.2f}%)")
    
    # Prepare serialization data
    logger.info("\n Preparing serialization...")
    
    # Store complete feature information
    preprocessor_data = {
        'pipeline': pipeline,
        'n_features_in': X_sample.shape[1],
        'n_features_out': X_transformed.shape[1],
        'feature_names_in': X_sample.columns.tolist(),
        'feature_names_out': X_transformed.columns.tolist(),
        'is_fitted': True,
        'metadata': {
            'elevation_thresholds': pipeline.elevation_processor.THRESHOLDS,
            'proximity_window': pipeline.elevation_processor.PROXIMITY_WINDOW,
            'uncertainty_amplification': pipeline.elevation_processor.UNCERTAINTY_AMPLIFICATION,
            'normalization_applied': pipeline.normalize,
            'timestamp': pd.Timestamp.now().isoformat(),
            'training_samples': len(X_sample),
            'chaos_statistics': chaos_stats
        }
    }
    
    # Add scaler information if present
    if pipeline.scaler is not None:
        preprocessor_data['scaler_n_features'] = pipeline.scaler.n_features_in_
        preprocessor_data['scaler_feature_names'] = pipeline.scaler.feature_names_in_.tolist() if hasattr(pipeline.scaler, 'feature_names_in_') else None
    
    # Save to disk
    output_path = artifacts_dir / "preprocessor.pkl"
    
    with open(output_path, 'wb') as f:
        pickle.dump(preprocessor_data, f)
    
    logger.info(f"\n Preprocessor saved to: {output_path}")
    logger.info(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
    
    # Verification: Load and test
    logger.info("\n Verification: Loading and testing...")
    
    with open(output_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    loaded_pipeline = loaded_data['pipeline']
    
    # Test transform on a small sample
    test_sample = X_sample.head(10).copy()
    test_transformed = loaded_pipeline.elevation_processor.transform(test_sample)
    test_transformed = loaded_pipeline.aspect_transformer.transform(test_transformed)
    test_transformed = loaded_pipeline.soil_consolidator.transform(test_transformed)
    
    if loaded_pipeline.scaler is not None:
        numerical_cols = test_transformed.select_dtypes(include=[np.number]).columns.tolist()
        numerical_data = test_transformed[numerical_cols].values
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            scaled_data = loaded_pipeline.scaler.transform(numerical_data)
            test_transformed[numerical_cols] = scaled_data
    
    logger.info(f"✓ Transform test passed: {test_sample.shape} → {test_transformed.shape}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PREPROCESSING PIPELINE SUMMARY")
    print("=" * 70)
    print(f"\n Input:")
    print(f"   Features: {preprocessor_data['n_features_in']}")
    print(f"   Sample columns: {preprocessor_data['feature_names_in'][:5]}...")
    
    print(f"\n Output:")
    print(f"   Features: {preprocessor_data['n_features_out']}")
    print(f"   Sample columns: {preprocessor_data['feature_names_out'][:5]}...")
    
    print(f"\n Transformations Applied:")
    print(f"   ✓ Elevation binning + threshold detection")
    print(f"   ✓ Aspect sin/cos encoding")
    print(f"   ✓ Soil type consolidation")
    print(f"   ✓ Numerical scaling: {'YES' if pipeline.normalize else 'NO'}")
    
    print(f"\n  Chaos Detection:")
    print(f"   Thresholds: {preprocessor_data['metadata']['elevation_thresholds']} m")
    print(f"   Window: ±{preprocessor_data['metadata']['proximity_window']} m")
    print(f"   Amplification: {preprocessor_data['metadata']['uncertainty_amplification']}×")
    
    print(f"\n Artifact:")
    print(f"   Path: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"   Fitted: {preprocessor_data['is_fitted']}")
    
    print("\n" + "=" * 70)
    print(" PREPROCESSING PIPELINE READY!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run: python src/generate_ensemble_models.py")
    print("  2. Or use in your code:")
    print("     from preprocessing import PreprocessorLoader")
    print("     loader = PreprocessorLoader('src/data/artifacts/preprocessor.pkl')")
    print("     X_transformed = loader.transform(X_raw)")
    

if __name__ == "__main__":
    main()

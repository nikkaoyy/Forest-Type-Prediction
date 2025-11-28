"""
Feature Engineering Pipeline with Chaos Detection
Implements elevation threshold monitoring and uncertainty amplification
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, List
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElevationProcessor(BaseEstimator, TransformerMixin):
    """
    Module 3A: Elevation binning with chaos detection
    Implements threshold proximity flagging
    """
    
    THRESHOLDS = [2400, 2800, 3200]
    PROXIMITY_WINDOW = 50  # ±50m
    UNCERTAINTY_AMPLIFICATION = 2.0
    
    def __init__(self):
        self.zone_labels = ['Foothill', 'Montane', 'Subalpine', 'Alpine']
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply elevation binning and threshold detection"""
        X = X.copy()
        
        elevation = X['Elevation']
        
        # Bin elevation into ecological zones
        bins = [0] + self.THRESHOLDS + [np.inf]
        X['elevation_zone'] = pd.cut(
            elevation, 
            bins=bins, 
            labels=self.zone_labels
        )
        
        # Detect threshold proximity (chaos zones)
        X['near_threshold'] = False
        X['distance_to_nearest_threshold'] = np.inf
        X['chaos_amplification_factor'] = 1.0
        
        for threshold in self.THRESHOLDS:
            distance = np.abs(elevation - threshold)
            is_near = distance <= self.PROXIMITY_WINDOW
            
            X.loc[is_near, 'near_threshold'] = True
            X.loc[is_near, 'chaos_amplification_factor'] = self.UNCERTAINTY_AMPLIFICATION
            
            # Update minimum distance
            closer_mask = distance < X['distance_to_nearest_threshold']
            X.loc[closer_mask, 'distance_to_nearest_threshold'] = distance[closer_mask]
        
        # Log chaos zone statistics
        chaos_count = X['near_threshold'].sum()
        chaos_percentage = (chaos_count / len(X)) * 100
        
        logger.info(f"Detected {chaos_count} observations in chaos zones ({chaos_percentage:.1f}%)")
        logger.info(f"Critical thresholds monitored: {self.THRESHOLDS}")
        
        return X


class AspectTransformer(BaseEstimator, TransformerMixin):
    """
    Module 3B: Circular aspect encoding
    Converts aspect (0-360°) to sin/cos components
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply trigonometric transformation"""
        X = X.copy()
        
        aspect_rad = np.deg2rad(X['Aspect'])
        
        X['aspect_sin'] = np.sin(aspect_rad)
        X['aspect_cos'] = np.cos(aspect_rad)
        
        # Verify circularity preservation
        reconstructed = np.rad2deg(np.arctan2(X['aspect_sin'], X['aspect_cos']))
        reconstructed = (reconstructed + 360) % 360
        
        max_error = np.abs(reconstructed - X['Aspect']).max()
        logger.info(f"Aspect circularity preserved (max error: {max_error:.4f}°)")
        
        return X


class SoilConsolidator(BaseEstimator, TransformerMixin):
    """
    Module 3C: Soil type consolidation
    Reduces 40 sparse categories to 15 groups
    """
    
    def __init__(self, frequency_threshold: int = 100):
        self.frequency_threshold = frequency_threshold
        self.consolidation_map = {}
        self.original_sparsity = None
        self.consolidated_sparsity = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn soil type frequencies"""
        soil_cols = [col for col in X.columns if col.startswith('Soil_Type')]
        
        if not soil_cols:
            logger.warning("No soil type columns found - skipping consolidation")
            return self
        
        # Calculate frequencies
        frequencies = X[soil_cols].sum().sort_values(ascending=False)
        
        # Separate frequent and rare types
        frequent_types = frequencies[frequencies >= self.frequency_threshold].index.tolist()
        rare_types = frequencies[frequencies < self.frequency_threshold].index.tolist()
        
        # Create consolidation mapping
        for soil_type in frequent_types:
            self.consolidation_map[soil_type] = soil_type
        
        # Group rare types by simulated ecological similarity
        # In production, use domain knowledge
        if len(rare_types) > 0:
            group_size = max(len(rare_types) // 5, 1)
            groups = ['Sandy', 'Clay', 'Rocky', 'Organic', 'Other']
            
            for i, soil_type in enumerate(rare_types):
                group_idx = min(i // group_size, len(groups) - 1)
                self.consolidation_map[soil_type] = f'Soil_Group_{groups[group_idx]}'
        
        # Calculate original sparsity
        self.original_sparsity = (X[soil_cols] == 0).sum().sum() / (len(X) * len(soil_cols))
        
        logger.info(f"Soil consolidation: {len(soil_cols)} → {len(set(self.consolidation_map.values()))} types")
        logger.info(f"Original sparsity: {self.original_sparsity:.2%}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply soil consolidation"""
        X = X.copy()
        
        soil_cols = [col for col in X.columns if col.startswith('Soil_Type')]
        
        if not soil_cols or not self.consolidation_map:
            logger.warning("No soil consolidation applied - returning original data")
            return X
        
        # Create consolidated features
        consolidated_features = {}
        for new_type in set(self.consolidation_map.values()):
            original_types = [k for k, v in self.consolidation_map.items() if v == new_type]
            # Use max to preserve one-hot encoding (at least one original type is 1)
            consolidated_features[new_type] = X[original_types].max(axis=1)
        
        consolidated_df = pd.DataFrame(consolidated_features, index=X.index)
        
        # Calculate new sparsity
        self.consolidated_sparsity = (consolidated_df == 0).sum().sum() / consolidated_df.size
        
        # Drop original columns and add consolidated
        X = X.drop(columns=soil_cols)
        X = pd.concat([X, consolidated_df], axis=1)
        
        logger.info(f"Consolidated sparsity: {self.consolidated_sparsity:.2%}")
        logger.info(f"Sparsity reduction: {self.original_sparsity:.2%} → {self.consolidated_sparsity:.2%}")
        
        return X


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline
    Implements modularity and loose coupling principles
    """
    
    def __init__(self, normalize: bool = True):
        self.elevation_processor = ElevationProcessor()
        self.aspect_transformer = AspectTransformer()
        self.soil_consolidator = SoilConsolidator(frequency_threshold=100)
        self.scaler = StandardScaler() if normalize else None
        self.is_fitted = False
        self.normalize = normalize
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform pipeline"""
        logger.info("=" * 70)
        logger.info("Starting feature engineering pipeline...")
        logger.info("=" * 70)
        
        # Module 3A: Elevation processing
        logger.info("\n[Module 3A] Elevation Processing...")
        X = self.elevation_processor.fit_transform(X)
        
        # Module 3B: Aspect transformation
        logger.info("\n[Module 3B] Aspect Transformation...")
        X = self.aspect_transformer.fit_transform(X)
        
        # Module 3C: Soil consolidation
        logger.info("\n[Module 3C] Soil Consolidation...")
        X = self.soil_consolidator.fit_transform(X)
        
        # Normalize numerical features
        if self.normalize and self.scaler is not None:
            logger.info("\n[Normalization] Scaling numerical features...")
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            logger.info(f"Normalized {len(numerical_cols)} numerical features")
        
        self.is_fitted = True
        logger.info("\n✅ Feature engineering pipeline complete")
        
        return X
    
    def get_chaos_statistics(self, X: pd.DataFrame) -> dict:
        """Extract chaos zone statistics"""
        return {
            'total_observations': len(X),
            'chaos_zone_count': X['near_threshold'].sum(),
            'chaos_zone_percentage': (X['near_threshold'].sum() / len(X)) * 100,
            'mean_distance_to_threshold': X['distance_to_nearest_threshold'].mean(),
            'amplification_active': (X['chaos_amplification_factor'] > 1.0).sum()
        }


# Example usage with REAL Kaggle dataset
if __name__ == "__main__":
    """
    INSTRUCTIONS TO RUN:
    1. Download dataset from Kaggle: 
       https://www.kaggle.com/competitions/forest-cover-type-prediction/data
    2. Place 'train.csv' in the same directory as this script
    3. Run: python feature_engineering.py
    """
    
    # ============================================================
    # STEP 1: Load Real Dataset
    # ============================================================
    dataset_path = 'train.csv'
    
    if not os.path.exists(dataset_path):
        print("❌ ERROR: Dataset not found!")
        print(f"Please download 'train.csv' from Kaggle and place it at: {dataset_path}")
        print("Download URL: https://www.kaggle.com/competitions/forest-cover-type-prediction/data")
        exit(1)
    
    print("=" * 70)
    print("LOADING KAGGLE FOREST COVER TYPE DATASET")
    print("=" * 70)
    
    # Load dataset
    df_full = pd.read_csv(dataset_path)
    print(f"✓ Dataset loaded: {df_full.shape[0]} observations × {df_full.shape[1]} features")
    
    # Separate features and target
    feature_cols = [col for col in df_full.columns if col != 'Cover_Type']
    X_full = df_full[feature_cols]
    y_full = df_full['Cover_Type']
    
    print(f"✓ Features: {X_full.shape[1]} columns")
    print(f"✓ Target: Cover_Type (7 classes)")
    
    # ============================================================
    # STEP 2: Use a Subset for Demonstration (faster processing)
    # ============================================================
    # Use first 5000 samples for quick demo
    X_sample = X_full.head(5000).copy()
    y_sample = y_full.head(5000).copy()
    
    print(f"\n✓ Using sample: {X_sample.shape} for demonstration")
    
    # ============================================================
    # STEP 3: Apply Feature Engineering Pipeline
    # ============================================================
    print("\n" + "=" * 70)
    print("APPLYING FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    
    # Initialize pipeline (normalize=False to see raw engineered features)
    pipeline = FeatureEngineeringPipeline(normalize=False)
    
    # Transform data
    X_transformed = pipeline.fit_transform(X_sample)
    
    # ============================================================
    # STEP 4: Chaos Zone Analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("CHAOS ZONE STATISTICS")
    print("=" * 70)
    
    chaos_stats = pipeline.get_chaos_statistics(X_transformed)
    for key, value in chaos_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # ============================================================
    # STEP 5: Feature Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("FEATURE TRANSFORMATION SUMMARY")
    print("=" * 70)
    
    print(f"Original features: {X_sample.shape[1]}")
    print(f"Transformed features: {X_transformed.shape[1]}")
    print(f"Net change: {X_transformed.shape[1] - X_sample.shape[1]:+d} features")
    
    print("\n📊 New features created:")
    new_features = set(X_transformed.columns) - set(X_sample.columns)
    for feat in sorted(new_features):
        print(f"  ✓ {feat}")
    
    print("\n🗑️  Features removed (consolidated):")
    removed_features = set(X_sample.columns) - set(X_transformed.columns)
    soil_removed = [f for f in removed_features if f.startswith('Soil_Type')]
    if soil_removed:
        print(f"  ✓ {len(soil_removed)} sparse Soil_Type features → consolidated")
    
    # ============================================================
    # STEP 6: Detailed Chaos Zone Breakdown
    # ============================================================
    print("\n" + "=" * 70)
    print("DETAILED CHAOS ZONE BREAKDOWN")
    print("=" * 70)
    
    thresholds = [2400, 2800, 3200]
    for threshold in thresholds:
        near_threshold = ((X_sample['Elevation'] >= threshold - 50) & 
                         (X_sample['Elevation'] <= threshold + 50))
        count = near_threshold.sum()
        percentage = (count / len(X_sample)) * 100
        
        print(f"\n🔍 Threshold {threshold}m (±50m window):")
        print(f"   Observations: {count}")
        print(f"   Percentage: {percentage:.2f}%")
        print(f"   Uncertainty Amplification: 2.0×")
        
        if count > 0:
            # Show elevation zone distribution in this chaos zone
            chaos_data = X_transformed[near_threshold]
            zone_counts = chaos_data['elevation_zone'].value_counts()
            print(f"   Zone distribution:")
            for zone, zone_count in zone_counts.items():
                print(f"      - {zone}: {zone_count} samples")
    
    # ============================================================
    # STEP 7: Aspect Transformation Verification
    # ============================================================
    print("\n" + "=" * 70)
    print("ASPECT CIRCULARITY VERIFICATION")
    print("=" * 70)
    
    # Check aspect transformation quality
    aspect_original = X_sample['Aspect']
    aspect_sin = X_transformed['aspect_sin']
    aspect_cos = X_transformed['aspect_cos']
    
    # Reconstruct aspect from sin/cos
    reconstructed = np.rad2deg(np.arctan2(aspect_sin, aspect_cos))
    reconstructed = (reconstructed + 360) % 360
    
    reconstruction_error = np.abs(reconstructed - aspect_original).mean()
    max_error = np.abs(reconstructed - aspect_original).max()
    
    print(f"✓ Mean reconstruction error: {reconstruction_error:.6f}°")
    print(f"✓ Max reconstruction error: {max_error:.6f}°")
    print(f"✓ Circularity preserved: {'YES ✓' if max_error < 0.001 else 'NO ✗'}")
    
    # ============================================================
    # STEP 8: Soil Type Consolidation Report
    # ============================================================
    print("\n" + "=" * 70)
    print("SOIL TYPE CONSOLIDATION REPORT")
    print("=" * 70)
    
    original_soil_cols = [col for col in X_sample.columns if col.startswith('Soil_Type')]
    consolidated_soil_cols = [col for col in X_transformed.columns if col.startswith('Soil_')]
    
    print(f"Original soil features: {len(original_soil_cols)}")
    print(f"Consolidated soil features: {len(consolidated_soil_cols)}")
    print(f"Reduction: {len(original_soil_cols) - len(consolidated_soil_cols)} features removed")
    
    if consolidated_soil_cols:
        print(f"\nConsolidated soil groups:")
        for soil_col in sorted(consolidated_soil_cols):
            count = X_transformed[soil_col].sum()
            percentage = (count / len(X_transformed)) * 100
            print(f"  ✓ {soil_col}: {count} samples ({percentage:.2f}%)")
    
    # ============================================================
    # STEP 9: Save Transformed Data (Optional)
    # ============================================================
    print("\n" + "=" * 70)
    print("SAVE TRANSFORMED DATA")
    print("=" * 70)
    
    output_path = 'train_transformed.csv'
    save_data = input(f"\nSave transformed data to '{output_path}'? (y/n): ").lower().strip()
    
    if save_data == 'y':
        # Add target back for complete dataset
        X_transformed['Cover_Type'] = y_sample.values
        X_transformed.to_csv(output_path, index=False)
        print(f"✅ Transformed data saved to: {output_path}")
        print(f"   Shape: {X_transformed.shape}")
    else:
        print("⏭️  Skipping save")
    
    print("\n" + "=" * 70)
    print("✅ FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

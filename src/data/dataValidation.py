"""
Data Validation Module with Circuit Breaker Pattern
Implements fault-tolerant schema validation and drift detection
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ValidationResult:
    """Validation result container"""
    is_valid: bool
    errors: list
    warnings: list
    metrics: Dict[str, float]


class CircuitBreaker:
    """Circuit breaker for fault isolation"""
    
    def __init__(self, failure_threshold: int = 3, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN - rejecting request")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Reset on successful execution"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Increment failure count"""
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if timeout expired"""
        return True  # Simplified for demo


class ForestDataValidator:
    """
    Validates forest cover dataset with fault tolerance
    Implements ISO 9000 quality gates
    """
    
    EXPECTED_FEATURES = 54  # Updated: 54 features (without Cover_Type)
    ELEVATION_RANGE = (1859, 3858)
    SLOPE_RANGE = (0, 90)
    ASPECT_RANGE = (0, 360)
    
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(failure_threshold=3)
        self.baseline_stats = None
    
    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate data schema and ranges
        ISO 9000: Process-oriented quality control
        """
        errors = []
        warnings = []
        metrics = {}
        
        # Check feature count
        if df.shape[1] != self.EXPECTED_FEATURES:
            errors.append(f"Expected {self.EXPECTED_FEATURES} features, got {df.shape[1]}")
        
        # Check completeness
        null_counts = df.isnull().sum()
        null_percentage = (null_counts / len(df) * 100).round(2)
        
        if null_counts.sum() > 0:
            warnings.append(f"Found {null_counts.sum()} null values")
            metrics['null_percentage'] = float(null_percentage.max())
        else:
            metrics['null_percentage'] = 0.0
            logger.info("✓ 100% data completeness verified")
        
        # Validate elevation range
        if 'Elevation' in df.columns:
            elevation = df['Elevation']
            out_of_range = ((elevation < self.ELEVATION_RANGE[0]) | 
                           (elevation > self.ELEVATION_RANGE[1])).sum()
            
            if out_of_range > 0:
                errors.append(f"Elevation out of range: {out_of_range} observations")
            
            metrics['elevation_mean'] = float(elevation.mean())
            metrics['elevation_std'] = float(elevation.std())
        
        # Validate aspect circularity
        if 'Aspect' in df.columns:
            aspect = df['Aspect']
            if ((aspect < 0) | (aspect > 360)).any():
                errors.append("Aspect values outside [0, 360] range")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
    
    def detect_drift(self, 
                     df_production: pd.DataFrame,
                     df_baseline: pd.DataFrame,
                     threshold: float = 0.05) -> Tuple[bool, Dict]:
        """
        Detect distributional drift using KS test
        Six Sigma: Statistical process control
        """
        drift_detected = False
        drift_features = []
        p_values = {}
        
        numerical_cols = df_production.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in df_baseline.columns:
                # Kolmogorov-Smirnov test
                statistic, p_value = ks_2samp(
                    df_baseline[col].dropna(),
                    df_production[col].dropna()
                )
                
                p_values[col] = float(p_value)
                
                if p_value < threshold:
                    drift_detected = True
                    drift_features.append(col)
                    logger.warning(f"Drift detected in {col}: p-value={p_value:.4f}")
        
        return drift_detected, {
            'drift_detected': drift_detected,
            'affected_features': drift_features,
            'p_values': p_values
        }
    
    def validate_with_circuit_breaker(self, df: pd.DataFrame) -> ValidationResult:
        """Execute validation with fault tolerance"""
        try:
            return self.circuit_breaker.call(self.validate_schema, df)
        except Exception as e:
            logger.error(f"Validation failed with circuit breaker: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[str(e)],
                warnings=["Circuit breaker triggered"],
                metrics={}
            )


# Example usage with REAL Kaggle dataset
if __name__ == "__main__":
    """
    INSTRUCTIONS TO RUN:
    1. Download dataset from Kaggle: 
       https://www.kaggle.com/competitions/forest-cover-type-prediction/data
    2. Place 'train.csv' in the same directory as this script
    3. Run: python data_validation.py
    """
    
    # ============================================================
    # STEP 1: Load Real Dataset
    # ============================================================
    dataset_path = 'train.csv'  # ← Change this if your file is elsewhere
    
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
    print(f"✓ Columns: {list(df_full.columns[:10])}... (showing first 10)")
    
    # Separate features and target
    feature_cols = [col for col in df_full.columns if col != 'Cover_Type']
    X_full = df_full[feature_cols]
    y_full = df_full['Cover_Type']
    
    print(f"✓ Features extracted: {X_full.shape[1]} columns")
    print(f"✓ Target variable: Cover_Type (7 classes)")
    
    # ============================================================
    # STEP 2: Split into Train/Test for Validation Demo
    # ============================================================
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_full
    )
    
    print(f"\n✓ Train set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    # ============================================================
    # STEP 3: Initialize Validator
    # ============================================================
    validator = ForestDataValidator()
    
    # ============================================================
    # TEST 1: Validate Real Training Data
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 1: VALIDATE REAL TRAINING DATA")
    print("=" * 70)
    
    result_train = validator.validate_with_circuit_breaker(X_train)
    
    print(f"Valid: {result_train.is_valid}")
    print(f"Errors: {result_train.errors if result_train.errors else 'None ✓'}")
    print(f"Warnings: {result_train.warnings if result_train.warnings else 'None ✓'}")
    print(f"\nMetrics:")
    for key, value in result_train.metrics.items():
        print(f"  - {key}: {value}")
    
    # ============================================================
    # TEST 2: Validate Test Data (should be consistent)
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 2: VALIDATE TEST DATA")
    print("=" * 70)
    
    result_test = validator.validate_with_circuit_breaker(X_test)
    
    print(f"Valid: {result_test.is_valid}")
    print(f"Errors: {result_test.errors if result_test.errors else 'None ✓'}")
    
    # ============================================================
    # TEST 3: Detect Drift (Simulate Production Shift)
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 3: DRIFT DETECTION (SIMULATED PRODUCTION DATA)")
    print("=" * 70)
    
    # Simulate production drift: shift elevation by 100m
    X_production = X_test.copy()
    X_production['Elevation'] = X_production['Elevation'] + 100
    
    print("Simulating production data with Elevation shift: +100m")
    
    drift_detected, drift_report = validator.detect_drift(
        df_production=X_production,
        df_baseline=X_train,
        threshold=0.05
    )
    
    print(f"\nDrift Detected: {'YES ⚠️' if drift_detected else 'NO ✓'}")
    print(f"Affected Features: {drift_report['affected_features']}")
    
    if drift_detected:
        print("\nP-values for drifted features:")
        for feature in drift_report['affected_features']:
            p_val = drift_report['p_values'][feature]
            print(f"  - {feature}: p-value = {p_val:.6f}")
    
    # ============================================================
    # TEST 4: Circuit Breaker Stress Test
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 4: CIRCUIT BREAKER FAULT TOLERANCE")
    print("=" * 70)
    
    # Create intentionally invalid data
    invalid_data_list = [
        X_train.drop(columns=['Elevation']),  # Missing critical feature
        X_train.drop(columns=['Aspect', 'Slope']),  # Multiple missing
        pd.DataFrame({'wrong_col': [1, 2, 3]})  # Completely wrong schema
    ]
    
    print("Triggering 3 consecutive validation failures...\n")
    
    for i, invalid_data in enumerate(invalid_data_list, 1):
        print(f"Attempt {i}:")
        result = validator.validate_with_circuit_breaker(invalid_data)
        print(f"  Valid: {result.is_valid}")
        print(f"  Circuit State: {validator.circuit_breaker.state.value}")
        print(f"  Failure Count: {validator.circuit_breaker.failure_count}\n")
    
    # Try one more after circuit opened
    print("Attempt 4 (circuit should be OPEN):")
    try:
        result = validator.validate_with_circuit_breaker(invalid_data_list[0])
        print("  Result:", result)
    except Exception as e:
        print(f"  ❌ Request rejected: {e}")
    
    # ============================================================
    # SUMMARY STATISTICS
    # ============================================================
    print("\n" + "=" * 70)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\nElevation Statistics:")
    print(f"  Mean: {X_train['Elevation'].mean():.2f}m")
    print(f"  Std Dev: {X_train['Elevation'].std():.2f}m")
    print(f"  Range: [{X_train['Elevation'].min()}, {X_train['Elevation'].max()}]")
    
    print(f"\nAspect Statistics:")
    print(f"  Mean: {X_train['Aspect'].mean():.2f}°")
    print(f"  Range: [{X_train['Aspect'].min()}, {X_train['Aspect'].max()}]")
    
    print(f"\nSlope Statistics:")
    print(f"  Mean: {X_train['Slope'].mean():.2f}°")
    print(f"  Range: [{X_train['Slope'].min()}, {X_train['Slope'].max()}]")
    
    # Check for critical thresholds (chaos zones)
    elevation_thresholds = [2400, 2800, 3200]
    print(f"\n🔍 Chaos Zone Analysis (±50m windows):")
    for threshold in elevation_thresholds:
        count = ((X_train['Elevation'] >= threshold - 50) & 
                 (X_train['Elevation'] <= threshold + 50)).sum()
        percentage = (count / len(X_train)) * 100
        print(f"  Near {threshold}m: {count} observations ({percentage:.2f}%)")
    
    # Additional analysis: Soil Type Sparsity
    print(f"\n🔍 Soil Type Sparsity Analysis:")
    soil_cols = [col for col in X_train.columns if col.startswith('Soil_Type')]
    if soil_cols:
        soil_data = X_train[soil_cols]
        total_zeros = (soil_data == 0).sum().sum()
        total_cells = soil_data.size
        sparsity = (total_zeros / total_cells) * 100
        print(f"  Soil features: {len(soil_cols)}")
        print(f"  Total sparsity: {sparsity:.2f}%")
        
        # Show most frequent soil types
        print(f"\n  Most frequent soil types:")
        soil_counts = soil_data.sum().sort_values(ascending=False).head(5)
        for soil_type, count in soil_counts.items():
            percentage = (count / len(X_train)) * 100
            print(f"    {soil_type}: {count} samples ({percentage:.2f}%)")
    
    print("\n" + "=" * 70)
    print("✅ VALIDATION COMPLETE")
    print("=" * 70)

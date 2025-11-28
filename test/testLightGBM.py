"""
Test Ensemble Models
Diagnostic script to verify ensemble functionality
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import sys

# Add src and src/data to path - THIS IS THE KEY FIX
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "data"))

from generateLightGBM import LightGBMTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_lightgbm():
    """Test LightGBM model comprehensively"""
    
    print("=" * 70)
    print("LIGHTGBM MODEL DIAGNOSTIC TEST")
    print("=" * 70)
    
    # Paths
    train_file = Path("data") / "raw" / "train.csv"
    preprocessor_path = Path("data")/ "artifacts" / "preprocessor.pkl"
    lightgbm_model_path = Path("models") / "lightgbm_model.pkl"
    
    # Check files
    if not lightgbm_model_path.exists():
        logger.error(f" LightGBM model not found: {lightgbm_model_path}")
        logger.info("   Run: python src/generateLightGBM.py")
        return
    
    if not preprocessor_path.exists():
        logger.error(f" Preprocessor not found: {preprocessor_path}")
        logger.info("   Run: python src/generateLightGBM.py")
        return
    
    if not train_file.exists():
        logger.error(f" Training data not found: {train_file}")
        logger.info("   Run: python src/generateLightGBM.py")
        return
    
    # Load model first to check expected features
    logger.info("\n Loading model to check expectations...")
    with open(lightgbm_model_path, 'rb') as f:
        ensemble_data = pickle.load(f)
    expected_features = ensemble_data['model'].n_features_in_
    logger.info(f"  Model expects: {expected_features} features")
    
    # Load data
    logger.info("\n Loading training data...")
    df = pd.read_csv(train_file)
    X = df.drop(['Id', 'Cover_Type'], axis=1)
    y = df['Cover_Type'] - 1  # Convert to 0-6
    
    logger.info(f"✓ Loaded {len(df)} samples")
    logger.info(f"  Initial features: {X.shape[1]}")
    logger.info(f"  Classes: {y.nunique()} (range: {y.min()}-{y.max()})")
    
    # Load preprocessor
    logger.info("\n Loading preprocessor...")
    with open(preprocessor_path, 'rb') as f:
        preprocessor_data = pickle.load(f)
    
    pipeline = preprocessor_data['pipeline']
    
    # Preprocess
    logger.info(" Preprocessing data...")
    X_transformed = pipeline.elevation_processor.transform(X)
    X_transformed = pipeline.aspect_transformer.transform(X_transformed)
    X_transformed = pipeline.soil_consolidator.transform(X_transformed)
    
    if pipeline.scaler is not None:
        # Get the numerical columns that were actually scaled during training
        if hasattr(pipeline.scaler, 'feature_names_in_'):
            scaler_features = pipeline.scaler.feature_names_in_.tolist()
            logger.info(f"  Scaler was trained on {len(scaler_features)} features")
            
            # Check which features are available now
            available_cols = X_transformed.columns.tolist()
            logger.info(f"  Currently have {len(available_cols)} features")
            
            # Find missing features
            missing_features = [f for f in scaler_features if f not in available_cols]
            if missing_features:
                logger.warning(f"  Missing features: {missing_features}")
            
            # Find extra features
            extra_features = [f for f in available_cols if f not in scaler_features]
            if extra_features:
                logger.info(f"  Extra features (not in scaler): {extra_features}")
            
            # Only use features that exist in both
            numerical_cols = [col for col in scaler_features if col in available_cols]
            logger.info(f"  Will scale {len(numerical_cols)} common features")
        else:
            # Last resort: get all numerical columns
            numerical_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
            logger.info(f"  Auto-detected {len(numerical_cols)} numerical features")
        
        logger.info(f"  Scaler expects {pipeline.scaler.n_features_in_} features")
        
        # If we have fewer features than expected, we need to match the scaler's expectations
        if len(numerical_cols) < pipeline.scaler.n_features_in_:
            logger.error(f"  Feature mismatch: have {len(numerical_cols)}, need {pipeline.scaler.n_features_in_}")
            logger.error("  Cannot proceed with scaling - feature mismatch")
            logger.info("  Proceeding WITHOUT scaling...")
            X_transformed = X_transformed[numerical_cols]  # Just select the features we have
        else:
            numerical_data = X_transformed[numerical_cols].values
            
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                scaled_data = pipeline.scaler.transform(numerical_data)
                X_transformed[numerical_cols] = scaled_data
    
    logger.info(f"✓ Preprocessing complete: {X.shape[1]} → {X_transformed.shape[1]} features")
    
    # Final feature count check
    if X_transformed.shape[1] != expected_features:
        logger.error(f"\n FEATURE MISMATCH!")
        logger.error(f"  Preprocessed data has: {X_transformed.shape[1]} features")
        logger.error(f"  Model expects: {expected_features} features")
        logger.error(f"\n  Your preprocessing pipeline has changed since model training!")
        logger.error(f"  You need to regenerate BOTH preprocessor and model:")
        logger.error(f"    1. Delete: data/artifacts/preprocessor.pkl")
        logger.error(f"    2. Delete: models/lightgbm_model.pkl")
        logger.error(f"    3. Run: python src/generatePreprocessor.py")
        logger.error(f"    4. Run: python src/generateLightGBM.py")
        logger.error(f"\n  Exiting test...")
        return
    
    # Load ensemble
    logger.info("\n Loading ensemble...")
    trainer = LightGBMTrainer(random_state=ensemble_data['random_state'])
    trainer.model = ensemble_data['model']
    trainer.model_scores = ensemble_data['model_scores']
    trainer.is_fitted = ensemble_data['is_fitted']
    
    logger.info(f"✓ Loaded model")
    
    # Test 1: Individual model predictions
    print("\n" + "=" * 70)
    print("TEST 1: INDIVIDUAL MODEL PREDICTIONS")
    print("=" * 70)
    
    X_array = X_transformed.values if isinstance(X_transformed, pd.DataFrame) else X_transformed
    
    y_pred_individual = trainer.model.predict(X_array)
    acc = accuracy_score(y, y_pred_individual)
    f1 = f1_score(y, y_pred_individual, average='weighted')
    
    print(f"\nLightGBM Model:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (weighted): {f1:.4f}")
    print(f"  Predictions range: {y_pred_individual.min()}-{y_pred_individual.max()}")
    print(f"  Unique classes predicted: {np.unique(y_pred_individual)}")
    
    # Test 2: Ensemble probability predictions
    print("\n" + "=" * 70)
    print("TEST 2: ENSEMBLE PROBABILITY PREDICTIONS")
    print("=" * 70)
    
    try:
        proba = trainer.predict_proba(X_array)
        print(f"\n✓ Probability shape: {proba.shape}")
        print(f"  Expected: ({len(X)}, 7)")
        print(f"  Probability sum (should be ~1.0): {proba[0].sum():.6f}")
        print(f"  Min probability: {proba.min():.6f}")
        print(f"  Max probability: {proba.max():.6f}")
        print(f"\nSample probabilities (first observation):")
        for i, p in enumerate(proba[0]):
            print(f"  Class {i}: {p:.4f}")
    except Exception as e:
        print(f"\n Error in predict_proba: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: LightGBM class predictions
    print("\n" + "=" * 70)
    print("TEST 3: LIGHTGBM CLASS PREDICTIONS")
    print("=" * 70)
    
    try:
        y_pred = trainer.predict(X_array)
        
        print(f"\n✓ Predictions shape: {y_pred.shape}")
        print(f"  Expected: ({len(X)},)")
        print(f"  Predictions range: {y_pred.min()}-{y_pred.max()}")
        print(f"  Expected range: 0-6")
        print(f"  Unique classes predicted: {sorted(np.unique(y_pred))}")
        
        # Calculate metrics
        acc = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average='macro')
        f1_weighted = f1_score(y, y_pred, average='weighted')
        
        print(f"\n LightGBM Performance:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        
    except Exception as e:
        print(f"\n Error in predict: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Confusion matrix
    print("\n" + "=" * 70)
    print("TEST 4: CONFUSION MATRIX")
    print("=" * 70)
    
    try:
        cm = confusion_matrix(y, y_pred)
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print("     ", "  ".join([f"C{i}" for i in range(7)]))
        for i, row in enumerate(cm):
            print(f"C{i}:", "  ".join([f"{val:4d}" for val in row]))
        
        # Per-class accuracy
        print("\n Per-Class Performance:")
        for cls in range(7):
            true_count = (y == cls).sum()
            pred_count = (y_pred == cls).sum()
            correct = cm[cls, cls]
            cls_accuracy = correct / true_count if true_count > 0 else 0
            
            print(f"  Class {cls} (orig {cls+1}): "
                  f"true={true_count:5d}, pred={pred_count:5d}, "
                  f"correct={correct:5d}, acc={cls_accuracy:.4f}")
        
    except Exception as e:
        print(f"\n Error in confusion matrix: {e}")
    
    # Test 5: Sample predictions
    print("\n" + "=" * 70)
    print("TEST 5: SAMPLE PREDICTIONS (First 10)")
    print("=" * 70)
    
    try:
        print("\n  ID | True | Pred | Confidence")
        print("  " + "-" * 35)
        for i in range(min(10, len(y))):
            true_cls = y.iloc[i] if isinstance(y, pd.Series) else y[i]
            pred_cls = y_pred[i]
            confidence = proba[i, pred_cls]
            match = "✓" if true_cls == pred_cls else "✗"
            
            print(f"  {i:3d} | {true_cls:4d} | {pred_cls:4d} | {confidence:.4f} {match}")
    except Exception as e:
        print(f"\n Error in sample predictions: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    if acc > 0.0:
        print("\n✓ LightGBM is working correctly!")
        print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        
        if acc < 0.5:
            print("\n  WARNING: Low accuracy detected")
            print("   - Check if features are scaled correctly")
            print("   - Verify label encoding (should be 0-6)")
            print("   - Review model hyperparameters")
        elif acc > 0.99:
            print("\n  WARNING: Suspiciously high accuracy")
            print("   - This is training set - expect overfitting")
            print("   - Check CV scores for realistic performance")
    else:
        print("\n✗ Ensemble predictions are failing!")
        print("   - Check error messages above")
        print("   - Verify data types (DataFrame vs numpy)")
        print("   - Ensure label encoding matches (0-6)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_lightgbm()
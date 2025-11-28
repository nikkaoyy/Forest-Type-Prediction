"""
Generate LightGBM Model
Train LightGBM model and save as lightgbm_model.pkl
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import time

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️  LightGBM not installed. Install with: pip install lightgbm")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightGBMTrainer:
    """
    Train LightGBM model for forest cover prediction
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize trainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.model_scores = {}
        self.is_fitted = False
        
    def build_model(self) -> lgb.LGBMClassifier:
        """
        Build LightGBM model with optimized hyperparameters
        
        Returns:
            LightGBM classifier
        """
        logger.info("Building LightGBM model...")
        
        model = lgb.LGBMClassifier(
            n_estimators=400,
            num_leaves=64,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
            objective='multiclass',
            num_class=7,
            
        )
        
        return model
    
    def train_model(self, X_train, y_train) -> Tuple[Any, Dict]:
        """
        Train LightGBM model and collect metrics
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model and metrics dictionary
        """
        logger.info("Training LightGBM...")
        
        # Build model
        self.model = self.build_model()
        
        # Track training time
        start_time = time.time()
        
        # Train model
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Predictions
        y_pred = self.model.predict(X_train)
        
        # Calculate metrics
        self.model_scores = {
            'accuracy': accuracy_score(y_train, y_pred),
            'f1_macro': f1_score(y_train, y_pred, average='macro'),
            'f1_weighted': f1_score(y_train, y_pred, average='weighted'),
            'training_time': training_time
        }
        
        logger.info(f"  ✓ LightGBM - Accuracy: {self.model_scores['accuracy']:.4f}, "
                   f"F1 (macro): {self.model_scores['f1_macro']:.4f}, "
                   f"Time: {training_time:.2f}s")
        
        return self.model, self.model_scores
    
    def cross_validate_model(self, X_train, y_train, cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation on the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of CV folds
            
        Returns:
            CV metrics dictionary
        """
        logger.info(f"Cross-validating LightGBM ({cv_folds} folds)...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                            random_state=self.random_state)
        
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        cv_metrics = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        logger.info(f"  ✓ LightGBM CV - Mean: {cv_metrics['cv_mean']:.4f} "
                   f"(±{cv_metrics['cv_std']:.4f})")
        
        return cv_metrics
    
    def fit(self, X_train, y_train, perform_cv: bool = True) -> 'LightGBMTrainer':
        """
        Train LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            perform_cv: Whether to perform cross-validation
            
        Returns:
            Self for chaining
        """
        logger.info("=" * 70)
        logger.info("LIGHTGBM MODEL TRAINING")
        logger.info("=" * 70)
        
        # Train model
        self.train_model(X_train, y_train)
        
        # Cross-validate (optional)
        if perform_cv:
            cv_metrics = self.cross_validate_model(X_train, y_train)
            self.model_scores.update(cv_metrics)
        
        self.is_fitted = True
        
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        
        return self
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Get probability predictions
        
        Args:
            X: Features
            
        Returns:
            Probability array of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict_proba(X)
    
    def predict(self, X) -> np.ndarray:
        """
        Get class predictions
        
        Args:
            X: Features (DataFrame or ndarray)
            
        Returns:
            Predicted classes (0-6, add 1 for original labels 1-7)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names=None) -> pd.DataFrame:
        """
        Get feature importance from the model
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None
        
        importance_df = pd.DataFrame({
            'importance': self.model.feature_importances_
        })
        
        if feature_names is not None and len(feature_names) == len(importance_df):
            importance_df.index = feature_names
        
        return importance_df
    
    def save(self, filepath: str):
        """
        Save model to pickle file
        
        Args:
            filepath: Path to save file
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model.")
        
        model_data = {
            'model': self.model,
            'model_scores': self.model_scores,
            'is_fitted': self.is_fitted,
            'random_state': self.random_state,
            'metadata': {
                'model_type': 'LightGBM',
                'timestamp': pd.Timestamp.now().isoformat(),
                'version': '1.0.0'
            }
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✓ Model saved to: {filepath}")
        logger.info(f"  File size: {filepath.stat().st_size / 1024:.2f} KB")
    
    @staticmethod
    def load(filepath: str) -> 'LightGBMTrainer':
        """
        Load model from pickle file
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            Loaded trainer
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        trainer = LightGBMTrainer(
            random_state=model_data['random_state']
        )
        trainer.model = model_data['model']
        trainer.model_scores = model_data['model_scores']
        trainer.is_fitted = model_data['is_fitted']
        
        logger.info(f"✓ Model loaded from: {filepath}")
        logger.info(f"  Type: {model_data['metadata']['model_type']}")
        logger.info(f"  Version: {model_data['metadata']['version']}")
        
        return trainer
    
    def summary(self):
        """Print model summary"""
        print("=" * 70)
        print("LIGHTGBM MODEL SUMMARY")
        print("=" * 70)
        
        print(f"\nStatus: {'✓ Fitted' if self.is_fitted else '✗ Not fitted'}")
        print(f"Model type: LightGBM Classifier")
        
        if self.model_scores:
            print("\n" + "-" * 70)
            print("MODEL PERFORMANCE")
            print("-" * 70)
            
            print(f"\nAccuracy: {self.model_scores['accuracy']:.4f}")
            print(f"F1 (macro): {self.model_scores['f1_macro']:.4f}")
            print(f"F1 (weighted): {self.model_scores['f1_weighted']:.4f}")
            print(f"Training time: {self.model_scores['training_time']:.2f}s")
            
            if 'cv_mean' in self.model_scores:
                print(f"\nCV accuracy: {self.model_scores['cv_mean']:.4f} "
                     f"(±{self.model_scores['cv_std']:.4f})")
        
        print("\n" + "=" * 70)


def load_and_preprocess_data(preprocessor_path: Path, train_file: Path):
    """
    Load raw data and apply preprocessing pipeline
    
    Args:
        preprocessor_path: Path to preprocessor.pkl
        train_file: Path to train.csv
        
    Returns:
        Tuple of (X_transformed, y, feature_names)
    """
    logger.info("\n📂 Loading raw data...")
    df = pd.read_csv(train_file)
    logger.info(f"✓ Loaded {len(df)} samples")
    
    # Prepare data
    X = df.drop(['Id', 'Cover_Type'], axis=1)
    y = df['Cover_Type']
    
    # Convert labels from 1-7 to 0-6 for sklearn compatibility
    y = y - 1
    
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Classes: {y.nunique()} (labels: {y.min()}-{y.max()})")
    
    # Load and apply preprocessor
    if preprocessor_path.exists():
        logger.info("\n🔧 Applying preprocessing pipeline...")
        
        with open(preprocessor_path, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        pipeline = preprocessor_data['pipeline']
        
        # Apply transformations step by step
        X_transformed = pipeline.elevation_processor.transform(X)
        X_transformed = pipeline.aspect_transformer.transform(X_transformed)
        X_transformed = pipeline.soil_consolidator.transform(X_transformed)
        
        # Apply scaling if available
        if pipeline.scaler is not None:
            # Get numerical columns
            numerical_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
            
            logger.info(f"  Found {len(numerical_cols)} numerical columns")
            logger.info(f"  Scaler expects {pipeline.scaler.n_features_in_} features")
            
            # Ensure we have the right number of features
            if len(numerical_cols) != pipeline.scaler.n_features_in_:
                logger.warning(f"  ⚠️ Feature mismatch! Adjusting...")
                
                # Use only the features that match the scaler
                if len(numerical_cols) > pipeline.scaler.n_features_in_:
                    numerical_cols = numerical_cols[:pipeline.scaler.n_features_in_]
                else:
                    logger.error(f"  ❌ Too few features: {len(numerical_cols)} < {pipeline.scaler.n_features_in_}")
                    logger.error("  Regenerate preprocessor with: python generatePreprocessor.py")
                    raise ValueError("Feature count mismatch - regenerate preprocessor")
            
            # Scale numerical features
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                numerical_data = X_transformed[numerical_cols].values
                scaled_data = pipeline.scaler.transform(numerical_data)
                X_transformed[numerical_cols] = scaled_data
            
            logger.info(f"✓ Scaling applied to {len(numerical_cols)} features")
        
        feature_names = X_transformed.columns.tolist()
        logger.info(f"✓ Preprocessing complete: {X.shape[1]} → {X_transformed.shape[1]} features")
        
        return X_transformed, y, feature_names
    else:
        logger.warning("⚠️  Preprocessor not found, using raw features")
        logger.warning("   Run: python generatePreprocessor.py")
        return X, y, X.columns.tolist()


def main():
    """Main function to generate lightgbm_model.pkl"""
    
    print("=" * 70)
    print("FOREST COVER LIGHTGBM MODEL GENERATOR")
    print("=" * 70)
    
    # Paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    artifacts_dir = data_dir / "data" /"artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = raw_dir / "train.csv"
    preprocessor_path = artifacts_dir / "preprocessor.pkl"
    
    # Check if training data exists
    if not train_file.exists():
        logger.error(f"❌ Training data not found: {train_file}")
        logger.info(f"   Current directory: {Path.cwd()}")
        logger.info(f"   Looking for: {train_file.absolute()}")
        logger.info("   Please ensure train.csv is in: src/data/raw/")
        return
    
    # Load and preprocess data
    try:
        X, y, feature_names = load_and_preprocess_data(preprocessor_path, train_file)
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        return
    
    # Initialize trainer
    logger.info("\n🚀 Initializing LightGBM trainer...")
    trainer = LightGBMTrainer(random_state=42)
    
    # Train model
    logger.info("\n🎯 Starting model training...")
    trainer.fit(X, y, perform_cv=True)
    
    # Evaluate model
    logger.info("\n📊 Evaluating model performance...")
    y_pred = trainer.predict(X)
    
    # Note: y and y_pred are both 0-6 now
    model_accuracy = accuracy_score(y, y_pred)
    model_f1 = f1_score(y, y_pred, average='weighted')
    
    logger.info(f"✓ Model accuracy: {model_accuracy:.4f}")
    logger.info(f"✓ Model F1 (weighted): {model_f1:.4f}")
    
    # Show class distribution
    logger.info(f"\n📈 Class Distribution (0-6, add 1 for original labels):")
    for cls in sorted(y.unique()):
        count = (y == cls).sum()
        pred_count = (y_pred == cls).sum()
        logger.info(f"  Class {cls} (original {cls+1}): {count} actual, {pred_count} predicted")
    
    # Save model
    output_path = artifacts_dir / "lightgbm_model.pkl"
    trainer.save(output_path)
    
    # Show feature importance
    importance_df = trainer.get_feature_importance(feature_names)
    
    if importance_df is not None:
        logger.info("\n🔍 Top 10 Most Important Features:")
        importance_sorted = importance_df.sort_values('importance', ascending=False)
        
        for i, (feature_name, row) in enumerate(importance_sorted.head(10).iterrows(), 1):
            logger.info(f"  {i:2d}. {feature_name:40s} {row['importance']:.4f}")
    
    # Print summary
    print("\n")
    trainer.summary()
    
    print("\n" + "=" * 70)
    print("✅ MODEL GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\n💾 Saved to: {output_path}")
    print(f"   Ready for deployment!")
    print("\nNext steps:")
    print("  1. Test predictions: python src/predict.py")
    print("  2. Start API: uvicorn api.app:app --reload")
    print("  3. Generate submission: python src/generate_submission.py")


if __name__ == "__main__":
    main()
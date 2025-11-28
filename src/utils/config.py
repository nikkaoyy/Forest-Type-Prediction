"""
Forest Cover Type Prediction - System Configuration
Centralized configuration for all system components
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

# Root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports"

# Create directories if they don't exist
for directory in [ARTIFACTS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

class DataConfig:
    """Data loading and processing configuration"""
    
    # File paths
    TRAIN_FILE = RAW_DATA_DIR / "train.csv"
    TEST_FILE = RAW_DATA_DIR / "test.csv"
    SUBMISSION_FILE = RAW_DATA_DIR / "sample_submission.csv"
    
    # Dataset properties
    N_SAMPLES = 15120
    N_FEATURES_RAW = 56  # Including Id and target
    N_FEATURES_INPUT = 54  # Excluding Id and target
    N_CLASSES = 7
    TARGET_COLUMN = "Cover_Type"
    ID_COLUMN = "Id"
    
    # Feature groups
    NUMERICAL_FEATURES = [
        'Elevation',
        'Aspect',
        'Slope',
        'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Horizontal_Distance_To_Fire_Points',
        'Hillshade_9am',
        'Hillshade_Noon',
        'Hillshade_3pm'
    ]
    
    WILDERNESS_FEATURES = [f'Wilderness_Area{i}' for i in range(1, 5)]
    SOIL_FEATURES = [f'Soil_Type{i}' for i in range(1, 41)]
    
    # Validation split
    VALIDATION_SIZE = 0.2
    RANDOM_STATE = 42


# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

class FeatureEngineeringConfig:
    """Feature engineering and preprocessing configuration"""
    
    # Elevation processing (chaos detection thresholds)
    ELEVATION_THRESHOLDS = [2400, 2800, 3200]  # meters
    ELEVATION_ZONES = {
        'Foothill': (1859, 2400),
        'Montane': (2400, 2800),
        'Subalpine': (2800, 3200),
        'Alpine': (3200, 3858)
    }
    
    # Chaos theory parameters
    PROXIMITY_WINDOW = 50  # ±50 meters around thresholds
    UNCERTAINTY_AMPLIFICATION = 2.0  # Multiply uncertainty by 2x near thresholds
    
    # Aspect transformation
    ASPECT_CIRCULAR = True  # Use sin/cos transformation
    
    # Soil consolidation
    SOIL_FREQUENCY_THRESHOLD = 100  # Minimum samples to keep soil type
    SOIL_GROUPS = {
        'Sandy_Soils': [7, 8, 11, 15, 25],
        'Clay_Soils': [1, 2, 3, 4, 5, 6, 9, 13],
        'Rocky_Soils': [16, 17, 18, 19, 20, 21, 22, 23, 26, 27],
        'Organic_Soils': [28, 30, 31, 32, 33, 35],
        'Other_Soils': [34, 36, 37, 38, 39, 40]
    }
    
    # Distance interactions
    CREATE_DISTANCE_INTERACTIONS = True
    DISTANCE_FEATURES = [
        'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Horizontal_Distance_To_Fire_Points'
    ]
    
    # Hillshade ratios
    CREATE_HILLSHADE_RATIOS = True
    
    # Normalization
    APPLY_SCALING = True
    SCALER_TYPE = 'standard'  # 'standard', 'robust', 'minmax'


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

class ModelConfig:
    """Model training and ensemble configuration"""
    
    # Cross-validation
    CV_FOLDS = 5
    CV_STRATEGY = 'stratified'  # 'stratified' or 'spatial_blocked'
    SPATIAL_BLOCK_SIZE = None  # For spatial CV (future implementation)
    
    # Hyperparameter optimization
    OPTUNA_TRIALS = 100
    OPTUNA_TIMEOUT = 3600  # 1 hour
    OPTIMIZATION_METRIC = 'accuracy'
    
    # Base models
    BASE_MODELS = {
        'RandomForest': {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        },
        'XGBoost': {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        },
        'LightGBM': {
            'n_estimators': 400,
            'num_leaves': 64,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    }
    
    # Ensemble configuration
    ENSEMBLE_METHOD = 'weighted_voting'  # 'weighted_voting' or 'stacking'
    ENSEMBLE_WEIGHTS = [0.3, 0.4, 0.3]  # RF, XGB, LGB
    
    # Stacking (if used)
    STACKING_META_LEARNER = 'LogisticRegression'
    STACKING_CV_FOLDS = 5
    
    # Target accuracy
    TARGET_ACCURACY = 0.952  # 95.2%


# ============================================================================
# UNCERTAINTY QUANTIFICATION CONFIGURATION
# ============================================================================

class UncertaintyConfig:
    """Uncertainty estimation configuration"""
    
    # Uncertainty types
    COMPUTE_ALEATORIC = True  # Entropy-based
    COMPUTE_EPISTEMIC = True  # Ensemble variance
    
    # Aleatoric uncertainty (entropy)
    ENTROPY_BASE = 2  # log2 for bits
    NORMALIZE_ENTROPY = True
    MAX_ENTROPY = 2.807  # log2(7) for 7 classes
    
    # Epistemic uncertainty (model disagreement)
    VARIANCE_THRESHOLD = 0.1  # Flag high disagreement
    
    # Combined uncertainty
    COMBINE_METHOD = 'euclidean'  # sqrt(aleatoric² + epistemic²)
    
    # Threshold amplification
    APPLY_THRESHOLD_AMPLIFICATION = True
    AMPLIFICATION_FACTOR = 2.0
    
    # Confidence scoring
    CONFIDENCE_MIN = 0.0
    CONFIDENCE_MAX = 1.0
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    LOW_CONFIDENCE_THRESHOLD = 0.5


# ============================================================================
# MONITORING AND DRIFT DETECTION CONFIGURATION
# ============================================================================

class MonitoringConfig:
    """Model monitoring and drift detection configuration"""
    
    # Drift detection
    ENABLE_DRIFT_DETECTION = True
    DRIFT_METRICS = ['KL_divergence', 'PSI', 'KS_test']
    
    # KL-divergence thresholds
    KL_MINOR_THRESHOLD = 0.01
    KL_MAJOR_THRESHOLD = 0.05
    
    # PSI thresholds
    PSI_MINOR_THRESHOLD = 0.1
    PSI_MAJOR_THRESHOLD = 0.25
    
    # Performance monitoring
    ACCURACY_DROP_WARNING = 0.03  # 3% drop
    ACCURACY_DROP_CRITICAL = 0.05  # 5% drop triggers retraining
    
    # Confidence degradation
    TRACK_CONFIDENCE_TRENDS = True
    CONFIDENCE_WINDOW_WEEKS = 4
    
    # Threshold monitoring
    MONITOR_THRESHOLDS = True
    THRESHOLD_ALERT_FREQUENCY = 0.2  # Alert if 20% in chaos zones
    
    # Alerting
    ALERT_CHANNELS = ['log', 'grafana']  # 'log', 'email', 'slack', 'pagerduty', 'grafana'
    ALERT_SEVERITY_LEVELS = ['minor', 'major', 'critical']


# ============================================================================
# DEPLOYMENT CONFIGURATION
# ============================================================================

class DeploymentConfig:
    """Deployment and serving configuration"""
    
    # API configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_WORKERS = 4
    API_TIMEOUT = 30  # seconds
    
    # SLA targets
    SLA_AVAILABILITY = 0.999  # 99.9%
    SLA_LATENCY_P50 = 50  # ms
    SLA_LATENCY_P95 = 100  # ms
    SLA_LATENCY_P99 = 200  # ms
    
    # Real-time inference
    REALTIME_MAX_LATENCY = 100  # ms
    REALTIME_BATCH_SIZE = 1
    
    # Batch inference
    BATCH_SIZE = 1000
    BATCH_MAX_SAMPLES = 1000000
    BATCH_PARALLELISM = True
    BATCH_GPU_ACCELERATION = False  # Enable if GPU available
    
    # Caching
    ENABLE_REDIS_CACHE = True
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_TTL = 3600  # 1 hour
    CACHE_EVICTION_POLICY = 'LRU'
    
    # Model registry
    MLFLOW_TRACKING_URI = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME = "forest_cover_prediction"
    MODEL_REGISTRY_STAGE = "production"  # 'staging' or 'production'
    
    # Storage
    S3_BUCKET = "forest-cover-models"  # If using S3
    S3_REGION = "us-east-1"
    LOCAL_MODEL_PATH = MODELS_DIR
    
    # Database
    DB_HOST = "localhost"
    DB_PORT = 5432
    DB_NAME = "forest_cover_logs"
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
    
    # Monitoring dashboard
    GRAFANA_URL = "http://localhost:3000"
    PROMETHEUS_PORT = 9090


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

class LoggingConfig:
    """Logging configuration"""
    
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Log files
    MAIN_LOG_FILE = LOGS_DIR / "forest_cover.log"
    ERROR_LOG_FILE = LOGS_DIR / "errors.log"
    PREDICTION_LOG_FILE = LOGS_DIR / "predictions.log"
    
    # Log rotation
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
    LOG_BACKUP_COUNT = 5
    
    # Structured logging
    ENABLE_JSON_LOGGING = False
    LOG_TO_CONSOLE = True
    LOG_TO_FILE = True


# ============================================================================
# EXPERIMENT TRACKING CONFIGURATION
# ============================================================================

class ExperimentConfig:
    """Experiment tracking configuration"""
    
    TRACK_EXPERIMENTS = True
    EXPERIMENT_BACKEND = "mlflow"  # 'mlflow', 'wandb', 'tensorboard'
    
    # MLflow specific
    MLFLOW_TRACKING_URI = DeploymentConfig.MLFLOW_TRACKING_URI
    MLFLOW_ARTIFACT_LOCATION = str(ARTIFACTS_DIR)
    
    # Metrics to track
    TRACK_METRICS = [
        'accuracy',
        'f1_macro',
        'f1_weighted',
        'log_loss',
        'training_time',
        'inference_time'
    ]
    
    # Parameters to track
    TRACK_PARAMS = True
    
    # Artifacts to save
    SAVE_ARTIFACTS = [
        'model',
        'preprocessor',
        'feature_importances',
        'confusion_matrix',
        'classification_report'
    ]


# ============================================================================
# TESTING CONFIGURATION
# ============================================================================

class TestingConfig:
    """Testing and validation configuration"""
    
    # Unit testing
    RUN_UNIT_TESTS = True
    TEST_DATA_SIZE = 100
    
    # Integration testing
    RUN_INTEGRATION_TESTS = True
    
    # Performance testing
    RUN_PERFORMANCE_TESTS = True
    PERFORMANCE_TEST_SAMPLES = 10000
    MAX_INFERENCE_TIME = 0.001  # 1ms per sample
    
    # Reproducibility
    ENSURE_REPRODUCIBILITY = True
    FIXED_SEEDS = [42, 123, 456, 789, 1011]
    MAX_ACCURACY_VARIANCE = 0.015  # 1.5% variance allowed


# ============================================================================
# CHAOS DETECTION CONFIGURATION
# ============================================================================

class ChaosDetectionConfig:
    """Chaos theory and sensitivity detection configuration"""
    
    # Enable chaos detection
    ENABLE_CHAOS_DETECTION = True
    
    # Critical thresholds (elevation)
    CRITICAL_THRESHOLDS = FeatureEngineeringConfig.ELEVATION_THRESHOLDS
    PROXIMITY_WINDOW = FeatureEngineeringConfig.PROXIMITY_WINDOW
    
    # Sensitivity analysis
    ENABLE_SENSITIVITY_ANALYSIS = True
    PERTURBATION_RANGE = 50  # ±50m for elevation
    PERTURBATION_STEPS = 10
    
    # Butterfly effect detection
    TRACK_BUTTERFLY_EFFECTS = True
    BUTTERFLY_THRESHOLD = 0.5  # Class change with small perturbation
    
    # Tipping point identification
    IDENTIFY_TIPPING_POINTS = True
    TIPPING_POINT_WINDOW = 100  # ±100m analysis window


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(config_name: str = 'all') -> Dict:
    """
    Get configuration dictionary
    
    Args:
        config_name: Name of config class or 'all' for everything
        
    Returns:
        Configuration dictionary
    """
    configs = {
        'data': DataConfig,
        'feature_engineering': FeatureEngineeringConfig,
        'model': ModelConfig,
        'uncertainty': UncertaintyConfig,
        'monitoring': MonitoringConfig,
        'deployment': DeploymentConfig,
        'logging': LoggingConfig,
        'experiment': ExperimentConfig,
        'testing': TestingConfig,
        'chaos': ChaosDetectionConfig
    }
    
    if config_name == 'all':
        return {name: cls for name, cls in configs.items()}
    
    return configs.get(config_name.lower())


def print_config_summary():
    """Print a summary of all configurations"""
    print("=" * 70)
    print("FOREST COVER TYPE PREDICTION - CONFIGURATION SUMMARY")
    print("=" * 70)
    
    print("\n PROJECT STRUCTURE:")
    print(f"  Root: {ROOT_DIR}")
    print(f"  Data: {DATA_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Logs: {LOGS_DIR}")
    
    print("\n DATA CONFIGURATION:")
    print(f"  Samples: {DataConfig.N_SAMPLES}")
    print(f"  Features: {DataConfig.N_FEATURES_INPUT}")
    print(f"  Classes: {DataConfig.N_CLASSES}")
    
    print("\n FEATURE ENGINEERING:")
    print(f"  Elevation thresholds: {FeatureEngineeringConfig.ELEVATION_THRESHOLDS}")
    print(f"  Chaos proximity: ±{FeatureEngineeringConfig.PROXIMITY_WINDOW}m")
    print(f"  Uncertainty amplification: {FeatureEngineeringConfig.UNCERTAINTY_AMPLIFICATION}x")
    
    print("\n MODEL CONFIGURATION:")
    print(f"  Base models: {list(ModelConfig.BASE_MODELS.keys())}")
    print(f"  Ensemble: {ModelConfig.ENSEMBLE_METHOD}")
    print(f"  Target accuracy: {ModelConfig.TARGET_ACCURACY:.1%}")
    
    print("\n MONITORING:")
    print(f"  Drift detection: {MonitoringConfig.ENABLE_DRIFT_DETECTION}")
    print(f"  Metrics: {MonitoringConfig.DRIFT_METRICS}")
    print(f"  Critical accuracy drop: {MonitoringConfig.ACCURACY_DROP_CRITICAL:.1%}")
    
    print("\n DEPLOYMENT:")
    print(f"  API: {DeploymentConfig.API_HOST}:{DeploymentConfig.API_PORT}")
    print(f"  SLA: {DeploymentConfig.SLA_AVAILABILITY:.1%} uptime")
    print(f"  Latency (P95): {DeploymentConfig.SLA_LATENCY_P95}ms")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_config_summary()

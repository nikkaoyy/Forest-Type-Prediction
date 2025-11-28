"""
Forest Cover Type Prediction - FastAPI Application
Real-time inference API with uncertainty quantification
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Forest Cover Type Prediction API",
    description="Predict forest cover type from topographic features with uncertainty quantification",
    version="1.0.0"
)

# Global variables for models
preprocessor = None
models = None
weights = None

# Pydantic models for request/response
class PredictionInput(BaseModel):
    """Input features for prediction"""
    Elevation: float = Field(..., ge=1859, le=3858, description="Elevation in meters")
    Aspect: float = Field(..., ge=0, le=360, description="Aspect in degrees")
    Slope: float = Field(..., ge=0, le=90, description="Slope in degrees")
    Horizontal_Distance_To_Hydrology: float = Field(..., ge=0, description="Horizontal distance to hydrology")
    Vertical_Distance_To_Hydrology: float = Field(..., description="Vertical distance to hydrology")
    Horizontal_Distance_To_Roadways: float = Field(..., ge=0, description="Horizontal distance to roadways")
    Horizontal_Distance_To_Fire_Points: float = Field(..., ge=0, description="Horizontal distance to fire points")
    Hillshade_9am: int = Field(..., ge=0, le=255, description="Hillshade at 9am")
    Hillshade_Noon: int = Field(..., ge=0, le=255, description="Hillshade at noon")
    Hillshade_3pm: int = Field(..., ge=0, le=255, description="Hillshade at 3pm")
    Wilderness_Area1: int = Field(..., ge=0, le=1, description="Wilderness area 1")
    Wilderness_Area2: int = Field(..., ge=0, le=1, description="Wilderness area 2")
    Wilderness_Area3: int = Field(..., ge=0, le=1, description="Wilderness area 3")
    Wilderness_Area4: int = Field(..., ge=0, le=1, description="Wilderness area 4")
    Soil_Type1: int = Field(0, ge=0, le=1, description="Soil type 1")
    Soil_Type2: int = Field(0, ge=0, le=1, description="Soil type 2")
    Soil_Type3: int = Field(0, ge=0, le=1, description="Soil type 3")
    Soil_Type4: int = Field(0, ge=0, le=1, description="Soil type 4")
    Soil_Type5: int = Field(0, ge=0, le=1, description="Soil type 5")
    Soil_Type6: int = Field(0, ge=0, le=1, description="Soil type 6")
    Soil_Type7: int = Field(0, ge=0, le=1, description="Soil type 7")
    Soil_Type8: int = Field(0, ge=0, le=1, description="Soil type 8")
    Soil_Type9: int = Field(0, ge=0, le=1, description="Soil type 9")
    Soil_Type10: int = Field(0, ge=0, le=1, description="Soil type 10")
    Soil_Type11: int = Field(0, ge=0, le=1, description="Soil type 11")
    Soil_Type12: int = Field(0, ge=0, le=1, description="Soil type 12")
    Soil_Type13: int = Field(0, ge=0, le=1, description="Soil type 13")
    Soil_Type14: int = Field(0, ge=0, le=1, description="Soil type 14")
    Soil_Type15: int = Field(0, ge=0, le=1, description="Soil type 15")
    Soil_Type16: int = Field(0, ge=0, le=1, description="Soil type 16")
    Soil_Type17: int = Field(0, ge=0, le=1, description="Soil type 17")
    Soil_Type18: int = Field(0, ge=0, le=1, description="Soil type 18")
    Soil_Type19: int = Field(0, ge=0, le=1, description="Soil type 19")
    Soil_Type20: int = Field(0, ge=0, le=1, description="Soil type 20")
    Soil_Type21: int = Field(0, ge=0, le=1, description="Soil type 21")
    Soil_Type22: int = Field(0, ge=0, le=1, description="Soil type 22")
    Soil_Type23: int = Field(0, ge=0, le=1, description="Soil type 23")
    Soil_Type24: int = Field(0, ge=0, le=1, description="Soil type 24")
    Soil_Type25: int = Field(0, ge=0, le=1, description="Soil type 25")
    Soil_Type26: int = Field(0, ge=0, le=1, description="Soil type 26")
    Soil_Type27: int = Field(0, ge=0, le=1, description="Soil type 27")
    Soil_Type28: int = Field(0, ge=0, le=1, description="Soil type 28")
    Soil_Type29: int = Field(0, ge=0, le=1, description="Soil type 29")
    Soil_Type30: int = Field(0, ge=0, le=1, description="Soil type 30")
    Soil_Type31: int = Field(0, ge=0, le=1, description="Soil type 31")
    Soil_Type32: int = Field(0, ge=0, le=1, description="Soil type 32")
    Soil_Type33: int = Field(0, ge=0, le=1, description="Soil type 33")
    Soil_Type34: int = Field(0, ge=0, le=1, description="Soil type 34")
    Soil_Type35: int = Field(0, ge=0, le=1, description="Soil type 35")
    Soil_Type36: int = Field(0, ge=0, le=1, description="Soil type 36")
    Soil_Type37: int = Field(0, ge=0, le=1, description="Soil type 37")
    Soil_Type38: int = Field(0, ge=0, le=1, description="Soil type 38")
    Soil_Type39: int = Field(0, ge=0, le=1, description="Soil type 39")
    Soil_Type40: int = Field(0, ge=0, le=1, description="Soil type 40")

class PredictionOutput(BaseModel):
    """Output prediction with uncertainty"""
    cover_type: int = Field(..., description="Predicted forest cover type (1-7)")
    cover_type_name: str = Field(..., description="Cover type name")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: Dict[int, float] = Field(..., description="Class probabilities")
    uncertainty: Dict[str, float] = Field(..., description="Uncertainty metrics")
    warnings: List[str] = Field(default=[], description="Prediction warnings")

# Cover type mapping
COVER_TYPES = {
    1: 'Spruce/Fir',
    2: 'Lodgepole Pine',
    3: 'Ponderosa Pine',
    4: 'Cottonwood/Willow',
    5: 'Aspen',
    6: 'Douglas-fir',
    7: 'Krummholz'
}

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global preprocessor, models, weights
    
    try:
        # Load preprocessor
        preprocessor_path = Path("data/artifacts/preprocessor.pkl")
        with open(preprocessor_path, 'rb') as f:
            preprocessor_data = pickle.load(f)
        preprocessor = preprocessor_data['pipeline']
        logger.info(f"✓ Preprocessor loaded from {preprocessor_path}")
        
        # Load ensemble
        ensemble_path = Path("data/artifacts/ensemble_models.pkl")
        with open(ensemble_path, 'rb') as f:
            ensemble_data = pickle.load(f)
        models = ensemble_data['models']
        weights = ensemble_data['ensemble_weights']
        logger.info(f"✓ Ensemble loaded from {ensemble_path}")
        logger.info(f"  Models: {list(models.keys())}")
        
    except Exception as e:
        logger.error(f"✗ Failed to load models: {e}")
        raise

def calculate_uncertainty(probabilities: np.ndarray, elevation: float) -> Dict[str, float]:
    """Calculate aleatoric and epistemic uncertainty"""
    # Aleatoric uncertainty (entropy)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    max_entropy = np.log2(7)  # 7 classes
    aleatoric = entropy / max_entropy
    
    # Epistemic uncertainty (model variance - simplified)
    epistemic = 0.05  # Placeholder - would need individual model predictions
    
    # Total uncertainty
    total = np.sqrt(aleatoric**2 + epistemic**2)
    
    # Chaos amplification near thresholds
    thresholds = [2400, 2800, 3200]
    near_threshold = any(abs(elevation - t) <= 50 for t in thresholds)
    
    if near_threshold:
        total *= 2.0
    
    confidence = 1 - total
    confidence = max(0, min(1, confidence))  # Clip to [0, 1]
    
    return {
        'aleatoric': float(aleatoric),
        'epistemic': float(epistemic),
        'total': float(total),
        'confidence_score': float(confidence),
        'near_threshold': near_threshold
    }

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Forest Cover Type Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": models is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Make a prediction"""
    try:
        # Convert input to DataFrame
        input_dict = input_data.dict()
        df = pd.DataFrame([input_dict])
        
        # Preprocess
        X_transformed = preprocessor.elevation_processor.transform(df)
        X_transformed = preprocessor.aspect_transformer.transform(X_transformed)
        X_transformed = preprocessor.soil_consolidator.transform(X_transformed)
        
        if preprocessor.scaler is not None:
            numerical_cols = X_transformed.select_dtypes(include=[np.number]).columns
            X_transformed[numerical_cols] = preprocessor.scaler.transform(X_transformed[numerical_cols])
        
        # Get ensemble predictions
        ensemble_proba = np.zeros(7)
        for i, (name, model) in enumerate(models.items()):
            proba = model.predict_proba(X_transformed)[0]
            ensemble_proba += weights[i] * proba
        
        # Get prediction
        prediction = int(np.argmax(ensemble_proba) + 1)  # Classes 1-7
        confidence = float(np.max(ensemble_proba))
        
        # Calculate uncertainty
        uncertainty_metrics = calculate_uncertainty(ensemble_proba, input_data.Elevation)
        
        # Generate warnings
        warnings = []
        if uncertainty_metrics['near_threshold']:
            warnings.append(f"Prediction near ecological threshold (±50m of {2400}m, {2800}m, or {3200}m)")
        if confidence < 0.5:
            warnings.append("Low confidence prediction - manual review recommended")
        if uncertainty_metrics['total'] > 0.7:
            warnings.append("High uncertainty detected")
        
        # Prepare response
        return PredictionOutput(
            cover_type=prediction,
            cover_type_name=COVER_TYPES[prediction],
            confidence=confidence,
            probabilities={i+1: float(p) for i, p in enumerate(ensemble_proba)},
            uncertainty=uncertainty_metrics,
            warnings=warnings
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(input_data: List[PredictionInput]):
    """Batch prediction endpoint"""
    results = []
    
    for sample in input_data:
        try:
            result = await predict(sample)
            results.append(result.dict())
        except Exception as e:
            results.append({"error": str(e)})
    
    return {"predictions": results, "count": len(results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

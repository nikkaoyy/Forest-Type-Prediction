"""
Uncertainty Quantification Module
Implements aleatoric and epistemic uncertainty estimation
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """
    Quantifies prediction uncertainty by decomposing into:
    - Aleatoric uncertainty (data uncertainty / irreducible)
    - Epistemic uncertainty (model uncertainty / reducible)
    """
    
    # Elevation thresholds for chaos zone detection (from Workshop 1)
    ELEVATION_THRESHOLDS = [2400, 2800, 3200]
    PROXIMITY_WINDOW = 50  # ±50m
    UNCERTAINTY_AMPLIFICATION = 2.0
    
    def __init__(self, n_classes: int = 7):
        """
        Initialize uncertainty quantifier
        
        Args:
            n_classes: Number of target classes
        """
        self.n_classes = n_classes
        self.max_entropy = np.log2(n_classes)  # Maximum possible entropy
    
    def calculate_aleatoric_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate aleatoric (data) uncertainty using Shannon entropy
        
        Aleatoric uncertainty reflects irreducible randomness in the data,
        such as class overlap in ecological transition zones.
        
        Args:
            probabilities: Probability predictions (n_samples, n_classes)
            
        Returns:
            Array of normalized entropy values [0, 1]
        """
        # Calculate Shannon entropy
        entropies = entropy(probabilities.T, base=2)  # Use base 2 for bits
        
        # Normalize by maximum possible entropy
        normalized_entropy = entropies / self.max_entropy
        
        return normalized_entropy
    
    def calculate_epistemic_uncertainty(self, 
                                       model_probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate epistemic (model) uncertainty from ensemble disagreement
        
        Epistemic uncertainty reflects model uncertainty that could be reduced
        with more training data or better models.
        
        Args:
            model_probabilities: Dictionary mapping model names to their probability predictions
                               {'random_forest': proba_rf, 'xgboost': proba_xgb, ...}
            
        Returns:
            Array of variance-based uncertainty values
        """
        # Stack all model predictions
        all_probs = np.stack(list(model_probabilities.values()), axis=0)
        
        # Calculate variance across models for each class
        variances = np.var(all_probs, axis=0)
        
        # Aggregate variance across classes (using mean)
        epistemic_uncertainty = np.mean(variances, axis=1)
        
        return epistemic_uncertainty
    
    def calculate_total_uncertainty(self,
                                   aleatoric: np.ndarray,
                                   epistemic: np.ndarray) -> np.ndarray:
        """
        Combine aleatoric and epistemic uncertainties
        
        Args:
            aleatoric: Aleatoric uncertainty values
            epistemic: Epistemic uncertainty values
            
        Returns:
            Total uncertainty (Euclidean combination)
        """
        total_uncertainty = np.sqrt(aleatoric**2 + epistemic**2)
        return total_uncertainty
    
    def detect_threshold_proximity(self, elevation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect observations near ecological threshold zones
        
        Args:
            elevation: Array of elevation values
            
        Returns:
            Tuple of (near_threshold_flags, distances_to_nearest_threshold)
        """
        near_threshold = np.zeros(len(elevation), dtype=bool)
        distances = np.full(len(elevation), np.inf)
        
        for threshold in self.ELEVATION_THRESHOLDS:
            distance = np.abs(elevation - threshold)
            is_near = distance <= self.PROXIMITY_WINDOW
            
            near_threshold |= is_near
            distances = np.minimum(distances, distance)
        
        return near_threshold, distances
    
    def apply_chaos_amplification(self,
                                 uncertainty: np.ndarray,
                                 elevation: np.ndarray) -> np.ndarray:
        """
        Amplify uncertainty for observations near ecological thresholds
        
        Implements chaos theory principle: small changes near tipping points
        can lead to large changes in outcomes.
        
        Args:
            uncertainty: Base uncertainty values
            elevation: Elevation values for threshold detection
            
        Returns:
            Amplified uncertainty values
        """
        near_threshold, _ = self.detect_threshold_proximity(elevation)
        
        # Apply amplification factor to observations near thresholds
        amplified_uncertainty = uncertainty.copy()
        amplified_uncertainty[near_threshold] *= self.UNCERTAINTY_AMPLIFICATION
        
        # Log statistics
        n_amplified = near_threshold.sum()
        pct_amplified = (n_amplified / len(elevation)) * 100
        logger.info(f"Chaos amplification applied to {n_amplified} observations ({pct_amplified:.1f}%)")
        
        return amplified_uncertainty
    
    def quantify(self,
                ensemble_probabilities: np.ndarray,
                model_probabilities: Dict[str, np.ndarray],
                elevation: np.ndarray = None,
                apply_amplification: bool = True) -> Dict[str, np.ndarray]:
        """
        Complete uncertainty quantification pipeline
        
        Args:
            ensemble_probabilities: Final ensemble probability predictions (n_samples, n_classes)
            model_probabilities: Individual model probabilities for epistemic calculation
            elevation: Elevation values for chaos amplification (optional)
            apply_amplification: Whether to apply chaos amplification
            
        Returns:
            Dictionary containing all uncertainty metrics
        """
        logger.info("Calculating uncertainty metrics...")
        
        # Calculate aleatoric uncertainty
        aleatoric = self.calculate_aleatoric_uncertainty(ensemble_probabilities)
        
        # Calculate epistemic uncertainty
        epistemic = self.calculate_epistemic_uncertainty(model_probabilities)
        
        # Calculate total uncertainty
        total = self.calculate_total_uncertainty(aleatoric, epistemic)
        
        # Apply chaos amplification if elevation provided
        if elevation is not None and apply_amplification:
            total_amplified = self.apply_chaos_amplification(total, elevation)
            near_threshold, distances = self.detect_threshold_proximity(elevation)
        else:
            total_amplified = total
            near_threshold = np.zeros(len(total), dtype=bool)
            distances = np.full(len(total), np.nan)
        
        # Calculate confidence score
        confidence = 1 - total_amplified
        confidence = np.clip(confidence, 0, 1)  # Ensure [0, 1] range
        
        results = {
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'total': total,
            'total_amplified': total_amplified,
            'confidence': confidence,
            'near_threshold': near_threshold,
            'distance_to_threshold': distances
        }
        
        logger.info(f"✓ Uncertainty quantification complete")
        logger.info(f"  Mean aleatoric: {aleatoric.mean():.4f}")
        logger.info(f"  Mean epistemic: {epistemic.mean():.4f}")
        logger.info(f"  Mean total: {total_amplified.mean():.4f}")
        logger.info(f"  Mean confidence: {confidence.mean():.4f}")
        
        return results
    
    def generate_prediction_report(self,
                                  predictions: np.ndarray,
                                  probabilities: np.ndarray,
                                  uncertainty_metrics: Dict,
                                  sample_idx: int = 0) -> Dict:
        """
        Generate detailed prediction report for a single observation
        
        Args:
            predictions: Predicted class labels
            probabilities: Class probabilities
            uncertainty_metrics: Output from quantify()
            sample_idx: Index of sample to report
            
        Returns:
            Dictionary with detailed prediction information
        """
        report = {
            'prediction': int(predictions[sample_idx]),
            'confidence': float(probabilities[sample_idx, predictions[sample_idx]]),
            'probabilities': probabilities[sample_idx].tolist(),
            'uncertainty': {
                'aleatoric': float(uncertainty_metrics['aleatoric'][sample_idx]),
                'epistemic': float(uncertainty_metrics['epistemic'][sample_idx]),
                'total': float(uncertainty_metrics['total_amplified'][sample_idx]),
                'confidence_score': float(uncertainty_metrics['confidence'][sample_idx])
            },
            'warnings': []
        }
        
        # Add warnings based on uncertainty
        if uncertainty_metrics['near_threshold'][sample_idx]:
            report['warnings'].append({
                'type': 'threshold_proximity',
                'message': f"Prediction near ecological threshold (±{self.PROXIMITY_WINDOW}m)",
                'distance': float(uncertainty_metrics['distance_to_threshold'][sample_idx]),
                'recommendation': 'Consider field verification'
            })
        
        if uncertainty_metrics['total_amplified'][sample_idx] > 0.7:
            report['warnings'].append({
                'type': 'high_uncertainty',
                'message': 'High prediction uncertainty detected',
                'recommendation': 'Manual review recommended'
            })
        
        return report


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("UNCERTAINTY QUANTIFICATION DEMO")
    print("=" * 70)
    
    # Simulate ensemble predictions
    np.random.seed(42)
    n_samples = 100
    n_classes = 7
    
    # Simulate probability predictions from three models
    rf_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
    xgb_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
    lgb_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
    
    # Ensemble (weighted average)
    ensemble_proba = 0.3 * rf_proba + 0.4 * xgb_proba + 0.3 * lgb_proba
    
    # Simulate elevation data
    elevation = np.random.randint(1859, 3858, n_samples)
    
    # Initialize quantifier
    quantifier = UncertaintyQuantifier(n_classes=7)
    
    # Quantify uncertainty
    uncertainty_metrics = quantifier.quantify(
        ensemble_probabilities=ensemble_proba,
        model_probabilities={
            'random_forest': rf_proba,
            'xgboost': xgb_proba,
            'lightgbm': lgb_proba
        },
        elevation=elevation,
        apply_amplification=True
    )
    
    # Generate sample prediction report
    predictions = np.argmax(ensemble_proba, axis=1)
    report = quantifier.generate_prediction_report(
        predictions=predictions,
        probabilities=ensemble_proba,
        uncertainty_metrics=uncertainty_metrics,
        sample_idx=0
    )
    
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTION REPORT")
    print("=" * 70)
    print(f"Prediction: Class {report['prediction']}")
    print(f"Confidence: {report['confidence']:.4f}")
    print(f"\nUncertainty Metrics:")
    print(f"  Aleatoric: {report['uncertainty']['aleatoric']:.4f}")
    print(f"  Epistemic: {report['uncertainty']['epistemic']:.4f}")
    print(f"  Total: {report['uncertainty']['total']:.4f}")
    print(f"  Confidence Score: {report['uncertainty']['confidence_score']:.4f}")
    
    if report['warnings']:
        print(f"\nWarnings:")
        for warning in report['warnings']:
            print(f"  - {warning['message']}")
            print(f"    Recommendation: {warning['recommendation']}")
    
    print("\n✅ Demo complete!")

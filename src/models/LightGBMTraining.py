"""
Model Evaluation Module
Comprehensive evaluation metrics and visualizations
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations
    """
    
    COVER_TYPE_NAMES = {
        1: 'Spruce/Fir',
        2: 'Lodgepole Pine',
        3: 'Ponderosa Pine',
        4: 'Cottonwood/Willow',
        5: 'Aspen',
        6: 'Douglas-fir',
        7: 'Krummholz'
    }
    
    def __init__(self, class_names: Dict[int, str] = None):
        """
        Initialize evaluator
        
        Args:
            class_names: Optional mapping of class indices to names
        """
        self.class_names = class_names or self.COVER_TYPE_NAMES
    
    def calculate_metrics(self, y_true, y_pred, y_proba=None) -> Dict:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating evaluation metrics...")
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        metrics['per_class'] = {}
        for i, class_idx in enumerate(sorted(self.class_names.keys())):
            metrics['per_class'][class_idx] = {
                'name': self.class_names[class_idx],
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        # Probability-based metrics
        if y_proba is not None:
            try:
                # Multi-class log loss
                metrics['log_loss'] = log_loss(y_true, y_proba)
                
                # One-vs-Rest AUC (if binary problem or multi-class)
                if len(np.unique(y_true)) == 2:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['auc_roc_ovr'] = roc_auc_score(
                        y_true, y_proba, 
                        multi_class='ovr', 
                        average='weighted'
                    )
            except Exception as e:
                logger.warning(f"Could not calculate probability-based metrics: {e}")
        
        logger.info(f"✓ Overall Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"✓ F1-Score (Macro): {metrics['f1_macro']:.4f}")
        logger.info(f"✓ F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        
        return metrics
    
    def generate_confusion_matrix(self, 
                                 y_true, 
                                 y_pred, 
                                 normalize: bool = False,
                                 save_path: str = None) -> np.ndarray:
        """
        Generate and optionally visualize confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the figure (optional)
            
        Returns:
            Confusion matrix array
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Visualize
        plt.figure(figsize=(12, 10))
        
        class_labels = [self.class_names[i] for i in sorted(self.class_names.keys())]
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''), 
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Confusion matrix saved to: {save_path}")
        
        plt.close()
        
        return cm
    
    def generate_classification_report(self, y_true, y_pred) -> str:
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        target_names = [self.class_names[i] for i in sorted(self.class_names.keys())]
        
        report = classification_report(
            y_true, 
            y_pred,
            target_names=target_names,
            digits=4
        )
        
        return report
    
    def evaluate_chaos_zones(self, 
                            y_true, 
                            y_pred, 
                            elevation, 
                            thresholds=[2400, 2800, 3200],
                            window=50) -> Dict:
        """
        Evaluate model performance in chaos zones (near elevation thresholds)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            elevation: Elevation values
            thresholds: Elevation thresholds to check
            window: Proximity window (±m)
            
        Returns:
            Dictionary with chaos zone performance metrics
        """
        logger.info("Evaluating performance in chaos zones...")
        
        results = {
            'overall': {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            },
            'zones': {}
        }
        
        for threshold in thresholds:
            # Identify observations near this threshold
            near_threshold = np.abs(elevation - threshold) <= window
            
            if near_threshold.sum() > 0:
                zone_y_true = y_true[near_threshold]
                zone_y_pred = y_pred[near_threshold]
                
                results['zones'][threshold] = {
                    'n_observations': int(near_threshold.sum()),
                    'percentage': float((near_threshold.sum() / len(elevation)) * 100),
                    'accuracy': float(accuracy_score(zone_y_true, zone_y_pred)),
                    'f1_score': float(f1_score(zone_y_true, zone_y_pred, average='weighted'))
                }
                
                logger.info(f"  Threshold {threshold}m: "
                          f"Accuracy={results['zones'][threshold]['accuracy']:.4f}, "
                          f"n={results['zones'][threshold]['n_observations']}")
        
        return results
    
    def plot_feature_importance(self, 
                               model,
                               feature_names,
                               top_n=20,
                               save_path=None):
        """
        Plot feature importance for tree-based models
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            top_n: Number of top features to display
            save_path: Path to save figure
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Feature importance plot saved to: {save_path}")
        
        plt.close()
    
    def generate_evaluation_report(self,
                                  y_true,
                                  y_pred,
                                  y_proba=None,
                                  elevation=None,
                                  output_dir='reports/') -> Dict:
        """
        Generate comprehensive evaluation report with visualizations
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            elevation: Elevation values for chaos zone analysis (optional)
            output_dir: Directory to save reports and figures
            
        Returns:
            Complete evaluation dictionary
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info("GENERATING COMPREHENSIVE EVALUATION REPORT")
        logger.info("=" * 70)
        
        report = {}
        
        # Calculate metrics
        report['metrics'] = self.calculate_metrics(y_true, y_pred, y_proba)
        
        # Confusion matrix
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        report['confusion_matrix'] = self.generate_confusion_matrix(
            y_true, y_pred, normalize=True, save_path=cm_path
        )
        
        # Classification report
        report['classification_report'] = self.generate_classification_report(
            y_true, y_pred
        )
        
        # Chaos zone evaluation
        if elevation is not None:
            report['chaos_zones'] = self.evaluate_chaos_zones(
                y_true, y_pred, elevation
            )
        
        # Save report as JSON
        json_path = os.path.join(output_dir, 'evaluation_report.json')
        with open(json_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_report = {
                'metrics': report['metrics'],
                'chaos_zones': report.get('chaos_zones', {})
            }
            json.dump(json_report, f, indent=2)
        
        logger.info(f"\n✓ Evaluation report saved to: {json_path}")
        logger.info(f"✓ Confusion matrix saved to: {cm_path}")
        
        logger.info("\n" + "=" * 70)
        logger.info("CLASSIFICATION REPORT")
        logger.info("=" * 70)
        print(report['classification_report'])
        
        return report


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("MODEL EVALUATION DEMO")
    print("=" * 70)
    
    # Simulate predictions
    np.random.seed(42)
    n_samples = 500
    n_classes = 7
    
    # Generate synthetic true labels
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Generate synthetic predictions (with some accuracy)
    y_pred = y_true.copy()
    # Add some errors
    error_idx = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    y_pred[error_idx] = np.random.randint(0, n_classes, len(error_idx))
    
    # Generate synthetic probabilities
    y_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
    
    # Generate synthetic elevation
    elevation = np.random.randint(1859, 3858, n_samples)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Generate comprehensive report
    report = evaluator.generate_evaluation_report(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        elevation=elevation,
        output_dir='demo_reports/'
    )
    
    print("\n Demo complete! Check demo_reports/ for outputs.")

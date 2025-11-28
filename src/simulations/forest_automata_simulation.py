"""
Forest Cover Type Cellular Automata Simulation - INTEGRATED VERSION
====================================================================
Simulates ecological succession with REAL LightGBM model integration
Validates automata rules against trained model predictions

Author: Forest Cover Type Prediction Team
Date: December 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import pickle
from pathlib import Path
import json
from typing import Dict, Tuple, List, Optional
import logging
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForestAutomataSimulation:
    """
    Cellular automata simulation with LightGBM model integration
    
    Features:
    - Real feature engineering using preprocessor.pkl
    - LightGBM predictions for validation and guidance
    - Hybrid mode: automata + model corrections
    - Chaos zone detection with model validation
    """
    
    # Forest cover types (matching Kaggle dataset)
    COVER_TYPES = {
        0: {'name': 'Empty', 'color': '#E8F5E9'},
        1: {'name': 'Spruce/Fir', 'color': '#1B5E20'},
        2: {'name': 'Lodgepole Pine', 'color': '#2E7D32'},
        3: {'name': 'Ponderosa Pine', 'color': '#388E3C'},
        4: {'name': 'Cottonwood/Willow', 'color': '#66BB6A'},
        5: {'name': 'Aspen', 'color': '#81C784'},
        6: {'name': 'Douglas-fir', 'color': '#4CAF50'},
        7: {'name': 'Krummholz', 'color': '#A5D6A7'}
    }
    
    # Elevation zones (meters - real scale)
    ELEVATION_ZONES = {
        'FOOTHILL': {'min': 1859, 'max': 2400, 'dominant': [1, 5]},
        'MONTANE': {'min': 2400, 'max': 2800, 'dominant': [2, 3, 4]},
        'SUBALPINE': {'min': 2800, 'max': 3200, 'dominant': [6, 2]},
        'ALPINE': {'min': 3200, 'max': 3858, 'dominant': [7]}
    }
    
    # Chaos thresholds (real elevation values)
    CHAOS_THRESHOLDS = [2400, 2800, 3200]
    CHAOS_PROXIMITY = 100  # ±100m proximity window
    
    def __init__(self, 
                 grid_size: int = 100, 
                 model_path: str = Path(r"C:\Users\nikka\Downloads\Forest-Cover-Type-Prediction----Kaggle\Forest-Cover-Type-Prediction----Kaggle\ForestTypePredict\src\models\lightgbm_model.pkl"),
                 preprocessor_path: str = Path(r"C:\Users\nikka\Downloads\Forest-Cover-Type-Prediction----Kaggle\Forest-Cover-Type-Prediction----Kaggle\ForestTypePredict\src\data\artifacts\preprocessor.pkl"),
                 use_model_corrections: bool = True):
        """
        Initialize simulation with LightGBM integration
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            model_path: Path to trained LightGBM model
            preprocessor_path: Path to preprocessing pipeline
            use_model_corrections: Use model to correct automata predictions
        """
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.use_model_corrections = use_model_corrections
        
        # Environmental features (persistent)
        self.elevation = np.zeros((grid_size, grid_size), dtype=float)
        self.aspect = np.zeros((grid_size, grid_size), dtype=float)
        self.slope = np.zeros((grid_size, grid_size), dtype=float)
        self.hillshade_9am = np.zeros((grid_size, grid_size), dtype=float)
        self.hillshade_noon = np.zeros((grid_size, grid_size), dtype=float)
        self.hillshade_3pm = np.zeros((grid_size, grid_size), dtype=float)
        self.horizontal_dist_hydrology = np.zeros((grid_size, grid_size), dtype=float)
        self.vertical_dist_hydrology = np.zeros((grid_size, grid_size), dtype=float)
        self.horizontal_dist_roadways = np.zeros((grid_size, grid_size), dtype=float)
        self.horizontal_dist_firepoints = np.zeros((grid_size, grid_size), dtype=float)
        self.wilderness_area = np.zeros((grid_size, grid_size), dtype=int)
        self.soil_type = np.zeros((grid_size, grid_size), dtype=int)
        
        self.generation = 0
        
        # Load ML model and preprocessor
        self.model = None
        self.preprocessor = None
        self.model_loaded = self._load_model(model_path, preprocessor_path)
        
        # Initialize grid with realistic features
        self._initialize_grid()
        
        # Statistics tracking
        self.history = {
            'generation': [],
            'populations': [],
            'chaos_events': [],
            'model_agreements': [],
            'model_corrections': []
        }
    
    def _load_model(self, model_path: str, preprocessor_path: str) -> bool:
        """Load trained model and preprocessor"""
        try:
            # ✨ Add project root to sys.path to resolve 'src' module imports
            import sys
            # Get the directory where THIS script is located (Forest-Type-Prediction/)
            project_root = Path(__file__).parent.resolve()
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
                logger.info(f"✓ Added project root to sys.path: {project_root}")
            
            # Convert to Path objects
            model_path = Path(model_path)
            preprocessor_path = Path(preprocessor_path)
            
            logger.info(f"📂 Looking for model at: {model_path}")
            logger.info(f"📂 Model exists: {model_path.exists()}")
            logger.info(f"📂 Looking for preprocessor at: {preprocessor_path}")
            logger.info(f"📂 Preprocessor exists: {preprocessor_path.exists()}")
            
            # Load model
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                # Handle both dict and direct model object
                if isinstance(model_data, dict):
                    self.model = model_data.get('model', model_data)
                else:
                    self.model = model_data
                logger.info(f"✓ LightGBM model loaded from {model_path}")
            else:
                logger.warning(f"⚠️  Model not found at {model_path}")
                return False
            
            # Load preprocessor
            if preprocessor_path.exists():
                with open(preprocessor_path, 'rb') as f:
                    prep_data = pickle.load(f)
                # ✅ FIX: Handle direct FeatureEngineeringPipeline object
                if isinstance(prep_data, dict):
                    self.preprocessor = prep_data.get('pipeline', prep_data)
                else:
                    # Object is already the pipeline directly
                    self.preprocessor = prep_data
                logger.info(f"✓ Preprocessor loaded from {preprocessor_path}")
                logger.info(f"✓ Preprocessor type: {type(self.preprocessor).__name__}")
            else:
                logger.warning(f"⚠️  Preprocessor not found at {preprocessor_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Could not load model: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            return False
    
    def _initialize_grid(self):
        """Initialize grid with realistic elevation and features"""
        logger.info("Initializing grid with realistic features...")
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Create elevation gradient (higher at top) + noise
                base_elev = 1859 + (self.grid_size - i) / self.grid_size * (3858 - 1859)
                noise = np.random.normal(0, 150)
                self.elevation[i, j] = np.clip(base_elev + noise, 1859, 3858)
                
                # Aspect (0-360 degrees) - somewhat correlated with position
                self.aspect[i, j] = (j / self.grid_size * 180 + np.random.uniform(-30, 30)) % 360
                
                # Slope (0-66 degrees) - higher at elevation transitions
                elev_gradient = abs(i - self.grid_size/2) / (self.grid_size/2)
                self.slope[i, j] = np.clip(elev_gradient * 40 + np.random.uniform(0, 15), 0, 66)
                
                # Hillshade (0-255) - based on aspect and slope
                aspect_rad = np.radians(self.aspect[i, j])
                self.hillshade_9am[i, j] = np.clip(200 + 55 * np.cos(aspect_rad - np.pi/4), 0, 255)
                self.hillshade_noon[i, j] = np.clip(230 + 25 * np.cos(aspect_rad), 0, 255)
                self.hillshade_3pm[i, j] = np.clip(200 + 55 * np.cos(aspect_rad + np.pi/4), 0, 255)
                
                # Distance to hydrology (0-1397m horizontal, -173 to 601m vertical)
                self.horizontal_dist_hydrology[i, j] = np.random.uniform(0, 800)
                self.vertical_dist_hydrology[i, j] = np.random.uniform(-100, 400)
                
                # Distance to roadways (0-6890m)
                self.horizontal_dist_roadways[i, j] = np.random.uniform(0, 4000)
                
                # Distance to fire points (0-6993m)
                self.horizontal_dist_firepoints[i, j] = np.random.uniform(0, 4000)
                
                # Wilderness area (1-4) - divided by quadrants
                if i < self.grid_size/2 and j < self.grid_size/2:
                    self.wilderness_area[i, j] = 1
                elif i < self.grid_size/2:
                    self.wilderness_area[i, j] = 2
                elif j < self.grid_size/2:
                    self.wilderness_area[i, j] = 3
                else:
                    self.wilderness_area[i, j] = 4
                
                # Soil type (1-40) - elevation dependent
                elev = self.elevation[i, j]
                if elev < 2400:
                    self.soil_type[i, j] = np.random.choice([1, 2, 3, 10, 29])
                elif elev < 2800:
                    self.soil_type[i, j] = np.random.choice([4, 5, 6, 11, 12, 13])
                elif elev < 3200:
                    self.soil_type[i, j] = np.random.choice([14, 15, 16, 17, 18])
                else:
                    self.soil_type[i, j] = np.random.choice([19, 20, 21, 22, 23, 38, 39])
                
                # Initialize forest cover based on elevation zone
                zone = self._get_elevation_zone(self.elevation[i, j])
                if np.random.random() < 0.25:
                    self.grid[i, j] = 0  # Empty
                else:
                    self.grid[i, j] = np.random.choice(zone['dominant'])
        
        logger.info(f"✓ Grid initialized: {self.grid_size}x{self.grid_size} with realistic features")
    
    def _get_elevation_zone(self, elevation: float) -> Dict:
        """Get elevation zone for a given elevation value"""
        for zone_name, zone in self.ELEVATION_ZONES.items():
            if zone['min'] <= elevation < zone['max']:
                return zone
        return self.ELEVATION_ZONES['ALPINE']
    
    def _is_near_threshold(self, elevation: float) -> bool:
        """Check if elevation is near a chaos threshold"""
        for threshold in self.CHAOS_THRESHOLDS:
            if abs(elevation - threshold) < self.CHAOS_PROXIMITY:
                return True
        return False
    
    def _get_neighbors(self, x: int, y: int) -> List[int]:
        """Get Moore neighborhood (8 neighbors)"""
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                nx, ny = x + i, y + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    neighbors.append(self.grid[nx, ny])
        return neighbors
    
    def _create_feature_vector(self, i: int, j: int) -> pd.DataFrame:
        """Create feature vector for a cell (compatible with preprocessor)"""
        features = {
            'Elevation': self.elevation[i, j],
            'Aspect': self.aspect[i, j],
            'Slope': self.slope[i, j],
            'Horizontal_Distance_To_Hydrology': self.horizontal_dist_hydrology[i, j],
            'Vertical_Distance_To_Hydrology': self.vertical_dist_hydrology[i, j],
            'Horizontal_Distance_To_Roadways': self.horizontal_dist_roadways[i, j],
            'Horizontal_Distance_To_Fire_Points': self.horizontal_dist_firepoints[i, j],
            'Hillshade_9am': self.hillshade_9am[i, j],
            'Hillshade_Noon': self.hillshade_noon[i, j],
            'Hillshade_3pm': self.hillshade_3pm[i, j],
        }
        
        # Add Wilderness_Area one-hot encoding
        for wa in range(1, 5):
            features[f'Wilderness_Area{wa}'] = 1 if self.wilderness_area[i, j] == wa else 0
        
        # Add Soil_Type one-hot encoding
        for st in range(1, 41):
            features[f'Soil_Type{st}'] = 1 if self.soil_type[i, j] == st else 0
        
        return pd.DataFrame([features])
    
    def _predict_with_model(self, i: int, j: int) -> Optional[int]:
        """Get model prediction for a cell"""
        if not self.model_loaded:
            return None
        
        try:
            df = self._create_feature_vector(i, j)
            
            # Apply preprocessor if available
            if self.preprocessor is not None:
                df_transformed = self.preprocessor.transform(df)
            else:
                df_transformed = df
            
            # Predict (LightGBM returns 1-7, we keep that)
            prediction = self.model.predict(df_transformed)[0]
            return int(prediction)
            
        except Exception as e:
            # Log first error only (to avoid spam)
            if not hasattr(self, '_prediction_error_logged'):
                logger.error(f"❌ Prediction failed at ({i},{j}): {e}")
                import traceback
                traceback.print_exc()
                self._prediction_error_logged = True
            return None
    
    def _apply_rules(self):
        """Apply cellular automata rules with optional model corrections"""
        new_grid = self.grid.copy()
        chaos_count = 0
        agreements = 0
        corrections = 0
        total_predictions = 0
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                current = self.grid[i, j]
                neighbors = self._get_neighbors(i, j)
                elev = self.elevation[i, j]
                zone = self._get_elevation_zone(elev)
                is_chaos = self._is_near_threshold(elev)
                
                # Count neighbor types
                neighbor_counts = {}
                for n in neighbors:
                    if n != 0:
                        neighbor_counts[n] = neighbor_counts.get(n, 0) + 1
                
                # ============================================
                # AUTOMATA RULES (Same as before)
                # ============================================
                automata_prediction = current
                
                # Rule 1: Colonization (empty cells)
                if current == 0:
                    non_empty = [n for n in neighbors if n != 0]
                    if len(non_empty) >= 3:
                        if neighbor_counts:
                            most_common = max(neighbor_counts.items(), key=lambda x: x[1])[0]
                            if most_common in zone['dominant']:
                                if np.random.random() < 0.3:
                                    automata_prediction = most_common
                
                # Rule 2: Ecological succession
                elif current != 0:
                    total_neighbors = len([n for n in neighbors if n != 0])
                    
                    # Succession transitions
                    if current == 2 and elev > 2800 and neighbor_counts.get(6, 0) >= 2:
                        if np.random.random() < 0.1:
                            automata_prediction = 6
                    
                    if current == 5 and elev < 2400 and neighbor_counts.get(1, 0) >= 2:
                        if np.random.random() < 0.1:
                            automata_prediction = 1
                    
                    if current == 3 and 2400 < elev < 2800 and neighbor_counts.get(2, 0) >= 2:
                        if np.random.random() < 0.08:
                            automata_prediction = 2
                    
                    # Rule 3: Competition (isolation)
                    if total_neighbors < 2:
                        if np.random.random() < 0.15:
                            automata_prediction = 0
                    
                    # Rule 4: Overcrowding
                    if total_neighbors == 8:
                        if np.random.random() < 0.1:
                            automata_prediction = 0
                
                # Rule 5: Chaos zone - amplified transitions
                if is_chaos and current not in zone['dominant'] and current != 0:
                    if np.random.random() < 0.2:
                        suitable = np.random.choice(zone['dominant'])
                        automata_prediction = suitable
                        chaos_count += 1
                
                # ============================================
                # MODEL VALIDATION & CORRECTION
                # ============================================
                if self.model_loaded and current != 0:
                    model_prediction = self._predict_with_model(i, j)
                    
                    if model_prediction is not None:
                        total_predictions += 1
                        
                        # Check agreement
                        if automata_prediction == model_prediction:
                            agreements += 1
                        
                        # Apply model correction if enabled
                        if self.use_model_corrections:
                            # Use model in chaos zones or when automata wants to change
                            if is_chaos or automata_prediction != current:
                                # Blend: 70% model, 30% automata in chaos zones
                                if is_chaos:
                                    if np.random.random() < 0.7:
                                        new_grid[i, j] = model_prediction
                                        corrections += 1
                                    else:
                                        new_grid[i, j] = automata_prediction
                                else:
                                    # Outside chaos: trust automata more
                                    new_grid[i, j] = automata_prediction
                            else:
                                new_grid[i, j] = automata_prediction
                        else:
                            new_grid[i, j] = automata_prediction
                else:
                    new_grid[i, j] = automata_prediction
        
        self.grid = new_grid
        self.generation += 1
        
        # Track statistics
        agreement_rate = agreements / total_predictions if total_predictions > 0 else 0
        self._update_statistics(chaos_count, agreement_rate, corrections)
    
    def _update_statistics(self, chaos_count: int, agreement_rate: float, corrections: int):
        """Update simulation statistics"""
        unique, counts = np.unique(self.grid, return_counts=True)
        populations = dict(zip(unique.tolist(), counts.tolist()))
        
        self.history['generation'].append(self.generation)
        self.history['populations'].append(populations)
        self.history['chaos_events'].append(chaos_count)
        self.history['model_agreements'].append(agreement_rate)
        self.history['model_corrections'].append(corrections)
    
    def run(self, generations: int = 100, verbose: bool = True):
        """Run simulation for N generations"""
        logger.info(f"Starting simulation for {generations} generations...")
        logger.info(f"Model corrections: {'ENABLED' if self.use_model_corrections else 'DISABLED'}")
        logger.info(f"Model loaded: {self.model_loaded}")
        
        # Test single prediction before running
        if self.model_loaded:
            logger.info("🧪 Testing model prediction on cell (0,0)...")
            test_pred = self._predict_with_model(0, 0)
            logger.info(f"✓ Test prediction result: {test_pred}")
        
        for gen in range(generations):
            self._apply_rules()
            
            if verbose and (gen + 1) % 10 == 0:
                avg_agreement = np.mean(self.history['model_agreements'][-10:])
                total_preds = sum(1 for a in self.history['model_agreements'][-10:] if a > 0)
                logger.info(f"Gen {gen + 1}/{generations} | Model agreement: {avg_agreement:.2%} | Predictions made: {total_preds}")
        
        logger.info("✓ Simulation complete")
    
    def visualize(self, save_path: str = None):
        """Visualize current state"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Forest cover map
        colors = [self.COVER_TYPES[i]['color'] for i in range(8)]
        cmap = ListedColormap(colors)
        
        im1 = axes[0].imshow(self.grid, cmap=cmap, vmin=0, vmax=7)
        axes[0].set_title(f'Forest Cover (Gen {self.generation})', 
                         fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        cbar1 = plt.colorbar(im1, ax=axes[0], ticks=range(8), fraction=0.046)
        cbar1.set_ticklabels([self.COVER_TYPES[i]['name'] for i in range(8)], fontsize=8)
        
        # Elevation map
        im2 = axes[1].imshow(self.elevation, cmap='terrain', vmin=1859, vmax=3858)
        axes[1].set_title('Elevation (m)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046)
        cbar2.set_label('Meters', rotation=270, labelpad=15, fontsize=9)
        
        # Chaos zones overlay
        chaos_mask = np.zeros_like(self.elevation)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self._is_near_threshold(self.elevation[i, j]):
                    chaos_mask[i, j] = 1
        
        axes[2].imshow(self.elevation, cmap='terrain', vmin=1859, vmax=3858, alpha=0.5)
        axes[2].imshow(chaos_mask, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
        axes[2].set_title('Chaos Zones (Red)', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Visualization saved to {save_path}")
        
        plt.show()
    
    def plot_statistics(self, save_path: str = None):
        """Plot comprehensive statistics"""
        if not self.history['generation']:
            logger.warning("No history to plot")
            return
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        generations = self.history['generation']
        
        # 1. Population dynamics
        ax1 = fig.add_subplot(gs[0, :])
        for cover_type in range(1, 8):
            populations = [pop.get(cover_type, 0) for pop in self.history['populations']]
            ax1.plot(generations, populations, 
                    label=self.COVER_TYPES[cover_type]['name'], linewidth=2)
        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_ylabel('Cell Count', fontsize=11)
        ax1.set_title('Population Dynamics Over Time', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # 2. Chaos events
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(generations, self.history['chaos_events'], 
                color='red', linewidth=2, label='Chaos Transitions')
        ax2.fill_between(generations, self.history['chaos_events'], alpha=0.3, color='red')
        ax2.set_xlabel('Generation', fontsize=11)
        ax2.set_ylabel('Transition Events', fontsize=11)
        ax2.set_title('Chaos Zone Activity', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Model agreement rate
        ax3 = fig.add_subplot(gs[1, 1])
        if self.model_loaded and self.history['model_agreements']:
            ax3.plot(generations, [a*100 for a in self.history['model_agreements']], 
                    color='blue', linewidth=2, label='Agreement Rate')
            ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
            ax3.set_xlabel('Generation', fontsize=11)
            ax3.set_ylabel('Agreement %', fontsize=11)
            ax3.set_title('Automata-Model Agreement', fontsize=12, fontweight='bold')
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 100)
        else:
            ax3.text(0.5, 0.5, 'Model not loaded', ha='center', va='center', fontsize=12)
            ax3.axis('off')
        
        # 4. Model corrections
        ax4 = fig.add_subplot(gs[2, 0])
        if self.use_model_corrections and self.history['model_corrections']:
            ax4.plot(generations, self.history['model_corrections'], 
                    color='green', linewidth=2, label='Model Corrections')
            ax4.fill_between(generations, self.history['model_corrections'], 
                           alpha=0.3, color='green')
            ax4.set_xlabel('Generation', fontsize=11)
            ax4.set_ylabel('Corrections', fontsize=11)
            ax4.set_title('Model-Guided Corrections', fontsize=12, fontweight='bold')
            ax4.legend(loc='best', fontsize=9)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Model corrections disabled', ha='center', va='center', fontsize=12)
            ax4.axis('off')
        
        # 5. Cover type distribution (final state)
        ax5 = fig.add_subplot(gs[2, 1])
        final_pop = self.history['populations'][-1]
        types = [self.COVER_TYPES[k]['name'] for k in sorted(final_pop.keys()) if k != 0]
        counts = [final_pop[k] for k in sorted(final_pop.keys()) if k != 0]
        colors = [self.COVER_TYPES[k]['color'] for k in sorted(final_pop.keys()) if k != 0]
        
        ax5.barh(types, counts, color=colors, edgecolor='black', linewidth=0.5)
        ax5.set_xlabel('Cell Count', fontsize=11)
        ax5.set_title('Final Cover Type Distribution', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Statistics plot saved to {save_path}")
        
        plt.show()
    
    def validate_with_model(self, sample_rate: int = 5) -> Dict:
        """
        Comprehensive validation against LightGBM model
        
        Args:
            sample_rate: Sample every Nth cell (default: 5)
        
        Returns:
            Validation metrics dictionary
        """
        if not self.model_loaded:
            logger.warning("❌ No model loaded for validation")
            return {}
        
        logger.info("🔬 Validating automata output with LightGBM model...")
        
        automata_labels = []
        model_predictions = []
        elevations = []
        chaos_flags = []
        
        # Sample grid
        for i in range(0, self.grid_size, sample_rate):
            for j in range(0, self.grid_size, sample_rate):
                if self.grid[i, j] != 0:  # Skip empty cells
                    automata_labels.append(self.grid[i, j])
                    elevations.append(self.elevation[i, j])
                    chaos_flags.append(self._is_near_threshold(self.elevation[i, j]))
                    
                    # Get model prediction
                    pred = self._predict_with_model(i, j)
                    model_predictions.append(pred if pred is not None else self.grid[i, j])
        
        if len(automata_labels) == 0:
            logger.warning("⚠️  No samples to validate")
            return {}
        
        # Calculate metrics
        accuracy = accuracy_score(automata_labels, model_predictions)
        
        # Separate chaos vs non-chaos accuracy
        chaos_indices = [i for i, c in enumerate(chaos_flags) if c]
        normal_indices = [i for i, c in enumerate(chaos_flags) if not c]
        
        chaos_acc = 0
        normal_acc = 0
        
        if chaos_indices:
            chaos_actual = [automata_labels[i] for i in chaos_indices]
            chaos_pred = [model_predictions[i] for i in chaos_indices]
            chaos_acc = accuracy_score(chaos_actual, chaos_pred)
        
        if normal_indices:
            normal_actual = [automata_labels[i] for i in normal_indices]
            normal_pred = [model_predictions[i] for i in normal_indices]
            normal_acc = accuracy_score(normal_actual, normal_pred)
        
        # Confusion matrix
        cm = confusion_matrix(automata_labels, model_predictions, labels=list(range(1, 8)))
        
        results = {
            'overall_accuracy': accuracy,
            'chaos_zone_accuracy': chaos_acc,
            'normal_zone_accuracy': normal_acc,
            'total_samples': len(automata_labels),
            'chaos_samples': len(chaos_indices),
            'normal_samples': len(normal_indices),
            'confusion_matrix': cm.tolist()
        }
        
        # Print results
        logger.info("=" * 60)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Overall Accuracy:      {accuracy:.2%}")
        logger.info(f"Chaos Zone Accuracy:   {chaos_acc:.2%} ({len(chaos_indices)} samples)")
        logger.info(f"Normal Zone Accuracy:  {normal_acc:.2%} ({len(normal_indices)} samples)")
        logger.info(f"Total Samples:         {len(automata_labels)}")
        logger.info("=" * 60)
        
        return results
    
    def export_state(self, filepath: str):
        """Export simulation state to JSON"""
        state = {
            'generation': self.generation,
            'grid_size': self.grid_size,
            'model_loaded': self.model_loaded,
            'use_model_corrections': self.use_model_corrections,
            'grid': self.grid.tolist(),
            'elevation': self.elevation.tolist(),
            'history': {
                'generations': self.history['generation'],
                'populations': self.history['populations'],
                'chaos_events': self.history['chaos_events'],
                'model_agreements': self.history['model_agreements'],
                'model_corrections': self.history['model_corrections']
            },
            'metadata': {
                'cover_types': self.COVER_TYPES,
                'elevation_zones': self.ELEVATION_ZONES,
                'chaos_thresholds': self.CHAOS_THRESHOLDS
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"✓ State exported to {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run forest automata simulation with LightGBM integration"""
    print("=" * 70)
    print("🌲 FOREST COVER TYPE CELLULAR AUTOMATA + LIGHTGBM INTEGRATION")
    print("=" * 70)
    
    # Configuration
    GRID_SIZE = 100
    GENERATIONS = 50
    BASE_DIR = Path(__file__).parent
    MODEL_PATH = Path(r"C:\Users\nikka\Downloads\Forest-Cover-Type-Prediction----Kaggle\Forest-Cover-Type-Prediction----Kaggle\ForestTypePredict\src\models\lightgbm_model.pkl")
    PREPROCESSOR_PATH = Path(r"C:\Users\nikka\Downloads\Forest-Cover-Type-Prediction----Kaggle\Forest-Cover-Type-Prediction----Kaggle\ForestTypePredict\src\data\artifacts\preprocessor.pkl")
    USE_MODEL_CORRECTIONS = True
    
    # Initialize simulation
    print(f"\n🔧 Initializing simulation (Grid: {GRID_SIZE}x{GRID_SIZE})...")
    sim = ForestAutomataSimulation(
        grid_size=GRID_SIZE,
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        use_model_corrections=USE_MODEL_CORRECTIONS
    )
    
    # Visualize initial state
    print("\n📊 Initial State:")
    sim.visualize(save_path="automata_initial_integrated.png")
    
    # Run simulation
    print(f"\n🚀 Running simulation for {GENERATIONS} generations...")
    sim.run(generations=GENERATIONS, verbose=True)
    
    # Visualize final state
    print("\n📊 Final State:")
    sim.visualize(save_path="automata_final_integrated.png")
    
    # Plot comprehensive statistics
    print("\n📈 Generating Statistics...")
    sim.plot_statistics(save_path="automata_statistics_integrated.png")
    
    # Validate with model
    if sim.model_loaded:
        print("\n🔬 Running Model Validation...")
        validation = sim.validate_with_model(sample_rate=3)
        
        if validation:
            print("\n📊 Key Metrics:")
            print(f"  • Overall Accuracy: {validation['overall_accuracy']:.2%}")
            print(f"  • Chaos Zone Accuracy: {validation['chaos_zone_accuracy']:.2%}")
            print(f"  • Normal Zone Accuracy: {validation['normal_zone_accuracy']:.2%}")
    
    # Export state
    print("\n💾 Exporting State...")
    sim.export_state("automata_state_integrated.json")
    
    print("\n" + "=" * 70)
    print("✅ SIMULATION COMPLETE")
    print("=" * 70)
    print("\nOutputs:")
    print("  📸 automata_initial_integrated.png")
    print("  📸 automata_final_integrated.png")
    print("  📊 automata_statistics_integrated.png")
    print("  💾 automata_state_integrated.json")
    
    # Summary
    if sim.model_loaded:
        avg_agreement = np.mean(sim.history['model_agreements'])
        total_corrections = sum(sim.history['model_corrections'])
        print(f"\n🤝 Model Integration Summary:")
        print(f"  • Average Agreement Rate: {avg_agreement:.2%}")
        print(f"  • Total Model Corrections: {total_corrections}")
        print(f"  • Correction Mode: {'ENABLED' if USE_MODEL_CORRECTIONS else 'DISABLED'}")


if __name__ == "__main__":
    main()

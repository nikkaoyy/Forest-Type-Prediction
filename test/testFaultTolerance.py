import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path to import data_validation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataValidation import CircuitBreaker, ForestDataValidator, CircuitState
import logging

logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker fault isolation"""
    
    def setUp(self):
        """Initialize test fixtures"""
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=60)
    
    def test_circuit_remains_closed_on_success(self):
        """Circuit should remain CLOSED on successful calls"""
        def successful_function():
            return "success"
        
        result = self.circuit_breaker.call(successful_function)
        
        self.assertEqual(result, "success")
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 0)
    
    def test_circuit_opens_after_threshold_failures(self):
        """Circuit should OPEN after exceeding failure threshold"""
        def failing_function():
            raise ValueError("Simulated failure")
        
        # Trigger failures
        for i in range(3):
            with self.assertRaises(ValueError):
                self.circuit_breaker.call(failing_function)
        
        self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)
        self.assertEqual(self.circuit_breaker.failure_count, 3)
    
    def test_open_circuit_rejects_requests(self):
        """OPEN circuit should reject all requests"""
        def failing_function():
            raise ValueError("Failure")
        
        # Open the circuit
        for i in range(3):
            try:
                self.circuit_breaker.call(failing_function)
            except ValueError:
                pass
        
        # Circuit should be OPEN now
        self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)
        
        # Verify rejection
        with self.assertRaises(Exception) as context:
            self.circuit_breaker.call(failing_function)
        
        self.assertIn("Circuit breaker is OPEN", str(context.exception))


class TestDataValidationFaultTolerance(unittest.TestCase):
    """Test data validation with fault tolerance"""
    
    def setUp(self):
        """Initialize validator and test data"""
        self.validator = ForestDataValidator()
        
        # Valid dataset with EXACTLY 54 features (matching EXPECTED_FEATURES)
        np.random.seed(42)
        self.valid_data = pd.DataFrame({
            'Elevation': np.random.randint(1859, 3858, 100),
            'Aspect': np.random.randint(0, 360, 100),
            'Slope': np.random.randint(0, 40, 100),
            'Horizontal_Distance_To_Hydrology': np.random.randint(0, 400, 100),
            'Vertical_Distance_To_Hydrology': np.random.randint(-150, 150, 100),
            'Horizontal_Distance_To_Roadways': np.random.randint(0, 7000, 100),
            'Horizontal_Distance_To_Fire_Points': np.random.randint(0, 7000, 100),
            'Hillshade_9am': np.random.randint(0, 255, 100),
            'Hillshade_Noon': np.random.randint(0, 255, 100),
            'Hillshade_3pm': np.random.randint(0, 255, 100),
            # Add Wilderness Area (4 features)
            **{f'Wilderness_Area{i}': np.random.choice([0, 1], 100) for i in range(1, 5)},
            # Add Soil Types (40 features)
            **{f'Soil_Type{i}': np.random.choice([0, 1], 100) for i in range(1, 41)}
        })
        
        # Verify we have exactly 54 features
        assert self.valid_data.shape[1] == 54, f"Expected 54 features, got {self.valid_data.shape[1]}"
    
    def test_validation_passes_for_valid_data(self):
        """Valid data should pass all checks"""
        result = self.validator.validate_schema(self.valid_data)
        
        self.assertTrue(result.is_valid, f"Validation failed with errors: {result.errors}")
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(result.metrics['null_percentage'], 0.0)
    
    def test_validation_detects_missing_features(self):
        """Validator should detect missing features"""
        invalid_data = self.valid_data.drop(columns=['Soil_Type10', 'Soil_Type11'])
        
        result = self.validator.validate_schema(invalid_data)
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
        self.assertIn("Expected 54 features", result.errors[0])
    
    def test_validation_detects_out_of_range_elevation(self):
        """Validator should detect elevation range violations"""
        invalid_data = self.valid_data.copy()
        invalid_data.loc[0, 'Elevation'] = 5000  # Out of range
        
        result = self.validator.validate_schema(invalid_data)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any('Elevation out of range' in error for error in result.errors))
    
    def test_drift_detection_identifies_distribution_shift(self):
        """Drift detection should identify shifted distributions"""
        # Create drifted dataset
        drifted_data = self.valid_data.copy()
        drifted_data['Elevation'] = drifted_data['Elevation'] + 300  # Significant shift
        
        drift_detected, report = self.validator.detect_drift(
            df_production=drifted_data,
            df_baseline=self.valid_data,
            threshold=0.05
        )
        
        self.assertTrue(drift_detected)
        self.assertIn('Elevation', report['affected_features'])
        self.assertLess(report['p_values']['Elevation'], 0.05)
    
    def test_circuit_breaker_integration(self):
        """Circuit breaker should protect validation pipeline"""
        # Reset circuit breaker for this test
        self.validator.circuit_breaker = CircuitBreaker(failure_threshold=3)
        
        # Create malformed data that will cause repeated failures
        invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
        
        # Trigger multiple failures
        for i in range(3):
            result = self.validator.validate_with_circuit_breaker(invalid_data)
            self.assertFalse(result.is_valid)
        
        # Verify circuit opened
        self.assertEqual(
            self.validator.circuit_breaker.state, 
            CircuitState.OPEN
        )


class TestGracefulDegradation(unittest.TestCase):
    """Test graceful degradation scenarios"""
    
    def setUp(self):
        """Initialize test data"""
        np.random.seed(42)
    
    def test_partial_feature_availability(self):
        """System should handle partial feature availability"""
        partial_data = pd.DataFrame({
            'Elevation': np.random.randint(2000, 3000, 50),
            'Aspect': np.random.randint(0, 360, 50),
            # Missing Slope and other features
        })
        
        validator = ForestDataValidator()
        result = validator.validate_schema(partial_data)
        
        # Should fail validation but not crash
        self.assertFalse(result.is_valid)
        self.assertIsInstance(result.errors, list)
    
    def test_null_value_handling(self):
        """System should gracefully handle null values"""
        # Create valid 54-feature dataset first
        data_with_nulls = pd.DataFrame({
            'Elevation': [2500, None, 2700, 2900],
            'Aspect': [180, 90, None, 270],
            'Slope': [15, 20, 25, None],
            'Horizontal_Distance_To_Hydrology': [100, 200, 300, 400],
            'Vertical_Distance_To_Hydrology': [-50, 0, 50, 100],
            'Horizontal_Distance_To_Roadways': [500, 1000, 1500, 2000],
            'Horizontal_Distance_To_Fire_Points': [600, 1200, 1800, 2400],
            'Hillshade_9am': [100, 150, 200, 250],
            'Hillshade_Noon': [200, 220, 240, 255],
            'Hillshade_3pm': [100, 120, 140, 160],
            **{f'Wilderness_Area{i}': [1, 0, 0, 1] for i in range(1, 5)},
            **{f'Soil_Type{i}': [0, 1, None, 0] for i in range(1, 41)}
        })
        
        validator = ForestDataValidator()
        result = validator.validate_schema(data_with_nulls)
        
        # Should detect nulls but provide metrics
        self.assertGreater(len(result.warnings), 0)
        self.assertIn('null_percentage', result.metrics)
        self.assertGreater(result.metrics['null_percentage'], 0)
    
    def test_extreme_elevation_values(self):
        """System should handle extreme elevation values"""
        extreme_data = pd.DataFrame({
            'Elevation': [1859, 3858, 2500, 2800],  # Min, max, and normal values
            'Aspect': [0, 180, 90, 270],
            'Slope': [0, 40, 20, 30],
            'Horizontal_Distance_To_Hydrology': [100, 200, 300, 400],
            'Vertical_Distance_To_Hydrology': [-50, 0, 50, 100],
            'Horizontal_Distance_To_Roadways': [500, 1000, 1500, 2000],
            'Horizontal_Distance_To_Fire_Points': [600, 1200, 1800, 2400],
            'Hillshade_9am': [100, 150, 200, 250],
            'Hillshade_Noon': [200, 220, 240, 255],
            'Hillshade_3pm': [100, 120, 140, 160],
            **{f'Wilderness_Area{i}': [1, 0, 0, 1] for i in range(1, 5)},
            **{f'Soil_Type{i}': [0, 1, 0, 0] for i in range(1, 41)}
        })
        
        validator = ForestDataValidator()
        result = validator.validate_schema(extreme_data)
        
        # Should pass validation (values are within range)
        self.assertTrue(result.is_valid, f"Validation failed with errors: {result.errors}")


def run_fault_tolerance_tests():
    """Run complete fault tolerance test suite"""
    print("=" * 70)
    print("FAULT TOLERANCE TEST SUITE")
    print("=" * 70)
    print("NOTE: This test suite uses SIMULATED data")
    print("      No train.csv file is required")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestCircuitBreaker))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidationFaultTolerance))
    suite.addTests(loader.loadTestsFromTestCase(TestGracefulDegradation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print(f"Success rate: 100.0% ✅")
        print("\n✅ All tests passed! Your validation code is working correctly.")
        print("   You can now proceed to validate the real dataset with:")
        print("   → python dataValidation.py")
    else:
        success_count = result.testsRun - len(result.failures) - len(result.errors)
        success_rate = (success_count / result.testsRun * 100) if result.testsRun > 0 else 0
        print(f"Success rate: {success_rate:.1f}% ⚠️")
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_fault_tolerance_tests()
    exit(0 if success else 1)

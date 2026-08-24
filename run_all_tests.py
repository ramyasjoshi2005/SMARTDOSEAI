import sys
import unittest

# Ensure the root directory is in python search path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import unit tests
from tests.test_validation import *
from tests.test_smart_dose import *
from tests.test_app_routes import TestAppRoutes

class TestValidationWrapper(unittest.TestCase):
    def test_validation_diabetes_missing_critical(self):
        test_validation_diabetes_missing_critical()

    def test_validation_diabetes_out_of_range(self):
        test_validation_diabetes_out_of_range()

    def test_validation_diabetes_valid(self):
        test_validation_diabetes_valid()

    def test_validation_kidney_missing_critical(self):
        test_validation_kidney_missing_critical()

    def test_validation_kidney_valid(self):
        test_validation_kidney_valid()

    def test_validation_aliases(self):
        test_validation_aliases()

class TestSmartDoseWrapper(unittest.TestCase):
    def test_get_smart_dose_diabetes_mild_no_contra(self):
        test_get_smart_dose_diabetes_mild_no_contra()

    def test_get_smart_dose_diabetes_mild_contra(self):
        test_get_smart_dose_diabetes_mild_contra()

    def test_get_smart_dose_heart_mild_no_contra(self):
        test_get_smart_dose_heart_mild_no_contra()

    def test_get_smart_dose_heart_mild_contra(self):
        test_get_smart_dose_heart_mild_contra()

    def test_get_smart_dose_missing_creatinine_unknown_status(self):
        test_get_smart_dose_missing_creatinine_unknown_status()

if __name__ == "__main__":
    print("==================================================")
    print("Running SmartDoseAI Upgrade Test Suite")
    print("==================================================")
    
    suite = unittest.TestSuite()
    
    # Add app route tests
    suite.addTest(unittest.makeSuite(TestAppRoutes))
    # Add validation tests
    suite.addTest(unittest.makeSuite(TestValidationWrapper))
    # Add smart dose tests
    suite.addTest(unittest.makeSuite(TestSmartDoseWrapper))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        sys.exit(1)

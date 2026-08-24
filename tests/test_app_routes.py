import unittest
import os
import json
from app import app

class TestAppRoutes(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_login_page(self):
        rv = self.client.get("/")
        self.assertEqual(rv.status_code, 200)
        self.assertIn(b"Login", rv.data)

    def test_reception_dashboard(self):
        rv = self.client.get("/reception_dashboard")
        self.assertEqual(rv.status_code, 200)

    def test_doctor_dashboard(self):
        rv = self.client.get("/doctor_dashboard/Dr.%20Sharma")
        self.assertEqual(rv.status_code, 200)
        self.assertIn(b"Dr. Sharma", rv.data)

    def test_patient_list(self):
        rv = self.client.get("/history")
        self.assertEqual(rv.status_code, 200)

    def test_select_patient_for_twin(self):
        rv = self.client.get("/digital_twin")
        self.assertEqual(rv.status_code, 200)
        self.assertIn(b"Digital Twin Simulation", rv.data)

    def test_ml_results_page(self):
        rv = self.client.get("/ml_results")
        self.assertEqual(rv.status_code, 200)
        self.assertIn(b"ML Results", rv.data)

    def test_digital_twin_get(self):
        rv = self.client.get("/digital_twin/neha%20kulkarni")
        self.assertEqual(rv.status_code, 200)
        self.assertIn(b"neha kulkarni", rv.data)

    def test_digital_twin_post(self):
        rv = self.client.post("/digital_twin/neha%20kulkarni", data={
            "sim_HbA1c": "7.0",
            "sim_Fasting_Glucose": "130.0"
        })
        self.assertEqual(rv.status_code, 200)
        self.assertIn(b"neha kulkarni", rv.data)

if __name__ == "__main__":
    unittest.main()

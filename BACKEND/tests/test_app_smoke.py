import os

os.environ.setdefault("SKIP_STARTUP_LOAD", "1")

from fastapi.testclient import TestClient
from app import app


def test_predict_endpoint_smoke():
    client = TestClient(app)
    payload = {
        "person_age": 28,
        "person_income": 45000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3,
        "loan_intent": "EDUCATION",
        "loan_amnt": 5000,
        "loan_int_rate": 9.5,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "logistic_regression" in data


def test_metrics_summary_smoke():
    client = TestClient(app)
    response = client.get("/metrics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "latency_ms" in data

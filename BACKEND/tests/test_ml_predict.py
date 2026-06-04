from ml_predict import predict


def test_predict_returns_keys():
    payload = {
        "person_age": 35,
        "person_income": 52000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5,
        "loan_intent": "PERSONAL",
        "loan_amnt": 8000,
        "loan_int_rate": 12.5,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 7,
    }
    result = predict(payload)
    assert "logistic_regression" in result
    assert "random_forest" in result
    lr = result["logistic_regression"]
    if lr:
        assert "probability" in lr
        assert "confidence_band" in lr
    rf = result["random_forest"]
    if rf:
        assert "probability" in rf
        assert "confidence_band" in rf

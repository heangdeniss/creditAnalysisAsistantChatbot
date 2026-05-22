from feature_store import encode_applicant, feature_contract, standardize_applicant
from job_queue import get_job, submit_job
from model_server_client import external_generation_enabled
from time import sleep


def _sample_applicant():
    return {
        "person_age": 35,
        "person_income": 52000,
        "person_home_ownership": "rent",
        "person_emp_length": 5,
        "loan_intent": "personal",
        "loan_amnt": 8000,
        "loan_int_rate": 12.5,
        "cb_person_default_on_file": "n",
        "cb_person_cred_hist_length": 7,
    }


def test_feature_contract_standardizes_and_encodes():
    applicant = standardize_applicant(_sample_applicant())
    assert applicant["person_home_ownership"] == "RENT"
    assert applicant["loan_intent"] == "PERSONAL"

    vector = encode_applicant(applicant)
    contract = feature_contract()
    assert len(vector) == len(contract["feature_columns"])


def test_in_process_job_queue_runs_callable():
    job = submit_job("unit_test", lambda: {"ok": True})
    assert job["status"] in {"queued", "running", "succeeded"}

    finished = None
    for _ in range(100):
        current = get_job(job["job_id"])
        if current and current["status"] == "succeeded":
            finished = current
            break
        sleep(0.01)

    assert finished is not None
    assert finished["result"] == {"ok": True}


def test_model_server_is_disabled_by_default():
    assert external_generation_enabled() is False

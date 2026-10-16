import os
os.environ.setdefault("PYTHONPATH", ".")
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def _seed_events(experiment_id: str, control_n: int, control_succ: int,
                 treatment_n: int, treatment_succ: int):
    events = []
    for i in range(control_n):
        events.append({"event_id": f"{experiment_id}_c_{i}", "experiment_id": experiment_id,
                       "variant": "control", "user_id": f"c{i}", "metric": "thumbs",
                       "value": 1 if i < control_succ else -1})
    for i in range(treatment_n):
        events.append({"event_id": f"{experiment_id}_t_{i}", "experiment_id": experiment_id,
                       "variant": "treatment", "user_id": f"t{i}", "metric": "thumbs",
                       "value": 1 if i < treatment_succ else -1})
    r = client.post("/events", json={"events": events})
    assert r.status_code == 200


def test_results_empty_experiment():
    r = client.get("/experiments/exp_hybrid_vs_vector/results")
    assert r.status_code == 200
    body = r.json()
    assert body["experiment_id"] == "exp_hybrid_vs_vector"
    assert body["status"] == "running"
    assert "variants" in body


def test_results_are_computed_from_events():
    exp = "exp_test_stats"
    r = client.post("/experiments", json={"id": exp, "name": "stats test"})
    # is 409 if it already exists from a previous run; ignore
    assert r.status_code in (201, 409)

    _seed_events(exp, control_n=200, control_succ=20, treatment_n=200, treatment_succ=40)

    r = client.get(f"/experiments/{exp}/results")
    assert r.status_code == 200
    body = r.json()

    rates = {v["variant"]: v["thumbs_up_rate"] for v in body["variants"]}
    assert rates["control"] == 0.10
    assert rates["treatment"] == 0.20

    # treatment clearly better -> z-test significant
    assert body["test"] is not None
    assert body["test"]["significant"] is True
    assert body["test"]["absolute_lift"] == 0.10
    assert body["verdict"] == "treatment better"
    # SRM should be fine on a 200/200 split
    assert body["srm"]["srm_detected"] is False


def test_events_are_idempotent_within_experiment():
    exp = "exp_idem"
    client.post("/experiments", json={"id": exp, "name": "idem"})
    payload = {"events": [
        {"event_id": "same_id", "experiment_id": exp, "variant": "control",
         "user_id": "x", "metric": "thumbs", "value": 1}
    ]}
    assert client.post("/events", json=payload).json()["stored"] == 1
    # same event_id again -> no-op
    assert client.post("/events", json=payload).json()["stored"] == 0

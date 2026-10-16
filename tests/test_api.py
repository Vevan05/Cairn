from fastapi.testclient import TestClient
from api.main import app
import json

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "vector_count" in body


def test_search_basic():
    r = client.post("/search", json={"query": "hello", "user_id": "u1"})
    assert r.status_code == 200
    body = r.json()
    assert body["variant"] in ("control", "treatment")
    assert "results" in body
    assert body["experiment_id"] == "exp_hybrid_vs_vector"


def test_events_idempotent():
    payload = {"events": [
        {"event_id": "e1", "variant": "control", "user_id": "u", "event_type": "thumbs", "value": 1},
    ]}
    r1 = client.post("/events", json=payload)
    r2 = client.post("/events", json=payload)
    assert r1.status_code == 200
    assert r2.status_code == 200
    # both should report stored exactly once for the single unique id
    assert r1.json()["stored"] == 1
    assert r2.json()["stored"] == 1


def test_create_and_list_experiments():
    client.post("/experiments", json={"id": "expX", "name": "TestExp", "status": "running"})
    r = client.get("/experiments")
    assert r.status_code == 200
    exp_ids = [e["id"] for e in r.json()["experiments"]]
    assert "expX" in exp_ids


def test_experiment_results():
    client.post("/experiments", json={"id": "expY", "name": "Another", "status": "running"})
    r = client.get("/experiments/expY/results")
    assert r.status_code == 200
    assert r.json()["experiment_id"] == "expY"

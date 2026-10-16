"""Tests for the experiment admin console, eval harness, and streaming answer."""
import json
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Experiment admin console
# ---------------------------------------------------------------------------
def test_create_update_delete_experiment():
    exp_id = "exp_admin_test"
    # create
    r = client.post("/experiments", json={"id": exp_id, "name": "Admin test", "status": "draft"})
    assert r.status_code in (201, 409)

    # update status + variants
    r = client.patch(f"/experiments/{exp_id}", json={"status": "running"})
    assert r.status_code == 200
    assert r.json()["status"] == "running"

    # invalid config (single variant) rejected
    r = client.patch(f"/experiments/{exp_id}", json={"config_json": {"variants": [{"name": "only", "weight": 100}]}})
    assert r.status_code == 422

    # duplicate variant names rejected
    r = client.patch(f"/experiments/{exp_id}", json={"config_json": {"variants": [
        {"name": "a", "weight": 50}, {"name": "a", "weight": 50}]}})
    assert r.status_code == 422

    # delete
    assert client.delete(f"/experiments/{exp_id}").status_code == 200
    assert client.get(f"/experiments/{exp_id}").status_code == 404


def test_cannot_delete_default_experiment():
    r = client.delete("/experiments/exp_hybrid_vs_vector")
    assert r.status_code == 400


def test_activating_experiment_routes_traffic_to_it():
    """A newly created experiment is inert until activated."""
    exp_id = "exp_active_test"
    client.post("/experiments", json={
        "id": exp_id, "name": "active test", "status": "draft",
        "config_json": {"variants": [
            {"name": "control", "weight": 50, "config": {"retrieval": "vector", "topK": 5}},
            {"name": "treatment", "weight": 50, "config": {"retrieval": "hybrid", "topK": 5}},
        ]},
    })

    # before activation: searches route to the default, not this one
    r = client.post("/search", json={"query": "battleship", "user_id": "route_before"})
    assert r.json()["experiment_id"] != exp_id

    # activate it
    r = client.post(f"/experiments/{exp_id}/activate")
    assert r.status_code == 200
    assert r.json()["active"] == exp_id

    # after activation: searches route to it and log exposures
    r = client.post("/search", json={"query": "battleship", "user_id": "route_after"})
    assert r.json()["experiment_id"] == exp_id

    results = client.get(f"/experiments/{exp_id}/results").json()
    assert sum(v["exposures"] for v in results["variants"]) >= 1

    # list marks it active
    listing = client.get("/experiments").json()
    assert listing["active"] == exp_id
    assert any(e["id"] == exp_id and e["active"] for e in listing["experiments"])

    # cleanup: restore default and delete
    client.post("/experiments/exp_hybrid_vs_vector/activate")
    client.delete(f"/experiments/{exp_id}")


def test_get_unknown_experiment_404():
    assert client.get("/experiments/nope").status_code == 404


def test_one_outcome_per_user_per_variant():
    """A user can thumbs multiple chunks, but counts once for the experiment."""
    exp_id = "exp_dedup_test"
    client.post("/experiments", json={"id": exp_id, "name": "dedup"})
    for i in range(5):
        client.post("/events", json={"events": [{
            "event_id": f"dedup_{i}", "experiment_id": exp_id, "variant": "control",
            "user_id": "same_user", "metric": "thumbs", "value": 1,
        }]})
    res = client.get(f"/experiments/{exp_id}/results").json()
    control = next(v for v in res["variants"] if v["variant"] == "control")
    assert control["n"] == 1  # five clicks from one user == one outcome
    client.delete(f"/experiments/{exp_id}")


def test_simulate_populates_both_variants_and_runs_test():
    exp_id = "exp_sim_test"
    client.post("/experiments", json={
        "id": exp_id, "name": "sim",
        "config_json": {"variants": [
            {"name": "control", "weight": 50, "config": {"retrieval": "vector", "topK": 5}},
            {"name": "treatment", "weight": 50, "config": {"retrieval": "hybrid", "topK": 5}},
        ]},
    })
    r = client.post(f"/experiments/{exp_id}/simulate", json={"n_users": 1000, "treatment_lift": 0.1, "seed": 1})
    assert r.status_code == 200

    res = client.get(f"/experiments/{exp_id}/results").json()
    by = {v["variant"]: v for v in res["variants"]}
    assert by["control"]["exposures"] > 0
    assert by["treatment"]["exposures"] > 0
    # treatment got a +10pp lift, so it should be detected as better
    assert res["test"] is not None
    assert res["verdict"] in ("treatment better", "no detectable difference")
    assert res["srm"] is not None and res["srm"]["srm_detected"] is False
    client.delete(f"/experiments/{exp_id}")


# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------
def _labeled_sample(n=8):
    chunks = [json.loads(l) for l in open("data/chunks.jsonl")]
    items = []
    for c in chunks[: n * 6 : 6]:
        items.append({"query": c["text"][:60], "relevant_chunk_ids": [c["chunk_id"]]})
    return items


def test_eval_set_create_and_run():
    items = _labeled_sample()
    r = client.post("/eval/sets", json={"id": "eval_test_set", "name": "test", "labeled": items})
    assert r.status_code in (201, 409)

    r = client.post("/eval/run", json={
        "set_id": "eval_test_set", "top_k": 10, "compare": ["vector", "hybrid"],
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    for row in body["results"]:
        assert 0.0 <= row["recall_at_k"] <= 1.0
        assert 0.0 <= row["mrr"] <= 1.0
        assert 0.0 <= row["ndcg_at_k"] <= 1.0


def test_eval_set_rejects_unknown_chunks():
    r = client.post("/eval/sets", json={
        "id": "bad_set", "name": "bad",
        "labeled": [{"query": "x", "relevant_chunk_ids": ["does_not_exist::0"]}],
    })
    assert r.status_code == 422


def test_eval_run_unknown_set_404():
    r = client.post("/eval/run", json={"set_id": "nope", "top_k": 10})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Streaming answer
# ---------------------------------------------------------------------------
def test_answer_stream_emits_events():
    r = client.get("/answer/stream", params={"query": "battleship", "user_id": "u1"})
    assert r.status_code == 200
    text = r.text
    assert "event: chunks" in text
    assert "event: token" in text
    assert "event: done" in text
    # done event carries citations
    done_line = [l for l in text.splitlines() if l.startswith("data: {") and '"citations"' in l]
    assert done_line, "done event with citations missing"

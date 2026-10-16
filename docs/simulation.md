# Simulating an experiment

Cairn ships with a traffic simulator so an A/B test can reach statistical
significance without real users. This is the same guide as the in-app
**Docs** page (`/docs`).

## Why

A new experiment starts empty, and live traffic is a trickle — reaching
significance can take thousands of users. An A/B test also needs both arms
populated; with a single visitor all traffic lands in one variant and the
z-test cannot run.

Simulation generates many distinct users, assigns each with the **real**
bucketing function, and logs exposures and thumbs with a configurable lift.
Everything goes through the same helpers as live traffic, so results,
p-values, and the SRM check are computed identically.

## UI

1. Experiments tab → create an experiment (e.g. control 50 / treatment 50).
2. **Activate** it so it serves `/search` traffic.
3. In the **Simulate traffic** row set `users` and `lift`.
4. Click **Run simulation** and watch the tables and stat cards populate.

| Field | Meaning | Typical |
|---|---|---|
| `users` | number of synthetic users | 500–2000 |
| `lift` | absolute boost to thumbs-up rate for non-control variants | 0.05–0.15 |

The first variant is the control; others get `control_rate + lift`
(base control rate is 10%).

## Scenario cheat-sheet

| Goal | users | lift | Expected verdict |
|---|---|---|---|
| Clear winner | 2000 | 0.15 | treatment better |
| Modest effect | 1000 | 0.08 | treatment better |
| A/A (no real difference) | 1500 | 0.00 | no detectable difference |
| Underpowered | 60 | 0.10 | not enough data (provisional) |
| Effect too small to detect | 800 | 0.01 | no detectable difference |

## API

```bash
# create + activate
curl -s -X POST localhost:8000/experiments \
  -H 'Content-Type: application/json' \
  -d '{"id":"demo","name":"Demo","config_json":{"variants":[
       {"name":"control","weight":50,"config":{"retrieval":"vector","topK":5}},
       {"name":"treatment","weight":50,"config":{"retrieval":"hybrid","topK":5}}]}}'
curl -s -X POST localhost:8000/experiments/demo/activate

# simulate 1000 users with a +10pp treatment lift (seed = reproducible)
curl -s -X POST localhost:8000/experiments/demo/simulate \
  -H 'Content-Type: application/json' \
  -d '{"n_users":1000,"treatment_lift":0.10,"seed":1}'

# read the computed stats
curl -s localhost:8000/experiments/demo/results
```

## Reading the results

- **Thumbs-up rate** — successes ÷ outcomes per variant (primary metric).
- **absolute lift** — treatment rate − control rate (percentage points).
- **z / p-value** — two-proportion z-test; `p < 0.05` is significant.
- **SRM** — chi-square check that the split matches configured weights;
  a flagged SRM means a bucketing/logging bug.
- **verdict** — plain-language summary, guarded against peeking
  ("not enough data" below 100 outcomes).

## Live vs simulated

- Real 👍/👎 count toward the same experiment.
- The unit of analysis is the **user**: at most one exposure and one vote per
  user per variant. Change the `user_id` box in the header to create a new user.
- Simulated and real traffic mix safely.

## Under the hood

```python
for each synthetic user:
    variant = assign_variant(exp_id, user_id, variants)   # SHA-256 bucketing
    log_exposure(exp_id, variant, user_id)
    if random() < feedback_rate:
        value = +1 if random() < (control_rate + lift_if_treatment) else -1
        log_outcome(exp_id, variant, "thumbs", value, user_id)
```

No special-casing: the results endpoint, z-test, and SRM chi-square read the
same stores that live traffic writes to.

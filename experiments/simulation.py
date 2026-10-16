"""Experiment simulations: see the A/B layer work end-to-end.

Run:  PYTHONPATH=. python experiments/simulation.py

Three simulations, each producing a table:
  1. A/A  — no real difference; false-positive rate should be ~alpha.
  2. A/B  — known lift; reports detection power.
  3. SRM  — intentionally broken bucketer; should be flagged.

Also prints live bucketing validation (split %, determinism, independence).
"""
from __future__ import annotations

import random
from collections import defaultdict

from experiments import (
    assign_variant,
    bucket_hash,
    two_proportion_z,
    welch_t,
    srm_chi_square,
)

VARIANTS = [{"name": "control", "weight": 50}, {"name": "treatment", "weight": 50}]


# ----------------------------------------------------------------------------
# Bucketing validation
# ----------------------------------------------------------------------------
def show_bucketing(n_users: int = 20000) -> None:
    print("\n" + "=" * 64)
    print("BUCKETING VALIDATION")
    print("=" * 64)
    counts: dict[str, int] = defaultdict(int)
    for i in range(n_users):
        uid = f"user_{i}"
        counts[assign_variant("exp_demo", uid, VARIANTS)] += 1

    total = sum(counts.values())
    print(f"Users: {total}")
    for name, c in counts.items():
        print(f"  {name:<10} {c:>7}  ({100 * c / total:.2f}%)  expected 50.00%")

    # determinism
    same = all(
        assign_variant("exp_demo", f"user_{i}", VARIANTS)
        == assign_variant("exp_demo", f"user_{i}", VARIANTS)
        for i in range(1000)
    )
    print(f"  determinism (same user -> same variant): {same}")

    # independence across experiments
    corr_sample = 5000
    x = [bucket_hash("exp_A", f"u{i}") for i in range(corr_sample)]
    y = [bucket_hash("exp_B", f"u{i}") for i in range(corr_sample)]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / len(x)
    sx = (sum((a - mx) ** 2 for a in x) / len(x)) ** 0.5
    sy = (sum((b - my) ** 2 for b in y) / len(y)) ** 0.5
    corr = cov / (sx * sy) if sx and sy else 0.0
    print(f"  cross-experiment correlation: {corr:+.4f} (want ~0)")


# ----------------------------------------------------------------------------
# A/A simulation
# ----------------------------------------------------------------------------
def simulate_aa(n_trials: int = 1000, n_per_arm: int = 2000, base_rate: float = 0.10,
                alpha: float = 0.05, seed: int = 7) -> float:
    rng = random.Random(seed)
    false_pos = 0
    for _ in range(n_trials):
        c = sum(rng.random() < base_rate for _ in range(n_per_arm))
        t = sum(rng.random() < base_rate for _ in range(n_per_arm))
        res = two_proportion_z(c / n_per_arm, n_per_arm, t / n_per_arm, n_per_arm)
        if res["p_value"] < alpha:
            false_pos += 1
    fp_rate = false_pos / n_trials
    print("\n" + "=" * 64)
    print("A/A SIMULATION  (no real difference)")
    print("=" * 64)
    print(f"  trials:              {n_trials}")
    print(f"  per arm:             {n_per_arm}")
    print(f"  base rate:           {base_rate:.0%}")
    print(f"  false-positive rate: {fp_rate:.3f}   (target alpha={alpha})")
    print(f"  verdict:             {'OK' if abs(fp_rate - alpha) <= 0.02 else 'TOO HIGH/LOW — investigate'}")
    return fp_rate


# ----------------------------------------------------------------------------
# A/B simulation with known effect (power)
# ----------------------------------------------------------------------------
def simulate_ab(lift: float = 0.02, n_trials: int = 500, n_per_arm: int = 3000,
                base_rate: float = 0.10, alpha: float = 0.05, seed: int = 11) -> float:
    rng = random.Random(seed)
    detected = 0
    for _ in range(n_trials):
        c = sum(rng.random() < base_rate for _ in range(n_per_arm))
        t = sum(rng.random() < (base_rate + lift) for _ in range(n_per_arm))
        res = two_proportion_z(c / n_per_arm, n_per_arm, t / n_per_arm, n_per_arm)
        if res["p_value"] < alpha:
            detected += 1
    power = detected / n_trials
    print("\n" + "=" * 64)
    print("A/B SIMULATION  (known lift, measuring power)")
    print("=" * 64)
    print(f"  control rate:  {base_rate:.2%}")
    print(f"  treatment:     {base_rate + lift:.2%}   (lift +{lift:.2%})")
    print(f"  per arm:       {n_per_arm}   trials: {n_trials}")
    print(f"  detection power: {power:.3f}   (want >= 0.80)")
    print(f"  verdict:         {'OK' if power >= 0.80 else 'UNDERPOWERED — need more samples'}")
    return power


# ----------------------------------------------------------------------------
# SRM: broken bucketer detection
# ----------------------------------------------------------------------------
def show_srm() -> None:
    print("\n" + "=" * 64)
    print("SAMPLE RATIO MISMATCH  (intentionally broken bucketer)")
    print("=" * 64)
    expected = [5000, 5000]

    cases = [
        ("healthy 50/50",       [4987, 5013]),
        ("mild drift 52/48",    [5200, 4800]),
        ("broken 60/40",        [6000, 4000]),
        ("broken 90/10",        [9000, 1000]),
    ]
    print(f"  {'case':<20} {'observed':<18} {'chi2':>9} {'p':>8}  flag")
    for name, obs in cases:
        r = srm_chi_square(obs, expected)
        flag = "SRM_WARNING" if r["srm_detected"] else "ok"
        print(f"  {name:<20} {str(obs):<18} {r['chi2']:>9.2f} {r['p_value']:>8.2f}  {flag}")


# ----------------------------------------------------------------------------
# Full experiment: bucket -> outcomes -> stats
# ----------------------------------------------------------------------------
def simulate_full_experiment(n_users: int = 20000, control_rate: float = 0.10,
                             treatment_rate: float = 0.13, seed: int = 3) -> None:
    """End-to-end: assign users, simulate outcomes, compute the result table."""
    rng = random.Random(seed)
    exposures: dict[str, int] = defaultdict(int)
    successes: dict[str, int] = defaultdict(int)

    for i in range(n_users):
        uid = f"user_{i}"
        variant = assign_variant("exp_full", uid, VARIANTS)
        exposures[variant] += 1
        rate = control_rate if variant == "control" else treatment_rate
        if rng.random() < rate:
            successes[variant] += 1

    print("\n" + "=" * 64)
    print("FULL EXPERIMENT RUN  (bucketing + outcomes + stats)")
    print("=" * 64)
    print(f"  {'variant':<12} {'n':>7} {'thumbs_up':>10} {'rate':>8}")
    for v in ("control", "treatment"):
        n = exposures[v]
        s = successes[v]
        print(f"  {v:<12} {n:>7} {s:>10} {(s / n if n else 0):>8.2%}")

    c_n, t_n = exposures["control"], exposures["treatment"]
    c_p = successes["control"] / c_n
    t_p = successes["treatment"] / t_n

    z = two_proportion_z(c_p, c_n, t_p, t_n)
    ratio = srm_chi_square([c_n, t_n], [n_users // 2, n_users // 2])

    print(f"\n  primary metric (thumbs_up_rate):")
    print(f"    control   {c_p:.4f}")
    print(f"    treatment {t_p:.4f}")
    print(f"    absolute lift  {t_p - c_p:+.4f}")
    print(f"    relative lift  {100 * (t_p - c_p) / c_p:+.1f}%")
    print(f"    z = {z['z']}, p = {z['p_value']}, significant = {z['significant']}")
    print(f"    verdict: {'treatment better' if z['significant'] and t_p > c_p else 'no detectable difference'}")
    print(f"  SRM check: chi2={ratio['chi2']:.2f}, p={ratio['p_value']:.2f}, flag={ratio['flag']}")

    # latency example (Welch's t-test)
    lat_c, lat_t = 120.0, 114.0
    w = welch_t(lat_c, 400.0, c_n, lat_t, 380.0, t_n)
    print(f"\n  guardrail (p95 latency ms): control {lat_c}, treatment {lat_t}")
    print(f"    Welch t = {w['t']}, p = {w['p_value']}, significant = {w['significant']}")


def main() -> None:
    show_bucketing()
    simulate_aa()
    simulate_ab()
    show_srm()
    simulate_full_experiment()
    print("\n" + "=" * 64)
    print("Done. These are the numbers you'd surface in /experiments/{id}/results")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()

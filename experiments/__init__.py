"""Experiments: deterministic bucketing, stats, SRM - pure Python (standard library only), tested."""
from __future__ import annotations

import hashlib
import math

__all__ = ["bucket_hash", "assign_variant", "two_proportion_z", "welch_t", "srm_chi_square"]


def bucket_hash(experiment_id: str, user_id: str, mod: int = 10000) -> int:
    key = f"{experiment_id}:{user_id}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % mod


def assign_variant(experiment_id: str, user_id: str, variants: list[dict]) -> str:
    bucket = bucket_hash(experiment_id, user_id, mod=10000)
    total_weight = sum(v["weight"] for v in variants)
    cumulative = 0
    normalized_bucket = bucket / 10000.0 * total_weight
    for v in variants:
        cumulative += v["weight"]
        if normalized_bucket < cumulative:
            return v["name"]
    return variants[-1]["name"]


def two_proportion_z(p1: float, n1: int, p2: float, n2: int) -> dict:
    if n1 == 0 or n2 == 0:
        return {"z": 0.0, "p_value": 1.0, "significant": False}
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1:
        return {"z": 0.0, "p_value": 1.0, "significant": False}
    se = math.sqrt(p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2))
    z = (p1 - p2) / se
    from math import erfc
    p_value = erfc(abs(z) / math.sqrt(2))
    return {"z": round(z, 4), "p_value": round(p_value, 4), "significant": p_value < 0.05}


def welch_t(mean1: float, var1: float, n1: int, mean2: float, var2: float, n2: int) -> dict:
    if n1 == 0 or n2 == 0:
        return {"t": 0.0, "p_value": 1.0, "significant": False}
    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        return {"t": 0.0, "p_value": 1.0, "significant": False}
    t = (mean1 - mean2) / se
    numerator = (var1 / n1 + var2 / n2) ** 2
    denominator = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    df = numerator / denominator if denominator > 0 else min(n1, n2) - 1
    from math import erfc
    p_value = erfc(abs(t) / math.sqrt(2))
    return {"t": round(t, 4), "p_value": round(p_value, 4), "significant": p_value < 0.05, "df": round(df, 1)}


def srm_chi_square(observed: list[int], expected: list[int]) -> dict:
    if len(observed) != len(expected):
        return {"chi2": 0.0, "srm_detected": True, "flag": "invalid_sizes"}
    chi2 = 0.0
    for o, e in zip(observed, expected):
        if e > 0:
            chi2 += (o - e) ** 2 / e
        else:
            chi2 = float("inf")
    n = len(observed)
    df = n - 1
    from math import erfc
    p_value = 1.0
    if chi2 != float("inf") and df > 0:
        p_value = min(1.0, erfc(math.sqrt(chi2 / (2 * df))))
    srm_detected = p_value < 0.05
    return {
        "chi2": round(chi2, 4),
        "df": df,
        "p_value": round(p_value, 4),
        "srm_detected": srm_detected,
        "flag": "SRM_WARNING" if srm_detected else "ok",
    }

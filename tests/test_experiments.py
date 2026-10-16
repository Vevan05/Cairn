from experiments import bucket_hash, assign_variant, two_proportion_z, welch_t, srm_chi_square


def test_bucket_stable():
    assert bucket_hash("e", "u") == bucket_hash("e", "u")


def test_same_user_same_variant():
    v1 = assign_variant("e", "u", [{"name": "c", "weight": 50}, {"name": "t", "weight": 50}])
    v2 = assign_variant("e", "u", [{"name": "c", "weight": 50}, {"name": "t", "weight": 50}])
    assert v1 == v2


def test_split_weights():
    counts = {"c": 0, "t": 0}
    for i in range(10000):
        v = assign_variant("e", f"u{i}", [{"name": "c", "weight": 50}, {"name": "t", "weight": 50}])
        counts[v] += 1
    assert abs(counts["c"] - 5000) < 300
    assert abs(counts["t"] - 5000) < 300


def test_z_test_same_rate_not_significant():
    r = two_proportion_z(0.3, 100, 0.3, 100)
    assert r["significant"] is False


def test_srm_good_split_not_flagged():
    r = srm_chi_square([4980, 5020], [5000, 5000])
    assert not r["srm_detected"]


def test_srm_bad_split_flagged():
    r = srm_chi_square([4500, 5500], [5000, 5000])
    assert r["srm_detected"] is True


def test_welch_t_same_means_not_significant():
    r = welch_t(10.0, 1.0, 100, 10.0, 1.0, 100)
    assert r["significant"] is False

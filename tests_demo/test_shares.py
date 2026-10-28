import random

from demo.shares import split_counts


def test_proportional_split():
    assert split_counts(10, {"a": .5, "b": .3, "c": .2}, ["a", "b", "c"]) == {"a": 5, "b": 3, "c": 2}


def test_sums_to_B_for_random_cases():
    rng = random.Random(0)
    for _ in range(200):
        members = ["a", "b", "c"][: rng.randint(1, 3)]
        shares = {m: rng.random() for m in members}
        B = rng.randint(0, 60)
        counts = split_counts(B, shares, members)
        assert sum(counts.values()) == B
        assert set(counts) == set(members)
        assert all(v >= 0 for v in counts.values())


def test_zero_share_gets_nothing():
    assert split_counts(9, {"a": 1.0, "b": 0.0}, ["a", "b"]) == {"a": 9, "b": 0}


def test_ties_break_by_member_order():
    assert split_counts(1, {"a": .5, "b": .5}, ["b", "a"]) == {"b": 1, "a": 0}


def test_missing_or_all_zero_shares_split_evenly():
    assert split_counts(6, {}, ["a", "b", "c"]) == {"a": 2, "b": 2, "c": 2}

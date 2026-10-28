import numpy as np

from experiments.external_audits.daccn.reference import zero_flags

TARGETS = range(9, 31)  # 22 steps, list indices 9..30 of the 37 ten-day composites


def test_zero_flags_all_valid():
    valid = np.ones(37, dtype=bool)
    is_zero, is_valid = zero_flags(valid, TARGETS, 9)
    assert is_zero.shape == (22,) and not is_zero.any()
    assert is_valid.all()


def test_zero_flags_isolated_gap():
    valid = np.ones(37, dtype=bool)
    valid[15] = False
    is_zero, is_valid = zero_flags(valid, TARGETS, 9)
    assert not is_valid[15 - 9]
    assert not is_zero[15 - 9]  # interpolated from 14 and 16
    assert not is_zero.any()


def test_zero_flags_edge_gap_variant_b_like():
    # No winter imagery: composites 0..8 and 31..36 empty; target 9 (step 0) empty too.
    valid = np.zeros(37, dtype=bool)
    valid[10:31] = True
    is_zero, is_valid = zero_flags(valid, TARGETS, 9)
    assert is_zero[0]  # target 9: invalid, nothing valid among 0..8 -> zero
    assert not is_zero[1:].any()  # targets 10..30 are valid themselves


def test_zero_flags_long_gap():
    # Composites 12..30 inclusive empty, everything else valid.
    # Target i (12 <= i <= 30) is invalid. Before-search covers i-9..i-1, after-search i+1..i+9.
    # - before valid iff i-9 <= 11  <=> i <= 20
    # - after  valid iff i+9 >= 31  <=> i >= 22
    # zero iff before invalid OR after invalid  <=> i >= 21 OR i <= 21  -> every i in 12..30 is zero
    #   (i = 21 has before 12..20 all empty and after 22..30 all empty).
    valid = np.ones(37, dtype=bool)
    valid[12:31] = False
    is_zero, is_valid = zero_flags(valid, TARGETS, 9)
    expected = np.zeros(22, dtype=bool)
    expected[12 - 9 : 31 - 9] = True
    assert is_zero.tolist() == expected.tolist()
    assert (~is_valid[12 - 9 : 31 - 9]).all() and is_valid[: 12 - 9].all()


def test_zero_flags_gap_interpolated_when_both_sides_within_search():
    # Composites 12..20 empty (9 wide). Target 12 has before 3..11 (valid) and after 13..21: 21 valid -> interpolated.
    # Target 20 has before 11..19: 11 valid; after 21..29 valid -> interpolated. No zeros.
    valid = np.ones(37, dtype=bool)
    valid[12:21] = False
    is_zero, _ = zero_flags(valid, TARGETS, 9)
    assert not is_zero.any()

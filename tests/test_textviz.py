import pytest

from simengine.textviz import BLOCKS, resample, sparkline


def test_lowest_and_highest_use_extreme_blocks() -> None:
    line = sparkline([1.0, 5.0, 3.0])
    assert line[0] == BLOCKS[0]
    assert line[1] == BLOCKS[-1]


def test_increasing_series_never_decreases() -> None:
    line = sparkline(list(range(20)))
    levels = [BLOCKS.index(ch) for ch in line]
    assert levels == sorted(levels)
    assert levels[0] == 0 and levels[-1] == len(BLOCKS) - 1


def test_flat_series_is_constant() -> None:
    line = sparkline([7.0, 7.0, 7.0, 7.0])
    assert len(set(line)) == 1


def test_length_is_capped_by_width() -> None:
    assert len(sparkline(list(range(500)), width=60)) == 60
    assert len(sparkline([1.0, 2.0, 3.0], width=60)) == 3


def test_resample_preserves_mean_for_equal_buckets() -> None:
    out = resample([1.0, 3.0, 5.0, 7.0], 2)
    assert list(out) == [2.0, 6.0]


def test_bad_input_rejected() -> None:
    with pytest.raises(ValueError):
        sparkline([])
    with pytest.raises(ValueError):
        sparkline([1.0, 2.0], width=0)
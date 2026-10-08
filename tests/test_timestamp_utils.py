"""Tests for timestamp matching used by the MCAP converter."""

import pytest

from galbot_mcap2lerobot.timestamp_utils import TimestampIndex


@pytest.mark.parametrize(
    ("timestamps", "targets"),
    [
        ([10, 20, 30], [-1, 10, 15, 29, 30, 99]),
        ([30, 10, 20], [5, 15, 25, 35]),
        ([10, 10, 20, 20], [0, 10, 15, 20, 21]),
        ([9, 1, 7, 3], [0, 2, 5, 8, 10]),
    ],
)
def test_nearest_timestamp_matches_linear_search(timestamps: list[int], targets: list[int]) -> None:
    index = TimestampIndex(timestamps)

    for target in targets:
        expected_frame_index, expected_timestamp = min(
            enumerate(timestamps), key=lambda item: abs(item[1] - target)
        )
        assert index.nearest(target) == (expected_timestamp, expected_frame_index)


def test_timestamp_index_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty timestamp sequence"):
        TimestampIndex([])

"""Utilities for matching timestamped sensor streams."""

from bisect import bisect_left
from collections.abc import Sequence


class TimestampIndex:
    """Find the closest source-frame timestamp without scanning the full stream."""

    def __init__(self, timestamps: Sequence[int]) -> None:
        """Index timestamps while retaining each frame's original position."""
        if len(timestamps) == 0:
            raise ValueError("Cannot index an empty timestamp sequence.")

        sorted_frames = sorted(enumerate(timestamps), key=lambda item: item[1])
        self.timestamps = [timestamp for _, timestamp in sorted_frames]
        self.frame_indices = [frame_index for frame_index, _ in sorted_frames]

    def nearest(self, target_timestamp: int) -> tuple[int, int]:
        """Return ``(timestamp, original_frame_index)`` for the closest frame.

        Ties retain the behavior of ``min(..., key=abs(timestamp - target))``:
        the earliest frame in the input sequence wins.
        """
        right = bisect_left(self.timestamps, target_timestamp)
        if right == 0:
            return self.timestamps[0], self.frame_indices[0]
        if right == len(self.timestamps):
            left_timestamp = self.timestamps[-1]
            left = bisect_left(self.timestamps, left_timestamp)
            return left_timestamp, self.frame_indices[left]

        left_timestamp = self.timestamps[right - 1]
        left = bisect_left(self.timestamps, left_timestamp)
        right_timestamp = self.timestamps[right]

        left_distance = target_timestamp - left_timestamp
        right_distance = right_timestamp - target_timestamp
        if left_distance < right_distance:
            return left_timestamp, self.frame_indices[left]
        if right_distance < left_distance:
            return right_timestamp, self.frame_indices[right]

        if self.frame_indices[left] <= self.frame_indices[right]:
            return left_timestamp, self.frame_indices[left]
        return right_timestamp, self.frame_indices[right]

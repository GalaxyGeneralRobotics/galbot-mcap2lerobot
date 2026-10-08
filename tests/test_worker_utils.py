"""Tests for parent-side worker supervision."""

from queue import Empty

import pytest

from galbot_mcap2lerobot.worker_utils import raise_for_worker_failures, wait_for_workers


class FakeQueue:
    """Queue stub with a fixed sequence of progress events."""

    def __init__(self, events=()):
        """Store events to return to the worker monitor."""
        self.events = list(events)

    def get(self, timeout):
        """Return the next event or raise when the stub is empty."""
        if not self.events:
            raise Empty
        return self.events.pop(0)


class FakeProcess:
    """Minimal process stub exposing its name and exit status."""

    def __init__(self, name="Worker-0", exitcode=0):
        """Initialize the stub process state."""
        self.name = name
        self.exitcode = exitcode


class FakeProgressBar:
    """Progress bar stub recording updates made by the monitor."""

    def __init__(self):
        """Initialize the recorded progress state."""
        self.descriptions = []
        self.updates = []

    def set_description(self, description):
        """Record the current worker label."""
        self.descriptions.append(description)

    def refresh(self):
        """Match the tqdm interface without rendering output."""
        pass

    def reset(self, total):
        """Record the total work for the current phase."""
        self.total = total

    def update(self, value):
        """Record a progress increment."""
        self.updates.append(value)


def test_worker_error_event_is_returned_and_worker_is_marked_done():
    progress_queue = FakeQueue(
        [
            ("worker_error", 0, "broken.mcap: invalid image"),
            ("worker_done", 0, None),
        ]
    )
    bar = FakeProgressBar()

    errors = wait_for_workers(progress_queue, [FakeProcess(exitcode=1)], [bar])

    assert errors == [(0, "broken.mcap: invalid image")]
    assert bar.descriptions == ["Worker 1 failed"]


def test_crashed_worker_without_completion_event_does_not_hang():
    process = FakeProcess(exitcode=1)
    bar = FakeProgressBar()
    errors = wait_for_workers(FakeQueue(), [process], [bar])

    assert errors == [(0, "Worker-0 exited with status 1 without a completion event")]
    assert bar.descriptions == ["Worker 1 failed"]
    with pytest.raises(RuntimeError, match="without a completion event"):
        raise_for_worker_failures(errors, [process])


def test_successful_exit_without_completion_event_does_not_hang():
    process = FakeProcess(exitcode=0)
    errors = wait_for_workers(FakeQueue(), [process], [FakeProgressBar()])

    assert errors == [(0, "Worker-0 exited with status 0 without a completion event")]
    with pytest.raises(RuntimeError, match="without a completion event"):
        raise_for_worker_failures(errors, [process])


def test_successful_progress_events_are_applied_until_worker_done():
    progress_queue = FakeQueue(
        [
            ("set_file", 0, "episode.mcap"),
            ("set_phase", 0, ("aligning", 3)),
            ("advance", 0, 2),
            ("worker_done", 0, None),
        ]
    )
    bar = FakeProgressBar()

    errors = wait_for_workers(progress_queue, [FakeProcess()], [bar])

    assert errors == []
    assert bar.descriptions == ["Worker 1 episode.mcap", "Worker 1 aligning"]
    assert bar.total == 3
    assert bar.updates == [2]


def test_worker_failure_prevents_merge():
    process = FakeProcess(exitcode=1)

    with pytest.raises(RuntimeError, match="refusing to merge partial output") as error:
        raise_for_worker_failures([(0, "broken.mcap: invalid image")], [process])
    assert "broken.mcap: invalid image" in str(error.value)


def test_process_exit_failure_is_reported_when_worker_event_is_missing():
    process = FakeProcess(exitcode=1)

    with pytest.raises(RuntimeError, match="Worker-0 exited with status 1"):
        raise_for_worker_failures([], [process])


def test_successful_workers_allow_merge():
    raise_for_worker_failures([], [FakeProcess(exitcode=0)])

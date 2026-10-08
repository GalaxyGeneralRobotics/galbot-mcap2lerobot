"""Utilities for supervising parallel MCAP conversion workers."""

from __future__ import annotations

from queue import Empty
from typing import Any


def wait_for_workers(
    progress_queue: Any,
    processes: list[Any],
    progress_bars: list[Any],
) -> list[tuple[int, str]]:
    """Consume worker progress until all workers finish or exit unexpectedly.

    Returns worker errors reported by ``process_subset`` and unexpected process exits as
    ``(worker_index, message)`` pairs. Any process that exits without a completion event is
    treated as finished here so the parent can join it and report the failure instead of
    waiting forever for a queue message.
    """
    completed_workers: set[int] = set()
    worker_errors: list[tuple[int, str]] = []

    while len(completed_workers) < len(processes):
        try:
            event, worker_idx, value = progress_queue.get(timeout=0.2)
        except Empty:
            for worker_idx, process in enumerate(processes):
                if worker_idx not in completed_workers and process.exitcode is not None:
                    completed_workers.add(worker_idx)
                    worker_errors.append(
                        (
                            worker_idx,
                            f"{process.name} exited with status {process.exitcode} "
                            "without a completion event",
                        )
                    )
                    bar = progress_bars[worker_idx]
                    bar.set_description(f"Worker {worker_idx + 1} failed")
                    bar.refresh()
            continue

        bar = progress_bars[worker_idx]
        if event == "set_file":
            bar.set_description(f"Worker {worker_idx + 1} {value}")
            bar.refresh()
        elif event == "set_phase":
            phase_name, phase_total = value
            bar.reset(total=int(phase_total))
            bar.set_description(f"Worker {worker_idx + 1} {phase_name}")
            bar.refresh()
        elif event == "advance":
            bar.update(int(value))
        elif event == "worker_error":
            worker_errors.append((worker_idx, str(value)))
            bar.set_description(f"Worker {worker_idx + 1} failed")
            bar.refresh()
        elif event == "worker_done":
            completed_workers.add(worker_idx)

    return worker_errors


def raise_for_worker_failures(
    worker_errors: list[tuple[int, str]],
    processes: list[Any],
) -> None:
    """Raise before merge when a worker reported an error or exited unexpectedly."""
    reported_worker_indices = {worker_idx for worker_idx, _ in worker_errors}
    failures = list(worker_errors)
    for worker_idx, process in enumerate(processes):
        if process.exitcode not in (None, 0) and worker_idx not in reported_worker_indices:
            failures.append((worker_idx, f"{process.name} exited with status {process.exitcode}"))

    if failures:
        details = "\n".join(f"- Worker {idx + 1}: {message}" for idx, message in failures)
        raise RuntimeError(
            f"Worker conversion failed; refusing to merge partial output:\n{details}"
        )

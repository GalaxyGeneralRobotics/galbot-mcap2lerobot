"""Regression tests for the multi-file merge metadata bug.

Background
----------
``merge_datasets`` renumbers ``episode_index`` globally across source folders.
Before this was hardened, ``episodes_stats.jsonl`` and ``tasks.jsonl`` /
``task_index`` were *not* renumbered alongside it. Two source folders that both
started at episode 0 therefore collided under the same merged index, and:

* the loader silently kept only one episode's stats, skewing normalization;
* frames were relabelled with another folder's task text.

Nothing in the pipeline raised. These tests pin that behaviour down so the
failure mode cannot return silently.

The fixtures are synthetic LeRobot v2.1 folders: real ``meta/`` JSON, a real
one-row parquet per episode, and zero-byte ``.mp4`` placeholders (the merge
path only copies video files, it never decodes them). No MCAP, no robot and no
``lerobot`` import is required.
"""

import json
import os

import pandas as pd
import pytest

from galbot_mcap2lerobot.merge_lerobot_v2_1 import merge_datasets

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
VIDEO_KEY = "observation.images.front_head_camera_left_color"

JOINT_NAMES = [f"joint_{i}" for i in range(6)]


# ============================================================
# Fixture builders
# ============================================================


def _info() -> dict:
    """Minimal but schema-valid LeRobot v2.1 ``info.json``."""
    return {
        "codebase_version": "v2.1",
        "robot_type": "galbot_g1",
        "fps": 30,
        "chunks_size": 1000,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "splits": {},
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [len(JOINT_NAMES)],
                "names": JOINT_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": [len(JOINT_NAMES)],
                "names": JOINT_NAMES,
            },
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
            },
        },
    }


def _write_jsonl(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_source(
    root,
    episodes: list,
    tasks: list,
    *,
    frames_per_episode: int = 3,
    skip_parquet_for: tuple = (),
    skip_video_for: tuple = (),
) -> str:
    """Build one synthetic LeRobot v2.1 source folder.

    Args:
        root: Directory to create the dataset in.
        episodes: ``[(episode_index, task_index), ...]``.
        tasks: ``[(task_index, task_text), ...]`` for this folder's tasks.jsonl.
        frames_per_episode: Rows written to each episode's parquet.
        skip_parquet_for: episode indices to deliberately leave without a
            parquet file (used to exercise the abort paths).
        skip_video_for: episode indices to deliberately leave without a video.

    Returns:
        The folder path, as a string.
    """
    root = str(root)
    os.makedirs(os.path.join(root, "meta"), exist_ok=True)

    with open(os.path.join(root, "meta", "info.json"), "w") as f:
        json.dump(_info(), f)

    _write_jsonl(
        os.path.join(root, "meta", "episodes.jsonl"),
        [
            {
                "episode_index": ep_index,
                "tasks": [text for t_index, text in tasks if t_index == task_index],
                "length": frames_per_episode,
            }
            for ep_index, task_index in episodes
        ],
    )

    # Per-episode stats carry a fingerprint value so a test can tell which
    # source episode a merged stats entry actually came from.
    _write_jsonl(
        os.path.join(root, "meta", "episodes_stats.jsonl"),
        [
            {
                "episode_index": ep_index,
                "stats": {
                    "observation.state": {
                        "min": [float(fingerprint(root, ep_index))] * len(JOINT_NAMES),
                        "max": [float(fingerprint(root, ep_index))] * len(JOINT_NAMES),
                        "mean": [float(fingerprint(root, ep_index))] * len(JOINT_NAMES),
                        "std": [0.0] * len(JOINT_NAMES),
                        "count": [frames_per_episode],
                    }
                },
            }
            for ep_index, _ in episodes
        ],
    )

    _write_jsonl(
        os.path.join(root, "meta", "tasks.jsonl"),
        [{"task_index": t_index, "task": text} for t_index, text in tasks],
    )

    for ep_index, task_index in episodes:
        chunk = ep_index // _info()["chunks_size"]

        if ep_index not in skip_parquet_for:
            parquet_path = os.path.join(
                root, DATA_PATH.format(episode_chunk=chunk, episode_index=ep_index)
            )
            os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
            pd.DataFrame(
                {
                    "episode_index": [ep_index] * frames_per_episode,
                    "index": list(range(frames_per_episode)),
                    "task_index": [task_index] * frames_per_episode,
                    "frame_index": list(range(frames_per_episode)),
                }
            ).to_parquet(parquet_path, index=False)

        if ep_index not in skip_video_for:
            video_path = os.path.join(
                root,
                VIDEO_PATH.format(episode_chunk=chunk, video_key=VIDEO_KEY, episode_index=ep_index),
            )
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            # The merge path copies videos, it never decodes them.
            open(video_path, "wb").close()

    return root


def fingerprint(root: str, episode_index: int) -> int:
    """A per-(folder, episode) marker embedded in stats so it can be traced."""
    return (hash(os.path.basename(root)) % 1000) * 10 + episode_index


def read_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ============================================================
# Tests
# ============================================================


def test_merge_remaps_episode_stats_instead_of_colliding(tmp_path):
    """Two folders both starting at episode 0 must keep separate stats.

    Pre-fix, ``episodes_stats.jsonl`` entries were concatenated with their
    original folder-local ``episode_index``, so both folders' episode 0
    collided and one silently overwrote the other.
    """
    a = make_source(tmp_path / "a", episodes=[(0, 0)], tasks=[(0, "pick up the red block")])
    b = make_source(tmp_path / "b", episodes=[(0, 0)], tasks=[(0, "open the drawer")])
    out = str(tmp_path / "merged")

    merge_datasets([a, b], out)

    stats = read_jsonl(os.path.join(out, "meta", "episodes_stats.jsonl"))

    assert len(stats) == 2, "one stats entry per merged episode"
    assert sorted(s["episode_index"] for s in stats) == [0, 1], (
        "stats must follow the globally renumbered episode_index"
    )

    by_index = {s["episode_index"]: s for s in stats}
    assert by_index[0]["stats"]["observation.state"]["min"][0] == float(fingerprint(a, 0))
    assert by_index[1]["stats"]["observation.state"]["min"][0] == float(fingerprint(b, 0)), (
        "episode 1's stats must come from folder b, not a duplicate of folder a"
    )


def test_merge_remaps_task_index_to_correct_text(tmp_path):
    """Distinct tasks that share a folder-local index must stay distinct.

    Both folders call their task ``task_index == 0`` but mean different things.
    Pre-fix the merged parquet kept the raw index, so folder b's frames were
    relabelled with folder a's task text.
    """
    a = make_source(tmp_path / "a", episodes=[(0, 0)], tasks=[(0, "pick up the red block")])
    b = make_source(tmp_path / "b", episodes=[(0, 0)], tasks=[(0, "open the drawer")])
    out = str(tmp_path / "merged")

    merge_datasets([a, b], out)

    tasks = read_jsonl(os.path.join(out, "meta", "tasks.jsonl"))
    text_by_index = {t["task_index"]: t["task"] for t in tasks}

    assert len(tasks) == 2, "two different task texts must not be deduped into one"
    assert set(text_by_index.values()) == {"pick up the red block", "open the drawer"}

    ep0 = pd.read_parquet(os.path.join(out, DATA_PATH.format(episode_chunk=0, episode_index=0)))
    ep1 = pd.read_parquet(os.path.join(out, DATA_PATH.format(episode_chunk=0, episode_index=1)))

    assert text_by_index[ep0["task_index"].iloc[0]] == "pick up the red block"
    assert text_by_index[ep1["task_index"].iloc[0]] == "open the drawer", (
        "folder b's frames must keep folder b's task text"
    )


def test_merge_dedupes_identical_task_text(tmp_path):
    """The same task text under different folder-local indices collapses to one."""
    a = make_source(tmp_path / "a", episodes=[(0, 0)], tasks=[(0, "pick up the red block")])
    b = make_source(tmp_path / "b", episodes=[(0, 7)], tasks=[(7, "pick up the red block")])
    out = str(tmp_path / "merged")

    merge_datasets([a, b], out)

    tasks = read_jsonl(os.path.join(out, "meta", "tasks.jsonl"))
    assert len(tasks) == 1, "identical task text should share a single merged index"

    merged_index = tasks[0]["task_index"]
    for episode_index in (0, 1):
        df = pd.read_parquet(
            os.path.join(out, DATA_PATH.format(episode_chunk=0, episode_index=episode_index))
        )
        assert (df["task_index"] == merged_index).all()


def test_merge_reindexes_frames_contiguously(tmp_path):
    """The global ``index`` column must be contiguous across merged episodes."""
    a = make_source(tmp_path / "a", episodes=[(0, 0)], tasks=[(0, "task a")], frames_per_episode=3)
    b = make_source(tmp_path / "b", episodes=[(0, 0)], tasks=[(0, "task b")], frames_per_episode=4)
    out = str(tmp_path / "merged")

    merge_datasets([a, b], out)

    ep0 = pd.read_parquet(os.path.join(out, DATA_PATH.format(episode_chunk=0, episode_index=0)))
    ep1 = pd.read_parquet(os.path.join(out, DATA_PATH.format(episode_chunk=0, episode_index=1)))

    assert ep0["index"].tolist() == [0, 1, 2]
    assert ep1["index"].tolist() == [3, 4, 5, 6], "frame indices must continue, not restart"
    assert (ep1["episode_index"] == 1).all()


def test_merge_aborts_on_missing_parquet(tmp_path):
    """A missing parquet must abort, not leave a ghost episode in metadata."""
    a = make_source(
        tmp_path / "a",
        episodes=[(0, 0), (1, 0)],
        tasks=[(0, "task a")],
        skip_parquet_for=(1,),
    )
    out = str(tmp_path / "merged")

    with pytest.raises(RuntimeError, match="Parquet file not found"):
        merge_datasets([a], out)


def test_episode_zero_fallback_is_not_applied_to_other_episodes(tmp_path):
    """The ``episode_000000`` fallback must stay scoped to ``old_index == 0``.

    Per-mcap worker folders always name their single episode 0, so the merge
    falls back to ``data/chunk-000/episode_000000.parquet``. Applying that
    fallback to any other index would silently substitute episode 0's frames
    for a different, missing episode. Episode 0 exists here, so an unscoped
    fallback would find it and succeed instead of raising.
    """
    a = make_source(
        tmp_path / "a",
        episodes=[(0, 0), (2, 0)],
        tasks=[(0, "task a")],
        skip_parquet_for=(2,),
    )
    out = str(tmp_path / "merged")

    with pytest.raises(RuntimeError, match="Parquet file not found"):
        merge_datasets([a], out)


def test_merge_aborts_on_missing_video(tmp_path):
    """A missing video must abort rather than claim a video that isn't there."""
    a = make_source(
        tmp_path / "a",
        episodes=[(0, 0), (1, 0)],
        tasks=[(0, "task a")],
        skip_video_for=(1,),
    )
    out = str(tmp_path / "merged")

    with pytest.raises(RuntimeError, match="Video not found"):
        merge_datasets([a], out)


def test_merge_aborts_on_unresolvable_task_index(tmp_path):
    """A ``task_index`` absent from tasks.jsonl must abort, not become NaN."""
    a = make_source(
        tmp_path / "a",
        episodes=[(0, 5)],  # parquet references task 5...
        tasks=[(0, "task a")],  # ...but tasks.jsonl only declares task 0
    )
    out = str(tmp_path / "merged")

    with pytest.raises(RuntimeError, match=r"has no entry in tasks\.jsonl"):
        merge_datasets([a], out)


def test_merge_updates_info_totals(tmp_path):
    """``info.json`` totals must reflect the merged dataset, not folder 0."""
    a = make_source(tmp_path / "a", episodes=[(0, 0)], tasks=[(0, "task a")], frames_per_episode=3)
    b = make_source(tmp_path / "b", episodes=[(0, 0)], tasks=[(0, "task b")], frames_per_episode=4)
    out = str(tmp_path / "merged")

    merge_datasets([a, b], out)

    with open(os.path.join(out, "meta", "info.json")) as f:
        info = json.load(f)

    assert info["total_episodes"] == 2
    assert info["total_frames"] == 7
    assert info["total_tasks"] == 2
    assert info["splits"] == {"train": "0:2"}

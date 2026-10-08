"""Convert MCAP files to LeRobot dataset format.

This module provides functionality to convert MCAP (Message Capture) files,
which contain ROS message data, into LeRobot dataset format. It handles:
- Parsing MCAP files and extracting image, camera info, and robot state data
- Converting protobuf messages to Python dictionaries
- Processing and aligning multi-camera image streams
- Creating LeRobot dataset with proper feature definitions
- Multi-process batch processing of MCAP files

The module supports Galbot humanoid robot data with multiple camera streams
and joint sensor data.

Usage:
    python -m galbot_mcap2lerobot.galbot_mcap_to_lerobot

    Place .mcap files in MCAP_DIR (default: ./mcap) and run. Output goes to
    OUTPUT_DIR (default: ./output). Configure topics and export settings in
    galbot_mcap2lerobot/config.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
from pathlib import Path
import shutil
from typing import Any

import cv2
from google.protobuf import descriptor_pb2, message_factory
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import make_reader
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

# Import local config and merge module
from . import config
from .joint_constants import (
    GRIPPER_GROUPS,
    GRIPPER_VALUE_SCALE,
    JOINT_GROUP_ORDER,
    JOINT_NAMES,
    JOINT_STRUCTURE,
)
from .merge_lerobot_v2_1 import merge_datasets
from .worker_utils import raise_for_worker_failures, wait_for_workers

# ----------------------------
# Logging Configuration
# ----------------------------
logger = logging.getLogger("MCAP2LeRobot")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Global cache for protobuf message classes to avoid repeated parsing
_PROTOBUF_CACHE: dict[str, Any] = {}


# ============================================================
# Protobuf Utilities
# ============================================================


def protobuf_to_dict(msg: Any) -> dict[str, Any] | Any:
    """Recursively convert protobuf Message to Python dict.

    Handles nested messages, repeated fields, and map types. Compatible with
    the new upb runtime in protobuf >= 4.x.

    Args:
        msg: The protobuf message object to convert.

    Returns:
        A Python dictionary, or the original value if not a protobuf message.
    """
    from google.protobuf.descriptor import FieldDescriptor

    if not hasattr(msg, "ListFields"):
        return msg

    result: dict[str, Any] = {}
    for field, value in msg.ListFields():
        if (
            field.type == FieldDescriptor.TYPE_MESSAGE
            and field.message_type.has_options
            and field.message_type.GetOptions().map_entry
        ):
            result[field.name] = {k: protobuf_to_dict(v) for k, v in value.items()}
        elif field.type == FieldDescriptor.TYPE_MESSAGE:
            if field.is_repeated:
                result[field.name] = [protobuf_to_dict(v) for v in value]
            else:
                result[field.name] = protobuf_to_dict(value)
        else:
            if getattr(value, "__class__", None).__name__ == "RepeatedScalarContainer":
                result[field.name] = list(value)
            else:
                result[field.name] = value
    return result


def decode_protobuf(schema: Any, data: bytes) -> dict[str, Any] | None:
    """Dynamically decode Protobuf data using MCAP-embedded schema descriptors.

    Uses the schema information from MCAP to dynamically create protobuf
    message classes and decode binary data. Caches message classes for reuse.

    Args:
        schema: The MCAP schema object containing protobuf descriptor information.
        data: Binary data containing the serialized protobuf message.

    Returns:
        A Python dictionary of the decoded message, or None if decoding fails.
    """
    if not data or not schema or schema.encoding != "protobuf":
        return None

    if schema.name in _PROTOBUF_CACHE:
        msg_class = _PROTOBUF_CACHE[schema.name]
    else:
        fd_set = descriptor_pb2.FileDescriptorSet()
        fd_set.ParseFromString(schema.data)
        pool = message_factory.GetMessages(fd_set.file)
        msg_class = pool.get(schema.name) or next(iter(pool.values()))
        _PROTOBUF_CACHE[schema.name] = msg_class

    msg = msg_class()
    msg.ParseFromString(data)
    return protobuf_to_dict(msg)


# ============================================================
# McapParser — extract data from .mcap files
# ============================================================


class McapParser:
    """Parser for MCAP files to extract ROS message data.

    Reads an MCAP file and extracts image bytes, camera calibration info, and
    robot state messages. Each message is stored with its log_time timestamp.

    Attributes:
        image_topics: ROS topic names for camera image streams.
        info_topics: ROS topic names for camera info messages.
        state_topics: ROS topic names for robot state messages.
        save_parsed_info: Whether to dump decoded messages to JSON for debugging.
    """

    def __init__(
        self,
        image_topics: list[str],
        info_topics: list[str],
        state_topics: list[str],
        save_parsed_info: bool = False,
        enable_undistort: bool = False,
    ) -> None:
        """Initialize parser topic configuration and optional debug behavior."""
        self.image_topics = image_topics
        self.info_topics = info_topics
        self.state_topics = state_topics
        self.save_parsed_info = save_parsed_info
        self.enable_undistort = enable_undistort

    def parse(
        self, file_path: str | Path, output_dir: str = "./parsed_json"
    ) -> tuple[
        dict[str, list[tuple[int, dict[str, Any]]]],
        dict[str, list[tuple[int, dict[str, Any]]]],
        dict[str, dict[str, Any]],
    ]:
        """Parse MCAP file and extract messages from configured topics.

        Args:
            file_path: Path to the MCAP file.
            output_dir: Directory for debug JSON output.

        Returns:
            (image_buffers, state_buffers, camera_info_buffers)
            - image_buffers:  {topic: [(timestamp_ns, decoded_msg), ...]}
            - state_buffers:  {topic: [(timestamp_ns, decoded_msg), ...]}
            - camera_info_buffers: {topic: decoded_camera_info_dict}
        """
        if self.save_parsed_info:
            os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Reading MCAP: {file_path}")

        image_buffers: dict[str, list[tuple[int, dict[str, Any]]]] = {
            t: [] for t in self.image_topics
        }
        state_buffers: dict[str, list[tuple[int, dict[str, Any]]]] = {
            t: [] for t in self.state_topics
        }
        camera_info_buffers: dict[str, dict[str, Any]] = {t: {} for t in self.info_topics}

        with open(file_path, "rb") as f:
            reader = make_reader(f)
            topic_cnt_dict: dict[str, int] = {}
            for schema, channel, message in reader.iter_messages():
                topic = channel.topic
                ts = message.log_time
                decoded = decode_protobuf(schema, message.data)

                if topic not in topic_cnt_dict:
                    topic_cnt_dict[topic] = 0
                else:
                    topic_cnt_dict[topic] += 1

                if topic in self.info_topics and decoded:
                    camera_info_buffers[topic] = decoded

                if topic in self.image_topics:
                    image_buffers[topic].append((ts, decoded))
                    continue

                if topic in self.state_topics:
                    state_buffers[topic].append((ts, decoded))

                # Optionally save decoded messages to JSON for debugging
                if (
                    self.save_parsed_info
                    and decoded is not None
                    and topic in (self.info_topics + self.state_topics + self.image_topics)
                ):
                    safe_topic = topic.replace("/", "_").strip("_")
                    if topic in self.info_topics:
                        topic_dir = os.path.join(output_dir, "camera_parameters")
                        filename = f"{safe_topic}.json"
                    else:
                        topic_dir = os.path.join(output_dir, safe_topic)
                        filename = f"{ts}.json"
                    os.makedirs(topic_dir, exist_ok=True)
                    filepath = os.path.join(topic_dir, filename)
                    if not os.path.exists(filepath):
                        try:
                            with open(filepath, "w", encoding="utf-8") as fp:
                                json.dump(decoded, fp, ensure_ascii=False, indent=2)
                        except Exception as e:
                            logger.warning(f"Failed to write JSON for {topic}: {e}")

        return image_buffers, state_buffers, camera_info_buffers


# ============================================================
# McapToLeRobotDataset — build LeRobot dataset from parsed data
# ============================================================


class McapToLeRobotDataset:
    """Converter from parsed MCAP data to LeRobot dataset format.

    Handles dataset creation, feature definitions, multi-camera frame alignment,
    joint state interpolation, and episode generation.

    Attributes:
        repo_id: HuggingFace repository ID or local dataset name.
        output_root: Root directory for the output dataset.
        fps: Frame rate for video encoding.
        robot_type: Type identifier for the robot (e.g., "G1_V2.2B").
    """

    def __init__(
        self,
        repo_id: str,
        output_root: str,
        fps: int = 30,
        robot_type: str = "G1_V2.2B",
        use_videos: bool = True,
        video_codec: str = "av1",
        video_pix_fmt: str = "yuv420p",
        primary_camera_topic: str = "/front_head_camera/left_color/image_raw",
        use_velocity: bool = False,
        use_effort: bool = False,
        enable_undistort: bool = False,
    ) -> None:
        """Initialize dataset conversion options and output targets."""
        self.repo_id = repo_id
        self.output_root = Path(output_root)
        self.fps = fps
        self.robot_type = robot_type
        self.use_videos = use_videos
        self.video_codec = video_codec
        self.video_pix_fmt = video_pix_fmt
        self.primary_camera_topic = primary_camera_topic
        self.use_velocity = use_velocity
        self.use_effort = use_effort
        self.enable_undistort = enable_undistort
        self.dataset: LeRobotDataset | None = None

    # ------------------------------------------------------------------
    # Joint flattening — CUSTOMIZE THIS FOR YOUR ROBOT
    # ------------------------------------------------------------------
    @staticmethod
    def flatten_scaled_device_robot_data(
        joint_sensor_map: dict[str, dict[str, list[float]]],
        use_velocity: bool = False,
        use_effort: bool = False,
    ) -> np.ndarray:
        """Flatten joint sensor data into a 1-D array.

        Order: leg(5) -> head(2) -> left_arm(7) -> left_gripper(1) ->
               right_arm(7) -> right_gripper(1) = 23 dimensions.

        Gripper position/velocity values are divided by 1000 and stored as
        meter-scale gripper widths.

        Args:
            joint_sensor_map: Nested dict from Galbot's protobuf, e.g.:
                {"leg": {"position": [...], "velocity": [...]}, ...}
            use_velocity: Append velocity values after position.
            use_effort: Append effort/torque values after position.

        Returns:
            np.ndarray of shape (N,) with dtype float32.
        """
        flat_list: list[float] = []
        for part in JOINT_GROUP_ORDER:
            if part not in joint_sensor_map:
                continue
            part_data = joint_sensor_map[part]
            position_values = part_data.get("position", [])
            if part in GRIPPER_GROUPS:
                flat_list.extend(float(v) * GRIPPER_VALUE_SCALE for v in position_values)
            else:
                flat_list.extend(position_values)
            if use_velocity:
                velocity_values = part_data.get("velocity", [])
                if part in GRIPPER_GROUPS:
                    flat_list.extend(float(v) * GRIPPER_VALUE_SCALE for v in velocity_values)
                else:
                    flat_list.extend(velocity_values)
            if use_effort:
                flat_list.extend(part_data.get("effort", []))
        return np.array(flat_list, dtype=np.float32)

    # ------------------------------------------------------------------
    # Feature definitions — CUSTOMIZE dimensions for your robot
    # ------------------------------------------------------------------
    def _build_features(
        self,
        camera_info_buffers: dict[str, dict[str, Any]],
        state_buffers: dict[str, list[tuple[int, dict[str, Any]]]],
    ) -> dict[str, dict[str, Any]]:
        """Build feature schema for the LeRobot dataset.

        Returns:
            Dict mapping feature names to their definitions (dtype, shape, etc.).
        """
        features: dict[str, dict[str, Any]] = {}

        features["observation.state"] = {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        }

        # Camera features — built dynamically from camera_info_buffers
        for info_topic, info in camera_info_buffers.items():
            cam_key = info_topic.replace("/", "_").strip("_").replace("camera_info", "").strip("_")
            cam_name = cam_key or info_topic.split("/")[-2]
            W = info.get("width", 640) if isinstance(info, dict) else 640
            H = info.get("height", 480) if isinstance(info, dict) else 480
            features[f"observation.images.{cam_name}"] = {
                "dtype": "video",
                "shape": (3, H, W),
                "names": ["channels", "height", "width"],
                "info": {
                    "video.height": H,
                    "video.width": W,
                    "video.codec": self.video_codec,
                    "video.pix_fmt": self.video_pix_fmt,
                    "video.is_depth_map": False,
                    "video.fps": self.fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            }
        return features

    def create_dataset(
        self,
        camera_info_buffers: dict[str, dict[str, Any]],
        state_buffers: dict[str, list[tuple[int, dict[str, Any]]]],
        repo_id_override: str | None = None,
    ) -> LeRobotDataset:
        """Create a new LeRobot dataset instance."""
        repo = repo_id_override or self.repo_id
        features = self._build_features(camera_info_buffers, state_buffers)
        self.dataset = LeRobotDataset.create(
            repo_id=repo,
            root=str(self.output_root),
            fps=self.fps,
            robot_type=self.robot_type,
            features=features,
            use_videos=self.use_videos,
        )
        return self.dataset

    # ------------------------------------------------------------------
    # Image decoding
    # ------------------------------------------------------------------
    def _decode_image_bytes(self, data_bytes: bytes, fmt: str, W: int, H: int) -> np.ndarray:
        """Decode image bytes to a (H, W, 3) BGR numpy array.

        Tries JPEG decode first, then raw RGB reshape, then embedded JPEG search.

        Args:
            data_bytes: Raw bytes from the ROS image message.
            fmt: Format hint (e.g., "rgb8", "jpeg").
            W: Expected image width.
            H: Expected image height.

        Returns:
            np.ndarray of shape (H, W, 3), dtype uint8, BGR channel order.

        Raises:
            ValueError: If all decoding strategies fail.
        """
        if not data_bytes:
            raise ValueError("Empty image bytes")

        # Strategy 1: JPEG header detected
        if data_bytes[:2] == b"\xff\xd8":
            arr = np.frombuffer(data_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode failed on jpeg bytes")
            return img

        # Strategy 2: Raw RGB array
        expected = H * W * 3
        if len(data_bytes) == expected:
            return np.frombuffer(data_bytes, dtype=np.uint8).reshape((H, W, 3))

        # Strategy 3: Search for embedded JPEG
        start = data_bytes.find(b"\xff\xd8")
        if start != -1:
            end = data_bytes.find(b"\xff\xd9", start)
            if end != -1:
                jpeg = data_bytes[start : end + 2]
                arr = np.frombuffer(jpeg, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img

        raise ValueError(
            "Image bytes length "
            f"{len(data_bytes)} doesn't match expected {expected} "
            "and no embedded JPEG found"
        )

    # ------------------------------------------------------------------
    # Episode conversion
    # ------------------------------------------------------------------
    def add_mcap_episode(
        self,
        image_buffers: dict[str, list[tuple[int, Any]]],
        state_buffers: dict[str, list[tuple[int, Any]]],
        camera_info_buffers: dict[str, dict[str, Any]],
        task_instruction: str = "",
        progress_queue: multiprocessing.Queue | None = None,
        worker_idx: int | None = None,
        progress_label: str | None = None,
    ) -> None:
        """Convert one MCAP file's data into a LeRobot episode.

        Uses the front head left camera as the primary timing reference.
        Joint states are linearly interpolated to each head camera frame timestamp.
        Other cameras use nearest-neighbor matching.

        Args:
            image_buffers: {topic: [(timestamp_ns, decoded_msg), ...]}
            state_buffers: {topic: [(timestamp_ns, decoded_msg), ...]}
            camera_info_buffers: {topic: camera_info_dict}
            task_instruction: Task description for this episode.
        """
        if self.dataset is None:
            self.create_dataset(camera_info_buffers, state_buffers)

        # --- Primary camera topic (CUSTOMIZE for your robot) ---
        head_topic = self.primary_camera_topic
        if head_topic not in image_buffers:
            raise RuntimeError(f"Primary camera topic '{head_topic}' not found")

        # Build camera name + resolution lookup
        cam_wh = {
            t: (
                t.replace("/", "_").strip("_").replace("image_raw", "").strip("_"),
                camera_info_buffers.get(t.replace("image_raw", "camera_info"), {}).get(
                    "width", 640
                ),
                camera_info_buffers.get(t.replace("image_raw", "camera_info"), {}).get(
                    "height", 480
                ),
            )
            for t in image_buffers
        }

        # --- State interpolation ---
        if not config.STATE_TOPICS:
            raise RuntimeError("STATE_TOPICS is empty in config.py")

        s_topic = config.STATE_TOPICS[0]
        state_list = state_buffers.get(s_topic, [])
        if not state_list:
            raise RuntimeError(f"State topic '{s_topic}' not found or empty")

        s_joint_sm_decoded_list = [
            self.flatten_scaled_device_robot_data(s_decoded["joint_sensor_map"])
            for _, s_decoded in state_list
        ]
        s_timestamps = [ts for ts, _ in state_list]

        interpolate_func = interp1d(
            s_timestamps,
            s_joint_sm_decoded_list,
            kind="linear",
            axis=0,
            bounds_error=False,
            fill_value=(s_joint_sm_decoded_list[0], s_joint_sm_decoded_list[-1]),
        )

        head_frames = image_buffers[head_topic]
        head_timestamps = np.array([h[0] for h in head_frames])
        interpolated_states = interpolate_func(head_timestamps)

        image_times = {
            t: [(ts, idx) for idx, (ts, _) in enumerate(frames)]
            for t, frames in image_buffers.items()
        }

        # --- Build frame list ---
        all_frames: list[dict[str, Any]] = []
        max_sync_gap_ns = config.MAX_CAMERA_SYNC_GAP_FRAMES * (1_000_000_000 / self.fps)

        # Report current file.
        if progress_queue is not None and worker_idx is not None:
            progress_queue.put(("set_file", worker_idx, progress_label or task_instruction))
            progress_queue.put(("set_phase", worker_idx, ("aligning", len(head_frames))))

        align_updates = 0
        for head_idx, (head_ts, head_decoded) in enumerate(head_frames):
            frame_data: dict[str, Any] = {}

            # Decode primary camera image
            data_bytes = (
                head_decoded.get("data")
                or head_decoded.get("data_buffer")
                or head_decoded.get("image")
                or head_decoded.get("jpeg")
                if isinstance(head_decoded, dict)
                else head_decoded
            )
            fmt = (
                head_decoded.get("format", head_decoded.get("encoding", "rgb8"))
                if isinstance(head_decoded, dict)
                else "rgb8"
            )
            cam_name, W, H = cam_wh[head_topic]
            img = self._decode_image_bytes(data_bytes, fmt, W, H)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_data[f"observation.images.{cam_name}"] = np.transpose(img_rgb, (2, 0, 1))

            # Decode and align other cameras (nearest neighbor)
            for topic, frames in image_buffers.items():
                if topic == head_topic:
                    continue
                frames_with_ts = image_times.get(topic, [])
                if not frames_with_ts:
                    continue
                _closest_ts, idx = min(frames_with_ts, key=lambda x: abs(x[0] - head_ts))
                if abs(_closest_ts - head_ts) > max_sync_gap_ns:
                    # Closest available frame for this camera is farther from
                    # the primary camera's timestamp than the allowed sync
                    # gap (e.g. this camera dropped out mid-episode).
                    # Reusing it anyway would silently freeze this camera's
                    # video for the affected frames with no indication in
                    # the output. Abort rather than write out misaligned data.
                    raise RuntimeError(
                        f"[{progress_label or self.repo_id}] Camera '{topic}' frame is "
                        f"{abs(_closest_ts - head_ts) / 1e6:.1f}ms from head frame {head_idx} "
                        f"(limit {max_sync_gap_ns / 1e6:.1f}ms) - refusing to align with a "
                        f"stale frame. Check '{topic}' for dropped frames around this timestamp."
                    )
                _, decoded = frames[idx]
                data_bytes = (
                    decoded.get("data")
                    or decoded.get("data_buffer")
                    or decoded.get("image")
                    or decoded.get("jpeg")
                    if isinstance(decoded, dict)
                    else decoded
                )
                fmt = (
                    decoded.get("format", decoded.get("encoding", "rgb8"))
                    if isinstance(decoded, dict)
                    else "rgb8"
                )
                cam_name, W, H = cam_wh[topic]
                img = self._decode_image_bytes(data_bytes, fmt, W, H)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame_data[f"observation.images.{cam_name}"] = np.transpose(img_rgb, (2, 0, 1))

            # Joint state (interpolated)
            frame_data["observation.state"] = interpolated_states[head_idx].astype(np.float32)
            all_frames.append(frame_data)

            if progress_queue is not None and worker_idx is not None:
                align_updates += 1
                if align_updates >= 10:
                    progress_queue.put(("advance", worker_idx, align_updates))
                    align_updates = 0

        if align_updates and progress_queue is not None and worker_idx is not None:
            progress_queue.put(("advance", worker_idx, align_updates))

        # --- Action = next frame's state ---
        if progress_queue is not None and worker_idx is not None:
            progress_queue.put(("set_phase", worker_idx, ("writing", len(all_frames))))
        write_updates = 0
        for i in range(len(all_frames)):
            if i < len(all_frames) - 1:
                all_frames[i]["action"] = all_frames[i + 1]["observation.state"]
            else:
                all_frames[i]["action"] = all_frames[i]["observation.state"]

            self.dataset.add_frame(all_frames[i], task=task_instruction)

            if progress_queue is not None and worker_idx is not None:
                write_updates += 1
                if write_updates >= 10:
                    progress_queue.put(("advance", worker_idx, write_updates))
                    write_updates = 0

        if write_updates and progress_queue is not None and worker_idx is not None:
            progress_queue.put(("advance", worker_idx, write_updates))

        logger.info(
            f"Starting save_episode() video encoding... this may take a while "
            f"(codec={self.video_codec}, pix_fmt={self.video_pix_fmt})."
        )
        self.dataset.save_episode()
        logger.info(f"Saved episode for MCAP -> dataset {self.repo_id}")


# ============================================================
# Human-readable modality metadata
# ============================================================


def generate_modality_json() -> dict[str, Any]:
    """Generate a readable summary of the state/action and camera layout."""
    return {
        "state": JOINT_STRUCTURE,
        "action": JOINT_STRUCTURE,
        "state_names": JOINT_NAMES,
        "action_names": JOINT_NAMES,
        "video": {
            "left_arm_camera_color": {"original_key": "observation.images.left_arm_camera_color"},
            "front_head_camera_left_color": {
                "original_key": "observation.images.front_head_camera_left_color"
            },
            "front_head_camera_right_color": {
                "original_key": "observation.images.front_head_camera_right_color"
            },
            "right_arm_camera_color": {"original_key": "observation.images.right_arm_camera_color"},
        },
        "annotation": {"human.task_description": {"original_key": "task"}},
    }


def save_modality_json(output_dir: str) -> None:
    """Save the human-readable modality summary next to standard metadata."""
    os.makedirs(output_dir, exist_ok=True)
    modality_path = os.path.join(output_dir, "modality.json")
    with open(modality_path, "w", encoding="utf-8") as f:
        json.dump(generate_modality_json(), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved modality.json to {modality_path}")


def ensure_info_joint_names(meta_dir: str) -> None:
    """Ensure info.json carries explicit joint names for state/action vectors."""
    info_path = os.path.join(meta_dir, "info.json")
    if not os.path.isfile(info_path):
        logger.warning(f"info.json not found, cannot add joint names: {info_path}")
        return

    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    features = info.get("features", {})
    for feature_key in ("observation.state", "action"):
        feature = features.get(feature_key)
        if isinstance(feature, dict):
            feature["shape"] = [len(JOINT_NAMES)]
            feature["names"] = JOINT_NAMES

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    logger.info(f"Ensured joint names in {info_path}")


# ============================================================
# Multi-process batch processing
# ============================================================


def split_list(lst: list[Any], n: int) -> list[list[Any]]:
    """Split a list into n approximately equal sublists."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def is_complete_dataset_dir(dataset_dir: Path) -> bool:
    """Check whether an existing sub-dataset looks complete.

    Rules:
    1. meta/info.json exists and total_episodes/total_frames are > 0
    2. info.total_episodes == number of lines in episodes.jsonl
    3. info.total_frames == sum(length) in episodes.jsonl
    """
    info_path = dataset_dir / "meta" / "info.json"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not info_path.exists() or not episodes_path.exists():
        return False

    try:
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        total_episodes = int(info.get("total_episodes", 0))
        total_frames = int(info.get("total_frames", 0))
        if total_episodes <= 0 or total_frames <= 0:
            return False

        episodes = []
        with open(episodes_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                episodes.append(json.loads(line))

        if total_episodes != len(episodes):
            return False

        frame_sum = 0
        for ep in episodes:
            frame_sum += int(ep.get("length", 0))
        return total_frames == frame_sum
    except Exception:
        return False


def process_subset(
    subset_files: list[Path],
    output_dir: str,
    export_config: dict[str, Any],
    subset_idx: int,
    progress_queue: multiprocessing.Queue | None = None,
    skip_existing: bool = False,
) -> None:
    """Worker function: process a subset of MCAP files.

    Each worker creates its own McapParser to avoid cross-process serialization.
    Each MCAP file becomes a separate sub-dataset, later merged by main().
    """
    parser = McapParser(
        config.IMAGE_TOPICS,
        config.IMAGE_INFO_TOPICS,
        config.STATE_TOPICS,
        False,
        False,
    )
    try:
        for mcap_path in subset_files:
            try:
                logger.info(f"[Worker {subset_idx + 1}] Processing {mcap_path}")
                image_buffers, state_buffers, camera_info_buffers = parser.parse(
                    str(mcap_path), output_dir
                )

                # Create temporary worker sub-directory (will be cleaned up after merge)
                dataset_name = f".tmp_worker_{subset_idx + 1}_{mcap_path.stem}"
                output_subdir = os.path.join(output_dir, dataset_name)

                converter = McapToLeRobotDataset(
                    repo_id=dataset_name,
                    output_root=output_subdir,
                    fps=export_config.get("fps", 30),
                    robot_type=export_config.get("robot_type", "G1_V2.2B"),
                    use_videos=export_config.get("use_videos", True),
                    video_codec=export_config.get("video_codec", "av1"),
                    video_pix_fmt=export_config.get("video_pix_fmt", "yuv420p"),
                    primary_camera_topic=export_config.get(
                        "primary_camera_topic",
                        "/front_head_camera/left_color/image_raw",
                    ),
                )

                task_instruction = export_config.get("task", "")
                converter.add_mcap_episode(
                    image_buffers,
                    state_buffers,
                    camera_info_buffers,
                    task_instruction,
                    progress_queue=progress_queue,
                    worker_idx=subset_idx,
                    progress_label=mcap_path.name,
                )

                save_modality_json(os.path.join(output_subdir, "meta"))
                ensure_info_joint_names(os.path.join(output_subdir, "meta"))
                logger.info(f"[Worker {subset_idx + 1}] Finished {mcap_path}")

            except Exception as e:
                logger.error(
                    f"[Worker {subset_idx + 1}] Error processing {mcap_path}: {e}",
                    exc_info=True,
                )
                if progress_queue is not None:
                    progress_queue.put(("worker_error", subset_idx, f"{mcap_path}: {e}"))
                raise
    finally:
        if progress_queue is not None:
            progress_queue.put(("worker_done", subset_idx, None))


# ============================================================
# Main entry point
# ============================================================


def main(
    mcap_dir: str,
    output_dir: str,
    export_config: dict[str, Any] | None = None,
) -> None:
    """Process MCAP files and convert to a merged LeRobot dataset.

    1. Find all .mcap files in mcap_dir
    2. Split across NUM_WORKERS parallel processes
    3. Each worker converts MCAP files to sub-datasets
    4. Merge all sub-datasets into a single dataset

    Args:
        mcap_dir: Directory containing .mcap files.
        output_dir: Output directory for the converted dataset.
        export_config: Export settings dict.
    """
    export_config = export_config or config.EXPORT_CONFIG

    mcap_files = sorted(Path(mcap_dir).glob("*.mcap"))
    if not mcap_files:
        logger.warning(f"No .mcap files found in {mcap_dir}")
        return

    total_files = len(mcap_files)
    num_workers = min(config.NUM_WORKERS, total_files)
    subsets = split_list(mcap_files, num_workers)
    progress_queue: multiprocessing.Queue = multiprocessing.Queue()

    # Record paths of generated sub-datasets for merging
    generated_datasets = []
    for i, subset in enumerate(subsets):
        for mcap_path in subset:
            dataset_name = f".tmp_worker_{i + 1}_{mcap_path.stem}"
            generated_datasets.append(os.path.join(output_dir, dataset_name))

    # Launch worker processes
    processes = []
    for i, subset in enumerate(subsets):
        p = multiprocessing.Process(
            target=process_subset,
            args=(subset, output_dir, export_config, i, progress_queue),
            name=f"Worker-{i}",
        )
        processes.append(p)
        p.start()
        print(f"Started process {i}, PID: {p.pid}")

    bars = [
        tqdm(
            total=0,
            desc=f"Worker {i + 1}",
            position=i,
            dynamic_ncols=True,
            leave=True,
            unit="step",
        )
        for i in range(num_workers)
    ]
    worker_errors = wait_for_workers(progress_queue, processes, bars)

    for bar in bars:
        bar.close()

    for p in processes:
        p.join()
        status = "OK" if p.exitcode == 0 else f"FAILED (exit code {p.exitcode})"
        print(f"Process {p.name}: {status}")
    raise_for_worker_failures(worker_errors, processes)

    # Merge all sub-datasets into final output
    try:
        merge_datasets(source_folders=generated_datasets, output_folder=output_dir)
        logger.info(f"Merged dataset saved to: {output_dir}")

        save_modality_json(os.path.join(output_dir, "meta"))
        ensure_info_joint_names(os.path.join(output_dir, "meta"))

        # Clean up temporary worker directories
        for tmp_dir in generated_datasets:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
        logger.info("Cleaned up temporary files")
    except Exception as e:
        logger.error(f"Failed to merge datasets: {e}", exc_info=True)
        raise RuntimeError(f"Failed to merge converted datasets in {output_dir}") from e

    logger.info("All processes finished.")


def cli_main() -> None:
    """Command-line entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Convert Galbot MCAP files to LeRobot dataset format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  galbot-mcap2lerobot /path/to/mcap_files /path/to/output_dataset
        """,
    )
    parser.add_argument(
        "mcap_dir",
        type=str,
        help="Directory containing .mcap files",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for the converted dataset",
    )
    args = parser.parse_args()
    main(
        mcap_dir=args.mcap_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    cli_main()

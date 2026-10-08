# mcap2lerobot: Galbot MCAP to LeRobot Dataset Converter

**English** | [中文](./README.zh-CN.md)

Convert ROS MCAP recordings from a Galbot G1 humanoid robot into [LeRobot v2.1](https://github.com/huggingface/lerobot) datasets for imitation learning.

## Version Scope (Important)

This project targets the **LeRobotDataset v2.1 format**.

This README discusses the **dataset format version**, not the `lerobot` software release version. These two must be distinguished:

| Category | Example Versions | Meaning |
|----------|------------------|---------|
| LeRobot software version | `v0.4.0`, `v0.5.0` | Release versions of the Hugging Face `lerobot` codebase |
| LeRobotDataset format version | `v2.1`, `v3.0` | Dataset directory layout, metadata organization, and loading format |

As of the official `lerobot` `v0.5.0` release, the main dataset format remains **LeRobotDataset v3.0**.

If you need **LeRobotDataset v3.0**, use the conversion script provided by `lerobot` `v0.4.0+`:

```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=<HF_USER/DATASET_ID>
```

### Dataset Version vs Code Tag

The dataset `meta/info.json` field `codebase_version` should match the LeRobot code tag used for training:

| Dataset Version | Recommended LeRobot Code Version | Notes |
|-----------------|----------------------------------|-------|
| `v2.1` | `v0.3.x` | Legacy dataset structure; still used by many older pipelines |
| `v3.0` | `v0.4+`  | New dataset structure |

## What This Does

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  MCAP Files │────▶│  mcap2lerobot    │────▶│  LeRobot Dataset    │
│  (ROS msgs) │     │  (this tool)     │     │  (parquet + video)  │
└─────────────┘     └──────────────────┘     └─────────────────────┘
```

**Input**: MCAP files containing ROS protobuf messages (4 camera streams + joint states)

**Output**: LeRobot v2.1 dataset with:
- `data/` — Parquet files (joint states + actions, 23-D vectors)
- `videos/` — AV1-encoded camera streams (4 cameras, 30 FPS)
- `meta/` — Dataset metadata (`info.json`, `modality.json`, `episodes.jsonl`, etc.)

## Galbot G1 Data Specification

### Joint Structure (23 dimensions)

| Index Range | Joint Group     | Count | Notes                      |
|-------------|-----------------|-------|----------------------------|
| 0–4         | leg             | 5     | Leg joints                 |
| 5–6         | head            | 2     | Head pan/tilt              |
| 7–13        | left_arm        | 7     | Left arm joints            |
| 14          | left_gripper    | 1     | Raw value / 1000, width in meters |
| 15–21       | right_arm       | 7     | Right arm joints           |
| 22          | right_gripper   | 1     | Raw value / 1000, width in meters |

### Camera Streams (4 cameras)

| ROS Topic                                      | Dataset Key                                         | Resolution |
|------------------------------------------------|-----------------------------------------------------|------------|
| `/left_arm_camera/color/image_raw`             | `observation.images.left_arm_camera_color`          | 480×640    |
| `/front_head_camera/left_color/image_raw`      | `observation.images.front_head_camera_left_color`   | 480×640    |
| `/front_head_camera/right_color/image_raw`     | `observation.images.front_head_camera_right_color`  | 480×640    |
| `/right_arm_camera/color/image_raw`            | `observation.images.right_arm_camera_color`         | 480×640    |

### State Topic

| ROS Topic               | Content                              |
|-------------------------|--------------------------------------|
| `singorix/wbcs/sensor`  | Joint positions (+ velocity/effort)  |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/GalaxyGeneralRobotics/galbot-mcap2lerobot.git
cd galbot-mcap2lerobot

# Install the package
pip install .
```

## Usage

```bash
galbot-mcap2lerobot <mcap_directory> <output_directory>
```

**Example:**
```bash
galbot-mcap2lerobot ./mcap ./output
```

**Note**: Configure robot ROS topics, task description, and other settings in `src/galbot_mcap2lerobot/config.py` before running. See [Configuration Reference](#configuration-reference) below.

## Output Dataset Structure

```
output_galbot/galbot_lerobot_dataset/
├── meta/
│   ├── info.json                  # Dataset metadata (features, fps, robot_type)
│   ├── modality.json              # Human-readable joint, camera, and annotation layout
│   ├── episodes.jsonl             # Per-episode metadata
│   ├── tasks.jsonl                # Task descriptions
│   └── episodes_stats.jsonl       # Per-episode statistics (min/max/mean/std)
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet # Joint state + action data
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.left_arm_camera_color/
        │   ├── episode_000000.mp4
        │   └── ...
        ├── observation.images.front_head_camera_left_color/
        │   └── ...
        ├── observation.images.front_head_camera_right_color/
        │   └── ...
        └── observation.images.right_arm_camera_color/
            └── ...
```

### Parquet Schema

Each `episode_XXXXXX.parquet` contains:

| Column              | Type          | Description                           |
|---------------------|---------------|---------------------------------------|
| `timestamp`         | float32       | Relative time from episode start (s)  |
| `frame_index`       | int64         | Frame number within episode           |
| `index`             | int64         | Global frame index across dataset     |
| `episode_index`     | int64         | Episode number                        |
| `task_index`        | int64         | Task description index                |
| `observation.state` | list[float32] | 23-D joint positions                  |
| `action`            | list[float32] | 23-D joint actions (= next state)     |

## Conversion Pipeline

### Step 1: MCAP Parsing

The `McapParser` reads each `.mcap` file and extracts:
- **image_buffers**: `{topic: [(timestamp, decoded_msg), ...]}` for all 4 cameras
- **state_buffers**: `{topic: [(timestamp, decoded_msg), ...]}` for joint states
- **camera_info_buffers**: `{topic: camera_info_dict}` for calibration data

Protobuf messages are decoded dynamically using schema descriptors embedded in the MCAP file.

### Step 2: Timestamp Alignment

The **front head left camera** is used as the timing reference (primary clock):

1. For each head camera frame timestamp `t`:
   - **Joint states**: Linearly interpolated to `t` using `scipy.interpolate.interp1d`
   - **Other cameras**: Nearest frame by timestamp

### Step 3: Data Flattening

Joint sensor data is a nested protobuf structure:

```python
# Raw protobuf structure after decoding:
joint_sensor_map = {
    "leg":            {"position": [5 floats], "velocity": [...], "effort": [...]},
    "right_arm":      {"position": [7 floats], ...},
    "right_gripper":  {"position": [1 float],  ...},
    "left_arm":       {"position": [7 floats], ...},
    "left_gripper":   {"position": [1 float],  ...},
    "head":           {"position": [2 floats], ...},
}
```

Flattened to a 23-D array in order: `leg -> head -> left_arm -> left_gripper -> right_arm -> right_gripper`

Gripper values are divided by 1000 and stored as meter-scale widths.

### Step 4: Action Definition

```
action[t] = observation.state[t+1]   (next frame's state)
action[last] = observation.state[last]  (repeat last)
```

### Step 5: Multi-Process + Merge

- MCAP files are split across `NUM_WORKERS` processes
- Each process converts its files to separate sub-datasets
- All sub-datasets are merged into a single dataset with reindexed episodes
- If an MCAP conversion or worker fails, the command reports the failure and skips the merge. A merge failure also exits with an error instead of reporting success.

## Configuration Reference

### `src/galbot_mcap2lerobot/config.py`

```python
# Input/Output paths
MCAP_DIR = "./mcap"          # Directory containing .mcap files
OUTPUT_DIR = "./output"      # Base output directory

# Export settings
EXPORT_CONFIG = {
    "enable": True,
    "repo_id": "galbot/galbot_lerobot_dataset",  # Dataset name
    "fps": 30,                                    # Video frame rate
    "robot_type": "G1_V2.2B",                     # Robot type in LeRobot metadata
    "use_videos": True,                           # Store image observations as videos
    "video_codec": "av1",                         # Video codec (e.g. av1/h264/hevc)
    "video_pix_fmt": "yuv420p",                   # Video pixel format
    "save_parsed_info": False,                    # Save debug JSON
    "enable_undistort": False,                    # Camera undistortion
    "primary_camera_topic": "/front_head_camera/left_color/image_raw",  # Timeline reference camera
    "task": "your task description here",         # Task annotation
}

# ROS topics
IMAGE_TOPICS = [
    "/left_arm_camera/color/image_raw",
    "/front_head_camera/left_color/image_raw",
    "/front_head_camera/right_color/image_raw",
    "/right_arm_camera/color/image_raw",
]

IMAGE_INFO_TOPICS = [
    "/left_arm_camera/color/camera_info",
    "/front_head_camera/left_color/camera_info",
    "/front_head_camera/right_color/camera_info",
    "/right_arm_camera/color/camera_info",
]

# Galbot G1 (fixed value): custom protobuf topic with joint_sensor_map field
STATE_TOPICS = [
    "singorix/wbcs/sensor",
]

NUM_WORKERS = 10  # Parallel processes
```

## Dependencies

```
numpy>=1.21.0
opencv-python>=4.5.0
scipy>=1.7.0
mcap>=0.0.1
protobuf>=3.20.0
lerobot==0.3.3
ffmpeg-python>=0.2.0
tqdm>=4.62.0
pandas>=1.3.0
termcolor>=2.0.0
```

Requires `ffmpeg` system package for AV1 video encoding:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## Utility Tools

### Dataset Merging

Merge multiple LeRobot datasets into one:

```bash
merge-lerobot-v2-1 \
    --source_folders dataset_a dataset_b dataset_c \
    --output_folder merged_dataset
```

### MCAP Inspection

Inspect a raw `.mcap` file without running the full conversion pipeline — useful for checking topics, schemas, and message counts, or previewing decoded messages before converting.

```bash
# List topics/schemas/message counts
inspect-mcap <file.mcap>

# Decode and print sample messages from a topic
inspect-mcap <file.mcap> --topic /front_head_camera/left_color/image_raw --limit 5

# List embedded metadata records
inspect-mcap <file.mcap> --metadata
```

## License

See the repository root for license information.

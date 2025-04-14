
# Human Tracker

A modular human tracking system using multiple tracker types from BoxMOT.

## Features

- Support for multiple trackers: `DeepOCSORT`, `BoostTrack`, `StrongSORT`, `BoTSORT`, `ByteTrack`
- Optimized parameters for dancing scenarios with occlusions
- Visualization of tracks and trajectories
- Performance statistics

## Installation Instructions

### Setting Up the Environment

#### Clone the repository

```bash
git clone <repo_url>
cd <repo_folder>
```

#### Create and activate a virtual environment

**Using `venv` (Python's built-in module):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using `conda`:**

```bash
conda create -n tracker_env python=3.10
conda activate tracker_env
```

#### Install required dependencies

```bash
pip install -r requirements.txt
```

#### Download YOLOv8 model (if not downloaded automatically)

Make sure `yolo12n.pt` is available or specify your model path using the `--model` argument.

#### Download ReID model for trackers (optional, improves tracking)

Refer to BoxMOT documentation for ReID model download instructions.

## Usage

### Basic Usage

```bash
python track.py --video path/to/video.mp4
```

### Command Line Arguments

| Argument          | Description                                                                                 | Default       |
|-------------------|---------------------------------------------------------------------------------------------|---------------|
| `--video`         | Path to input video                                                                         | `1.mp4`       |
| `--output`        | Path to output video (processed)                                                            | `1_out.mp4`   |
| `--tracker`       | Tracker type to use (`deepocsort`, `boosttrack`, `strongsort`, `botsort`, `bytetrack`)     | `deepocsort`  |
| `--model`         | Path to YOLO model                                                                          | `yolo8n.pt`   |
| `--conf`          | Confidence threshold for detections (0.0-1.0)                                               | `0.6`         |
| `--device`        | Device to use for inference (`cuda`, `cpu`)                                                 | `cuda`        |
| `--no-show`       | Disable visualization during processing                                                     | `False`       |
| `--no-trajectory` | Disable trajectory visualization                                                            | `False`       |

### Examples

- **Process video with default settings (DeepOCSORT tracker):**

```bash
python main.py --video path/to/video.mp4
```

- **Change tracker to BoostTrack for better occlusion handling:**

```bash
python main.py --video path/to/video.mp4 --tracker boosttrack
```

- **Process video on CPU with ByteTrack (fastest tracker):**

```bash
python main.py --video path/to/video.mp4 --tracker bytetrack --device cpu
```

- **Higher detection confidence for cleaner results:**

```bash
python main.py --video path/to/video.mp4 --conf 0.8
```

- **Process video without displaying (batch processing):**

```bash
python main.py --video path/to/video.mp4 --no-show
```

- **For resource-constrained devices:**

```bash
python main.py --video path/to/video.mp4 --tracker bytetrack --no-trajectory
```

## Performance Considerations

- GPU acceleration is highly recommended for real-time performance.
- BoostTrack and DeepOCSORT provide the best tracking quality for dance videos with complex movements.
- ByteTrack is the fastest tracker but may struggle with complex occlusions.
- Disable trajectory visualization (`--no-trajectory`) for better performance on low-end devices.
- Adjust confidence threshold based on scene complexity (lower for challenging scenes, higher for cleaner output).

## Requirements

- Python 3.8 or newer
- PyTorch 1.8 or newer
- BoxMOT library
- Ultralytics YOLO
- OpenCV
- CUDA-capable GPU (recommended)

## Troubleshooting

- **ReID model missing**: If you get an error about missing ReID model weights, download it manually as described in the installation steps.
- **CUDA out of memory**: Try reducing batch size or switch to CPU mode with `--device cpu`.
- **Slow performance**: Consider using ByteTrack tracker and disabling trajectory visualization.
- **Detection issues**: Adjust the confidence threshold with `--conf` parameter (lower for more detections).




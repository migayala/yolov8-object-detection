# YOLOv8 Object Detection

Real-time object detection using Ultralytics YOLOv8 with support for image inference and webcam streaming.

---

## Features

- Detect objects in static images
- Real-time webcam object detection with live FPS display
- **Professional logging system** with timestamped output
- **Performance metrics**: FPS and inference time tracking
- **Save outputs**: Export annotated images and videos
- **YAML configuration support** for reproducible experiments
- Modular code structure (`src/main.py`)
- Organized project layout for ML workflows
- Uses Ultralytics YOLOv8 (Nano version by default)
- Reproducible environment via `requirements.txt`
- Designed for use with VS Code and virtual environments
- Easy to extend (tracking, fine-tuning, UI apps, etc.)

---

## Project Structure
```
yolov8-object-detection/
├── src/
│   └── main.py          # Main detection script with logging & metrics
├── data/
│   ├── images/          # Input images
│   └── videos/          # Input videos (optional)
├── outputs/             # Saved detection results (auto-created)
├── config.yaml          # Configuration file for parameters
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, generate one:
```bash
pip install ultralytics opencv-python
pip freeze > requirements.txt
```

---

## Usage

### Quick Start with Config File

The easiest way to run the application:

```bash
# Edit config.yaml with your preferences, then:
python -m src.main --config config.yaml
```

### Run Object Detection on an Image

Place an image inside `data/images/`, then run:
```bash
python -m src.main --mode image --image data/images/memekid.jpg
```

**Save the annotated output:**
```bash
python -m src.main --mode image --image data/images/memekid.jpg --output outputs/
```

YOLOv8 will:
- Load the image
- Detect objects
- Log results with timestamps to the terminal
- Display inference time
- Show an annotated window
- Save the result to `outputs/` (if specified)

Press any key to close the window.

### Run Real-Time Webcam Detection

```bash
python -m src.main --mode webcam
```

**Record webcam detection to video:**
```bash
python -m src.main --mode webcam --output outputs/
```

This will:
- Open your default webcam
- Run YOLOv8 on each frame
- Draw bounding boxes and labels
- **Display live FPS and inference time on screen**
- Save video to `outputs/` (if specified)

Press `q` to exit.

---

## Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML config file | None |
| `--mode` | `image` or `webcam` | `image` |
| `--model` | Path to YOLO model file (`.pt`) | `yolov8n.pt` |
| `--image` | Image path for image mode | `data/images/street.jpg` |
| `--conf` | Confidence threshold | `0.5` |
| `--cam-index` | Webcam device index | `0` |
| `--output` | Directory to save outputs | None |
| `--verbose` | Enable verbose logging | `False` |

**Examples:**
```bash
# Use config file
python -m src.main --config config.yaml

# High-accuracy image detection with saved output
python -m src.main --mode image --model yolov8s.pt --conf 0.6 --output outputs/

# Webcam detection with verbose logging
python -m src.main --mode webcam --verbose --output outputs/
```

---

## Example Output

**Terminal output with logging:**
```
2025-12-31 14:30:15 - INFO - Loading model: yolov8n.pt
2025-12-31 14:30:16 - INFO - Running inference on image: data/images/memekid.jpg
2025-12-31 14:30:17 - INFO - Inference completed in 0.245s
2025-12-31 14:30:17 - INFO - Detected: person with confidence 0.92
2025-12-31 14:30:17 - INFO - Detected: bicycle with confidence 0.87
2025-12-31 14:30:17 - INFO - Detected: car with confidence 0.80
2025-12-31 14:30:17 - INFO - Total detections: 3
2025-12-31 14:30:17 - INFO - Saved annotated image to: outputs/memekid_detected_20251231_143017.jpg
2025-12-31 14:30:17 - INFO - Press any key to close the window...
```

**Webcam mode shows:**
- Live FPS counter on screen
- Inference time per frame (in milliseconds)
- Bounding boxes with class labels

An annotated detection window will appear automatically.

---

## Tech Stack

- Ultralytics YOLOv8
- OpenCV (cv2)
- Python 3.10+
- VS Code (recommended)
- GitHub SSH workflow

---

## Future Enhancements (Roadmap)

- Add object tracking (ByteTrack or DeepSORT)
- Add a Streamlit or Gradio UI
- Export detection results to a log file
- Model performance comparison (YOLOv8n, YOLOv8s, YOLOv8m)
- Fine-tune YOLO on a custom dataset
- Add a Dockerfile for deployment
- Create a HuggingFace Spaces demo

---

## License

MIT License (optional)

---

## Contributing

Pull requests are welcome. For major changes, open an issue to discuss details before submitting a PR.

---

## Support

If you find this project helpful, consider starring the repository.
from ultralytics import YOLO # type: ignore[import]
import cv2
import argparse
import logging
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_image_inference(
    model_path: str,
    image_path: str,
    conf: float = 0.5,
    output_dir: Optional[str] = None
):
    """
    Run YOLOv8 on a single image and print detections.

    Args:
        model_path: Path to YOLO model file
        image_path: Path to input image
        conf: Confidence threshold for detections
        output_dir: Optional directory to save annotated output
    """
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)

    logger.info(f"Running inference on image: {image_path}")
    start_time = time.time()

    # Run prediction on the image
    results = model.predict(source=image_path, conf=conf, verbose=False)

    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.3f}s")

    # Process and log detections
    detection_count = 0
    for result in results:
        boxes = result.boxes  # bounding boxes
        for box in boxes: # type: ignore
            cls_id = int(box.cls)        # class index (e.g. 0 = person)
            score = float(box.conf)      # confidence score
            label = model.names[cls_id]  # class name (e.g. "person")
            logger.info(f"Detected: {label} with confidence {score:.2f}")
            detection_count += 1

    logger.info(f"Total detections: {detection_count}")

    # Save output if directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(image_path).stem
        save_path = output_path / f"{image_name}_detected_{timestamp}.jpg"

        annotated_img = results[0].plot()
        cv2.imwrite(str(save_path), annotated_img)
        logger.info(f"Saved annotated image to: {save_path}")

    # Display result
    annotated_img = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated_img)
    logger.info("Press any key to close the window...")

    cv2.waitKey(0)
    cv2.destroyAllWindows()        

def run_webcam_inference(
    model_path: str,
    conf: float = 0.5,
    cam_index: int = 0,
    output_dir: Optional[str] = None
):
    """
    Run YOLOv8 on a webcam stream in real time.
    Press 'q' to exit the window.

    Args:
        model_path: Path to YOLO model file
        conf: Confidence threshold for detections
        cam_index: Webcam device index
        output_dir: Optional directory to save video output
    """
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)

    logger.info(f"Opening webcam with index {cam_index}")
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        logger.error(f"Could not open webcam with index {cam_index}")
        return

    # Get video properties for FPS calculation
    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0

    # Setup video writer if output directory specified
    video_writer = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_save_path = output_path / f"webcam_detection_{timestamp}.mp4"

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(video_save_path), fourcc, 20.0, (frame_width, frame_height)
        )
        logger.info(f"Recording video to: {video_save_path}")

    logger.info("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Could not read frame from webcam")
            break

        # Run prediction on the current frame
        inference_start = time.time()
        results = model.predict(frame, conf=conf, verbose=False)
        inference_time = time.time() - inference_start

        # Plot detections on the frame
        annotated_frame = results[0].plot()

        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:  # Update FPS every 30 frames
            fps_end_time = time.time()
            fps = 30 / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time

        # Add FPS and inference time to frame
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f} | Inference: {inference_time*1000:.1f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Show the annotated frame
        cv2.imshow("YOLOv8 Webcam", annotated_frame)

        # Save frame to video if recording
        if video_writer:
            video_writer.write(annotated_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer:
        video_writer.release()
        logger.info("Video saved successfully")
    cv2.destroyAllWindows()
    logger.info(f"Average FPS: {fps:.1f}")


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from: {config_path}")
    return config


def parse_args():
    """
    Parse command-line arguments for selecting mode and options.
    """
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection Demo")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (overrides other arguments if provided).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="image",
        choices=["image", "webcam"],
        help="Run on a single image or live webcam stream.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLOv8 model file (e.g., yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/images/street.jpg",
        help="Path to input image when mode='image'.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--cam-index",
        type=int,
        default=0,
        help="Webcam index when mode='webcam'. Usually 0 is the default camera.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Directory to save output images/videos (e.g., 'outputs/').",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )

    return parser.parse_args()


def main():
    """
    Entry point: decide whether to run in image or webcam mode.
    """
    args = parse_args()

    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        mode = config.get('mode', args.mode)
        model = config.get('model', args.model)
        image = config.get('image', args.image)
        conf = config.get('conf', args.conf)
        cam_index = config.get('cam_index', args.cam_index)
        output = config.get('output', args.output)
    else:
        mode = args.mode
        model = args.model
        image = args.image
        conf = args.conf
        cam_index = args.cam_index
        output = args.output

    logger.info(f"Starting YOLOv8 Object Detection in {mode} mode")

    if mode == "image":
        image_path = Path(image)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            raise FileNotFoundError(f"Image not found: {image_path}")
        run_image_inference(model, str(image_path), conf=conf, output_dir=output)
    else:
        run_webcam_inference(model, conf=conf, cam_index=cam_index, output_dir=output)


if __name__ == "__main__":
    main()

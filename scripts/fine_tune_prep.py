"""SortSense AI — YOLOv26 Fine-Tuning & Dataset Pipeline Utility.

Solves GitHub Issue #4:
"AI: Fine-tune(?) and Set up the AI Model In The Backend"

This utility provides automated dataset configuration, label validation,
and fine-tuning orchestration for training the YOLOv26 model on custom
waste sorting data (Plastic, Paper, Metal, General).
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("FineTunePrep")

# Standard SortSense class mapping
SORTSENSE_CLASSES: dict[int, str] = {
    0: "plastic",
    1: "paper",
    2: "metal",
    3: "general",
}


def generate_dataset_yaml(output_path: str = "dataset.yaml", root_dir: str = "./dataset") -> str:
    """Generate YOLO format dataset.yaml for YOLOv26 training.

    Args:
        output_path: Target path to write dataset.yaml.
        root_dir: Base directory containing train/val/test splits.
    """
    yaml_content = f"""# SortSense Waste Classification Dataset Configuration for YOLOv26
path: {root_dir}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {len(SORTSENSE_CLASSES)}

# Class names matching strict SortSense bin mapping:
# 0 -> plastic
# 1 -> paper
# 2 -> metal
# 3 -> general
names:
  0: plastic
  1: paper
  2: metal
  3: general
"""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml_content, encoding="utf-8")
    logger.info(f"Generated dataset YAML at: {dest.resolve()}")
    return str(dest.resolve())


def validate_yolo_labels(labels_dir: str) -> dict[str, int]:
    """Validate annotation text files in YOLO format (<class_id> <x_center> <y_center> <w> <h>).

    Ensures class_id is within 0..3 and coordinates are normalized within [0, 1].
    """
    path = Path(labels_dir)
    stats = {"valid_files": 0, "invalid_files": 0, "total_annotations": 0}
    if not path.exists():
        logger.warning(f"Labels directory '{labels_dir}' does not exist.")
        return stats

    for label_file in path.glob("*.txt"):
        is_file_valid = True
        with open(label_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    is_file_valid = False
                    break
                try:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    if cls_id not in SORTSENSE_CLASSES:
                        is_file_valid = False
                        break
                    if any(c < 0.0 or c > 1.0 for c in coords):
                        is_file_valid = False
                        break
                    stats["total_annotations"] += 1
                except ValueError:
                    is_file_valid = False
                    break

        if is_file_valid:
            stats["valid_files"] += 1
        else:
            stats["invalid_files"] += 1

    return stats


def train_yolov26(
    data_yaml: str = "dataset.yaml",
    base_model: str = "yolo26n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "cpu",
    output_dir: str = "runs/train",
) -> str | None:
    """Fine-tune YOLOv26 on custom waste sorting dataset.

    Returns the path to the best fine-tuned model weights.
    """
    logger.info(
        f"Starting YOLOv26 fine-tuning: base={base_model}, data={data_yaml}, epochs={epochs}, imgsz={imgsz}"
    )

    try:
        from ultralytics import YOLO

        model = YOLO(base_model)
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=output_dir,
            name="sortsense_yolov26",
            exist_ok=True,
        )
        best_weights = Path(output_dir) / "sortsense_yolov26" / "weights" / "best.pt"
        logger.info(f"Training completed successfully. Weights saved at: {best_weights}")
        return str(best_weights)
    except ImportError:
        logger.warning(
            "ultralytics package is not installed. To run actual fine-tuning: pip install ultralytics"
        )
        return None
    except Exception as exc:
        logger.error(f"Error during training: {exc}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="SortSense YOLOv26 Fine-Tuning Pipeline Utility")
    parser.add_argument("--gen-yaml", action="store_true", help="Generate dataset.yaml")
    parser.add_argument("--validate-labels", type=str, help="Validate YOLO label directory")
    parser.add_argument("--train", action="store_true", help="Start training loop")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Input image resolution (640 for ESP32)"
    )
    args = parser.parse_args()

    if args.gen_yaml:
        generate_dataset_yaml()

    if args.validate_labels:
        stats = validate_yolo_labels(args.validate_labels)
        print("Validation Results:", json.dumps(stats, indent=2))

    if args.train:
        train_yolov26(epochs=args.epochs, imgsz=args.imgsz)

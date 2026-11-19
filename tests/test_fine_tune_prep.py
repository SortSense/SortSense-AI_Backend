"""Unit tests for YOLOv26 fine-tuning dataset preparation script."""

from pathlib import Path

from scripts.fine_tune_prep import (
    SORTSENSE_CLASSES,
    generate_dataset_yaml,
    train_yolov26,
    validate_yolo_labels,
)


def test_sortsense_classes_mapping():
    """Verify script classes match project specification."""
    assert SORTSENSE_CLASSES[0] == "plastic"
    assert SORTSENSE_CLASSES[1] == "paper"
    assert SORTSENSE_CLASSES[2] == "metal"
    assert SORTSENSE_CLASSES[3] == "general"


def test_generate_dataset_yaml(tmp_path: Path):
    """Verify generate_dataset_yaml creates valid YAML content."""
    yaml_file = tmp_path / "test_dataset.yaml"
    result_path = generate_dataset_yaml(output_path=str(yaml_file), root_dir="./custom_data")

    assert Path(result_path).exists()
    content = yaml_file.read_text(encoding="utf-8")
    assert "nc: 4" in content
    assert "0: plastic" in content
    assert "1: paper" in content
    assert "2: metal" in content
    assert "3: general" in content


def test_validate_yolo_labels(tmp_path: Path):
    """Verify validation of YOLO format bounding box annotations."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    # Valid annotation file
    valid_file = labels_dir / "frame_001.txt"
    valid_file.write_text("0 0.5 0.5 0.2 0.3\n2 0.8 0.7 0.1 0.15\n")

    # Invalid annotation file (class_id 99 outside 0..3)
    invalid_file = labels_dir / "frame_002.txt"
    invalid_file.write_text("99 0.5 0.5 0.2 0.2\n")

    # Invalid coordinate file (> 1.0)
    invalid_file2 = labels_dir / "frame_003.txt"
    invalid_file2.write_text("1 1.5 0.5 0.2 0.2\n")

    stats = validate_yolo_labels(str(labels_dir))
    assert stats["valid_files"] == 1
    assert stats["invalid_files"] == 2
    assert stats["total_annotations"] == 2


def test_train_yolov26_fallback():
    """Verify graceful handling when ultralytics is not in test environment."""
    result = train_yolov26(epochs=1)
    # ultralytics is not installed in minimal CI test, returns None without crashing
    assert result is None or isinstance(result, str)

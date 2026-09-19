from pathlib import Path
import os
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset_yolo" / "data.yaml"
BEST_MODEL = BASE_DIR / "runs" / "yolo26_training" / "weights" / "best.pt"
RUNS_DIR = BASE_DIR / "runs"

IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = min(8, os.cpu_count() or 1)


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    if not BEST_MODEL.exists():
        raise FileNotFoundError(f"Best model not found: {BEST_MODEL}")

    if torch.cuda.is_available():
        device = 0
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("Testing on CPU")

    model = YOLO(str(BEST_MODEL))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=WORKERS,
        plots=True,
        project=str(RUNS_DIR),
        name="yolo26_test",
        exist_ok=True,
    )

    print("\nTest Results")
    print("------------------------")
    print(f"Precision:   {metrics.box.mp:.4f}")
    print(f"Recall:      {metrics.box.mr:.4f}")
    print(f"mAP@50:      {metrics.box.map50:.4f}")
    print(f"mAP@50-95:   {metrics.box.map:.4f}")
    print(f"Results:     {metrics.save_dir}")


if __name__ == "__main__":
    main()

from pathlib import Path
import os
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset_yolo" / "data.yaml"
RUNS_DIR = BASE_DIR / "runs"
LOCAL_MODEL = BASE_DIR / "yolo26n.pt"

MODEL_NAME = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else "yolo26n.pt"
RUN_NAME = "yolo26_training"

EPOCHS = 200
IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = min(8, os.cpu_count() or 1)


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    if torch.cuda.is_available():
        device = 0
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("Training on CPU")

    print("Dataset:", DATA_YAML)
    print("Model:", MODEL_NAME)

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        patience=20,
        seed=42,
        workers=WORKERS,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        amp=True,
        cache=False,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.10,
        scale=0.40,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.10,
        close_mosaic=10,
        plots=True,
    )

    print("\nTraining finished.")
    print("Best weights:", model.trainer.best)


if __name__ == "__main__":
    main()

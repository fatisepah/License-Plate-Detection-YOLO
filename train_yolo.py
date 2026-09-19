from pathlib import Path
import torch
from ultralytics import YOLO

DATA_YAML = Path(r"D:\Projects\LabelStudioML\dataset_yolo\data.yaml")

RUNS_DIR = Path(r"D:\Projects\LabelStudioML\runs")

MODEL_NAME = "yolo11n.pt"

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 8


def main():
    if torch.cuda.is_available():
        device = 0
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("Training on CPU")

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        patience=20,
        seed=42,
        workers=0,
        project=str(RUNS_DIR),
        name="yolo_training",
        # ---------------------
        # Augmentation
        # ---------------------
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.10,
        scale=0.40,
        shear=2.0,
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

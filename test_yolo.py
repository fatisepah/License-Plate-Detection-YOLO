from pathlib import Path
import torch
from ultralytics import YOLO


DATA_YAML = Path(
    r"D:\Projects\LabelStudioML\dataset_yolo\data.yaml"
)

BEST_MODEL = Path(
    r"D:\Projects\LabelStudioML\runs\yolo_training\weights\best.pt"
)

TEST_OUTPUT = Path(
    r"D:\Projects\LabelStudioML\runs"
)


def main():
    device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    model = YOLO(
        str(BEST_MODEL)
    )

    metrics = model.val(
        data=str(DATA_YAML),

        split="test",

        imgsz=640,
        batch=8,

        device=device,

        plots=True,

        project=str(TEST_OUTPUT),
        name="yolo_test",
    )

    print("\nTest Results")
    print("------------------------")

    print(
        "Precision:",
        metrics.box.mp
    )

    print(
        "Recall:",
        metrics.box.mr
    )

    print(
        "mAP@50:",
        metrics.box.map50
    )

    print(
        "mAP@50-95:",
        metrics.box.map
    )

    print(
        "\nResults directory:",
        metrics.save_dir
    )


if __name__ == "__main__":
    main()
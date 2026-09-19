from pathlib import Path
import random
import shutil
import yaml

# -----------------------------
# Settings
# -----------------------------

SOURCE_ROOT = Path(r"D:\Projects\LabelStudioML\dataset_raw")
OUTPUT_ROOT = Path(r"D:\Projects\LabelStudioML\dataset_yolo")

IMAGES_DIR = SOURCE_ROOT / "images"
LABELS_DIR = SOURCE_ROOT / "labels"
CLASSES_FILE = SOURCE_ROOT / "classes.txt"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# -----------------------------
# Helper functions
# -----------------------------

def get_images():
    images = [
        p for p in IMAGES_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise RuntimeError(
            f"No images found in: {IMAGES_DIR}"
        )

    return images


def load_classes():
    if not CLASSES_FILE.exists():
        raise FileNotFoundError(
            f"classes.txt not found: {CLASSES_FILE}"
        )

    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        classes = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return classes


def create_directories():
    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_ROOT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )


def copy_sample(image_path, split):
    label_path = LABELS_DIR / f"{image_path.stem}.txt"

    destination_image = (
        OUTPUT_ROOT
        / "images"
        / split
        / image_path.name
    )

    destination_label = (
        OUTPUT_ROOT
        / "labels"
        / split
        / f"{image_path.stem}.txt"
    )

    shutil.copy2(
        image_path,
        destination_image
    )

    if label_path.exists():
        shutil.copy2(
            label_path,
            destination_label
        )
    else:
        print(
            f"WARNING: label not found for {image_path.name}"
        )


def create_yaml(classes):
    data = {
        "path": str(OUTPUT_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            i: name
            for i, name in enumerate(classes)
        }
    }

    yaml_path = OUTPUT_ROOT / "data.yaml"

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True
        )

    print(f"\ndata.yaml created:")
    print(yaml_path)


# -----------------------------
# Main
# -----------------------------

def main():
    random.seed(SEED)

    images = get_images()
    classes = load_classes()

    random.shuffle(images)

    total = len(images)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = train_end + int(
        total * VAL_RATIO
    )

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    create_directories()

    for image in train_images:
        copy_sample(
            image,
            "train"
        )

    for image in val_images:
        copy_sample(
            image,
            "val"
        )

    for image in test_images:
        copy_sample(
            image,
            "test"
        )

    create_yaml(classes)

    print("\nDataset split completed.")
    print("-------------------------")
    print(f"Total: {total}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(val_images)}")
    print(f"Test: {len(test_images)}")


if __name__ == "__main__":
    main()
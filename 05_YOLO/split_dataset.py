from pathlib import Path
import argparse
import random
import shutil

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = Path(__file__).resolve().parent
SOURCE_IMAGE_DIR = PROJECT_ROOT / "04_Training_Data" / "images"


def ensure_structure(yolo_dir):
    for path in (
        yolo_dir / "images",
        yolo_dir / "labels",
        yolo_dir / "train" / "images",
        yolo_dir / "train" / "labels",
        yolo_dir / "val" / "images",
        yolo_dir / "val" / "labels",
    ):
        path.mkdir(parents=True, exist_ok=True)


def copy_source_images(source_dir, image_dir):
    image_files = sorted(source_dir.glob("*.tif"))
    if not image_files:
        raise FileNotFoundError(f"No .tif images found in {source_dir}")

    for image_path in image_files:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        cv2.imwrite(str(image_dir / image_path.name), image)

    return image_files


def copy_split(files, image_dir, label_dir, image_output_dir, label_output_dir):
    copied_images = 0
    copied_labels = 0

    for image_path in files:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"Skipping {image_path.name}: missing {label_path.name}")
            continue

        shutil.copy2(image_dir / image_path.name, image_output_dir / image_path.name)
        shutil.copy2(label_path, label_output_dir / label_path.name)
        copied_images += 1
        copied_labels += 1

    return copied_images, copied_labels


def split_dataset(yolo_dir, source_image_dir, train_ratio, seed):
    ensure_structure(yolo_dir)

    image_dir = yolo_dir / "images"
    label_dir = yolo_dir / "labels"
    copy_source_images(source_image_dir, image_dir)

    files = sorted(image_dir.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No .tif images found in {image_dir}")

    rng = random.Random(seed)
    rng.shuffle(files)

    split_index = int(train_ratio * len(files))
    train_files = files[:split_index]
    val_files = files[split_index:]

    train_images, train_labels = copy_split(
        train_files,
        image_dir,
        label_dir,
        yolo_dir / "train" / "images",
        yolo_dir / "train" / "labels",
    )
    val_images, val_labels = copy_split(
        val_files,
        image_dir,
        label_dir,
        yolo_dir / "val" / "images",
        yolo_dir / "val" / "labels",
    )

    print(f"Copied {len(files)} images into {image_dir}")
    print(f"Train: {train_images} images, {train_labels} labels")
    print(f"Val: {val_images} images, {val_labels} labels")


def parse_args():
    parser = argparse.ArgumentParser(description="Create YOLO train/val split.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yolo-dir", type=Path, default=YOLO_DIR)
    parser.add_argument("--source-image-dir", type=Path, default=SOURCE_IMAGE_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset(
        args.yolo_dir.resolve(),
        args.source_image_dir.resolve(),
        args.train_ratio,
        args.seed,
    )

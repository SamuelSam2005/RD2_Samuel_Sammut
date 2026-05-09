from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "04_Training_Data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "labels"
# ArcGIS class values:
# 1 - Buildings
# 2 - Roads
# 3 - Non Dev Land
# 4 - Public Open Spaces
VALUE_TO_CLASS = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
}
CLASS_NAME_TO_ID = {str(value): class_id for value, class_id in VALUE_TO_CLASS.items()}


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def bbox_to_yolo_polygon(box, width, height):
    xmin, ymin, xmax, ymax = box
    points = (
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    )
    return [
        coordinate
        for x, y in points
        for coordinate in (clamp(x / width), clamp(y / height))
    ]


def convert_xml_label(xml_path):
    root = ET.parse(xml_path).getroot()

    width = float(root.findtext("size/width", default="0"))
    height = float(root.findtext("size/height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"{xml_path} has invalid image size")

    lines = []
    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        if class_name not in CLASS_NAME_TO_ID:
            continue

        box_node = obj.find("bndbox")
        if box_node is None:
            continue

        box = tuple(
            float(box_node.findtext(tag, default="0"))
            for tag in ("xmin", "ymin", "xmax", "ymax")
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        values = bbox_to_yolo_polygon(box, width, height)
        coords = " ".join(f"{value:.6f}" for value in values)
        lines.append(f"{CLASS_NAME_TO_ID[class_name]} {coords}")

    return lines


def normalize_contour(contour, width, height):
    return [(x / width, y / height) for x, y in contour]


def convert_mask_label(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    height, width = mask.shape
    lines = []

    for class_value, class_id in VALUE_TO_CLASS.items():
        binary = (mask == class_value).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) < 3:
                continue

            contour = contour.squeeze()
            if contour.ndim != 2:
                continue

            coords = " ".join(
                f"{x:.6f} {y:.6f}"
                for x, y in normalize_contour(contour, width, height)
            )
            lines.append(f"{class_id} {coords}")

    return lines


def convert_labels(dataset_dir, output_dir):
    label_dir = dataset_dir / "labels"
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    xml_files = sorted(label_dir.glob("*.xml"))
    mask_files = sorted(label_dir.glob("*.tif")) + sorted(label_dir.glob("*.tiff"))

    if xml_files:
        files = xml_files
        converter = convert_xml_label
        mode = "XML bounding boxes"
    elif mask_files:
        files = mask_files
        converter = convert_mask_label
        mode = "raster masks"
    else:
        raise FileNotFoundError(f"No .xml, .tif, or .tiff labels found in {label_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    object_count = 0
    for label_path in tqdm(files, desc=f"Converting {mode}"):
        lines = converter(label_path)
        output_path = output_dir / f"{label_path.stem}.txt"
        output_path.write_text("\n".join(lines), encoding="utf-8")
        written += 1
        object_count += len(lines)

    print(f"Wrote {written} label files to {output_dir}")
    print(f"Wrote {object_count} YOLO objects")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ArcGIS training labels to YOLO txt labels."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Dataset folder containing images/ and labels/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where YOLO .txt labels will be written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_labels(args.dataset_dir.resolve(), args.output_dir.resolve())

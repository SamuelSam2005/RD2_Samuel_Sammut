import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform as rasterio_transform
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "06_Results"
DEFAULT_MODEL = RESULTS_DIR / "runs" / "birzebbugia_rgb_yolo_seg" / "weights" / "best.pt"
DEFAULT_SOURCE = Path(r"C:\Users\samue\OneDrive\Desktop\RD2_Project\01_Data\Satellite\Birzebb.tif")
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "inference"

CLASS_NAMES = {
    0: "Buildings",
    1: "Roads",
    2: "Non-developed Land",
    3: "Public Open Spaces",
}

CLASS_COLORS = {
    0: (0, 0, 255),
    1: (255, 165, 0),
    2: (0, 255, 0),
    3: (255, 0, 0),
}

CLASS_THRESHOLDS = {
    0: 0.35,
    1: 0.35,
    2: 0.40,
    3: 0.35,
}

MIN_AREAS = {
    0: 10,
    1: 20,
    2: 20,
    3: 20,
}


def load_geotiff(geotiff_path):
    with rasterio.open(geotiff_path) as src:
        band_count = src.count
        if band_count >= 3:
            image = src.read([1, 2, 3])
        else:
            band = src.read(1)
            image = np.stack([band, band, band], axis=0)

        image = np.transpose(image, (1, 2, 0))
        image = normalize_to_uint8(image)
        return image, src.transform, src.crs, src.bounds


def normalize_to_uint8(image):
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    output = np.zeros_like(image, dtype=np.uint8)
    for band_index in range(image.shape[2]):
        band = image[:, :, band_index]
        low, high = np.nanpercentile(band, (2, 98))
        if high <= low:
            output[:, :, band_index] = 0
            continue
        scaled = (band - low) * 255.0 / (high - low)
        output[:, :, band_index] = np.clip(scaled, 0, 255).astype(np.uint8)
    return output


def pixel_to_gps(col, row, transform, src_crs):
    x, y = xy(transform, row, col)
    if src_crs is None:
        return x, y
    lon, lat = rasterio_transform(src_crs, "EPSG:4326", [x], [y])
    return lon[0], lat[0]


def get_mask_boundary_gps(mask, transform, src_crs, sample_every=10):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    for contour in contours:
        if len(contour) < 3:
            continue

        sampled = contour[::sample_every]
        coords = []
        for point in sampled:
            col, row = point[0]
            lon, lat = pixel_to_gps(int(col), int(row), transform, src_crs)
            coords.append([lon, lat])

        if len(coords) >= 3:
            polygons.append(coords)

    return polygons


def iter_tiles(width, height, tile_size, overlap):
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            tile_h = y_end - y
            tile_w = x_end - x

            if tile_h < tile_size // 3 or tile_w < tile_size // 3:
                continue

            yield x, y, x_end, y_end


def pad_tile(tile, tile_size):
    tile_h, tile_w = tile.shape[:2]
    if tile_h == tile_size and tile_w == tile_size:
        return tile

    return cv2.copyMakeBorder(
        tile,
        0,
        tile_size - tile_h,
        0,
        tile_size - tile_w,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def run_tiled_inference(model_path, geotiff_path, output_dir, conf_threshold, tile_size, overlap, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    print(f"Loading GeoTIFF: {geotiff_path}")
    image, transform, crs, bounds = load_geotiff(geotiff_path)
    height, width = image.shape[:2]

    print(f"Image shape: {image.shape}")
    print(f"CRS: {crs}")
    print(f"Bounds: {bounds}")

    class_scores = {class_id: np.zeros((height, width), dtype=np.float32) for class_id in CLASS_NAMES}
    coverage = np.zeros((height, width), dtype=np.float32)

    tiles = list(iter_tiles(width, height, tile_size, overlap))
    total_detections = 0
    print(f"Processing {len(tiles)} tiles ({tile_size}x{tile_size}, overlap {overlap})...")

    for index, (x, y, x_end, y_end) in enumerate(tiles, 1):
        tile = image[y:y_end, x:x_end].copy()
        tile_h, tile_w = tile.shape[:2]
        tile_for_model = pad_tile(tile, tile_size)
        # tile_for_model = cv2.GaussianBlur(tile_for_model, (5, 5), 0)

        result = model.predict(
            source=tile_for_model,
            task="segment",
            conf=conf_threshold,
            iou=0.3,
            imgsz=tile_size,
            device=device,
            verbose=False,
        )[0]

        coverage[y:y_end, x:x_end] += 1

        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            total_detections += len(masks)

            for mask, class_id, confidence in zip(masks, classes, confidences):
                if class_id not in class_scores:
                    continue
                mask = cv2.resize(mask, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
                mask = mask[:tile_h, :tile_w]
                class_scores[class_id][y:y_end, x:x_end] += mask * float(confidence)

        if index % 25 == 0 or index == len(tiles):
            print(f"  Processed {index}/{len(tiles)} tiles")

    coverage[coverage == 0] = 1
    print(f"Raw detections before merging: {total_detections}")

    overlay = np.zeros_like(image, dtype=np.uint8)
    gps_data = {class_id: [] for class_id in CLASS_NAMES}
    final_masks = {}

    for class_id, class_name in CLASS_NAMES.items():
        score = class_scores[class_id] / coverage
        threshold = CLASS_THRESHOLDS[class_id]
        mask_binary = (score > threshold).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        kept = 0

        for component_id in range(1, num_labels):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < MIN_AREAS[class_id]:
                continue

            component_mask = (labels == component_id).astype(np.uint8)
            centroid_col = int(centroids[component_id][0])
            centroid_row = int(centroids[component_id][1])
            lon, lat = pixel_to_gps(centroid_col, centroid_row, transform, crs)
            boundary_polygons = get_mask_boundary_gps(component_mask, transform, crs)

            gps_data[class_id].append(
                {
                    "class": class_name,
                    "confidence": float(score[component_mask > 0].mean()),
                    "centroid_gps": {"longitude": lon, "latitude": lat},
                    "boundary_polygons": boundary_polygons,
                    "area_pixels": area,
                }
            )
            kept += 1

            color = CLASS_COLORS[class_id]
            for channel in range(3):
                overlay[:, :, channel][component_mask == 1] = color[channel]

        final_masks[class_id] = mask_binary
        print(f"{class_name}: kept {kept} objects")

    alpha = 0.5
    vis_image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    stem = Path(geotiff_path).stem

    output_image_path = output_dir / f"{stem}_segmented_tiled.png"
    cv2.imwrite(str(output_image_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization: {output_image_path}")

    output_legend_path = output_dir / f"{stem}_with_legend_tiled.png"
    save_legend_figure(vis_image, gps_data, output_legend_path, stem)
    print(f"Saved legend figure: {output_legend_path}")

    output_json_path = output_dir / f"{stem}_gps_data_tiled.json"
    save_gps_json(output_json_path, geotiff_path, crs, bounds, tile_size, overlap, conf_threshold, gps_data)
    print(f"Saved GPS data: {output_json_path}")

    output_masks_npz = output_dir / f"{stem}_class_masks.npz"
    np.savez_compressed(output_masks_npz, **{str(k): v for k, v in final_masks.items()})
    print(f"Saved class masks: {output_masks_npz}")

    open_spaces = gps_data[3]
    if open_spaces:
        output_open_spaces_path = output_dir / f"{stem}_open_spaces_tiled.json"
        with output_open_spaces_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "public_open_spaces_count": len(open_spaces),
                    "locations": open_spaces,
                },
                file,
                indent=2,
            )
        print(f"Saved Public Open Spaces data: {output_open_spaces_path}")

    print_summary(gps_data)
    return gps_data


def save_legend_figure(vis_image, gps_data, output_path, stem):
    fig, ax = plt.subplots(figsize=(20, 16), facecolor="white", dpi=300)
    ax.imshow(vis_image)
    ax.axis("off")
    ax.set_title(f"Tiled Segmentation Results - {stem}", fontsize=18, weight="bold", pad=20)

    legend_patches = []
    for class_id, name in CLASS_NAMES.items():
        color = tuple(channel / 255 for channel in CLASS_COLORS[class_id])
        count = len(gps_data[class_id])
        legend_patches.append(mpatches.Patch(color=color, label=f"{name} ({count})"))

    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=14,
        framealpha=0.95,
        edgecolor="black",
        facecolor="white",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.5, facecolor="white")
    plt.close(fig)


def save_gps_json(output_path, geotiff_path, crs, bounds, tile_size, overlap, conf_threshold, gps_data):
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "geotiff": str(geotiff_path),
                "crs": str(crs),
                "bounds": {
                    "left": bounds.left,
                    "bottom": bounds.bottom,
                    "right": bounds.right,
                    "top": bounds.top,
                },
                "inference_params": {
                    "tile_size": tile_size,
                    "overlap": overlap,
                    "conf_threshold": conf_threshold,
                    "class_thresholds": {CLASS_NAMES[k]: v for k, v in CLASS_THRESHOLDS.items()},
                    "min_areas": {CLASS_NAMES[k]: v for k, v in MIN_AREAS.items()},
                },
                "detections": {CLASS_NAMES[k]: v for k, v in gps_data.items()},
                "summary": {
                    "total_detections": sum(len(v) for v in gps_data.values()),
                    "by_class": {CLASS_NAMES[k]: len(v) for k, v in gps_data.items()},
                },
            },
            file,
            indent=2,
        )


def print_summary(gps_data):
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    for class_id, class_name in CLASS_NAMES.items():
        print(f"{class_name}: {len(gps_data[class_id])}")

    open_spaces = gps_data[3]
    if open_spaces:
        print("\nPUBLIC OPEN SPACES SAMPLE")
        for index, space in enumerate(open_spaces[:5], 1):
            gps = space["centroid_gps"]
            print(
                f"  {index}. GPS ({gps['longitude']:.6f}, {gps['latitude']:.6f}), "
                f"confidence {space['confidence']:.3f}, area {space['area_pixels']} px"
            )
        if len(open_spaces) > 5:
            print(f"  ... and {len(open_spaces) - 5} more")


def find_geotiffs(source):
    source = Path(source)
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(source.rglob("*.tif")) + sorted(source.rglob("*.tiff"))
    raise FileNotFoundError(f"Source not found: {source}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run tiled geospatial YOLO segmentation inference.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--device", default=0, help='Use 0 for GPU or "cpu".')
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model.resolve()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    geotiff_files = find_geotiffs(source)
    if not geotiff_files:
        raise FileNotFoundError(f"No GeoTIFF files found in: {source}")

    print(f"Found {len(geotiff_files)} GeoTIFF file(s)")
    for geotiff_path in geotiff_files:
        run_output_dir = output_dir / f"{geotiff_path.stem}_results_tiled"
        run_tiled_inference(
            model_path=model_path,
            geotiff_path=geotiff_path,
            output_dir=run_output_dir,
            conf_threshold=args.conf,
            tile_size=args.tile_size,
            overlap=args.overlap,
            device=args.device,
        )


if __name__ == "__main__":
    main()

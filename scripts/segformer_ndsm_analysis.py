from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio


# Edit these paths if your project structure changes.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEGFORMER_IMAGE = (
    PROJECT_ROOT
    / "06_Results"
    / "segformer"
    / "large_image_pred_color_Birzebbugia.tif"
)
NDSM_PATH = (
    PROJECT_ROOT
    / "02_Preprocessed"
    / "nDSM"
    / "ndsm_FINAL.tif"
)
FALLBACK_NDSM_PATH = PROJECT_ROOT / "02_Preprocessed" / "nDSM" / "b_ndsm_clean"
OUTPUT_CSV = PROJECT_ROOT / "06_Results" / "segformer_ndsm_stats.csv"

# Class colours in RGB order, based on the SegFormer palette.
BUILDING_RGB = np.array([230, 25, 75], dtype=np.uint8)

# Combined "Other Land" variable: roads, POS/grey, and non-developed land.
OTHER_LAND_RGBS = np.array(
    [
        [0, 130, 200],  # Roads / blue
        [60, 180, 75],  # Non-developed land / green
        [160, 160, 160],  # POS or grey class
    ],
    dtype=np.uint8,
)


def load_segformer_rgb(path: Path) -> np.ndarray:
    """Load the colour-coded SegFormer output as an RGB array."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read SegFormer image: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"Unsupported SegFormer image shape: {image.shape}")

    return image


def print_unique_colours(rgb_image: np.ndarray) -> None:
    """Print all unique RGB colours in the SegFormer raster."""
    unique_colours = np.unique(rgb_image.reshape(-1, 3), axis=0)

    print("\nUnique RGB colours found in SegFormer prediction:")
    for colour in unique_colours:
        print(f"RGB({colour[0]}, {colour[1]}, {colour[2]})")
    print(f"Total unique colours: {len(unique_colours)}\n")


def resize_to_ndsm_shape(rgb_image: np.ndarray, ndsm_shape: tuple[int, int]) -> np.ndarray:
    """Resize the SegFormer RGB raster to match the nDSM rows and columns."""
    ndsm_height, ndsm_width = ndsm_shape
    seg_height, seg_width = rgb_image.shape[:2]

    if (seg_height, seg_width) == (ndsm_height, ndsm_width):
        return rgb_image

    print(
        "Shape mismatch detected. "
        f"SegFormer: {(seg_height, seg_width)}, nDSM: {(ndsm_height, ndsm_width)}"
    )
    print("Resizing SegFormer raster with nearest-neighbour interpolation.\n")

    return cv2.resize(
        rgb_image,
        (ndsm_width, ndsm_height),
        interpolation=cv2.INTER_NEAREST,
    )


def resolve_ndsm_path(primary_path: Path, fallback_path: Path) -> Path:
    """Use the primary nDSM path, then fall back to a file or folder path."""
    if primary_path.exists():
        return primary_path

    if fallback_path.is_file():
        print(f"Primary nDSM not found. Using fallback nDSM: {fallback_path}\n")
        return fallback_path

    if fallback_path.is_dir():
        tif_files = sorted(
            list(fallback_path.glob("*.tif")) + list(fallback_path.glob("*.tiff"))
        )
        if tif_files:
            print(f"Primary nDSM not found. Using fallback nDSM: {tif_files[0]}\n")
            return tif_files[0]

    raise FileNotFoundError(
        "Could not find nDSM raster at either path:\n"
        f"Primary: {primary_path}\n"
        f"Fallback: {fallback_path}"
    )


def clean_height_values(values: np.ndarray, nodata: float | int | None) -> np.ndarray:
    """Remove NoData, NaN, infinite, and negative height values."""
    values = values.astype("float64", copy=False)
    valid = np.isfinite(values) & (values >= 0)

    if nodata is not None:
        valid &= values != nodata

    return values[valid]


def calculate_stats(class_name: str, values: np.ndarray) -> dict[str, float | int | str]:
    """Calculate summary height statistics for one segmentation class."""
    if values.size == 0:
        return {
            "Class": class_name,
            "Pixel count": 0,
            "Mean height": np.nan,
            "Median height": np.nan,
            "Standard deviation": np.nan,
            "Minimum height": np.nan,
            "Maximum height": np.nan,
        }

    return {
        "Class": class_name,
        "Pixel count": int(values.size),
        "Mean height": float(np.mean(values)),
        "Median height": float(np.median(values)),
        "Standard deviation": float(np.std(values)),
        "Minimum height": float(np.min(values)),
        "Maximum height": float(np.max(values)),
    }


def main() -> None:
    segformer_rgb = load_segformer_rgb(SEGFORMER_IMAGE)
    print_unique_colours(segformer_rgb)

    ndsm_path = resolve_ndsm_path(NDSM_PATH, FALLBACK_NDSM_PATH)
    with rasterio.open(ndsm_path) as ndsm_src:
        ndsm = ndsm_src.read(1)
        ndsm_nodata = ndsm_src.nodata

    segformer_rgb = resize_to_ndsm_shape(segformer_rgb, ndsm.shape)

    # Build class masks from exact RGB colour matches.
    building_mask = np.all(segformer_rgb == BUILDING_RGB, axis=2)
    other_land_mask = np.any(
        np.all(segformer_rgb[:, :, None, :] == OTHER_LAND_RGBS[None, None, :, :], axis=3),
        axis=2,
    )

    # Extract and clean nDSM heights under each class mask.
    building_heights = clean_height_values(ndsm[building_mask], ndsm_nodata)
    other_land_heights = clean_height_values(ndsm[other_land_mask], ndsm_nodata)

    results = pd.DataFrame(
        [
            calculate_stats("Buildings", building_heights),
            calculate_stats("Other Land", other_land_heights),
        ]
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)

    print("nDSM height statistics:")
    print(results.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

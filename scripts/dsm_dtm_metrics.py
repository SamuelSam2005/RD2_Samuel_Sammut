from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize


# Edit these paths if your project structure changes.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NDSM_PATH = PROJECT_ROOT / "02_Preprocessed" / "nDSM" / "ndsm_FINAL.tif"
GROUND_TRUTH_SHP = PROJECT_ROOT / "03_Annotations" / "Birzebbugia_Labels.shp"
OUTPUT_CSV = PROJECT_ROOT / "06_Results" / "dsm_dtm_metrics.csv"

# Class mapping from the ArcGIS labels:
# 1 = Buildings, 2 = Roads, 3 = Non-developed land, 4 = POS.
BUILDING_CLASS = 1
OTHER_LAND_CLASSES = [2, 3, 4]

# DSM/DTM building rule. Pixels above this nDSM height are treated as Buildings.
BUILDING_HEIGHT_THRESHOLD_METRES = 2.0


def load_ndsm(path: Path) -> tuple[np.ndarray, rasterio.Affine, object]:
    """Load the nDSM raster and return its array, transform, and CRS."""
    if not path.exists():
        raise FileNotFoundError(f"nDSM raster not found: {path}")

    with rasterio.open(path) as src:
        ndsm = src.read(1).astype("float32")
        nodata = src.nodata

        valid = np.isfinite(ndsm)
        if nodata is not None:
            valid &= ndsm != nodata
        valid &= ndsm >= 0

        ndsm[~valid] = np.nan
        return ndsm, src.transform, src.crs


def rasterize_ground_truth(
    shapefile_path: Path,
    target_shape: tuple[int, int],
    target_transform: rasterio.Affine,
    target_crs: object,
) -> np.ndarray:
    """Rasterize ground-truth polygons to the nDSM grid."""
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Ground-truth shapefile not found: {shapefile_path}")

    ground_truth = gpd.read_file(shapefile_path)
    if ground_truth.crs != target_crs:
        ground_truth = ground_truth.to_crs(target_crs)

    shapes = (
        (geometry, int(class_value))
        for geometry, class_value in zip(ground_truth.geometry, ground_truth["Class"])
        if geometry is not None and not geometry.is_empty
    )

    return rasterize(
        shapes,
        out_shape=target_shape,
        transform=target_transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )


def calculate_metrics(
    class_name: str,
    prediction_mask: np.ndarray,
    truth_mask: np.ndarray,
    evaluation_mask: np.ndarray,
) -> dict[str, float | int | str]:
    """Calculate IoU, precision, and recall for one binary class mask."""
    true_positive = int((prediction_mask & truth_mask & evaluation_mask).sum())
    false_positive = int((prediction_mask & ~truth_mask & evaluation_mask).sum())
    false_negative = int((~prediction_mask & truth_mask & evaluation_mask).sum())

    iou_denominator = true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    iou = true_positive / iou_denominator if iou_denominator else np.nan
    precision = (
        true_positive / precision_denominator if precision_denominator else np.nan
    )
    recall = true_positive / recall_denominator if recall_denominator else np.nan

    return {
        "Method": "DSM/DTM Study",
        "Class": class_name,
        "IoU (%)": round(iou * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "True Positive": true_positive,
        "False Positive": false_positive,
        "False Negative": false_negative,
    }


def main() -> None:
    ndsm, transform, crs = load_ndsm(NDSM_PATH)
    ground_truth = rasterize_ground_truth(GROUND_TRUTH_SHP, ndsm.shape, transform, crs)

    # Evaluate only annotated pixels with valid nDSM values.
    evaluation_mask = np.isfinite(ndsm) & (ground_truth > 0)

    truth_buildings = evaluation_mask & (ground_truth == BUILDING_CLASS)
    truth_other_land = evaluation_mask & np.isin(ground_truth, OTHER_LAND_CLASSES)

    predicted_buildings = evaluation_mask & (ndsm > BUILDING_HEIGHT_THRESHOLD_METRES)
    predicted_other_land = evaluation_mask & ~predicted_buildings

    results = pd.DataFrame(
        [
            calculate_metrics(
                "Buildings",
                predicted_buildings,
                truth_buildings,
                evaluation_mask,
            ),
            calculate_metrics(
                "Other Land",
                predicted_other_land,
                truth_other_land,
                evaluation_mask,
            ),
        ]
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)

    print("DSM/DTM metrics:")
    print(results.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

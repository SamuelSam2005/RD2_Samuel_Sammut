from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject


# ---------- PATHS ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASKS_PATH = (
    PROJECT_ROOT
    / "06_Results"
    / "inference"
    / "Birzebb_results_tiled"
    / "Birzebb_class_masks.npz"
)

PREFERRED_NDSM_PATH = (
    PROJECT_ROOT
    / "02_Preprocessed"
    / "nDSM"
    / "Birzebbuga_nDSM2018_Clean.tif"
)

FALLBACK_NDSM_PATH = PROJECT_ROOT / "02_Preprocessed" / "nDSM" / "b_ndsm_clean"

OUTPUT_CSV = PROJECT_ROOT / "06_Results" / "ndsm_class_height_statistics.csv"

REFERENCE_GEOTIFF = PROJECT_ROOT / "01_Data" / "Satellite" / "Birzebb.tif"

CLASS_NAMES = {
    "0": "Buildings",
    "1": "Roads",
    "2": "Non-developed Land",
    "3": "Public Open Spaces",
}


def get_ndsm_path():
    if PREFERRED_NDSM_PATH.exists():
        return PREFERRED_NDSM_PATH
    if FALLBACK_NDSM_PATH.exists():
        return FALLBACK_NDSM_PATH
    raise FileNotFoundError(
        "nDSM raster not found. Checked:\n"
        f"{PREFERRED_NDSM_PATH}\n"
        f"{FALLBACK_NDSM_PATH}"
    )


def load_ndsm_aligned_to_reference(ndsm_path, reference_path):
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference GeoTIFF not found:\n{reference_path}")

    with rasterio.open(reference_path) as ref:
        ref_shape = (ref.height, ref.width)
        ref_transform = ref.transform
        ref_crs = ref.crs

    with rasterio.open(ndsm_path) as src:
        ndsm = src.read(1).astype("float32")
        nodata = src.nodata

        if nodata is not None:
            ndsm[ndsm == nodata] = np.nan

        if ndsm.shape == ref_shape and src.crs == ref_crs and src.transform == ref_transform:
            return ndsm

        aligned = np.full(ref_shape, np.nan, dtype="float32")
        reproject(
            source=ndsm,
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    return aligned


def main():
    if not MASKS_PATH.exists():
        raise FileNotFoundError(f"Prediction masks not found:\n{MASKS_PATH}")

    ndsm_path = get_ndsm_path()

    print("Loading prediction masks...")
    masks = np.load(MASKS_PATH)

    print(f"Loading and aligning nDSM:\n{ndsm_path}")
    ndsm = load_ndsm_aligned_to_reference(ndsm_path, REFERENCE_GEOTIFF)

    ndsm[ndsm < 0] = np.nan

    rows = []

    for class_id, class_name in CLASS_NAMES.items():
        if class_id not in masks:
            print(f"Class {class_id} missing from masks. Skipping.")
            continue

        mask = masks[class_id].astype(bool)

        if mask.shape != ndsm.shape:
            raise ValueError(
                f"Shape mismatch for {class_name}: mask {mask.shape}, nDSM {ndsm.shape}. "
                "The prediction mask and nDSM must have the same dimensions."
            )

        values = ndsm[mask]
        values = values[~np.isnan(values)]

        if len(values) == 0:
            rows.append(
                {
                    "Class": class_name,
                    "Pixel_Count": 0,
                    "Mean_Height": None,
                    "Median_Height": None,
                    "Min_Height": None,
                    "Max_Height": None,
                    "Std_Height": None,
                }
            )
            continue

        rows.append(
            {
                "Class": class_name,
                "Pixel_Count": int(len(values)),
                "Mean_Height": float(np.mean(values)),
                "Median_Height": float(np.median(values)),
                "Min_Height": float(np.min(values)),
                "Max_Height": float(np.max(values)),
                "Std_Height": float(np.std(values)),
            }
        )

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nHeight statistics:")
    print(df)
    print(f"\nSaved to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()

# RD2 Project Workspace

This workspace is organised so the assignment materials can be found by stage of the workflow.

## Folder Guide

| Folder | Contents |
| --- | --- |
| `01_Data/` | Original input datasets, including satellite imagery, DTM files, and boundary data. |
| `02_Preprocessed/` | Processed raster datasets such as aligned rasters and nDSM outputs. |
| `03_Annotations/` | GIS annotation and label shapefiles. |
| `04_Training_Data/` | Exported image tiles, XML labels, model definition files, and training metadata. |
| `05_YOLO/` | YOLO dataset files, training/validation split, scripts, model weights, and dataset config. |
| `06_Results/` | Final outputs, inference products, metric CSV files, run plots, and visual result images. |
| `ArcGIS/` | ArcGIS Pro project files and geodatabase content. |
| `scripts/` | Standalone analysis scripts used to calculate metrics from DSM/DTM/nDSM results. |
| `99_Archive_Generated/` | Non-essential generated cache files moved out of working folders for cleanliness. |

## Key Entry Points

- YOLO training script: `05_YOLO/train_yolo.py`
- YOLO inference script: `05_YOLO/run_inference.py`
- YOLO dataset config: `05_YOLO/dataset.yaml`
- DSM/DTM metrics script: `scripts/dsm_dtm_metrics.py`
- SegFormer nDSM analysis script: `scripts/segformer_ndsm_analysis.py`

## Main Results

- Inference outputs: `06_Results/inference/`
- YOLO training and validation plots: `06_Results/runs/`
- MMSEG visual outputs: `06_Results/mmseg_visuals/`
- Summary CSV files:
  - `06_Results/dsm_dtm_metrics.csv`
  - `06_Results/ndsm_class_height_statistics.csv`
  - `06_Results/segformer_ndsm_stats.csv`

## Notes

- Raw `Attard` satellite raster files are stored with the other satellite imagery in `01_Data/Satellite/`.
- Python bytecode cache folders were moved to `99_Archive_Generated/python_cache/`; they are not required to run the project.
- The project is not currently a Git repository, so keep a backup copy before making large structural changes.

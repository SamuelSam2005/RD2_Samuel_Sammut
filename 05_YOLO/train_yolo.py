# train_yolo_seg.py
from ultralytics import YOLO
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR.parent / "06_Results"
DATASET_YAML = PROJECT_DIR / "dataset.yaml"
RUN_NAME = "birzebbugia_rgb_yolo_seg"

def main():
    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Missing dataset.yaml at: {DATASET_YAML}")

    model = YOLO("yolo11n-seg.pt")

    model.train(
        data=str(DATASET_YAML),
        task="segment",
        epochs=50,
        imgsz=640,
        batch=8,
        workers=2,
        project=str(RESULTS_DIR / "runs"),
        name=RUN_NAME,
        exist_ok=True,
        patience=15,
        device=0,   # use "cpu" if no GPU
    )

    print("Training complete.")
    print(f"Results saved in: {RESULTS_DIR / 'runs' / RUN_NAME}")

if __name__ == "__main__":
    main()

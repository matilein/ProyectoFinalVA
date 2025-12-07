from pathlib import Path
from ultralytics import YOLO

BASE = Path.home() / "ProyectoFinalVA"

def train_rgb_baseline():
    model = YOLO("yolo11n.pt")

    model.train(
        data=BASE / "YOLO/RGB/YOLO_rgb.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        name="RGB_baseline",
        project=BASE / "runs/detect",
        task="detect"
    )

def train_t_baseline():
    model = YOLO("yolo11n.pt")

    model.train(
        data=BASE / "YOLO/RGB/YOLO_t.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        name="T_baseline",
        project=BASE / "runs/detect",
        task="detect"
    )


if __name__ == "__main__":
    train_rgb_baseline()
    train_t_baseline()

import os
import yaml
import torch
from math import floor
from pathlib import Path
from ultralytics import YOLO
from ultralytics import settings

dataset_dirpath = Path("../CVDatasets/bdd100k_yolo_format/dataset.yaml").resolve()  # BDD
# dataset_dirpath = Path("../CVDatasets/VOC_yolo_format/dataset.yaml").resolve()  # VOC
yolo_cfg_file_path = Path("./config/yolov8.yaml").resolve()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if torch.cuda.is_available():
        devices = ",".join([str(i) for i in range(torch.cuda.device_count())])
        workers = min(8, int(floor(os.cpu_count() / torch.cuda.device_count() * 0.95)))
    else:
        devices = device
        workers = int(floor(os.cpu_count() * 0.9))
    # Get yolo inference overrides
    overrides = dict()
    overrides["data"] = dataset_dirpath
    overrides["device"] = devices
    overrides["workers"] = workers
    overrides["verbose"] = True

    # Load yolo cfg
    yolo_cfg = {}
    if yolo_cfg_file_path is not None:
        with open(yolo_cfg_file_path.as_posix(), "r") as f:
            yolo_cfg.update(yaml.safe_load(f))
    yolo_cfg.update(overrides)

    # Load model
    yolo = YOLO(model=yolo_cfg["model"], task=yolo_cfg["task"])
    yolo.overrides.update(yolo_cfg)

    # Train the model
    results = yolo.train()


if __name__ == "__main__":
    # Track with mlflow
    # Environment variables needed:
    # MLFLOW_EXPERIMENT_NAME=Yolo_train_bdd
    # MLFLOW_TRACKING_URI=http://10.8.33.50:5050 (MLFlow server in the TDL server)
    settings.update({"mlflow": True})
    main()

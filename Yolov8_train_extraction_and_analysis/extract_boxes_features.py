import math
import os
import hydra
import yaml
from pathlib import Path
import torch
from ood_detect_fns.uncertainty_estimation import Hook
from ood_detect_fns.uncertainty_estimation import BoxFeaturesExtractor
from omegaconf import DictConfig
from ultralytics import YOLO
import warnings

from datasets_loaders import ood_datasets_loaders, ind_datasets_loaders

ind_dataset_dirpath = {
    "bdd": Path("../CVDatasets/bdd100k_yolo_format/dataset.yaml").resolve(),
    "voc": Path("../CVDatasets/VOC_yolo_format/dataset.yaml").resolve(),
}
ood_datasets_parent_directory = Path("../CVDatasets/").resolve()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# EXTRACT_ENTROPIES = False
EXTRACT_IND = True


@hydra.main(version_base=None, config_path="config", config_name="config_ood_box_detection.yaml")
def run(cfg: DictConfig):
    print(
        f"Extracting boxes from InD {cfg.ind_dataset}, OoD {cfg.ood_datasets}, "
        f"{'stds ' if cfg.return_variances else ''}"
        f"hooked {cfg.hooked_module} output {cfg.hook_output} "
        f"roi_output_size {cfg.roi_output_sizes} "
        f"roi_sampling_ratio {cfg.roi_sampling_ratio} "
        f"Train_conf: {cfg.train_predict_confidence} "
        f"Inf_conf: {cfg.inference_predict_confidence} "
        f"model {cfg.model}/{cfg.name}"
    )

    # Directory where samples will be saved
    ls_samples_dir = Path(cfg.latent_samples_folder).resolve()
    if os.path.exists(ls_samples_dir):
        warnings.warn(f"Destination folder {ls_samples_dir} already exists!")
    os.makedirs(ls_samples_dir, exist_ok=True)

    ###############################################################
    # Define dataloaders
    ###############################################################
    ind_dataset_dict = ind_datasets_loaders(cfg=cfg, ind_dataset_dirpath=ind_dataset_dirpath[cfg.ind_dataset])
    ood_datasets_dict = ood_datasets_loaders(parent_directory=ood_datasets_parent_directory, cfg=cfg)
    ###########################################################
    # Load model
    ###########################################################
    yolo = load_yolo_model_with_overrides(cfg, data=ind_dataset_dirpath[cfg.ind_dataset].as_posix())

    # Hook layers
    hooked_layers = [Hook(yolo.model.model._modules[str(layer)]) for layer in cfg.hooked_module]
    # Instantiate samples extractor
    samples_extractor = BoxFeaturesExtractor(
        model=yolo,
        hooked_layers=hooked_layers,
        device=device,
        output_sizes=cfg.roi_output_sizes,
        sampling_ratio=cfg.roi_sampling_ratio,
        return_raw_predictions=cfg.return_raw_predictions,
        return_stds=cfg.return_variances,
        hook_layer_output=cfg.hook_output,
        architecture="yolov8"
    )
    ##########################################################
    # Extract samples
    ##########################################################
    # =============== InD ========================
    if EXTRACT_IND:
        # ind_entropies = {}
        for split, data_loader in ind_dataset_dict.items():
            # Create identifiable file names
            file_name = (
                f"{ls_samples_dir}/ind_{cfg.ind_dataset}_{split}_boxes_"
                f"{'stds_' if cfg.return_variances else ''}"
                f"hooked{'_'.join(map(str, cfg.hooked_module))}_"
                f"output{cfg.hook_output}_"
                f"roi_s{''.join(map(str, cfg.roi_output_sizes))}_"
                f"roi_sr{cfg.roi_sampling_ratio}"
                f"{'_trc_' + str(int(cfg.train_predict_confidence * 100)) if split == 'train' else ''}"
                f"{'_infc_' + str(int(cfg.inference_predict_confidence * 100)) if split == 'valid' else ''}"
            )
            if not os.path.exists(file_name + "_ls_samples.pt"):
                # Extract and save latent space samples
                print(f"\nExtracting InD {cfg.ind_dataset} {split} boxes ls samples.")
                latent_activations = samples_extractor.get_ls_samples(
                    data_loader,
                    predict_conf=cfg.train_predict_confidence if split == "train" else cfg.inference_predict_confidence,
                )
                if cfg.save_latent_samples:
                    print(f"\nSaving InD {split} boxes ls samples in {file_name}_ls_samples.pt")
                    torch.save(latent_activations, file_name + "_ls_samples.pt")
            else:
                print(f"{file_name}_ls_samples.pt already exists!")

    # ================= OoD ====================
    # ood_entropies = {}
    for ood_dataset, data_loader in ood_datasets_dict.items():
        # Create identifiable file names
        file_name = (
            f"{ls_samples_dir}/ood_{ood_dataset}_boxes_"
            f"{'stds_' if cfg.return_variances else ''}"
            f"hooked{'_'.join(map(str, cfg.hooked_module))}_"
            f"output{cfg.hook_output}_"
            f"roi_s{''.join(map(str, cfg.roi_output_sizes))}_"
            f"roi_sr{cfg.roi_sampling_ratio}"
            f"{'_infc_' + str(int(cfg.inference_predict_confidence * 100))}"
        )
        if not os.path.exists(file_name + "_ls_samples.pt"):
            # Extract and save latent space samples
            print(f"\nExtracting OoD {ood_dataset} boxes ls samples.")
            latent_activations = samples_extractor.get_ls_samples(
                data_loader,
                predict_conf=cfg.inference_predict_confidence
            )
            if cfg.save_latent_samples:
                print(f"\nSaving OoD {ood_dataset} boxes ls samples in {file_name}_ls_samples.pt")
                torch.save(latent_activations, file_name + "_ls_samples.pt")
        else:
            print(f"{file_name}_ls_samples.pt already exists!")

    print("Done!")


def load_yolo_model_with_overrides(cfg: DictConfig, data: str) -> torch.nn.Module:
    yolo_config_filepath = cfg.yolo_cfg_file_path
    if torch.cuda.is_available():
        devices = ",".join([str(i) for i in range(torch.cuda.device_count())])
        workers = int(math.floor(os.cpu_count() / torch.cuda.device_count() * 0.95))
    else:
        devices = device
        workers = int(math.floor(os.cpu_count() * 0.9))
    # Get yolo inference overrides
    overrides = dict()
    overrides["data"] = data
    overrides["device"] = devices
    overrides["batch"] = cfg.batch_size
    overrides["nbs"] = overrides["batch"]
    overrides["workers"] = workers
    overrides["model"] = cfg.model_filepath
    overrides["verbose"] = cfg.verbose_inference
    overrides["imgsz"] = cfg.imgsize

    # Load yolo cfg
    if not isinstance(yolo_config_filepath, Path):
        yolo_config_filepath = Path(yolo_config_filepath)
    yolo_cfg = {}
    if yolo_config_filepath is not None:
        with open(yolo_config_filepath.as_posix(), "r") as f:
            yolo_cfg.update(yaml.safe_load(f))
    yolo_cfg.update(overrides)

    # Load model
    yolo = YOLO(model=cfg.model_filepath, task=cfg.task)
    yolo.overrides.update(yolo_cfg)

    return yolo


if __name__ == "__main__":
    run()

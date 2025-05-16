from pathlib import Path
from typing import Dict
from omegaconf import DictConfig
import numpy as np
from ultralytics.data.loaders import LoadImagesAndVideos

ood_datapaths = {
    "bdd": {
        "coco": "id_bdd_ood_coco/val2017",
        "openimages": "OpenImages/ood_classes_rm_overlap/images",
        "coco_farther": "COCO/farther_images_wrt_bdd",
        "oi_farther": "OpenImages/ood_classes_rm_overlap/farther_images_wrt_bdd",
    },
    "voc": {
        "coco": "id_voc_ood_coco/val2017",
        "openimages": "OpenImages/ood_classes_rm_overlap/images",
        "coco_far": "COCO/far_images",
        "coco_near": "COCO/near_images",
        "oi_far": "OpenImages/id_voc_ood_openimages/ood_classes_rm_overlap/far_images",
        "oi_near": "OpenImages/id_voc_ood_openimages/ood_classes_rm_overlap/near_images"
    },
}


def ood_datasets_loaders(parent_directory: Path, cfg: DictConfig) -> Dict:
    assert cfg.ind_dataset in ("bdd", "voc")
    ood_datasets_loaders_dict = dict()
    for ood_dataset in cfg.ood_datasets:
        ood_datasets_loaders_dict[ood_dataset] = LoadImagesAndVideos(
            (parent_directory / ood_datapaths[cfg.ind_dataset][ood_dataset]).as_posix()
        )
    # Subset datasets to speed up dev and debugging
    if cfg.debug_mode:
        for ood_dataset in ood_datasets_loaders_dict.values():
            ood_dataset.files = ood_dataset.files[: cfg.debug_mode_dataset_length]
            ood_dataset.nf = cfg.debug_mode_dataset_length
            ood_dataset.ni = cfg.debug_mode_dataset_length

    return ood_datasets_loaders_dict


def ind_datasets_loaders(cfg, ind_dataset_dirpath: Path):
    # =========== InD BDD or VOC ===============
    assert cfg.ind_dataset in ("bdd", "voc")
    train_datapath = ind_dataset_dirpath.parent / "images/train"
    val_datapath = ind_dataset_dirpath.parent / "images/val"
    ind_dataset_dict = {"train": LoadImagesAndVideos(train_datapath), "valid": LoadImagesAndVideos(val_datapath)}
    # Reduce training and testing samples
    np.random.seed(cfg.random_seed)
    # Train set
    chosen_idx_train = np.random.choice(ind_dataset_dict["train"].nf, size=cfg.ind_train_samples, replace=False)
    ind_dataset_dict["train"].files = [ind_dataset_dict["train"].files[i] for i in chosen_idx_train]
    ind_dataset_dict["train"].nf = cfg.ind_train_samples
    ind_dataset_dict["train"].ni = cfg.ind_train_samples
    # Test set
    chosen_idx_valid = np.random.choice(ind_dataset_dict["valid"].nf, size=cfg.ind_valid_samples, replace=False)
    ind_dataset_dict["valid"].files = [ind_dataset_dict["valid"].files[i] for i in chosen_idx_valid]
    ind_dataset_dict["valid"].nf = cfg.ind_valid_samples
    ind_dataset_dict["valid"].ni = cfg.ind_valid_samples

    if cfg.debug_mode:
        # InD train
        ind_dataset_dict["train"].files = ind_dataset_dict["train"].files[: cfg.debug_mode_dataset_length]
        ind_dataset_dict["train"].nf = cfg.debug_mode_dataset_length
        ind_dataset_dict["train"].ni = cfg.debug_mode_dataset_length
        # InD valid
        ind_dataset_dict["valid"].files = ind_dataset_dict["valid"].files[: cfg.debug_mode_dataset_length]
        ind_dataset_dict["valid"].nf = cfg.debug_mode_dataset_length
        ind_dataset_dict["valid"].ni = cfg.debug_mode_dataset_length

    return ind_dataset_dict

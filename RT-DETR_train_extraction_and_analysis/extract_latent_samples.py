import os
import warnings
from typing import Dict, Union, List, Optional, Tuple
from torch.utils.data import DataLoader
import torch
import hydra
from omegaconf import DictConfig
from transformers import DetrImageProcessor, AutoProcessor
import numpy as np
from torchvision import transforms as transform_lib
from ood_detect_fns.uncertainty_estimation import BoxFeaturesExtractor, Hook


from fine_tune_bdd_voc import get_preprocessor
from utils import datasets_dict, CocoDetection, model_config_dict, CollatorDetrNoLabels, \
    BravoDataFolder, bravo_data_paths_dict
from models import models_dict

torch.set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy('file_system')


DOWNLOAD_MODEL = True
EXTRACT_IND = True
EXTRACT_OOD = True
SAVE_FC_PARAMS = True
VISUALIZE_N_IMAGES = 0
TENSOR_PRECISION = torch.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_file = "config_boxes.yaml"


@hydra.main(version_base=None, config_path="config", config_name=config_file)
def main(cfg: DictConfig) -> None:
    assert cfg.model_type in model_config_dict.keys()
    saved_images_dir = f"./inference_examples/{cfg.model_type}/{cfg.ind_dataset}"
    os.makedirs(saved_images_dir, exist_ok=True)
    print(
        f"Extracting from model {cfg.model_type} InD {cfg.ind_dataset}, OoD {cfg.ood_datasets}, "
        f"hooked {cfg.hooked_modules} output {cfg.hook_output} "
        f"{'roi_s ' + ''.join(map(str, cfg.roi_output_sizes)) + ' '}"
        f"{'roi_sr' + str(cfg.roi_sampling_ratio) + ' '}"
        f"Inf_conf: {cfg.inference_predict_confidence} "
    )
    processor = get_preprocessor(cfg.model_type, model_config_dict[cfg.model_type]["model_name"], DOWNLOAD_MODEL)
    # Path of pretrained model at t0
    # pretrained_model_path = f"./saved_models/{cfg.model_type}_model_{cfg.ind_dataset}_t0"
    pretrained_model_path = model_config_dict[cfg.model_type]["model_path"][cfg.ind_dataset]
    # Directory where samples will be saved
    if os.path.exists(cfg.latent_samples_folder):
        warnings.warn(f"Destination folder {cfg.latent_samples_folder} already exists!")
    os.makedirs(cfg.latent_samples_folder, exist_ok=True)
    # Get datasets
    ind_data_dict, text_queries = ind_datasets_loaders(cfg, processor)
    ood_data_dict = ood_datasets_loaders(cfg, processor, text_queries)

    # Load model
    model = models_dict[cfg.model_type].load_from_checkpoint(
        cfg.checkpoint,
        lr=1e-5,
        lr_backbone=1e-6,
        weight_decay=1e-4,
        n_labels=10 if cfg.ind_dataset == "bdd" else 20,
        train_loader=None,
        val_loader=None,
        download=DOWNLOAD_MODEL,
        pretrained_model_name=model_config_dict[cfg.model_type]["model_name"],
        pretrained_model_path=pretrained_model_path,
        tensor_precision=TENSOR_PRECISION,
        num_queries=100
    )
    model.to(device)
    # Possible layers to hook per architecture
    architecture = None
    if cfg.model_type == "DETR":
        # b0 means backbone module 0, etc.
        hooks_dict = {
            "b0": model.model.base_model.backbone.conv_encoder.model.encoder.stages[0],
            "b1": model.model.base_model.backbone.conv_encoder.model.encoder.stages[1],
            "b2": model.model.base_model.backbone.conv_encoder.model.encoder.stages[2],
            "b3": model.model.base_model.backbone.conv_encoder.model.encoder.stages[3],
            "bp": model.model.base_model.input_projection
        }
        architecture = "detr-backbone"
    elif cfg.model_type == "RTDETR":
        hooks_dict = {
            "b0": model.model.base_model.model.backbone.model.encoder.stages[0],
            "b1": model.model.base_model.model.backbone.model.encoder.stages[1],
            "b2": model.model.base_model.model.backbone.model.encoder.stages[2],
            "b3": model.model.base_model.model.backbone.model.encoder.stages[3],
            "e0": model.model.base_model.model.encoder.encoder[0],
        }
        if all(layer.startswith("b") for layer in cfg.hooked_modules):
            architecture = "rtdetr-backbone"
        elif all(layer.startswith("e") for layer in cfg.hooked_modules):
            architecture = "rtdetr-encoder"
        else:
            raise ValueError("Invalid hooked modules for RTDETR architecture!")
    # Hook layers
    hooked_layers = [Hook(hooks_dict[module]) for module in cfg.hooked_modules]

    # Instantiate samples extractor
    samples_extractor = BoxFeaturesExtractor(
        model=model,
        hooked_layers=hooked_layers,
        device=device,
        output_sizes=cfg.roi_output_sizes,
        sampling_ratio=cfg.roi_sampling_ratio,
        return_raw_predictions=cfg.return_raw_predictions,
        return_stds=cfg.return_variances,
        hook_layer_output=cfg.hook_output,
        # architecture="detr-backbone" if cfg.model_type=="DETR" else "owlv2"
        architecture=architecture
    )

    ##########################################################
    # Extract samples
    ##########################################################
    # =============== InD ========================
    if EXTRACT_IND:
        # ind_entropies = {}
        for split, data_loader in ind_data_dict.items():
            # Create identifiable file names
            file_name = (
                f"{cfg.latent_samples_folder}/ind_{cfg.ind_dataset}_{split}_"
                f"boxes_"
                f"{'stds_' if cfg.return_variances else ''}"
                f"hooked{'_'.join(cfg.hooked_modules)}_"
                f"output{cfg.hook_output}_"
                f"{'roi_s' + ''.join(map(str, cfg.roi_output_sizes)) + '_'}"
                f"{'roi_sr' + str(cfg.roi_sampling_ratio) + '_'}"
                f"{'trc_' + str(int(cfg.train_predict_confidence * 100)) if split == 'train' else ''}"
                f"{'infc_' + str(int(cfg.inference_predict_confidence * 100)) if split == 'valid' else ''}"
                f"_ls_samples.pt"
            )
            if not os.path.exists(file_name):
                # Extract and save latent space samples
                print(f"\nExtracting InD {cfg.ind_dataset} {split} boxes ls samples.")
                latent_activations = samples_extractor.get_ls_samples(
                    data_loader,
                    predict_conf=cfg.train_predict_confidence if split == "train" else cfg.inference_predict_confidence,
                )
                print(f"\nSaving InD {cfg.ind_dataset} {split} boxes ls samples in {file_name}")
                torch.save(latent_activations, file_name)
            else:
                print(f"{file_name} already exists!")
    del ind_data_dict
    if EXTRACT_OOD:
        # ================= OoD ====================
        # ood_entropies = {}
        for ood_dataset, data_loader in ood_data_dict.items():
            # Create identifiable file names
            file_name = (
                f"{cfg.latent_samples_folder}/ood_{ood_dataset}_"
                f"boxes_"
                f"{'stds_' if cfg.return_variances else ''}"
                f"hooked{'_'.join(cfg.hooked_modules)}_"
                f"output{cfg.hook_output}_"
                f"{'roi_s' + ''.join(map(str, cfg.roi_output_sizes)) + '_'}"
                f"{'roi_sr' + str(cfg.roi_sampling_ratio) + '_'}"
                f"{'infc_' + str(int(cfg.inference_predict_confidence * 100))}"
                f"_ls_samples.pt"
            )
            if not os.path.exists(file_name):
                # Extract and save latent space samples
                print(f"\nExtracting OoD {ood_dataset} boxes ls samples.")
                latent_activations = samples_extractor.get_ls_samples(
                    data_loader,
                    predict_conf=cfg.inference_predict_confidence
                )
                print(f"\nSaving OoD {ood_dataset} boxes ls samples in {file_name}")
                torch.save(latent_activations, file_name)
            else:
                print(f"{file_name} already exists!")

    if SAVE_FC_PARAMS:
        fc_params = model.model.class_embed[-1].state_dict()
        fc_params_file_name = cfg.checkpoint.split(".ckpt")[0].split("lightning_logs/")[1].replace("/", "_") + "_fc_params.pt"
        torch.save(fc_params, f"{cfg.latent_samples_folder}/{fc_params_file_name}")
        print(f"Saved fc_params in {cfg.latent_samples_folder}/{fc_params_file_name}")
    print("Done!")


def ind_datasets_loaders(
    cfg,
    processor
) -> Tuple[Dict[str, DataLoader], List[str]]:
    assert cfg.ind_dataset in ("bdd", "voc")
    train_dataset = CocoDetection(
        base_folder=datasets_dict[cfg.ind_dataset]["data_folder"],
        ann_file=datasets_dict[cfg.ind_dataset]["train_annotations"],
        images_path=datasets_dict[cfg.ind_dataset]["train_images"],
        id_type_int=datasets_dict[cfg.ind_dataset]["id_type_int"],
    )
    val_dataset = CocoDetection(
        base_folder=datasets_dict[cfg.ind_dataset]["data_folder"],
        ann_file=datasets_dict[cfg.ind_dataset]["val_annotations"],
        images_path=datasets_dict[cfg.ind_dataset]["val_images"],
        id_type_int=datasets_dict[cfg.ind_dataset]["id_type_int"],
    )
    text_queries = None
    # Get categories names in case we have the OWL model (vision-language model)
    if cfg.model_type == "OWL":
        text_queries = [value["name"] for value in train_dataset.coco.cats.values()]
        # label_names = [value["name"] for value in train_dataset.coco.cats.values()]
        # text_queries = [f"a photo of a {category}" for category in label_names]
    # Instantiate collator function class
    collator = Collator(model_type=cfg.model_type, processor=processor, texts=text_queries)
    # Reduce training and testing samples
    np.random.seed(cfg.random_seed)
    # Train set
    chosen_idx_train = np.random.choice(len(train_dataset), size=cfg.ind_train_samples, replace=False).astype(int)
    if cfg.debug_mode:
        chosen_idx_train = chosen_idx_train[:cfg.debug_mode_dataset_length]
    ind_train_subset = torch.utils.data.Subset(train_dataset, chosen_idx_train)
    # Valid set
    chosen_idx_valid = np.random.choice(len(val_dataset), size=cfg.ind_valid_samples, replace=False).astype(int)
    if cfg.debug_mode:
        chosen_idx_valid = chosen_idx_valid[:cfg.debug_mode_dataset_length]
    ind_val_subset = torch.utils.data.Subset(val_dataset, chosen_idx_valid)

    ind_dataset_dict = {
        "train": DataLoader(
            ind_train_subset,
            collate_fn=collator.collate_fn,
            batch_size=1,
            shuffle=True,
            num_workers=10,
        ),
        "valid": DataLoader(ind_val_subset, collate_fn=collator.collate_fn, batch_size=1, num_workers=10)
    }

    return ind_dataset_dict, text_queries

def ood_datasets_loaders(
    cfg: DictConfig,
    processor: Union[DetrImageProcessor, AutoProcessor],
    text_queries
) -> Dict[str, DataLoader]:
    ood_datasets_loaders_dict = dict()
    # Instantiate collator function class
    collator = Collator(model_type=cfg.model_type, processor=processor, texts=text_queries)
    # Reduce testing samples
    np.random.seed(cfg.random_seed)
    for ood_dataset_name in cfg.ood_datasets:
        ood_dataset = CocoDetection(
            base_folder=datasets_dict[ood_dataset_name]["data_folder"],
            ann_file=datasets_dict[ood_dataset_name]["val_annotations"][cfg.ind_dataset],
            images_path=datasets_dict[ood_dataset_name]["val_images"][cfg.ind_dataset],
            id_type_int=datasets_dict[ood_dataset_name]["id_type_int"],
        )
        if len(ood_dataset) > cfg.ind_valid_samples:
            chosen_idx_ood = np.random.choice(len(ood_dataset), size=cfg.ind_valid_samples, replace=False)
            ood_dataset = torch.utils.data.Subset(ood_dataset, chosen_idx_ood)
        if cfg.debug_mode:
            chosen_idx_ood = np.random.choice(len(ood_dataset), size=cfg.debug_mode_dataset_length, replace=False)
            ood_dataset = torch.utils.data.Subset(ood_dataset, chosen_idx_ood)
        ood_datasets_loaders_dict[ood_dataset_name] = DataLoader(
            ood_dataset,
            collate_fn=collator.collate_fn,
            batch_size=1,
            shuffle=False,
            num_workers=10
        )

    return ood_datasets_loaders_dict


class Collator:
    def __init__(
        self,
        model_type: str,
        processor: Union[DetrImageProcessor, AutoProcessor],
        texts: Optional[List[str]] = None,
    ):
        self.processor = processor
        self.texts = texts
        assert model_type in model_config_dict.keys()
        if model_type == "CLIP":
            self.collate_fn = self.collate_fn_clip
        else:
            self.collate_fn = self.collate_fn_detr

    def collate_fn_detr(self, batch):
        imgs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        if len(labels[0]["annotations"]) > 0:
            if "area" not in labels[0]["annotations"][0].keys():
                for ann in labels[0]["annotations"]:
                    # Dummy area just to comply with processor format
                    ann["area"] = 1
        encodings = self.processor(images=imgs, annotations=labels, return_tensors="pt")
        # encodings = self.processor(images=imgs, return_tensors="pt")
        pixel_values = encodings["pixel_values"] # remove batch dimension
        target = encodings["labels"] # remove batch dimension
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = target
        return batch

    def collate_fn_clip(self, batch):
        imgs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        encodings = self.processor(images=imgs, annotations=labels, return_tensors="pt")
        batch = {}
        batch['pixel_values'] = encodings['pixel_values']
        batch['img_ids'] =  [ann['image_id'] for ann in labels]

        return batch


if __name__ == '__main__':
    main()

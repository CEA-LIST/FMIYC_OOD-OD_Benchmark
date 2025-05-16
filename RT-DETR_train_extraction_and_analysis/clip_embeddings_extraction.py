import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from extract_latent_samples import ind_datasets_loaders, ood_datasets_loaders
from fine_tune_bdd_voc import get_preprocessor, DOWNLOAD_MODEL
from models import models_dict
from utils import model_config_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOWNLOAD_MODEL = False


@hydra.main(version_base=None, config_path="config", config_name="config_clip_embeddings.yaml")
def main(cfg: DictConfig):
    # processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
    # model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    assert cfg.model_type in model_config_dict.keys()
    saved_embeddings_dir = f"./extracted_latent_samples/{cfg.model_type}_embeddings"
    os.makedirs(saved_embeddings_dir, exist_ok=True)
    print(
        f"Extracting from model {cfg.model_type} InD {cfg.ind_dataset}, OoD {cfg.ood_datasets}, "
    )
    processor = get_preprocessor(cfg.model_type, model_config_dict[cfg.model_type]["model_name"], DOWNLOAD_MODEL)
    # Path of pretrained model at t0
    pretrained_model_path = model_config_dict[cfg.model_type]["model_path"][cfg.ind_dataset]

    # Load datasets
    ind_data_dict, text_queries = ind_datasets_loaders(cfg, processor)
    ood_data_dict = ood_datasets_loaders(cfg, processor, text_queries)
    del ind_data_dict["train"]
    model = models_dict[cfg.model_type](
        download=DOWNLOAD_MODEL,
        pretrained_model_name=model_config_dict[cfg.model_type]["model_name"],
        pretrained_model_path=pretrained_model_path
    )
    model.to(device)

    # ID data
    ind_embeddings = extract_embeddings(model, ind_data_dict["valid"], cfg.model_type, cfg.normalize)
    file_name = f"{saved_embeddings_dir}/ind_{cfg.ind_dataset}_val{'_norm' if cfg.normalize else ''}.npy"
    print(f"Saving ID {cfg.ind_dataset} CLIP embeddings in file {file_name}")
    np.save(file_name, ind_embeddings)

    # OOD datasets
    for ood_dataset_name in ood_data_dict:
        ood_embeddings = extract_embeddings(model, ood_data_dict[ood_dataset_name], cfg.model_type, cfg.normalize)
        file_name = f"{saved_embeddings_dir}/ood_{ood_dataset_name}_ind_{cfg.ind_dataset}{'_norm' if cfg.normalize else ''}.npy"
        print(f"Saving OOD {ood_dataset_name} {cfg.model_type} embeddings in file {file_name}")
        np.save(file_name, ood_embeddings)


def extract_embeddings(model, data_loader, model_type, normalize):
    model.eval()
    dataset_embeddings = []
    img_ids = []
    for image in tqdm(data_loader):
        outputs = model(pixel_values=image["pixel_values"])
        # outputs = model(
        #     pixel_values=image["pixel_values"].to(device),
        #     attention_mask=image["attention_mask"].to(device),
        #     input_ids=image["input_ids"].to(device)
        # )
        if model_type == "OWL":
            dataset_embeddings.append(outputs.vision_model_output.pooler_output.cpu().detach().numpy())
        elif model_type == "CLIP":
            dataset_embeddings.append(outputs.pooler_output.cpu().detach().numpy())
        else:
            raise NotImplementedError
        img_ids.append(image["img_ids"][0])
    dataset_embeddings = np.concatenate(dataset_embeddings, axis=0)
    if normalize:
        dataset_embeddings /= np.linalg.norm(dataset_embeddings, axis=1, keepdims=True)
    embeddings = {
        "embeddings": dataset_embeddings,
        "img_ids": img_ids,
    }
    return embeddings


if __name__ == '__main__':
    main()

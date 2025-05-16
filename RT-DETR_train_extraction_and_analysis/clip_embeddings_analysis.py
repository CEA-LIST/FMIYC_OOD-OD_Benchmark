import json
from collections import defaultdict
from typing import List, Dict

import numpy as np
import torch
import hydra
from PIL import Image
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from scipy.stats import wasserstein_distance_nd
from sympy.printing.pretty.pretty_symbology import line_width

from utils import model_config_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOWNLOAD_MODEL = True
N_SAMPLES = 2000
DIST_MEASURES = ["Cosine"]
IND_DATASETS = ["voc", "bdd"]
NBINS = 200
plt.rc('axes', axisbelow=True)


@hydra.main(version_base=None, config_path="config", config_name="config_clip_embeddings.yaml")
def main(cfg: DictConfig):
    # processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
    # model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    assert cfg.model_type in model_config_dict.keys()
    saved_embeddings_dir = f"./extracted_latent_samples/{cfg.model_type}_embeddings"

    print(
        f"Analyzing from model {cfg.model_type} InD {cfg.ind_dataset}, OoD {cfg.ood_datasets}, "
    )
    np.random.seed(cfg.random_seed)
    # Load ID data
    ind_embeddings = {}
    ood_embeddings = {}
    for ind_dataset_name in IND_DATASETS:
        file_name = f"{saved_embeddings_dir}/ind_{ind_dataset_name}_val{'_norm' if cfg.normalize else ''}.npy"
        dict_content = np.load(file_name, allow_pickle=True)
        try:
            ind_embeddings[ind_dataset_name] = {
                "embeddings": dict_content.item().get("embeddings"),
                "img_ids": dict_content.item().get("img_ids"),
            }
        except ValueError:
            raise ValueError(f"Error loading {file_name}")
        ind_embeddings[ind_dataset_name] = subset_dataset(ind_embeddings[ind_dataset_name], N_SAMPLES)

        # Load OOD datasets
        ood_embeddings[ind_dataset_name] = {}
        for ood_dataset_name in cfg.ood_datasets[ind_dataset_name]:
            file_name = f"{saved_embeddings_dir}/ood_{ood_dataset_name}_ind_{ind_dataset_name}{'_norm' if cfg.normalize else ''}.npy"
            dict_content = np.load(file_name, allow_pickle=True)
            try:
                ood_embeddings[ind_dataset_name][ood_dataset_name] = {
                    "embeddings": dict_content.item().get("embeddings"),
                    "img_ids": dict_content.item().get("img_ids"),
                }
            except ValueError:
                ood_embeddings[ind_dataset_name][ood_dataset_name] = {
                    "embeddings": dict_content,
                    "img_ids": [],
                }
            ood_embeddings[ind_dataset_name][ood_dataset_name] = subset_dataset(ood_embeddings[ind_dataset_name][ood_dataset_name], N_SAMPLES)

    distances = {}
    # Wasserstein distance
    if "Wasserstein" in DIST_MEASURES:
        for ood_dataset_name in cfg.ood_datasets:
            distances[f"Wasserstein {ood_dataset_name}"] = wasserstein_distance_nd(ind_embeddings, ood_embeddings[ood_dataset_name])
            print(f"Wass distance ID {cfg.ind_dataset} OOD {ood_dataset_name}: {distances[f'Wasserstein {ood_dataset_name}']}")

    if "Cosine" in DIST_MEASURES:
        for ind_dataset_name in IND_DATASETS:
            for ood_dataset_name in cfg.ood_datasets[ind_dataset_name]:
                distances[f"Cosine {ind_dataset_name} {ood_dataset_name}"] = cosine_dist(
                    dataset_a=ind_embeddings[ind_dataset_name]["embeddings"],
                    dataset_b=ood_embeddings[ind_dataset_name][ood_dataset_name]["embeddings"]
                )

        # for ood_dataset_name in cfg.ood_datasets:
        #     visualize_distribution(
        #         distances[f"Cosine {IND_DATASETS[0]} {ood_dataset_name}"],
        #         distances[f"Cosine {IND_DATASETS[1]} {ood_dataset_name}"],
        #         dataset_name_dict[IND_DATASETS[0]],
        #         dataset_name_dict[IND_DATASETS[1]],
        #         dataset_name_dict[ood_dataset_name],
        #     )
        if len(IND_DATASETS) == 2:
            # visualize_two_id_three_ood_cos_sim(distances_dict=distances)
            visualize_two_id_cos_sim(distances, cfg.ood_datasets["bdd"])
        elif len(IND_DATASETS) == 1 and IND_DATASETS[0] == "voc":
            if "coco_near" in cfg.ood_datasets:
                visualize_new_voc_benchmark(
                    cos_similarities_dict=distances,
                    id_dataset_name=IND_DATASETS[0]
                )
            if "coco_all_new" in cfg.ood_datasets:
                visualize_single_new_voc_benchmark(
                    cos_similarities_dict=distances,
                    id_dataset_name=IND_DATASETS[0],
                    ood_datasets_names=cfg.ood_datasets
                )
        else:
            raise NotImplementedError
    print("Done!")

def get_outlier_im_ids_from_similarities(similarities: np.ndarray, img_ids: List, max_mode: bool, n_candidates: int):
    sorted_indexes = np.argsort(similarities)
    if max_mode:
        sorted_indexes = sorted_indexes[-n_candidates:]
    else:
        sorted_indexes = sorted_indexes[:n_candidates]
    return list(np.array(img_ids)[sorted_indexes])


def cosine_dist(dataset_a, dataset_b):
    sims_chunk = np.matmul(dataset_a, dataset_b.T)
    sims_max = np.max(sims_chunk, axis=0)
    return sims_max


def subset_dataset(dataset: Dict, n_samples: int) -> Dict:
    if len(dataset["embeddings"]) > n_samples:
        all_inds = np.arange(len(dataset))
        np.random.shuffle(all_inds)
        dataset["embeddings"] = dataset["embeddings"][all_inds[:n_samples]]
        dataset["img_ids"] = dataset["img_ids"][all_inds[:n_samples]]
    return dataset

def visualize_distribution(dataset_a, dataset_b, ind_dataset_name_a, ind_dataset_name_b, ood_dataset_name):
    plt.grid(True)
    plt.hist(dataset_a, bins=NBINS, label=ind_dataset_name_a, alpha=0.7)
    plt.hist(dataset_b, bins=NBINS, label=ind_dataset_name_b, alpha=0.7)
    plt.title(f"CLIP embeddings cosine similarity wrt {ood_dataset_name}")
    plt.legend()
    plt.xlabel("Cosine similarity")
    plt.ylabel("Frequency")
    plt.xlim((0.1, 0.81))
    plt.ylim((0.0, 47.0))
    plt.tight_layout()
    plt.show()


def visualize_single_new_voc_benchmark(cos_similarities_dict, id_dataset_name: str, ood_datasets_names: List[str]):
    colors = ["tab:blue", "tab:orange"]
    fig, axs = plt.subplots(1, 1, sharey=True, figsize=(6, 4))
    axs.grid(True)
    for i, ood_ds_name in enumerate(ood_datasets_names):
        axs.hist(cos_similarities_dict[f"Cosine {id_dataset_name} {ood_ds_name}"], bins=NBINS,
                    label=ood_ds_name, alpha=0.7)
        mean = cos_similarities_dict[f"Cosine {id_dataset_name} {ood_ds_name}"].mean()
        axs.axvline(mean, ymin=0, ymax=1, color=colors[i], label=f"Mean: {mean:.3f}", linewidth=3)


    axs.set_xlabel(f"Cosine similarity to Single new VOC benchmark")
    axs.set_ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_new_voc_benchmark(cos_similarities_dict, id_dataset_name: str):
    color_1 = "tab:blue"
    color_2 = "tab:orange"
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
    axs[0].grid(True)
    axs[0].hist(cos_similarities_dict[f"Cosine {id_dataset_name} openimages_new"], bins=NBINS,
                label="New", alpha=0.7, color=color_1)
    oi_new_mean = cos_similarities_dict[f"Cosine {id_dataset_name} openimages_new"].mean()
    axs[0].axvline(oi_new_mean, ymin=0, ymax=1, color=color_1, label=f"Mean new: {oi_new_mean:.3f}", linewidth=3)
    axs[0].hist(cos_similarities_dict[f"Cosine {id_dataset_name} openimages_near"], bins=NBINS,
                label="Near", alpha=0.7, color=color_2)
    oi_near_mean = cos_similarities_dict[f"Cosine {id_dataset_name} openimages_near"].mean()
    axs[0].axvline(oi_near_mean, ymin=0, ymax=1, color=color_2, label=f"Mean near: {oi_near_mean:.3f}", linewidth=3)

    axs[0].set_xlabel(f"Cosine similarity to OpenImages")
    axs[0].set_ylabel("Frequency")
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(
        [handles[i] for i in (1,3)],
        [labels[i] for i in (1,3)], frameon=True, loc=(0.01, 0.8))
    # axs[0].set_xlim((0.1, 0.81))
    # axs[0].set_ylim((0.0, 47.0))

    axs[1].grid(True)
    axs[1].hist(cos_similarities_dict[f"Cosine {id_dataset_name} coco_new"], bins=NBINS,
                label="New", alpha=0.7)
    coco_new_mean = cos_similarities_dict[f"Cosine {id_dataset_name} coco_new"].mean()
    axs[1].axvline(coco_new_mean, ymin=0, ymax=1, color=color_1, label=f"Mean new: {coco_new_mean:.3f}", linewidth=3)
    axs[1].hist(cos_similarities_dict[f"Cosine {id_dataset_name} coco_near"], bins=NBINS,
                label="Near", alpha=0.7)
    coco_near_mean = cos_similarities_dict[f"Cosine {id_dataset_name} coco_near"].mean()
    axs[1].axvline(coco_near_mean, ymin=0, ymax=1, color=color_2, label=f"Mean new: {coco_near_mean:.3f}", linewidth=3)
    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(
        [handles[i] for i in (1, 3)],
        [labels[i] for i in (1, 3)], frameon=True, loc=(0.01, 0.8))

    # axs[1].legend()
    axs[1].set_xlabel(f"Cosine similarity to MS-COCO")
    # axs[1].set_xlim((0.1, 0.81))
    # axs[1].set_ylim((0.0, 47.0))
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend([handles[i] for i in (0,2)], [labels[i] for i in (0,2)], frameon=True, loc=(0.4, 0.8))
    plt.tight_layout()
    plt.show()

def visualize_two_id_cos_sim(distances_dict, ood_ds_names):
    id_ds_name_a = IND_DATASETS[0]
    id_ds_name_b = IND_DATASETS[1]
    ood_ds_name_a = ood_ds_names[0]
    ood_ds_name_b = ood_ds_names[1]

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
    axs[0].grid(True)
    axs[0].hist(distances_dict[f"Cosine {id_ds_name_a} {ood_ds_name_a}"], bins=NBINS, label=dataset_name_dict[id_ds_name_a], alpha=0.7)
    axs[0].hist(distances_dict[f"Cosine {id_ds_name_b} {ood_ds_name_a}"], bins=NBINS, label=dataset_name_dict[id_ds_name_b], alpha=0.7)

    axs[0].set_xlabel(f"Cosine similarity to {dataset_name_dict[ood_ds_name_a]}")
    axs[0].set_ylabel("Frequency")
    axs[0].set_xlim((0.1, 0.81))
    axs[0].set_ylim((0.0, 47.0))

    axs[1].grid(True)
    axs[1].hist(distances_dict[f"Cosine {id_ds_name_a} {ood_ds_name_b}"], bins=NBINS,
                label=dataset_name_dict[id_ds_name_a], alpha=0.7)
    axs[1].hist(distances_dict[f"Cosine {id_ds_name_b} {ood_ds_name_b}"], bins=NBINS,
                label=dataset_name_dict[id_ds_name_b], alpha=0.7)

    # axs[1].legend()
    axs[1].set_xlabel(f"Cosine similarity to {dataset_name_dict[ood_ds_name_b]}")
    axs[1].set_xlim((0.1, 0.81))
    axs[1].set_ylim((0.0, 47.0))

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=True, loc=(0.1, 0.8))
    plt.tight_layout()
    plt.show()


def visualize_two_id_three_ood_cos_sim(distances_dict):
    id_ds_name_a = "bdd"
    id_ds_name_b = "voc"
    # ood_ds_name_a = ood_ds_names[0]
    # ood_ds_name_b = ood_ds_names[1]

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
    axs[0].grid(True)
    axs[0].hist(distances_dict[f"Cosine {id_ds_name_a} openimages_new"], bins=NBINS, label=dataset_name_dict_three_plot[id_ds_name_a], alpha=0.7)
    axs[0].hist(distances_dict[f"Cosine {id_ds_name_b} openimages_new"], bins=NBINS, label=dataset_name_dict_three_plot["openimages_new"][id_ds_name_b], alpha=0.7)
    axs[0].hist(distances_dict[f"Cosine {id_ds_name_b} openimages_near"], bins=NBINS, label=dataset_name_dict_three_plot["openimages_near"], alpha=0.7)

    axs[0].set_xlabel(f"Cosine similarity to OpenImages")
    axs[0].set_ylabel("Frequency")
    axs[0].set_xlim((0.1, 0.81))
    axs[0].set_ylim((0.0, 47.0))

    axs[1].grid(True)

    axs[1].hist(distances_dict[f"Cosine {id_ds_name_a} coco_new"], bins=NBINS,
                label=dataset_name_dict_three_plot[id_ds_name_a], alpha=0.7)
    axs[1].hist(distances_dict[f"Cosine {id_ds_name_b} coco_new"], bins=NBINS,
                label=dataset_name_dict_three_plot["coco_new"][id_ds_name_b], alpha=0.7)
    axs[1].hist(distances_dict[f"Cosine {id_ds_name_b} coco_near"], bins=NBINS,
                label=dataset_name_dict_three_plot["coco_near"], alpha=0.7)
    # axs[1].legend()
    axs[1].set_xlabel(f"Cosine similarity to COCO")
    axs[1].set_xlim((0.1, 0.81))
    axs[1].set_ylim((0.0, 47.0))

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=True, loc=(0.77, 0.77))

    plt.tight_layout()
    plt.show()


def visualize_imgs_with_annotations(coco_dataset, images_path, images_ids_list, dataset_name, images_mode):
    color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "black", "magenta", "green", "aqua"] * 61
    # Divide by groups of four
    if len(images_ids_list) % 4 != 0:
        n_groups = (len(images_ids_list) // 4) + 1
    else:
        n_groups = len(images_ids_list) // 4
    for group in range(n_groups):
        images_group = images_ids_list[4 * group: 4 * (group + 1)]

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        ax = ax.ravel()
        for i, im_id in enumerate(images_group):
            image = Image.open(f"{images_path}/{coco_dataset.im_dict[im_id]['file_name']}")
            ann_ids = coco_dataset.get_annIds(im_id)
            annotations = coco_dataset.load_anns(ann_ids)
            for ann in annotations:
                bbox = ann['bbox']
                x, y, w, h = [int(b) for b in bbox]
                class_id = ann["category_id"]
                class_name = coco_dataset.load_cats(class_id)[0]["name"]
                # license = coco_dataset.get_imgLicenses(im)[0]["name"]
                color_ = color_list[class_id]
                rect = plt.Rectangle((x, y), w, h, linewidth=3, edgecolor=color_, facecolor='none')
                t_box = ax[i].text(x, y, class_name, color='red', fontsize=12)
                t_box.set_bbox(dict(boxstyle='square, pad=0', facecolor='white', alpha=0.8, edgecolor='blue'))
                ax[i].add_patch(rect)

            ax[i].axis('off')
            ax[i].imshow(image)
            ax[i].set_xlabel('Longitude')
            ax[i].set_title(f"N {4*group + i} Im ID: {im_id} ")
        plt.suptitle(f"{images_mode} {dataset_name} group {group + 1}/{n_groups}")
        plt.tight_layout()
        plt.show()


dataset_name_dict = {
    "bdd": "BDD100k",
    "voc": "Pascal-VOC",
    "coco": "MS-COCO",
    "openimages": "OpenImages",
    "openimages_new": "OpenImages New",
    "coco_new": "COCO New",
    "openimages_near": "OpenImages Near",
    "coco_near": "COCO Near",
}

dataset_name_dict_three_plot = {
    "bdd": "BDD: Farther",
    "voc": "Pascal-VOC",
    "coco": "MS-COCO",
    "openimages": "OpenImages",
    "openimages_new":
        {
            "bdd": "OI w/bdd",
            "voc": "VOC: Far",
        },
    "coco_new":
        {
            "bdd": "COCO w/bdd",
            "voc": "VOC: Far",
        },
    "openimages_near": "VOC: Near",
    "coco_near": "VOC: Near",
}

outliers_max_mode_dict = {
    "coco_new": True,
    "coco_near": False,
    "openimages_new": True,
    "openimages_near": False
}


class COCOParser:
    def __init__(self, anns_file):
        with open(anns_file, 'r') as f:
            coco = json.load(f)

        self.annIm_dict = defaultdict(list)
        # Dict of id: category pairs
        self.cat_dict = {}
        # Dict of original categories (copy of original entry)
        self.categories_original = {'categories': coco['categories']}
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {'licenses': coco['licenses']}
        self.info_dict = {'info': coco['info']}
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat
            self.cat_dict[cat['id']]["count"] = 0
        for ann in coco['annotations']:
            self.annIm_dict[ann['image_id']].append(ann)
            self.annId_dict[ann['id']] = ann
            self.cat_dict[ann["category_id"]]["count"] += 1
        for img in coco['images']:
            # Remove leading zeros from filenames
            # try:
            #     img['file_name'] = str(int(img['file_name'].split('.')[0])) + '.jpg'
            # except ValueError:
            #     pass
            self.im_dict[img['id']] = img

        # Licenses not actually needed per image
        # for license in coco['licenses']:
        #     self.licenses_dict[license['id']] = license

    def get_imgIds(self):
        return list(self.im_dict.keys())

    def get_annIds(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]

    def load_anns(self, ann_ids):
        im_ids = ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]

    def load_cats(self, class_ids):
        class_ids = class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]

    def get_imgLicenses(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]

    def get_img_info(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [self.im_dict[im_id] for im_id in im_ids]

    def get_img_ids_per_cat_name(self, cat_name):
        cat_id = [cat["id"] for cat in self.cat_dict.values() if cat['name'] == cat_name][0]
        return list(set([ann['image_id'] for ann in self.annId_dict.values() if ann['category_id'] == cat_id]))

if __name__ == '__main__':
    main()

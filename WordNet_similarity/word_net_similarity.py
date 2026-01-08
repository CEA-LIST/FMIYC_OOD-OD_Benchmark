import itertools
from math import ceil

import hydra
import numpy as np
from omegaconf import DictConfig
from runia.evaluation import COCOParser
from nltk.corpus import wordnet
import matplotlib.pyplot as plt



@hydra.main(version_base=None, config_path="", config_name="wordnet_config.yaml")
def main(cfg: DictConfig) -> None:
    #############################################################################
    # Read current benchmark annotations
    #############################################################################

    ind_annotations = {}
    ood_annotations = {}
    for ind_dataset_name in cfg.ind_annotations_path.keys():
        ind_annotations[ind_dataset_name] = COCOParser(cfg.ind_annotations_path[ind_dataset_name])
        ood_annotations[ind_dataset_name] = {}
        for ood_dataset_name in cfg.ood_annotations_paths[ind_dataset_name].keys():
            ood_annotations[ind_dataset_name][ood_dataset_name] = COCOParser(cfg.ood_annotations_paths[ind_dataset_name][ood_dataset_name])

    ind_categories = {}
    ood_categories_count = {}
    ood_top_categories = {}
    for ind_dataset_name in cfg.ind_annotations_path.keys():
        ind_categories[ind_dataset_name] = {
            "cat_names": [cat["name"] for cat in ind_annotations[ind_dataset_name].cat_dict.values()]
        }
        # Here we split the compound names into separate words, i.e. "potted plant" -> "potted", "plant"
        ind_categories[ind_dataset_name]["cat_names"] = list(
            itertools.chain(
                *[
                    cat.split(" ") if len(cat.split(" ")) > 1 else [cat] for cat in ind_categories[ind_dataset_name]["cat_names"]
                ]
            )
        )
        ind_categories[ind_dataset_name]["wordnet"] = [wordnet.synsets(cat)[0] for cat in ind_categories[ind_dataset_name]["cat_names"]]
        for ood_dataset_name, ood_dataset_annotations in ood_annotations[ind_dataset_name].items():
            ood_categories_count[ood_dataset_name] = {}
            ood_categories_count[ood_dataset_name]["non_zero"], ood_categories_count[ood_dataset_name][
                "zero_count"] = make_barplot_categories(
                ood_dataset_annotations,
                dataset_name=dataset_name_equiv[ind_dataset_name][ood_dataset_name],
                top_k=20,
                ascending=True,
                # save=f"./histogram_new_bm/{dataset_name_equiv[ind_dataset_name][ood_dataset_name]}_categories_count.pdf"
                save=None
            )
            ood_top_categories[ood_dataset_name] = {
                "cat_names": [cat for idx, cat in enumerate(ood_categories_count[ood_dataset_name]["non_zero"].keys()) if idx < cfg.top_k_categories]
            }
            ood_top_categories[ood_dataset_name]["cat_names"] = list(
                itertools.chain(
                    *[
                        cat.split(" ") if len(cat.split(" ")) > 1 else [cat] for cat in
                        ood_top_categories[ood_dataset_name]["cat_names"]
                    ]
                )
            )

            ood_top_categories[ood_dataset_name]["wordnet"] = [wordnet.synsets(cat)[0] for cat in
                                                           ood_top_categories[ood_dataset_name]["cat_names"]]
            wup_similarities = np.zeros((len(ind_categories[ind_dataset_name]["wordnet"]), len(ood_top_categories[ood_dataset_name]["wordnet"])))
            path_similarities = np.zeros((len(ind_categories[ind_dataset_name]["wordnet"]), len(ood_top_categories[ood_dataset_name]["wordnet"])))
            for i, ind_cat in enumerate(ind_categories[ind_dataset_name]["wordnet"]):
                for j, ood_cat in enumerate(ood_top_categories[ood_dataset_name]["wordnet"]):
                    wup_similarities[i, j] = ind_cat.wup_similarity(ood_cat)
                    path_similarities[i, j] = ind_cat.path_similarity(ood_cat)
            ood_top_categories[ood_dataset_name]["similarities"] = {
                "WuP": wup_similarities.max(axis=1),
                "Path": path_similarities.max(axis=1),
            }
    for sim_name in ["WuP", "Path",]:
        make_barplots_similarities(ood_top_categories, sim_name )
    print("Done")


dataset_name_equiv = {
    "voc": {
        "coco_near": "COCO-Near",
        "coco_far": "COCO-Far",
        "openimages_far": "OpenImages-Far",
        "openimages_near": "OpenImages-Near"
    },
    "bdd": {
        "coco_farther": "COCO-Farther",
        "openimages_farther": "OpenImages-Farther",
    }
}


def make_barplots_similarities(ood_similarities, similarity_type):
    y_pos = np.arange(len(ood_similarities))
    means = []
    stds = []
    for ood_dataset_name in fmiyc_datasets.keys():
        print(f"Mean {similarity_type} similarity {ood_dataset_name}: {round(np.mean(ood_similarities[ood_dataset_name]['similarities'][similarity_type]), 3)} "
                f"Std: {round(np.std(ood_similarities[ood_dataset_name]['similarities'][similarity_type]), 3)}")
        means.append(np.mean(ood_similarities[ood_dataset_name]["similarities"][similarity_type]))
        stds.append(np.std(ood_similarities[ood_dataset_name]["similarities"][similarity_type]))
    plt.rcParams['font.size'] = '16'
    plt.figure(figsize=(12, max(ceil(len(y_pos) * 0.352), 6)))
    plt.barh(y_pos, means, xerr=stds, align='center', alpha=0.8 )
    plt.yticks(y_pos, list(fmiyc_datasets.values()) )
    plt.title(f"WordNet {similarity_type} mean similarity to VOC and BDD categories")
    plt.xlabel(f"WordNet {similarity_type} Similarity")
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(f"wordnet_smilarity_{similarity_type}.pdf")
    plt.show()


def make_barplot_categories(dataset_annotations, dataset_name, ascending: bool, show_plot:bool = False, top_k=10, save=None, ):
    non_zero_categories = {}
    zero_freq_categories = []
    for cat in dataset_annotations.cat_dict.values():
        if cat["count"] == 0:
            zero_freq_categories.append(cat)
        else:
            non_zero_categories[cat["name"]] = cat["count"]
    # Sort by count
    non_zero_categories = dict(sorted(non_zero_categories.items(), key=lambda item: item[1], reverse=True if ascending else False))

    if show_plot:
        y_pos = np.arange(min(len(non_zero_categories), top_k))
        plt.rcParams['font.size'] = '16'
        # plt.figure(figsize=(6, max(ceil(len(y_pos)*0.176), 3)))
        plt.figure(figsize=(12, max(ceil(len(y_pos)*0.352), 6)))
        plt.barh(y_pos, list(non_zero_categories.values())[-top_k:],)
        plt.yticks(y_pos, list(non_zero_categories.keys())[-top_k:])
        plt.title(f"Top {len(y_pos)} category count for {dataset_name}")
        plt.xlabel("Count")
        plt.grid(axis='x')
        plt.tight_layout()
        if save:
            plt.savefig(save)
        plt.show()
    return non_zero_categories, zero_freq_categories


fmiyc_datasets = {
    "openimages_near": "OI Near",
    "openimages_far": "OI Far",
    "openimages_farther": "OI Farther",
    "coco_near": "COCO Near",
    "coco_far": "COCO Far",
    "coco_farther": "COCO Farther"
}

if __name__ == '__main__':
    main()

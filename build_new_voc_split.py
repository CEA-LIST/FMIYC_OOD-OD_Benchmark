import json
import copy
import os
import shutil
import hydra
from omegaconf import DictConfig
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from OI_downloader import download_all_images
from edit_COCO_json import COCOParser
import csv


@hydra.main(version_base=None, config_path="configs/New_benchmark", config_name="config_far_near.yaml")
def main(cfg: DictConfig) -> None:
    #############################################################################
    # Read current benchmark annotations
    #############################################################################
    ALL_OOD_DATASETS = cfg.OOD_DATASETS + cfg.NEW_OOD_DATASETS
    annotations = {}
    for dataset_name in cfg.CURRENT_BENCHMARK_ANNOTATIONS:
        annotations[dataset_name] = COCOParser(cfg.CURRENT_BENCHMARK_ANNOTATIONS[dataset_name])
    # COCO val annotations
    coco_train_annotations = COCOParser(cfg.COCO_PATHS["train_annotations"])
    coco_im_ids_benchmark = annotations["coco"].get_imgIds()
    # Read bboxes annotations for OpenImages
    # Since the current benchmark does not include OpenImages annotations, we need to load them and attach them
    open_images_cat_ids = read_csv_file(cfg.OPENIMAGES_PATHS["cat_ids"], fieldnames=["cat_id", "cat_name"])
    open_images_cat_ids = convert_cat_ids_oi(open_images_cat_ids)
    # Read Hierarchy JSON
    with open(cfg.OPENIMAGES_PATHS["hierarchy"], 'r', encoding='utf-8') as file:
        oi_hierarchy = json.load(file)
    oi_hierarchy_c = convert_oi_hierarchy(oi_hierarchy["Subcategory"], open_images_cat_ids)
    # Get a unique entry with super and meta category from the hierarchy
    oi_hierarchy_u = unfold_oi_categories(oi_hierarchy_c)
    # Get a catId dictionary
    oi_cat_dict = {}
    for cat in oi_hierarchy_u.values():
        oi_cat_dict[cat["id"]] = {
            "name": cat["name"],
            "supercategory": cat["supercategory"],
            "metacategory": cat["metacategory"],
            "count": 0
        }
    for idx, cat in oi_cat_dict.items():
        cat["id"] = idx
    # Modify the category dict for openimages, since currently it's the one from coco
    annotations["openimages"].cat_dict = oi_cat_dict
    # Read train set OpenImages bb annotations
    open_images_bboxes = read_csv_file(cfg.OPENIMAGES_PATHS["bboxes_ann"])
    # Get all train image ids
    oi_train_image_ids = {ann["ImageID"] for ann in open_images_bboxes.values()}
    # Get intersection with subset of OI
    oi_im_ids_benchmark = list(oi_train_image_ids.intersection(list(annotations["openimages"].im_dict.keys())))
    # Get the subset annotations
    open_images_bboxes_filtered = filter_open_im_bboxes(open_images_bboxes, oi_im_ids_benchmark)
    # Add image ids to OI annotations dict
    for im_id in oi_im_ids_benchmark:
        annotations["openimages"].annIm_dict[im_id] = []
    # Add boxes to OI annotations
    annotations["openimages"] = add_bboxes_to_annotations(
        annotations=annotations["openimages"],
        open_images_bboxes=open_images_bboxes_filtered,
        oi_hierarchy=oi_hierarchy_u
    )
    # Count current categories
    categories_count = {}
    for dataset_name, dataset_annotations in annotations.items():
        categories_count[dataset_name] = {}
        (categories_count[dataset_name]["non_zero"],
         categories_count[dataset_name]["zero_count"]) = make_barplot_categories(
            dataset_annotations=dataset_annotations,
            dataset_name=dataset_name
        )

    ###############################################################################
    # Automatic stage where overlapping categories and near categories are removed or moved to near ood
    ###############################################################################
    images_to_remove, images_to_near_ood = get_images_to_near_and_remove(
        annotations,
        ood_dataset_names=cfg.OOD_DATASETS,
        categories_to_remove=cfg.categories.CATS_TO_REMOVE,
        categories_to_near_ood=cfg.categories.CATS_TO_NEAR_OOD,
        ids_to_remove=cfg.manual_selection.REMOVE_IDS,
        ids_to_near=cfg.manual_selection.KEEP_TO_NEAR_IDS,
    )
    # Visualize images to remove by dataset and category
    visualize_removed_or_near_images(
        coco_dataset=annotations["openimages"],
        images_path=cfg.OPENIMAGES_PATHS["images_path"],
        num_imgs_to_disp=4,
        dataset_name="openimages",
        images_ids_list=images_to_remove["openimages"]["Person"],
        category_name="Person",
    )
    # Modify the benchmark
    annotations = split_old_benchmark(
        annotations=annotations,
        images_to_remove=images_to_remove,
        images_to_near_ood=images_to_near_ood,
        new_splits_annotations_paths=cfg.NEW_SPLITS_ANNOTATIONS_PATHS,
        ood_dataset_names=cfg.OOD_DATASETS,
    )
    #####################################################################################
    # First manual stage where remaining images from previous benchmark are manually checked
    #####################################################################################
    # Some images that should be eliminated might go to near, and some images that should go to near get eliminated
    # Remove overlap images and modify according to manually curated image ids
    all_images_to_remove = {}
    all_images_to_near = {}
    all_images_to_far = {}
    for ood_dataset_name in cfg.OOD_DATASETS:
        all_images_to_remove[ood_dataset_name] = [
            im_id for cat in cfg.manual_selection.UNFILTERED_IMAGES_TO_REMOVE[ood_dataset_name].values() for im_id in cat
        ]
        all_images_to_near[ood_dataset_name] = [
            im_id for cat in cfg.manual_selection.UNFILTERED_IMAGES_TO_NEAR[ood_dataset_name].values() for im_id in cat
        ]
        all_images_to_far[ood_dataset_name] = [
            im_id for cat in cfg.manual_selection.UNFILTERED_IMAGES_TO_FAR[ood_dataset_name].values() for im_id in cat
        ]
        annotations = remove_and_split_manual_inspection(
            annotations=annotations,
            images_to_remove=all_images_to_remove[ood_dataset_name],
            images_to_near_ood=all_images_to_near[ood_dataset_name],
            images_to_far_ood=all_images_to_far[ood_dataset_name],
            base_dataset_name=ood_dataset_name,
        )
    # Until here the new splits come from coco val and Openimages train, from pre-existing folders
    # The new added images will come from coco train and Openimages images need to be downloaded
    # Copy retained images into new folders
    for ood_dataset_name in ALL_OOD_DATASETS:
        original_images_path = cfg.COCO_PATHS["val_images"] if "coco" in ood_dataset_name else cfg.OPENIMAGES_PATHS[
            "images_path"]
        copy_images(
            dataset=annotations[ood_dataset_name],
            original_path=original_images_path,
            target_path=cfg.NEW_SPLITS_IMAGES_PATHS[ood_dataset_name],
        )

    # Redo categories count after new benchmark construction
    new_categories_count = {}
    for dataset_name, dataset_annotations in annotations.items():
        new_categories_count[dataset_name] = {}
        new_categories_count[dataset_name]["non_zero"], new_categories_count[dataset_name][
            "zero_count"] = make_barplot_categories(dataset_annotations, dataset_name)

    ##################################################################################
    # Second manual stage where first we get img ids candidates with non_overlapping or near categories
    ##################################################################################
    # Get the categories that have low counts in already existing ood datasets
    candidate_categories_new = {}
    for ood_dataset_name in ["openimages", "openimages_near", "coco"]:
        candidate_categories_new[ood_dataset_name] = [cat_name for cat_name, count in
                                                      new_categories_count[ood_dataset_name]["non_zero"].items() if
                                                      (count < 250 and count > 40)]
        # candidate_categories_new[ood_dataset_name] = [
        #     cat_name for cat_name, count in new_categories_count[ood_dataset_name]["non_zero"].items() if count > 40
        # ]
    candidate_categories_new["coco_near"] = [
        cat_name for cat_name, count in new_categories_count["coco_near"]["non_zero"].items() if (count > 20)
    ]
    # Delete shared categories within candidate categories inside each dataset
    for ood_dataset_name in cfg.OOD_DATASETS:
        candidate_categories_new[f"{ood_dataset_name}_near"] = list(
            set(candidate_categories_new[f"{ood_dataset_name}_near"]) - set(candidate_categories_new[ood_dataset_name])
        )
    # candidate_categories_new["coco_near"].remove("giraffe")
    if cfg.GET_OI_CANDIDATES:
        # Openimages new candidates download
        oi_img_ids_candidates = {d:{} for d in ["openimages", "openimages_near"]}
        for ood_dataset_name in oi_img_ids_candidates.keys():
            print(
                f"Getting {ood_dataset_name} candidates, "
                f"storing them in {cfg.NEW_SPLITS_CANDIDATES_IMAGES_PATHS[ood_dataset_name]}"
            )
            (oi_img_ids_candidates[ood_dataset_name]["candidates_ids"],
             oi_img_ids_candidates[ood_dataset_name]["candidates_ids_for_dl"]) = (
                get_oi_image_candidates(
                bboxes=open_images_bboxes,
                unwanted_im_ids=oi_im_ids_benchmark + list(cfg.manual_selection.FROM_CANDIDATES[ood_dataset_name]),
                wanted_categories=candidate_categories_new[ood_dataset_name],
                unwanted_categories=cfg.categories.CATS_TO_REMOVE["openimages"] + cfg.categories.VOC_CLASSES["openimages"],
                max_candidates=cfg.MAX_CANDIDATES,
                oi_hierarchy=oi_hierarchy_u
                )
            )

            oi_downloader_args = {
                "download_folder": cfg.NEW_SPLITS_CANDIDATES_IMAGES_PATHS[ood_dataset_name],
                "image_list": oi_img_ids_candidates[ood_dataset_name]["candidates_ids_for_dl"],
                "num_processes": 5
            }
            download_all_images(oi_downloader_args)
            # Visualize candidates with annotations
            visualize_candidate_imgs_oi(
                images_path=cfg.NEW_SPLITS_CANDIDATES_IMAGES_PATHS[ood_dataset_name],
                num_imgs_to_disp=cfg.VISUALIZE_OI_N_CANDIDATES,
                images_ids_list=oi_img_ids_candidates[ood_dataset_name]["candidates_ids"],
                bboxes=open_images_bboxes,
                oi_hierarchy=oi_hierarchy_u)

        # Manual checking images because of lack of rigorous labeling, for example, in many images,
        # humans are labeled as 'mammals'
    if cfg.GET_COCO_CANDIDATES:
        # Copy COCO candidates to temporary folder for manual checking
        coco_img_ids_candidates = {"coco": [], "coco_near": []}
        for ood_dataset_name in coco_img_ids_candidates.keys():
            print(
                f"Getting {ood_dataset_name} candidates, "
                f"storing them in {cfg.NEW_SPLITS_CANDIDATES_IMAGES_PATHS[ood_dataset_name]}"
            )
            coco_img_ids_candidates[ood_dataset_name] = get_coco_image_candidates(
                annotations=coco_train_annotations,
                unwanted_im_ids=coco_im_ids_benchmark + cfg.manual_selection.FROM_CANDIDATES[ood_dataset_name],
                wanted_categories=candidate_categories_new[ood_dataset_name],
                unwanted_categories=cfg.categories.CATS_TO_REMOVE["coco"] + cfg.categories.VOC_CLASSES["coco"],
                max_candidates=cfg.MAX_CANDIDATES,
            )

            # Copy COCO candidates to temporary folder for manual inspection
            copy_images(
                dataset=coco_train_annotations,
                original_path=cfg.COCO_PATHS["train_images"],
                target_path=cfg.NEW_SPLITS_CANDIDATES_IMAGES_PATHS[ood_dataset_name],
                image_ids=coco_img_ids_candidates[ood_dataset_name]
            )
    # COCO manual images check

    # Add manually added images from candidates
    for ood_dataset_name in ["coco", "coco_near"]:
        annotations[ood_dataset_name] = add_manual_selection_of_candidates_coco(
            annotations_dest=annotations[ood_dataset_name],
            annotations_orig=coco_train_annotations,
            images_to_add=cfg.manual_selection.FROM_CANDIDATES[ood_dataset_name],
        )
        # Copy manually added images from candidates to new paths
        copy_images(
            dataset=coco_train_annotations,
            original_path=cfg.COCO_PATHS["train_images"],
            target_path=cfg.NEW_SPLITS_IMAGES_PATHS[ood_dataset_name],
            image_ids=cfg.manual_selection.FROM_CANDIDATES[ood_dataset_name],
        )

    # Openimages Add manually checked images from candidates
    for ood_dataset_name in ["openimages", "openimages_near"]:
        download_oi_new_images(
            destination_folder=cfg.NEW_SPLITS_IMAGES_PATHS[ood_dataset_name],
            img_ids_list=cfg.manual_selection.FROM_CANDIDATES[ood_dataset_name],
        )
        annotations[ood_dataset_name] = add_manual_selection_of_candidates_oi(
            annotations_dest=annotations[ood_dataset_name],
            bboxes=open_images_bboxes,
            images_to_add=cfg.manual_selection.FROM_CANDIDATES[ood_dataset_name],
            images_path=cfg.NEW_SPLITS_IMAGES_PATHS[ood_dataset_name],
            oi_hierarchy=oi_hierarchy_u
        )
    # Save all new annotation files
    for ood_dataset_name in ALL_OOD_DATASETS:
        print(f"Saving {ood_dataset_name} annotations")
        save_annotation_file_coco_format(
            dataset=annotations[ood_dataset_name],
            out_file_name=cfg.NEW_SPLITS_ANNOTATIONS_PATHS[ood_dataset_name],
        )
    # Redo categories count after new benchmark construction
    final_categories_count = {}
    for dataset_name, dataset_annotations in annotations.items():
        final_categories_count[dataset_name] = {}
        final_categories_count[dataset_name]["non_zero"], final_categories_count[dataset_name][
            "zero_count"] = make_barplot_categories(dataset_annotations, dataset_name)
    # Visualize some of the new images and their annotations
    for dataset_name in ALL_OOD_DATASETS:
        visualize_annotations(
            coco_dataset=annotations[dataset_name],
            images_path=cfg.NEW_SPLITS_IMAGES_PATHS[dataset_name],
            num_imgs_to_disp=4,
            dataset_name=dataset_name
        )

    print("Done!")


def download_oi_new_images(destination_folder: str, img_ids_list: List):
    img_ids_list = [f"train/{im_id}" for im_id in img_ids_list]
    oi_downloader_args = {
        "download_folder": destination_folder,
        "image_list": img_ids_list,
        "num_processes": 5
    }
    download_all_images(oi_downloader_args)

def copy_images(dataset: COCOParser, original_path: str, target_path: str, image_ids: Optional[List] = None) -> None:
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if image_ids is None:
        # Get images ids
        image_ids = dataset.get_imgIds()

    for img_id in image_ids:
        src = f"{original_path}/{dataset.im_dict[img_id]['file_name']}"
        dst = f"{target_path}/{dataset.im_dict[img_id]['file_name']}"
        # dst = f"{target_path}/{img_id}.jpg"
        shutil.copyfile(src, dst)


def visualize_candidate_imgs_oi(images_path, num_imgs_to_disp, images_ids_list, bboxes, oi_hierarchy):
    color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "black", "magenta", "green", "aqua"] * 61
    # Divide by groups of four
    if len(images_ids_list) > num_imgs_to_disp:
        images_ids_list = images_ids_list[:num_imgs_to_disp]
    n_groups = len(images_ids_list) // 4
    for group in range(n_groups):
        images_group = images_ids_list[4 * group: 4 * (group + 1)]

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        ax = ax.ravel()
        for i, im_id in enumerate(images_group):
            image = Image.open(f"{images_path}/{im_id}.jpg")
            width = image.size[0]
            height = image.size[1]
            annotations = {ann_id: ann for ann_id, ann in bboxes.items() if ann["ImageID"] == im_id}
            for ann_id, ann in annotations.items():

                # cx = round((float(ann["XMin"]) + float(ann["XMax"])) * width / 2.0, 2)
                # cy = round((float(ann["YMin"]) + float(ann["YMax"])) * height / 2.0, 2)
                x = round(float(ann["XMin"]) * width, 2)
                y = round(float(ann["YMin"]) * height, 2)
                h = round((float(ann["YMax"]) - float(ann["YMin"])) * height, 2)
                w = round((float(ann["XMax"]) - float(ann["XMin"])) * width, 2)

                class_id = oi_hierarchy[ann["LabelName"]]["id"]
                class_name = oi_hierarchy[ann["LabelName"]]["name"]
                # license = coco_dataset.get_imgLicenses(im)[0]["name"]
                color_ = color_list[class_id]
                rect = plt.Rectangle((x, y), w, h, linewidth=3, edgecolor=color_, facecolor='none')
                t_box = ax[i].text(x, y, class_name, color='red', fontsize=12)
                t_box.set_bbox(dict(boxstyle='square, pad=0', facecolor='white', alpha=0.8, edgecolor='blue'))
                ax[i].add_patch(rect)

            ax[i].axis('off')
            ax[i].imshow(image)
            ax[i].set_xlabel('Longitude')
            ax[i].set_title(f"Im ID: {im_id} ")
        plt.suptitle(f"Candidate images annotations")
        plt.tight_layout()
        plt.show()

def get_oi_image_candidates(
    bboxes: Dict,
    unwanted_im_ids: List[str],
    wanted_categories: List[str],
    unwanted_categories: List[str],
    max_candidates: int,
    oi_hierarchy: Dict
):
    candidate_counter = 0
    candidate_im_ids = set()
    discarded_im_ids = set()
    with tqdm(total=max_candidates, desc="Getting OI img ids candidates") as pbar:
        for ann_id, ann in bboxes.items():
            if ann["ImageID"] not in unwanted_im_ids:
                if (oi_hierarchy[ann["LabelName"]]["name"] in wanted_categories and
                        ann["ImageID"] not in candidate_im_ids and ann["ImageID"] not in discarded_im_ids):
                    candidate_im_ids.add(ann["ImageID"])
                    candidate_counter += 1
                    pbar.update(1)
                if oi_hierarchy[ann["LabelName"]]["name"] in unwanted_categories:
                    discarded_im_ids.add(ann["ImageID"])
                    if ann["ImageID"] in candidate_im_ids:
                        candidate_im_ids.remove(ann["ImageID"])
                        candidate_counter -= 1
                        pbar.update(-1)
            if candidate_counter >= max_candidates:
                break

    candidate_im_ids = candidate_im_ids - discarded_im_ids
    # Adapt for downloading script, since images belong to training set
    candidate_im_ids_for_dl = [f"train/{im_id}" for im_id in candidate_im_ids]
    return list(candidate_im_ids), candidate_im_ids_for_dl

def get_coco_image_candidates(
    annotations: COCOParser,
    unwanted_im_ids: List[str],
    wanted_categories: List[str],
    unwanted_categories: List[str],
    max_candidates: int,
):
    candidate_counter = 0
    candidate_im_ids = set()
    discarded_im_ids = set()
    with tqdm(total=max_candidates, desc="Getting COCO img ids candidates") as pbar:
        for ann_id, ann in annotations.annId_dict.items():
            if ann["image_id"] not in unwanted_im_ids:
                if (annotations.cat_dict[ann["category_id"]]["name"] in wanted_categories and
                        ann["image_id"] not in candidate_im_ids and ann["image_id"] not in discarded_im_ids):
                    candidate_im_ids.add(ann["image_id"])
                    candidate_counter += 1
                    pbar.update(1)
                if annotations.cat_dict[ann["category_id"]]["name"] in unwanted_categories:
                    discarded_im_ids.add(ann["image_id"])
                    if ann["image_id"] in candidate_im_ids:
                        candidate_im_ids.remove(ann["image_id"])
                        candidate_counter -= 1
                        pbar.update(-1)
            if candidate_counter >= max_candidates:
                break

    candidate_im_ids = candidate_im_ids - discarded_im_ids

    return list(candidate_im_ids)

def read_csv_file(file_path: str, fieldnames=None):
    file_dict = {}
    with open(file_path, mode='r') as csv_file:
        file_reader = csv.DictReader(csv_file, fieldnames)
        line_count = 0
        for row in file_reader:
            file_dict[line_count] = row
            line_count += 1
            # if line_count > 1000000:
            #     break
        print(f'Processed {line_count} lines.')
    return file_dict

def convert_cat_ids_oi(cat_ids_dict):
    new_dict = {}
    for cat in cat_ids_dict.values():
        new_dict[cat["cat_id"]] = cat["cat_name"]
    return new_dict

def convert_oi_hierarchy(oi_hierarchy_node, cat_ids):
    converted_h = {}
    for subcat in oi_hierarchy_node:
        converted_h[subcat["LabelName"]] = {"cat_name": cat_ids[subcat["LabelName"]]}
        if "Subcategory" in subcat.keys():
            converted_h[subcat["LabelName"]]["Subcategory"] = convert_oi_hierarchy(subcat["Subcategory"], cat_ids)
        if "Part" in subcat.keys():
            converted_h[subcat["LabelName"]]["Part"] = convert_oi_hierarchy(subcat["Part"], cat_ids)
    return converted_h

def make_barplot_categories(dataset_annotations, dataset_name):
    non_zero_categories = {}
    zero_freq_categories = []
    for cat in dataset_annotations.cat_dict.values():
        if cat["count"] == 0:
            zero_freq_categories.append(cat)
        else:
            non_zero_categories[cat["name"]] = cat["count"]
    # Sort by count
    non_zero_categories = dict(sorted(non_zero_categories.items(), key=lambda item: item[1], reverse=False))
    y_pos = np.arange(len(non_zero_categories))
    plt.figure(figsize=(6, max(int(len(non_zero_categories)*0.176), 3)))
    plt.barh(y_pos, non_zero_categories.values(),)
    plt.yticks(y_pos, non_zero_categories.keys())
    plt.title(f"Category Count for {dataset_name}")
    plt.tight_layout()
    plt.xlabel("Count")
    plt.grid(axis='x')
    plt.show()
    return non_zero_categories, zero_freq_categories


def add_bboxes_to_annotations(annotations, open_images_bboxes, oi_hierarchy):
    for ann_id, ann in open_images_bboxes.items():
        width = annotations.im_dict[ann["ImageID"]]["width"]
        height = annotations.im_dict[ann["ImageID"]]["height"]
        # cx = round((float(ann["XMin"]) + float(ann["XMax"])) * width / 2.0, 2)
        # cy = round((float(ann["YMin"]) + float(ann["YMax"])) * height / 2.0, 2)
        x = round(float(ann["XMin"]) * width, 2)
        y = round(float(ann["YMin"]) * height, 2)
        h = round((float(ann["YMax"]) - float(ann["YMin"])) * height, 2)
        w = round((float(ann["XMax"]) - float(ann["XMin"])) * width, 2)
        bbox = [x, y, w, h]
        # Feed annIm_dict dictionary
        annotations.annIm_dict[ann["ImageID"]].append(
            {
                "bbox": bbox,
                "iscrowd": int(ann["IsGroupOf"]),
                "image_id": ann["ImageID"],
                "category_id": oi_hierarchy[ann["LabelName"]]["id"],
                "id": ann_id
            }
        )
        # Feed annID_dict dictionary
        annotations.annId_dict[ann_id] = {
            "iscrowd": int(ann["IsGroupOf"]),
            "bbox": bbox,
            "category_id": oi_hierarchy[ann["LabelName"]]["id"],
            "image_id": ann["ImageID"],
            "id": ann_id
        }
        annotations.cat_dict[oi_hierarchy[ann["LabelName"]]["id"]]["count"] += 1
    return annotations


def unfold_oi_categories(oi_hierarchy_node, parents=[], depth=0):
    categories = []
    for subcat_code, subcat in oi_hierarchy_node.items():
        if depth == 0:
            categories.append(
                {
                    subcat_code: {
                        "name": subcat["cat_name"],
                        "supercategory": subcat["cat_name"],
                        "metacategory": subcat["cat_name"],
                    }
                }
            )
        elif depth == 1:
            categories.append(
                {
                    subcat_code: {
                        "name": subcat["cat_name"],
                        "supercategory": parents[0],
                        "metacategory": parents[0],
                    }
                }
            )
        elif depth >= 2:
            categories.append(
                {
                    subcat_code: {
                        "name": subcat["cat_name"],
                        "supercategory": parents[-1],
                        "metacategory": parents[-2],
                    }
                }
            )
        if "Subcategory" in subcat.keys():
            categories.extend(
                unfold_oi_categories(subcat["Subcategory"], parents + [subcat["cat_name"]], depth + 1)
            )
    if depth == 0:
        categories = {k:v for cat in categories for k,v in cat.items()}
        for idx, cat in enumerate(categories.values()):
            cat['id'] = idx + 1
    return categories

def save_annotation_file_coco_format(dataset, out_file_name):
    # Get images ids
    img_ids = dataset.get_imgIds()

    # Subset the dictionary
    ann_ids = dataset.get_annIds(img_ids)
    anns = dataset.load_anns(ann_ids)
    imgs_info = dataset.get_img_info(img_ids)

    # Build new dictionary
    coco_dict = {
        'info': dataset.info_dict['info'],
        'licenses': dataset.licenses_dict['licenses'],
        'images': imgs_info,
        'annotations': anns,
        'categories': list(dataset.cat_dict.values())
    }
    print(f"Saved {len(imgs_info)} images")
    # Save dictionary as json
    with open(out_file_name, "w") as outfile:
        json.dump(coco_dict, outfile)

def visualize_annotations(coco_dataset, images_path, num_imgs_to_disp, dataset_name):
    color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "black", "magenta", "green", "aqua"] * 61
    total_images = len(coco_dataset.get_imgIds())  # total number of images
    sel_im_idxs = np.random.permutation(total_images)[:num_imgs_to_disp]
    img_ids = coco_dataset.get_imgIds()
    selected_img_ids = [img_ids[i] for i in sel_im_idxs]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    ax = ax.ravel()
    for i, im in enumerate(selected_img_ids):
        image = Image.open(f"{images_path}/{coco_dataset.im_dict[im]['file_name']}")
        ann_ids = coco_dataset.get_annIds(im)
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
        ax[i].set_title(f"Im ID: {im}")
    plt.suptitle(f"{dataset_name} annotations")
    plt.tight_layout()
    plt.show()

def filter_open_im_bboxes(open_images_bboxes: Dict, open_images_im_ids: List) -> Dict:
    ann_kept = {idx:ann for idx, ann in tqdm(open_images_bboxes.items()) if ann["ImageID"] in open_images_im_ids}
    return ann_kept

def get_images_to_near_and_remove(
        annotations_dict: Dict,
        ood_dataset_names: List,
        categories_to_remove: Dict,
        categories_to_near_ood: Dict,
        ids_to_remove: Dict,
        ids_to_near: Dict
) -> Tuple[Dict, Dict]:
    # Find images to remove with the overlapping categories
    images_to_remove = {"openimages": {"all_cats": []}, "coco": {"all_cats": []}}
    # Find images defined as near ood because of semantic similarity to ID categories
    images_to_near_ood = {"openimages": {"all_cats": []}, "coco": {"all_cats": []}}
    for ood_dataset_name in ood_dataset_names:
        for cat in categories_to_remove[ood_dataset_name]:
            images_to_remove[ood_dataset_name][cat] = annotations_dict[ood_dataset_name].get_img_ids_per_cat_name(cat)
            images_to_remove[ood_dataset_name]["all_cats"].extend(
                annotations_dict[ood_dataset_name].get_img_ids_per_cat_name(cat))
        for cat in categories_to_near_ood[ood_dataset_name]:
            images_to_near_ood[ood_dataset_name][cat] = annotations_dict[ood_dataset_name].get_img_ids_per_cat_name(cat)
            images_to_near_ood[ood_dataset_name]["all_cats"].extend(
                annotations_dict[ood_dataset_name].get_img_ids_per_cat_name(cat))
    # Filter images that will be kept for near ood subset, in remove lists
    for ood_dataset_name in ood_dataset_names:
        # Add remove ids if not present
        images_to_remove[ood_dataset_name]["all_cats"].extend(ids_to_remove[ood_dataset_name])
        images_to_remove[ood_dataset_name]["all_cats"] = list(
            set(images_to_remove[ood_dataset_name]["all_cats"])
        )
        # Delete ids that will be kept for near ood
        images_to_remove[ood_dataset_name]["all_cats"] = [
            im_id for im_id in images_to_remove[ood_dataset_name]["all_cats"] if
            im_id not in ids_to_near[ood_dataset_name]
        ]
    # Filter images that will be removed bc of overlap, in near ood lists
    for ood_dataset_name in ood_dataset_names:
        # Add keep ids if not present
        images_to_near_ood[ood_dataset_name]["all_cats"].extend(ids_to_near[ood_dataset_name])
        images_to_near_ood[ood_dataset_name]["all_cats"] = list(
            set(images_to_near_ood[ood_dataset_name]["all_cats"])
        )
        # Remove images that will be removed
        images_to_near_ood[ood_dataset_name]["all_cats"] = [
            im_id for im_id in images_to_near_ood[ood_dataset_name]["all_cats"] if
            im_id not in ids_to_remove[ood_dataset_name]
        ]
    # Check no intersection in remove and near ood ids
    for ood_dataset_name in ood_dataset_names:
        # intersect = list(
        #     set(images_to_remove[ood_dataset_name]["all_cats"]).intersection(
        #         set(images_to_near_ood[ood_dataset_name]["all_cats"])
        #     )
        # )
        # if len(intersect) > 0:
        #     for im_id in intersect:
        #         images_to_near_ood[ood_dataset_name]["all_cats"].remove(im_id)

        images_to_near_ood[ood_dataset_name]["all_cats"] = list(
            set(images_to_near_ood[ood_dataset_name]["all_cats"]) - set(images_to_remove[ood_dataset_name]["all_cats"])
        )
    return images_to_remove, images_to_near_ood

def visualize_removed_or_near_images(coco_dataset, images_path, num_imgs_to_disp, dataset_name, images_ids_list, category_name):
    color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "black", "magenta", "green", "aqua"] * 61
    if num_imgs_to_disp < len(images_ids_list):
        selected_img_ids = np.random.permutation(images_ids_list)[:num_imgs_to_disp]
    else:
        selected_img_ids = images_ids_list

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    ax = ax.ravel()
    for i, im in enumerate(selected_img_ids):
        image = Image.open(f"{images_path}/{coco_dataset.im_dict[im]['file_name']}")
        ann_ids = coco_dataset.get_annIds(im)
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
        ax[i].set_title(f"Im ID: {im} category: {category_name}")
    plt.suptitle(f"{dataset_name} annotations")
    plt.tight_layout()
    plt.show()

def remove_and_split_manual_inspection(
    annotations: Dict,
    images_to_remove: List,
    images_to_near_ood: List,
    images_to_far_ood: List,
    base_dataset_name: str
) -> Dict:
    old_dataset_name = base_dataset_name
    # Remove images from annotations bc of overlap
    for im_id in images_to_remove:
        if im_id in annotations[old_dataset_name].im_dict.keys():
            del annotations[old_dataset_name].im_dict[im_id]
        if im_id in annotations[old_dataset_name].annIm_dict.keys():
            for ann in annotations[old_dataset_name].annIm_dict[im_id]:
                annotations[old_dataset_name].cat_dict[ann["category_id"]]["count"] -= 1
                del annotations[old_dataset_name].annId_dict[ann["id"]]
            del annotations[old_dataset_name].annIm_dict[im_id]

    # Add to new dataset near near and delete from old dataset
    near_dataset_name = old_dataset_name + "_near"
    # Remove images from annotations bc of overlap
    for im_id in images_to_near_ood:
        if im_id in annotations[old_dataset_name].im_dict.keys():
            annotations[near_dataset_name].im_dict[im_id] = copy.deepcopy(annotations[old_dataset_name].im_dict[im_id])
            del annotations[old_dataset_name].im_dict[im_id]

        if im_id in annotations[old_dataset_name].annIm_dict.keys():
            annotations[near_dataset_name].annIm_dict[im_id] = copy.deepcopy(
                annotations[old_dataset_name].annIm_dict[im_id]
            )
            for ann in annotations[old_dataset_name].annIm_dict[im_id]:
                annotations[near_dataset_name].cat_dict[ann["category_id"]]["count"] += 1
                annotations[old_dataset_name].cat_dict[ann["category_id"]]["count"] -= 1
                annotations[near_dataset_name].annId_dict[ann["id"]] = copy.deepcopy(
                    annotations[old_dataset_name].annId_dict[ann["id"]]
                )
                del annotations[old_dataset_name].annId_dict[ann["id"]]
            del annotations[old_dataset_name].annIm_dict[im_id]

    for im_id in images_to_far_ood:
        if im_id in annotations[near_dataset_name].im_dict.keys():
            annotations[old_dataset_name].im_dict[im_id] = copy.deepcopy(annotations[near_dataset_name].im_dict[im_id])
            del annotations[near_dataset_name].im_dict[im_id]

        if im_id in annotations[near_dataset_name].annIm_dict.keys():
            annotations[old_dataset_name].annIm_dict[im_id] = copy.deepcopy(
                annotations[near_dataset_name].annIm_dict[im_id]
            )
            for ann in annotations[near_dataset_name].annIm_dict[im_id]:
                annotations[old_dataset_name].cat_dict[ann["category_id"]]["count"] += 1
                annotations[near_dataset_name].cat_dict[ann["category_id"]]["count"] -= 1
                annotations[old_dataset_name].annId_dict[ann["id"]] = copy.deepcopy(
                    annotations[near_dataset_name].annId_dict[ann["id"]]
                )
                del annotations[near_dataset_name].annId_dict[ann["id"]]
            del annotations[near_dataset_name].annIm_dict[im_id]
    return annotations

def add_manual_selection_of_candidates_coco(
    annotations_dest: COCOParser,
    annotations_orig: COCOParser,
    images_to_add: List,
) -> COCOParser:

    for im_id in images_to_add:
        if im_id in annotations_orig.im_dict.keys():
            annotations_dest.im_dict[im_id] = copy.deepcopy(annotations_orig.im_dict[im_id])
        else:
            print(f"{im_id} not in original annotations")

        if im_id in annotations_orig.annIm_dict.keys():
            annotations_dest.annIm_dict[im_id] = copy.deepcopy(
                annotations_orig.annIm_dict[im_id]
            )
            for ann in annotations_orig.annIm_dict[im_id]:
                annotations_dest.cat_dict[ann["category_id"]]["count"] += 1
                annotations_orig.cat_dict[ann["category_id"]]["count"] -= 1
                annotations_dest.annId_dict[ann["id"]] = copy.deepcopy(
                    annotations_orig.annId_dict[ann["id"]]
                )

    return annotations_dest


def add_manual_selection_of_candidates_oi(
    annotations_dest: COCOParser,
    bboxes: Dict,
    images_to_add: List,
    images_path: str,
    oi_hierarchy: Dict,
) -> COCOParser:

    for im_id in tqdm(images_to_add, desc="Adding new OpenImages Images annotations"):
        image = Image.open(f"{images_path}/{im_id}.jpg")
        width = image.size[0]
        height = image.size[1]
        annotations_dest.im_dict[im_id] = {
            "file_name": f"{im_id}.jpg",
            "height": height,
            "width": width,
            "id": im_id,
            "license": 1,
        }
        annotations_im = {ann_id: ann for ann_id, ann in bboxes.items() if ann["ImageID"] == im_id}
        for ann_id, ann in annotations_im.items():
            if ann_id in annotations_dest.annId_dict.keys():
                if im_id == annotations_dest.annId_dict[ann_id]["image_id"]:
                    continue
                else:
                    raise ValueError(
                        f"Annotation {ann_id} already in annotations, "
                        f"associated to img {annotations_dest.annId_dict[ann_id]['image_id']} "
                        f"attempted to associate to img {im_id}"
                    )

            # cx = round((float(ann["XMin"]) + float(ann["XMax"])) * width / 2.0, 2)
            # cy = round((float(ann["YMin"]) + float(ann["YMax"])) * height / 2.0, 2)
            x = round(float(ann["XMin"]) * width, 2)
            y = round(float(ann["YMin"]) * height, 2)
            h = round((float(ann["YMax"]) - float(ann["YMin"])) * height, 2)
            w = round((float(ann["XMax"]) - float(ann["XMin"])) * width, 2)
            bbox = [x, y, w, h]
            category_id = oi_hierarchy[ann["LabelName"]]["id"]
            # category_name = oi_hierarchy[ann["LabelName"]]["name"]
            annotations_dest.annIm_dict[ann["ImageID"]].append(
                {
                    "bbox": bbox,
                    "iscrowd": int(ann["IsGroupOf"]),
                    "image_id": ann["ImageID"],
                    "category_id": category_id,
                    "id": ann_id
                }
            )
            # Feed annID_dict dictionary
            annotations_dest.annId_dict[ann_id] = {
                "iscrowd": int(ann["IsGroupOf"]),
                "bbox": bbox,
                "category_id": category_id,
                "image_id": ann["ImageID"],
                "id": ann_id
            }
            annotations_dest.cat_dict[oi_hierarchy[ann["LabelName"]]["id"]]["count"] += 1
    return annotations_dest


def split_old_benchmark(
    annotations: Dict,
    images_to_remove: Dict,
    images_to_near_ood: Dict,
    new_splits_annotations_paths: Dict,
    ood_dataset_names: List,
) -> Dict:
    near_ood_annotations = {}
    for ood_dataset_name in ood_dataset_names:
        # Remove images from annotations bc of overlap
        for im_id in images_to_remove[ood_dataset_name]["all_cats"]:
            if im_id in annotations[ood_dataset_name].im_dict.keys():
                del annotations[ood_dataset_name].im_dict[im_id]
            if im_id in annotations[ood_dataset_name].annIm_dict.keys():
                for ann in annotations[ood_dataset_name].annIm_dict[im_id]:
                    annotations[ood_dataset_name].cat_dict[ann["category_id"]]["count"] -= 1
                    del annotations[ood_dataset_name].annId_dict[ann["id"]]

                del annotations[ood_dataset_name].annIm_dict[im_id]
        # Create the 'near' ood datasets if asked to do so
        if f"{ood_dataset_name}_near" in new_splits_annotations_paths.keys():
            # Subset the dictionary
            near_ann_ids = annotations[ood_dataset_name].get_annIds(images_to_near_ood[ood_dataset_name]["all_cats"])
            near_anns = annotations[ood_dataset_name].load_anns(near_ann_ids)
            near_imgs_info = annotations[ood_dataset_name].get_img_info(images_to_near_ood[ood_dataset_name]["all_cats"])
            near_cat_dict = copy.deepcopy(annotations[ood_dataset_name].cat_dict)
            for cat in near_cat_dict.values():
                cat["count"] = 0

            for im_id in images_to_near_ood[ood_dataset_name]["all_cats"]:
                if im_id in annotations[ood_dataset_name].im_dict.keys():
                    del annotations[ood_dataset_name].im_dict[im_id]
                if im_id in annotations[ood_dataset_name].annIm_dict.keys():
                    for ann in annotations[ood_dataset_name].annIm_dict[im_id]:
                        annotations[ood_dataset_name].cat_dict[ann["category_id"]]["count"] -= 1
                        near_cat_dict[ann["category_id"]]["count"] += 1
                        del annotations[ood_dataset_name].annId_dict[ann["id"]]
                    del annotations[ood_dataset_name].annIm_dict[im_id]

            # Build new dictionary
            near_ood_annotations[f"{ood_dataset_name}_near"] = {
                'info': annotations[ood_dataset_name].info_dict['info'],
                'licenses': annotations[ood_dataset_name].licenses_dict['licenses'],
                'images': near_imgs_info,
                'annotations': near_anns,
                'categories': list(near_cat_dict.values())
            }
            # Save to temporary file, to be able to use the COCO parser format
            with open(new_splits_annotations_paths[f"{ood_dataset_name}_near"], "w") as outfile:
                json.dump(near_ood_annotations[f"{ood_dataset_name}_near"], outfile)
            # Read the new file
            annotations[f"{ood_dataset_name}_near"] = COCOParser(new_splits_annotations_paths[f"{ood_dataset_name}_near"])

    return annotations


if __name__ == "__main__":
    main()

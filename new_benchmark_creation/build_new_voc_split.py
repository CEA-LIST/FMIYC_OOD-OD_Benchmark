import json
import hydra
from omegaconf import DictConfig

from OI_downloader import download_all_images
from coco_parser import COCOParser
from utils import read_csv_file, convert_cat_ids_oi, convert_oi_hierarchy, unfold_oi_categories, \
    filter_open_im_bboxes, add_bboxes_to_annotations, make_barplot_categories, get_images_to_near_and_remove, \
    visualize_removed_or_near_images, split_old_benchmark, remove_and_split_manual_inspection, copy_images, \
    save_annotation_file_coco_format, visualize_annotations, download_oi_new_images, \
    add_manual_selection_of_candidates_oi, add_manual_selection_of_candidates_coco, get_coco_image_candidates, \
    visualize_candidate_imgs_oi, get_oi_image_candidates


@hydra.main(version_base=None, config_path="configs", config_name="config_far_near.yaml")
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


if __name__ == "__main__":
    main()

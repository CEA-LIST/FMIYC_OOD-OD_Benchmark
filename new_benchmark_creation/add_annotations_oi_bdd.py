import json
import hydra
from omegaconf import DictConfig

from utils import read_csv_file, convert_cat_ids_oi, convert_oi_hierarchy, unfold_oi_categories, \
    filter_open_im_bboxes, add_bboxes_to_annotations, make_barplot_categories, get_images_to_near_and_remove, \
    visualize_removed_or_near_images, split_old_benchmark, remove_and_split_manual_inspection, copy_images, \
    save_annotation_file_coco_format, visualize_annotations
from coco_parser import COCOParser


@hydra.main(version_base=None, config_path="configs", config_name="config_farther_split_bdd.yaml")
def main(cfg: DictConfig) -> None:
    #############################################################################
    # Read current benchmark annotations
    #############################################################################
    # ALL_OOD_DATASETS = cfg.OOD_DATASETS + cfg.NEW_OOD_DATASETS
    annotations = {}
    for dataset_name in cfg.CURRENT_BENCHMARK_ANNOTATIONS:
        annotations[dataset_name] = COCOParser(cfg.CURRENT_BENCHMARK_ANNOTATIONS[dataset_name])

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

    save_annotation_file_coco_format(
        dataset=annotations["openimages"],
        out_file_name=cfg.openimages_bboxes,
    )

if __name__ == '__main__':
    main()

import json
from collections import defaultdict
from typing import Dict, List, Optional, Union


class COCOParser:
    def __init__(self, anns_file: str, using_subset: Optional[List[Union[str, int]]] = False):
        with open(anns_file, "r") as f:
            coco = json.load(f)

        self.annIm_dict = defaultdict(list)
        # Dict of id: category pairs
        self.cat_dict = {}
        # Dict of original categories (copy of original entry)
        self.categories_original = {"categories": coco["categories"]}
        self.annId_dict = {}
        self.im_dict = {}
        if "licenses" in coco:
            self.licenses_dict = {"licenses": coco["licenses"]}
        else:
            self.licenses_dict = {}
        if "info" in coco:
            self.info_dict = {"info": coco["info"]}
        else:
            self.info_dict = {}
        for cat in coco["categories"]:
            self.cat_dict[cat["id"]] = cat
            self.cat_dict[cat["id"]]["count"] = 0
        for ann in coco["annotations"]:
            if using_subset and ann["image_id"] in using_subset:
                self.annIm_dict[ann["image_id"]].append(ann)
                self.annId_dict[ann["id"]] = ann
                self.cat_dict[ann["category_id"]]["count"] += 1
            elif not using_subset:
                self.annIm_dict[ann["image_id"]].append(ann)
                self.annId_dict[ann["id"]] = ann
                self.cat_dict[ann["category_id"]]["count"] += 1
        for img in coco["images"]:
            if using_subset and img["id"] in using_subset:
                self.im_dict[img["id"]] = img
            elif not using_subset:
                self.im_dict[img["id"]] = img

        # Licenses not actually needed per image
        # for license in coco['licenses']:
        #     self.licenses_dict[license['id']] = license

    def get_imgIds(self):
        return list(self.im_dict.keys())

    def get_annIds(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann["id"] for im_id in im_ids for ann in self.annIm_dict[im_id]]

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
        cat_id = [cat["id"] for cat in self.cat_dict.values() if cat["name"] == cat_name][0]
        return list(
            set(
                [
                    ann["image_id"]
                    for ann in self.annId_dict.values()
                    if ann["category_id"] == cat_id
                ]
            )
        )

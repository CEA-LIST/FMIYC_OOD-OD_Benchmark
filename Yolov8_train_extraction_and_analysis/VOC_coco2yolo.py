import os
from tqdm import tqdm
from pathlib import Path
import yaml
from collections import defaultdict
import json

from BDD_coco2yolo import draw_detect

DESTINATION_FOLDER = Path(f"../CVDatasets/VOC_yolo_format/").resolve()
VOC_DATASET_PATH = Path("../CVDatasets/VOC_0712_converted").resolve()
VOC_ANNOTATIONS_VALID = "../CVDatasets/VOC_0712_converted/val_coco_format.json"
VOC_ANNOTATIONS_TRAIN = "../CVDatasets/VOC_0712_converted/voc0712_train_all.json"
SHOW_N_ANNOTATIONS = 0


def save_annotation_voc_coco2yolo(
    original_dataset_path: Path, destination_dataset_path: Path, split: str, data: dict, annotations: dict
):
    assert split in ["train", "val", "test"]
    image_filepath = data["id"] + ".jpg"
    image_filepath = original_dataset_path / image_filepath
    assert image_filepath.exists(), f"{image_filepath} does not exist"
    # New image will only be a symbolic link
    new_image_filepath = destination_dataset_path / "images" / split / image_filepath.name
    os.makedirs(new_image_filepath.parent, exist_ok=True)
    # Create symbolic link to real image
    os.symlink(image_filepath.as_posix(), new_image_filepath.as_posix(), target_is_directory=False)
    # save labels
    label_filepath = destination_dataset_path / "labels" / split / f"{new_image_filepath.stem}.txt"
    os.makedirs(label_filepath.parent, exist_ok=True)

    with open(label_filepath.as_posix(), "w") as f:
        for annotation in annotations:
            bbox = annotation["bbox"]
            # Annotations seem to be already in the cx, cy, w, h format
            bbox[0] += 0.5 * bbox[2]  # In case center needs to be adjusted to the right
            bbox[1] += 0.5 * bbox[3]  # In case center needs to be adjusted to the right
            # Scale bboxes
            bbox[0] /= data["width"]
            bbox[1] /= data["height"]
            bbox[2] /= data["width"]
            bbox[3] /= data["height"]
            # Subtract -1 from category number to make it from 0 to 9 (They are from 1 to 10)
            f.write("%g %.6f %.6f %.6f %.6f\n" % (annotation["category_id"] - 1, *bbox))


def main() -> None:
    """
    The current script transforms the BDD100k dataset into yolo format, using symbolic links to save disk space
    """
    # Check the destination folder doesn't exist
    if DESTINATION_FOLDER.exists():
        print(f"Destination directory path already exists: {DESTINATION_FOLDER.as_posix}")
        return
    print(f"Creating destination folder: {DESTINATION_FOLDER.as_posix()}")
    os.makedirs(DESTINATION_FOLDER, exist_ok=False)

    # Train set
    images_train_path = VOC_DATASET_PATH / "JPEGImages"
    voc_train = COCOParser(VOC_ANNOTATIONS_TRAIN)
    for data in tqdm(voc_train.im_dict.values(), desc="Train set"):
        # data = data[0]
        ann_ids = voc_train.get_annIds([data["id"]])
        annotations = voc_train.load_anns(ann_ids)
        image_info = voc_train.get_img_info([data["id"]])
        assert image_info[0]["file_name"] == data["file_name"]
        save_annotation_voc_coco2yolo(
            original_dataset_path=images_train_path,
            destination_dataset_path=DESTINATION_FOLDER,
            split="train",
            data=data,
            annotations=annotations,
        )

    # Validation set
    bdd_val = COCOParser(VOC_ANNOTATIONS_VALID)
    images_val_path = VOC_DATASET_PATH / "JPEGImages"
    for data in tqdm(bdd_val.im_dict.values(), desc="Val set"):
        ann_ids = bdd_val.get_annIds([data["id"]])
        annotations = bdd_val.load_anns(ann_ids)
        image_info = bdd_val.get_img_info([data["id"]])
        assert image_info[0]["file_name"] == data["file_name"]
        save_annotation_voc_coco2yolo(
            original_dataset_path=images_val_path,
            destination_dataset_path=DESTINATION_FOLDER,
            split="val",
            data=data,
            annotations=annotations,
        )

    # Dataset file
    print(f"Creating dataset file in {(DESTINATION_FOLDER / 'dataset.yaml').as_posix()}")
    # Subtract -1 from category number to make it from 0 to 9 (They are from 1 to 10)
    categories_dict = {el_list["id"] - 1: el_list["name"] for el_list in voc_train.categories_original["categories"]}
    d = {
        "path": (DESTINATION_FOLDER).as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "",
        "names": categories_dict,
    }  # dictionary

    with open((DESTINATION_FOLDER / "dataset.yaml").as_posix(), "w") as f:
        yaml.dump(d, f, sort_keys=False)

    if SHOW_N_ANNOTATIONS:
        for i, data in enumerate(bdd_val.im_dict.values()):
            img_file_path = data["id"] + ".jpg"
            img_file_path = images_val_path / img_file_path
            # img_file_path = images_val_path / data["file_name"]
            label_filepath = DESTINATION_FOLDER / "labels" / "val" / f"{img_file_path.stem}.txt"
            draw_detect(label_filepath, categories_dict)
            if i >= SHOW_N_ANNOTATIONS:
                break

    print("Done!")


class COCOParser:
    def __init__(self, anns_file):
        with open(anns_file, "r") as f:
            coco = json.load(f)

        self.annIm_dict = defaultdict(list)
        # Dict of id: category pairs
        self.cat_dict = {}
        # Dict of original categories (copy of original entry)
        self.categories_original = {"categories": coco["categories"]}
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {"licenses": coco["licenses"]}
        self.info_dict = {"info": coco["info"]}
        for ann in coco["annotations"]:
            self.annIm_dict[ann["image_id"]].append(ann)
            self.annId_dict[ann["id"]] = ann
        for img in coco["images"]:
            # Remove leading zeros from filenames
            img["file_name"] = str(int(img["file_name"].split(".")[0])) + ".jpg"
            self.im_dict[img["id"]] = img
        for cat in coco["categories"]:
            self.cat_dict[cat["id"]] = cat
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


if __name__ == "__main__":
    main()

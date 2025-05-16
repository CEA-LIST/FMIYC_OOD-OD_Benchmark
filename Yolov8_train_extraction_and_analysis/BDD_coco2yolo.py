import os

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import yaml
from collections import defaultdict
import json

DESTINATION_FOLDER = Path(f"../CVDatasets/bdd100k_yolo_format/").resolve()
BDD_DATASET_PATH = Path("../CVDatasets/bdd100k").resolve()
BDD_ANNOTATIONS_VALID = "../CVDatasets/bdd100k/val_bdd_converted.json"
BDD_ANNOTATIONS_TRAIN = "../CVDatasets/bdd100k/train_bdd_converted.json"
SHOW_N_ANNOTATIONS = 0


def save_annotation_coco2yolo(
    original_dataset_path: Path, destination_dataset_path: Path, split: str, data: dict, annotations: dict
):
    assert split in ["train", "val", "test"]
    image_filepath = original_dataset_path / data["file_name"]
    assert image_filepath.exists()
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
    images_train_path = BDD_DATASET_PATH / "images" / "100k" / "train"
    bdd_train = BDDParser(BDD_ANNOTATIONS_TRAIN)
    for data in tqdm(bdd_train.im_dict.values(), desc="Train set"):
        # data = data[0]
        ann_ids = bdd_train.get_annIds([data["id"]])
        annotations = bdd_train.load_anns(ann_ids)
        image_info = bdd_train.get_img_info([data["id"]])
        assert image_info[0]["file_name"] == data["file_name"]
        save_annotation_coco2yolo(
            original_dataset_path=images_train_path,
            destination_dataset_path=DESTINATION_FOLDER,
            split="train",
            data=data,
            annotations=annotations,
        )

    # Validation set
    bdd_val = BDDParser(BDD_ANNOTATIONS_VALID)
    images_val_path = BDD_DATASET_PATH / "images" / "100k" / "val"
    for data in tqdm(bdd_val.im_dict.values(), desc="Val set"):
        ann_ids = bdd_val.get_annIds([data["id"]])
        annotations = bdd_val.load_anns(ann_ids)
        image_info = bdd_val.get_img_info([data["id"]])
        assert image_info[0]["file_name"] == data["file_name"]
        save_annotation_coco2yolo(
            original_dataset_path=images_val_path,
            destination_dataset_path=DESTINATION_FOLDER,
            split="val",
            data=data,
            annotations=annotations,
        )

    # Dataset file
    print(f"Creating dataset file in {(DESTINATION_FOLDER / 'dataset.yaml').as_posix()}")
    # Subtract -1 from category number to make it from 0 to 9 (They are from 1 to 10)
    categories_dict = {el_list["id"] - 1: el_list["name"] for el_list in bdd_train.categories_original["categories"]}
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
            img_file_path = images_val_path / data["file_name"]
            label_filepath = DESTINATION_FOLDER / "labels" / "val" / f"{img_file_path.stem}.txt"
            draw_detect(label_filepath, categories_dict)
            if i >= SHOW_N_ANNOTATIONS:
                break

    print("Done!")


class BDDParser:
    def __init__(self, anns_file):
        with open(anns_file, "r") as f:
            bdd = json.load(f)

        self.annIm_dict = defaultdict(list)
        self.annId_dict = {}
        self.im_dict = {}
        # Dict of original categories (copy of original entry)
        self.categories_original = {"categories": bdd["categories"]}
        for ann in bdd["annotations"]:
            self.annIm_dict[ann["image_id"]].append(ann)
            self.annId_dict[ann["id"]] = ann
        for img in bdd["images"]:
            # Remove leading zeros from filenames
            # img['file_name'] = str(int(img['file_name'].split('.')[0])) + '.jpg'
            self.im_dict[img["id"]] = img

    def get_imgIds(self):
        return list(self.im_dict.keys())

    def get_annIds(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann["id"] for im_id in im_ids for ann in self.annIm_dict[im_id]]

    def load_anns(self, ann_ids):
        im_ids = ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]

    def get_img_info(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [self.im_dict[im_id] for im_id in im_ids]


def draw_detect(label_filepath: Path, categories_dict, scale=0.5):
    sample = pd.read_csv(label_filepath.as_posix(), delimiter=" ", header=None)
    image_filepath = Path(label_filepath.as_posix().replace("/labels/", "/images/"))
    image_filepath = image_filepath.parent / (image_filepath.stem + ".jpg")
    image = np.array(cv2.cvtColor(cv2.imread(image_filepath.as_posix()), cv2.COLOR_BGR2RGB))
    height, width, depth = image.shape
    color = (255, 0, 0)
    for idx, annot in sample.iterrows():
        # annot = annot.split()
        class_idx = int(annot[0])
        x, y, w, h = float(annot[1]), float(annot[2]), float(annot[3]), float(annot[4])
        xmin = int((x * width) - (w * width) / 2.0)
        ymin = int((y * height) - (h * height) / 2.0)
        xmax = int((x * width) + (w * width) / 2.0)
        ymax = int((y * height) + (h * height) / 2.0)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, categories_dict[int(class_idx)], (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    dpi = 100
    fig, ax = plt.subplots(1, 1, figsize=(int(width / dpi * scale), int(height / dpi * scale)), dpi=dpi)
    ax.imshow(image)
    ax.axis("off")
    plt.show()


if __name__ == "__main__":
    main()

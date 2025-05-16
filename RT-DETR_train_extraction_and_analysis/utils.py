import os
from typing import Union, List, Tuple, Any, Optional

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.colors
from torchvision.datasets import ImageFolder
from transformers import DetrImageProcessor, AutoProcessor, RTDetrImageProcessor
import albumentations as A


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
            self,
            base_folder: str,
            ann_file: str,
            images_path: str,
            id_type_int: bool,
            transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        ann_file = os.path.join(base_folder, ann_file)
        data_path = os.path.join(base_folder, images_path)
        super(CocoDetection, self).__init__(data_path, ann_file, transforms=transforms)
        self.id_type_int = id_type_int  # If False the id_type is a string
        # Correct category ids starting from 1
        for anno in self.coco.anns.values():
            anno["category_id"] = anno["category_id"] - 1

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        if self.id_type_int:
            if not isinstance(idx, int):
                idx = int(idx)
            img, target = super(CocoDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
        # String id_type
        else:
            if isinstance(idx, str):
                im_id = [idx]
                # idx = self.ids.index(im_id[0])
            else:
                try:
                    idx = int(idx)
                    im_id = [self.ids[idx]]
                except ValueError:
                    raise ValueError(f"Index must be of type integer or string, got {type(idx)} instead.")

            img = self._load_image(im_id)
            target = self._load_target(im_id)

            if self.transforms is not None:
                img, target = self.transforms(img, target)
            image_id = im_id[0]

        target = {'image_id': image_id, 'annotations': target}
        return img, target


def save_results_one_pred(
    pil_img,
    img_name,
    scores,
    labels,
    boxes,
    id2label,
    # idx,
    save_folder,
    # dataset_name,
    # inference_threshold
):
    hex = (
        "#042AFF",
        "#0BDBEB",
        "#F3F3F3",
        "#00DFB7",
        "#111F68",
        "#FF6FDD",
        "#FF444F",
        "#CCED00",
        "#00F344",
        "#BD00FF",
        "#00B4FF",
        "#DD00BA",
        "#00FFFF",
        "#26C000",
        "#01FFB3",
        "#7D24FF",
        "#7B0068",
        "#FF1B6C",
        "#FC6D2F",
        "#A2FF0B",
    )
    color_class_dict = {i:matplotlib.colors.to_rgb(color) for i,color in enumerate(hex)}
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    for score, label, (xmin, ymin, xmax, ymax) in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=color_class_dict[label], linewidth=1))
        text = f'{id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.1))
    plt.axis('off')
    # plt.savefig(os.path.join(save_folder, f"pred_sample_{dataset_name}_{idx}_inft{int(inference_threshold * 100)}.jpg"))
    plt.savefig(f"{os.path.join(save_folder, img_name)}.jpg")


class CollatorDetrNoLabels:
    def __init__(
        self,
        processor: Union[DetrImageProcessor, AutoProcessor],
    ):
        self.processor = processor

    def collate_fn(self, batch):
        imgs = [item[0] for item in batch]
        img_paths = [item[2] for item in batch]
        img_names = [item[3] for item in batch]
        orig_sizes = [img.shape[1::] for img in imgs]
        # labels = [item[1] for item in batch]
        encodings = self.processor(images=imgs, return_tensors="pt")
        pixel_values = encodings["pixel_values"] # remove batch dimension
        # target = encodings["labels"] # remove batch dimension
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        batch = {}
        # batch["images"] = imgs
        batch["img_names"] = img_names
        batch["img_paths"] = img_paths
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['orig_sizes'] = orig_sizes
        # Almost dummy field to comply with the Imagelevel Feature extractor
        batch["labels"] = [
            {
                "orig_size": torch.tensor(size),
                "image_id": im_name
            } for size, im_name in zip(orig_sizes, img_names)
        ]
        # batch['labels'] = target
        return batch


datasets_dict = {
    "bdd": {
        "data_folder": "../CVDatasets/bdd100k/",
        "train_annotations": "train_bdd_converted.json",
        "val_annotations": "val_bdd_converted.json",
        "train_images": "images/100k/train",
        "val_images": "images/100k/val",
        "id_type_int": True
    },
    "voc": {
        "data_folder": "../CVDatasets/VOC_0712_converted/",
        "train_annotations": "voc0712_train_all.json",
        "val_annotations": "val_coco_format.json",
        "train_images": "JPEGImages",
        "val_images": "JPEGImages",
        "id_type_int": False
    },
    "coco": {
        "data_folder": "../CVDatasets/COCO/",
        "val_annotations": {
            "bdd": "annotations/instances_val2017_ood_wrt_bdd_rm_overlap.json",
            "voc": "annotations/instances_val2017_ood_rm_overlap.json",
        },
        "val_images": "val2017",
        "id_type_int": True
    },
    "openimages": {
        "data_folder": "../CVDatasets/OpenImages/",
        "val_annotations": {
            "bdd": "ood_classes_rm_overlap/COCO-Format/val_coco_format_int_id.json",
            "voc": "ood_classes_rm_overlap/COCO-Format/val_coco_format_int_id.json",
        },
        "val_images": "ood_classes_rm_overlap/images",
        "id_type_int": True
    },
    "openimages_new": {
        "data_folder": "../CVDatasets/OpenImages/",
        "val_annotations": {
            "bdd": "ood_classes_rm_overlap/COCO-Format/new_oi_wrt_bdd_int_id.json",
            "voc": "id_voc_ood_openimages/ood_classes_rm_overlap/COCO-Format/new_oi_wrt_voc_int_id.json",
        },
        "val_images": {
            "bdd": "ood_classes_rm_overlap/new_images_wrt_bdd",
            "voc": "id_voc_ood_openimages/ood_classes_rm_overlap/new_images",
        },
        "id_type_int": True
    },
    "openimages_all_new": {
        "data_folder": "../CVDatasets/OpenImages/id_voc_ood_openimages/",
        "val_annotations": {
            "bdd": "",
            "voc": "ood_classes_rm_overlap/COCO-Format/all_new_oi_wrt_voc.json",
        },
        "val_images": {
            "bdd": "",
            "voc": "ood_classes_rm_overlap/all_new_images",
        },
        "id_type_int": False
    },
    "openimages_near": {
        "data_folder": "../CVDatasets/OpenImages/id_voc_ood_openimages/",
        "val_annotations": {
            "bdd": "",
            "voc": "ood_classes_rm_overlap/COCO-Format/near_oi_wrt_voc_int_id.json",
        },
        "val_images": {
            "bdd": "",
            "voc": "ood_classes_rm_overlap/near_images",
        },
        "id_type_int": True
    },
    "coco_new": {
        "data_folder": "../CVDatasets/COCO/",
        "val_annotations": {
            "bdd": "annotations/new_coco_wrt_bdd.json",
            "voc": "annotations/new_coco_wrt_voc.json",
        },
        "val_images": {
            "bdd": "new_images_wrt_bdd",
            "voc": "new_images",
        },
        "id_type_int": True
    },
    "coco_all_new": {
        "data_folder": "../CVDatasets/COCO/",
        "val_annotations": {
            "bdd": "",
            "voc": "annotations/all_new_coco_wrt_voc.json",
        },
        "val_images": {
            "bdd": "",
            "voc": "all_new_images",
        },
        "id_type_int": True
    },
    "coco_near": {
        "data_folder": "../CVDatasets/COCO/",
        "val_annotations": {
            "bdd": "",
            "voc": "annotations/near_coco_wrt_voc.json",
        },
        "val_images": {
            "bdd": "",
            "voc": "near_images",
        },
        "id_type_int": True
    },
}


model_config_dict = {
    "DETR": {
        "model_name": "facebook/detr-resnet-50",
        "processor": DetrImageProcessor,
        "saved_processor_path": "./saved_models/DETR_processor_t0",
        "model_path": {
            "bdd": f"./saved_models/DETR_model_bdd_t0",
            "voc": f"./saved_models/DETR_model_voc_t0"
        }
    },
    "RTDETR": {
        "model_name": "PekingU/rtdetr_r50vd",
        "processor": RTDetrImageProcessor,
        "saved_processor_path": "./saved_models/RTDETR_processor_t0",
        "model_path": {
            "bdd": f"./saved_models/RTDETR_model_bdd_t0",
            "voc": f"./saved_models/RTDETR_model_voc_t0"
        }
    },
    "OWL": {
        "model_name": "google/owlv2-base-patch16-ensemble",
        "processor": AutoProcessor,
        "saved_processor_path": "./saved_models/OWL_processor_t0",
        "model_path": {
            "bdd": f"./saved_models/OWL_model_t0",
            "voc": f"./saved_models/OWL_model_t0"
        }
    },
    "CLIP": {
        "model_name": "openai/clip-vit-base-patch32",
        "processor": AutoProcessor,
        "saved_processor_path": "./saved_models/CLIP_processor_t0",
        "model_path": {
            "bdd": f"./saved_models/CLIP_model_t0",
            "voc": f"./saved_models/CLIP_model_t0"
        }
    }
}

bravo_data_paths_dict = {
    "bravo_syn_obj": "../CVDatasets/BRAVO/synobj",
    "bravo_real_obj": "../CVDatasets/BRAVO/real_anom",
    "bravo_syn_rain": "../CVDatasets/BRAVO/bravo_synrain",
    "bravo_syn_flare": "../CVDatasets/BRAVO/bravo_synflare",
    "bravo_outofctx": "../CVDatasets/BRAVO/bravo_outofcontext",
    "acdc_fog": "../CVDatasets/BRAVO/ACDC/fog/test",
    "acdc_night": "../CVDatasets/BRAVO/ACDC/night/test",
    "acdc_rain": "../CVDatasets/BRAVO/ACDC/rain/test",
    "acdc_snow": "../CVDatasets/BRAVO/ACDC/snow/test"
}


class Transforms:
    """
    Transforms (dummy) Class to Apply Albumanetations transforms to
    PyTorch ImageFolder dataset class\n
    See:
        https://albumentations.ai/docs/examples/example/
        https://stackoverflow.com/questions/69151052/using-imagefolder-with-albumentations-in-pytorch
        https://github.com/albumentations-team/albumentations/issues/1010
    """
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        # return self.transforms(image=np.array(img), bboxes=args[0] if len(args)>0 else None)["image"]
        if len(args)>0:
            annotations = args[0]
            bboxes = []
            for annotation in annotations:
                bboxes.append(annotation["bbox"] + [annotation["category_id"]])
            transformed_data = self.transforms(image=np.array(img), bboxes=bboxes)
            for bbox, annotation in zip(transformed_data["bboxes"], annotations):
                annotation["bbox"] = bbox[:4]
            return transformed_data["image"], annotations
        else:
            return self.transforms(image=np.array(img))["image"]


def get_train_aug_transforms():
    train_augment_and_transform = A.Compose(
        [
            # A.RandomSizedBBoxSafeCrop(height=max_size, width=max_size, p=0.2),
            A.OneOf(
                [
                    A.Blur(blur_limit=7, p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.5),
                    A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
                ],
                p=0.1,
            ),
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(format="coco", clip=True, min_area=25),
    )
    return Transforms(train_augment_and_transform)
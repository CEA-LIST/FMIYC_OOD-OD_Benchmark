from datetime import datetime
import os
import PIL
import torch
import hydra
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import DeformableDetrImageProcessor, DetrImageProcessor, RTDetrImageProcessor, AutoProcessor
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from coco_eval import CocoEvaluator
from tqdm import tqdm
from omegaconf import DictConfig

from models import models_dict
from utils import CocoDetection, datasets_dict, save_results_one_pred, get_train_aug_transforms

torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOWNLOAD_MODEL = True
TRAIN = False
RESUME_TRAINING = False  # Either False or a lightning log date of existing checkpoint
MLFLOW_RUN_ID = None  # Either None or an existing run_id to resume training
CHECKPOINT_NAME = None
INFERENCE_EXAMPLES = 40

TENSOR_PRECISION = torch.float32

IND_DATASET = "voc"
INFERENCE_DIR = f"./inference_examples/{IND_DATASET}"
TRAIN_EXAMPLES_FOLDER = "train_set_examples"
SHOW_N_IMAGES_TRAIN = 10


@hydra.main(version_base=None, config_path="config", config_name="config_train.yaml")
def main(cfg: DictConfig):
    experiment_name = f"{cfg.model_type} {IND_DATASET} train"
    if not RESUME_TRAINING:
        current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        current_date_time = RESUME_TRAINING
    pl.seed_everything(cfg.random_seed)
    checkpoint_dirpath = f'lightning_logs/{cfg.model_type}_{current_date_time}_ind_{IND_DATASET}'
    processor = get_preprocessor(cfg.model_type, cfg.model_name, DOWNLOAD_MODEL)
    # Path of pretrained model at t0
    pretrained_model_path = f"./saved_models/{cfg.model_type}_model_{IND_DATASET}_t0"

    train_transforms = get_train_aug_transforms()
    train_dataset = CocoDetection(
        base_folder=datasets_dict[IND_DATASET]["data_folder"],
        ann_file=datasets_dict[IND_DATASET]["train_annotations"],
        images_path=datasets_dict[IND_DATASET]["train_images"],
        id_type_int=True if IND_DATASET == "bdd" else False,
        transforms=train_transforms
    )
    val_dataset = CocoDetection(
        base_folder=datasets_dict[IND_DATASET]["data_folder"],
        ann_file=datasets_dict[IND_DATASET]["val_annotations"],
        images_path=datasets_dict[IND_DATASET]["val_images"],
        id_type_int=True if IND_DATASET == "bdd" else False,
        transforms=None
    )

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    id2label = draw_n_images_gt_data(
        train_dataset,
        destination_folder=TRAIN_EXAMPLES_FOLDER,
        n_images=SHOW_N_IMAGES_TRAIN
    )

    def collate_fn(batch):
        imgs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        encoding = processor(images=imgs, annotations=labels, return_tensors="pt")
        # pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        # encoding = processor.pad(encoding, return_tensors="pt")
        if all(isinstance(label["image_id"], str) for label in labels):
            for encoded_label, string_label in zip(encoding["labels"], labels):
                encoded_label["image_id"] = string_label["image_id"]
            # encoding["labels"]. = [label["image_id"] for label in labels]
        for im_label in encoding["labels"]:
            im_label["boxes"] = im_label["boxes"].to(TENSOR_PRECISION)
            im_label["area"] = im_label["area"].to(TENSOR_PRECISION)

        batch = {}
        batch['pixel_values'] = encoding['pixel_values'].to(TENSOR_PRECISION)
        if not cfg.model_type == "RTDETR":
            batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = encoding["labels"]
        return batch

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=cfg.train_batch_size, shuffle=True, num_workers=10)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2, num_workers=10)

    if TRAIN:
        print(
            f"Training model: InD {IND_DATASET}, {cfg.model_type} batch_size {cfg.train_batch_size},"
            f"lr {cfg.learning_rate}, lr_backbone {cfg.learning_rate_backbone}, Resume {RESUME_TRAINING} "
            f"MLFlow_run {MLFLOW_RUN_ID}"
        )
        model = models_dict[cfg.model_type](
            lr=cfg.learning_rate,
            lr_backbone=cfg.learning_rate_backbone,
            weight_decay=1e-4,
            n_labels=len(id2label),
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            download=DOWNLOAD_MODEL,
            pretrained_model_name=cfg.model_name,
            pretrained_model_path=pretrained_model_path,
            tensor_precision=TENSOR_PRECISION,
            num_queries=cfg.num_queries,
        )

        # Test dataloader snippet
        # batch = next(iter(train_dataloader))
        # pixel_values, target = train_dataset[0]
        # print(pixel_values.shape)
        # print(target)
        # outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
        # print(outputs.logits.shape)

        mlf_logger = MLFlowLogger(experiment_name=experiment_name, log_model=True, run_id=MLFLOW_RUN_ID)
        # mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri="http://10.8.33.50:5050", log_model=True)
        mlf_logger.log_hyperparams(cfg)
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dirpath,
            monitor="validation_loss",
            mode="min",
            every_n_epochs=1,
            save_top_k=1,
            save_last=True,
            save_on_train_epoch_end=False)
        early_stopping = EarlyStopping('validation_loss', patience=16, mode='min')
        trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            gradient_clip_val=0.1,
            logger=mlf_logger,
            log_every_n_steps=500,
            callbacks=[checkpoint_callback, early_stopping],
            # strategy='ddp_find_unused_parameters_true'
        )
        trainer.fit(model, ckpt_path=os.path.join(checkpoint_callback.dirpath, "last.ckpt") if RESUME_TRAINING else None)
    else:
        print(f"Loading model from {checkpoint_dirpath}/{CHECKPOINT_NAME}, InD {IND_DATASET}")
        model = models_dict[cfg.model_type].load_from_checkpoint(
            os.path.join(checkpoint_dirpath, CHECKPOINT_NAME),
            lr=cfg.learning_rate,
            lr_backbone=cfg.learning_rate_backbone,
            weight_decay=1e-4,
            n_labels=len(id2label),
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            download=DOWNLOAD_MODEL,
            pretrained_model_name=cfg.model_name,
            pretrained_model_path=pretrained_model_path,
            tensor_precision=TENSOR_PRECISION,
            num_queries=cfg.num_queries
        )
    # Evaluate model
    model.to(device)
    print(f"Evaluating model: InD {IND_DATASET}, {cfg.model_type} batch_size {cfg.train_batch_size}, Resume {RESUME_TRAINING} "
          f"inference threshold {cfg.inference_threshold}")
    evaluate_model_coco(
        val_dataset,
        val_dataloader,
        device,
        model,
        processor,
        IND_DATASET,
        cfg.inference_threshold,
        cfg.model_type
    )

    # Inference examples
    if INFERENCE_EXAMPLES > 0:
        inference_examples_dir = f"./inference_examples/{IND_DATASET}/{cfg.model_type}_{current_date_time}"
        if not os.path.exists(inference_examples_dir):
            os.makedirs(inference_examples_dir)
        save_inference_examples(
            val_dataset,
            model,
            processor,
            id2label,
            INFERENCE_EXAMPLES,
            inference_examples_dir,
            cfg.inference_threshold,
            IND_DATASET,
            collate_fn,
            cfg.model_type
        )


def get_preprocessor(model_type, model_name, download_model):
    preprocessors_dict = {
        "DETR": DetrImageProcessor,
        "DefDETR": DeformableDetrImageProcessor,
        "RTDETR": RTDetrImageProcessor,
        "OWL": AutoProcessor,
        "CLIP": AutoProcessor
    }
    if download_model:
        processor = preprocessors_dict[model_type].from_pretrained(model_name)
        processor.save_pretrained(f"./saved_models/{model_type}_processor_t0")
    else:
        processor = preprocessors_dict[model_type].from_pretrained(f"saved_models/{model_type}_processor_t0")
    return processor


def draw_n_images_gt_data(gt_data, destination_folder, n_images):
    # based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    image_ids = gt_data.coco.getImgIds()
    cats = gt_data.coco.cats
    # id2label_orig = {k: v['name'] for k, v in cats.items()}
    id2label_minus_one = {k-1: v['name'] for k, v in cats.items()}
    for _ in range(n_images):
        # let's pick a random image
        image_id = [image_ids[np.random.randint(0, len(image_ids))]]
        print('Image n°{}'.format(image_id))
        image, annotations = gt_data.__getitem__(image_id[0])
        save_one_image_train(Image.fromarray(image), image_id[0], annotations["annotations"], destination_folder, id2label_minus_one)

    return id2label_minus_one


def save_one_image_train(image:PIL.Image, image_id, annotations, destination_folder, id2label):
    draw = ImageDraw.Draw(image, "RGBA")

    for annotation in annotations:
        box = annotation['bbox']
        class_idx = annotation['category_id']
        x, y, w, h = tuple(box)
        draw.rectangle((x, y, x + w, y + h), outline='red', width=2)
        draw.text((x, y), id2label[class_idx], fill='white')

    image.save(f"./{destination_folder}/test_img_{IND_DATASET}_{image_id}.jpg")

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def evaluate_model_coco(val_dataset, val_dataloader, device, model, processor, ind_dataset_name, inference_threshold, model_type):
    # initialize evaluator with ground truth (gt)
    evaluator = CocoEvaluator(coco_gt=val_dataset.coco, iou_types=["bbox"])

    print("Running evaluation...")
    for idx, batch in enumerate(tqdm(val_dataloader)):
        # get the inputs
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask=None
        if not model_type == "RTDETR":
            pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        # turn into a list of dictionaries (one item for each example in the batch)
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=inference_threshold)

        # provide to metric
        # metric expects a list of dictionaries, each item
        # containing image_id, category_id, bbox and score keys
        if ind_dataset_name == "bdd":
            predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        else:
            predictions = {target['image_id']: output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)
        if len(predictions) > 0:
            evaluator.update(predictions)
        else:
            print(f"0 predictions in image ids {[label['image_id'] for label in labels]}")

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator


def save_inference_examples(
    val_dataset,
    model,
    processor,
    id2label,
    n_images,
    inference_dir,
    inference_threshold,
    ind_dataset_name,
    collate_fn,
    model_type
):
    assert ind_dataset_name in ["bdd", "voc"]
    viz_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=1, num_workers=10)
    for idx, batch in enumerate(tqdm(viz_dataloader)):
    # for i in range(n_images):
        # We can use the image_id in target to know which image it is
        # pixel_values, target = val_dataset[i]
        # pixel_values = pixel_values.unsqueeze(0).to(device)
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = None
        if not model_type == "RTDETR":
            pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in
              batch["labels"]]  # these are in DETR format, resized + normalized
        with torch.no_grad():
            # forward pass to get class logits and bounding boxes
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        if isinstance(labels[0]['image_id'], str):
            image_id = [labels[0]['image_id']]
        else:
            image_id = labels[0]['image_id'].item()
        # load image based on ID
        image = val_dataset.coco.loadImgs(image_id)[0]
        img_name = image['file_name'].split('.jpg')[0] + f"_infth{str(int(100*inference_threshold))}"
        image = Image.open(
            os.path.join(
                datasets_dict[IND_DATASET]["data_folder"],
                datasets_dict[IND_DATASET]["val_images"],
                image['file_name']
            )
        )

        # postprocess model outputs
        width, height = image.size
        postprocessed_outputs = processor.post_process_object_detection(
            outputs,
            target_sizes=[(height, width)],
            threshold=inference_threshold
        )
        results = postprocessed_outputs[0]
        save_results_one_pred(
            image,
            img_name,
            results['scores'],
            results['labels'],
            results['boxes'],
            id2label,
            inference_dir,
        )

        if idx >= n_images - 1:
            break


if __name__ == '__main__':
    main()

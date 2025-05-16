from typing import List, Union, Tuple

import torch
import pytorch_lightning as pl
from torch import TensorType
from torch.utils.data import DataLoader
from transformers import RTDetrForObjectDetection, CLIPVisionModel
from transformers.image_transforms import center_to_corners_format


class RTDetr(pl.LightningModule):
    def __init__(
            self,
            lr: float,
            lr_backbone: float,
            weight_decay: float,
            n_labels: int,
            train_loader: DataLoader,
            val_loader: DataLoader,
            download: bool,
            pretrained_model_name: str,
            pretrained_model_path: str,
            tensor_precision: torch.dtype,
            **kwargs
    ):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        if download:
            self.model = RTDetrForObjectDetection.from_pretrained(
                pretrained_model_name,
                revision="main",
                num_labels=n_labels,
                ignore_mismatched_sizes=True,
                torch_dtype=tensor_precision,
                **kwargs
            )
            self.model.to(tensor_precision)
            self.model.save_pretrained(pretrained_model_path)
        else:
            self.model = RTDetrForObjectDetection.from_pretrained(pretrained_model_path)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.val_loader = val_loader

    def forward_and_postprocess(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: torch.Tensor,
        orig_sizes: torch.Tensor,
        threshold: float
    ) -> List:
        outputs = self(pixel_values=pixel_values, pixel_mask=pixel_mask)
        results = self.post_process_object_detection(
            outputs,
            target_sizes=orig_sizes,
            threshold=threshold
        )
        return results

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)
        # return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=8,
                    min_lr=1e-7,
                    verbose=True
                ),
                "monitor": "validation_loss",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    @staticmethod
    def post_process_object_detection(
        outputs, threshold: float = 0.5, target_sizes: Union[TensorType, List[Tuple]] = None
    ) -> List:
        """
        Converts the raw output of [`DetrForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes
        # This would be the decoder last hidden state before the final linear layer
        last_hidden_states = outputs.last_hidden_state
        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        prob = torch.nn.functional.sigmoid(out_logits)
        scores, labels = prob.max(-1)

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(out_bbox)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for s, l, b, lg, h in zip(scores, labels, boxes, out_logits, last_hidden_states):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            logit = lg[s > threshold]
            hidden = h[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box, "logits": logit, "last_hidden": hidden})

        return results


class CLIPVision(pl.LightningModule):
    def __init__(
            self,
            download: bool,
            pretrained_model_name: str,
            pretrained_model_path: str,
    ):
        super().__init__()
        if download:
            self.model = CLIPVisionModel.from_pretrained(pretrained_model_name)
            self.model.save_pretrained(pretrained_model_path)
        else:
            self.model = CLIPVisionModel.from_pretrained(pretrained_model_path)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values.to(self.device))

        return outputs


models_dict = {
    "RTDETR": RTDetr,
    "CLIP": CLIPVision
}
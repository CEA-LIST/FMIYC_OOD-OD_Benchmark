# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Based on https://github.com/fregu856/deeplabv3
#    Fabio Arnez, probabilistic adaptation
#    Daniel Montoya
import numpy as np
import pytorch_lightning as pl
from torch.nn.functional import avg_pool2d
from torch.utils.data import DataLoader
from warnings import warn
import torch
from torch import Tensor
from tqdm import tqdm
from typing import Tuple, Union, Any, List, Dict
from numpy import ascontiguousarray, ndarray
from torchvision.ops import roi_align, nms

SUPPORTED_OBJECT_DETECTION_ARCHITECTURES = [
    "yolov8",
    "rcnn",
    "detr-backbone",
    "owlv2",
    "rtdetr-backbone",
    "rtdetr-encoder",
]


class Hook:
    """
    This class will catch the input and output of any torch layer during forward/backward pass.

    Args:
        module (torch.nn.Module): Layer block from Neural Network Module
        backward (bool): backward-poss hook
    """

    def __init__(self, module: torch.nn.Module, backward: bool = False):
        """
        This class will catch the input and output of any torch layer during forward/backward pass.

        Args:
            module (torch.nn.Module): Layer block from Neural Network Module
            backward (bool): backward-poss hook
        """
        assert isinstance(module, torch.nn.Module), "module must be a pytorch module"
        assert isinstance(backward, bool), "backward must be a boolean"
        self.input = None
        self.output = None
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, outputs):
        self.input = inputs
        self.output = outputs

    def close(self):
        self.hook.remove()


def get_mean_or_fullmean_ls_sample(latent_sample: Tensor, method: str = "fullmean") -> Tensor:
    """
    Get either the mean (get a $W \times C$-sized vector) or fullmean (get a $C$-sized vector)
    from the convolutional activation map. (From a $C \times H \times W_sized convolutional
    activation map)

    Args:
        latent_sample: Convolutional activation map
        method: Either 'mean' or 'fullmean'

    Returns:
        The reduced activation map
    """
    assert method in ("mean", "fullmean")
    if method == "mean":
        latent_sample = torch.mean(latent_sample, dim=3, keepdim=True)
        latent_sample = torch.squeeze(latent_sample)
    # fullmean
    else:
        latent_sample = torch.mean(latent_sample, dim=3, keepdim=True)
        latent_sample = torch.mean(latent_sample, dim=2, keepdim=True)
        latent_sample = torch.squeeze(latent_sample)
    return latent_sample


def get_variance_ls_sample(latent_sample: Tensor) -> Tensor:
    """
    Get the variance for each channel from the convolutional activation map.

    Args:
        latent_sample: Convolutional activation map

    Returns:
        The reduced activation map
    """
    latent_sample = torch.var(latent_sample, dim=3, keepdim=True)
    latent_sample = torch.var(latent_sample, dim=2, keepdim=True)
    latent_sample = torch.squeeze(latent_sample)
    return latent_sample


def get_std_ls_sample(latent_sample: Tensor) -> Tensor:
    """
    Get the standard deviation for each channel from the convolutional activation map.

    Args:
        latent_sample: Convolutional activation map

    Returns:
        The reduced activation map
    """
    latent_sample = torch.std(latent_sample, dim=3, keepdim=True)
    latent_sample = torch.std(latent_sample, dim=2, keepdim=True)
    latent_sample = torch.squeeze(latent_sample)
    return latent_sample


class BoxFeaturesExtractor:
    def __init__(
        self,
        model: torch.nn.Module,
        hooked_layers: List[Hook],
        device: torch.device,
        output_sizes: Tuple[int],
        architecture: str,
        sampling_ratio: int = -1,
        return_raw_predictions: bool = False,
        return_stds: bool = False,
        hook_layer_output: bool = False,
        rcnn_extraction_type: str = None,
        extract_noise_entropies: bool = False,
        dropblock_size: int = 0,
        dropblock_prob: float = 0.0,
        n_mcd_reps: int = 1,
    ):
        assert (
            architecture in SUPPORTED_OBJECT_DETECTION_ARCHITECTURES
        ), f"Only {SUPPORTED_OBJECT_DETECTION_ARCHITECTURES} are supported"
        assert rcnn_extraction_type in ("rpn_inter", "rpn_head", "shortcut", "backbone", None)
        self.model = model
        self.hooked_layers = hooked_layers
        self.device = device
        self.architecture = architecture
        self.return_raw_predictions = return_raw_predictions
        self.return_stds = return_stds
        self.hook_layer_output = hook_layer_output
        self.output_sizes = output_sizes
        self.n_hooked_reps = len(output_sizes)
        self.sampling_ratio = sampling_ratio
        self.rcnn_extraction_type = rcnn_extraction_type
        # When hooking the input, a direct layer Hook is expected
        if len(hooked_layers) == 1 and not hook_layer_output:
            self.hooked_layers = self.hooked_layers[0]
        # When hooking output, a list of Hooked layers is expected
        if hook_layer_output and self.rcnn_extraction_type != "rpn_inter":
            assert (
                len(hooked_layers) == self.n_hooked_reps
            ), "Specify an equal number of hooked layers and output sizes"
        # RPN intermediate extraction is tricky, the architecture was modified to catch intermediate
        # representations by using a list stored created during the forward
        # Also the backbone and the RPN final, output a dictionary of five tensors
        if self.architecture == "rcnn" and self.rcnn_extraction_type != "shortcut":
            self.output_sizes = self.output_sizes * 5
            self.n_hooked_reps = 5
        # In case of performing MCD sampling
        self.extract_noise_entropies = extract_noise_entropies
        self.dropblock_sizes = dropblock_size
        self.dropblock_probs = dropblock_prob
        self.n_mcd_reps = n_mcd_reps

    def get_ls_samples(
        self, data_loader: Union[DataLoader, Any], predict_conf=0.25, **kwargs
    ) -> dict:
        if hasattr(data_loader, "batch_sampler"):
            assert data_loader.batch_sampler.batch_size == 1, "Only batch size 1 is supported"
        elif hasattr(data_loader, "batch_size"):
            assert data_loader.batch_size == 1, "Only batch size 1 is supported"
        elif hasattr(data_loader, "bs"):
            assert data_loader.bs == 1, "Only batch size 1 is supported"
        else:
            raise AttributeError(
                "Data loader must have attribute batch size and should be equal to 1"
            )
        results = {}
        no_obj_imgs = []
        if self.return_stds:
            results["stds"] = []
        with torch.no_grad():
            with tqdm(total=len(data_loader), desc="Extracting latent space box samples") as pbar:
                # for impath, image, im_counter in data_loader:
                for loader_contents in data_loader:
                    if self.architecture == "yolov8":
                        (impath, image, im_counter) = loader_contents
                        try:
                            int(impath[0].split("/")[-1].split(".")[0])
                            im_id = impath[0].split("/")[-1].split(".")[0].lstrip("0")
                        except ValueError:
                            im_id = impath[0].split("/")[-1].split(".")[0]

                    elif self.architecture == "rcnn":
                        image = loader_contents
                        impath = [image[0]["file_name"]]
                        im_id = image[0]["image_id"]
                    elif self.architecture == "owlv2":
                        image = (
                            loader_contents["input_ids"].to(self.device),
                            loader_contents["attention_mask"].to(self.device),
                            loader_contents["pixel_values"].to(self.device),
                            loader_contents["orig_size"],
                        )
                        impath = [loader_contents["labels"][0]["image_id"]]
                    # DETR or RTDETR
                    else:
                        image = (
                            loader_contents["pixel_values"].to(self.device),
                            loader_contents["pixel_mask"].to(self.device),
                            torch.stack(
                                [target["orig_size"] for target in loader_contents["labels"]], dim=0
                            ).to(self.device),
                        )
                        impath = [loader_contents["labels"][0]["image_id"]]
                        im_id = loader_contents["labels"][0]["image_id"].item()
                    result_img, found_obj_flag = self._get_samples_one_image(
                        image=image, predict_conf=predict_conf, **kwargs
                    )
                    results[im_id] = {"means": [], "features": [], "logits": [], "boxes": []}
                    if found_obj_flag:
                        for result_type, result_value in result_img.items():
                            results[im_id][result_type].append(result_value)
                    else:
                        # impath is a list, with batch size 1 we only need the first element (the string)
                        no_obj_imgs.append(impath[0])
                    # Update progress bar
                    pbar.update(1)
                for im_id in results.keys():
                    for result_type, result_value in results[im_id].items():
                        results[im_id][result_type] = (
                            torch.cat(result_value, dim=0)
                            if len(result_value) > 0
                            else result_value
                        )
        results["no_obj"] = no_obj_imgs
        # print("Latent representation vector size: ", results["means"].shape[1])
        print(f"No objects in {len(no_obj_imgs)} images")
        return results

    def _get_samples_one_image(
        self, image: Union[Tensor, ndarray], predict_conf: float, **kwargs
    ) -> Tuple[Dict[str, Tensor], bool]:
        # Found objects flag
        found_objs_flag = True
        # Final results dictionary
        results = {}
        if self.architecture == "yolov8":
            img_shape = image[0].shape[:2]  # Height, width
            # Hook the Predict module
            hook_detect = Hook(self.model.model.model._modules["22"])
            # Perform inference just once per image
            pred_img = self.model(image, conf=predict_conf, **kwargs)
            if len(pred_img[0]) > 0:
                activation_detect = hook_detect.output[0]
                results["logits"] = self.yolo_get_logits(
                    prediction=activation_detect,
                    conf_thres=predict_conf,
                    iou_thres=self.model.predictor.args.iou,
                    classes=self.model.predictor.args.classes,
                    agnostic=self.model.predictor.args.agnostic_nms,
                    max_det=self.model.predictor.args.max_det,
                )
                assert len(results["logits"]) == len(pred_img[0])
            boxes = pred_img[0].boxes.xyxy

        elif self.architecture == "rcnn":
            img_shape = image[0]["height"], image[0]["width"]
            pred_img = self.model(image)
            if isinstance(pred_img, list):
                pred_img = pred_img[0]
            if isinstance(pred_img, dict):
                pred_img = pred_img["instances"]
            # The output of the rcnn seems to be already in the format xyxy
            boxes = pred_img.pred_boxes.tensor
            if "latent_feature" in pred_img._fields.keys():
                # Store previous-to-last features
                results["features"] = pred_img.latent_feature
            if "inter_feat" in pred_img._fields.keys():
                results["logits"] = pred_img.inter_feat
            elif "logits" in pred_img._fields.keys():
                results["logits"] = pred_img.logits

        elif self.architecture == "owlv2":
            img_shape = image[3][0]
            pred_img = self.model.forward_and_postprocess(
                input_ids=image[0],
                attention_mask=image[1],
                pixel_values=image[2],
                orig_sizes=image[3],
                threshold=predict_conf,
            )[
                0
            ]  # Batch size 1, therefore just one image
            boxes = pred_img["boxes"]
            results["features"] = pred_img["last_hidden"]
            results["logits"] = pred_img["logits"]
        # DETR or RTDETR
        else:
            img_shape = (image[2][0][0].item(), image[2][0][1].item())
            # Made a custom function on the Detr class to handle this input
            pred_img = self.model.forward_and_postprocess(
                pixel_values=image[0],
                pixel_mask=image[1],
                orig_sizes=image[2],
                threshold=predict_conf,
            )[
                0
            ]  # Batch size 1, therefore just one image
            boxes = pred_img["boxes"]
            results["features"] = pred_img["last_hidden"]
            results["logits"] = pred_img["logits"]
        n_detected_objects = boxes.shape[0]
        if n_detected_objects == 0:
            # return None
            # Get whole image as single object if no objects are detected
            boxes = Tensor([0.0, 0.0, img_shape[1], img_shape[0]]).reshape(1, -1).to(self.device)
            n_detected_objects = 1
            found_objs_flag = False
        # Catch the latent activations
        if self.architecture == "rcnn" and self.rcnn_extraction_type == "rpn_inter":
            # Ugly but still a simple solution for a complex architecture
            if hasattr(self.model, "model"):
                latent_mcd_sample = (
                    self.model.model.proposal_generator.rpn_head.rpn_intermediate_output
                )
            else:
                latent_mcd_sample = self.model.proposal_generator.rpn_head.rpn_intermediate_output

        # Yolo, DETR, or other rcnn locations
        else:
            if self.hook_layer_output:
                latent_mcd_sample = [layer.output for layer in self.hooked_layers]
            else:
                latent_mcd_sample = self.hooked_layers.input
                # Input might be a one-element tuple, containing the desired list
                if len(latent_mcd_sample) == 1 and self.n_hooked_reps != 1:
                    try:
                        assert len(latent_mcd_sample[0]) == self.n_hooked_reps
                        latent_mcd_sample = latent_mcd_sample[0]
                    except AssertionError:
                        print("Cannot find a suitable latent space sample")
        # Check if rcnn backbone output
        if (
            self.architecture == "rcnn"
            and len(latent_mcd_sample) == 1
            and isinstance(latent_mcd_sample[0], dict)
            and self.rcnn_extraction_type == "backbone"
        ):
            latent_mcd_sample = [v for k, v in latent_mcd_sample[0].items()]
        # Check for rpn_head output extraction
        if (
            self.architecture == "rcnn"
            and len(latent_mcd_sample) == 1
            and isinstance(latent_mcd_sample[0], tuple)
            and len(latent_mcd_sample[0]) == 2
            and self.rcnn_extraction_type == "rpn_head"
        ):
            latent_mcd_sample = [
                torch.cat([obj_logit, anch_delta], dim=1)
                for obj_logit, anch_delta in zip(latent_mcd_sample[0][0], latent_mcd_sample[0][1])
            ]
        if self.architecture == "owlv2":
            latent_mcd_sample = [
                latent_mcd_sample[0][0][:, 1:, :].reshape(
                    1,
                    self.model.model.config.vision_config.hidden_size,
                    int(
                        self.model.model.config.vision_config.image_size
                        / self.model.model.config.vision_config.patch_size
                    ),
                    int(
                        self.model.model.config.vision_config.image_size
                        / self.model.model.config.vision_config.patch_size
                    ),
                )
            ]
        if self.architecture == "rtdetr-encoder":
            latent_mcd_sample = [
                latent_mcd_sample[0][0].permute(0, 2, 1).reshape(-1, 256, 20, 20).contiguous()
            ]
        # Deterministic algorithm
        n_objects_means, n_objects_stds = reduce_features_to_rois(
            latent_mcd_sample=latent_mcd_sample,
            output_sizes=self.output_sizes,
            boxes=boxes,
            img_shape=img_shape,
            sampling_ratio=self.sampling_ratio,
            n_hooked_reps=self.n_hooked_reps,
            n_detected_objects=n_detected_objects,
            return_stds=self.return_stds,
        )
        results["means"] = torch.cat(n_objects_means, dim=0)
        results["boxes"] = boxes
        if self.return_stds:
            results["stds"] = torch.cat(n_objects_stds, dim=0)
        if self.return_raw_predictions:
            results["raw_preds"] = pred_img
        return results, found_objs_flag

    @staticmethod
    def yolo_get_logits(
        prediction,
        conf_thres,
        iou_thres,
        classes=None,
        agnostic=False,
        multi_label=False,
        # labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        # max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
        # in_place=True,
        # rotated=False,
    ):
        # Checks
        assert (
            0 <= conf_thres <= 1
        ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

        # Settings
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
        # if not rotated:
        #     if in_place:
        #         prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        #     else:
        #         prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x.split((4, nc, nm), 1)

            if multi_label:
                i, j = torch.where(cls > conf_thres)
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[
                    x[:, 4].argsort(descending=True)[:max_nms]
                ]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            scores = x[:, 4]  # scores

            boxes = x[:, :4] + c  # boxes (offset by class)
            i = nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections

            output[xi] = torch.log(cls[i])

        return torch.cat(output, dim=0)


class BoxFeaturesExtractorAnomalyLoader(BoxFeaturesExtractor):
    def get_ls_samples(
        self, data_loader: Union[DataLoader, Any], predict_conf=0.25, **kwargs
    ) -> dict:
        results = {"means": []}
        no_obj_imgs = 0
        if self.return_stds:
            results["stds"] = []
        with torch.no_grad():
            with tqdm(total=len(data_loader), desc="Extracting latent space box samples") as pbar:
                for image, label in data_loader:
                    # Here, a BGR 2 RGB inversion is performed, since the torch Dataloader seems to feed yolo
                    # Images in the wrong ordering
                    image = [ascontiguousarray(image[0].numpy().transpose(1, 2, 0)[..., ::-1])]
                    result_img, found_obj_flag = self._get_samples_one_image(
                        image=image, predict_conf=predict_conf
                    )
                    for result_type, result_value in result_img.items():
                        results[result_type].append(result_value)
                    if not found_obj_flag:
                        # impath is a list, with batch size 1 we only need the first element (the string)
                        no_obj_imgs += 1
                    # Update progress bar
                    pbar.update(1)
                for result_type, result_value in results.items():
                    results[result_type] = torch.cat(result_value, dim=0)
        results["no_obj"] = no_obj_imgs
        print("Latent representation vector size: ", results["means"].shape[1])
        print(f"No objects in {no_obj_imgs} images")
        return results


def reduce_features_to_rois(
    latent_mcd_sample: List[Tensor],
    output_sizes: Tuple[int],
    boxes: Tensor,
    img_shape: Tuple[int, ...],
    sampling_ratio: int,
    n_hooked_reps: int,
    n_detected_objects: int,
    return_stds: bool = False,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    This function takes as input the bounding boxes predictions, the latent representations, and the output sizes
    to obtain the means and optionally the variances of these representations
    Args:
        latent_mcd_sample: The latent representation samples
        output_sizes: Tuple of output sizes as integers
        boxes: Box predictions in the format xyxy
        img_shape: Tuple with the image shape
        sampling_ratio: Roi Align sampling ratio
        n_hooked_reps: Number of hooked latent representations
        n_detected_objects: Number of detected objects in the image
        return_stds: Whether to return standard deviations

    Returns:
        A tuple with the means and standard deviations of rois
    """
    # Extract latent space ROIs
    # Several representations per layer
    rois = [
        roi_align(
            latent_mcd_sample[i],
            [boxes],
            output_size=output_sizes[i],
            spatial_scale=latent_mcd_sample[i].shape[3] / img_shape[1],
            sampling_ratio=sampling_ratio,
            aligned=True,
        )
        for i in range(n_hooked_reps)
    ]
    # Get means and optionally stds from rois
    n_objects_means = []
    n_objects_stds = []

    for i in range(n_detected_objects):
        object_i_latent_means = []
        object_i_latent_stds = []
        for j in range(n_hooked_reps):
            object_i_latent_means.append(torch.mean(rois[j][i], dim=(1, 2)).reshape(-1))
            if return_stds:
                object_i_latent_stds.append(torch.std(rois[j][i], dim=(1, 2)).reshape(-1))
        n_objects_means.append(torch.cat(object_i_latent_means, dim=0).reshape(1, -1))
        if return_stds:
            n_objects_stds.append(torch.cat(object_i_latent_stds, dim=0).reshape(1, -1))

    return n_objects_means, n_objects_stds


def get_aggregated_data_dict(
    data_dict: Dict,
    dataset_name: str,
    aggregated_data_dict: Dict,
    no_obj_dict: Dict,
    non_empty_predictions_ids: Dict,
) -> Tuple[Dict, Dict, Dict]:
    if "no_obj" in data_dict[dataset_name].keys():
        no_obj_dict[dataset_name] = data_dict[dataset_name].pop("no_obj")
    all_features = []
    for im_results in data_dict[f"{dataset_name}"].values():
        if len(im_results["features"]) > 0:
            all_features.append(im_results["features"])
    if len(all_features) > 0:
        aggregated_data_dict[f"{dataset_name} features"] = (
            torch.cat(all_features, dim=0).cpu().numpy()
        )

    all_logits = []
    for im_results in data_dict[f"{dataset_name}"].values():
        if len(im_results["logits"]) > 0:
            all_logits.append(im_results["logits"])
    if len(all_logits) > 0:
        aggregated_data_dict[f"{dataset_name} logits"] = torch.cat(all_logits, dim=0).cpu().numpy()

    all_means = []
    non_empty_predictions_ids[dataset_name] = []
    for im_id, im_results in data_dict[f"{dataset_name}"].items():
        if len(im_results["means"]) > 0:
            all_means.append(im_results["means"])
            non_empty_predictions_ids[dataset_name].extend([im_id] * len(im_results["means"]))
    aggregated_data_dict[dataset_name] = torch.cat(all_means, dim=0).cpu().numpy()
    return aggregated_data_dict, no_obj_dict, non_empty_predictions_ids


def associate_precalculated_baselines_with_raw_predictions(
    data_dict: Dict[str, Dict[str, torch.Tensor]],
    dataset_name: str,
    ood_baselines_dict: Dict[str, np.ndarray],
    baselines_names: List[str],
    non_empty_ids: List[str],
    is_ood: bool,
):
    for idx, im_id in enumerate(non_empty_ids):
        for baseline_name in baselines_names:
            if not baseline_name in data_dict[im_id].keys():
                data_dict[im_id][baseline_name] = []
            if is_ood:
                data_dict[im_id][baseline_name].append(
                    ood_baselines_dict[f"{dataset_name} {baseline_name}"][idx]
                )
            else:
                data_dict[im_id][baseline_name].append(ood_baselines_dict[f"{baseline_name}"][idx])
    return data_dict

import json
from collections import defaultdict
from typing import Dict, List, Optional, Union

import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import softmax
import numpy as np
import torch
from tqdm import tqdm


class OpenSetEvaluator:
    def __init__(self, id_dataset_name, ground_truth_annotations_path, metric_2007: bool):
        ground_truth_annotations = COCOParser(ground_truth_annotations_path)
        # self.output_dir = output_dir
        self._dataset_name = id_dataset_name
        self._class_names = [cat["name"] for cat in ground_truth_annotations.cat_dict.values()] + [
            "unknown"
        ]
        self.total_num_class = len(ground_truth_annotations.cat_dict) + 1
        self.unknown_class_index = self.total_num_class - 1
        self.num_known_classes = len(ground_truth_annotations.cat_dict)
        self.known_classes = self._class_names[: self.num_known_classes]
        self._is_2007 = metric_2007

    def reset(self):
        # class name -> list of prediction strings
        self._predictions = defaultdict(list)

    def process(self, image_id, boxes, scores, classes):
        for box, score, cls in zip(boxes, scores, classes):
            xmin, ymin, xmax, ymax = box
            # The inverse of data loading logic in `datasets/pascal_voc.py`
            xmin += 1
            ymin += 1
            self._predictions[cls].append(
                f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
            )

    def evaluate(
        self,
        test_annotations_path: str,
        is_ood: bool,
        get_known_classes_metrics: bool,
        using_subset: Optional[List[Union[str, int]]] = False,
    ) -> Dict[str, float]:
        """
        Returns:
            dict: A dict of "AP", "AP50", and "AP75".
        """
        # Read annotations file
        test_annotations = COCOParser(test_annotations_path, using_subset)
        # Get the predictions per class
        predictions = defaultdict(list)
        for clsid, lines in self._predictions.items():
            predictions[clsid].extend(lines)

        aps = defaultdict(list)  # iou -> ap per class
        recs = defaultdict(list)
        precs = defaultdict(list)
        all_recs = defaultdict(list)
        all_precs = defaultdict(list)
        unk_det_as_knowns = defaultdict(list)
        num_unks = defaultdict(list)
        tp_plus_fp_cs = defaultdict(list)
        fp_os = defaultdict(list)

        for cls_id, cls_name in enumerate(self._class_names):
            lines = predictions.get(cls_id, [""])

            for thresh in [
                50,
            ]:
                # for thresh in range(50, 100, 5):
                (
                    rec,
                    prec,
                    ap,
                    unk_det_as_known,
                    num_unk,
                    tp_plus_fp_closed_set,
                    fp_open_set,
                ) = voc_eval(
                    lines,
                    test_annotations,
                    cls_name,
                    ovthresh=thresh / 100.0,
                    use_07_metric=self._is_2007,
                    is_ood=is_ood,
                )
                aps[thresh].append(ap * 100)
                unk_det_as_knowns[thresh].append(unk_det_as_known)
                num_unks[thresh].append(num_unk)
                all_precs[thresh].append(prec)
                all_recs[thresh].append(rec)
                tp_plus_fp_cs[thresh].append(tp_plus_fp_closed_set)
                fp_os[thresh].append(fp_open_set)
                try:
                    recs[thresh].append(rec[-1] * 100)
                    precs[thresh].append(prec[-1] * 100)
                except:
                    recs[thresh].append(0)
                    precs[thresh].append(0)

        results_2d = {}
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        if get_known_classes_metrics:
            results_2d["mAP"] = mAP[50]

        wi = self.compute_WI_at_many_recall_level(all_recs, tp_plus_fp_cs, fp_os)
        results_2d["WI"] = wi[0.8][50] * 100

        total_num_unk_det_as_known = {iou: np.sum(x) for iou, x in unk_det_as_knowns.items()}
        # total_num_unk = num_unks[50][0]
        # self.logger.info('num_unk ' + str(total_num_unk))
        results_2d["AOSE"] = total_num_unk_det_as_known[50]
        if num_unk > 0:
            results_2d["nOSE"] = round(total_num_unk_det_as_known[50] * 100 / num_unk, 3)
        else:
            results_2d["nOSE"] = 0.0

        # Known
        if get_known_classes_metrics:
            results_2d.update(
                {
                    "AP_K": np.mean(aps[50][: self.num_known_classes]),
                    "P_K": np.mean(precs[50][: self.num_known_classes]),
                    "R_K": np.mean(recs[50][: self.num_known_classes]),
                }
            )

        # Unknown
        results_2d.update(
            {
                "AP_U": np.mean(aps[50][-1]),
                "P_U": np.mean(precs[50][-1]),
                "R_U": np.mean(recs[50][-1]),
            }
        )
        results_head = list(results_2d.keys())
        results_data = [[float(results_2d[k]) for k in results_2d]]

        return {metric: round(x, 3) for metric, x in zip(results_head, results_data[0])}

    def compute_WI_at_many_recall_level(self, recalls, tp_plus_fp_cs, fp_os):
        wi_at_recall = {}
        # for r in range(1, 10):
        for r in [8]:
            r = r / 10
            wi = self.compute_WI_at_a_recall_level(recalls, tp_plus_fp_cs, fp_os, recall_level=r)
            wi_at_recall[r] = wi
        return wi_at_recall

    def compute_WI_at_a_recall_level(self, recalls, tp_plus_fp_cs, fp_os, recall_level=0.5):
        wi_at_iou = {}
        for iou, recall in recalls.items():
            tp_plus_fps = []
            fps = []
            for cls_id, rec in enumerate(recall):
                if cls_id in range(self.num_known_classes) and len(rec) > 0:
                    index = min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))
                    tp_plus_fp = tp_plus_fp_cs[iou][cls_id][index]
                    tp_plus_fps.append(tp_plus_fp)
                    fp = fp_os[iou][cls_id][index]
                    fps.append(fp)
            if len(tp_plus_fps) > 0:
                wi_at_iou[iou] = np.mean(fps) / np.mean(tp_plus_fps)
            else:
                wi_at_iou[iou] = 0
        return wi_at_iou


def voc_eval(
    predictions_par_class,
    test_annotations,
    classname,
    ovthresh=0.5,
    use_07_metric=True,
    is_ood=True,
):
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    # if not is_ood:
    for imagename in test_annotations.annIm_dict.keys():
        # If is_ood, all objects in dataset are ood
        if is_ood:
            if classname == "unknown":
                R = [obj for obj in test_annotations.annIm_dict[imagename]]
            else:
                R = []
        else:
            R = [
                obj
                for obj in test_annotations.annIm_dict[imagename]
                if test_annotations.cat_dict[obj["category_id"]]["name"] == classname
            ]
        bbox = np.array([convert_xywh_to_xyxy(x["bbox"]) for x in R])
        difficult = np.array([False for x in R]).astype(bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        if isinstance(imagename, int):
            imagename = str(imagename)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    splitlines = [x.strip().split(" ") for x in predictions_par_class]
    image_ids = [x[0] for x in splitlines]
    # If there exists detections for this class
    if len(image_ids[0]) > 0:
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
    else:
        image_ids = []

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        if image_ids[d] in class_recs.keys():
            R = class_recs[image_ids[d]]
        else:
            continue
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            overlaps = compute_overlaps(BBGT, bb)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    if npos > 0:
        rec = tp / float(npos)
    elif npos == 0:
        rec = tp
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    # compute unknown det as known
    unknown_class_recs = {}
    n_unk = 0
    for imagename in test_annotations.annIm_dict.keys():
        # If is_ood, all objects in dataset are ood
        if is_ood:
            R = [obj for obj in test_annotations.annIm_dict[imagename]]
        else:
            R = [
                obj
                for obj in test_annotations.annIm_dict[imagename]
                if test_annotations.cat_dict[obj["category_id"]]["name"] == "unknown"
            ]
        bbox = np.array([convert_xywh_to_xyxy(x["bbox"]) for x in R])
        difficult = np.array([False for x in R]).astype(bool)
        det = [False] * len(R)
        n_unk = n_unk + sum(~difficult)
        if isinstance(imagename, int):
            imagename = str(imagename)
        unknown_class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    if classname == "unknown":
        return rec, prec, ap, 0, n_unk, None, None

    # Go down each detection and see if it has an overlap with an unknown object.
    # If so, it is an unknown object that was classified as known.
    is_unk = np.zeros(nd)
    for d in range(nd):
        if image_ids[d] in unknown_class_recs.keys():
            R = unknown_class_recs[image_ids[d]]
        else:
            continue
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            overlaps = compute_overlaps(BBGT, bb)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            is_unk[d] = 1.0

    is_unk_sum = np.sum(is_unk)
    tp_plus_fp_closed_set = tp + fp
    fp_open_set = np.cumsum(is_unk)

    return rec, prec, ap, is_unk_sum, n_unk, tp_plus_fp_closed_set, fp_open_set


def compute_overlaps(BBGT, bb):
    # compute overlaps
    # intersection
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    uni = (
        (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
        + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
        - inters
    )

    return inters / uni


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_open_set_detection_one_method(
    id_dataset_name: str,
    id_gt_annotations_path: str,
    predictions_dict: Dict,
    method_name: str,
    threshold: float,
    test_gt_annotations_path: str,
    metric_2007: bool,
    evaluating_ood: bool,
    get_known_classes_metrics: bool,
    using_subset: Optional[List[Union[str, int]]] = False,
) -> Dict[str, float]:
    evaluator = OpenSetEvaluator(id_dataset_name, id_gt_annotations_path, metric_2007=metric_2007)
    evaluator.reset()
    for im_id, im_pred in predictions_dict.items():
        if len(im_pred["boxes"]) > 0:
            labels, scores = get_labels_and_scores_from_logits(im_pred["logits"])
            boxes = get_boxes_from_precalculated(im_pred["boxes"])
            # Postprocess according to score and threshold
            unk_boxes = np.where(predictions_dict[im_id][method_name] < threshold)
            labels[unk_boxes] = evaluator.unknown_class_index
            # Add results to evaluator
            evaluator.process(im_id, boxes, scores, labels)
    evaluation_results = evaluator.evaluate(
        test_gt_annotations_path,
        is_ood=evaluating_ood,
        get_known_classes_metrics=get_known_classes_metrics,
        using_subset=using_subset,
    )
    return evaluation_results


def get_boxes_from_precalculated(boxes):
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    elif isinstance(boxes, np.ndarray):
        boxes = boxes
    elif isinstance(boxes, list):
        boxes = np.array(boxes)
    else:
        raise ValueError("boxes must be a torch.Tensor, np.ndarray or list")
    return boxes


def get_labels_and_scores_from_logits(logits):
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    elif isinstance(logits, np.ndarray):
        logits = logits
    elif isinstance(logits, list):
        logits = np.array(logits)
    else:
        raise ValueError("logits must be a torch.Tensor, np.ndarray or list")
    # if logits.shape[1] == 21 or logits.shape[1] == 11:
    #     logits = logits[:, :-1]
    scores = softmax(logits, axis=-1).max(axis=-1)
    pred_classes = np.argmax(logits, axis=-1)
    return pred_classes, scores


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


def convert_xywh_to_xyxy(bbox):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]


def get_overall_open_set_results(
    ind_dataset_name: str,
    ind_gt_annotations_path: str,
    ind_data_dict: Dict,
    ood_data_dict: Dict,
    ood_datasets_names: List[str],
    ood_annotations_paths: Dict[str, str],
    methods_names: List[str],
    methods_thresholds: Dict[str, float],
    metric_2007: bool,
    evaluate_on_ind: bool,
    get_known_classes_metrics: bool,
    using_id_val_subset: Optional[List[Union[str, int]]] = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    open_set_results = {}
    if evaluate_on_ind:
        open_set_results[ind_dataset_name] = {}
        for baseline_name in methods_names:
            open_set_results[ind_dataset_name][
                baseline_name
            ] = evaluate_open_set_detection_one_method(
                id_dataset_name=ind_dataset_name,
                id_gt_annotations_path=ind_gt_annotations_path,
                predictions_dict=ind_data_dict["valid"],
                method_name=baseline_name,
                threshold=methods_thresholds[baseline_name],
                test_gt_annotations_path=ind_gt_annotations_path,
                metric_2007=metric_2007,
                evaluating_ood=False,
                get_known_classes_metrics=True,
                using_subset=using_id_val_subset,
            )
    for ood_dataset_name in tqdm(
        ood_datasets_names, desc=f"Evaluating OSOD on OOD datasets {ood_datasets_names}"
    ):
        open_set_results[ood_dataset_name] = {}
        for baseline_name in methods_names:
            open_set_results[ood_dataset_name][
                baseline_name
            ] = evaluate_open_set_detection_one_method(
                id_dataset_name=ind_dataset_name,
                id_gt_annotations_path=ind_gt_annotations_path,
                predictions_dict=ood_data_dict[ood_dataset_name],
                method_name=baseline_name,
                threshold=methods_thresholds[baseline_name],
                test_gt_annotations_path=ood_annotations_paths[ood_dataset_name],
                metric_2007=metric_2007,
                evaluating_ood=True,
                get_known_classes_metrics=get_known_classes_metrics,
            )
    return open_set_results


def convert_osod_results_to_pandas_df(
    open_set_results: Dict[str, Dict[str, float]],
    methods_names: List[str],
    save_method_as_data: bool,
):
    if save_method_as_data:
        col_names = ["Method"] + list(open_set_results[list(open_set_results.keys())[0]].keys())
    else:
        col_names = list(open_set_results[list(open_set_results.keys())[0]].keys())
    new_dict = {}
    for method_name in methods_names:
        if save_method_as_data:
            new_dict[method_name] = [method_name] + list(open_set_results[method_name].values())
        else:
            new_dict[method_name] = list(open_set_results[method_name].values())
    df = pd.DataFrame.from_dict(new_dict, orient="index", columns=col_names)
    return df


def convert_osod_results_to_hierarchical_pandas_df(
    osod_results_a: Dict[str, Dict[str, float]],
    osod_results_b: Dict[str, Dict[str, float]],
    methods_names: List[str],
    save_method_as_data: bool,
    datasets_names: List[str],
):
    if save_method_as_data:
        col_names = ["Method"] + list(osod_results_a[list(osod_results_a.keys())[0]].keys())
    else:
        col_names = list(osod_results_a[list(osod_results_a.keys())[0]].keys())
    columns = pd.MultiIndex.from_product([datasets_names, col_names], names=["Dataset", "Metric"])
    new_dict = {}
    for method_name in methods_names:
        if save_method_as_data:
            new_dict[method_name] = (
                [method_name]
                + list(osod_results_a[method_name].values())
                + list(osod_results_b[method_name].values())
            )
        else:
            new_dict[method_name] = list(osod_results_a[method_name].values()) + list(
                osod_results_b[method_name].values()
            )
    df = pd.DataFrame.from_dict(new_dict, orient="index", columns=columns)
    return df


def plot_two_osod_datasets_metrics(
    osod_results_a: Dict[str, Dict[str, float]],
    osod_results_b: Dict[str, Dict[str, float]],
    methods_names: List[str],
    datasets_names: List[str],
    metrics_to_plot: List[str],
    show_plot: bool,
):
    x = np.arange(len(metrics_to_plot))  # the label locations
    width = 1 / (len(methods_names) * 2 + 1)  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained", figsize=(4 * len(methods_names), 6))

    for method in methods_names:
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            osod_results_a.loc[method][metrics_to_plot],
            width,
            label=f"{method} {datasets_names[0]}",
        )
        ax.bar_label(rects, padding=3, fontsize=8, fmt="%.2f")
        multiplier += 1
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            osod_results_b.loc[method][metrics_to_plot],
            width,
            label=f"{method} {datasets_names[1]}",
        )
        ax.bar_label(rects, padding=3, fontsize=8, fmt="%.2f")
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Percentage")
    ax.set_title(f"OSOD metrics for {datasets_names[0]} and {datasets_names[1]}")
    ax.set_xticks(x + 0.5 - 0.5 * width, metrics_to_plot)
    # ax.legend(loc='upper left', ncols=3)
    ax.legend(ncols=max(1, int(len(methods_names) / 3)))
    ax.set_ylim(0, 100)
    if show_plot:
        plt.show()
    return fig


def plot_two_osod_datasets_per_metric(
    osod_results_a: Dict[str, Dict[str, float]],
    osod_results_b: Dict[str, Dict[str, float]],
    methods_names: List[str],
    datasets_names: List[str],
    metric_to_plot: str,
    show_plot: bool,
):
    # colors = ['tab:blue', 'tab:orange']
    x = np.arange(len(methods_names))  # the label locations
    width = 1 / (len(datasets_names) + 1)  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained", figsize=(1.5 * len(methods_names), 5))
    ax.grid(axis="y", linestyle="--")
    for dataset, dataset_name in zip([osod_results_a, osod_results_b], datasets_names):
        offset = width * multiplier
        rects = ax.bar(x + offset, dataset[metric_to_plot], width, label=f"{dataset_name}")
        ax.bar_label(rects, padding=3, fontsize=8, fmt="%.2f")
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Percentage")
    ax.set_title(f"OSOD {metric_to_plot} for {datasets_names[0]} and {datasets_names[1]}")
    ax.set_xticks(x + 0.5 - width, methods_names)
    # ax.legend(loc='upper left', ncols=3)
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[-2:], labels[-2:], frameon=True)

    ax.legend()
    ax.set_ylim(0, 100)
    if show_plot:
        plt.show()
    return fig


def convert_osod_results_for_mlflow_logging(
    open_set_results: Dict[str, Dict[str, Dict[str, float]]],
    ood_datasets_names: List[str],
    methods_names: List[str],
) -> Dict[str, float]:
    results_for_mlflow = {}
    for ood_dataset_name in ood_datasets_names:
        for baseline_name in methods_names:
            for metric_name, value in open_set_results[ood_dataset_name][baseline_name].items():
                results_for_mlflow[f"{ood_dataset_name} {baseline_name} {metric_name}"] = value
    return results_for_mlflow


def get_n_unk_ood_dataset(annotations_path: COCOParser):
    annotations = COCOParser(annotations_path)
    im_ids = annotations.get_imgIds()
    ann_ids = annotations.get_annIds(im_ids)
    return len(ann_ids)

from typing import Dict, List
import hydra
import mlflow
import numpy as np
import torch
from ood_detect_fns.metrics import subset_boxes
from ood_detect_fns.rcnn import remove_background_dimension
from ood_detect_fns.uncertainty_estimation import remove_latent_features, calculate_all_baselines, \
    get_baselines_thresholds, get_aggregated_data_dict, associate_precalculated_baselines_with_raw_predictions
from ood_detect_fns.uncertainty_estimation.evaluation import log_evaluate_larex
from ood_detect_fns.uncertainty_estimation.open_set_evaluation import \
    get_overall_open_set_results, convert_osod_results_for_mlflow_logging, convert_osod_results_to_pandas_df, \
    plot_two_osod_datasets_metrics, plot_two_osod_datasets_per_metric
from omegaconf import DictConfig
from os.path import join as op_join
import warnings

# Filter the append warning from pandas
warnings.simplefilter(action="ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASELINES_NAMES = ["vim", "mdist", "msp", "knn", "energy", "ash", "dice", "react", "gen", "dice_react", "ddu"]
POSTPROCESSORS = ["KNN", "MD", "GMM"]
VISUALIZE_POSTPROCESSOR = "MD"
# OSOD evaluation parameters
GET_OSOD_PLOTS = True
BASE_OOD_DATASETS = ["coco", "oi"]
OSOD_METRICS_TO_PLOT = ["nOSE", "AP_U", "P_U", "R_U"]

config_file = "config_boxes.yaml"


@hydra.main(version_base=None, config_path="configs/Boxes-evaluation", config_name=config_file)
def main(cfg: DictConfig):
    print_config(cfg)
    #################################################################
    # Load precalculated data
    #################################################################
    latent_samples_folder = (f"./Extracted_latent_samples/boxes/ind_{cfg.ind_dataset}/"
                             f"{'_'.join(map(str, cfg.hooked_modules))}")
    # Last layer parameters for ViM calculations
    fc_params = torch.load(
        f=op_join(latent_samples_folder, f"fc_params{'_vos' if cfg.using_vos else ''}{'_regnet' if cfg.using_regnet else ''}.pt"), map_location="cpu"
    )

    # InD
    ind_data_splits = ["test", "valid"]
    # In this case the validation set acts as the training set, and the test set as the valid set for ID samples
    correspondent_splits = {
        "valid": "train",
        "test": "valid",
    }
    ind_data_dict: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = dict()
    aggregated_ind_data_dict: Dict[str, np.ndarray] = dict()
    non_empty_preds_ind_im_ids: Dict[str, List[str]] = dict()
    # Track images with no objects found from varying the confidence of predictions
    ind_no_obj: Dict[str, np.ndarray] = dict()
    for file_split in ind_data_splits:
        split = correspondent_splits[file_split]
        ind_file_name = get_ind_file_name(cfg, file_split)
        # Load InD latent space activations
        ind_data_dict[f"{split}"] = torch.load(
            f=op_join(latent_samples_folder, ind_file_name), map_location=device
        )
        aggregated_ind_data_dict, ind_no_obj, non_empty_preds_ind_im_ids = get_aggregated_data_dict(
            data_dict=ind_data_dict,
            dataset_name=split,
            aggregated_data_dict=aggregated_ind_data_dict,
            no_obj_dict=ind_no_obj,
            non_empty_predictions_ids=non_empty_preds_ind_im_ids
        )

    # OoD
    ood_data_dict: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = dict()
    aggregated_ood_data_dict: Dict[str, np.ndarray] = dict()
    non_empty_preds_ood_im_ids: Dict[str, List[str]] = dict()
    # Track images with no objects found from varying the confidence of predictions
    ood_no_obj: Dict[str, np.ndarray] = dict()
    for ood_dataset_name in cfg.ood_datasets:
        ood_file_name = get_ood_file_name(cfg, ood_dataset_name)
        # Load OoD latent space activations
        ood_data_dict[f"{ood_dataset_name}"] = torch.load(
            f=op_join(latent_samples_folder, ood_file_name), map_location=device
        )
        aggregated_ood_data_dict, ood_no_obj, non_empty_preds_ood_im_ids = get_aggregated_data_dict(
            data_dict=ood_data_dict,
            dataset_name=ood_dataset_name,
            aggregated_data_dict=aggregated_ood_data_dict,
            no_obj_dict=ood_no_obj,
            non_empty_predictions_ids=non_empty_preds_ood_im_ids
        )
    ###########################################################################
    # Calculate baselines
    ###########################################################################
    if cfg.vim_remove_background_dimension:
        aggregated_ind_data_dict, aggregated_ood_data_dict, fc_params = remove_background_dimension(
            fc_params=fc_params,
            ind_data_dict=aggregated_ind_data_dict,
            ood_data_dict=aggregated_ood_data_dict,
            ood_names=cfg.ood_datasets,
        )
    # Subset boxes since they are a huge number!
    aggregated_ind_data_dict, aggregated_ood_data_dict, non_empty_preds_ind_im_ids, non_empty_preds_ood_im_ids = subset_boxes(
        ind_dict=aggregated_ind_data_dict,
        ood_dict=aggregated_ood_data_dict,
        ind_train_limit=cfg.ind_train_boxes_max,
        ood_limit=cfg.ood_boxes_max,
        random_seed=cfg.random_seed,
        ood_names=cfg.ood_datasets,
        non_empty_predictions_id=non_empty_preds_ind_im_ids,
        non_empty_predictions_ood=non_empty_preds_ood_im_ids,
    )
    aggregated_ind_data_dict, aggregated_ood_data_dict, ood_baselines_scores_dict = calculate_all_baselines(
        baselines_names=BASELINES_NAMES,
        ind_data_dict=aggregated_ind_data_dict,
        ood_data_dict=aggregated_ood_data_dict,
        fc_params=fc_params,
        cfg=cfg,
        num_classes=10 if cfg.ind_dataset == "bdd" else 20
    )
    aggregated_ind_data_dict, aggregated_ood_data_dict = remove_latent_features(
        id_data=aggregated_ind_data_dict,
        ood_data=aggregated_ood_data_dict,
        ood_names=cfg.ood_datasets
    )
    baselines_thresholds = get_baselines_thresholds(
        baselines_names=BASELINES_NAMES,
        baselines_scores_dict=aggregated_ind_data_dict,
        z_score_percentile=cfg.z_score_thresholds
    )
    # Associate calculated baselines scores with raw predictions dicts
    # OOD
    for ood_dataset_name in cfg.ood_datasets:
        ood_data_dict[ood_dataset_name] = associate_precalculated_baselines_with_raw_predictions(
            data_dict=ood_data_dict[ood_dataset_name],
            dataset_name=ood_dataset_name,
            ood_baselines_dict=ood_baselines_scores_dict,
            baselines_names=BASELINES_NAMES,
            non_empty_ids=non_empty_preds_ood_im_ids[ood_dataset_name],
            is_ood=True
        )
    # InD
    ind_data_dict["valid"] = associate_precalculated_baselines_with_raw_predictions(
        data_dict=ind_data_dict["valid"],
        dataset_name="valid",
        ood_baselines_dict=aggregated_ind_data_dict,
        baselines_names=BASELINES_NAMES,
        non_empty_ids=non_empty_preds_ind_im_ids["valid"],
        is_ood=False
    )

    #######################################################################
    # Setup MLFLow
    #######################################################################
    # Setup MLFlow for experiment tracking
    experiment_name = cfg.mlflow_experiment_name
    mlflow.set_tracking_uri("http://my_mlflow_server")
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(
            name=experiment_name,
        )
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    mlflow_run_name = get_mlflow_run_name(cfg)

    ##########################################################################
    # Start the evaluation run
    ##########################################################################
    # Define mlflow run to log metrics and parameters
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=mlflow_run_name) as run:
        print(f"MlFlow experiment name: {cfg.mlflow_experiment_name}, run name: {mlflow_run_name}")
        print("run_id: {}; status: {}".format(run.info.run_id, run.info.status))
        # Log parameters in config with mlflow
        mlflow.log_params(cfg)
        # Log latent space size (total number of channels across layers taken into account)
        mlflow.log_param("ls_size", aggregated_ind_data_dict["train"].shape[1])
        # Log the number of used InD samples
        if len(ind_no_obj) > 0:
            for split, no_obj in ind_no_obj.items():
                mlflow.log_metric(
                    f"No_obj_{split}",
                    round((len(no_obj) / ind_datasets_lengths[cfg.ind_dataset][split]) * 100, 2)
                )
        if len(ood_no_obj) > 0:
            # Log the number of used OoD samples
            for ood_dataset_name, no_obj in ood_no_obj.items():
                mlflow.log_metric(
                    f"No_obj_{ood_dataset_name}",
                    round((len(no_obj) / ood_datasets_lengths[cfg.ind_dataset][ood_dataset_name]) * 100, 2)
                )
        _, best_postprocessors_dict, postprocessor_thresholds, aggregated_ood_data_dict = log_evaluate_larex(
            cfg=cfg,
            baselines_names=BASELINES_NAMES,
            ind_data_dict=aggregated_ind_data_dict,
            ood_data_dict=aggregated_ood_data_dict,
            ood_baselines_scores=ood_baselines_scores_dict,
            mlflow_run_name=mlflow_run_name,
            mlflow_logging=True,
            visualize_score=VISUALIZE_POSTPROCESSOR,
            postprocessors=POSTPROCESSORS,
        )

        for ood_dataset_name in cfg.ood_datasets:
            best_postp_names = [best_postprocessors_dict[p_name]["best_comp"] for p_name in POSTPROCESSORS]
            ood_data_dict[ood_dataset_name] = associate_precalculated_baselines_with_raw_predictions(
                data_dict=ood_data_dict[ood_dataset_name],
                dataset_name=ood_dataset_name,
                ood_baselines_dict=aggregated_ood_data_dict,
                baselines_names=best_postp_names,
                non_empty_ids=non_empty_preds_ood_im_ids[ood_dataset_name],
                is_ood=True
            )
        open_set_results = get_overall_open_set_results(
            ind_dataset_name=cfg.ind_dataset,
            ind_gt_annotations_path=cfg.ind_annotations_path[cfg.ind_dataset],
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_datasets_names=cfg.ood_datasets,
            ood_annotations_paths=cfg.ood_annotations_paths[cfg.ind_dataset],
            methods_names=BASELINES_NAMES + best_postp_names,
            methods_thresholds={**baselines_thresholds, **postprocessor_thresholds},
            metric_2007=cfg.metric_2007,
            evaluate_on_ind=False,
            get_known_classes_metrics=cfg.get_known_classes_metrics,
            using_id_val_subset=list(ind_data_dict["valid"].keys())
        )
        osod_pd_dfs = {}
        for ood_dataset_name in cfg.ood_datasets:
            osod_pd_dfs[ood_dataset_name] = convert_osod_results_to_pandas_df(
                open_set_results=open_set_results[ood_dataset_name],
                methods_names=BASELINES_NAMES + best_postp_names,
                save_method_as_data=True
            )
            mlflow.log_table(osod_pd_dfs[ood_dataset_name], f"osod/{ood_dataset_name}_osod_results.csv")

        if GET_OSOD_PLOTS:
            if "coco_near" in cfg.ood_datasets and "oi_near" in cfg.ood_datasets and cfg.ind_dataset == "voc":
                for ood_dataset_name in BASE_OOD_DATASETS:
                    osod_fig = plot_two_osod_datasets_metrics(
                        osod_results_a=osod_pd_dfs[f"{ood_dataset_name}_new"],
                        osod_results_b=osod_pd_dfs[f"{ood_dataset_name}_near"],
                        methods_names=BASELINES_NAMES + best_postp_names,
                        datasets_names=[f"{ood_dataset_name}_new", f"{ood_dataset_name}_near"],
                        metrics_to_plot=OSOD_METRICS_TO_PLOT,
                        show_plot=False
                    )
                    mlflow.log_figure(figure=osod_fig, artifact_file=f"osod/{ood_dataset_name}_new_near.png")
                    for metric_name in OSOD_METRICS_TO_PLOT:
                        metric_plot = plot_two_osod_datasets_per_metric(
                            osod_results_a=osod_pd_dfs[f"{ood_dataset_name}_new"],
                            osod_results_b=osod_pd_dfs[f"{ood_dataset_name}_near"],
                            methods_names=BASELINES_NAMES + best_postp_names,
                            datasets_names=[f"{ood_dataset_name}_new", f"{ood_dataset_name}_near"],
                            metric_to_plot=metric_name,
                            show_plot=False
                        )
                        mlflow.log_figure(
                            figure=metric_plot,
                            artifact_file=f"osod/{ood_dataset_name}_new_near_{metric_name}.png"
                        )
            elif "coco_new" in cfg.ood_datasets and "oi_new" in cfg.ood_datasets:
                osod_fig = plot_two_osod_datasets_metrics(
                    osod_results_a=osod_pd_dfs["coco_new"],
                    osod_results_b=osod_pd_dfs["oi_new"],
                    methods_names=BASELINES_NAMES + best_postp_names,
                    datasets_names=["coco_new", "oi_new"],
                    metrics_to_plot=OSOD_METRICS_TO_PLOT,
                    show_plot=False
                )
                mlflow.log_figure(figure=osod_fig, artifact_file=f"osod/new_coco_oi_osod.png")
                for metric_name in OSOD_METRICS_TO_PLOT:
                    metric_plot = plot_two_osod_datasets_per_metric(
                        osod_results_a=osod_pd_dfs["coco_new"],
                        osod_results_b=osod_pd_dfs["oi_new"],
                        methods_names=BASELINES_NAMES + best_postp_names,
                        datasets_names=["coco_new", "oi_new"],
                        metric_to_plot=metric_name,
                        show_plot=False
                    )
                    mlflow.log_figure(
                        figure=metric_plot,
                        artifact_file=f"osod/new_coco_oi_{metric_name}.png"
                    )
        open_set_results_mlflow = convert_osod_results_for_mlflow_logging(
            open_set_results=open_set_results,
            ood_datasets_names=cfg.ood_datasets,
            methods_names=BASELINES_NAMES + best_postp_names,
        )
        # Log Open set metrics
        mlflow.log_metrics(open_set_results_mlflow, step=0)
        mlflow.end_run()


def print_config(cfg):
    print(
        f"Analyzing from InD {cfg.ind_dataset}, OoD {cfg.ood_datasets}, boxes, "
        f"{'std_devs ' if cfg.return_variances else ''}"
        f"hooked {cfg.hooked_modules} output {cfg.hook_output} "
        f"roi_s{''.join(map(str, cfg.roi_output_sizes))} "
        f"roi_sr{cfg.roi_sampling_ratio} "
        f"Inf_conf: {cfg.inference_predict_confidence} "
        f"{'entropy ' if cfg.dropblock_entropy == True else ''}"
        f"{'dp' + str(cfg.drop_prob) + ' ' if cfg.dropblock_entropy == True else ''}"
        f"{'ds' + str(cfg.drop_size) + ' ' if cfg.dropblock_entropy == True else ''}"
        f"{'mcd' + str(cfg.n_mcd) + ' ' if cfg.dropblock_entropy == True else ''}"
        f"{'vos ' if cfg.using_vos else ''}"
        f"{'regnet ' if cfg.using_regnet else ''}"
    )



def get_ind_file_name(cfg, file_split):
    return (
        f"ind_{cfg.ind_dataset}_{file_split}_"
        f"roi_s{''.join(map(str, cfg.roi_output_sizes))}_"
        f"roi_sr{cfg.roi_sampling_ratio}_"
        f"infc_{str(int(cfg.inference_predict_confidence * 100))}_"
        f"{'h_z_' if cfg.dropblock_entropy == True else ''}"
        f"{'dp' + str(int(10 * cfg.drop_prob)) + '_' if cfg.dropblock_entropy == True else ''}"
        f"{'ds' + str(cfg.drop_size) + '_' if cfg.dropblock_entropy == True else ''}"
        f"{'mcd' + str(cfg.n_mcd) + '_' if cfg.dropblock_entropy == True else ''}"
        f"{'vos_' if cfg.using_vos else ''}"
        f"{'regnet_' if cfg.using_regnet else ''}"
        f"ls_samples.pt"
    )


def get_ood_file_name(cfg, ood_dataset_name):
    return (
        f"ood_{ood_dataset_name}_"
        f"roi_s{''.join(map(str, cfg.roi_output_sizes))}_"
        f"roi_sr{cfg.roi_sampling_ratio}_"
        f"infc_{str(int(cfg.inference_predict_confidence * 100))}_"
        f"{'h_z_' if cfg.dropblock_entropy == True else ''}"
        f"{'dp' + str(int(10 * cfg.drop_prob)) + '_' if cfg.dropblock_entropy == True else ''}"
        f"{'ds' + str(cfg.drop_size) + '_' if cfg.dropblock_entropy == True else ''}"
        f"{'mcd' + str(cfg.n_mcd) + '_' if cfg.dropblock_entropy == True else ''}"
        f"{'vos_' if cfg.using_vos else ''}"
        f"{'regnet_' if cfg.using_regnet else ''}"
        f"ls_samples.pt"
    )


def get_mlflow_run_name(cfg):
    return (
        # f"h{cfg.hooked_module}_"
        f"h{'_'.join(map(str, cfg.hooked_modules))}_"
        f"{'stds_' if cfg.return_variances else ''}"
        # f"out{cfg.hook_output}_"
        f"roi_s{''.join(map(str, cfg.roi_output_sizes))}_"
        # f"roi_sr{cfg.roi_sampling_ratio}_"
        f"infc_{str(int(cfg.inference_predict_confidence * 100))}_"
        f"kn_{cfg.k_neighbors}"
        f"{'_h_z_' if cfg.dropblock_entropy == True else ''}"
        f"{'dp' + str(int(10 * cfg.drop_prob)) + '_' if cfg.dropblock_entropy == True else ''}"
        f"{'ds' + str(cfg.drop_size) + '_' if cfg.dropblock_entropy == True else ''}"
        f"{'mcd' + str(cfg.n_mcd) if cfg.dropblock_entropy == True else ''}"
        f"{'_vos' if cfg.using_vos else ''}"
        f"{'_regnet' if cfg.using_regnet else ''}"
        f"{'_new_bm' if 'oi_new' in cfg.ood_datasets else ''}"
    )


ind_datasets_lengths = {
    "bdd": {
        "train": 69853,
        "valid": 10000,
        "test": 2000  # For boxes evaluation
    },
    "voc": {
        "train": 16551,
        "valid": 4952,
        "test": 2030
    }
}

ood_datasets_lengths = {
    "bdd": {
        "coco": 1880,
        "openimages": 1761,
        "oi_new": 1695,
        "coco_new": 1873
    },
    "voc": {
        "coco": 930,
        "openimages": 1761,
        "coco_new": 938,
        "coco_all_new": 2116,
        "coco_near": 1174,
        "oi_new": 1179,
        "oi_all_new": 2079,
        "oi_near": 908,
    }
}


if __name__ == "__main__":
    main()
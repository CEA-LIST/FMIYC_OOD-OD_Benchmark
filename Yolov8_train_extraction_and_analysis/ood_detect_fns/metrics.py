# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Fabio Arnez
#    Daniel Montoya

from typing import Union, Tuple, Dict, List, Optional
import numpy as np
import torch
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import auc
import torchmetrics.functional as tmf
import seaborn as sns

from .uncertainty_estimation import postprocessors_dict


def get_hz_detector_results(
    detect_exp_name: str,
    ind_samples_scores: np.ndarray,
    ood_samples_scores: np.ndarray,
    return_results_for_mlflow: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Calculates the metrics relevant OoD detection: AUROC, FPR, AUPR, TPR, precision, recall,
    and classification thresholds. Can optionally format results for mlflow logging (no @ allowed).
    Automatically inverts labels if AUROC<0.5.

    Args:
        detect_exp_name: Name of the current experiment. This will be of the name of the row
         of the returned pandas df
        ind_samples_scores: Array of InD scores
        ood_samples_scores: Array of OoD scores
        return_results_for_mlflow: Optionally return AUROC, FPR and AUPR formatted for mlflow
         logging

    Returns:
        (pd.Dataframe): Results in a pandas dataframe format and optionally a dictionary with
            results for mlflow
    """
    assert isinstance(detect_exp_name, str), "detect_exp_name must be a string"
    assert isinstance(ind_samples_scores, np.ndarray), "ind_samples_scores must be a numpy array"
    assert isinstance(ood_samples_scores, np.ndarray), "ood_samples_scores must be a numpy array"
    assert isinstance(
        return_results_for_mlflow, bool
    ), "return_results_for_mlflow must be a boolean"
    labels_ind_test = np.ones((ind_samples_scores.shape[0], 1))  # positive class
    labels_ood_test = np.zeros((ood_samples_scores.shape[0], 1))  # negative class

    ind_samples_scores = np.expand_dims(ind_samples_scores, 1)
    ood_samples_scores = np.expand_dims(ood_samples_scores, 1)

    scores = np.vstack((ind_samples_scores, ood_samples_scores))
    labels = np.vstack((labels_ind_test, labels_ood_test))
    labels = labels.astype("int32")

    results_table = pd.DataFrame(
        columns=[
            "experiment",
            "auroc",
            "fpr@95",
            "aupr",
            "fpr",
            "tpr",
            "roc_thresholds",
            "precision",
            "recall",
            "pr_thresholds",
        ]
    )

    roc_auc = tmf.auroc(torch.from_numpy(scores), torch.from_numpy(labels))

    fpr, tpr, roc_thresholds = tmf.roc(torch.from_numpy(scores), torch.from_numpy(labels))

    fpr_95 = fpr[torch.where(tpr >= 0.95)[0][0]]

    precision, recall, pr_thresholds = tmf.precision_recall_curve(
        torch.from_numpy(scores), torch.from_numpy(labels)
    )
    aupr = auc(recall.numpy(), precision.numpy())

    results_table = results_table.append(
        {
            "experiment": detect_exp_name,
            "auroc": roc_auc.item(),
            "fpr@95": fpr_95.item(),
            "aupr": aupr,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_thresholds": roc_thresholds.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "pr_thresholds": pr_thresholds.tolist(),
        },
        ignore_index=True,
    )

    results_table.set_index("experiment", inplace=True)

    if not return_results_for_mlflow:
        return results_table
    results_for_mlflow = results_table.loc[detect_exp_name, ["auroc", "fpr@95", "aupr"]].to_dict()
    # MLFlow doesn't accept the character '@'
    results_for_mlflow["fpr_95"] = results_for_mlflow.pop("fpr@95")
    return results_table, results_for_mlflow


def plot_roc_ood_detector(results_table, plot_title: str = "Plot Title"):
    """
    Plot ROC curve from the results table from the function get_hz_detector_results.

    Args:
        results_table: Pandas table obtained with the get_hz_detector_results function
        plot_title: Title of the plot

    """
    plt.figure(figsize=(8, 6))
    for i in results_table.index:
        # print(i)
        plt.plot(
            results_table.loc[i]["fpr"],
            results_table.loc[i]["tpr"],
            label=i + ", AUROC={:.4f}".format(results_table.loc[i]["auroc"]),
        )

    plt.plot([0, 1], [0, 1], color="orange", linestyle="--")
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title(plot_title, fontweight="bold", fontsize=15)
    plt.legend(prop={"size": 12}, loc="lower right")
    plt.show()


def save_roc_ood_detector(
    results_table: pd.DataFrame, postprocessors: List[str], plot_title: str = "Plot Title"
) -> plt.Figure:
    """
    Returns a ROC plot figure that can be saved or logged with mlflow. Does not display any
    figure to screen

    Args:
        results_table (pd.Dataframe): Dataframe with results as rows and experiments names as
            indexes
        postprocessors: List of strings of postprocessors names
        plot_title (str): Title of the plot

    Returns:
        (plt.Figure): A figure to be saved or logged with mlflow
    """
    assert isinstance(results_table, pd.DataFrame), "results_table must be a pandas dataframe"
    assert isinstance(plot_title, str), "plot_title must be a string"
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in results_table.index:
        if any([postp in i for postp in postprocessors]):
            ax.plot(
                results_table.loc[i]["fpr"],
                results_table.loc[i]["tpr"],
                label=i + ", AUROC={:.4f}".format(results_table.loc[i]["auroc"]),
                linestyle="solid",
                linewidth=3.0,
            )
        else:
            ax.plot(
                results_table.loc[i]["fpr"],
                results_table.loc[i]["tpr"],
                label=i + ", AUROC={:.4f}".format(results_table.loc[i]["auroc"]),
                linestyle="dashed",
                linewidth=1.7,
            )

    ax.plot([0, 1], [0, 1], color="orange", linestyle="--")
    ax.set_xticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_ylabel("True Positive Rate", fontsize=15)
    ax.set_title(plot_title, fontweight="bold", fontsize=15)
    ax.legend(prop={"size": 12}, loc="lower right")
    return fig


def save_scores_plots(
    scores_ind: np.ndarray,
    ood_scores_dict: Dict,
    ood_datasets_list: List,
    ind_dataset_name: str,
    post_processor_name: str = "LaREM",
) -> Dict:
    """
    InD and OoD agnostic function that takes as input the InD numpy ndarray with the LaRED scores,
    a dictionary of OoD LaRED scores, a list of the names of the OoD dataset, and the name of the
    InD dataset, and returns a histogram of pairwise comparisons, that can be saved to a
    file, logged with mlflow, or shown in screen

    Args:
        scores_ind: InD LaRED scores as numpy ndarray
        ood_scores_dict: Dictionary keys as ood datasets names and values as ndarrays of
            LaRED scores per each
        ood_datasets_list: List of OoD datasets names
        ind_dataset_name: String with the name of the InD dataset
        post_processor_name: String with the name of the post-processing function. One of "LaRED", "LaREM", or "LaREK"

    Returns:
        Dictionary of plots where the keys are the plot names and the values are the figures
    """
    assert isinstance(scores_ind, np.ndarray), "scores_ind must be a numpy array"
    assert isinstance(ood_scores_dict, dict), "ood_lared_scores_dict must be a dictionary"
    assert hasattr(ood_datasets_list, "__iter__"), "ood_datasets_list must be an iterable"
    assert all(isinstance(item, str) for item in ood_datasets_list), (
        "ood_datasets_list items must" " be strings"
    )
    assert isinstance(ind_dataset_name, str), "ind_dataset_name must be a string"
    assert post_processor_name in postprocessors_dict.keys()
    df_scores_ind = pd.DataFrame(scores_ind, columns=[f"{post_processor_name} score"])
    df_scores_ind.insert(0, "Dataset", "")
    df_scores_ind.loc[:, "Dataset"] = ind_dataset_name
    ood_df_dict = {}
    for ood_dataset_name in ood_datasets_list:
        ood_df_dict[ood_dataset_name] = pd.DataFrame(
            ood_scores_dict[ood_dataset_name], columns=[f"{post_processor_name} score"]
        )
        ood_df_dict[ood_dataset_name].insert(0, "Dataset", "")
        ood_df_dict[ood_dataset_name].loc[:, "Dataset"] = ood_dataset_name

    plots_dict = {}
    for ood_dataset_name in ood_datasets_list:
        df_h_z_scores = pd.concat([df_scores_ind, ood_df_dict[ood_dataset_name]]).reset_index(
            drop=True
        )
        plots_dict[f"{ood_dataset_name}_{post_processor_name}_scores"] = sns.displot(
            df_h_z_scores, x=f"{post_processor_name} score", hue="Dataset", kind="hist", fill=True
        )

    return plots_dict


def get_pred_scores_plots(
    experiment: Dict, ood_datasets_list: list, title: str, ind_dataset_name: str
):
    """
    Function that takes as input an experiment dictionary (one classification technique), a list
    of ood datasets, a plot title, and the InD dataset name and returns a plot of the predictive
    score density

    Args:
        experiment: Dictionary with keys 'InD':ndarray, 'x_axis':str, and 'plot_name':str and other
            keys as ood dataset names with values as ndarray
        ood_datasets_list: List with OoD datasets names
        title: Title of the plot
        ind_dataset_name: String with the name of the InD dataset

    Returns:
        Figure with the density scores of the InD and the OoD datasets
    """
    assert isinstance(experiment, dict)
    assert hasattr(ood_datasets_list, "__iter__"), "ood_datasets_list must be an iterable"
    assert all(isinstance(item, str) for item in ood_datasets_list), (
        "ood_datasets_list items must" " be strings"
    )
    assert isinstance(title, str)
    assert isinstance(ind_dataset_name, str)
    df_pred_h_scores_ind = pd.DataFrame(experiment["InD"], columns=[experiment["x_axis"]])
    df_pred_h_scores_ind.insert(0, "Dataset", "")
    df_pred_h_scores_ind.loc[:, "Dataset"] = ind_dataset_name
    ood_df_dict = {}
    for ood_dataset_name in ood_datasets_list:
        ood_df_dict[ood_dataset_name] = pd.DataFrame(
            experiment[ood_dataset_name], columns=[experiment["x_axis"]]
        )
        ood_df_dict[ood_dataset_name].insert(0, "Dataset", "")
        ood_df_dict[ood_dataset_name].loc[:, "Dataset"] = ood_dataset_name

    all_dfs = [df_pred_h_scores_ind]
    all_dfs.extend(list(ood_df_dict.values()))
    df_pred_h_scores = pd.concat(all_dfs).reset_index(drop=True)

    ax = sns.displot(
        df_pred_h_scores, x=experiment["x_axis"], hue="Dataset", kind="hist", fill=True
    ).set(title=title)
    plt.tight_layout()
    plt.legend(loc="best")
    return ax


def log_evaluate_postprocessors(
    ind_dict: Dict[str, np.ndarray],
    ood_dict: Dict[str, np.ndarray],
    ood_datasets_names: List[str],
    experiment_name_extension: str = "",
    return_density_scores: Union[None, str] = None,
    log_step: Union[int, None] = None,
    mlflow_logging: bool = False,
    postprocessors=None,
    cfg: DictConfig = None,
) -> Dict[str, Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]]:
    """
    Function that takes as input InD numpy arrays of entropies and one dictionary for all OoD
    datasets and returns LaRED and LaREM results in the form of a pandas dataframe.
    Optionally logs to a running mlflow experiment.

    Args:
        ind_dict: InD data in the format {"train": np.ndarray, "valid": np.ndarray, "train labels": np.ndarray,
         "valid labels": np.ndarray}
        ood_dict: OoD dictionary where keys are the OoD datasets and the values are the
            numpy arrays of latent representations
        ood_datasets_names: List of strings with the names of the OOD datasets
        experiment_name_extension: Extra string to add to the default experiment name, useful for
            PCA experiments
        return_density_scores: return one of the postprocessor density scores for further analysis. Either 'LaRED',
            'LaREM' or 'LaREK'
        log_step: optional step useful for PCA experiments. None if not performing PCA with
            several components
        mlflow_logging: Optionally log to an existing mlflow run
        postprocessors: List of postprocessors to apply to precalculated ls samples.
            Default: ["LaRED", "LaREM", "LaREK"]
        cfg: Configuration class, useful for postprocessor parameters

    Returns:
        Pandas dataframe with results, optionally LaRED density score
    """
    assert isinstance(ind_dict["train"], np.ndarray)
    assert isinstance(ind_dict["valid"], np.ndarray)
    assert isinstance(ood_dict, dict)
    assert isinstance(experiment_name_extension, str)
    if return_density_scores is not None:
        # assert return_density_scores in ("LaRED", "LaREM", "LaREK", "LaREcM")
        assert return_density_scores in postprocessors_dict.keys()
    if log_step is not None:
        assert isinstance(log_step, int), "log_step is either None or an integer"
    assert isinstance(mlflow_logging, bool)
    if postprocessors is None:
        # postprocessors = ["LaRED", "LaREM", "LaREK", "LaREcM"]
        postprocessors = postprocessors_dict.keys()

    # Initialize df to store all the results
    overall_metrics_df = pd.DataFrame(
        columns=[
            "auroc",
            "fpr@95",
            "aupr",
            "fpr",
            "tpr",
            "roc_thresholds",
            "precision",
            "recall",
            "pr_thresholds",
        ]
    )
    ##############################
    # Calculate scores
    ##############################
    # Initialize dictionaries
    ind_scores_dict = {}
    ood_scores_dict = {}
    for postprocessor in postprocessors:
        # Instantiate postprocessor
        postp_instance = postprocessors_dict[postprocessor]
        postp_instance.__init__(cfg=cfg)
        postp_instance.setup_flag = False
        # Train postprocessor
        postp_instance.setup(ind_dict["train"], ind_train_labels=ind_dict["train labels"])
        # InD Inference
        ind_scores_dict[postprocessor] = postp_instance.postprocess(
            ind_dict["valid"], pred_labels=ind_dict["valid labels"]
        )
        # OoD Inference
        ood_scores_dict[postprocessor] = {}
        for ood_dataset_name in ood_datasets_names:
            ood_scores_dict[postprocessor][ood_dataset_name] = postp_instance.postprocess(
                ood_dict[ood_dataset_name], pred_labels=ood_dict[f"{ood_dataset_name} labels"]
            )

    #########################
    # Prepare logging of results
    postprocessors_experiments = {}
    for ood_dataset_name in ood_datasets_names:
        for postprocessor in postprocessors:
            postprocessors_experiments[f"{ood_dataset_name} {postprocessor}"] = {
                "InD": ind_scores_dict[postprocessor],
                "OoD": ood_scores_dict[postprocessor][ood_dataset_name],
            }

    # Log Results
    for experiment_name, experiment in postprocessors_experiments.items():
        experiment_name = experiment_name + experiment_name_extension
        r_df, r_mlflow = get_hz_detector_results(
            detect_exp_name=experiment_name,
            ind_samples_scores=experiment["InD"],
            ood_samples_scores=experiment["OoD"],
            return_results_for_mlflow=True,
        )
        # Add OoD dataset to metrics name
        if "PCA" in experiment_name:
            r_mlflow = {
                f"{' '.join(experiment_name.split()[:-1])}_{k}": v for k, v in r_mlflow.items()
            }

        else:
            r_mlflow = {f"{experiment_name}_{k}": v for k, v in r_mlflow.items()}
        if mlflow_logging:
            mlflow.log_metrics(r_mlflow, step=log_step)
        overall_metrics_df = overall_metrics_df.append(r_df)

    results = {"results_df": overall_metrics_df}
    if return_density_scores is not None:
        results["InD"] = ind_scores_dict[return_density_scores]
        results["OoD"] = ood_scores_dict[return_density_scores]
    return results


def select_and_log_best_larex(
    overall_metrics_df: pd.DataFrame,
    n_pca_components_list: Union[list, Tuple],
    technique: str,
    multiple_ood_datasets_flag: bool,
    log_mlflow: bool = False,
) -> Tuple[float, float, float, int]:
    """
    Takes as input a Dataframe with the columns 'auroc', 'aupr' and 'fpr@95', a list of PCA number
    of components, and the name of the technique: either 'LaRED' or 'LaREM', and logs to and
    existing mlflow run the best metrics

    Args:
        overall_metrics_df: Pandas DataFrame with the LaRED or LaREM experiments results
        n_pca_components_list: List with the numbers of PCA components
        technique: Either 'LaRED', 'LaREM' or 'LaREK'
        multiple_ood_datasets_flag: Flag that indicates whether there are multiple ood datasets or not
        log_mlflow: Log to mlflow boolean flag

    Returns:
        Tuple with the best auroc, aupr, fpr and the N components.
    """
    assert isinstance(overall_metrics_df, pd.DataFrame)
    assert hasattr(n_pca_components_list, "__iter__")
    assert isinstance(log_mlflow, bool)
    assert technique in postprocessors_dict.keys(), f"Got {technique}"
    means_df = pd.DataFrame(columns=["auroc", "fpr@95", "aupr"])
    temp_df = pd.DataFrame(columns=["auroc", "fpr@95", "aupr"])
    # Calculate mean of no PCA run
    for row_name in overall_metrics_df.index:
        if technique in row_name and "anomalies" not in row_name and "PCA" not in row_name:
            temp_df = temp_df.append(overall_metrics_df.loc[row_name, ["auroc", "fpr@95", "aupr"]])
    means_temp_df = temp_df.mean()
    means_df = means_df.append(pd.DataFrame(dict(means_temp_df), index=[technique]))
    if multiple_ood_datasets_flag:
        stds_df = pd.DataFrame(columns=["auroc", "fpr@95", "aupr"])
        stds_temp_df = temp_df.std()
        stds_df = stds_df.append(pd.DataFrame(dict(stds_temp_df), index=[technique]))

    # Calculate means of PCA runs
    for n_components in n_pca_components_list:
        temp_df = pd.DataFrame(columns=["auroc", "fpr@95", "aupr"])
        for row_name in overall_metrics_df.index:
            if (
                technique in row_name
                and f"PCA {n_components}" in row_name
                and row_name.split(f"PCA {n_components}")[-1] == ""
            ):
                temp_df = temp_df.append(
                    overall_metrics_df.loc[row_name, ["auroc", "fpr@95", "aupr"]]
                )
        means_temp_df = temp_df.mean()
        means_df = means_df.append(
            pd.DataFrame(dict(means_temp_df), index=[f"{technique} PCA {n_components}"])
        )
        if multiple_ood_datasets_flag:
            stds_temp_df = temp_df.std()
            stds_df = stds_df.append(
                pd.DataFrame(dict(stds_temp_df), index=[f"{technique} PCA {n_components}"])
            )
    best_index = means_df[means_df.auroc == means_df.auroc.max()].index[0]
    # Here we assume the convention that 0 PCA components would mean the no PCA case
    if "PCA" in best_index:
        best_n_comps = int(best_index.split()[-1])
    else:
        best_n_comps = 0

    if log_mlflow:
        mlflow.log_metric(f"{technique}_auroc_mean", means_df.loc[best_index, "auroc"])
        mlflow.log_metric(f"{technique}_aupr_mean", means_df.loc[best_index, "aupr"])
        mlflow.log_metric(f"{technique}_fpr95_mean", means_df.loc[best_index, "fpr@95"])
        mlflow.log_metric(f"Best {technique}", best_n_comps)
        if multiple_ood_datasets_flag:
            mlflow.log_metric(f"{technique}_auroc_std", stds_df.loc[best_index, "auroc"])
            mlflow.log_metric(f"{technique}_aupr_std", stds_df.loc[best_index, "aupr"])
            mlflow.log_metric(f"{technique}_fpr95_std", stds_df.loc[best_index, "fpr@95"])
    return (
        means_df.loc[best_index, "auroc"],
        means_df.loc[best_index, "aupr"],
        means_df.loc[best_index, "fpr@95"],
        best_n_comps,
    )


def subset_boxes(
    ind_dict: Dict[str, np.ndarray],
    ood_dict: Dict[str, np.ndarray],
    ind_train_limit: int,
    ood_limit: int,
    random_seed: int,
    ood_names: List[str],
    non_empty_predictions_id: Optional[Dict[str, List]] = None,
    non_empty_predictions_ood: Optional[Dict[str, List]] = None,
) -> Union[
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List], Dict[str, List]],
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
]:
    """
    Function that subsets a given number of box predictions into a smaller number of them, to speed up caluclations
    during evaluation.

    Args:
        ind_dict: InD data dictionary, with the entries 'train' and 'valid'
        ood_dict: OoD data dictionary where each ood dataset is its own key-value pair
        ind_train_limit: Max number of allowed InD train boxes
        ood_limit: Max number of allowed OoD boxes
        random_seed: Random generator seed
        ood_names: List with the names of the OOD datasets
        non_empty_predictions_id: List with the ids of the images that have non-empty predictions in the InD valid dataset
        non_empty_predictions_ood: List with the ids of the images that have non-empty predictions in the OoD datasets

    Returns:
        Tuple of InD and OoD subset dictionaries
    """
    np.random.seed(random_seed)
    # Subset train
    if ind_dict["train"].shape[0] > ind_train_limit:
        print(
            f"Subsetting train set to {ind_train_limit} from {ind_dict['train'].shape[0]} extracted boxes"
        )
        chosen_idx_train = np.random.choice(
            ind_dict["train"].shape[0], size=ind_train_limit, replace=False
        )
        ind_dict["train"] = ind_dict["train"][chosen_idx_train]
        if "train logits" in ind_dict.keys():
            ind_dict["train logits"] = ind_dict["train logits"][chosen_idx_train, :]
        if "train features" in ind_dict.keys():
            ind_dict["train features"] = ind_dict["train features"][chosen_idx_train, :]

    # Subset InD valid to be the same size as the ood length
    if ind_dict["valid"].shape[0] > ood_limit:
        print(
            f"Subsetting valid set to {ood_limit} from {ind_dict['valid'].shape[0]} extracted boxes"
        )
        chosen_idx_valid = np.random.choice(
            ind_dict["valid"].shape[0], size=ood_limit, replace=False
        )
        ind_dict["valid"] = ind_dict["valid"][chosen_idx_valid]
        if "valid logits" in ind_dict.keys():
            ind_dict["valid logits"] = ind_dict["valid logits"][chosen_idx_valid, :]
        if "valid features" in ind_dict.keys():
            ind_dict["valid features"] = ind_dict["valid features"][chosen_idx_valid, :]
        if non_empty_predictions_id is not None:
            non_empty_predictions_id["valid"] = [
                non_empty_predictions_id["valid"][i] for i in chosen_idx_valid
            ]

    # Subset OoD
    for ood_dataset_name in ood_names:
        data = ood_dict[ood_dataset_name]
        if data.shape[0] > ood_limit:
            print(
                f"Subsetting {ood_dataset_name} to {ood_limit} from {data.shape[0]} extracted boxes"
            )
            chosen_idx_ood = np.random.choice(data.shape[0], size=ood_limit, replace=False)
            ood_dict[ood_dataset_name] = data[chosen_idx_ood]
            if f"{ood_dataset_name} logits" in ood_dict.keys():
                ood_dict[f"{ood_dataset_name} logits"] = ood_dict[f"{ood_dataset_name} logits"][
                    chosen_idx_ood, :
                ]
            if f"{ood_dataset_name} features" in ood_dict.keys():
                ood_dict[f"{ood_dataset_name} features"] = ood_dict[f"{ood_dataset_name} features"][
                    chosen_idx_ood, :
                ]
            if non_empty_predictions_ood is not None:
                non_empty_predictions_ood[ood_dataset_name] = [
                    non_empty_predictions_ood[ood_dataset_name][i] for i in chosen_idx_ood
                ]

    if non_empty_predictions_id is not None and non_empty_predictions_ood is not None:
        return ind_dict, ood_dict, non_empty_predictions_id, non_empty_predictions_ood
    return ind_dict, ood_dict


# Commented here, since the baselines dictionary should be found now in the baselines script
# baseline_name_dict = {
#     "pred_h": {
#         "plot_title": "Predictive H distribution",
#         "x_axis": "Predictive H score",
#         "plot_name": "pred_h",
#     },
#     "mi": {
#         "plot_title": "Predictive MI distribution",
#         "x_axis": "Predictive MI score",
#         "plot_name": "pred_mi",
#     },
#     "msp": {
#         "plot_title": "Predictive MSP distribution",
#         "x_axis": "Predictive MSP score",
#         "plot_name": "pred_msp",
#     },
#     "energy": {
#         "plot_title": "Predictive energy score distribution",
#         "x_axis": "Predictive energy score",
#         "plot_name": "pred_energy",
#     },
#     "mdist": {
#         "plot_title": "Mahalanobis Distance distribution",
#         "x_axis": "Mahalanobis Distance score",
#         "plot_name": "pred_mdist",
#     },
#     "knn": {
#         "plot_title": "kNN distance distribution",
#         "x_axis": "kNN Distance score",
#         "plot_name": "pred_knn",
#     },
#     "ash": {
#         "plot_title": "ASH score distribution",
#         "x_axis": "ASH score",
#         "plot_name": "ash_score",
#     },
#     "dice": {
#         "plot_title": "DICE score distribution",
#         "x_axis": "DICE score",
#         "plot_name": "dice_score",
#     },
#     "react": {
#         "plot_title": "ReAct score distribution",
#         "x_axis": "ReAct score",
#         "plot_name": "react_score",
#     },
#     "dice_react": {
#         "plot_title": "DICE + ReAct score distribution",
#         "x_axis": "DICE + ReAct score",
#         "plot_name": "dice_react_score",
#     },
#     "filtered_energy": {
#         "plot_title": "Predictive filtered energy score distribution",
#         "x_axis": "Predictive filtered energy score",
#         "plot_name": "pred_filtered_energy",
#     },
#     "filtered_ash": {
#         "plot_title": "ASH filtered score distribution",
#         "x_axis": "ASH filtered score",
#         "plot_name": "filtered_ash_score",
#     },
#     "filtered_react": {
#         "plot_title": "ReAct filtered score distribution",
#         "x_axis": "Filtered ReAct score",
#         "plot_name": "filtered_react_score",
#     },
#     "filtered_dice": {
#         "plot_title": "DICE filtered score distribution",
#         "x_axis": "DICE filtered score",
#         "plot_name": "filtered_dice_score",
#     },
#     "filtered_dice_react": {
#         "plot_title": "DICE + ReAct filtered score distribution",
#         "x_axis": "DICE + ReAct filtered score",
#         "plot_name": "filtered_dice_react_score",
#     },
#     "raw_energy": {
#         "plot_title": "Predictive raw energy score distribution",
#         "x_axis": "Predictive raw energy score",
#         "plot_name": "pred_raw_energy",
#     },
#     "raw_ash": {
#         "plot_title": "ASH raw score distribution",
#         "x_axis": "ASH raw score",
#         "plot_name": "raw_ash_score",
#     },
#     "raw_react": {
#         "plot_title": "ReAct raw score distribution",
#         "x_axis": "raw ReAct score",
#         "plot_name": "raw_react_score",
#     },
#     "raw_dice": {
#         "plot_title": "DICE raw score distribution",
#         "x_axis": "DICE raw score",
#         "plot_name": "raw_dice_score",
#     },
#     "raw_dice_react": {
#         "plot_title": "DICE + ReAct raw score distribution",
#         "x_axis": "DICE + ReAct raw score",
#         "plot_name": "raw_dice_react_score",
#     },
# }

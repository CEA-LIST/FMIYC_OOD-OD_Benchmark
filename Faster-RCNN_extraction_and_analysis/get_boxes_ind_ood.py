"""
Script for performing Model outputs extraction for the InD and OoD datasets. Analysis is performed in another script
"""
import core
import os
import sys
import torch
from shutil import copyfile

from helper_fns import visualize_predictions

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), "src", "detr"))

# Detectron imports
from detectron2.engine import launch

# Project imports
from core.setup import setup_config, setup_arg_parser
from inference.inference_utils import get_inference_output_dir, build_predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from ood_detect_fns.uncertainty_estimation import BoxFeaturesExtractor, Hook
from dataloader_helper_functions import (
    build_in_distribution_valid_test_dataloader_args,
    build_data_loader,
    build_ood_dataloader_args, build_ind_voc_train_dataloader_args,
)

EXTRACT_IND = True
EXTRACT_OOD = True

VISUALIZE_N_SAMPLES = 10


def main(args) -> None:
    """
    The current script has as only purpose to get the Monte Carlo Dropout samples, save them,
    and then calculate the entropy and save those quantities for further analysis. This will do this for the InD BDD set
    and one chosen OoD set
    :param args: Configuration class parameters
    :return: None
    """
    # Setup config
    cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)
    # Make sure only 1 data point is processed at a time. This simulates
    # deployment.
    cfg.defrost()
    # cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 1

    cfg.MODEL.DEVICE = device.type
    # Set up number of cpu threads#
    # torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

    # Create inference output directory and copy inference config file to keep
    # track of experimental settings
    inference_output_dir = get_inference_output_dir(
        cfg["OUTPUT_DIR"],
        args.test_dataset,
        args.inference_config,
        args.image_corruption_level,
    )
    ind_dataset = args.config_file.split("-Detection")[0][-3:].lower()
    assert ind_dataset in ("bdd", "voc")
    assert cfg.PROBABILISTIC_INFERENCE.OOD_DATASET in (
        "coco_ood_val_bdd",
        "openimages_ood_val",
        "coco_ood_val",
        "bdd_custom_val",
        "coco_ood_new",
        "coco_ood_near",
        "coco_ood_all_new",
        "openimages_ood_new",
        "openimages_ood_near",
        "openimages_ood_all_new",
        "coco_ood_new_wrt_bdd",
        "openimages_ood_new_wrt_bdd"
    )

    using_vos = True if "vos.yaml" in args.config_file else False
    using_regnet = True if "regnetx.yaml" in args.config_file else False
    using_vos_or_regnet = using_vos or using_regnet
    os.makedirs(inference_output_dir, exist_ok=True)
    copyfile(
        args.inference_config,
        os.path.join(inference_output_dir, os.path.split(args.inference_config)[-1]),
    )
    # Samples save folder
    SAVE_FOLDER = (f"./Extracted_latent_samples/boxes/ind_{ind_dataset}/"
                   f"{'_'.join(map(str, cfg.PROBABILISTIC_INFERENCE.BOXES.HOOKED_LAYERS))}")
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    ##################################################################################
    # Prepare predictor and data loaders
    ##################################################################################
    # Build predictor
    predictor = build_predictor(cfg)
    # Hook the layer
    if not using_regnet:
        hook_dictionary = {
            "Backbone": predictor.model.backbone,
            "RPN_head": predictor.model.proposal_generator.rpn_head,
            "RPN_inter": predictor.model.proposal_generator.rpn_head.conv,
            "Bb2": predictor.model.backbone.bottom_up.res2[0].shortcut,
            "Bb3": predictor.model.backbone.bottom_up.res3[0].shortcut,
            "Bb4": predictor.model.backbone.bottom_up.res4[0].shortcut,
            "Bb5": predictor.model.backbone.bottom_up.res5[0].shortcut,
        }

    else:
        hook_dictionary = {
            "Backbone": predictor.model.backbone,
            "RPN_head": predictor.model.proposal_generator.rpn_head,
            "RPN_inter": predictor.model.proposal_generator.rpn_head.conv,
            "Bb2": predictor.model.backbone.bottom_up.s2,
            "Bb3": predictor.model.backbone.bottom_up.s3,
        }

    extraction_type_dictionary = {
        "Backbone": "backbone",
        "RPN_head": "rpn_head",
        "RPN_inter": "rpn_inter",
        "Bb2": "shortcut",
        "Bb3": "shortcut",
        "Bb4": "shortcut",
        "Bb5": "shortcut",
    }
    hooked_layers = [
        Hook(
            hook_dictionary[layer]
        ) for layer in cfg.PROBABILISTIC_INFERENCE.BOXES.HOOKED_LAYERS
    ]
    # Put model in evaluation mode
    predictor.model.eval()

    if EXTRACT_IND:
        # Build In Distribution valid and test data loader
        if ind_dataset == "voc" or args.test_dataset == "voc_custom_val":
            # The following split proportions correspond to: 0.5 if ind is VOC (and test is also VOC)
            # and to 0.01 when BDD is InD and VOC is OoD (test dataset voc)
            train_voc_args = build_ind_voc_train_dataloader_args(cfg=cfg,
                                                                 split_proportion=0.5 if ind_dataset == "voc" else 0.01)
            (
                ind_test_dl_args,
                _,
            ) = build_in_distribution_valid_test_dataloader_args(
                cfg, dataset_name=args.test_dataset, split_proportion=0.41
            )
            ind_dataset_dict = {
                "valid": build_data_loader(**train_voc_args),
                "test": build_data_loader(**ind_test_dl_args)
            }
            del ind_test_dl_args
            del train_voc_args
        # BDD
        else:
            (
                ind_valid_dl_args,
                ind_test_dl_args,
            ) = build_in_distribution_valid_test_dataloader_args(
                cfg, dataset_name=args.test_dataset, split_proportion=0.8
            )
            ind_dataset_dict = {
                "valid": build_data_loader(**ind_valid_dl_args),
                "test": build_data_loader(**ind_test_dl_args)
            }
            del ind_valid_dl_args
            del ind_test_dl_args

    if EXTRACT_OOD:
        # Build Out of Distribution test data loader
        ood_data_loader_args = build_ood_dataloader_args(cfg)
        ood_test_data_loader = build_data_loader(**ood_data_loader_args)
        ood_ds_name = ood_ds_name_dict[cfg.PROBABILISTIC_INFERENCE.OOD_DATASET]
        del ood_data_loader_args

    if VISUALIZE_N_SAMPLES:
        # InD
        if EXTRACT_IND:
            visualize_predictions(
                data_loader=ind_dataset_dict["test"],
                dataset_name=ind_dataset,
                is_ind=True,
                model=predictor,
                energy_threshold=0 if not using_vos_or_regnet else 8.868,
                save_directory=f"Inference_examples/ind_{ind_dataset}{'/vos' if using_vos else ''}{'/vos_regnet' if using_regnet else ''}",
                cfg=cfg,
                num_samples=VISUALIZE_N_SAMPLES
            )
        if EXTRACT_OOD:
            # OoD
            visualize_predictions(
                data_loader=ood_test_data_loader,
                dataset_name=ood_ds_name,
                is_ind=False,
                model=predictor,
                energy_threshold=0 if not using_vos else 8.868,
                save_directory=f"Inference_examples/ind_{ind_dataset}{'/vos' if using_vos else ''}{'/vos_regnet' if using_regnet else ''}",
                cfg=cfg,
                num_samples=VISUALIZE_N_SAMPLES
            )

    samples_extractor = BoxFeaturesExtractor(
        model=predictor,
        hooked_layers=hooked_layers,
        device=device,
        output_sizes=cfg.PROBABILISTIC_INFERENCE.BOXES.ROI_OUTPUT_SIZES,
        architecture='rcnn',
        sampling_ratio=cfg.PROBABILISTIC_INFERENCE.BOXES.ROI_SAMPLING_RATIO,
        return_raw_predictions=False,
        return_stds=False,
        hook_layer_output=True,
        rcnn_extraction_type=extraction_type_dictionary[cfg.PROBABILISTIC_INFERENCE.BOXES.HOOKED_LAYERS[0]],
        extract_noise_entropies=cfg.PROBABILISTIC_INFERENCE.BOXES.EXTRACT_DROPBLOCK_ENTROPY,
        dropblock_prob=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.DROPBLOCK_PROB,
        dropblock_size=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.DROPBLOCK_SIZE,
        n_mcd_reps=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.N_MCD_STEPS
    )

    print(
        f"Extracting from layers {cfg.PROBABILISTIC_INFERENCE.BOXES.HOOKED_LAYERS} "
        f"prob {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST} "
        f"{'imlvl ' if not cfg.PROBABILISTIC_INFERENCE.BOXES.ENABLE else 'boxes'}"
        f"{'entropy ' if cfg.PROBABILISTIC_INFERENCE.BOXES.EXTRACT_DROPBLOCK_ENTROPY == True else ''}"
        f"{'dp' + str(cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.DROPBLOCK_PROB) + ' ' if cfg.PROBABILISTIC_INFERENCE.BOXES.EXTRACT_DROPBLOCK_ENTROPY == True else ''}"
        f"{'ds' + str(cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.DROPBLOCK_SIZE) + ' ' if cfg.PROBABILISTIC_INFERENCE.BOXES.EXTRACT_DROPBLOCK_ENTROPY == True else ''}"
        f"{'mcd' + str(cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.N_MCD_STEPS) + ' ' if cfg.PROBABILISTIC_INFERENCE.BOXES.EXTRACT_DROPBLOCK_ENTROPY == True else ''}"
    )
    ###################################################################################################
    # Perform MCD inference and save samples
    ###################################################################################################
    if EXTRACT_IND:
        for split, dataloader in ind_dataset_dict.items():
            # Get Monte-Carlo samples
            print(f"Extracting InD {ind_dataset} {split} latent samples {'boxes' if cfg.PROBABILISTIC_INFERENCE.BOXES.ENABLE else 'imlvl'}")
            ind_samples = samples_extractor.get_ls_samples(data_loader=dataloader)
            # Save MC samples
            file_name = (
                f"{SAVE_FOLDER}/ind_{ind_dataset}_{split}_"
                f"{'roi_s' + ''.join(map(str, cfg.PROBABILISTIC_INFERENCE.BOXES.ROI_OUTPUT_SIZES)) + '_' if cfg.PROBABILISTIC_INFERENCE.BOXES.ENABLE else ''}"
                f"{'roi_sr' + str(cfg.PROBABILISTIC_INFERENCE.BOXES.ROI_SAMPLING_RATIO) + '_' if cfg.PROBABILISTIC_INFERENCE.BOXES.ENABLE else ''}"
                f"infc_{str(int(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST * 100))}_"
                f"{'h_z_' if cfg.PROBABILISTIC_INFERENCE.BOXES.EXTRACT_DROPBLOCK_ENTROPY == True else ''}"
                f"{'dp'+str(int(10*cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.DROPBLOCK_PROB))+'_' if cfg.PROBABILISTIC_INFERENCE.BOXES.EXTRACT_DROPBLOCK_ENTROPY  == True else ''}"
                f"{'ds'+str(cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.DROPBLOCK_SIZE)+'_' if cfg.PROBABILISTIC_INFERENCE.BOXES.EXTRACT_DROPBLOCK_ENTROPY  == True else ''}"
                f"{'mcd'+str(cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.N_MCD_STEPS)+'_' if cfg.PROBABILISTIC_INFERENCE.BOXES.EXTRACT_DROPBLOCK_ENTROPY  == True else ''}"
                f"{'vos_' if using_vos else ''}"
                f"{'regnet_' if using_regnet else ''}"
                f"ls_samples.pt"
            )
            torch.save(ind_samples, file_name)
            print(f"Saved ind {ind_dataset} {split} in {file_name}")
        del ind_samples

    if EXTRACT_OOD:
        # OoD
        # # Get Monte-Carlo samples
        print(f"Extracting OOD {ood_ds_name} latent samples {'boxes' if cfg.PROBABILISTIC_INFERENCE.BOXES.ENABLE else 'imlvl'}")
        ood_samples = samples_extractor.get_ls_samples(data_loader=ood_test_data_loader)
        file_name = (
            f"{SAVE_FOLDER}/ood_{ood_ds_name}_"
            f"{'roi_s' + ''.join(map(str, cfg.PROBABILISTIC_INFERENCE.BOXES.ROI_OUTPUT_SIZES)) + '_' if cfg.PROBABILISTIC_INFERENCE.BOXES.ENABLE else ''}"
            f"{'roi_sr' + str(cfg.PROBABILISTIC_INFERENCE.BOXES.ROI_SAMPLING_RATIO) + '_' if cfg.PROBABILISTIC_INFERENCE.BOXES.ENABLE else ''}"
            f"infc_{str(int(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST * 100))}_"
            f"{'h_z_' if cfg.PROBABILISTIC_INFERENCE.BOXES.EXTRACT_DROPBLOCK_ENTROPY == True else ''}"
            f"{'dp'+str(int(10*cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.DROPBLOCK_PROB))+'_' if cfg.PROBABILISTIC_INFERENCE.BOXES.EXTRACT_DROPBLOCK_ENTROPY  == True else ''}"
            f"{'ds'+str(cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.DROPBLOCK_SIZE)+'_' if cfg.PROBABILISTIC_INFERENCE.BOXES.EXTRACT_DROPBLOCK_ENTROPY  == True else ''}"
            f"{'mcd'+str(cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.N_MCD_STEPS)+'_' if cfg.PROBABILISTIC_INFERENCE.BOXES.EXTRACT_DROPBLOCK_ENTROPY  == True else ''}"
            f"{'vos_' if using_vos else ''}"
            f"{'regnet_' if using_regnet else ''}"
            f"ls_samples.pt"
        )
        torch.save(ood_samples, file_name)
        print(f"Saved ood {ood_ds_name} in {file_name}")


    # Analysis of the calculated samples is performed in another script!
    fc_params = predictor.model.roi_heads.box_predictor.cls_score.state_dict()
    torch.save(fc_params, f"{SAVE_FOLDER}/fc_params{'_vos' if using_vos else ''}{'_regnet' if using_regnet else ''}.pt")
    print(f"Saved fc_params in {SAVE_FOLDER}/fc_params{'_vos' if using_vos else ''}{'_regnet' if using_regnet else ''}.pt")
    print("Done!")


ood_ds_name_dict = {
    "coco_ood_new": "coco_new",
    "coco_ood_near": "coco_near",
    "openimages_ood_new": "oi_new",
    "openimages_ood_near": "oi_near",
    "coco_ood_val_bdd": "coco",
    "coco_ood_val": "coco",
    "openimages_ood_val": "openimages",
    "coco_ood_new_wrt_bdd": "coco_new",
    "openimages_ood_new_wrt_bdd": "oi_new",
}


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    print("Command Line Args:", args)
    # This function checks if there are multiple gpus, then it launches the distributed inference, otherwise it
    # just launches the main function, i.e., would act as a function wrapper passing the args to main
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

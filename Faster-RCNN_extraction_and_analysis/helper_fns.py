from typing import Tuple
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ugly but fast way to test to hook the backbone: get raw output, apply dropblock
# inside the following function
dropout_ext = torch.nn.Dropout(p=0.5)


def visualize_predictions(
    data_loader: DataLoader,
    dataset_name: str,
    is_ind: bool,
    model: torch.nn.Module,
    energy_threshold: float,
    save_directory: str,
    cfg,
    num_samples: int,
):
    if num_samples > 0:
        with torch.no_grad():
            # InD
            for idx, input_im in enumerate(tqdm(data_loader, total=num_samples, desc=f"Saving predictions for {dataset_name}")):
                outputs = model(input_im)

                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)
                # if dataset_dir in input_im[0]['file_name']:
                #     file_name = input_im[0]['file_name'].split(dataset_dir)[1].split('.jpg')[0]
                # else:
                #     file_name = input_im[0]['file_name'].split('.jpg')[0]
                if '/' in input_im[0]['file_name']:
                    file_name = input_im[0]['file_name'].split('/')[-1].split('.jpg')[0]
                else:
                    file_name = input_im[0]['file_name'].split('.jpg')[0]
                    file_name = file_name.split('.png')[0]
                model.visualize_inference(input_im,
                                          outputs,
                                          savedir=save_directory,
                                          name=f"{'ind' if is_ind else 'ood'}_{dataset_name}_{file_name}_"
                                               f"c{int(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST * 100)}"
                                               f"{'_vos' if energy_threshold == 8.868 else ''}",
                                          cfg=cfg,
                                          energy_threshold=energy_threshold,
                                          inference_threshold=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
                if idx >= num_samples:
                    break

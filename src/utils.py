""" Python utilities """
from torch import nn
import torch
from pytorch_msssim import ssim

import csv
import numpy as np
import os
import time
from os.path import join

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from skimage import exposure

def readBaselineCPSNR(path):
    """
    Reads the baseline cPSNR scores from `path`.
    Args:
        filePath: str, path/filename of the baseline cPSNR scores
    Returns:
        scores: dict, of {'imagexxx' (str): score (float)}
    """
    scores = dict()
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            scores[row[0].strip()] = float(row[1].strip())
    return scores


def getImageSetDirectories(root_dir):
    """
    List scene subdirectories under root_dir.
    """
    return [join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(join(root_dir, d))]


class collateFunction():
    """ Util class to create padded batches of data. """

    def __init__(self, config, min_L=32):
        """
        Args:
            min_L: int, pad length
        """
        self.config = config
        self.min_L = min_L

    def __call__(self, batch):
        return self.collateFunction(batch)

    def collateFunction(self, batch):
        """
        Custom collate function to adjust a variable number of low-res images.
        Args:
            batch: list of imageset dicts
        Returns:
            padded_lr_batch: tensor (B, min_L, C, H, W), low-resolution images
            padded_lm_batch: tensor (B, min_L, H, W), low-resolution masks
            alpha_batch: tensor (B, min_L), low-resolution indicators (0 if padded view, 1 otherwise)
            hr_batch: tensor (B, H_hr, W_hr), high-resolution images (None if not train)
            hm_batch: tensor (B, H_hr, W_hr), high-resolution status maps (None if not train)
            isn_batch: list of imageset names
        """

        lr_batch = []   # list of [L, C, H, W] tensors
        lm_batch = []   # list of [L, H, W] tensors
        alpha_batch = []  # list of [L] tensors
        hr_batch = []   # list of [H_hr, W_hr] tensors
        hm_batch = []   # list of [H_hr, W_hr] tensors
        isn_batch = []  # list of names (str)

        train_batch = True

        for imageset in batch:
            lrs = imageset['lr']         # Tensor, shape [L, C, H, W]
            lr_maps = imageset['lr_maps']  # Tensor, shape [L, H, W]
            L, C, H, W = lrs.shape

            # Pad or truncate so each scene has exactly min_L frames
            if L >= self.min_L:
                lr_batch.append(lrs[:self.min_L])         # [min_L, C, H, W]
                lm_batch.append(lr_maps[:self.min_L])     # [min_L, H, W]
                alpha_batch.append(torch.ones(self.min_L))
            else:
                # Need to pad
                pad_L = self.min_L - L
                pad_lrs = torch.zeros((pad_L, C, H, W), dtype=lrs.dtype)
                pad_lm = torch.zeros((pad_L, H, W), dtype=lr_maps.dtype)

                lr_batch.append(torch.cat([lrs, pad_lrs], dim=0))     # [min_L, C, H, W]
                lm_batch.append(torch.cat([lr_maps, pad_lm], dim=0))   # [min_L, H, W]

                alpha_batch.append(torch.cat([
                    torch.ones(L), 
                    torch.zeros(pad_L)
                ], dim=0))  # [min_L]

            hr = imageset['hr']        # Tensor or None
            if train_batch and hr is not None:
                hr_batch.append(hr)    # each [H_hr, W_hr]
            else:
                train_batch = False

            hm_batch.append(imageset['hr_map'])  # each [H_hr, W_hr]
            isn_batch.append(imageset['name'])

        # Stack everything along the batch dimension
        # padded_lr_batch: [B, min_L, C, H, W]
        padded_lr_batch = torch.stack(lr_batch, dim=0)
        # padded_lm_batch: [B, min_L, H, W]
        padded_lm_batch = torch.stack(lm_batch, dim=0)
        # alpha_batch: [B, min_L]
        alpha_batch = torch.stack(alpha_batch, dim=0)

        if train_batch:
            # hr_batch: [B, H_hr, W_hr]
            hr_batch = torch.stack(hr_batch, dim=0)
            # hm_batch: [B, H_hr, W_hr]
            hm_batch = torch.stack(hm_batch, dim=0)
        else:
            hr_batch = None
            hm_batch = None

        ########## data augmentation (if requested) ##########
        if self.config["training"]["data_arguments"]:
            # shapes: lr [B, min_L, C, H, W], lm [B, min_L, H, W],
            # hr [B, H_hr, W_hr], hm [B, H_hr, W_hr]
            np.random.seed(int(1000 * time.time()) % 2**32)
            if np.random.random() <= self.config["training"]["probability of flipping horizontally"]:
                padded_lr_batch = torch.flip(padded_lr_batch, [4])  # flip width dim of LR
                padded_lm_batch = torch.flip(padded_lm_batch, [3])  # flip width dim of LR masks
                hr_batch = torch.flip(hr_batch, [2])  # flip width dim of HR
                hm_batch = torch.flip(hm_batch, [2])  # flip width dim of HR masks

            np.random.seed(int(1000 * time.time()) % 2**32)
            if np.random.random() <= self.config["training"]["probability of flipping vertically"]:
                padded_lr_batch = torch.flip(padded_lr_batch, [3])  # flip height dim of LR
                padded_lm_batch = torch.flip(padded_lm_batch, [2])  # flip height dim of LR masks
                hr_batch = torch.flip(hr_batch, [1])  # flip height dim of HR
                hm_batch = torch.flip(hm_batch, [1])  # flip height dim of HR masks

            np.random.seed(int(1000 * time.time()) % 2**32)
            k_num = np.random.choice(
                a=self.config["training"]["corresponding angles(x90)"],
                replace=True,
                p=self.config["training"]["probability of rotation"]
            )
            # Rotate around (H, W) dims for both LR and HR
            padded_lr_batch = torch.rot90(padded_lr_batch, k=k_num, dims=[3, 4])
            padded_lm_batch = torch.rot90(padded_lm_batch, k=k_num, dims=[2, 3])
            hr_batch = torch.rot90(hr_batch, k=k_num, dims=[1, 2])
            hm_batch = torch.rot90(hm_batch, k=k_num, dims=[1, 2])
            np.random.seed(int(1000 * time.time()) % 2**32)

        return padded_lr_batch, padded_lm_batch, alpha_batch, hr_batch, hm_batch, isn_batch


def get_loss(srs, hrs, hr_maps, metric='cMSE'):
    """
    Computes ESA loss for each instance in a batch.
    Args:
        srs: tensor (B, W, H), super resolved images
        hrs: tensor (B, W, H), high-res images
        hr_maps: tensor (B, W, H), high-res status maps
    Returns:
        loss: tensor (B), metric for each super resolved image.
    """
    # ESA Loss: https://kelvins.esa.int/proba-v-super-resolution/scoring/

    if metric == 'L1':
        border = 3
        max_pixels_shifts = 2 * border
        size_image = hrs.shape[1]
        srs = srs.squeeze(1)
        cropped_predictions = srs[:, border:size_image - border, border:size_image - border]

        X = []
        for i in range(max_pixels_shifts + 1):  # range(7)
            for j in range(max_pixels_shifts + 1):  # range(7)
                cropped_labels = hrs[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
                cropped_y_mask = hr_maps[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
                cropped_labels_masked = cropped_labels * cropped_y_mask
                cropped_predictions_masked = cropped_predictions * cropped_y_mask
                total_pixels_masked = torch.sum(cropped_y_mask, dim=[1,2])

                # bias brightness
                b = (1.0 / total_pixels_masked) * torch.sum(cropped_labels_masked-cropped_predictions_masked, dim=[1,2])
                b = b.unsqueeze(-1).unsqueeze(-1)

                corrected_cropped_predictions = cropped_predictions_masked + b
                corrected_cropped_predictions = corrected_cropped_predictions * cropped_y_mask

                l1_loss = (1.0 / total_pixels_masked) * torch.sum(torch.abs(cropped_labels_masked-corrected_cropped_predictions) , dim=[1,2])
                X.append(l1_loss)
        X = torch.stack(X)
        min_l1 = torch.min(X, 0)[0]
        loss = -10 * torch.log10(min_l1)

        return loss

    if metric == 'L2':
        border = 3
        max_pixels_shifts = 2 * border
        size_image = hrs.shape[1]
        srs = srs.squeeze(1)
        cropped_predictions = srs[:, border:size_image - border, border:size_image - border]

        X = []
        for i in range(max_pixels_shifts + 1):  # range(7)
            for j in range(max_pixels_shifts + 1):  # range(7)
                cropped_labels = hrs[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
                cropped_y_mask = hr_maps[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
                cropped_labels_masked = cropped_labels * cropped_y_mask
                cropped_predictions_masked = cropped_predictions * cropped_y_mask
                total_pixels_masked = torch.sum(cropped_y_mask, dim=[1,2])

                # bias brightness
                b = (1.0 / total_pixels_masked) * torch.sum(cropped_labels_masked-cropped_predictions_masked, dim=[1,2])
                b = b.unsqueeze(-1).unsqueeze(-1)

                corrected_cropped_predictions = cropped_predictions_masked + b
                corrected_cropped_predictions = corrected_cropped_predictions * cropped_y_mask

                corrected_mse = (1.0 / total_pixels_masked) * torch.sum((cropped_labels_masked-corrected_cropped_predictions)**2, dim=[1, 2])
                cPSNR = 10.0 * torch.log10((1.0**2)/corrected_mse)
                X.append(cPSNR)
        X = torch.stack(X)
        max_cPSNR = torch.max(X, 0)[0]

        return max_cPSNR

    if metric == 'SSIM':
        border = 3
        max_pixels_shifts = 2 * border
        size_image = hrs.shape[1]
        srs = srs.squeeze(1)
        cropped_predictions = srs[:, border:size_image - border, border:size_image - border]

        X = []
        for i in range(max_pixels_shifts + 1):  # range(7)
            for j in range(max_pixels_shifts + 1):  # range(7)
                cropped_labels = hrs[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
                cropped_y_mask = hr_maps[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
                cropped_labels_masked = cropped_labels * cropped_y_mask
                cropped_predictions_masked = cropped_predictions * cropped_y_mask

                total_pixels_masked = torch.sum(cropped_y_mask, dim=[1,2])

                # bias brightness
                b = (1.0 / total_pixels_masked) * torch.sum(cropped_labels_masked-cropped_predictions_masked, dim=[1,2])
                b = b.unsqueeze(-1).unsqueeze(-1)
                corrected_cropped_predictions = cropped_predictions_masked + b
                corrected_cropped_predictions = corrected_cropped_predictions * cropped_y_mask

                Y = []
                for k in range(corrected_cropped_predictions.shape[0]):
                    cSSIM = ssim(corrected_cropped_predictions[k].unsqueeze(0).unsqueeze(0),
                                 cropped_labels_masked[k].unsqueeze(0).unsqueeze(0),
                                 data_range=1.0,
                                 size_average=False)
                    Y.append(cSSIM)
                Y = torch.stack(Y).squeeze(-1)
                X.append(Y)
        X = torch.stack(X)
        max_cSSIM = torch.max(X, 0)[0]

        return max_cSSIM

    return -1


def get_crop_mask(patch_size, crop_size):
    """
    Computes a mask to crop borders.
    Args:
        patch_size: int, size of patches
        crop_size: int, size to crop (border)
    Returns:
        torch_mask: tensor (1, 1, 3*patch_size, 3*patch_size), mask
    """
    mask = np.ones((1, 1, 3 * patch_size, 3 * patch_size))  # crop_mask for loss (B, C, W, H)
    mask[0, 0, :crop_size, :] = 0
    mask[0, 0, -crop_size:, :] = 0
    mask[0, 0, :, :crop_size] = 0
    mask[0, 0, :, -crop_size:] = 0
    torch_mask = torch.from_numpy(mask).type(torch.FloatTensor)
    return torch_mask

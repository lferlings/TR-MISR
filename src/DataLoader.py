import os
import glob
import heapq
from os.path import join, basename, isfile
from collections import OrderedDict

import numpy as np
from skimage import io
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

import torch
from torch.utils.data import Dataset


def get_patch(img, x, y, size=32):
    """
    Extract a square patch from (x,y) top-left, supports multi-dim arrays.
    """
    return img[..., x:(x + size), y:(y + size)]


class ImageSet(OrderedDict):
    """
    Holds all assets for one scene: lr, lr_maps, hr, hr_map, clearances.
    """
    def __repr__(self):
        info = f"{'name':>10} : {self['name']}"
        for k, v in self.items():
            if hasattr(v, 'shape'):
                info += f"\n{k:>10} : {v.shape} {type(v).__name__} ({v.dtype})"
            else:
                info += f"\n{k:>10} : {type(v).__name__} ({v})"
        return info


# Landsat QA_PIXEL bit for 'Clear' flag
CLEAR_BIT = 6
# Multispectral low-res bands
LR_BANDS = [1,2,3,4,5,6,7,9]


def read_imageset(imset_dir,
                  create_patches=False,
                  patch_size=64,
                  seed=None,
                  top_k=None,
                  map_depend=True,
                  std_depend=True):
    """
    Load one Landsat scene directory:
      - lr: [N_frames, C_bands, H, W]
      - lr_maps: [N_frames, H, W]
      - hr: [H', W'] or None
      - hr_map: [H', W'] or None
      - clearances: [N_frames]
    """
    # load clearance scores
    clear_path = join(imset_dir, 'clearance.npy')
    if not isfile(clear_path):
        raise FileNotFoundError(f"clearance.npy not found in {imset_dir}")
    clearances = np.load(clear_path)

    # find QA_PIXEL paths and derive frame IDs
    qa_paths = sorted(glob.glob(join(imset_dir, 'QA', '*QA_PIXEL*.TIF')))
    frame_ids = [basename(p)[:-len('_QA_PIXEL.TIF')] for p in qa_paths]

    # load multispectral LR and QA masks per frame
    lr_list, mask_list = [], []
    for fid in frame_ids:
        bands = []
        for b in LR_BANDS:
            fpath = join(imset_dir, 'LR', f"{fid}_B{b}.TIF")
            bands.append(io.imread(fpath).astype(np.float32))
        lr_list.append(np.stack(bands, axis=0))  # [C, H, W]
        qa = io.imread(join(imset_dir, 'QA', f"{fid}_QA_PIXEL.TIF")).astype(np.uint16)
        mask_list.append(((qa >> CLEAR_BIT) & 1).astype(bool))  # [H, W]

    lr_stack = np.stack(lr_list, axis=0)       # [N, C, H, W]
    mask_stack = np.stack(mask_list, axis=0)   # [N, H, W]

    # ensure counts match
    assert lr_stack.shape[0] == mask_stack.shape[0] == clearances.shape[0], (
        f"Counts mismatch: frames={lr_stack.shape[0]}, masks={mask_stack.shape[0]}, scores={clearances.shape[0]}"
    )

    # select top_k clearest frames
    if top_k is not None and top_k > 0:
        k = min(top_k, len(clearances))
        idxs = heapq.nlargest(k, range(len(clearances)), clearances.take)
    else:
        idxs = list(range(len(clearances)))
        idxs.sort(key=lambda i: clearances[i], reverse=True)

    lr_stack = lr_stack[idxs]
    mask_stack = mask_stack[idxs]
    clearances = clearances[idxs]

    # align each frame to the first
    for i in range(1, lr_stack.shape[0]):
        shift_vec = phase_cross_correlation(
            lr_stack[0, 0], lr_stack[i, 0],
            reference_mask=mask_stack[0],
            moving_mask=mask_stack[i],
            return_error=False,
            overlap_ratio=0.99
        )
        lr_stack[i] = shift(lr_stack[i], shift_vec, mode='constant', cval=0)
        mask_stack[i] = shift(mask_stack[i].astype(np.uint8), shift_vec,
                               mode='constant', cval=0).astype(bool)

    # mask out unclear pixels
    if map_depend:
        lr_stack = lr_stack * mask_stack[:, None, :, :]

    # standardize clear pixels per-band
    if std_depend:
        N, C, H, W = lr_stack.shape
        for i in range(N):
            for c in range(C):
                img = lr_stack[i, c]
                m = mask_stack[i]
                if m.any():
                    mu = img[m].mean()
                    sigma = img[m].std() + 1e-4
                    lr_stack[i, c] = np.where(m, (img - mu) / sigma, 0)
                else:
                    lr_stack[i, c] = 0

    # load high-res pan band
    hr_files = glob.glob(join(imset_dir, 'HR', '*.TIF'))
    if hr_files:
        hr_img = io.imread(hr_files[0]).astype(np.float32)
        hr_map = np.ones_like(hr_img, dtype=bool)
    else:
        hr_img, hr_map = None, None

    # optional random patch cropping
    if create_patches:
        if seed is not None:
            np.random.seed(seed)
        max_x = lr_stack.shape[2] - patch_size
        max_y = lr_stack.shape[3] - patch_size
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        lr_stack = get_patch(lr_stack, x, y, patch_size)
        mask_stack = get_patch(mask_stack, x, y, patch_size)
        if hr_img is not None:
            hr_img = get_patch(hr_img, x * 3, y * 3, patch_size * 3)
            hr_map = get_patch(hr_map, x * 3, y * 3, patch_size * 3)

    # pack into ImageSet using torch.tensor instead of from_numpy
    imageset = ImageSet(
        name=basename(imset_dir),
        lr=torch.tensor(lr_stack, dtype=torch.float32),
        lr_maps=torch.tensor(mask_stack, dtype=torch.float32),
        hr=torch.tensor(hr_img, dtype=torch.float32) if hr_img is not None else None,
        hr_map=torch.tensor(hr_map, dtype=torch.float32) if hr_map is not None else None,
        clearances=clearances
    )
    return imageset


class ImagesetDataset(Dataset):
    """
    PyTorch Dataset for loading multiple scenes.
    """
    def __init__(self, scene_dirs, config, seed=None,
                 top_k=-1, map_depend=True, std_depend=True):
        super().__init__()
        self.scene_dirs = scene_dirs
        self.config = config
        self.seed = seed
        self.top_k = top_k
        self.map_depend = map_depend
        self.std_depend = std_depend

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene = self.scene_dirs[idx]
        return read_imageset(
            scene,
            create_patches=self.config.get('create_patches', False),
            patch_size=self.config.get('patch_size', 64),
            seed=self.seed,
            top_k=self.top_k,
            map_depend=self.map_depend,
            std_depend=self.std_depend
        )

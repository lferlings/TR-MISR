#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from skimage import io
from tqdm import tqdm

# bit index for “Clear” in Landsat 8/9 QA_PIXEL (0-based)
CLEAR_BIT = 6


def list_scene_dirs(data_root):
    """
    Returns a list of all scene directories under data_root.
    If subdirs 'train', 'val', or 'test' exist, scenes under those are used.
    Otherwise, scenes are taken directly under data_root.
    """
    root = Path(data_root)
    scenes = []
    for split in ('train', 'val', 'test'):
        split_dir = root / split
        if split_dir.is_dir():
            scenes.extend([p for p in split_dir.iterdir() if p.is_dir()])
    if not scenes:
        scenes = [p for p in root.iterdir() if p.is_dir()]
    return scenes


def save_clearance_scores(scene_dirs):
    """
    For each scene, read QA_PIXEL TIFs, decode clear bit,
    sum clear pixels → clearance score, save clearance.npy.
    """
    for scene in tqdm(scene_dirs, desc="Scenes"):
        qa_dir = scene / "QA"
        if not qa_dir.exists():
            tqdm.write(f"⚠️  no QA/ in {scene}, skipping")
            continue

        qa_files = sorted(qa_dir.glob("*QA_PIXEL*.TIF"))
        if not qa_files:
            tqdm.write(f"⚠️  no QA_PIXEL files in {qa_dir}, skipping")
            continue

        scores = []
        for qa_path in qa_files:
            qa = io.imread(str(qa_path)).astype(np.uint16)
            clear_mask = ((qa >> CLEAR_BIT) & 1).astype(np.uint8)
            scores.append(int(clear_mask.sum()))

        scores = np.array(scores, dtype=np.uint32)
        np.save(scene / "clearance.npy", scores)
        tqdm.write(f"Saved {len(scores)} scores for {scene.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-frame Landsat-8 clearance scores for split data"
    )
    parser.add_argument(
        "--data_root", "-d",
        type=str,
        required=True,
        help="Root folder containing train/, val/, and/or test/ subdirs"
    )
    args = parser.parse_args()

    scenes = list_scene_dirs(args.data_root)
    save_clearance_scores(scenes)
    print("✅ clearance.npy written for each scene")

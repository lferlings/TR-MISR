#!/usr/bin/env python3
import argparse
import re
import shutil
import random
from pathlib import Path

def organize_landsat_with_split(input_dir: Path, output_dir: Path,
                                 train_frac: float, val_frac: float,
                                 seed: int):
    """
    Scan input_dir for Landsat-8 .TIF files, group by scene prefix,
    split into train/val/test, and copy into
    output_dir/{train,val,test}/<scene_id>/{LR,HR,QA}.
    """
    # regex to split prefix and band or QA
    pat = re.compile(r"(?P<prefix>.+)_(?:B(?P<band>\d+)|(?P<qa>QA_PIXEL))\.TIF$",
                     re.IGNORECASE)
    # collect all prefixes
    prefixes = set()
    for tif in input_dir.glob("*.TIF"):
        m = pat.match(tif.name)
        if m:
            prefixes.add(m.group("prefix"))
    prefixes = sorted(prefixes)

    # shuffle and split
    random.seed(seed)
    random.shuffle(prefixes)
    n = len(prefixes)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_p = set(prefixes[:n_train])
    val_p = set(prefixes[n_train:n_train + n_val])
    test_p = set(prefixes[n_train + n_val:])

    # helper to determine split
    def which_split(pref):
        if pref in train_p:
            return 'train'
        if pref in val_p:
            return 'val'
        return 'test'

    # bands classification
    LR_BANDS = {str(b) for b in [1,2,3,4,5,6,7,9]}
    HR_BAND = '8'

    # process each file
    for tif in input_dir.glob("*.TIF"):
        m = pat.match(tif.name)
        if not m:
            print(f"Skipping unrecognized: {tif.name}")
            continue
        prefix = m.group("prefix")
        band = m.group("band")
        is_qa = bool(m.group("qa"))
        split = which_split(prefix)
        # prepare target subdirs
        base = output_dir / split / prefix
        lr_dir = base / 'LR'
        hr_dir = base / 'HR'
        qa_dir = base / 'QA'
        for d in (lr_dir, hr_dir, qa_dir):
            d.mkdir(parents=True, exist_ok=True)
        # copy
        if band == HR_BAND:
            shutil.copy2(tif, hr_dir / tif.name)
        elif band in LR_BANDS:
            shutil.copy2(tif, lr_dir / tif.name)
        elif is_qa:
            shutil.copy2(tif, qa_dir / tif.name)
        else:
            # ignore other bands
            continue

    print(f"Split complete: {len(train_p)} train, {len(val_p)} val, {len(test_p)} test scenes.")


def main():
    p = argparse.ArgumentParser(
        description="Organize Landsat-8 .TIFs into train/val/test splits for TR-MISR"
    )
    p.add_argument('input_dir', type=Path,
                   help="Folder with all LC08_..._TIF files")
    p.add_argument('output_dir', type=Path,
                   help="Root for train/, val/, test/ subdirs")
    p.add_argument('--train_frac', type=float, default=0.8,
                   help="Fraction of scenes for training")
    p.add_argument('--val_frac', type=float, default=0.1,
                   help="Fraction of scenes for validation")
    p.add_argument('--seed', type=int, default=42,
                   help="Random seed for shuffling scenes")
    args = p.parse_args()
    organize_landsat_with_split(args.input_dir, args.output_dir,
                                 args.train_frac, args.val_frac, args.seed)

if __name__ == '__main__':
    main()

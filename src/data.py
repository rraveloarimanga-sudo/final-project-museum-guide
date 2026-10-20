"""Data loading, label encoding, stratified splits and PyTorch Dataset.

The dataset is produced by scripts/make_wikiart_subset.py as an ImageFolder tree:

    data/wikiart_style5/
        Impressionism/s03_r0457.jpg ...
        Baroque/...
        metadata.csv   # path,label,label_idx,id,style,genre,artist,shard,row,orig_width,orig_height
        subset_info.json

`load_raw()` reads that metadata.csv. Everything downstream relies on `load_dataframe()`
returning a DataFrame with:
    image   : absolute path to the JPEG (str)
    style   : str label (e.g. "Impressionism")
    genre   : str label (e.g. "portrait")
    artist  : str
Only `load_raw()` is specific to the subset layout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.config import SEED, SUBSET_DIR, TARGET, TEST_SIZE, VAL_SIZE


def load_raw() -> pd.DataFrame:
    """Read metadata.csv of the built subset and resolve image paths to absolute files."""
    meta = SUBSET_DIR / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(
            f"{meta} not found. Build the subset first, e.g.\n"
            f"  python scripts/make_wikiart_subset.py --label style "
            f"--classes Impressionism Baroque Cubism Ukiyo_e Pop_Art "
            f"--per-class 300 --size 128 --out {SUBSET_DIR}"
        )
    df = pd.read_csv(meta)
    df["image"] = df["path"].map(lambda p: str((SUBSET_DIR / p).resolve()))
    return df


def load_dataframe(min_per_class: int = 20) -> pd.DataFrame:
    df = load_raw().copy()
    df = df.dropna(subset=["image", TARGET]).reset_index(drop=True)
    # Drop classes too small to appear in all three splits
    counts = df[TARGET].value_counts()
    keep = counts[counts >= min_per_class].index
    dropped = sorted(set(counts.index) - set(keep))
    if dropped:
        print(f"[data] dropping rare classes (<{min_per_class}): {dropped}")
    return df[df[TARGET].isin(keep)].reset_index(drop=True)


def encode_labels(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    classes = sorted(df[TARGET].unique())
    index = {c: i for i, c in enumerate(classes)}
    return df[TARGET].map(index).to_numpy(), classes


def split_indices(y: np.ndarray) -> dict[str, np.ndarray]:
    """Stratified train / val / test split, fixed seed. Test is touched once, at the end."""
    idx = np.arange(len(y))
    trainval, test = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    rel_val = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        trainval, test_size=rel_val, stratify=y[trainval], random_state=SEED
    )
    return {"train": train, "val": val, "test": test}


def open_image(item) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, np.ndarray):
        return Image.fromarray(item).convert("RGB")
    path = Path(item)
    if not path.is_absolute():
        path = DATA_DIR / path
    return Image.open(path).convert("RGB")


class ArtworkDataset(Dataset):
    """Optionally pre-decodes every image once into memory (cache_size x cache_size uint8).

    On CPU this is the difference between re-reading 1500 JPEGs from disk every epoch
    (very slow, worse from iCloud) and decoding them a single time up front.
    """

    def __init__(self, images, labels, transform, cache_size: int | None = None):
        self.images = list(images)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.transform = transform
        self.cache = None
        if cache_size is not None:
            from tqdm import tqdm
            self.cache = [
                np.asarray(open_image(im).resize((cache_size, cache_size)))
                for im in tqdm(self.images, desc="caching images", leave=False)
            ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        src = self.cache[i] if self.cache is not None else self.images[i]
        return self.transform(open_image(src)), self.labels[i]


def standardise_from_train(datasets: dict) -> tuple[list[float], list[float]]:
    """Per-channel x <- (x - mean) / std, with mean and std computed on the TRAIN images only.

    Val and test reuse the train statistics: they stand for future visitors, so looking at
    them (even for a mean) would leak information. Requires datasets built with cache_size.
    """
    from torchvision import transforms as T

    train = datasets["train"]
    if train.cache is None:
        raise ValueError("standardise_from_train needs datasets built with cache_size")
    pix = np.stack(train.cache).astype(np.float32) / 255.0   # (N, H, W, 3), same scale as ToTensor
    mean = pix.mean(axis=(0, 1, 2)).tolist()
    std = pix.std(axis=(0, 1, 2)).tolist()
    norm = T.Normalize(mean, std)
    for ds in datasets.values():
        ds.transform = T.Compose([ds.transform, norm])
    print(f"[data] train pixel mean (RGB) {[round(m, 3) for m in mean]} | std {[round(s, 3) for s in std]}")
    return mean, std


def make_datasets(df: pd.DataFrame, transforms: dict, cache_size: int | None = None) -> tuple[dict, list[str]]:
    """transforms: {"train": tf, "val": tf, "test": tf}. cache_size preloads images in memory."""
    y, classes = encode_labels(df)
    splits = split_indices(y)
    datasets = {
        name: ArtworkDataset(df["image"].iloc[ix], y[ix], transforms[name], cache_size)
        for name, ix in splits.items()
    }
    return datasets, classes

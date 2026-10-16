"""Data loading, label encoding, stratified splits and PyTorch Dataset.

Everything downstream only relies on `load_dataframe()` returning a DataFrame with:
    image       : file path (str) OR PIL.Image OR HxWx3 uint8 array
    style       : str label (e.g. "Impressionism")
    genre       : str label (e.g. "portrait")
    title, artist, description : optional text, used by the search component
Only `load_raw()` needs to change to match the course's data file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.config import DATA_DIR, SEED, TARGET, TEST_SIZE, VAL_SIZE


def load_raw() -> pd.DataFrame:
    """Adapter around the provided data-loading file.

    TODO: replace the body with a call to the function in the course's Python file,
    then rename its columns to the schema documented above.
    """
    csv = DATA_DIR / "artworks.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found — plug the course loader into src/data.py::load_raw()"
        )
    return pd.read_csv(csv)


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
    def __init__(self, images, labels, transform):
        self.images = list(images)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(open_image(self.images[i])), self.labels[i]


def make_datasets(df: pd.DataFrame, transforms: dict) -> tuple[dict, list[str]]:
    """transforms: {"train": tf, "val": tf, "test": tf}"""
    y, classes = encode_labels(df)
    splits = split_indices(y)
    datasets = {
        name: ArtworkDataset(df["image"].iloc[ix], y[ix], transforms[name])
        for name, ix in splits.items()
    }
    return datasets, classes

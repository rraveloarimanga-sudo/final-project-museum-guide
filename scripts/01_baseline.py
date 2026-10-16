"""Step 1 — Classical baseline the neural net must beat.

Logistic regression on cheap, hand-made features (downscaled pixels + colour histogram).
This is the "would a simpler model do the job?" reference the director asked about.
Run: python scripts/01_baseline.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import FIG_DIR, SEED
from src.data import encode_labels, load_dataframe, open_image, split_indices
from src.evaluate import plot_confusion, summary
from src.utils import save_metrics, set_seed, timer

FEAT_SIZE = 32   # downscaled pixel grid
HIST_BINS = 16   # per channel


def features(item):
    img = open_image(item).resize((FEAT_SIZE, FEAT_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    pixels = arr.flatten()
    hist = np.concatenate([
        np.histogram(arr[:, :, c], bins=HIST_BINS, range=(0, 1), density=True)[0]
        for c in range(3)
    ])
    return np.concatenate([pixels, hist])


def main():
    set_seed(SEED)
    df = load_dataframe()
    y, classes = encode_labels(df)
    splits = split_indices(y)

    X = np.stack([features(it) for it in tqdm(df["image"], desc="features")])
    scaler = StandardScaler().fit(X[splits["train"]])
    Xs = scaler.transform(X)

    timings: dict = {}
    with timer("baseline_fit", timings):
        clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
        clf.fit(Xs[splits["train"]], y[splits["train"]])

    results = {"n_features": int(X.shape[1]), "classes": classes, "timings_s": timings}
    for name in ("val", "test"):
        ix = splits[name]
        pred = clf.predict(Xs[ix])
        results[name] = summary(y[ix], pred, classes)
        print(f"[baseline] {name} accuracy = {results[name]['accuracy']:.3f}")
        if name == "test":
            plot_confusion(y[ix], pred, classes, FIG_DIR / "baseline_confusion_test.png",
                           "Baseline (logistic regression) — test")

    save_metrics("baseline", results)


if __name__ == "__main__":
    main()

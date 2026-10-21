"""Step 6 — Benchmark for the director: accuracy vs cost on a modest CPU.

Reads the accuracies saved by the previous scripts, then measures on THIS machine, one image at a
time (what a kiosk does): time to answer, number of parameters, size on disk.
Speed does not depend on the values of the weights, only on the architecture, so we time
fresh models with the same shapes as the trained ones.
Run: python scripts/06_benchmark.py   (after 01, 02, 04 and 05)
"""

# ---------------------------------------------------------------------------------------------
# 1. Libraries
# ---------------------------------------------------------------------------------------------
import json                    # json: read the result files written by the previous scripts
import sys                     # sys: lets us change Python's search path (used just below)
import time                    # time: precise stopwatch, to measure answer times
from pathlib import Path       # Path: handles file paths; used below to locate the project folder

# Add the project folder (two levels above this file) to Python's search path,
# so that "from src.xxx import ..." works.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np                                    # numpy: arrays of numbers
import torch                                          # PyTorch: runs the two neural networks
from PIL import Image                                 # PIL: image handling (resize, like script 01)
from sklearn.linear_model import LogisticRegression   # the simple model of script 01
from torch import nn                                  # nn: building blocks (Linear layer)
from torchvision import models                        # models: the ResNet-18 architecture

from src.config import METRICS_DIR, MLP_IMG_SIZE, SEED    # results folder, 64 px, seed 42
from src.mlp import MLP                                    # the dense network class of script 02
from src.utils import save_metrics, set_seed               # save results, fix randomness

N_REPEAT = 100                 # number of timed predictions per model (more = more reliable average)


# ---------------------------------------------------------------------------------------------
# 2. Helper: read a result file if it exists
# ---------------------------------------------------------------------------------------------
def load(name):
    path = METRICS_DIR / f"{name}.json"                     # e.g. outputs/metrics/resnet.json
    # Return the content as a Python dictionary, or None if the script has not been run yet
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------------------------------------
# 3. Helper: average time of one call to fn, in milliseconds
# ---------------------------------------------------------------------------------------------
def ms_per_call(fn, n=N_REPEAT):
    for _ in range(5):
        fn()                                   # warm-up: the first calls are always slower
    t0 = time.perf_counter()                   # start the stopwatch
    for _ in range(n):
        fn()                                   # n timed calls
    return (time.perf_counter() - t0) / n * 1000    # seconds / n calls * 1,000 = ms per call


# ---------------------------------------------------------------------------------------------
# 4. Main program
# ---------------------------------------------------------------------------------------------
def main():
    set_seed(SEED)                             # fix randomness (seed 42) -> reproducible results
    torch.set_num_threads(1)                   # a modest kiosk: use ONE processor core only
    # A fake 128 x 128 colour photo (random pixels): the content does not change the timing
    photo = Image.fromarray(np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8))

    # ---- 4.1 Simple model: hand-made features + logistic regression (same as script 01)
    def features(img):
        arr = np.asarray(img.resize((32, 32)), dtype=np.float32) / 255.0      # 32 x 32 px, 0-1
        # 16-bin colour histogram for each of the 3 channels (red, green, blue)
        hist = [np.histogram(arr[:, :, c], bins=16, range=(0, 1), density=True)[0] for c in range(3)]
        return np.concatenate([arr.flatten(), *hist])     # 3,072 pixels + 48 histogram values = 3,120
    # A logistic regression of the same shape (3,120 inputs -> 5 styles), fitted on random data:
    # only its shape matters for the timing
    clf = LogisticRegression(max_iter=200).fit(np.random.randn(50, 3120), np.arange(50) % 5)
    # Time = compute the 3,120 features from the photo + predict the style
    t_simple = ms_per_call(lambda: clf.predict(features(photo)[None]))
    n_simple = clf.coef_.size + clf.intercept_.size       # 5 x 3,120 weights + 5 biases = 15,605

    # ---- 4.2 Dense network 12,288 -> 512 -> 256 -> 5 (same shape as script 02)
    mlp = MLP(MLP_IMG_SIZE * MLP_IMG_SIZE * 3, 5, hidden=(512, 256)).eval()   # eval = answer mode
    x_mlp = torch.randn(1, 3, MLP_IMG_SIZE, MLP_IMG_SIZE)     # one fake 64 x 64 colour image
    with torch.no_grad():                                     # no gradients needed to answer
        t_mlp = ms_per_call(lambda: mlp(x_mlp))               # average answer time
    n_mlp = sum(p.numel() for p in mlp.parameters())          # 6,424,581 parameters

    # ---- 4.3 ResNet-18 with a 5-class head (same shape as script 04), 224 x 224 input
    resnet = models.resnet18(weights=None)        # the architecture only: weights do not change speed
    resnet.fc = nn.Linear(512, 5)                 # same new last layer as in script 04
    resnet.eval()                                 # answer mode
    x_res = torch.randn(1, 3, 224, 224)           # one fake 224 x 224 colour image
    with torch.no_grad():                         # no gradients needed to answer
        t_res = ms_per_call(lambda: resnet(x_res), n=30)      # fewer repeats: each call is slower
    n_res = sum(p.numel() for p in resnet.parameters())       # 11,179,077 parameters

    # ---- 4.4 Test accuracies saved by scripts 01, 02 and 04 (None if a script was not run)
    acc = {
        "simple": (load("baseline") or {}).get("test", {}).get("accuracy"),
        "mlp": (load("mlp") or {}).get("test", {}).get("accuracy"),
        "resnet": (load("resnet") or {}).get("test", {}).get("accuracy"),
    }
    # One row per model: accuracy, time per image, number of parameters
    rows = {
        "simple": {"test_acc": acc["simple"], "ms_per_image": t_simple, "n_params": n_simple},
        "mlp": {"test_acc": acc["mlp"], "ms_per_image": t_mlp, "n_params": n_mlp},
        "resnet": {"test_acc": acc["resnet"], "ms_per_image": t_res, "n_params": n_res},
    }
    for r in rows.values():
        r["size_mb"] = r["n_params"] * 4 / 1e6    # each parameter = 4 bytes (float32); 1 MB = 10^6 bytes

    # ---- 4.5 Print the comparison table
    print(f"{'model':8s} {'test acc':>9s} {'ms/image (1 core)':>18s} {'params':>12s} {'size MB':>8s}")
    for name, r in rows.items():
        a = "n/a" if r["test_acc"] is None else f"{r['test_acc']:.1%}"     # "n/a" if not measured
        print(f"{name:8s} {a:>9s} {r['ms_per_image']:>18.1f} {r['n_params']:>12,} {r['size_mb']:>8.1f}")

    # ---- 4.6 Add the search results of script 05, then save everything
    search = load("search")                       # outputs/metrics/search.json, if it exists
    if search:
        print("\nsearch:", {k: v for k, v in search.items() if k in ("tfidf", "embeddings")})
    save_metrics("benchmark", {"classifiers": rows, "search": search, "threads": 1})   # benchmark.json


# Run main() only when the file is launched directly: python scripts/06_benchmark.py
if __name__ == "__main__":
    main()

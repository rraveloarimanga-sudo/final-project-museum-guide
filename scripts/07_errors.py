"""Step 7 — Look at the ResNet's mistakes one by one.

Reloads the fine-tuned ResNet saved by script 04 (no retraining), predicts the test set again and
saves a grid of the test paintings of one true style that were given another style
(default: true Cubism predicted as Pop_Art, the biggest confusion of the matrix).
Run: python scripts/07_errors.py
     python scripts/07_errors.py Baroque Impressionism      (any other pair of styles)
"""

# ---------------------------------------------------------------------------------------------
# 1. Libraries
# ---------------------------------------------------------------------------------------------
import sys                     # sys: search path + read the two style names typed after the command
from pathlib import Path       # Path: handles file paths; used below to locate the project folder

# Add the project folder (two levels above this file) to Python's search path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt                 # matplotlib: draws the grid of paintings
import torch                                    # PyTorch: runs the network
from PIL import Image                           # PIL: opens the original image files
from torch import nn                            # nn: the Linear layer of the 5-style head
from torch.utils.data import DataLoader         # DataLoader: serves the test images in batches
from torchvision import models                  # models: the ResNet-18 architecture

from src.config import BATCH_SIZE, FIG_DIR, OUT_DIR, SEED     # shared settings and folders
from src.data import load_dataframe, make_datasets            # same data and same split as before
from src.transforms import resnet_transforms                  # same preprocessing as script 04
from src.utils import set_seed                                # fix randomness

# The two styles to inspect: typed after the command, or Cubism -> Pop_Art by default
TRUE_STYLE = sys.argv[1] if len(sys.argv) > 2 else "Cubism"
PREDICTED_STYLE = sys.argv[2] if len(sys.argv) > 2 else "Pop_Art"


def main():
    # ---- 1. Same paintings and same split as scripts 01, 02 and 04 (same seed)
    set_seed(SEED)                                              # seed 42 -> identical split
    df = load_dataframe()                                       # the 1,500 paintings
    tf = resnet_transforms(False)                               # no random mirror: we only predict
    datasets, classes = make_datasets(df, {"train": tf, "val": tf, "test": tf}, cache_size=128)
    test = datasets["test"]                                     # the 225 test paintings
    loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)   # keep the order: no shuffle

    # ---- 2. Rebuild the network and load the weights saved by script 04 (no retraining)
    model = models.resnet18(weights=None)                       # empty ResNet-18 architecture
    model.fc = nn.Linear(model.fc.in_features, len(classes))    # same 5-style last layer
    model.load_state_dict(torch.load(OUT_DIR / "resnet18_finetuned.pt", map_location="cpu"))
    model.eval()                                                # answer mode

    # ---- 3. Predict every test painting, keep the probabilities to show the confidence
    probs = []
    with torch.no_grad():                                       # no gradients needed to answer
        for xb, _ in loader:
            probs.append(torch.softmax(model(xb), dim=1))       # 5 scores -> 5 probabilities
    probs = torch.cat(probs)                                    # (225, 5)
    pred = probs.argmax(1)                                      # predicted style index per painting
    true = test.labels                                          # true style index per painting

    # ---- 4. Keep the mistakes of the chosen type: true TRUE_STYLE, predicted PREDICTED_STYLE
    t, p = classes.index(TRUE_STYLE), classes.index(PREDICTED_STYLE)   # style names -> indices
    wrong = [i for i in range(len(true)) if true[i] == t and pred[i] == p]
    print(f"{len(wrong)} test paintings of style {TRUE_STYLE} predicted as {PREDICTED_STYLE}")
    if not wrong:                                               # nothing to show: stop here
        return

    # ---- 5. Draw them in a grid, with the network's confidence and the artist's name
    cols = 5                                                    # 5 paintings per row
    rows = (len(wrong) + cols - 1) // cols                      # enough rows for all of them
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.8 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")                                          # hide the axes of every cell
    artists = df["artist"].tolist()                             # artist of each painting
    positions = dict(zip(df["image"], range(len(df))))          # image path -> row in df
    for ax, i in zip(axes.flat, wrong):
        path = test.images[i]                                   # original image file
        ax.imshow(Image.open(path).convert("RGB"))              # show the painting itself
        artist = artists[positions[path]]                       # look up its artist
        ax.set_title(f"{artist[:22]}\n{PREDICTED_STYLE} {probs[i, p]:.0%} | "
                     f"{TRUE_STYLE} {probs[i, t]:.0%}", fontsize=8)   # confidence for both styles
    fig.suptitle(f"ResNet errors (test): true {TRUE_STYLE}, predicted {PREDICTED_STYLE}")
    fig.tight_layout()
    out = FIG_DIR / f"resnet_errors_{TRUE_STYLE}_as_{PREDICTED_STYLE}.png"
    fig.savefig(out, dpi=120)                                   # save the figure
    print(f"[saved] {out}")


# Run main() only when the file is launched directly: python scripts/07_errors.py
if __name__ == "__main__":
    main()

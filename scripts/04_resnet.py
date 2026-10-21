"""Step 4 — Pretrained convolutional network (ResNet-18), transfer learning.

Two phases, as in course 7:
  A. feature extraction : freeze the whole pretrained network, train only a new last layer (512 -> 5).
  B. fine-tuning        : also unfreeze the last block (layer4), with a 10x smaller learning rate.
Model selection on VALIDATION only; the test set is opened once, at the very end.
Run: python scripts/04_resnet.py
"""

# ---------------------------------------------------------------------------------------------
# 1. Python standard libraries
# ---------------------------------------------------------------------------------------------
import sys                     # sys: lets us change Python's search path (used just below)
import time                    # time: precise stopwatch, used at the end to time one prediction
from pathlib import Path       # Path: handles file paths; used below to locate the project folder

# Add the project folder (two levels above this file) to the places where Python looks for code,
# so that "from src.xxx import ..." works when the script is run from anywhere.
sys.path.append(str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------------------------
# 2. Deep-learning libraries
# ---------------------------------------------------------------------------------------------
import torch                                # PyTorch: tensors (arrays of numbers) and all the maths
from torch import nn                        # nn: building blocks of networks (Linear, BatchNorm...)
from torch.utils.data import DataLoader     # DataLoader: serves the images in batches of 64
from torchvision import models              # models: catalogue of ready-made architectures (ResNet)

# ---------------------------------------------------------------------------------------------
# 3. Our own project code (written in src/ for the dense network, reused here unchanged)
# ---------------------------------------------------------------------------------------------
from src.config import BATCH_SIZE, FIG_DIR, NUM_WORKERS, OUT_DIR, SEED   # shared settings
from src.data import load_dataframe, make_datasets                      # read data, split it
from src.evaluate import plot_confusion, predict, summary              # evaluation helpers
from src.train import fit                                              # the training loop
from src.transforms import resnet_transforms                           # 224 px + ImageNet norm
from src.utils import get_device, save_metrics, set_seed, timer        # misc helpers

# Images are kept in memory at 128 x 128 px (their stored size);
# resnet_transforms enlarges them to 224 x 224, the size ResNet expects.
CACHE_SIZE = 128

# ---------------------------------------------------------------------------------------------
# 5. Helper: count parameters (all of them, or only the trainable ones)
# ---------------------------------------------------------------------------------------------
def count(model, trainable_only=False):
    # numel() = number of values in one weight tensor; we add them up over the whole network.
    # With trainable_only=True we keep only tensors that accept corrections (requires_grad).
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# ---------------------------------------------------------------------------------------------
# 6. Main program
# ---------------------------------------------------------------------------------------------
def main():
    # ---- 6.1 Data: exactly the same paintings and the same split as the dense network
    set_seed(SEED)                          # fix randomness (seed 42) -> reproducible results
    device = get_device()                   # compute on the CPU, like the museum kiosk
    df = load_dataframe()                   # table of the 1,500 paintings: image path + style
    transforms = {"train": resnet_transforms(True),     # train: resize 224, random mirror, normalise
                  "val": resnet_transforms(False),      # val:   resize 224, normalise (no mirror)
                  "test": resnet_transforms(False)}     # test:  same as val
    # Split into train (1,049) / val (226) / test (225), images cached in memory at 128 px
    datasets, classes = make_datasets(df, transforms, cache_size=CACHE_SIZE)
    # One DataLoader per split: batches of 64, shuffled order for train only
    loaders = {k: DataLoader(v, batch_size=BATCH_SIZE, shuffle=(k == "train"),
                             num_workers=NUM_WORKERS) for k, v in datasets.items()}

    # ---- 6.2 Model: ResNet-18 architecture + weights already trained on ImageNet
    #      (11.7 M parameters learned on 1.2 M photos, 1,000 categories)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():            # freeze every pretrained weight:
        p.requires_grad = False             # they will not receive any correction
    # Replace the last layer (512 -> 1,000 ImageNet classes) by a new one (512 -> 5 styles).
    # A new layer starts with random weights and is trainable by default.
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)                # move the network to the chosen device (CPU)

    timings: dict = {}                      # notebook for durations
    metrics = {"classes": classes,          # notebook for results: the 5 style names
               "n_params_total": count(model)}          # and the total number of parameters

    # ---- 6.3 Phase A: only the new last layer learns (512 x 5 + 5 = 2,565 parameters)
    metrics["phaseA_trainable"] = count(model, trainable_only=True)      # should be 2,565
    print(f"[A] trainable parameters: {metrics['phaseA_trainable']:,} / {metrics['n_params_total']:,}")
    with timer("phaseA_train", timings):    # stopwatch around phase A
        # Same training loop as the dense network: forward -> CrossEntropyLoss (softmax + -ln)
        # -> backward -> Adam step. Max 10 epochs, step size 1e-3, early stop after 3 bad epochs.
        hist_a, _ = fit(model, loaders["train"], loaders["val"], device,
                        epochs=10, lr=1e-3, patience=3)


# Run main() only when the file is launched directly: python scripts/04_resnet.py
if __name__ == "__main__":
    main()


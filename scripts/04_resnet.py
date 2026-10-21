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


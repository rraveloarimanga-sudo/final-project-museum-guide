"""Step 4 — Pretrained convolutional network (ResNet-18), transfer learning.

Two phases, as in course 7:
  A. feature extraction : freeze the whole pretrained network, train only a new last layer (512 -> 5).
  B. fine-tuning        : also unfreeze the last block (layer4), with a 10x smaller learning rate.
Model selection on VALIDATION only; the test set is opened once, at the very end.
Run: python scripts/04_resnet.py
"""

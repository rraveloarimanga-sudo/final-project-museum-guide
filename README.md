# Museum Guide — Neural components

Intelligent guide for a (fictional) French fine-arts museum:

1. **Recognise** the style / genre of a painting from a photo (MLP from PyTorch primitives, then pretrained ResNet).
2. **Answer** natural-language questions ("a stormy sea", "portraits of children") by retrieving matching artworks.
3. **Recommend**: is a neural network worth it on modest kiosk CPUs, versus a classical model?

Every number in the presentation is produced by a script in this repo and saved under `outputs/`.

## Layout

```
src/
  config.py        # paths, seeds, hyper-parameters (single source of truth)
  data.py          # loading, label encoding, stratified splits, PyTorch Dataset
  utils.py         # seeding, device, timing, JSON metrics
scripts/
  00_eda.py        # class balance, image sizes, sample grid
  01_baseline.py   # classical model (logistic regression)        [step 2]
  02_mlp.py        # feed-forward net from primitives              [step 3]
  03_diagnose.py   # failure diagnosis experiments                 [step 4]
  04_resnet.py     # pretrained ResNet (features + fine-tune)      [step 5]
  05_search.py     # natural-language artwork retrieval            [step 6]
  06_benchmark.py  # CPU latency / size for the recommendation     [step 7]
outputs/           # metrics (*.json), figures (*.png) — committed
data/              # raw data — NOT committed
docs/              # git workflow, experiment log
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/00_eda.py
```

All scripts run on CPU. Seed is fixed in `src/config.py`.

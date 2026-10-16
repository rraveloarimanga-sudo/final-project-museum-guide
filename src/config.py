"""Single source of truth for paths and hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
METRICS_DIR = OUT_DIR / "metrics"
CACHE_DIR = OUT_DIR / "cache"

SEED = 42

# Which label the guide predicts: "style" or "genre"
TARGET = "style"

# Splits (stratified): 70 / 15 / 15
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Image sizes — CPU budget
MLP_IMG_SIZE = 64        # 64x64x3 = 12 288 inputs to the MLP
RESNET_IMG_SIZE = 224

# Training defaults (CPU-friendly)
BATCH_SIZE = 64
NUM_WORKERS = 0          # safest on laptops / macOS

for d in (FIG_DIR, METRICS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

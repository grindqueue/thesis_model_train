# Parental Control Image Classifier — EfficientNetB0

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-damilareisaac%2Fparental--control--efficientnet--b0-blue)](https://huggingface.co/damilareisaac/parental-control-efficientnet-b0)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20beff)](https://www.kaggle.com/datasets/sofialitvin/dataset-images)

A multi-label image classifier that detects NSFW/harmful content across 4 categories: **alcohol**, **drugs**, **sexual**, and **extremism**. Built with TensorFlow/Keras on top of EfficientNetB0, trained on ~79 k images.

---

## 🤗 Pre-trained Model

> **The trained model is freely available on Hugging Face Hub — no sign-in required.**
>
> **[damilareisaac/parental-control-efficientnet-b0](https://huggingface.co/damilareisaac/parental-control-efficientnet-b0)**

### Quick start (inference only — no training needed)

```bash
pip install huggingface_hub tensorflow pillow
```

```python
import tensorflow as tf, numpy as np, json
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "damilareisaac/parental-control-efficientnet-b0"

# Downloads and caches model + metadata (~44 MB, first run only)
model_path = hf_hub_download(REPO, "parental_control_b0.keras")
meta       = json.load(open(hf_hub_download(REPO, "model_metadata.json")))

model = tf.keras.models.load_model(model_path)

img = Image.open("image.jpg").convert("RGB").resize(tuple(meta["input_size"]))
arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
scores = model.predict(arr)[0]

for label, score in zip(meta["labels"], scores):
    flagged = score > meta["optimal_thresholds"][label]
    print(f"{label:<12} {score:.3f}  {'⚠️  FLAGGED' if flagged else '✅ ok'}")
```

### Model performance

| Label | Val Accuracy | Val Loss |
|---|---|---|
| alcohol | **99.6%** | — |
| extremism | **99.5%** | — |
| sexual | **99.0%** | — |
| drugs | **98.2%** | — |
| **Overall best val_loss** | — | **0.0948** |

### Files available on HF Hub

| File | Size | Purpose |
|---|---|---|
| `parental_control_b0.keras` | ~44 MB | Full model — inference & fine-tuning |
| `model_metadata.json` | < 1 KB | Labels, thresholds, input spec |
| `training_history.png` | 197 KB | Loss & accuracy curves |
| `threshold_calibration.png` | 81 KB | Per-label threshold calibration |

---

## About the model

### Architecture

| Property | Value |
|---|---|
| Base | EfficientNetB0 (ImageNet pre-trained) |
| Total parameters | 4,383,655 |
| Input size | 224 × 224 × 3 (RGB) |
| Output | 4 independent sigmoid scores (multi-label) |
| Labels | `alcohol`, `drugs`, `sexual`, `extremism` |
| Precision | mixed_float16 during training |

### Training strategy

Training uses a two-phase fine-tuning approach:

| Phase | Epochs | Layers trained | Learning rate |
|---|---|---|---|
| 1 — Head only | 20 | Custom classification head | 1e-3 |
| 2 — Fine-tune | 40 | Head + top 30% of EfficientNetB0 backbone | 2e-5 |

Loss is weighted binary cross-entropy with higher penalties for harder/rarer classes:

| Label | Weight |
|---|---|
| alcohol | 1.5 |
| drugs | 1.5 |
| sexual | 2.0 |
| extremism | 2.5 |

### Dataset

~79 k images from Kaggle [`sofialitvin/dataset-images`](https://www.kaggle.com/datasets/sofialitvin/dataset-images):

| Class | Images |
|---|---|
| normal | 23,332 |
| alcohol | 13,837 |
| tobacco (→ merged into drugs) | 14,025 |
| sexual | 14,178 |
| drugs | 5,649 |
| extremism | 7,912 |

> `tobacco` images are merged into the `drugs` label during training.

---

## Training from scratch

### 1. Install Python 3.12

```bash
brew install python@3.12          # macOS (Homebrew)
sudo apt install python3.12       # Ubuntu/Debian
winget install Python.Python.3.12 # Windows
```

> Python 3.12 is required for `tensorflow-metal` compatibility on Apple Silicon. Python 3.11+ works fine on Linux/Windows.

### 2. Create a virtual environment and install dependencies

**Apple Silicon (M-series Mac)**
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install "tensorflow==2.18.0" tensorflow-metal
pip install scikit-learn matplotlib pillow pandas kaggle
```

**NVIDIA GPU (Linux / Windows)**
```bash
python3 -m venv venv
source venv/bin/activate          # Linux
# venv\Scripts\activate           # Windows
pip install tensorflow[and-cuda]  # pulls CUDA deps automatically
pip install scikit-learn matplotlib pillow pandas kaggle
```

**CPU only (any platform)**
```bash
python3 -m venv venv
source venv/bin/activate
pip install tensorflow
pip install scikit-learn matplotlib pillow pandas kaggle
```

### 3. Download the dataset

```bash
# Requires a Kaggle API token at ~/.kaggle/kaggle.json
kaggle datasets download -d sofialitvin/dataset-images -p dataset/ --unzip
```

### 4. Configure paths

Either edit the two constants at the top of `train.py`:

```python
DATA_DIR    = "/path/to/dataset/DATASET_IMAGES"
WORKING_DIR = "/path/to/output"
```

Or pass them as environment variables (no code change needed):

```bash
export DATA_DIR="/path/to/dataset/DATASET_IMAGES"
export WORKING_DIR="/path/to/output"
python train.py
```

### 5. Run training

```bash
source venv/bin/activate
python train.py
```

**Resumable** — if interrupted, re-running `train.py` picks up from the last completed phase automatically.

---

## GPU support

The script auto-detects and uses the best available accelerator:

| Hardware | Backend | Install |
|---|---|---|
| Apple Silicon (M-series) | Metal | `pip install tensorflow-metal` |
| NVIDIA GPU | CUDA | `pip install tensorflow[and-cuda]` |
| Any CPU | CPU | No extra packages — automatic fallback |

For multi-GPU NVIDIA setups, `MirroredStrategy` is used automatically when more than one GPU is detected.

### Estimated training times

| Hardware | Time per epoch | Total (60 epochs) |
|---|---|---|
| Apple M3 / M4 (Metal, batch 128) | ~4 min | ~4 hrs |
| Apple M1 / M2 (Metal, batch 128) | ~6–8 min | ~6–8 hrs |
| NVIDIA RTX 4090 (CUDA, batch 128) | ~2–3 min | ~2–3 hrs |
| NVIDIA RTX 3080 (CUDA, batch 128) | ~4–5 min | ~4–5 hrs |
| NVIDIA T4 (Google Colab, batch 64) | ~8–10 min | ~8–10 hrs |
| Modern CPU only (batch 32) | ~40–60 min | ~2–3 days |

> **Tip (macOS):** Prevent your Mac from sleeping during a long run:
> ```bash
> caffeinate -disum -w $(pgrep -f "python train.py") &
> ```

### Reducing training time

| Technique | How | Effect |
|---|---|---|
| Larger batch size | Set `BATCH_SIZE = 128` or higher | 2–4× fewer steps/epoch |
| Enable GPU | See GPU Support above | 10–50× vs CPU |
| Fewer epochs | Reduce `PHASE1_EPOCHS` / `PHASE2_EPOCHS` | Linear reduction |
| Google Colab (free T4) | Upload dataset to Google Drive, run there | Free GPU |

---

## Outputs

All outputs are written to `WORKING_DIR/`:

```
output/
  checkpoints/
    phase1_best.keras          ← best Phase 1 weights
    phase2_best.keras          ← best Phase 2 weights (primary result)
  exported_model/
    parental_control_b0.keras  ← full exported model (~44 MB)
    model_metadata.json        ← labels, thresholds, input spec
    training_history.png
    threshold_calibration.png
  logs/
    training.log
  train_clean.csv / val_clean.csv / test_clean.csv
```


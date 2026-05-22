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

The script auto-detects the best available accelerator at startup via `setup_gpu()` and configures training accordingly. No code changes are required when switching hardware.

### How detection works

```
startup
  │
  ├─ arm64 macOS? ──yes──► tensorflow-metal installed? ──yes──► Metal GPU (/GPU:0)
  │                                                     └──no──► CPU (with warning)
  │
  ├─ CUDA GPU present? ──yes──► 1 GPU? ──► OneDeviceStrategy (/gpu:0)
  │                             └─ >1 GPU? ─► MirroredStrategy (all GPUs)
  │
  └─ nothing found ──► CPU fallback (warning logged)
```

Mixed-precision (`float16` compute, `float32` weights) is enabled automatically on all GPU paths for a significant speed boost.

### Apple Silicon (M-series)

Requires **TF 2.18** + `tensorflow-metal 1.2.0` (TF 2.21+ is not yet supported by tensorflow-metal):

```bash
pip install "tensorflow==2.18.0" tensorflow-metal
```

What happens at runtime:
- The Metal plugin is auto-registered when TensorFlow is imported — no explicit `import` needed.
- The GPU appears as `/physical_device:GPU:0` with device name `METAL`.
- Memory growth is enabled automatically so the GPU doesn't pre-allocate all unified memory.
- `mixed_float16` is applied — Metal supports float16 natively, giving ~2× throughput over float32.

> **Sleep prevention (macOS):** The laptop will go to sleep during long runs unless you run:
> ```bash
> caffeinate -disum -w $(pgrep -f "python train.py") &
> ```
> `-d` prevents display sleep, `-i` idle sleep, `-s` AC sleep, `-u` keeps user session active.

### NVIDIA GPU (CUDA)

Works on Linux and Windows with a CUDA-capable GPU (Kepler/GTX 700+):

```bash
pip install tensorflow[and-cuda]   # installs CUDA/cuDNN automatically
```

What happens at runtime:
- `mixed_float16` is applied. Full benefit requires Volta architecture (GTX 1080 Ti / V100) or newer — older cards run float16 but without tensor core acceleration.
- Memory growth is enabled — the GPU allocates VRAM on demand rather than reserving it all upfront.
- **Single GPU:** `OneDeviceStrategy(/gpu:0)` — all training on one device.
- **Multiple GPUs:** `MirroredStrategy` is selected automatically. Gradients are synchronised across all devices at each step; effective batch size scales with the number of GPUs.

Minimum VRAM requirements at `BATCH_SIZE = 128`:

| GPU VRAM | Batch size |
|---|---|
| 4 GB | Reduce to 32–64 |
| 8 GB | 64–128 |
| 12 GB+ | 128 (default) |
| 24 GB+ | 256+ for faster training |

> If you hit OOM errors, reduce `BATCH_SIZE` at the top of `train.py`.

### Google Colab (free T4 GPU)

Colab provides a free NVIDIA T4 (16 GB VRAM). No local install needed:

1. Upload the dataset to Google Drive.
2. Open a new Colab notebook and mount Drive:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```
3. Clone this repo and set env vars:
   ```bash
   !git clone https://github.com/grindqueue/thesis_model_train.git
   %cd thesis_model_train
   !pip install scikit-learn matplotlib pillow pandas
   ```
   ```python
   import os
   os.environ["DATA_DIR"]    = "/content/drive/MyDrive/DATASET_IMAGES"
   os.environ["WORKING_DIR"] = "/content/drive/MyDrive/output"
   ```
4. Run training:
   ```bash
   !python train.py
   ```

> Colab disconnects after ~12 hrs of inactivity. Training is resumable — re-run `train.py` to continue from the last completed phase.

### CPU only

No extra packages. Set a smaller batch size to avoid excessive RAM use:

```python
BATCH_SIZE = 32   # reduce if RAM < 16 GB
```

CPU training is very slow (see timing table above). Use Colab or a cloud GPU instead for full training runs.

### Verifying your GPU is being used

Check the log output at the start of training:

```
# Apple Silicon
INFO  GPUs found : ['/physical_device:GPU:0']
INFO  Apple Silicon detected — tensorflow-metal plugin loaded
INFO  mixed_float16 precision enabled
INFO  Using GPU: /physical_device:GPU:0  [Metal (Apple Silicon)]

# NVIDIA single GPU
INFO  GPUs found : ['/physical_device:GPU:0']
INFO  mixed_float16 precision enabled
INFO  Using GPU: /physical_device:GPU:0  [CUDA]

# NVIDIA multi-GPU
INFO  GPUs found : ['/physical_device:GPU:0', '/physical_device:GPU:1']
INFO  mixed_float16 precision enabled
INFO  Multi-GPU: MirroredStrategy across 2 GPUs

# No GPU
WARNING  No GPU found — running on CPU (will be slow)
```



### Estimated training times

Both phases together (20 + 40 epochs, ~79 k images):

| Hardware | Time per epoch | Total (60 epochs) |
|---|---|---|
| Apple M3 / M4 (Metal, batch 128) | ~4 min | ~4 hrs |
| Apple M1 / M2 (Metal, batch 128) | ~6–8 min | ~6–8 hrs |
| NVIDIA RTX 4090 (CUDA, batch 128) | ~2–3 min | ~2–3 hrs |
| NVIDIA RTX 3080 (CUDA, batch 128) | ~4–5 min | ~4–5 hrs |
| NVIDIA T4 (Google Colab, batch 64) | ~8–10 min | ~8–10 hrs |
| Modern CPU only (batch 32) | ~40–60 min | ~2–3 days |

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


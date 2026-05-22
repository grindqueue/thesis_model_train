# Parental Control Image Classifier — EfficientNetB0

A multi-label image classifier that detects NSFW/harmful content across 4 categories: **alcohol**, **drugs**, **sexual**, and **extremism**. Built with TensorFlow/Keras on top of EfficientNetB0, exported to TFLite INT8 for on-device deployment.

---

## Dataset

Download from Kaggle: [`sofialitvin/dataset-images`](https://www.kaggle.com/datasets/sofialitvin/dataset-images)

Expected folder structure after download:

```
DATA_DIR/
  alcohol/
  drugs/
  tobacco/       ← merged into "drugs" label
  sexual/
  extremism/
  normal/
```

> **Note:** `tobacco` images are automatically merged into the `drugs` label during training.

---

## Setup

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

---

## Training

```bash
source venv/bin/activate
python train.py
```

Training runs in two phases:

| Phase | Epochs | What trains |
|---|---|---|
| 1 | 20 | Head only (backbone frozen) |
| 2 | 40 | Head + top 30% of backbone (fine-tune) |

**Resumable** — if interrupted, re-running `train.py` picks up from the last completed phase automatically.

### GPU Support

The script auto-detects and uses the best available accelerator:

| Hardware | Backend | Install |
|---|---|---|
| Apple Silicon (M-series) | Metal | `pip install tensorflow-metal` |
| NVIDIA GPU | CUDA | `pip install tensorflow[and-cuda]` |
| Any CPU | CPU | No extra packages — automatic fallback |

For multi-GPU NVIDIA setups, `MirroredStrategy` is used automatically when more than one GPU is detected.

### Estimated training times

Training time depends heavily on batch size (default 128) and hardware. The two phases together (20 + 40 epochs) on ~79 k images:

| Hardware | Approx. time per epoch | Approx. total (60 epochs) |
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
| Reduce dataset | Train on a class-balanced subset | Proportional reduction |
| Google Colab (free T4) | Upload dataset to Google Drive, run there | Free GPU |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | EfficientNetB0 (ImageNet weights) |
| Input size | 224 × 224 |
| Batch size | 128 |
| Phase 1 LR | 1e-3 |
| Phase 2 LR | 2e-5 |
| Backbone freeze | Bottom 70% frozen in Phase 2 |
| Precision | mixed_float16 (GPU) |
| Loss | Weighted binary cross-entropy |
| Label weights | alcohol=1.5, drugs=1.5, sexual=2.0, extremism=2.5 |

---

## Outputs

All outputs are written to `WORKING_DIR/`:

```
output/
  checkpoints/
    phase1_best.keras          ← best Phase 1 weights
    phase2_best.keras          ← best Phase 2 weights (use this)
  exported_model/
    parental_control_b0.keras           ← full float32 model (~17.5 MB)
    parental_control_b0_int8.tflite     ← INT8 quantised (~4.4 MB) ← deploy this
    model_metadata.json                 ← labels, thresholds, input spec
    training_history.png
    threshold_calibration.png
  logs/
    training.log
  train_clean.csv / val_clean.csv / test_clean.csv
```

---

## Model Size

| Format | Size | Parameters |
|---|---|---|
| `.keras` float32 | ~17.5 MB | 4,383,655 |
| `.keras` float16 | ~8.8 MB | 4,383,655 |
| TFLite INT8 | **~4.4 MB** | 4,383,655 |

---

## Results

Validation accuracy after full training (Phase 2, 40 epochs):

| Label | Val Accuracy |
|---|---|
| alcohol | ~99.6% |
| extremism | ~99.5% |
| sexual | ~99.0% |
| drugs | ~98.2% |

---

## Pre-trained Model

The trained model is published on Hugging Face Hub:
**[damilareisaac/parental-control-efficientnet-b0](https://huggingface.co/damilareisaac/parental-control-efficientnet-b0)**

### Download & run inference

```bash
pip install huggingface_hub tensorflow pillow
```

```python
import tensorflow as tf, numpy as np, json
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "damilareisaac/parental-control-efficientnet-b0"

# Download model and metadata (cached locally after first run)
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

### Available files on HF Hub

| File | Size | Purpose |
|---|---|---|
| `parental_control_b0.keras` | ~44 MB | Full model — inference & fine-tuning |
| `model_metadata.json` | < 1 KB | Labels, thresholds, input spec |
| `training_history.png` | 197 KB | Loss & accuracy curves |
| `threshold_calibration.png` | 81 KB | Per-label threshold calibration |
 

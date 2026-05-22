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
brew install python@3.12   # macOS
```

> Python 3.12 is required for `tensorflow-metal` (Apple Silicon GPU support).

### 2. Create a virtual environment and install dependencies

```bash
python3.12 -m venv venv
source venv/bin/activate

pip install tensorflow tensorflow-metal   # Apple Silicon (Metal GPU)
# pip install tensorflow                  # Linux / NVIDIA CUDA (metal not needed)
pip install scikit-learn matplotlib pillow pandas kaggle
```

### 3. Download the dataset

```bash
# Requires a Kaggle API token (~/.kaggle/kaggle.json or KAGGLE_TOKEN env var)
kaggle datasets download -d sofialitvin/dataset-images -p dataset/ --unzip
```

### 4. Configure paths

Edit the two path constants at the top of `train.py`:

```python
DATA_DIR    = "/path/to/dataset/DATASET_IMAGES"
WORKING_DIR = "/path/to/output"
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

| Hardware | Backend | How |
|---|---|---|
| Apple Silicon (M-series) | Metal | Install `tensorflow-metal` |
| NVIDIA GPU | CUDA | Standard TF CUDA support |
| No GPU | CPU | Automatic fallback (slow) |

**Prevent your Mac from sleeping during training:**
```bash
caffeinate -disum -w $(pgrep -f "python train.py") &
```

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
 

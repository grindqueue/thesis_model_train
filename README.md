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

## Deployment

The deployment target is `parental_control_b0_int8.tflite` + `model_metadata.json`.

```python
import tensorflow as tf, json, numpy as np
from PIL import Image

meta = json.load(open("exported_model/model_metadata.json"))
interp = tf.lite.Interpreter("exported_model/parental_control_b0_int8.tflite")
interp.allocate_tensors()

img = Image.open("image.jpg").resize(tuple(meta["input_size"]))
arr = np.expand_dims(np.array(img, dtype=np.float32), 0)

interp.set_tensor(interp.get_input_details()[0]["index"], arr)
interp.invoke()
scores = interp.get_tensor(interp.get_output_details()[0]["index"])[0]

for label, score, threshold in zip(
    meta["labels"], scores, meta["optimal_thresholds"].values()
):
    print(f"{label}: {score:.2f}  {'⚠️ FLAGGED' if score > threshold else 'ok'}")
```
 

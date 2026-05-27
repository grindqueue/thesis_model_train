"""
download_and_convert.py
========================
Downloads the model from HuggingFace and converts to TFLite.

BEFORE RUNNING:
    pip install --upgrade tensorflow==2.18.0

Then run:
    python download_and_convert.py
"""

import sys

# ── Step 0: Check TF version ──────────────────────────────────────────────────
print("Checking TensorFlow version ...")

import tensorflow as tf
import keras

tf_version = tuple(int(x) for x in tf.__version__.split(".")[:2])

print(f"  TensorFlow : {tf.__version__}")
print(f"  Keras      : {keras.__version__}")

if tf_version < (2, 18):
    print()
    print("=" * 60)
    print("  ❌ Your TensorFlow is too old to load this model.")
    print(f"     You have    : TF {tf.__version__}")
    print(f"     You need    : TF 2.18.0 or newer")
    print()
    print("  Run this command then re-run this script:")
    print("    pip install --upgrade tensorflow==2.18.0")
    print("=" * 60)
    sys.exit(1)

print("  ✅ TensorFlow version is compatible\n")

# ── Remaining imports ─────────────────────────────────────────────────────────
import json
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────────────────────
REPO = "damilareisaac/parental-control-efficientnet-b0"
DEST = Path("./parental_control_model")
DEST.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────

# ── Step 1: Download all available files ─────────────────────
print("=" * 55)
print("  Step 1 — Downloading files from Hugging Face")
print("=" * 55)

files = [
    "parental_control_b0.keras",
    "model_metadata.json",
    "training_history.png",
    "threshold_calibration.png",
]

for filename in files:
    print(f"\nDownloading {filename} ...")
    try:
        path = hf_hub_download(
            repo_id   = REPO,
            filename  = filename,
            local_dir = str(DEST),
        )
        size_mb = Path(path).stat().st_size / 1e6
        print(f"  ✅  {filename}  ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ⚠️  Skipped {filename}: {e}")

# ── Step 2: Load metadata ─────────────────────────────────────
print("\n" + "=" * 55)
print("  Step 2 — Reading model_metadata.json")
print("=" * 55)

meta   = json.loads((DEST / "model_metadata.json").read_text(encoding="utf-8"))
LABELS = meta["labels"]

print(f"Labels     : {LABELS}")
print(f"Input size : {meta['input_size']}")
print(f"Thresholds : {meta['optimal_thresholds']}")

# ── Step 3: Register custom objects and load Keras model ──────
print("\n" + "=" * 55)
print("  Step 3 — Loading .keras model")
print("=" * 55)

import keras as _keras

# ── FIX: Patch Layer base class to drop any unrecognised kwargs
#         that were saved by a newer Keras version (renorm*,
#         quantization_config, etc.)
_STRIP_KEYS = {
    "renorm", "renorm_clipping", "renorm_momentum",
    "quantization_config",
}

_orig_layer_init = _keras.layers.Layer.__init__

def _safe_layer_init(self, **kwargs):
    for k in _STRIP_KEYS:
        kwargs.pop(k, None)
    _orig_layer_init(self, **kwargs)

_orig_layer_from_config = _keras.layers.Layer.from_config.__func__

@classmethod
def _safe_layer_from_config(cls, config):
    for k in _STRIP_KEYS:
        config.pop(k, None)
    return _orig_layer_from_config(cls, config)

_keras.layers.Layer.__init__    = _safe_layer_init
_keras.layers.Layer.from_config = _safe_layer_from_config

print("  ✅ Keras layer patched (unknown saved kwargs will be ignored)")


@_keras.saving.register_keras_serializable(package="ParentalControl")
class LabelAccuracy(_keras.metrics.Metric):
    def __init__(self, label_idx, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_idx = label_idx
        self.correct   = self.add_weight(name="correct", initializer="zeros")
        self.total     = self.add_weight(name="total",   initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        t = y_true[:, self.label_idx]
        p = tf.cast(y_pred[:, self.label_idx] > 0.5, tf.float32)
        self.correct.assign_add(tf.reduce_sum(tf.cast(tf.equal(t, p), tf.float32)))
        self.total.assign_add(tf.cast(tf.shape(t)[0], tf.float32))

    def result(self):
        return self.correct / (self.total + 1e-8)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

    def get_config(self):
        cfg = super().get_config()
        cfg["label_idx"] = self.label_idx
        return cfg


@_keras.saving.register_keras_serializable(package="ParentalControl")
def weighted_bce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
    bce    = -(y_true * tf.math.log(y_pred)
               + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    w      = tf.constant([[1.5, 1.5, 2.0, 2.0]], dtype=tf.float32)
    return tf.reduce_mean(bce * w)


keras_path = DEST / "parental_control_b0.keras"
print(f"Loading {keras_path} ...")
model = tf.keras.models.load_model(str(keras_path))
print("✅ Model loaded")
print(f"   Input  : {model.input_shape}")
print(f"   Output : {model.output_shape}")

# ── Step 4: Quick inference test ─────────────────────────────
print("\n" + "=" * 55)
print("  Step 4 — Quick inference test")
print("=" * 55)

dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8).astype("float32")
inp   = tf.keras.applications.efficientnet.preprocess_input(dummy)
inp   = np.expand_dims(inp, axis=0)

scores = model(inp, training=False).numpy()[0].astype("float32")
print("Scores on random image:")
for label, score in zip(LABELS, scores):
    threshold = meta["optimal_thresholds"][label]
    flag      = "⚠️  FLAGGED" if score > threshold else "✅ ok"
    print(f"  {label:<12}: {score:.3f}   {flag}")

# ── Step 5: Cast model to float32 then convert to TFLite INT8 ─
print("\n" + "=" * 55)
print("  Step 5 — Converting to TFLite INT8")
print("  (model was mixed_float16; casting to float32 first)")
print("  (takes 2–5 minutes)")
print("=" * 55)

# The model was trained with mixed_float16. TFLite INT8 converter
# requires a pure float32 graph. We rebuild it in float32 by:
#   1. Switching the global policy to float32
#   2. Wrapping the model in a float32 tf.function via a
#      concrete function — this forces all ops to float32.

print("Building float32 concrete function ...")

@tf.function(input_signature=[tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32)])
def serving_fn(x):
    return tf.cast(model(x, training=False), tf.float32)

concrete_fn = serving_fn.get_concrete_function()
print("  ✅ float32 concrete function ready")

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_fn], serving_fn
)

def representative_data():
    for _ in range(200):
        img = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8).astype("float32")
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        yield [img.astype(np.float32)]

converter.optimizations              = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset     = representative_data
converter.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type       = tf.float32
converter.inference_output_type      = tf.float32

print("Converting ...")
tflite_model = converter.convert()

tflite_path = DEST / "parental_control_b0_int8.tflite"
tflite_path.write_bytes(tflite_model)

keras_mb  = keras_path.stat().st_size / 1e6
tflite_mb = tflite_path.stat().st_size / 1e6
print(f"✅ TFLite saved  |  .keras: {keras_mb:.1f} MB  →  .tflite: {tflite_mb:.1f} MB")

# ── Step 6: Verify TFLite ─────────────────────────────────────
print("\n" + "=" * 55)
print("  Step 6 — Verifying TFLite model")
print("=" * 55)

interp = tf.lite.Interpreter(model_path=str(tflite_path))
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]

print(f"Input  : {inp_d['shape']}  {inp_d['dtype']}")
print(f"Output : {out_d['shape']}  {out_d['dtype']}")
print(f"Order  : {LABELS}")

test_inp = inp.astype(np.float32)
interp.set_tensor(inp_d["index"], test_inp)
interp.invoke()
tflite_scores = interp.get_tensor(out_d["index"])[0]

print(f"\n{'Label':<12} {'Keras':>8} {'TFLite':>8} {'Match':>7}")
print("─" * 40)
for label, ks, ts in zip(LABELS, scores, tflite_scores):
    match = "✅" if abs(float(ks) - float(ts)) < 0.05 else "⚠️ "
    print(f"  {label:<12} {ks:>8.3f} {ts:>8.3f} {match:>7}")

# ── Final summary ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ✅ All done!")
print("=" * 55)
print(f"\nFiles saved to: {DEST.resolve()}")
print()
for f in sorted(DEST.rglob("*")):
    if f.is_file():
        print(f"  {f.name:<45}  {f.stat().st_size/1e6:.2f} MB")

print()
print("Next steps:")
print("  1. Copy these two files to /opt/parental_control/ on your Linux device:")
print("       parental_control_b0_int8.tflite")
print("       model_metadata.json")
print()
print("  2. scp command (run from this machine):")
print("       scp parental_control_model/parental_control_b0_int8.tflite  user@device-ip:/opt/parental_control/")
print("       scp parental_control_model/model_metadata.json               user@device-ip:/opt/parental_control/")
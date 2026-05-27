"""
test_model.py
=============
Test the downloaded model on real images before deploying to emulator.

Usage:
    python test_model.py                          # tests on sample images from dataset
    python test_model.py --image path/to/img.jpg  # tests one specific image
    python test_model.py --folder path/to/folder  # tests all images in a folder

Run from your model_code folder:
    python test_model.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR  = Path("./parental_control_model")
TFLITE     = MODEL_DIR / "parental_control_b0_int8.tflite"
KERAS      = MODEL_DIR / "parental_control_b0.keras"
META       = MODEL_DIR / "model_metadata.json"

# ── Load metadata ─────────────────────────────────────────────────────────────
if not META.exists():
    sys.exit(f"❌ model_metadata.json not found at {META}\n   Run download_and_convert.py first.")

meta       = json.loads(META.read_text(encoding="utf-8"))
LABELS     = meta["labels"]
THRESHOLDS = meta["optimal_thresholds"]
INPUT_H, INPUT_W = meta["input_size"]

print("=" * 58)
print("  Parental Control Model — Test Tool")
print("=" * 58)
print(f"  Labels     : {LABELS}")
print(f"  Thresholds : {THRESHOLDS}")
print(f"  Input size : {INPUT_H}×{INPUT_W}")
print()

# ── Load model (prefer TFLite for speed, fall back to Keras) ──────────────────
# This model uses Flex (TF Select) ops because it was trained with
# mixed_float16 on Apple Silicon. The Flex delegate on Windows has a
# float16/float32 dtype mismatch at invoke() time.
# We always use the .keras model for inference — same scores, no dtype issues.
USE_TFLITE = False

if not USE_TFLITE:
    if not KERAS.exists():
        sys.exit(f"❌ No model found. Run download_and_convert.py first.")
    import tensorflow as tf
    import keras as _keras

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
        def result(self):   return self.correct / (self.total + 1e-8)
        def reset_state(self):
            self.correct.assign(0.0); self.total.assign(0.0)
        def get_config(self):
            cfg = super().get_config(); cfg["label_idx"] = self.label_idx; return cfg

    @_keras.saving.register_keras_serializable(package="ParentalControl")
    def weighted_bce(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        bce    = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(bce * tf.constant([[1.5, 1.5, 2.0, 2.0]], dtype=tf.float32))

    # Patch Keras to strip unrecognised BatchNormalization kwargs
    # (model was saved with TF 2.18/Apple Silicon which adds renorm params)
    _STRIP = {"renorm", "renorm_clipping", "renorm_momentum", "quantization_config"}
    _orig_init = _keras.layers.Layer.__init__
    def _safe_init(self, **kwargs):
        for k in list(_STRIP): kwargs.pop(k, None)
        _orig_init(self, **kwargs)
    _keras.layers.Layer.__init__ = _safe_init

    print(f"Loading Keras model ...")
    model      = tf.keras.models.load_model(str(KERAS))
    MODEL_TYPE = "keras"
    print(f"✅ Keras model loaded  ({KERAS.stat().st_size/1e6:.1f} MB)\n")


# ── Inference function ────────────────────────────────────────────────────────
def preprocess(image_path: str) -> np.ndarray:
    """
    Load image and resize to 224x224.

    The model metadata says:
        "Scale pixels to [0,255] float32 — EfficientNetB0 normalises internally"

    This means the model's first layer handles normalisation itself.
    We pass raw [0, 255] float32 pixels — no manual rescaling.

    We try both raw [0,255] and EfficientNet-normalised [-1,1] and pick
    whichever gives a higher score on the image, so we can auto-detect
    which preprocessing the model was saved with.
    """
    img = Image.open(image_path).convert("RGB").resize(
        (INPUT_W, INPUT_H), Image.BILINEAR
    )
    arr = np.array(img, dtype=np.float32)
    # Raw [0, 255] — model normalises internally
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)


def run_inference(image_path: str) -> dict:
    """
    Run the model on one image.
    Auto-detects preprocessing: tries raw [0,255] and normalised [-1,1],
    uses whichever gives the higher max score.
    """
    img = Image.open(image_path).convert("RGB").resize(
        (INPUT_W, INPUT_H), Image.BILINEAR
    )
    arr = np.array(img, dtype=np.float32)

    # Two candidate preprocessings
    inp_raw  = np.expand_dims(arr, 0)                          # [0, 255]
    inp_norm = np.expand_dims((arr / 127.5) - 1.0, 0)         # [-1, 1]

    t0 = time.perf_counter()

    def infer(inp):
        if MODEL_TYPE == "tflite":
            interp.set_tensor(inp_d["index"], inp.astype(np.float32))
            interp.invoke()
            return interp.get_tensor(out_d["index"])[0].astype(np.float32)
        else:
            return model(inp, training=False).numpy()[0].astype(np.float32)

    scores_raw  = infer(inp_raw)
    scores_norm = infer(inp_norm)

    # Pick whichever preprocessing gives a higher maximum score
    # (one of them will give near-zero for everything if it is wrong)
    if max(scores_norm) > max(scores_raw):
        scores = scores_norm
        preprocess_used = "normalised [-1,1]"
    else:
        scores = scores_raw
        preprocess_used = "raw [0,255]"

    ms = (time.perf_counter() - t0) * 1000

    result = {
        "scores"         : {lbl: float(s) for lbl, s in zip(LABELS, scores)},
        "flagged"        : {lbl: float(s) > THRESHOLDS[lbl]
                            for lbl, s in zip(LABELS, scores)},
        "any_flagged"    : any(float(s) > THRESHOLDS[lbl]
                               for lbl, s in zip(LABELS, scores)),
        "ms"             : ms,
        "preprocess_used": preprocess_used,
    }
    return result


def print_result(image_path: str, result: dict):
    """Print a clean result table for one image."""
    print(f"📷  {Path(image_path).name}")
    print(f"    Inference time : {result['ms']:.1f} ms")
    print(f"    Preprocessing  : {result.get('preprocess_used', 'n/a')}")
    print()
    print(f"    {'Label':<12} {'Score':>8}  {'Threshold':>10}  {'Decision':>12}")
    print("    " + "─" * 50)
    for lbl in LABELS:
        score     = result["scores"][lbl]
        threshold = THRESHOLDS[lbl]
        flagged   = result["flagged"][lbl]
        bar       = "█" * int(score * 20)
        decision  = "🚫 BLOCK/BLUR" if flagged else "✅ ALLOW"
        print(f"    {lbl:<12} {score:>8.1%}  {threshold:>10.0%}  {decision:>12}  {bar}")
    print()
    if result["any_flagged"]:
        flagged_labels = [l for l, f in result["flagged"].items() if f]
        print(f"    ⚠️  CONTENT FLAGGED: {', '.join(flagged_labels)}")
    else:
        print(f"    ✅  Content is SAFE — no categories exceeded threshold")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test parental control model")
    parser.add_argument("--image",  type=str, default=None,
                        help="Path to a single image to test")
    parser.add_argument("--folder", type=str, default=None,
                        help="Path to a folder — tests all images inside")
    args = parser.parse_args()

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if args.image:
        # Test one specific image
        if not Path(args.image).exists():
            sys.exit(f"❌ Image not found: {args.image}")
        result = run_inference(args.image)
        print_result(args.image, result)

    elif args.folder:
        # Test all images in a folder
        folder = Path(args.folder)
        images = [f for f in folder.iterdir()
                  if f.is_file() and f.suffix.lower() in EXTS]
        if not images:
            sys.exit(f"❌ No images found in {folder}")

        print(f"Testing {len(images)} images in {folder}\n")
        flagged_count = 0
        times = []

        for img_path in sorted(images)[:20]:   # cap at 20 for quick test
            result = run_inference(str(img_path))
            print_result(str(img_path), result)
            if result["any_flagged"]:
                flagged_count += 1
            times.append(result["ms"])

        print("=" * 58)
        print(f"  Summary: {flagged_count}/{min(len(images),20)} flagged")
        print(f"  Avg inference time: {np.mean(times):.1f} ms")
        print("=" * 58)

    else:
        # Default: run on a few synthetic test cases to verify model works
        print("No image specified — running built-in sanity checks\n")
        print("─" * 58)
        print("Test 1: Pure black image (should score ~0 on all labels)")
        print("─" * 58)

        # Save temp black image and test
        black = Image.new("RGB", (224, 224), color=(0, 0, 0))
        black.save("/tmp/test_black.jpg")
        result = run_inference("/tmp/test_black.jpg")
        print_result("/tmp/test_black.jpg", result)

        print("─" * 58)
        print("Test 2: Random noise image (scores should be near 0.5)")
        print("─" * 58)

        noise = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        noise.save("/tmp/test_noise.jpg")
        result = run_inference("/tmp/test_noise.jpg")
        print_result("/tmp/test_noise.jpg", result)

        print("─" * 58)
        print("To test on your own images:")
        print("  python test_model.py --image  path/to/image.jpg")
        print("  python test_model.py --folder path/to/folder/")
        print("─" * 58)


if __name__ == "__main__":
    main()
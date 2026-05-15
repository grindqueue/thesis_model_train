"""
train.py — Parental Control EfficientNet-B0  |  School RTX GPU Edition
=======================================================================
Labels  : alcohol · drugs · sexual · extremism
Output  : exported_model/parental_control_b0_int8.tflite
          exported_model/model_metadata.json

QUICK START
───────────
1. Edit the CONFIG section below (DATA_DIR, WORKING_DIR).
2. Open a terminal in the same folder as this file.

   Linux / SSH:
       pip3 install tensorflow[and-cuda] scikit-learn pillow pandas matplotlib tqdm
       python3 train.py

   Windows:
       pip install tensorflow scikit-learn pillow pandas matplotlib tqdm
       python train.py

3. If the power goes out or you close the terminal, just run the same
   command again — training resumes from the last saved epoch automatically.

NOTES
─────
- Works on any NVIDIA GPU (RTX 2080, 3090, 4090, A100, …)
- Works on Windows and Linux — paths handled by pathlib.Path
- If no GPU found, falls back to CPU (slow but functional)
- Dataset folder structure expected:
      DATA_DIR/
        alcohol/
        drugs/
        tobacco/      ← merged into drugs label
        sexual/
        extremism/
        normal/
"""

# ═══════════════════════════════════════════════════════════════════════════
# ▶ CONFIG — EDIT THESE TWO LINES BEFORE RUNNING
# ═══════════════════════════════════════════════════════════════════════════
DATA_DIR = r"C:\Users\username\Downloads\parental_dataset"
WORKING_DIR = r"C:\Users\username\Documents\parental_control_output"
# Training hyperparameters — safe to leave as-is
# ═══════════════════════════════════════════════════════════════════════════

IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32        # reduce to 16 if GPU runs out of memory
SEED            = 42
PHASE1_EPOCHS   = 20        # head-only training
PHASE2_EPOCHS   = 40        # fine-tuning top 30% of backbone
PHASE1_LR       = 1e-3
PHASE2_LR       = 2e-5
FREEZE_PCT      = 0.70      # freeze bottom 70% of backbone in phase 2

# ═══════════════════════════════════════════════════════════════════════════
# 0. Imports
# ═══════════════════════════════════════════════════════════════════════════

import json
import logging
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import keras
import matplotlib
matplotlib.use("Agg")   # no display needed — works over SSH and on Windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image as PilImage
from sklearn.metrics import (
    average_precision_score, f1_score,
    precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model, mixed_precision
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    CSVLogger, EarlyStopping, LambdaCallback,
    ModelCheckpoint, ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization, Concatenate, Dense, Dropout,
    GlobalAveragePooling2D, RandomBrightness, RandomContrast,
    RandomFlip, RandomRotation, RandomZoom,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# ═══════════════════════════════════════════════════════════════════════════
# 1. Paths  (pathlib handles / vs \ automatically)
# ═══════════════════════════════════════════════════════════════════════════

DATA_DIR    = Path(DATA_DIR)
WORKING_DIR = Path(WORKING_DIR)

CKPT_DIR    = WORKING_DIR / "checkpoints"
EXPORT_DIR  = WORKING_DIR / "exported_model"
LOG_DIR     = WORKING_DIR / "logs"
TRAIN_CSV   = WORKING_DIR / "train_clean.csv"
VAL_CSV     = WORKING_DIR / "val_clean.csv"
TEST_CSV    = WORKING_DIR / "test_clean.csv"

for d in [CKPT_DIR, EXPORT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Logging — writes to terminal AND logs/training.log simultaneously
#    So you can monitor remotely: tail -f logs/training.log
# ═══════════════════════════════════════════════════════════════════════════

log_path = LOG_DIR / "training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(str(log_path), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)
log.info(f"Python     : {sys.version}")
log.info(f"Platform   : {platform.platform()}")
log.info(f"TensorFlow : {tf.__version__}")
log.info(f"Working dir: {WORKING_DIR}")
log.info(f"Data dir   : {DATA_DIR}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Constants
# ═══════════════════════════════════════════════════════════════════════════

AUTOTUNE = tf.data.AUTOTUNE

# 4 labels — extremism added
LABELS = ["alcohol", "drugs", "sexual", "extremism"]

LABEL_MAP = {
    "normal"    : [0, 0, 0, 0],
    "alcohol"   : [1, 0, 0, 0],
    "drugs"     : [0, 1, 0, 0],
    "tobacco"   : [0, 1, 0, 0],   # merged into drugs
    "sexual"    : [0, 0, 1, 0],
    "extremism" : [0, 0, 0, 1],
}

# Per-label loss weights — extremism weighted highest (rare + high stakes)
LABEL_WEIGHTS = [1.5, 1.5, 2.0, 2.5]

# Safe image formats — GIF and WEBP excluded (TF cannot decode them safely)
EXTS             = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_PIL_FMTS = {"JPEG", "PNG", "BMP"}

# ═══════════════════════════════════════════════════════════════════════════
# 4. GPU setup — single GPU (school RTX)
# ═══════════════════════════════════════════════════════════════════════════

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    log.info(f"GPUs found : {[g.name for g in gpus]}")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # already initialised

    if gpus:
        mixed_precision.set_global_policy("mixed_float16")
        log.info("mixed_float16 enabled")
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        log.info(f"Using GPU: {gpus[0].name}\n")
    else:
        log.warning("No GPU found — running on CPU (will be slow)")
        strategy = tf.distribute.get_strategy()

    return strategy

# ═══════════════════════════════════════════════════════════════════════════
# 5. Custom metric
# ═══════════════════════════════════════════════════════════════════════════

@keras.saving.register_keras_serializable(package="ParentalControl")
class LabelAccuracy(tf.keras.metrics.Metric):

    def __init__(self, label_idx: int, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_idx = label_idx
        self.correct   = self.add_weight(name="correct", initializer="zeros")
        self.total     = self.add_weight(name="total",   initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        t = y_true[:, self.label_idx]
        p = tf.cast(y_pred[:, self.label_idx] > 0.5, tf.float32)
        self.correct.assign_add(
            tf.reduce_sum(tf.cast(tf.equal(t, p), tf.float32))
        )
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


def get_metrics():
    return [LabelAccuracy(i, f"{lbl}_acc") for i, lbl in enumerate(LABELS)]

# ═══════════════════════════════════════════════════════════════════════════
# 6. Loss — per-element BCE with per-label weights
# ═══════════════════════════════════════════════════════════════════════════

@keras.saving.register_keras_serializable(package="ParentalControl")
def weighted_bce(y_true, y_pred):
    """
    Per-element BCE → shape (batch, 4) → apply per-label weights → scalar.
    Extremism gets weight 2.5 because it is the rarest and highest-stakes label.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
    bce    = -(y_true * tf.math.log(y_pred)
               + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    w      = tf.constant([LABEL_WEIGHTS], dtype=tf.float32)  # (1, 4)
    return tf.reduce_mean(bce * w)

# ═══════════════════════════════════════════════════════════════════════════
# 7. Dataset helpers
# ═══════════════════════════════════════════════════════════════════════════

def find_dataset_root(base: Path) -> Path:
    """Walk subdirs to find the folder that contains class subfolders."""
    for p in [base] + sorted(base.rglob("*")):
        if not p.is_dir():
            continue
        subdirs = [d for d in p.iterdir() if d.is_dir()]
        has_images = any(
            any(f.suffix.lower() in EXTS for f in d.iterdir() if f.is_file())
            for d in subdirs
        )
        if len(subdirs) >= 2 and has_images:
            return p
    return base


def inspect_dataset(root: Path):
    log.info(f"Dataset root : {root}\n")
    log.info(f"{'Folder':<28} {'Images':>8}  {'Label'}")
    log.info("─" * 60)
    total = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        n = len([f for f in d.iterdir()
                 if f.is_file() and f.suffix.lower() in EXTS])
        total += n
        mapping = str(LABEL_MAP.get(d.name.lower(), "SKIPPED"))
        log.info(f"{d.name:<28} {n:>8}  {mapping}")
    log.info("─" * 60)
    log.info(f"{'TOTAL':<28} {total:>8}\n")


def build_dataframe(root: Path) -> pd.DataFrame:
    records, skipped = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        key = folder.name.lower().strip()
        if key not in LABEL_MAP:
            skipped.append(folder.name)
            continue
        lbl = LABEL_MAP[key]
        for img in folder.iterdir():
            if img.suffix.lower() in EXTS:
                records.append({
                    "path"      : str(img),
                    "alcohol"   : lbl[0],
                    "drugs"     : lbl[1],
                    "sexual"    : lbl[2],
                    "extremism" : lbl[3],
                })
    if skipped:
        log.info(f"Skipped folders (not in LABEL_MAP): {skipped}")
    return pd.DataFrame(records)


def remove_corrupt_images(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Drop images that are:
    - Corrupt / unreadable by PIL
    - Animated GIFs (multi-frame) disguised as .jpg
    - Wrong real format (WEBP/SVG/TIFF disguised as .jpg/.png)
    PIL reads actual file magic bytes — not the extension — so disguised
    files are always caught regardless of their filename.
    """
    good   = []
    counts = {"corrupt": 0, "animated": 0, "bad_format": 0}
    log.info(f"Scanning {len(df):,} {split_name} images ...")

    for i, row in enumerate(df.itertuples(), 1):
        if i % 5000 == 0:
            log.info(f"  {i:,} / {len(df):,}")
        try:
            with PilImage.open(row.path) as img:
                # Reject animated GIFs (any extension)
                if img.format == "GIF" or getattr(img, "n_frames", 1) > 1:
                    counts["animated"] += 1
                    continue
                # Reject WEBP / SVG / TIFF / ICO etc. disguised as jpg/png
                if img.format not in ALLOWED_PIL_FMTS:
                    counts["bad_format"] += 1
                    continue
                # Force full pixel decode — catches truncated files
                img.convert("RGB").load()
            good.append(row.Index)
        except Exception:
            counts["corrupt"] += 1

    removed = sum(counts.values())
    if removed:
        log.info(
            f"  Removed {removed:,}: "
            f"{counts['corrupt']} corrupt, "
            f"{counts['animated']} animated GIFs, "
            f"{counts['bad_format']} wrong format (WEBP/SVG/etc)"
        )
    else:
        log.info(f"  All {split_name} images valid ✓")

    return df.loc[good].reset_index(drop=True)


def get_splits(root: Path):
    """
    Cache split CSVs so the corruption scan only ever runs ONCE.
    On resume (after power cut / disconnect), loads CSVs instantly
    and skips the slow PIL scan entirely.
    """
    if TRAIN_CSV.exists() and VAL_CSV.exists() and TEST_CSV.exists():
        log.info("Cached splits found — skipping corruption scan")
        train_df = pd.read_csv(TRAIN_CSV)
        val_df   = pd.read_csv(VAL_CSV)
        test_df  = pd.read_csv(TEST_CSV)
        log.info(f"  Train:{len(train_df):,}  Val:{len(val_df):,}  Test:{len(test_df):,}\n")
        return train_df, val_df, test_df

    log.info("First run — scanning dataset ...\n")
    df = build_dataframe(root)
    log.info(f"Total images found : {len(df):,}")
    for lbl in LABELS:
        pos = df[lbl].sum()
        log.info(f"  {lbl:<12}: {pos:>6,} positive  ({pos/len(df):.1%})")
    safe = ((df[LABELS] == 0).all(axis=1)).sum()
    log.info(f"  {'normal':<12}: {safe:>6,} (all-zero)\n")

    df["_strat"] = df.apply(
        lambda r: f"{r.alcohol}{r.drugs}{r.sexual}{r.extremism}", axis=1
    )

    train_df, tmp = train_test_split(
        df, test_size=0.25, stratify=df["_strat"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        tmp, test_size=0.40, stratify=tmp["_strat"], random_state=SEED
    )

    train_df = remove_corrupt_images(train_df, "train")
    val_df   = remove_corrupt_images(val_df,   "val")
    test_df  = remove_corrupt_images(test_df,  "test")

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV,     index=False)
    test_df.to_csv(TEST_CSV,   index=False)
    log.info(
        f"\nSplits saved — "
        f"Train:{len(train_df):,}  Val:{len(val_df):,}  Test:{len(test_df):,}\n"
    )
    return train_df, val_df, test_df

# ═══════════════════════════════════════════════════════════════════════════
# 8. tf.data pipeline
# ═══════════════════════════════════════════════════════════════════════════

augment_layer = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.10),
    RandomZoom(0.10),
    RandomBrightness(0.15),
    RandomContrast(0.15),
], name="augmentation")


def load_image(path, label):
    """
    tf.io.decode_image reads magic bytes — not the file extension.
    Never crashes on disguised WEBP/GIF files that slipped past the scan.
    expand_animations=False → single frame even if somehow animated.
    """
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(
        raw, channels=3, dtype=tf.uint8, expand_animations=False
    )
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = tf.where(tf.math.is_finite(img), img, tf.zeros_like(img))
    return img, label


def make_ds(dataframe: pd.DataFrame,
            do_augment: bool = False,
            shuffle:    bool = False) -> tf.data.Dataset:
    paths  = tf.constant(dataframe["path"].values)
    labels = tf.constant(
        dataframe[LABELS].values, dtype=tf.float32
    )
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(dataframe), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if do_augment:
        ds = ds.map(
            lambda x, y: (augment_layer(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ═══════════════════════════════════════════════════════════════════════════
# 9. Model
# ═══════════════════════════════════════════════════════════════════════════

def build_and_compile(strategy,
                      lr: float,
                      backbone_trainable: bool = False,
                      freeze_pct: float = 0.0):
    """
    Build + compile inside strategy.scope() — required for MirroredStrategy
    and good practice for single-GPU too.
    """
    with strategy.scope():
        inp      = Input(shape=(*IMG_SIZE, 3), name="image")
        backbone = EfficientNetB0(
            include_top=False, weights="imagenet", input_tensor=inp
        )

        if backbone_trainable:
            backbone.trainable = True
            freeze_until = int(len(backbone.layers) * freeze_pct)
            for layer in backbone.layers[:freeze_until]:
                layer.trainable = False
            trainable_n = sum(
                np.prod(v.shape) for v in backbone.trainable_variables
            )
            log.info(
                f"Backbone: top {int((1-freeze_pct)*100)}% unfrozen "
                f"({trainable_n:,} trainable backbone params)"
            )
        else:
            backbone.trainable = False
            log.info("Backbone: fully frozen (head-only training)")

        x = GlobalAveragePooling2D(name="gap")(backbone.output)
        x = BatchNormalization(name="bn")(x)
        x = Dense(256, activation="swish",
                   kernel_regularizer=l2(1e-4), name="fc")(x)
        x = Dropout(0.35, name="drop")(x)

        # One output per label — 4 labels now including extremism
        outputs = [
            Dense(1, activation="sigmoid", dtype="float32", name=lbl)(x)
            for lbl in LABELS
        ]
        out   = Concatenate(name="scores")(outputs)
        model = Model(inputs=inp, outputs=out, name="ParentalControl_B0")

        model.compile(
            optimizer=Adam(learning_rate=lr, clipnorm=1.0),
            loss=weighted_bce,
            metrics=get_metrics(),
        )

    total     = model.count_params()
    trainable = sum(np.prod(v.shape) for v in model.trainable_variables)
    log.info(f"Total params     : {total:,}")
    log.info(f"Trainable params : {trainable:,}\n")
    return model, backbone


def load_model_safe(path: Path):
    """No custom_objects needed — registered serializable handles it."""
    return tf.keras.models.load_model(str(path))

# ═══════════════════════════════════════════════════════════════════════════
# 10. Resume detection
#     Checks what checkpoints already exist and returns where to continue.
#     This is what survives a power cut — just run the script again.
# ═══════════════════════════════════════════════════════════════════════════

def detect_resume_state() -> str:
    """
    Returns one of:
      'exported'    — everything done, just download outputs
      'phase2_done' — phase 2 trained, need to export
      'phase1_done' — phase 1 trained, need phase 2
      'fresh'       — nothing done yet
    """
    if (EXPORT_DIR / "parental_control_b0_int8.tflite").exists():
        return "exported"
    if (CKPT_DIR / "phase2_best.keras").exists():
        return "phase2_done"
    if (CKPT_DIR / "phase1_best.keras").exists():
        return "phase1_done"
    return "fresh"


def find_last_epoch_ckpt(phase: str):
    """
    Find the highest epoch checkpoint saved during an interrupted run.
    e.g. phase1_epoch_07.keras → resume from epoch 8.
    Returns (path, epoch_number) or (None, 0).
    """
    pattern  = f"{phase}_epoch_*.keras"
    ckpts    = sorted(CKPT_DIR.glob(pattern))
    if not ckpts:
        return None, 0
    latest   = ckpts[-1]
    # extract epoch number from filename e.g. phase1_epoch_07.keras
    try:
        epoch = int(latest.stem.split("_")[-1])
    except ValueError:
        epoch = 0
    return latest, epoch

# ═══════════════════════════════════════════════════════════════════════════
# 11. Training phases
# ═══════════════════════════════════════════════════════════════════════════

def make_callbacks(phase: str, monitor: str = "val_loss"):
    """
    Shared callback set for both phases.
    Per-epoch saves mean a power cut only ever loses < 1 epoch of work.
    """
    return [
        # Best model (by monitor metric)
        ModelCheckpoint(
            str(CKPT_DIR / f"{phase}_best.keras"),
            monitor=monitor, mode="max" if "acc" in monitor else "min",
            save_best_only=True, verbose=1
        ),
        # Every epoch — power-cut safety net
        ModelCheckpoint(
            str(CKPT_DIR / f"{phase}_epoch_{{epoch:02d}}.keras"),
            save_best_only=False, save_freq="epoch", verbose=0
        ),
        EarlyStopping(
            monitor="val_loss", patience=7,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.4,
            patience=3, min_lr=1e-9, verbose=1
        ),
        CSVLogger(str(LOG_DIR / f"{phase}.csv"), append=True),
    ]


def train_phase1(strategy, train_ds, val_ds):
    log.info("=" * 60)
    log.info("PHASE 1 — frozen backbone, training head only")
    log.info("=" * 60)

    model, _ = build_and_compile(strategy, lr=PHASE1_LR,
                                  backbone_trainable=False)

    # Resume mid-phase if interrupted
    last_ckpt, last_epoch = find_last_epoch_ckpt("phase1")
    initial_epoch = 0
    if last_ckpt and not (CKPT_DIR / "phase1_best.keras").exists():
        log.info(f"Resuming phase 1 from epoch {last_epoch} ({last_ckpt.name})")
        model = load_model_safe(last_ckpt)
        initial_epoch = last_epoch

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=make_callbacks("phase1", monitor="val_loss"),
        verbose=1,
    )

    best_loss = min(history.history["val_loss"])
    log.info(f"Phase 1 done — best val_loss: {best_loss:.4f}\n")
    return history


def train_phase2(strategy, train_ds, val_ds):
    log.info("=" * 60)
    log.info(f"PHASE 2 — fine-tuning top {int((1-FREEZE_PCT)*100)}% of backbone")
    log.info("=" * 60)

    model, _ = build_and_compile(strategy, lr=PHASE2_LR,
                                  backbone_trainable=True,
                                  freeze_pct=FREEZE_PCT)

    # Load phase 1 weights as starting point
    p1_ckpt = CKPT_DIR / "phase1_best.keras"
    p1      = load_model_safe(p1_ckpt)
    model.set_weights(p1.get_weights())
    del p1
    log.info("Phase 1 weights loaded into Phase 2 model\n")

    # Resume mid-phase if interrupted
    last_ckpt, last_epoch = find_last_epoch_ckpt("phase2")
    initial_epoch = 0
    if last_ckpt and not (CKPT_DIR / "phase2_best.keras").exists():
        log.info(f"Resuming phase 2 from epoch {last_epoch} ({last_ckpt.name})")
        model = load_model_safe(last_ckpt)
        initial_epoch = last_epoch

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=make_callbacks("phase2", monitor="val_loss"),
        verbose=1,
    )

    best_loss = min(history.history["val_loss"])
    log.info(f"Phase 2 done — best val_loss: {best_loss:.4f}\n")
    return history

# ═══════════════════════════════════════════════════════════════════════════
# 12. Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(model, test_ds):
    y_true_list, y_pred_list = [], []
    for imgs, labels in test_ds:
        preds = model(imgs, training=False).numpy()
        y_true_list.extend(labels.numpy())
        y_pred_list.extend(preds)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_bin  = (y_pred > 0.5).astype(int)

    log.info("\nPer-Label Test Metrics")
    log.info(f"{'Label':<14} {'AUC':>8} {'AP':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    log.info("─" * 60)
    for i, lbl in enumerate(LABELS):
        if y_true[:, i].sum() == 0:
            log.info(f"{lbl:<14}  no positive samples in test set")
            continue
        log.info(
            f"{lbl:<14}"
            f" {roc_auc_score(y_true[:,i], y_pred[:,i]):>8.4f}"
            f" {average_precision_score(y_true[:,i], y_pred[:,i]):>8.4f}"
            f" {f1_score(y_true[:,i], y_bin[:,i], zero_division=0):>8.4f}"
            f" {precision_score(y_true[:,i], y_bin[:,i], zero_division=0):>8.4f}"
            f" {recall_score(y_true[:,i], y_bin[:,i], zero_division=0):>8.4f}"
        )
    log.info("")
    return y_true, y_pred

# ═══════════════════════════════════════════════════════════════════════════
# 13. Threshold calibration
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_thresholds(y_true, y_pred) -> dict:
    """
    Find the threshold that maximises F1 per label.
    These thresholds go into model_metadata.json and are read by
    policy_engine.py at runtime to decide blur / block actions.
    """
    colors  = ["#1D9E75", "#BA7517", "#E24B4A", "#7B2D8B"]
    optimal = {}
    fig, axes = plt.subplots(1, len(LABELS), figsize=(5 * len(LABELS), 4))

    for i, (lbl, color) in enumerate(zip(LABELS, colors)):
        ax = axes[i]
        if y_true[:, i].sum() == 0:
            optimal[lbl] = 0.5
            ax.set_title(f"{lbl} — no positives")
            continue
        prec, rec, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
        f1s    = 2 * prec * rec / (prec + rec + 1e-8)
        best_t = round(float(thresholds[np.argmax(f1s[:-1])]), 3)
        optimal[lbl] = best_t
        ax.plot(thresholds, f1s[:-1], color=color, linewidth=2)
        ax.axvline(best_t, color="black", linestyle="--", alpha=0.7)
        ax.set_title(f"{lbl}  |  best t={best_t:.3f}", fontweight="bold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1")
        ax.grid(True, alpha=0.25)

    plt.suptitle("Threshold vs F1 per label", fontweight="bold")
    plt.tight_layout()
    out = EXPORT_DIR / "threshold_calibration.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    log.info(f"Threshold calibration plot saved: {out}")

    log.info("\nOptimal thresholds (used by policy_engine.py):")
    for lbl, t in optimal.items():
        log.info(f"  {lbl:<14}: {t}")
    log.info("")
    return optimal

# ═══════════════════════════════════════════════════════════════════════════
# 14. Training history plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_history(histories: list):
    combined, boundaries, acc = {}, [], 0
    for h in histories:
        for k, v in h.history.items():
            combined.setdefault(k, []).extend(v)
        if h is not histories[-1]:
            acc += len(h.history["loss"])
            boundaries.append(acc)

    epochs = range(1, len(combined["loss"]) + 1)
    colors = ["#1D9E75", "#BA7517", "#E24B4A", "#7B2D8B", "#534AB7"]
    plots  = [(f"{lbl}_acc", f"val_{lbl}_acc", f"{lbl.capitalize()} Acc", c)
              for lbl, c in zip(LABELS, colors)]
    plots += [("loss", "val_loss", "Weighted Loss", "#333333")]

    cols = 3
    rows = (len(plots) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

    for ax, (tk, vk, title, color) in zip(axes, plots):
        if tk not in combined:
            ax.set_visible(False)
            continue
        ax.plot(epochs, combined[tk], color=color, label="Train")
        ax.plot(epochs, combined[vk], color=color, alpha=0.45,
                linestyle="--", label="Val")
        for b in boundaries:
            ax.axvline(b, color="gray", linestyle=":", alpha=0.6)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    for ax in axes[len(plots):]:
        ax.set_visible(False)

    plt.suptitle("EfficientNet-B0 Training History  |  Parental Control",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = EXPORT_DIR / "training_history.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    log.info(f"Training history plot saved: {out}\n")

# ═══════════════════════════════════════════════════════════════════════════
# 15. Export — TFLite INT8 + metadata JSON
# ═══════════════════════════════════════════════════════════════════════════

def export_model(best_model, train_ds, optimal_thresholds: dict):
    EXPORT_DIR.mkdir(exist_ok=True)

    # Full Keras model (retraining / fine-tuning later)
    keras_path = EXPORT_DIR / "parental_control_b0.keras"
    best_model.save(str(keras_path))
    log.info(f".keras saved: {keras_path}")

    # TFLite INT8 quantised
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)

    def rep_data():
        for imgs, _ in train_ds.take(60):
            for img in imgs:
                yield [tf.expand_dims(img, 0)]

    converter.optimizations             = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset    = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type      = tf.float32
    converter.inference_output_type     = tf.float32

    tflite_bytes = converter.convert()
    tflite_path  = EXPORT_DIR / "parental_control_b0_int8.tflite"
    tflite_path.write_bytes(tflite_bytes)

    keras_mb  = keras_path.stat().st_size  / 1e6
    tflite_mb = tflite_path.stat().st_size / 1e6
    log.info(
        f"TFLite INT8: {tflite_mb:.1f} MB  "
        f"(was {keras_mb:.1f} MB, {keras_mb/tflite_mb:.1f}× smaller)"
    )

    # Metadata JSON — read by policy_engine.py at runtime
    metadata = {
        "model_name"         : "ParentalControl_EfficientNetB0_MultiLabel",
        "version"            : "2.0",
        "labels"             : LABELS,
        "label_indices"      : {l: i for i, l in enumerate(LABELS)},
        "input_size"         : list(IMG_SIZE),
        "preprocessing"      : "efficientnet.preprocess_input",
        "output_type"        : "multi_label_sigmoid",
        "output_order"       : LABELS,
        "optimal_thresholds" : optimal_thresholds,
        "label_weights"      : {l: w for l, w in zip(LABELS, LABEL_WEIGHTS)},
        "deployment_target"  : "Linux OS daemon (policy_engine.py)",
        "tflite_quantization": "INT8 with representative dataset",
        "trained_on"         : platform.platform(),
    }
    meta_path = EXPORT_DIR / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info(f"model_metadata.json saved: {meta_path}")

    log.info("\nOutput files:")
    for f in sorted(EXPORT_DIR.rglob("*")):
        if f.is_file():
            log.info(f"  {f.name:<55} {f.stat().st_size/1e6:.2f} MB")

    return tflite_path


# ═══════════════════════════════════════════════════════════════════════════
# 16. TFLite verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_tflite(tflite_path: Path, test_ds):
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    log.info(f"\nTFLite input  : {inp_d['shape']}  {inp_d['dtype']}")
    log.info(f"TFLite output : {out_d['shape']}  {out_d['dtype']}")
    log.info(f"Output order  : {LABELS}\n")

    imgs_b, lbls_b = next(iter(test_ds))
    for img, lbl in zip(imgs_b.numpy()[:5], lbls_b.numpy()[:5]):
        arr = np.expand_dims(img, 0).astype(np.float32)
        interp.set_tensor(inp_d["index"], arr)
        interp.invoke()
        s = interp.get_tensor(out_d["index"])[0]
        pred_str = "  ".join(f"{l}={s[i]:.2f}" for i, l in enumerate(LABELS))
        true_str = "  ".join(f"{l}={lbl[i]:.0f}" for i, l in enumerate(LABELS))
        log.info(f"  pred  {pred_str}")
        log.info(f"  true  {true_str}")
        log.info("")

    log.info("TFLite verified — ready to deploy\n")

# ═══════════════════════════════════════════════════════════════════════════
# 17. Main — fully resumable after power cut or SSH disconnect
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Validate paths
    if not DATA_DIR.exists():
        log.error(f"Data directory not found: {DATA_DIR}")
        log.error("Edit DATA_DIR at the top of this file and try again.")
        sys.exit(1)

    strategy = setup_gpu()

    # Dataset
    root = find_dataset_root(DATA_DIR)
    inspect_dataset(root)
    train_df, val_df, test_df = get_splits(root)

    train_ds = make_ds(train_df, do_augment=True,  shuffle=True)
    val_ds   = make_ds(val_df,   do_augment=False, shuffle=False)
    test_ds  = make_ds(test_df,  do_augment=False, shuffle=False)
    log.info(
        f"Batches — train:{len(train_ds)}  "
        f"val:{len(val_ds)}  test:{len(test_ds)}\n"
    )

    # Resume detection — figures out where we left off
    state = detect_resume_state()
    log.info(f"Resume state: {state}\n")

    if state == "exported":
        log.info("Already fully exported. Copy files from:")
        log.info(f"  {EXPORT_DIR}")
        return

    histories = []

    # Phase 1
    if state == "fresh":
        h1 = train_phase1(strategy, train_ds, val_ds)
        histories.append(h1)
    else:
        log.info("Phase 1 already done — skipping\n")

    # Phase 2
    if state in ("fresh", "phase1_done"):
        h2 = train_phase2(strategy, train_ds, val_ds)
        histories.append(h2)
    else:
        log.info("Phase 2 already done — loading best model\n")

    # Plot training curves
    if histories:
        plot_history(histories)

    # Load best model and evaluate
    log.info("Loading phase2_best.keras ...")
    best_model = load_model_safe(CKPT_DIR / "phase2_best.keras")
    log.info("Loaded\n")

    y_true, y_pred = evaluate(best_model, test_ds)
    optimal        = calibrate_thresholds(y_true, y_pred)
    tflite_path    = export_model(best_model, train_ds, optimal)
    verify_tflite(tflite_path, test_ds)

    log.info("Training complete!")
    log.info(f"  Outputs: {EXPORT_DIR}")
    log.info("  Deploy : parental_control_b0_int8.tflite + model_metadata.json")


if __name__ == "__main__":
    main()
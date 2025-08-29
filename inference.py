# inference.py — lazy init, ensemble-ready (fixed)

import os, json, time
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

# -------- CONFIG (match your training) --------
MODELS_DIR = os.getenv("MODELS_DIR", "models")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", os.path.join(MODELS_DIR, "class_names.json"))
TARGET_SIZE = (299, 299)    # trained with img_size=299
RESCALE = None              # trained WITHOUT rescale
DTYPE = np.float32

# -------- GLOBALS (lazy) --------
MODELS = None               # list[tf.keras.Model]
MODEL_NAMES = None          # list[str] pretty names aligned with MODELS
CLASS_NAMES = None          # list[str] index-ordered

# -------- Helpers --------
def clean_label(name: str) -> str:
    """Return text after the first '-' if present, else the original."""
    parts = name.split('-', 1)
    return parts[1].strip() if len(parts) == 2 else name.strip()

def model_short_name(path: str) -> str:
    """Pretty label from file path."""
    base = os.path.basename(path).lower()
    if "resnet101" in base or "resnet101" in base: return "ResNet101"
    if "resnet50" in base: return "ResNet50"
    if "efficientnet" in base and "b0" in base: return "EfficientNetB0"
    if "vgg16" in base: return "VGG16"
    return os.path.splitext(os.path.basename(path))[0]

def _apply_exif_orientation(pil_img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(pil_img)
    except Exception:
        return pil_img

def preprocess(pil_img: Image.Image) -> np.ndarray:
    """PIL -> (1, H, W, C) float32, matching training preprocessing."""
    pil_img = _apply_exif_orientation(pil_img)
    pil_img = pil_img.convert("RGB").resize(TARGET_SIZE)
    x = np.asarray(pil_img, dtype=DTYPE)
    if RESCALE is not None:
        x = x * RESCALE
    x = np.expand_dims(x, axis=0)
    return x

def _load_class_names(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"class_names.json not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if isinstance(loaded, dict):
        idxmax = max(loaded.values())
        names = [None]*(idxmax+1)
        for name, idx in loaded.items():
            names[idx] = name
        return names
    if isinstance(loaded, list):
        return loaded
    raise ValueError("class_names.json must be a list (idx-ordered) or dict (name->idx).")

def _load_models(dirpath: str):
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f"Models dir not found: {dirpath}")
    models, names = [], []
    for fname in sorted(os.listdir(dirpath)):
        if fname.endswith((".keras", ".h5")) and fname != "class_names.json":
            path = os.path.join(dirpath, fname)
            m = tf.keras.models.load_model(path, compile=False)
            models.append(m)
            names.append(model_short_name(path))
            print(f"[inference] loaded model: {path}")
    if not models:
        raise FileNotFoundError(f"No model files (*.keras/*.h5) found in {dirpath}")
    return models, names

def _check_output_dims(models, class_names):
    """Verify every model's output size == number of classes."""
    try:
        num_out = int(models[0].outputs[0].shape[-1])
    except Exception:
        num_out = len(class_names)
    if len(class_names) != num_out:
        raise ValueError(f"Class count mismatch: model outputs {num_out}, class_names has {len(class_names)}.")
    for m in models[1:]:
        try:
            n = int(m.outputs[0].shape[-1])
        except Exception:
            n = len(class_names)
        if n != num_out:
            raise ValueError("Ensemble models disagree on output size; ensure same classes/order for all.")

def init_models_if_needed():
    """Lazy initializer; safe to call multiple times."""
    global MODELS, MODEL_NAMES, CLASS_NAMES
    if CLASS_NAMES is None:
        CLASS_NAMES = _load_class_names(CLASS_NAMES_PATH)
        print(f"[inference] loaded {len(CLASS_NAMES)} classes from {CLASS_NAMES_PATH}")
    if MODELS is None or MODEL_NAMES is None:
        MODELS, MODEL_NAMES = _load_models(MODELS_DIR)
        _check_output_dims(MODELS, CLASS_NAMES)
        print(f"[inference] ensemble size: {len(MODELS)} -> {MODEL_NAMES}")

def get_clean_class_list():
    init_models_if_needed()
    return [clean_label(c) for c in CLASS_NAMES]

# Try eager-load (won't crash notebooks if files missing)
try:
    init_models_if_needed()
except Exception as e:
    print("[inference] deferred model load:", e)

# -------- Inference (ensemble) --------
def predict_ensemble_topk(pil_img, topk=3, aggregate="mean"):
    """
    Returns:
      - per_model: list[{'model','top1_class','top1_prob'}]
      - ensemble_topk: list[(class_name_clean, prob)]
      - probs: list[float] (final aggregated distribution)
      - latency_ms: float
      - classes_clean: list[str] (all classes cleaned)
    """
    init_models_if_needed()
    per_model_probs = {} 

    x = preprocess(pil_img)
    t0 = time.time()

    probs_list = []
    per_model = []

    # Per-model predictions with robust shape checks
    for m, name in zip(MODELS, MODEL_NAMES):
        p = m.predict(x, verbose=0)
        if not hasattr(p, "shape"):
            raise RuntimeError(f"Model {name} returned non-array prediction: {type(p)}")
        if p.ndim != 2 or p.shape[0] != 1:
            raise RuntimeError(f"Model {name} returned shape {p.shape}, expected (1, C)")
        p = p[0]
        if p.ndim != 1 or p.shape[0] != len(CLASS_NAMES):
            raise RuntimeError(
                f"Model {name} output length {p.shape[0]} != num classes {len(CLASS_NAMES)}"
            )
        p = p.astype(np.float64)
        probs_list.append(p)
        per_model_probs[name] = p.tolist() 

        # single-model top1
        idx = int(np.argmax(p))
        per_model.append({
            "model": name,
            "top1_class": clean_label(CLASS_NAMES[idx]),
            "top1_prob": float(p[idx])
        })

    if not probs_list:
        raise RuntimeError("No models produced predictions. Check MODELS_DIR and model files.")

    # Aggregate across models
    if aggregate == "gmean":
        stacked = np.stack(probs_list, axis=0)     # (M, C)
        logp = np.log(np.clip(stacked, 1e-12, 1.0))
        agg = np.exp(logp.mean(axis=0))
    else:
        agg = np.mean(probs_list, axis=0)

    s = float(agg.sum())
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError(f"Aggregated probabilities invalid (sum={s}).")
    agg = (agg / s).astype(np.float64)

    latency_ms = (time.time() - t0) * 1000.0

    # Ensemble top-k with cleaned labels
    idx_sorted = np.argsort(agg)[::-1]
    k = min(topk, len(idx_sorted))
    ensemble_topk = [(clean_label(CLASS_NAMES[i]), float(agg[i])) for i in idx_sorted[:k]]

    classes_clean = [clean_label(c) for c in CLASS_NAMES]
    return per_model, ensemble_topk, agg.tolist(), latency_ms, classes_clean, per_model_probs

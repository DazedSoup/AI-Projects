"""Train the three Keras MLP classifiers.

CLI:  python -m cyberarena.ml.train --model all|malware|phishing|network

Reads ``data/processed/<name>.npz`` (run ``python -m cyberarena.ml.datasets --all`` first) and writes
``artifacts/models/<name>.keras``, ``artifacts/models/<name>_preprocess.json`` and
``artifacts/metrics/<name>.json`` (see docs/contracts.md, "Classifier artifacts").
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cyberarena.ml.datasets import NAMES, PROCESSED_DIR, ROOT, SEED, load_processed

ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
LABEL_MAP = {"0": "benign", "1": "malicious"}


def build_model(n_features: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.2):
    import keras

    inputs = keras.Input(shape=(n_features,), name="features")
    x = inputs
    for i, units in enumerate(hidden):
        x = keras.layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = keras.layers.Dropout(dropout, name=f"dropout_{i}")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="p_malicious")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def _class_weight(y: np.ndarray) -> dict[int, float]:
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    return {c: float(len(y) / (2.0 * counts[c])) for c in (0, 1) if counts[c] > 0}


def fit(arrays: dict[str, np.ndarray], seed: int = SEED, epochs: int = 100, batch_size: int = 128,
        patience: int = 8, verbose: int = 0):  # fmt: skip
    """Fit an MLP on X_train with early stopping on X_val. Returns the trained model."""
    import keras

    keras.utils.set_random_seed(seed)
    model = build_model(arrays["X_train"].shape[1])
    model.fit(
        arrays["X_train"],
        arrays["y_train"].astype("float32"),
        validation_data=(arrays["X_val"], arrays["y_val"].astype("float32")),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=_class_weight(arrays["y_train"]),
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        ],
        verbose=verbose,
    )
    return model


def evaluate(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(p) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) == 2 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict(np.asarray(X, dtype=np.float32), batch_size=4096, verbose=0).reshape(-1)


def train_one(name: str, processed_dir: Path = PROCESSED_DIR, models_dir: Path = MODELS_DIR,
              metrics_dir: Path = METRICS_DIR, seed: int = SEED, verbose: int = 0) -> dict:  # fmt: skip
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")  # silence benign predict() retracing warnings
    arrays = load_processed(name, processed_dir)
    t0 = time.time()
    model = fit(arrays, seed=seed, verbose=verbose)
    elapsed = time.time() - t0

    metrics = evaluate(arrays["y_test"], predict(model, arrays["X_test"]))
    metrics.update(
        n_train=len(arrays["y_train"]),
        n_val=len(arrays["y_val"]),
        n_test=len(arrays["y_test"]),
        source=str(arrays["source"]),
        train_roc_auc=float(roc_auc_score(arrays["y_train"], predict(model, arrays["X_train"]))),
        train_seconds=round(elapsed, 1),
    )

    models_dir, metrics_dir = Path(models_dir), Path(metrics_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / f"{name}.keras")
    preprocess = {
        "feature_names": [str(f) for f in arrays["feature_names"]],
        "scaler_mean": [float(v) for v in arrays["scaler_mean"]],
        "scaler_scale": [float(v) for v in arrays["scaler_scale"]],
        "label_map": LABEL_MAP,
    }
    (models_dir / f"{name}_preprocess.json").write_text(json.dumps(preprocess, indent=2))
    (metrics_dir / f"{name}.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def format_table(results: dict[str, dict]) -> str:
    head = f"{'model':<9} {'n_train':>7} {'n_test':>6} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6} {'auc':>6}"
    rows = [head, "-" * len(head)]
    for name, m in results.items():
        rows.append(
            f"{name:<9} {m['n_train']:>7} {m['n_test']:>6} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['roc_auc']:>6.3f}"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train cyberarena classifiers")
    ap.add_argument("--model", choices=("all", *NAMES), default="all")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args(argv)
    results = {}
    for name in NAMES if args.model == "all" else (args.model,):
        m = train_one(name, seed=args.seed, verbose=args.verbose)
        print(f"[{name}] trained in {m['train_seconds']}s, test roc_auc={m['roc_auc']:.4f}")
        if m["roc_auc"] >= 0.999:
            print(f"[{name}] WARNING: near-perfect ROC-AUC, check for label leakage")
        results[name] = m
    print(format_table(results))


if __name__ == "__main__":
    main()

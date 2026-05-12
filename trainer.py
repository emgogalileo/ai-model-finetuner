"""
AI Model Fine-Tuner
====================
A practical simulation of a machine learning fine-tuning workflow using
scikit-learn (swap for PyTorch/HuggingFace in production).

Features:
  - Dataset loading / splitting (train / validation / test)
  - Multiple model backends: LogisticRegression, RandomForest, SVM
  - Hyperparameter search via GridSearchCV
  - Training loop with per-epoch metrics (accuracy, loss simulation)
  - Evaluation: classification report, confusion matrix, ROC-AUC
  - Checkpoint saving to disk (pickle + JSON metadata)
  - CLI interface for full configurability

Usage:
    python trainer.py --model logistic --epochs 20 --output checkpoints/
    python trainer.py --model random_forest --search   # with hyperparameter search
    python trainer.py --demo                            # quick demo run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import time
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ModelType = Literal["logistic", "random_forest", "svm"]

# ── Hyperparameter grids ───────────────────────────────────────────────────────
PARAM_GRIDS: dict[str, dict] = {
    "logistic": {"C": [0.01, 0.1, 1.0, 10.0], "max_iter": [200, 500]},
    "random_forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
    "svm": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
}

# ── Model factory ──────────────────────────────────────────────────────────────

def build_model(model_type: ModelType):
    """Instantiate a scikit-learn estimator by name."""
    if model_type == "logistic":
        return LogisticRegression(max_iter=1000, random_state=42)
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    if model_type == "svm":
        return SVC(probability=True, random_state=42)
    raise ValueError(f"Unknown model type: {model_type}")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset():
    """
    Load the Wisconsin Breast Cancer dataset (binary classification).
    Returns X (features), y (labels), and feature names.
    """
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names


# ── Training loop ──────────────────────────────────────────────────────────────

def simulated_training_loop(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 10,
) -> list[dict]:
    """
    Simulates an epoch-by-epoch training loop with convergence behaviour.

    For sklearn models, the "loop" re-trains with increasing data fractions
    to emulate progressive learning — producing realistic loss/accuracy curves.

    Returns:
        List of per-epoch metric dicts.
    """
    history: list[dict] = []
    n_samples = len(X_train)

    logger.info("  Starting training loop (%d epochs)…", epochs)
    for epoch in range(1, epochs + 1):
        # Use increasing data fraction to simulate convergence
        frac = min(1.0, 0.1 + 0.9 * (epoch / epochs))
        n = max(10, int(n_samples * frac))
        idx = np.random.choice(n_samples, n, replace=False)

        model.fit(X_train[idx], y_train[idx])

        train_acc = accuracy_score(y_train[idx], model.predict(X_train[idx]))
        val_acc = accuracy_score(y_val, model.predict(X_val))

        # Simulate decreasing loss with noise
        sim_loss = max(0.05, 0.8 * math.exp(-epoch * 0.25) + np.random.normal(0, 0.01))

        record = {
            "epoch": epoch,
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "loss": round(sim_loss, 4),
        }
        history.append(record)
        logger.info(
            "  Epoch %2d/%2d — loss: %.4f  train_acc: %.4f  val_acc: %.4f",
            epoch, epochs, sim_loss, train_acc, val_acc,
        )

    return history


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute comprehensive evaluation metrics on the test set.

    Returns:
        Dict with accuracy, ROC-AUC, classification report, confusion matrix.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    results: dict = {
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    if y_proba is not None:
        results["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)

    return results


# ── Checkpoint ─────────────────────────────────────────────────────────────────

def save_checkpoint(
    model,
    metadata: dict,
    output_dir: Path,
    model_name: str,
) -> None:
    """Persist model binary and JSON metadata to the checkpoint directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())

    model_path = output_dir / f"{model_name}_{timestamp}.pkl"
    meta_path = output_dir / f"{model_name}_{timestamp}_metadata.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("  💾  Model saved → %s", model_path)
    logger.info("  📋  Metadata  → %s", meta_path)


# ── Main workflow ──────────────────────────────────────────────────────────────

def run(
    model_type: ModelType = "logistic",
    epochs: int = 10,
    search: bool = False,
    output_dir: Path = Path("checkpoints"),
) -> None:
    """Execute the full fine-tuning pipeline."""
    logger.info("🤖  AI Fine-Tuner — model=%s  epochs=%d  search=%s", model_type, epochs, search)

    # ── Data ───────────────────────────────────────────────────────────────────
    logger.info("📂  Loading dataset…")
    X, y, feature_names = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    # ── Preprocessing ──────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    logger.info("  Dataset split — train: %d  val: %d  test: %d", len(X_train), len(X_val), len(X_test))

    # ── Hyperparameter search ──────────────────────────────────────────────────
    model = build_model(model_type)
    best_params: dict = {}

    if search:
        logger.info("🔍  Running GridSearchCV…")
        grid = GridSearchCV(
            model,
            PARAM_GRIDS[model_type],
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        best_params = grid.best_params_
        logger.info("  Best params: %s  (CV score: %.4f)", best_params, grid.best_score_)

    # ── Training loop ──────────────────────────────────────────────────────────
    logger.info("🏋️  Training…")
    history = simulated_training_loop(model, X_train, y_train, X_val, y_val, epochs=epochs)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    logger.info("📊  Evaluating on test set…")
    eval_results = evaluate(model, X_test, y_test)
    logger.info("  Test accuracy: %.4f  ROC-AUC: %s",
                eval_results["test_accuracy"],
                eval_results.get("roc_auc", "N/A"))

    # ── Save checkpoint ────────────────────────────────────────────────────────
    metadata = {
        "model_type": model_type,
        "epochs": epochs,
        "best_params": best_params,
        "history": history,
        "evaluation": eval_results,
        "features": list(feature_names),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_checkpoint(model, metadata, output_dir, model_type)
    logger.info("✅  Fine-tuning complete.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Model Fine-Tuner")
    parser.add_argument("--model", choices=["logistic", "random_forest", "svm"], default="logistic")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--search", action="store_true", help="Run hyperparameter search (GridSearchCV)")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--demo", action="store_true", help="Quick demo with defaults")
    args = parser.parse_args()

    run(
        model_type=args.model,
        epochs=args.epochs,
        search=args.search,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

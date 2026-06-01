"""
train.py
---------
End-to-end training pipeline.

Run:
    python src/train.py

Outputs (saved to models/):
    - classifier.joblib        trained classifier
    - pca.joblib               fitted PCA transform
    - embeddings_cache.npy     raw BERT embeddings (speeds up re-runs)
    - label_encoder.joblib     (unused, but good practice)
    - train_metadata.json      split sizes, base rate, timestamp
"""

import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import preprocess_dataframe
from src.label_simulation import generate_labels
from src.embeddings import get_embeddings
from src.evaluate import evaluate_model, save_evaluation_plots

DATA_PATH = Path("data/mtsamples.csv")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def load_and_prepare_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load MTSamples, preprocess, and generate labels."""
    print(f"\n[train] Loading data from {data_path} ...")
    df = pd.read_csv(data_path)

    print(f"[train] Raw rows: {len(df)}")
    print(f"[train] Columns: {df.columns.tolist()}")

    # MTSamples column is 'transcription'; handle alternate column names
    text_col = None
    for candidate in ["transcription", "text", "note", "content"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        raise ValueError(
            f"Could not find a text column. Available: {df.columns.tolist()}"
        )

    # Preprocess
    print("[train] Preprocessing notes ...")
    df = preprocess_dataframe(df, text_col=text_col)

    # Generate simulated labels
    df = generate_labels(df, text_col="clean_text")

    print(f"[train] Usable notes after preprocessing: {len(df)}")
    print(f"[train] Label distribution:\n{df['readmitted_30d'].value_counts(normalize=True)}")

    return df


def train(
    model_type: str = "xgboost",
    test_size: float = 0.2,
    random_state: int = 42,
    data_path: Path = DATA_PATH,
):
    """
    Full training pipeline.

    Parameters
    ----------
    model_type : str
        'logistic' or 'xgboost'
    test_size : float
        Fraction of data held out for evaluation.
    random_state : int
    data_path : Path

    Returns
    -------
    dict with evaluation metrics
    """
    start_time = time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Data preparation
    # ----------------------------------------------------------------
    df = load_and_prepare_data(data_path)

    # Stratified train/test split
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        stratify=df["readmitted_30d"],
        random_state=random_state,
    )
    print(f"\n[train] Train: {len(train_idx)} | Test: {len(test_idx)}")

    # ----------------------------------------------------------------
    # 2. Feature extraction
    # ----------------------------------------------------------------
    print("\n[train] Extracting ClinicalBERT embeddings ...")
    X_all, pca = get_embeddings(
        df,
        text_col="clean_text",
        use_cache=True,
        pca_components=64,
        fit_pca_on_train=True,
        train_indices=train_idx,
    )

    y = df["readmitted_30d"].values
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ----------------------------------------------------------------
    # 3. Model selection and training
    # ----------------------------------------------------------------
    print(f"\n[train] Training {model_type} classifier ...")

    if model_type == "logistic":
        classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            )),
        ])
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        classifier = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'logistic' or 'xgboost'.")

    # Cross-validation on training set
    print("[train] 5-fold cross-validation on training set ...")
    cv_scores = cross_val_score(
        classifier, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        scoring="roc_auc",
        n_jobs=-1,
    )
    print(f"[train] CV AUC-ROC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Final fit on full training set
    classifier.fit(X_train, y_train)

    # ----------------------------------------------------------------
    # 4. Evaluation
    # ----------------------------------------------------------------
    print("\n[train] Evaluating on held-out test set ...")
    metrics = evaluate_model(classifier, X_test, y_test, save_dir=RESULTS_DIR)

    # ----------------------------------------------------------------
    # 5. Persist artifacts
    # ----------------------------------------------------------------
    model_path = MODELS_DIR / "classifier.joblib"
    joblib.dump(classifier, model_path)
    print(f"\n[train] Model saved → {model_path}")

    # Save training metadata
    metadata = {
        "model_type": model_type,
        "trained_at": datetime.utcnow().isoformat(),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_base_rate": float(y_train.mean()),
        "test_base_rate": float(y_test.mean()),
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "test_metrics": metrics,
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    with open(MODELS_DIR / "train_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[train] Done in {metadata['elapsed_seconds']}s")
    print(f"[train] Test AUC-ROC: {metrics['auc_roc']:.3f}")
    print(f"[train] Results saved → {RESULTS_DIR}/")

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train readmission risk classifier")
    parser.add_argument(
        "--model", choices=["logistic", "xgboost"], default="xgboost",
        help="Classifier type (default: xgboost)"
    )
    parser.add_argument(
        "--data", type=Path, default=DATA_PATH,
        help="Path to MTSamples CSV"
    )
    args = parser.parse_args()

    train(model_type=args.model, data_path=args.data)

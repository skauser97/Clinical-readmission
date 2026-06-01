"""
embeddings.py
--------------
ClinicalBERT feature extraction.

Model: emilyalsentzer/Bio_ClinicalBERT
  - Fine-tuned on MIMIC-III clinical notes
  - 768-dimensional [CLS] token embeddings
  - 512 token limit (handled by truncation in preprocessing)

We extract the [CLS] token embedding as a fixed-length representation
of the entire note, then optionally apply PCA to reduce dimensionality
before passing to the classifier.

Memory note: processing the full MTSamples dataset (~5k notes) takes
roughly 2-4 minutes on CPU and ~500MB RAM. A GPU will run ~10x faster.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm


MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MODELS_DIR = Path(__file__).parent.parent / "models"
CACHE_PATH = MODELS_DIR / "embeddings_cache.npy"
PCA_PATH = MODELS_DIR / "pca.joblib"

MAX_LENGTH = 512   # ClinicalBERT hard limit
BATCH_SIZE = 16    # safe default for 8GB RAM; reduce to 8 if OOM
PCA_COMPONENTS = 64


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        print("[embeddings] Using GPU")
        return torch.device("cuda")
    print("[embeddings] Using CPU (this will be slow — consider a GPU)")
    return torch.device("cpu")


def load_model(model_name: str = MODEL_NAME):
    """Load tokenizer and model from HuggingFace (cached after first download)."""
    print(f"[embeddings] Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def extract_cls_embeddings(
    texts: List[str],
    tokenizer,
    model,
    device: Optional[torch.device] = None,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Extract [CLS] token embeddings for a list of texts.

    Parameters
    ----------
    texts : List[str]
        Cleaned note texts.
    tokenizer, model
        From load_model().
    device : torch.device
        Auto-detected if None.
    batch_size : int
        Reduce if running out of memory.
    max_length : int
        Token truncation limit.
    show_progress : bool

    Returns
    -------
    np.ndarray of shape (n_samples, 768)
    """
    if device is None:
        device = _get_device()

    model = model.to(device)
    all_embeddings = []

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting embeddings", unit="batch")

    with torch.no_grad():
        for start in iterator:
            batch_texts = texts[start: start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            # [CLS] token is at index 0 of the last hidden state
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)


def fit_pca(
    embeddings: np.ndarray,
    n_components: int = PCA_COMPONENTS,
    save: bool = True,
) -> tuple:
    """
    Fit PCA on training embeddings and return reduced embeddings + fitted PCA.

    Parameters
    ----------
    embeddings : np.ndarray (n_samples, 768)
    n_components : int
    save : bool
        If True, serialize PCA to models/pca.joblib.

    Returns
    -------
    (reduced_embeddings: np.ndarray, pca: PCA)
    """
    print(f"[embeddings] Fitting PCA: 768 → {n_components} dimensions ...")
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)

    explained = pca.explained_variance_ratio_.sum()
    print(f"[embeddings] PCA explains {explained:.1%} of variance")

    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(pca, PCA_PATH)
        print(f"[embeddings] PCA saved → {PCA_PATH}")

    return reduced, pca


def load_pca() -> PCA:
    """Load a previously fitted PCA."""
    if not PCA_PATH.exists():
        raise FileNotFoundError(f"PCA not found at {PCA_PATH}. Run train.py first.")
    return joblib.load(PCA_PATH)


def get_embeddings(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    use_cache: bool = True,
    pca_components: int = PCA_COMPONENTS,
    fit_pca_on_train: bool = True,
    train_indices: Optional[np.ndarray] = None,
) -> tuple:
    """
    Full embedding pipeline: load model → extract → PCA.

    Parameters
    ----------
    df : pd.DataFrame
    text_col : str
    use_cache : bool
        If True, cache raw embeddings to disk to avoid re-running BERT.
    pca_components : int
    fit_pca_on_train : bool
        If True, fit PCA on train_indices only (prevents leakage).
    train_indices : np.ndarray
        Indices of training samples (required if fit_pca_on_train=True).

    Returns
    -------
    (embeddings_pca: np.ndarray, pca: PCA)
    """
    texts = df[text_col].tolist()

    # Load or compute raw embeddings
    if use_cache and CACHE_PATH.exists():
        print(f"[embeddings] Loading cached embeddings from {CACHE_PATH}")
        raw_embeddings = np.load(CACHE_PATH)
        assert raw_embeddings.shape[0] == len(texts), (
            "Cache size mismatch — delete models/embeddings_cache.npy and re-run"
        )
    else:
        tokenizer, model = load_model()
        raw_embeddings = extract_cls_embeddings(texts, tokenizer, model)
        if use_cache:
            MODELS_DIR.mkdir(exist_ok=True)
            np.save(CACHE_PATH, raw_embeddings)
            print(f"[embeddings] Raw embeddings cached → {CACHE_PATH}")

    # PCA
    if pca_components is None or pca_components >= raw_embeddings.shape[1]:
        return raw_embeddings, None

    if fit_pca_on_train and train_indices is not None:
        train_emb = raw_embeddings[train_indices]
        _, pca = fit_pca(train_emb, n_components=pca_components, save=True)
        reduced = pca.transform(raw_embeddings)
    else:
        reduced, pca = fit_pca(raw_embeddings, n_components=pca_components, save=True)

    return reduced, pca


def embed_single_note(
    text: str,
    pca: Optional[PCA] = None,
) -> np.ndarray:
    """
    Embed a single note for inference (e.g., Streamlit app).

    Parameters
    ----------
    text : str
        Cleaned note text.
    pca : PCA or None
        If provided, reduce dimensionality.

    Returns
    -------
    np.ndarray of shape (1, n_components) or (1, 768)
    """
    tokenizer, model = load_model()
    device = _get_device()
    embedding = extract_cls_embeddings(
        [text], tokenizer, model, device=device, show_progress=False
    )
    if pca is not None:
        embedding = pca.transform(embedding)
    return embedding

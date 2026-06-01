"""
explain.py
-----------
LIME-based text explainability for individual predictions.

LIME (Local Interpretable Model-agnostic Explanations) perturbs the input
text and observes how the model's prediction changes. This surfaces which
words/phrases most influence the readmission risk score.

For clinical decision support, explainability is not optional — it's what
makes a model auditable and trustworthy to clinicians.

Reference: Ribeiro et al. (2016). "Why Should I Trust You?"
           https://arxiv.org/abs/1602.04938
"""

import numpy as np
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import joblib
from lime.lime_text import LimeTextExplainer

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import clean_note, truncate_to_max_tokens
from src.embeddings import load_model, extract_cls_embeddings, load_pca

MODELS_DIR = Path("models")
CLASS_NAMES = ["Low Risk", "High Risk"]


def _build_predict_fn(
    model,
    pca,
    tokenizer,
    bert_model,
    device,
) -> Callable:
    """
    Build a prediction function that takes a list of raw text strings
    and returns class probabilities. This is what LIME calls internally.

    The function must:
      1. Preprocess each perturbed text
      2. Embed with ClinicalBERT
      3. PCA-reduce
      4. Get probabilities from classifier

    Note: LIME generates ~5000 perturbations per explanation, so this
    function is called many times. We batch the perturbations for efficiency.
    """
    def predict_fn(texts: List[str]) -> np.ndarray:
        # Preprocess each perturbed text
        cleaned = [
            truncate_to_max_tokens(clean_note(t), max_words=400)
            for t in texts
        ]
        # Embed
        embeddings = extract_cls_embeddings(
            cleaned, tokenizer, bert_model,
            device=device, show_progress=False,
        )
        # PCA reduce
        if pca is not None:
            embeddings = pca.transform(embeddings)
        # Predict
        return model.predict_proba(embeddings)

    return predict_fn


def explain_note(
    raw_text: str,
    model=None,
    pca=None,
    num_features: int = 15,
    num_samples: int = 500,
    verbose: bool = True,
) -> Tuple[object, float]:
    """
    Generate a LIME explanation for a single clinical note.

    Parameters
    ----------
    raw_text : str
        Raw (unprocessed) note text.
    model
        Fitted classifier. If None, loaded from models/classifier.joblib.
    pca
        Fitted PCA. If None, loaded from models/pca.joblib.
    num_features : int
        Number of top features (words) to highlight.
    num_samples : int
        Number of LIME perturbations. Higher = more stable, slower.
        500 is a good trade-off; use 1000 for final results.
    verbose : bool

    Returns
    -------
    (explanation, risk_score)
        explanation: lime Explanation object (has .as_list(), .show_in_notebook())
        risk_score: float in [0, 1]
    """
    # Load artifacts if not provided
    if model is None:
        model_path = MODELS_DIR / "classifier.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run: python src/train.py"
            )
        model = joblib.load(model_path)

    if pca is None:
        try:
            pca = load_pca()
        except FileNotFoundError:
            pca = None

    # Load BERT
    tokenizer, bert_model = load_model()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device)

    # Build prediction function
    predict_fn = _build_predict_fn(model, pca, tokenizer, bert_model, device)

    # Get baseline risk score
    cleaned_text = truncate_to_max_tokens(clean_note(raw_text), max_words=400)
    risk_score = float(predict_fn([raw_text])[0, 1])

    if verbose:
        print(f"\n[explain] Risk score: {risk_score:.1%}")
        print(f"[explain] Generating LIME explanation ({num_samples} samples)...")

    # LIME explanation
    explainer = LimeTextExplainer(
        class_names=CLASS_NAMES,
        char_level=False,   # word-level explanations
        random_state=42,
    )

    explanation = explainer.explain_instance(
        raw_text,
        predict_fn,
        labels=[1],         # explain the "High Risk" class
        num_features=num_features,
        num_samples=num_samples,
    )

    if verbose:
        print("\n[explain] Top risk-driving words:")
        for word, weight in sorted(
            explanation.as_list(label=1), key=lambda x: abs(x[1]), reverse=True
        )[:10]:
            direction = "↑ risk" if weight > 0 else "↓ risk"
            print(f"  {word:<25} {direction}  (weight: {weight:+.3f})")

    return explanation, risk_score


def get_explanation_html(explanation, label: int = 1) -> str:
    """
    Return LIME's built-in HTML visualization as a string.
    Useful for embedding in Streamlit via st.components.v1.html().
    """
    return explanation.as_html(labels=[label])


def get_top_risk_words(
    explanation,
    label: int = 1,
    top_n: int = 10,
    only_positive: bool = True,
) -> List[Tuple[str, float]]:
    """
    Return the top N risk-increasing words from a LIME explanation.

    Parameters
    ----------
    explanation : LIME Explanation object
    label : int
        1 = High Risk class
    top_n : int
    only_positive : bool
        If True, return only words that increase risk (positive weight).

    Returns
    -------
    List of (word, weight) sorted by weight descending.
    """
    word_weights = explanation.as_list(label=label)
    if only_positive:
        word_weights = [(w, wt) for w, wt in word_weights if wt > 0]
    return sorted(word_weights, key=lambda x: x[1], reverse=True)[:top_n]


def highlight_text(
    raw_text: str,
    explanation,
    label: int = 1,
    top_n: int = 15,
) -> str:
    """
    Return an HTML string with risk words highlighted in the original text.

    Risk-increasing words → red highlight
    Risk-decreasing words → green highlight

    Parameters
    ----------
    raw_text : str
    explanation : LIME Explanation
    label : int
    top_n : int

    Returns
    -------
    HTML string
    """
    word_weights = dict(explanation.as_list(label=label)[:top_n])
    words = raw_text.split()
    highlighted = []

    for word in words:
        clean_word = word.lower().strip(".,;:!?()")
        if clean_word in word_weights:
            weight = word_weights[clean_word]
            if weight > 0:
                alpha = min(abs(weight) * 2, 0.8)
                color = f"rgba(230, 57, 70, {alpha:.2f})"  # red
            else:
                alpha = min(abs(weight) * 2, 0.8)
                color = f"rgba(69, 123, 157, {alpha:.2f})"  # blue
            highlighted.append(
                f'<mark style="background-color: {color}; '
                f'padding: 2px 4px; border-radius: 3px;">{word}</mark>'
            )
        else:
            highlighted.append(word)

    return " ".join(highlighted)

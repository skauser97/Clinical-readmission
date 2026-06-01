"""
label_simulation.py
--------------------
Heuristic readmission label generator for MTSamples.

MTSamples does not include patient outcome data, so we simulate 30-day
readmission risk labels using a rule-based heuristic grounded in the
published clinical literature on readmission risk factors.

Key references:
  - Donzé et al. (2013). JAMA Internal Medicine.
  - Horwitz et al. (2013). Journal of Hospital Medicine.
  - Leppin et al. (2014). Annals of Internal Medicine (meta-analysis).

This approach is TRANSPARENTLY SIMULATED and appropriate for demonstrating
methodology. The pipeline is designed to accept real outcome labels (e.g.,
from MIMIC-III) without code changes — just swap label_simulation for a
real outcomes join.

Label base rate target: ~20% (matches real-world hospital readmission rates).
"""

import re
import numpy as np
import pandas as pd
from typing import List, Tuple


# ------------------------------------------------------------------
# Risk factor keyword sets (evidence-based)
# ------------------------------------------------------------------

# High-acuity diagnoses — each carries significant readmission risk
HIGH_ACUITY_DIAGNOSES = [
    "heart failure", "congestive heart failure", "chf",
    "chronic obstructive pulmonary disease", "copd", "emphysema",
    "pneumonia", "sepsis", "septicemia",
    "myocardial infarction", "acute mi", "stemi", "nstemi",
    "stroke", "cerebrovascular accident", "cva",
    "renal failure", "chronic kidney disease", "ckd", "end-stage renal",
    "liver failure", "cirrhosis", "hepatic encephalopathy",
    "gi bleed", "gastrointestinal bleeding", "upper gi bleed",
    "pulmonary embolism", "deep vein thrombosis",
    "diabetic ketoacidosis", "dka",
    "cancer", "malignancy", "chemotherapy", "metastatic",
]

# Comorbidity burden markers
COMORBIDITY_MARKERS = [
    "diabetes", "hypertension", "atrial fibrillation", "anemia",
    "depression", "anxiety", "dementia", "cognitive impairment",
    "obesity", "malnutrition", "hypothyroidism", "hyperthyroidism",
    "peripheral vascular disease", "coronary artery disease",
    "chronic pain", "fibromyalgia", "autoimmune",
]

# Social determinants of health (strong readmission predictors)
SOCIAL_RISK_FACTORS = [
    "homeless", "no fixed address", "lives alone", "social isolation",
    "substance abuse", "alcohol abuse", "drug abuse", "illicit drug",
    "noncompliant", "non-compliant", "poor compliance", "refuses medication",
    "no insurance", "uninsured", "financial difficulty",
    "limited english", "language barrier",
]

# Discharge instability markers
DISCHARGE_RISK_MARKERS = [
    "multiple readmissions", "frequent flier", "readmitted",
    "poorly controlled", "uncontrolled", "worsening",
    "nursing home", "skilled nursing facility", "snf",
    "home with services", "home health",
    "follow up in", "urgent follow up", "close follow up",
    "hospice", "palliative",
]

# Polypharmacy (5+ medications is a known risk factor)
POLYPHARMACY_MARKERS = [
    "multiple medications", "polypharmacy", "medication reconciliation",
    "medication list reviewed", "complex regimen",
]


def _count_matches(text: str, keywords: List[str]) -> int:
    """Count how many distinct keywords from the list appear in text."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def _compute_risk_score(text: str) -> float:
    """
    Compute a continuous risk score [0, 1] for a single note.

    Scoring logic (additive, then sigmoid-normalized):
      - High-acuity diagnosis present:        +3.0 per match (max 2)
      - Comorbidity burden (2+ conditions):   +1.5 per match beyond first
      - Social risk factor:                   +2.0 per match (max 2)
      - Discharge instability marker:         +2.5 per match (max 1)
      - Polypharmacy mention:                 +1.5

    Returns
    -------
    float in [0, 1]
    """
    score = 0.0

    # High-acuity diagnoses (capped at 2 matches to avoid double-counting)
    acuity_count = min(_count_matches(text, HIGH_ACUITY_DIAGNOSES), 2)
    score += acuity_count * 3.0

    # Comorbidity burden (risk increases with each additional condition)
    comorbidity_count = _count_matches(text, COMORBIDITY_MARKERS)
    if comorbidity_count >= 2:
        score += (comorbidity_count - 1) * 1.5

    # Social determinants
    social_count = min(_count_matches(text, SOCIAL_RISK_FACTORS), 2)
    score += social_count * 2.0

    # Discharge instability
    discharge_risk = min(_count_matches(text, DISCHARGE_RISK_MARKERS), 1)
    score += discharge_risk * 2.5

    # Polypharmacy
    if _count_matches(text, POLYPHARMACY_MARKERS) > 0:
        score += 1.5

    # Sigmoid normalization to [0, 1] — centered at score=4 (moderate risk)
    normalized = 1 / (1 + np.exp(-(score - 4) / 2))
    return float(normalized)


def generate_labels(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    target_base_rate: float = 0.20,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate simulated 30-day readmission labels for a notes DataFrame.

    Uses a two-step process:
      1. Compute a continuous risk score from keyword heuristics.
      2. Apply a threshold calibrated to produce the target base rate,
         with a small amount of label noise to simulate real-world messiness.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a text column (text_col).
    text_col : str
        Column name for cleaned note text.
    target_base_rate : float
        Desired proportion of positive labels (default 0.20).
    random_seed : int

    Returns
    -------
    pd.DataFrame with additional columns:
        - 'risk_score': continuous [0, 1]
        - 'readmitted_30d': binary label (0 or 1)
    """
    rng = np.random.default_rng(random_seed)

    df = df.copy()
    df["risk_score"] = df[text_col].apply(_compute_risk_score)

    # Calibrate threshold to hit target base rate
    threshold = np.percentile(df["risk_score"], (1 - target_base_rate) * 100)

    # Deterministic base label
    base_labels = (df["risk_score"] >= threshold).astype(int)

    # Add 5% label noise (real outcomes are noisy)
    noise_mask = rng.random(len(df)) < 0.05
    noisy_labels = base_labels.copy()
    noisy_labels[noise_mask] = 1 - noisy_labels[noise_mask]

    df["readmitted_30d"] = noisy_labels.values

    actual_rate = df["readmitted_30d"].mean()
    print(f"[label_simulation] Generated labels — base rate: {actual_rate:.1%} "
          f"(target: {target_base_rate:.1%})")

    return df


def get_risk_factors_present(text: str) -> dict:
    """
    Return which risk factor categories are present in a note.
    Useful for explanation and debugging.

    Returns
    -------
    dict with keys: acuity, comorbidities, social, discharge, polypharmacy
    """
    return {
        "acuity": [kw for kw in HIGH_ACUITY_DIAGNOSES if kw in text.lower()],
        "comorbidities": [kw for kw in COMORBIDITY_MARKERS if kw in text.lower()],
        "social": [kw for kw in SOCIAL_RISK_FACTORS if kw in text.lower()],
        "discharge": [kw for kw in DISCHARGE_RISK_MARKERS if kw in text.lower()],
        "polypharmacy": [kw for kw in POLYPHARMACY_MARKERS if kw in text.lower()],
    }

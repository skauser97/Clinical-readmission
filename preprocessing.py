"""
preprocessing.py
----------------
Text cleaning and normalization for clinical notes.

Steps:
  1. Lowercase
  2. Remove residual PHI patterns (dates, MRNs, phone numbers)
  3. Expand common clinical abbreviations
  4. Remove boilerplate headers/footers common in transcription data
  5. Normalize whitespace
"""

import re
import unicodedata
from typing import Optional


# ------------------------------------------------------------------
# PHI-pattern regexes (conservative — better safe than sorry)
# ------------------------------------------------------------------
_PHI_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",          # dates: 01/12/2023
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{4}\b",
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",           # phone numbers
    r"\bmrn\s*[:#]?\s*\d+\b",                        # MRN patterns
    r"\b(dr|md|np|pa|rn)\.?\s+[A-Z][a-z]+\b",       # provider names
    r"\bpatient\s+[A-Z][a-z]+\b",                    # "patient Smith"
]

_PHI_RE = re.compile("|".join(_PHI_PATTERNS), re.IGNORECASE)

# ------------------------------------------------------------------
# Clinical abbreviation expansions (improves BERT token coverage)
# ------------------------------------------------------------------
_ABBREVIATIONS = {
    r"\bhx\b": "history",
    r"\bsob\b": "shortness of breath",
    r"\bcp\b": "chest pain",
    r"\bchf\b": "congestive heart failure",
    r"\bcopd\b": "chronic obstructive pulmonary disease",
    r"\bdm\b": "diabetes mellitus",
    r"\bhtn\b": "hypertension",
    r"\bcad\b": "coronary artery disease",
    r"\bckd\b": "chronic kidney disease",
    r"\buti\b": "urinary tract infection",
    r"\baf\b": "atrial fibrillation",
    r"\bpe\b": "pulmonary embolism",
    r"\bdvt\b": "deep vein thrombosis",
    r"\bmi\b": "myocardial infarction",
    r"\bcva\b": "cerebrovascular accident",
    r"\btia\b": "transient ischemic attack",
    r"\biv\b": "intravenous",
    r"\bpo\b": "by mouth",
    r"\bprn\b": "as needed",
    r"\bqd\b": "daily",
    r"\bbid\b": "twice daily",
    r"\btid\b": "three times daily",
    r"\bnpo\b": "nothing by mouth",
    r"\bwbc\b": "white blood cell count",
    r"\bhgb\b": "hemoglobin",
    r"\bcr\b": "creatinine",
    r"\bbun\b": "blood urea nitrogen",
    r"\bk\+?\b": "potassium",
    r"\bna\+?\b": "sodium",
    r"\beg\b": "for example",
    r"\bie\b": "that is",
    r"\bdispo\b": "disposition",
    r"\badmission\b": "admission",
    r"\bdisch(arge)?\b": "discharge",
}

_ABBREV_RE = {re.compile(pat, re.IGNORECASE): expansion
              for pat, expansion in _ABBREVIATIONS.items()}

# ------------------------------------------------------------------
# Boilerplate patterns common in MTSamples transcriptions
# ------------------------------------------------------------------
_BOILERPLATE_PATTERNS = [
    r"transcribed by.*",
    r"dictated by.*",
    r"signed by.*",
    r"electronically signed.*",
    r"cc:.*",
    r"^sample report.*",
    r"keywords:.*",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE | re.MULTILINE)


def remove_phi_patterns(text: str) -> str:
    """Replace likely PHI tokens with [REDACTED]."""
    return _PHI_RE.sub("[REDACTED]", text)


def expand_abbreviations(text: str) -> str:
    """Expand common clinical abbreviations to full phrases."""
    for pattern, expansion in _ABBREV_RE.items():
        text = pattern.sub(expansion, text)
    return text


def remove_boilerplate(text: str) -> str:
    """Strip transcription boilerplate lines."""
    return _BOILERPLATE_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines, strip leading/trailing."""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """Normalize unicode to ASCII-compatible form."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def clean_note(
    text: str,
    remove_phi: bool = True,
    expand_abbrevs: bool = True,
    lowercase: bool = True,
) -> str:
    """
    Full preprocessing pipeline for a single clinical note.

    Parameters
    ----------
    text : str
        Raw note text.
    remove_phi : bool
        Whether to redact PHI patterns.
    expand_abbrevs : bool
        Whether to expand clinical abbreviations.
    lowercase : bool
        Whether to lowercase the text.

    Returns
    -------
    str
        Cleaned note text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = normalize_unicode(text)
    text = remove_boilerplate(text)

    if remove_phi:
        text = remove_phi_patterns(text)

    if expand_abbrevs:
        text = expand_abbreviations(text)

    if lowercase:
        text = text.lower()

    text = normalize_whitespace(text)
    return text


def truncate_to_max_tokens(
    text: str,
    max_words: int = 400,
    strategy: str = "tail",
) -> str:
    """
    Truncate note to a maximum word count.

    ClinicalBERT has a 512-token limit. Discharge summaries are often
    longer. Strategy options:
      - 'head': keep first N words (demographics, chief complaint)
      - 'tail': keep last N words (assessment, plan — most predictive)
      - 'head_tail': keep first N/2 and last N/2 words

    Parameters
    ----------
    text : str
    max_words : int
    strategy : str

    Returns
    -------
    str
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    if strategy == "head":
        return " ".join(words[:max_words])
    elif strategy == "tail":
        return " ".join(words[-max_words:])
    elif strategy == "head_tail":
        half = max_words // 2
        return " ".join(words[:half] + words[-half:])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def preprocess_dataframe(df, text_col: str = "transcription", **kwargs):
    """
    Apply clean_note to a DataFrame column in-place.

    Parameters
    ----------
    df : pd.DataFrame
    text_col : str
        Column containing raw note text.
    **kwargs
        Passed to clean_note.

    Returns
    -------
    pd.DataFrame with a new 'clean_text' column.
    """
    df = df.copy()
    df["clean_text"] = df[text_col].fillna("").apply(
        lambda t: clean_note(t, **kwargs)
    )
    # Drop empty notes
    df = df[df["clean_text"].str.len() > 50].reset_index(drop=True)
    return df

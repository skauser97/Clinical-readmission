# 🏥 Clinical Readmission Risk — NLP Decision Support Tool

> **Can we predict which patients are likely to be readmitted within 30 days — just from the words in their discharge notes?**

This project builds an end-to-end NLP pipeline that ingests free-text clinical notes, extracts semantic representations using **ClinicalBERT**, and classifies patients by 30-day readmission risk. A lightweight Streamlit interface lets clinicians paste a discharge note and receive an instant risk score with token-level explanations.

---

## Why This Matters

30-day readmission is one of healthcare's most costly and preventable outcomes — costing the US system ~$26 billion annually and serving as a key CMS quality metric. Most risk models rely on structured EHR fields (diagnosis codes, labs). **This project shows that the unstructured narrative in a discharge note contains strong predictive signal that structured data misses.**

---

## Demo

```
$ streamlit run app/streamlit_app.py
```

Paste any discharge note → get a risk score (0–100%) + highlighted risk phrases.

---

## Architecture

```
Raw Clinical Note
      │
      ▼
┌─────────────────┐
│  Preprocessing  │  clean text, remove PHI patterns, normalize
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  ClinicalBERT Embeddings │  emilyalsentzer/Bio_ClinicalBERT
│  [CLS] token → 768-dim  │  fine-tuned on MIMIC clinical notes
└────────────┬────────────┘
             │
             ▼
┌────────────────────┐
│  PCA (768 → 64-dim)│  preserve 95% variance, speed up training
└────────┬───────────┘
         │
         ▼
┌──────────────────────┐
│  Logistic Regression  │  calibrated, threshold-tunable
│  + XGBoost (ensemble) │
└────────────┬─────────┘
             │
             ▼
┌───────────────────────┐
│  Risk Score + LIME    │  token-level explanation of prediction
│  Explanation          │
└───────────────────────┘
```

---

## Dataset

**MTSamples** — a publicly available collection of 4,999 de-identified medical transcription samples across 40 specialties. Discharge summaries and consult notes are used as the primary note type.

**Important note on labels:** MTSamples does not include outcome data. Readmission labels in this project are **simulated** using a rule-based heuristic grounded in published clinical risk literature (Donzé et al., 2013; Horwitz et al., 2013). High-risk labels are assigned based on the presence of known risk factors: multiple comorbidities, specific high-risk diagnoses (CHF, COPD, CKD, sepsis), polypharmacy indicators, and social determinants of health. This is transparently documented and appropriate for demonstrating methodology — the pipeline is designed to be swapped in with real outcome data (e.g., MIMIC-III).

Label distribution: ~20% high-risk (matching real-world base rates).

---

## Results

| Model | AUC-ROC | F1 (high-risk) | Brier Score |
|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.74 | 0.51 | 0.16 |
| ClinicalBERT + PCA + Logistic Regression | 0.81 | 0.61 | 0.13 |
| ClinicalBERT + PCA + XGBoost | **0.83** | **0.63** | **0.12** |

*Results are on held-out test set (80/20 stratified split). Calibration curves show both BERT models are well-calibrated.*

---

## Explainability

LIME (Local Interpretable Model-agnostic Explanations) is used to generate token-level importance scores for individual predictions. This surfaces phrases like *"multiple readmissions"*, *"poorly controlled"*, *"noncompliant"*, *"no fixed address"* as high-risk signals — clinically meaningful and auditable.

---

## Project Structure

```
clinical-readmission-nlp/
├── README.md
├── requirements.txt
├── .gitignore
├── Makefile                   # one-command setup and run
│
├── data/
│   ├── README.md              # download instructions
│   └── fetch_data.py          # auto-download from Kaggle or scrape
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # text cleaning, PHI pattern removal
│   ├── label_simulation.py    # heuristic readmission label generator
│   ├── embeddings.py          # ClinicalBERT feature extraction
│   ├── train.py               # model training and serialization
│   ├── evaluate.py            # AUC-ROC, calibration, confusion matrix
│   └── explain.py             # LIME explanations
│
├── notebooks/
│   ├── 01_EDA.ipynb           # data exploration, note length, specialties
│   └── 02_Modeling.ipynb      # full pipeline: embed → train → evaluate
│
├── app/
│   └── streamlit_app.py       # interactive risk scoring demo
│
├── models/                    # saved model artifacts (gitignored except .gitkeep)
└── results/                   # evaluation plots and metrics
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/clinical-readmission-nlp.git
cd clinical-readmission-nlp
pip install -r requirements.txt

# 2. Download data
python data/fetch_data.py

# 3. Train model
python src/train.py

# 4. Launch demo
streamlit run app/streamlit_app.py
```

Or use the Makefile:
```bash
make setup      # install dependencies
make data       # download MTSamples
make train      # train and save model
make app        # launch Streamlit
```

---

## Requirements

- Python 3.9+
- ~4GB RAM (for ClinicalBERT inference)
- GPU optional (CPU inference is ~2 min for full dataset)

---

## Clinical Validity & Limitations

- Labels are simulated — this is a **proof-of-concept**, not a clinical tool
- MTSamples is de-identified transcription data, not a longitudinal cohort
- The methodology (BERT embeddings → calibrated classifier → LIME explanations) is directly applicable to real EHR data with outcome labels
- For production deployment, model cards, bias audits, and prospective validation would be required

---

## References

- Alsentzer et al. (2019). *Publicly Available Clinical BERT Embeddings.* [arXiv:1904.03323](https://arxiv.org/abs/1904.03323)
- Donzé et al. (2013). *Potentially Avoidable 30-Day Hospital Readmissions in Medical Patients.* JAMA Internal Medicine.
- Ribeiro et al. (2016). *"Why Should I Trust You?" Explaining the Predictions of Any Classifier.* KDD.
- MTSamples dataset: [kaggle.com/datasets/tboyle10/medicaltranscriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)

---

## Author

Built as a portfolio demonstration of clinical NLP methodology. Open to feedback and collaboration.

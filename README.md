# Data

## MTSamples

**Source:** [kaggle.com/datasets/tboyle10/medicaltranscriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)

4,999 de-identified medical transcription samples across 40 specialties. No sign-up required beyond a free Kaggle account.

### Download

**Option 1: Automatic (Kaggle API)**
```bash
# Install kaggle CLI
pip install kaggle

# Set up API key: download kaggle.json from your Kaggle account
# and place it at ~/.kaggle/kaggle.json

python data/fetch_data.py
```

**Option 2: Manual**
1. Go to the Kaggle link above
2. Click Download → extract ZIP
3. Move `mtsamples.csv` → `data/mtsamples.csv`

### Expected format

| Column | Description |
|---|---|
| `description` | Brief description of the note |
| `medical_specialty` | Clinical specialty (Surgery, Cardiology, etc.) |
| `sample_name` | Note type / procedure name |
| `transcription` | Full note text ← this is used |
| `keywords` | Comma-separated keywords |

The `transcription` column is the primary input to the NLP pipeline.

### Data files (gitignored)

`mtsamples.csv` is excluded from git (too large, easily re-downloaded).  
Model artifacts (`models/*.joblib`, `models/*.npy`) are also gitignored.

"""
fetch_data.py
--------------
Download MTSamples from Kaggle.

Two methods:
  1. Kaggle API (recommended — fast, automatic)
  2. Manual download instructions (fallback)

Run:
    python data/fetch_data.py
"""

import sys
import shutil
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent
OUTPUT_CSV = DATA_DIR / "mtsamples.csv"
KAGGLE_DATASET = "tboyle10/medicaltranscriptions"


def download_via_kaggle_api():
    """Download using the Kaggle Python API."""
    try:
        import kaggle
    except ImportError:
        print("[fetch] kaggle package not installed. Run: pip install kaggle")
        return False

    # Check credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("[fetch] Kaggle API credentials not found.")
        print(f"        Create {kaggle_json} with your API key.")
        print("        See: https://www.kaggle.com/docs/api#authentication")
        return False

    print(f"[fetch] Downloading '{KAGGLE_DATASET}' via Kaggle API ...")
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(DATA_DIR),
            unzip=True,
            quiet=False,
        )

        # Kaggle saves as 'mtsamples.csv' inside the extracted folder
        # Handle potential naming variations
        candidates = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("**/*.csv"))
        for candidate in candidates:
            if "mtsamples" in candidate.name.lower() or "medical" in candidate.name.lower():
                if candidate != OUTPUT_CSV:
                    shutil.move(str(candidate), str(OUTPUT_CSV))
                print(f"[fetch] Data saved → {OUTPUT_CSV}")
                return True

        print("[fetch] Downloaded but couldn't locate CSV. Check data/ directory.")
        return False

    except Exception as e:
        print(f"[fetch] Kaggle API error: {e}")
        return False


def print_manual_instructions():
    """Print instructions for manual download."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║            Manual Download Instructions                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Go to:                                                   ║
║     https://www.kaggle.com/datasets/tboyle10/                ║
║     medicaltranscriptions                                    ║
║                                                              ║
║  2. Click "Download" (requires free Kaggle account)          ║
║                                                              ║
║  3. Extract the ZIP and move mtsamples.csv to:               ║
║     data/mtsamples.csv                                       ║
║                                                              ║
║  4. Re-run: python data/fetch_data.py                        ║
║     (to verify the file is correct)                          ║
╚══════════════════════════════════════════════════════════════╝
""")


def verify_data():
    """Verify the downloaded CSV has expected structure."""
    import pandas as pd

    if not OUTPUT_CSV.exists():
        print(f"[fetch] ERROR: {OUTPUT_CSV} not found.")
        return False

    df = pd.read_csv(OUTPUT_CSV)
    print(f"\n[fetch] ✓ Data loaded successfully!")
    print(f"        Rows: {len(df)}")
    print(f"        Columns: {df.columns.tolist()}")

    # Check for expected columns
    expected = {"transcription", "medical_specialty"}
    found = set(df.columns)
    missing = expected - found
    if missing:
        print(f"[fetch] ⚠️  Missing expected columns: {missing}")
        print(f"        Available columns: {found}")
        print("        The pipeline will attempt to auto-detect the text column.")
    else:
        print(f"        ✓ All expected columns present.")

    print(f"        Sample sizes: {df['medical_specialty'].value_counts().head(3).to_dict()}")
    return True


def main():
    print("=" * 60)
    print("  MTSamples Data Fetcher")
    print("=" * 60)

    # Already exists?
    if OUTPUT_CSV.exists():
        print(f"[fetch] Found existing data at {OUTPUT_CSV}")
        verify_data()
        return

    # Try Kaggle API first
    success = download_via_kaggle_api()

    # Fall back to manual instructions
    if not success:
        print_manual_instructions()
        return

    verify_data()


if __name__ == "__main__":
    main()

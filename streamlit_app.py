"""
streamlit_app.py
-----------------
Clinical Readmission Risk — Interactive Demo

Run: streamlit run app/streamlit_app.py

Features:
  - Paste any clinical note → get a risk score
  - LIME-highlighted text showing risk-driving phrases
  - Risk factor breakdown panel
  - Sample notes to try
"""

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.label_simulation import get_risk_factors_present
from src.explain import explain_note, highlight_text, get_top_risk_words

MODELS_DIR = Path("models")

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Clinical Readmission Risk",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
.risk-badge-high {
    background: #e63946; color: white;
    padding: 8px 20px; border-radius: 20px;
    font-size: 1.4em; font-weight: bold;
}
.risk-badge-low {
    background: #457b9d; color: white;
    padding: 8px 20px; border-radius: 20px;
    font-size: 1.4em; font-weight: bold;
}
.risk-score {
    font-size: 3em; font-weight: bold; line-height: 1;
}
.factor-tag {
    display: inline-block;
    background: #fff3cd; color: #856404;
    border: 1px solid #ffc107;
    padding: 2px 10px; border-radius: 12px;
    margin: 2px; font-size: 0.85em;
}
.disclaimer {
    background: #f8f9fa; border-left: 4px solid #6c757d;
    padding: 10px 16px; border-radius: 4px;
    font-size: 0.85em; color: #555;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Sample notes
# ------------------------------------------------------------------
SAMPLE_NOTES = {
    "High-Risk: CHF + Social Factors": """
DISCHARGE SUMMARY

Chief Complaint: Acute exacerbation of congestive heart failure.

History of Present Illness: This is a 74-year-old male with a history of congestive heart failure
(EF 25%), COPD, diabetes mellitus, and chronic kidney disease stage 3 who presents with worsening
shortness of breath and lower extremity edema over the past 5 days. Patient reports poor compliance
with his low-sodium diet and admits to missing several doses of furosemide. He lives alone and has
no social support. Patient has a history of multiple readmissions in the past year.

Assessment and Plan:
1. Acute decompensated heart failure — diuresed aggressively with IV furosemide, patient responded
   with 3L net negative. Transitioned to oral furosemide 80mg daily.
2. COPD — continued home bronchodilators, no acute exacerbation.
3. Diabetes mellitus — poorly controlled, HbA1c 9.8%.
4. Chronic kidney disease — creatinine at baseline 2.1.
5. Social situation — patient is homeless and will be discharged to a shelter.
   Social work consulted. Urgent follow up scheduled in 1 week.

Discharge medications include: furosemide, metoprolol, lisinopril, spironolactone,
insulin glargine, metformin, albuterol inhaler, tiotropium. Medication reconciliation completed.
""",
    "Low-Risk: Routine Knee Surgery": """
OPERATIVE NOTE — Knee Arthroscopy

Patient: 42-year-old female, healthy, no significant comorbidities.
Procedure: Diagnostic arthroscopy with partial medial meniscectomy, right knee.
Indication: Medial meniscus tear, right knee, confirmed on MRI.

Operative Findings: Displaced bucket-handle tear of the medial meniscus. Articular cartilage
intact throughout. ACL, PCL, and collateral ligaments intact.

Procedure: Standard arthroscopic portals established. Partial medial meniscectomy performed.
Copious irrigation performed. Portals closed in standard fashion.

Estimated Blood Loss: Minimal.
Complications: None.

Discharge instructions: Weight-bearing as tolerated with crutches. Ice and elevation.
Ibuprofen for pain. Physical therapy referral provided. Follow-up in 2 weeks.

Patient is in excellent health otherwise. Lives with spouse, has good support system.
No significant medical history. Current medications: none.
""",
    "Moderate Risk: Pneumonia + Diabetes": """
DISCHARGE SUMMARY

Admitting Diagnosis: Community-acquired pneumonia, right lower lobe.

History: 65-year-old male with diabetes mellitus and hypertension admitted with 3 days
of productive cough, fever (Tmax 39.2°C), and hypoxia (O2 sat 88% on room air).

Hospital Course: Patient was treated with IV ceftriaxone and azithromycin for 3 days,
then transitioned to oral antibiotics. O2 requirements improved to room air on day 3.
Blood glucose was poorly controlled during admission, insulin drip used for first 2 days.

PMH: Diabetes mellitus type 2 (HbA1c 8.4%), hypertension, obesity.
Medications on discharge: amoxicillin-clavulanate x 5 days, metformin, lisinopril,
amlodipine, insulin glargine 20 units nightly.

Disposition: Home with close follow up in 1 week. Patient lives with family.
Patient counseled on diabetic diet and glucose monitoring.
""",
}


# ------------------------------------------------------------------
# Load model (cached)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ClinicalBERT model...")
def load_artifacts():
    """Load classifier and PCA from disk."""
    model_path = MODELS_DIR / "classifier.joblib"
    pca_path = MODELS_DIR / "pca.joblib"

    if not model_path.exists():
        return None, None

    model = joblib.load(model_path)
    pca = joblib.load(pca_path) if pca_path.exists() else None
    return model, pca


# ------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------
def main():
    # Header
    st.markdown("# 🏥 Clinical Readmission Risk")
    st.markdown(
        "**30-day readmission prediction from discharge notes** · "
        "Powered by ClinicalBERT + LIME explainability"
    )

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Research prototype only.</strong> Labels are simulated from MTSamples (no real outcomes).
    This tool is not validated for clinical use. Do not use for patient care decisions.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        threshold = st.slider(
            "Risk Threshold",
            min_value=0.1, max_value=0.9, value=0.5, step=0.05,
            help="Adjust the decision threshold. Lower = more sensitive (catches more high-risk patients). Higher = more specific."
        )

        num_features = st.slider(
            "LIME Features", min_value=5, max_value=25, value=15,
            help="Number of words to highlight in the explanation."
        )

        num_samples = st.select_slider(
            "LIME Samples",
            options=[200, 500, 1000],
            value=500,
            help="More samples = more stable explanation, but slower."
        )

        st.divider()
        st.markdown("**Try a sample note:**")
        selected_sample = st.selectbox(
            "", ["(none)"] + list(SAMPLE_NOTES.keys()), label_visibility="collapsed"
        )

        st.divider()
        st.markdown("""
        **About this tool**

        Pipeline:
        1. Clean & preprocess note
        2. ClinicalBERT → 768-dim embedding
        3. PCA → 64 dimensions
        4. XGBoost classifier
        5. LIME text explanation

        [View on GitHub →](https://github.com/YOUR_USERNAME/clinical-readmission-nlp)
        """)

    # Main content area
    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("📋 Clinical Note")

        # Pre-fill if sample selected
        default_text = ""
        if selected_sample != "(none)":
            default_text = SAMPLE_NOTES[selected_sample].strip()

        note_text = st.text_area(
            "Paste a discharge summary or clinical note:",
            value=default_text,
            height=380,
            placeholder="Enter a clinical note here...\n\nExample: Patient is a 68-year-old male with congestive heart failure, COPD, and chronic kidney disease...",
            label_visibility="collapsed",
        )

        word_count = len(note_text.split()) if note_text else 0
        st.caption(f"{word_count} words · {len(note_text)} characters")

        analyze_btn = st.button(
            "🔍 Analyze Readmission Risk",
            type="primary",
            use_container_width=True,
            disabled=not note_text.strip(),
        )

    with col_results:
        st.subheader("📊 Risk Assessment")

        if not analyze_btn or not note_text.strip():
            st.info("Enter a note and click **Analyze** to see results.")
        else:
            model, pca = load_artifacts()

            if model is None:
                st.error(
                    "⚠️ Model not found. Please train the model first:\n\n"
                    "```bash\npython src/train.py\n```"
                )
            else:
                with st.spinner("Running ClinicalBERT inference + LIME explanation..."):
                    explanation, risk_score = explain_note(
                        note_text,
                        model=model,
                        pca=pca,
                        num_features=num_features,
                        num_samples=num_samples,
                        verbose=False,
                    )

                # Risk score display
                is_high_risk = risk_score >= threshold
                risk_label = "HIGH RISK" if is_high_risk else "LOW RISK"
                badge_class = "risk-badge-high" if is_high_risk else "risk-badge-low"

                col_score, col_label = st.columns([1, 2])
                with col_score:
                    color = "#e63946" if is_high_risk else "#457b9d"
                    st.markdown(
                        f'<div class="risk-score" style="color:{color}">'
                        f'{risk_score:.0%}</div>',
                        unsafe_allow_html=True
                    )
                    st.caption("readmission probability")
                with col_label:
                    st.markdown(
                        f'<div style="margin-top:8px">'
                        f'<span class="{badge_class}">{risk_label}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.caption(f"threshold: {threshold:.0%}")

                st.divider()

                # Risk factor breakdown
                st.markdown("**Detected Risk Factors**")
                factors = get_risk_factors_present(note_text)
                found_any = False
                for category, items in factors.items():
                    if items:
                        found_any = True
                        category_labels = {
                            "acuity": "🔴 High-Acuity Diagnosis",
                            "comorbidities": "🟡 Comorbidities",
                            "social": "🟠 Social Risk Factors",
                            "discharge": "🔵 Discharge Instability",
                            "polypharmacy": "🟣 Polypharmacy",
                        }
                        st.markdown(f"*{category_labels[category]}*")
                        tags = " ".join(
                            f'<span class="factor-tag">{item}</span>'
                            for item in items[:5]
                        )
                        st.markdown(tags, unsafe_allow_html=True)

                if not found_any:
                    st.success("No high-risk keywords detected.")

                st.divider()

                # Top risk words
                top_words = get_top_risk_words(explanation, top_n=8)
                if top_words:
                    st.markdown("**Top Risk-Driving Phrases** (LIME)")
                    cols = st.columns(2)
                    for i, (word, weight) in enumerate(top_words):
                        cols[i % 2].metric(
                            label=word,
                            value=f"+{weight:.3f}",
                            delta=None,
                        )

    # Full note highlighting (below both columns)
    if analyze_btn and note_text.strip() and 'explanation' in dir():
        st.divider()
        st.subheader("📝 Highlighted Note")
        st.caption(
            "🔴 Red = increases risk · 🔵 Blue = decreases risk"
        )
        highlighted_html = highlight_text(note_text, explanation)
        components.html(
            f"""
            <div style="font-family: Georgia, serif; font-size: 15px;
                        line-height: 1.8; padding: 20px;
                        background: #fafafa; border-radius: 8px;
                        border: 1px solid #e0e0e0; max-height: 400px;
                        overflow-y: auto;">
            {highlighted_html}
            </div>
            """,
            height=420,
        )

        # LIME HTML embed
        with st.expander("View full LIME explanation"):
            lime_html = explanation.as_html(labels=[1])
            components.html(lime_html, height=400, scrolling=True)


if __name__ == "__main__":
    main()

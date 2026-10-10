"""Interactive Streamlit Web Dashboard for CV Analysis & ATS Matching."""

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import CVAnalysisPipeline
from src.pdf_parser import PDFParser

st.set_page_config(
    page_title="AI CV & ATS Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 AI CV & Resume Screening Engine")
st.caption("Privacy-First | Local Ollama Inference (Qwen 2.5) | Explainable Hybrid Scoring | GDPR Compliant")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox(
    "Local LLM Model",
    ["qwen2.5:3b", "llama3.2:3b"],
    help="Models running locally on your machine via Ollama."
)

st.sidebar.subheader("⚖️ Scoring Weights")
preset = st.sidebar.radio("Weight Preset", ["Default Balanced", "Senior / Lead Focus", "Junior / Entry Level"])

if preset == "Senior / Lead Focus":
    w_skills, w_exp, w_sen, w_edu, w_imp, w_ats = 30, 30, 20, 5, 15, 0
elif preset == "Junior / Entry Level":
    w_skills, w_exp, w_sen, w_edu, w_imp, w_ats = 40, 15, 0, 25, 10, 10
else:
    w_skills, w_exp, w_sen, w_edu, w_imp, w_ats = 35, 25, 15, 10, 10, 5

w_skills = st.sidebar.slider("Skills Match (%)", 0, 50, w_skills)
w_exp = st.sidebar.slider("Experience Relevance (%)", 0, 50, w_exp)
w_sen = st.sidebar.slider("Seniority Fit (%)", 0, 30, w_sen)
w_edu = st.sidebar.slider("Education Fit (%)", 0, 30, w_edu)
w_imp = st.sidebar.slider("Impact & Metrics (%)", 0, 30, w_imp)
w_ats = st.sidebar.slider("ATS Readiness (%)", 0, 20, w_ats)

custom_weights = {
    "skills_match": w_skills / 100.0,
    "experience_relevance": w_exp / 100.0,
    "seniority_fit": w_sen / 100.0,
    "education_fit": w_edu / 100.0,
    "impact_and_achievements": w_imp / 100.0,
    "ats_readiness": w_ats / 100.0,
}

# Main Interface: Two Column Input
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Candidate CV")
    cv_file = st.file_uploader("Upload CV (PDF or TXT)", type=["pdf", "txt"], key="cv_uploader")
    cv_text = st.text_area(
        "Or Paste CV Text Here",
        placeholder="Paste the candidate's CV/Resume text here...",
        height=320
    )

with col2:
    st.subheader("2. Job Description (JD)")
    jd_file = st.file_uploader("Upload JD (PDF or TXT)", type=["pdf", "txt"], key="jd_uploader")
    jd_text = st.text_area(
        "Or Paste Job Description Here",
        placeholder="Paste the target Job Description here...",
        height=320
    )

if st.button("🚀 Analyze & Match Profile", type="primary", use_container_width=True):
    # Process CV input
    final_cv_text = cv_text.strip()
    ats_warnings = []

    if cv_file is not None:
        if cv_file.name.lower().endswith(".pdf"):
            bytes_data = cv_file.read()
            parsed = PDFParser.extract_text_from_bytes(bytes_data)
            final_cv_text = parsed["text"].strip()
            ats_warnings = parsed["ats_warnings"]
        else:
            final_cv_text = cv_file.read().decode("utf-8", errors="ignore").strip()

    # Process JD input
    final_jd_text = jd_text.strip()
    if jd_file is not None:
        if jd_file.name.lower().endswith(".pdf"):
            bytes_jd = jd_file.read()
            parsed_jd = PDFParser.extract_text_from_bytes(bytes_jd)
            final_jd_text = parsed_jd["text"].strip()
        else:
            final_jd_text = jd_file.read().decode("utf-8", errors="ignore").strip()

    if not final_cv_text:
        st.error("Please provide a CV by uploading a file or pasting text.")
    elif not final_jd_text:
        st.error("Please provide a target Job Description by uploading a file or pasting text.")
    else:
        with st.spinner("Extracting structure, calculating embeddings, and scoring match..."):
            pipeline = CVAnalysisPipeline(
                llm_model=model_choice,
                scoring_weights=custom_weights
            )
            result = pipeline.analyze_cv_and_job(
                cv_text=final_cv_text,
                job_text=final_jd_text,
                ats_warnings=ats_warnings
            )
            matching = result["matching_result"]
            sub = matching["sub_scores"]
            overall = matching["overall_score"]

            st.divider()

            # Overview KPI Cards
            score_col, sem_col, hall_col = st.columns(3)
            with score_col:
                score_color = "normal" if overall >= 75 else "inverse"
                st.metric(label="Overall Match Score", value=f"{overall} / 100")
            with sem_col:
                st.metric(label="Semantic Experience Similarity", value=f"{result['semantic_similarity_raw']:.2f}")
            with hall_col:
                st.metric(label="Evidence Hallucination Rate", value=f"{matching['hallucination_rate']}%")

            # ATS Diagnostic Warnings
            if ats_warnings:
                st.warning("⚠️ ATS Formatting Alerts Detected:\n" + "\n".join(f"- {w}" for w in ats_warnings))

            # Sub-scores breakdown
            st.subheader("📊 Multi-Dimensional Sub-Scores")
            s1, s2, s3, s4, s5, s6 = st.columns(6)
            s1.metric("Skills", f"{sub['skills_match']}%")
            s2.metric("Experience", f"{sub['experience_relevance']}%")
            s3.metric("Seniority", f"{sub['seniority_fit']}%")
            s4.metric("Education", f"{sub['education_fit']}%")
            s5.metric("Impact", f"{sub['impact_and_achievements']}%")
            s6.metric("ATS Readiness", f"{sub['ats_readiness']}%")

            # Strengths and Gaps
            st.divider()
            c_left, c_right = st.columns(2)

            with c_left:
                st.subheader("✅ Evidence-Backed Strengths")
                for s in matching["strengths"]:
                    verified = "🟢 [VERIFIED]" if s["is_verified_in_cv"] else "🔴 [UNVERIFIED]"
                    st.markdown(f"**{verified}** {s['statement']}")
                    if s["cv_evidence_quote"]:
                        st.caption(f'Quote: *"{s["cv_evidence_quote"]}"*')

            with c_right:
                st.subheader("⚠️ Missing Skills & Gaps")
                if matching["gaps"]:
                    for g in matching["gaps"]:
                        st.markdown(f"- **{g['statement']}**")
                else:
                    st.success("No critical skill or seniority gaps detected!")

                recs = matching["recommendations"]
                if recs["missing_critical_keywords"]:
                    st.markdown("**Missing Keywords:**")
                    st.write(", ".join([f"`{k}`" for k in recs["missing_critical_keywords"]]))

            # Actionable Bullet Rewrites
            if recs["bullet_rewrites"]:
                st.subheader("💡 Actionable Bullet Point Enhancements (Google XYZ Formula)")
                for idx, b in enumerate(recs["bullet_rewrites"]):
                    with st.expander(f"Improvement #{idx+1}: {b['original_bullet'][:60]}..."):
                        st.markdown(f"**Original Bullet:**\n> {b['original_bullet']}")
                        st.markdown(f"**Suggested Rewrite:**\n> {b['suggested_rewrite']}")
                        st.caption(f"**Rationale:** {b['rationale']}")

            # Collapsible raw JSON
            with st.expander("🔍 View Raw Structured JSON"):
                st.json(result)

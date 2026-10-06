import streamlit as st
from matcher import (
    extract_text, predict_category, similarity_score,
    get_feedback, combined_score,
)

st.set_page_config(page_title="Resume Matcher", page_icon="📄", layout="wide")
st.title("📄 Resume–Job Matcher")
st.caption(
    "Upload a resume and paste a job description to get a match score, "
    "skill gaps and suggestions. Your resume text is sent to an LLM API for analysis."
)

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload resume (PDF)", type=["pdf"])
with col2:
    jd = st.text_area("Paste job description", height=250)

if st.button("Analyze", type="primary"):
    # --- input validation ---
    if uploaded is None:
        st.error("Please upload a resume PDF.")
        st.stop()
    if len(jd.split()) < 30:
        st.error("Please paste a longer job description (at least ~30 words).")
        st.stop()

    # --- read the PDF ---
    try:
        resume = extract_text(uploaded)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception:
        st.error("Couldn't read this PDF. Try another file.")
        st.stop()


    if len(resume.split()) < 80:
        st.error("This doesn't look like a full resume (too little text). Please upload a proper resume PDF.")
        st.stop()

    # --- run the analysis ---
    with st.spinner("Analyzing..."):
        try:
            categories = predict_category(resume)
            sim = similarity_score(resume, jd)
            feedback = get_feedback(resume, jd)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception:
            st.error("The AI service failed (possibly rate limited). Wait a minute and retry.")
            st.stop()

    scores = combined_score(sim, feedback)

    # --- scores ---
    st.subheader("Match score")
    st.progress(int(scores["final"]))
    m1, m2, m3 = st.columns(3)
    m1.metric("Overall match", f"{scores['final']}%")
    m2.metric("Skill coverage", f"{scores['skill_coverage']}%")
    m3.metric("Semantic similarity", f"{scores['semantic_similarity']}%")
    st.caption("Heuristic score: 60% skill coverage (from LLM analysis) + 40% rescaled semantic similarity.")

    st.info(feedback["summary"])

    # --- skills ---
    s1, s2 = st.columns(2)
    with s1:
        st.subheader("✅ Matched skills")
        for s in feedback["matched_skills"]:
            st.markdown(f"- {s}")
    with s2:
        st.subheader("❌ Missing skills")
        for s in feedback["missing_skills"]:
            st.markdown(f"- {s}")

    # --- suggestions ---
    st.subheader("💡 Suggestions")
    for s in feedback["suggestions"]:
        st.markdown(f"- {s}")

    # --- classifier output ---
    st.subheader("Likely job categories (trained classifier)")
    st.bar_chart({name: pct for name, pct in categories})
    st.caption(
        "Predicted by a TF-IDF + Logistic Regression model (64.6% accuracy on 24 classes). "
        "Treat it as a hint, not a verdict."
    )

import json
import re
import joblib
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from utils import clean

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq()  # reads GROQ_API_KEY from .env 
_model = joblib.load("model.joblib")
_vec = joblib.load("vectorizer.joblib")


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise ValueError("No text found. The PDF may be a scanned image.")
    return text


def predict_category(resume_text, top_k=3):
    probs = _model.predict_proba(_vec.transform([clean(resume_text)]))[0]
    top = probs.argsort()[::-1][:top_k]
    return [(_model.classes_[i], round(float(probs[i]) * 100, 1)) for i in top]


def _embed_long(text, chunk_words=150):
    # MiniLM only reads ~256 tokens, so split long text into chunks and average
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]
    emb = embedder.encode(chunks or [""])
    return np.mean(emb, axis=0)


def similarity_score(resume, jd):
    a, b = _embed_long(resume), _embed_long(jd)
    return round(float(util.cos_sim(a, b)) * 100, 1)

def get_feedback(resume, jd, retries=2):
    prompt = f"""You are an expert recruiter. Compare the resume to the job description.
Return ONLY valid JSON with these keys:
- "matched_skills": list of skills the resume has that the job wants
- "missing_skills": list of important skills the job wants that the resume lacks
- "suggestions": list of 3-5 specific improvements to the resume
- "summary": one short paragraph on overall fit

RESUME:
{resume[:6000]}

JOB DESCRIPTION:
{jd[:4000]}"""
    last_err = None
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2000,
                reasoning_effort="low",
            )
            data = json.loads(resp.choices[0].message.content)
            for key in ("matched_skills", "missing_skills", "suggestions"):
                data.setdefault(key, [])
            data.setdefault("summary", "")
            return data
        except json.JSONDecodeError as e:
            last_err = e
    raise ValueError("The AI returned an invalid response. Please try again.") from last_err


def normalize_similarity(raw, low=30, high=75):
    # raw cosine scores for resume-vs-JD text usually fall between ~30 and ~75
    return max(0.0, min(100.0, (raw - low) / (high - low) * 100))


def combined_score(raw_sim, feedback):
    matched = len(feedback["matched_skills"])
    missing = len(feedback["missing_skills"])
    total = matched + missing
    coverage = matched / total * 100 if total else 0.0
    sim_norm = normalize_similarity(raw_sim)
    final = 0.6 * coverage + 0.4 * sim_norm
    return {
        "final": round(final, 1),
        "skill_coverage": round(coverage, 1),
        "semantic_similarity": round(sim_norm, 1),
    }


if __name__ == "__main__":
    resume = extract_text("sample_resume.pdf")
    jd = open("sample_jd.txt", encoding="utf-8").read()

    sim = similarity_score(resume, jd)
    feedback = get_feedback(resume, jd)
    print("Predicted categories:", predict_category(resume))
    print("Raw similarity:", sim)
    print("Scores:", combined_score(sim, feedback))

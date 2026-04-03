import streamlit as st
import joblib
import numpy as np
import scipy.sparse as sp
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🗞️",
    layout="centered",
)

# ── Custom CSS (dark theme matching screenshot) ───────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #0e1117; color: #ffffff; }

    /* Title */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }

    /* Warning banner */
    .warn-box {
        background-color: #1a2a3a;
        border-left: 4px solid #f0a500;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.5rem;
        color: #c9d8e8;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* Label above textarea */
    .field-label {
        font-size: 1rem;
        font-weight: 500;
        color: #d0d0d0;
        margin-bottom: 0.3rem;
    }

    /* Textarea override */
    textarea {
        background-color: #1e1e2e !important;
        color: #ffffff !important;
        border: 1px solid #333 !important;
        border-radius: 6px !important;
        font-size: 0.95rem !important;
    }

    /* Predict button */
    .stButton > button {
        background-color: #1e1e2e;
        color: #ffffff;
        border: 1px solid #555;
        border-radius: 6px;
        padding: 0.5rem 1.6rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #2e2e4e;
        border-color: #888;
    }

    /* Result cards */
    .result-real {
        background-color: #0d2b1f;
        border: 1px solid #1e7a4a;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-fake {
        background-color: #2b0d0d;
        border: 1px solid #7a1e1e;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .result-confidence {
        font-size: 1rem;
        color: #aaaaaa;
    }
    .signal-row {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)


# ── Load NLTK data ────────────────────────────────────────────────────────────
@st.cache_resource
def load_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

stop_words, lemmatizer = load_nltk()


# ── Load model bundle ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    bundle = joblib.load("fake_news_lr_model.pkl")
    return bundle["tfidf"], bundle["kmeans"], bundle["iso"], bundle["ensemble"]

try:
    tfidf, kmeans, iso, model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(text):
    cleaned  = clean_text(text)
    vec      = tfidf.transform([cleaned])
    cluster  = kmeans.predict(vec)[0]
    anomaly  = 0 if iso.predict(vec)[0] == 1 else 1
    extra    = np.array([[cluster, anomaly]])
    vec_full = sp.hstack((vec, extra))
    pred     = model.predict(vec_full)[0]
    label    = "REAL" if pred == 1 else "FAKE"

    confidence = None
    if hasattr(model, "predict_proba"):
        prob       = model.predict_proba(vec_full)[0]
        confidence = round(max(prob) * 100, 2)

    return label, confidence, cluster, anomaly


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🗞️ Fake News Detection System</p>', unsafe_allow_html=True)

st.markdown("""
<div class="warn-box">
    ⚠️ &nbsp;<strong>Note:</strong> This model was trained on historical political news datasets
    (ISOT &amp; LIAR). Predictions on very recent news, non-political topics, or
    region-specific stories may be less reliable. Real-time API training is included
    to improve coverage.
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"❌ Could not load model: `{load_error}`\n\nMake sure `fake_news_lr_model.pkl` is in the same folder as `app.py`.")
    st.stop()

st.write("Enter a news article below to classify it as **Real** or **Fake**.")

st.markdown('<p class="field-label">News Text</p>', unsafe_allow_html=True)
news_input = st.text_area("", height=180, placeholder="Paste a news headline or article here...", label_visibility="collapsed")

predict_btn = st.button("Predict")

if predict_btn:
    if not news_input.strip():
        st.warning("Please enter some news text first.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence, cluster, anomaly = predict(news_input)

        if label == "REAL":
            conf_text = f"{confidence}% confidence" if confidence else ""
            st.markdown(f"""
            <div class="result-real">
                <div class="result-label">✅ REAL NEWS</div>
                <div class="result-confidence">{conf_text}</div>
                <div class="signal-row">
                    <span>Cluster: {cluster}</span>
                    <span>Anomaly: {"Yes" if anomaly else "No"}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            conf_text = f"{confidence}% confidence" if confidence else ""
            st.markdown(f"""
            <div class="result-fake">
                <div class="result-label">❌ FAKE NEWS</div>
                <div class="result-confidence">{conf_text}</div>
                <div class="signal-row">
                    <span>Cluster: {cluster}</span>
                    <span>Anomaly: {"Yes" if anomaly else "No"}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#444; font-size:0.8rem;'>"
    "Hybrid Model · TF-IDF + K-Means + Isolation Forest · ISOT & LIAR Dataset"
    "</p>",
    unsafe_allow_html=True
)

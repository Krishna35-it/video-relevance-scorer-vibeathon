# relevance.py
from sentence_transformers import SentenceTransformer, util
import numpy as np
from utils import clean_text, chunk_text_words
import os

MODEL_NAME = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-mpnet-base-v2")
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def compute_similarity_metrics(title, transcript, description=None, chunk_size=200, overlap=50):
    title = clean_text(title)
    transcript = clean_text(transcript)
    if description:
        description = clean_text(description)

    chunks = chunk_text_words(transcript, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return {
            "score": 0,
            "mean_sim": 0.0,
            "max_sim": 0.0,
            "top_snippet": "",
            "sims": [],
            "chunks": []
        }

    model = get_model()
    title_emb = model.encode(title, convert_to_tensor=True)
    chunk_embs = model.encode(chunks, convert_to_tensor=True)
    sims = util.cos_sim(title_emb, chunk_embs).cpu().numpy().flatten().tolist()

    # if description exists, slightly combine semantics
    if description:
        desc_emb = model.encode(description, convert_to_tensor=True)
        avg_emb = (title_emb + desc_emb) / 2.0
        sims = util.cos_sim(avg_emb, chunk_embs).cpu().numpy().flatten().tolist()

    mean_sim = float(np.mean(sims))
    max_sim = float(np.max(sims))
    # composite scoring (tunable)
    raw_score = 0.6 * mean_sim + 0.4 * max_sim
    raw_score = max(0.0, min(1.0, raw_score))
    score_pct = int(round(raw_score * 100))

    top_idx = int(np.argmax(sims))
    top_snippet = chunks[top_idx]

    return {
        "score": score_pct,
        "mean_sim": mean_sim,
        "max_sim": max_sim,
        "top_snippet": top_snippet,
        "sims": sims,
        "chunks": chunks
    }

# Simple promotional detection (rule-based)
PROMO_KEYWORDS = [
    "visit our", "buy now", "order now", "discount", "subscribe", "sponsor", "sponsored", 
    "promo", "sale", "use code", "link below", "check out", "partner", "our product"
]

def detect_promotional_chunks(chunks):
    promo_flags = []
    for c in chunks:
        low = c.lower()
        flag = any(k in low for k in PROMO_KEYWORDS)
        promo_flags.append(flag)
    return promo_flags

def detect_offtopic_chunks(sims, threshold=0.3):
    # chunks with similarity below threshold considered off-topic
    return [s < threshold for s in sims]

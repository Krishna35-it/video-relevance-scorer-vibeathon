# app.py
import streamlit as st
from youtube_utils import download_youtube_audio_noffmpeg
from transcribe import transcribe_with_openai, transcribe_local_whisper
from relevance import compute_similarity_metrics, detect_promotional_chunks, detect_offtopic_chunks
from explain import generate_rationale_v1
from utils import clean_text
from dotenv import load_dotenv
import os
import csv, datetime
load_dotenv()

st.set_page_config(page_title="Video Relevance Scorer", layout="wide")
st.title("Video Relevance Scorer — Vibeathon")

st.sidebar.header("Settings")
use_local_whisper = st.sidebar.checkbox("Use local Whisper", value=False)
transcribe_model = st.sidebar.text_input("Transcribe model", value=os.getenv("OPENAI_MODEL_FOR_TRANSCRIBE","gpt-4o-mini-transcribe"))
llm_model = st.sidebar.text_input("LLM model", value=os.getenv("OPENAI_LLM_MODEL","gpt-4o"))

st.markdown("Input a YouTube URL *or* upload a video/audio file. Provide Title (required) and optional Description / Transcript.")
yt_url = st.text_input("YouTube URL")
uploaded_file = st.file_uploader("Upload audio/video (mp4, mp3, wav) (optional)", type=["mp4","mp3","wav"])
title = st.text_input("Video Title", "")
description = st.text_area("Video Description (optional)")
transcript_input = st.text_area("Paste transcript (optional)", height=200)

def log_run(url, title, score, verdict, transcript):
    """Append a run to runs.csv (safe: catches exceptions)."""
    try:
        with open("runs.csv","a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([url or "", title, score, verdict, len(transcript.split()) if transcript else 0, datetime.datetime.utcnow().isoformat()])
    except Exception as e:
        # don't crash the app for logging errors; just show a warning
        st.warning(f"Could not log run: {e}")


if st.button("Evaluate Relevance"):
    if not title:
        st.error("Please enter a Title.")
        st.stop()

    file_path = None
    transcript_text = ""
    if yt_url:
        st.info("Downloading audio from YouTube...")
        try:
            file_path = download_youtube_audio_noffmpeg(yt_url)
            st.success(f"Downloaded to {file_path}")
        except Exception as e:
            st.error(f"Download error: {e}")
            st.stop()
    elif uploaded_file:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp.close()
        file_path = tmp.name
        st.success(f"Saved uploaded file to {file_path}")

    # Transcription
    if transcript_input.strip():
        transcript_text = transcript_input
    elif file_path:
        st.info("Transcribing audio (this can take a while)...")
        try:
            if use_local_whisper:
                transcript_text = transcribe_local_whisper(file_path)
            else:
                transcript_text = transcribe_with_openai(file_path, model=transcribe_model)
            st.success("Transcription completed.")
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()
    else:
        st.error("No input provided. Upload a file, paste transcript, or give a YouTube URL.")
        st.stop()

    # Compute relevance (we only need final score)
    st.info("Computing semantic relevance...")
    result = compute_similarity_metrics(title, transcript_text, description=description)
    score = result.get("score", 0)

    # Determine verdict
    if score >= 75:
        verdict = "Strong match"
    elif score >= 40:
        verdict = "Partial match"
    else:
        verdict = "Low relevance"

    # Display only score and justification
    st.subheader("Relevance Score")
    st.metric(label="", value=f"{score}%")

    # Get short LLM justification
    st.subheader("Short justification")
    top_snippet = result.get("top_snippet","")
    try:
        rationale = generate_rationale_v1(title, score, top_snippet)
    except TypeError:
        # fallback if generate_rationale_v1 signature is old (only title,score)
        rationale = generate_rationale_v1(title, score)
    st.write(rationale)

    # Simple verdict label (redundant but clear)
    if verdict == "Strong match":
        st.success(f"Verdict: {verdict}")
    elif verdict == "Partial match":
        st.warning(f"Verdict: {verdict}")
    else:
        st.error(f"Verdict: {verdict}")

    # show progress bar
    try:
        st.progress(score / 100)
    except Exception:
        pass

    # Log the run safely
    log_run(yt_url, title, score, verdict, transcript_text)

# transcribe.py
import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI()

def transcribe_with_openai(filepath: str, model: str = None, language: str | None = None) -> str:
    """
    Transcribe an audio/video file using OpenAI's v1 Python client.
    Supports .mp3, .wav, .m4a, .webm, .mp4.
    """
    if model is None:
        model = os.getenv("OPENAI_MODEL_FOR_TRANSCRIBE", "gpt-4o-mini-transcribe")

    with open(filepath, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            language=language
        )

    # New v1 client returns a dict-like object containing "text"
    return response.get("text", "") if isinstance(response, dict) else getattr(response, "text", "")

# Optional local fallback (if you install whisper)
def transcribe_local_whisper(filepath, model_size="base"):
    """
    Local whisper fallback (requires `pip install -U openai-whisper` and ffmpeg).
    """
    try:
        import whisper
    except Exception as e:
        raise RuntimeError("Local whisper not installed. Install via `pip install -U openai-whisper`") from e

    model = whisper.load_model(model_size)
    result = model.transcribe(filepath)
    return result.get("text", "")

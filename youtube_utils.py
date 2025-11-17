# youtube_utils_noffmpeg.py
import yt_dlp
import tempfile
import os
from pathlib import Path
from typing import Optional

def download_youtube_audio_noffmpeg(url: str, out_dir: Optional[str] = None, prefer_exts=("m4a","webm","mp3","opus")) -> str:
    """
    Download best audio from a YouTube URL WITHOUT using ffmpeg postprocessing.
    Returns absolute path to the downloaded audio file.

    Strategy:
      - Ask yt-dlp for 'bestaudio' in native container.
      - Do NOT use FFmpegExtractAudio postprocessor.
      - After download, try to pick a preferred audio extension if available.

    Notes:
      - The returned file will often be .m4a, .webm, .opus, or .mp4 depending on YouTube.
      - This file is passed directly to the transcription endpoint.
      - If the transcription API fails due to unsupported codec, you'll need FFmpeg.
    """
    if out_dir is None:
        out_dir = tempfile.mkdtemp()
    else:
        os.makedirs(out_dir, exist_ok=True)

    # Create an output template. Keep original extension.
    out_template = os.path.join(out_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Do NOT set postprocessors that require ffmpeg
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # Determine downloaded filename(s) in out_dir
    files = list(Path(out_dir).glob("*"))
    if not files:
        raise FileNotFoundError(f"No files downloaded in {out_dir} for url {url}")

    # Prefer common audio extensions (m4a, webm, mp3, opus); otherwise pick the largest file
    preferred = None
    for ext in prefer_exts:
        candidates = [f for f in files if f.suffix.lower().lstrip('.') == ext.lower()]
        if candidates:
            # choose largest among candidates
            preferred = max(candidates, key=lambda p: p.stat().st_size)
            break

    if preferred is None:
        # fallback: choose the largest downloaded file (most likely the audio)
        preferred = max(files, key=lambda p: p.stat().st_size)

    return str(preferred.resolve())

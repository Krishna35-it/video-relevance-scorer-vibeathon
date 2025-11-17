# Video Relevance Scorer — Vibeathon MVP  
This project evaluates how relevant a video’s content is to its provided title or topic.
It supports YouTube URLs, manual uploads, automatic transcription using OpenAI Whisper, and generates a 0–100 relevance score with a short explanation.  

# Features  
- Accepts YouTube URL or uploaded video/audio file  
- Generates transcript using OpenAI Whisper (gpt-4o-mini-transcribe)  
- Computes semantic similarity between title and transcript  
- Produces a Relevance Score (0–100%)  
- Provides a short LLM-based justification  
- Logs every run into runs.csv  
- No FFmpeg dependency (YouTube audio downloaded directly)  

# Project Structure  

video-relevance-scorer/
│
├── app.py
├── transcribe.py
├── youtube_utils.py
├── relevance.py
├── explain.py
├── utils.py
├── requirements.txt
├── .env.example
├── runs.csv
└── README.md

# Installation  
## 1. Clone the repository  
```
git clone https://github.com/yourusername/video-relevance-scorer.git
cd video-relevance-scorer
```

## 2. Create a virtual environment  
```
python -m venv venv
```

Windows:
```
venv\Scripts\activate
```

Mac/Linux:
```
source venv/bin/activate
```  
## 3. Install required packages  
```
pip install -r requirements.txt
```  
## 4. Set up environment variables  

Create a .env file:  
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL_FOR_TRANSCRIBE=gpt-4o-mini-transcribe
OPENAI_LLM_MODEL=gpt-4o
SENTENCE_TRANSFORMER_MODEL=all-mpnet-base-v2
USE_OPENAI_EMBEDDINGS=false
```  
## Running the Application  
```  
streamlit run app.py
```  
The app will open in your browser.

# How It Works  
## 1. Audio Extraction  
- If a YouTube URL is provided, the app downloads the best available audio without needing FFmpeg.  
- If a file is uploaded, it is saved locally for transcription.  

## 2. Transcription  
- Transcript is generated using:  
```
client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=audio_file)
```  
## 3. Chunking  
- The transcript is split into overlapping chunks of approximately 200 words.  

## 4. Embedding Similarity  
- The title and transcript chunks are embedded using sentence-transformers.  
- Cosine similarity is calculated for each chunk.  
- Final score is computed using:  
```  
score = 0.6 * mean_similarity + 0.4 * max_similarity  
```  
## 5. Justification  
- A concise explanation is generated using the chat completion endpoint:  
```  
client.chat.completions.create(...)  
```  
## 6. Verdict  
- Final interpretation of the score:  
  * 75–100 : Strong match  
  * 40–74 : Partial match  
  * 0–39 : Low relevance  

## 7. Logging  
- Every evaluation is appended to runs.csv:  
  * URL, title, score, verdict, transcript_length, timestamp  

# Example Output  
```  
Relevance Score: 72%  
Short justification:  
"Partial match — the content is related to the political topic but not fully aligned with the provided title."  
Verdict: Partial match  
```

# Notes  
- Long videos may require additional transcription time.  
- YouTube formats vary; if a rare format fails to transcribe, try another video.  
- The LLM justification is intentionally constrained to avoid hallucination.  

# Future Enhancements  

- elevance heatmap visualization  
- Promotional/off-topic chunk classifier  
- WhisperX timestamped transcripts  
- Multiple language support  
- LangChain-based retrieval explanations  

# License  
This project is created for Vibeathon. You may extend or adapt it as needed.

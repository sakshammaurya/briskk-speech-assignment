# Speech-to-Text API (FastAPI + Whisper)

A small demonstration API for speech transcription, query suggestions, and a WebSocket endpoint that accepts complete audio payloads. This is a prototype, not a production-ready real-time streaming system.

## What is implemented

- `POST /api/voice-to-text`: transcribes an uploaded audio file using Whisper's `tiny` model.
- `GET /api/autocomplete?q=...`: returns simple template-based suggestions. Suggestions are **not** personalized, embedding-based, or popularity-ranked.
- `WS /ws/speech-to-search`: accepts one complete WAV-encoded audio file per WebSocket message and returns a transcription. It does not process raw microphone frames or perform noise reduction.

The Whisper model is loaded when the application process starts. The first startup can take time, and transcription speed depends on the machine. FFmpeg is required by Whisper.

## Requirements

- Python 3.10 or newer (use a version supported by your installed PyTorch/Whisper build)
- FFmpeg installed and available on your PATH
- Enough memory and disk space for PyTorch and the Whisper model

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive API docs.

## API usage

Upload a supported audio file (WAV, MP3, M4A, MPEG/MPGA, WebM, or OGG) to `POST /api/voice-to-text` as multipart form data using the field name `audio`. Uploads are limited to 10 MB.

Example with curl:

```bash
curl -X POST "http://127.0.0.1:8000/api/voice-to-text" \
  -F "audio=@sample.wav"
```

Autocomplete example:

```bash
curl "http://127.0.0.1:8000/api/autocomplete?q=find%20me"
```

## Run automated tests

The tests mock the transcription function, so they do not download Whisper weights or require FFmpeg. Install the API/test dependencies and run:

```bash
python -m pip install fastapi python-multipart "httpx>=0.27,<1.0" "pytest>=8,<9"
python -m pytest -q
```

GitHub Actions runs this lightweight test suite on pushes and pull requests. The tests cover autocomplete responses, invalid/empty/oversized uploads, the mocked transcription response, and the WebSocket response. They do not measure real transcription accuracy or latency.

## Limitations and next steps

- No denoising pipeline or measured before/after accuracy comparison is included.
- Autocomplete currently uses static templates rather than search history, Redis, embeddings, or ranking.
- WebSocket clients must send complete encoded WAV files per message; arbitrary microphone chunks may not be independently decodable.
- No deployment configuration is included yet.
- For production, add tests, rate limiting, authentication where appropriate, structured logging, request timeouts, and workload isolation for CPU-heavy transcription.

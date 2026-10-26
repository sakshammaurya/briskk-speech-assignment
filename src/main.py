from pathlib import Path
import tempfile

import whisper
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

app = FastAPI(title="Speech-to-Text API")
MAX_AUDIO_BYTES = 10 * 1024 * 1024
ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".mpeg", ".mpga", ".webm", ".ogg"}

# Load once per worker process. Whisper also requires FFmpeg to be installed.
model = whisper.load_model("tiny")


def transcribe_bytes(audio_data: bytes, suffix: str) -> str:
    """Write an encoded audio payload to a temporary file for Whisper/FFmpeg."""
    if not audio_data:
        raise ValueError("Audio payload is empty")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as audio_file:
        audio_file.write(audio_data)
        audio_file.flush()
        result = model.transcribe(audio_file.name)
    return result["text"].strip()


@app.post("/api/voice-to-text")
async def voice_to_text(audio: UploadFile = File(...)):
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Unsupported audio format")

    audio_data = await audio.read(MAX_AUDIO_BYTES + 1)
    if len(audio_data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds 10 MB limit")
    if not audio_data:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    try:
        text = transcribe_bytes(audio_data, suffix)
    except Exception:
        # Avoid exposing model/FFmpeg internals to callers.
        raise HTTPException(status_code=422, detail="Could not transcribe this audio file")
    finally:
        await audio.close()

    return {"text": text}


@app.get("/api/autocomplete")
async def autocomplete(q: str):
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter q must not be empty")
    # Demonstration suggestions only; this is not intent-aware ranking.
    return {"suggestions": [
        f"{query} red shoes",
        f"{query} blue jeans",
        f"{query} latest trends",
    ]}


@app.websocket("/ws/speech-to-search")
async def speech_to_search(websocket: WebSocket):
    """Accept one complete encoded audio file per message (not raw mic frames)."""
    await websocket.accept()
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            if len(audio_data) > MAX_AUDIO_BYTES:
                await websocket.send_json({"error": "Audio payload exceeds 10 MB limit"})
                continue
            try:
                # WebSocket payloads are expected to be WAV files.
                text = transcribe_bytes(audio_data, ".wav")
                await websocket.send_json({"text": text})
            except Exception:
                await websocket.send_json({"error": "Could not transcribe this audio payload"})
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

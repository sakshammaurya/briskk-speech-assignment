"""
FastAPI Speech-to-Text Service with Whisper Integration

Features:
- Audio file transcription endpoint
- WebSocket real-time transcription
- Autocomplete suggestions
- Configurable temp directory
- FFmpeg audio conversion
- Noise reduction with DeepFilterNet
- Redis-backed search tracking
"""

import numpy as np
from whisper import load_model  # OpenAI's Whisper model
from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException, Query, WebSocketDisconnect
from pathlib import Path
import whisper
import tempfile
import os
import logging
import uvicorn
import subprocess
from typing import Dict, List
import soundfile as sf
import librosa
import torch
import redis
import time
from sentence_transformers import SentenceTransformer  # For semantic search

# DeepFilterNet for noise reduction
from df import init_df, enhance

# Initialize FastAPI app
app = FastAPI(
    title="Speech-to-Text API",
    description="API for converting speech to text using OpenAI Whisper with noise reduction and autocomplete suggestions",
    version="1.0.0"
)

# Configuration
MODEL_NAME = os.getenv("WHISPER_MODEL", "base.en")
TEMP_DIR = Path(os.getenv("TEMP_DIR", tempfile.gettempdir()))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Initialize Whisper model
model = whisper.load_model(MODEL_NAME)

# Initialize Redis connection
redis_conn = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=False
)

# Initialize SentenceTransformer for semantic search
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_wav(input_path: Path, output_path: Path) -> None:
    """
    Convert any audio file to 16kHz WAV format using FFmpeg
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-ar', '16000',  # Sample rate
        '-ac', '1',       # Mono audio
        '-loglevel', 'error',
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg conversion failed: {e.stderr.decode()}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

@app.post("/api/voice-to-text", response_model=Dict[str, str])
async def voice_to_text(audio: UploadFile = File(...)) -> Dict[str, str]:
    """
    Convert uploaded audio file to text with optional noise reduction
    """
    input_path = output_path = denoised_path = None
    
    try:
        # Validate file type
        if not audio.filename.lower().endswith((".wav", ".mp3", ".ogg", ".m4a")):
            raise HTTPException(400, detail="Unsupported file format")

        # Validate file size
        if audio.size > MAX_FILE_SIZE:
            raise HTTPException(413, detail="File size exceeds 10MB limit")

        # Create temp files
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(audio.filename)[1],
            dir=TEMP_DIR,
            delete=False
        ) as orig_file, tempfile.NamedTemporaryFile(
            suffix=".wav",
            dir=TEMP_DIR,
            delete=False
        ) as wav_file:
            
            # Save original file
            content = await audio.read()
            orig_file.write(content)
            input_path = Path(orig_file.name)
            output_path = Path(wav_file.name)

        # Convert to standardized WAV format
        convert_to_wav(input_path, output_path)

        # Transcribe original audio
        original_result = model.transcribe(str(output_path))
        original_text = original_result['text'].strip()

        # Noise reduction processing
        try:
            data, sr = sf.read(str(output_path))
            
            # DeepFilterNet requires 48kHz
            target_sr = 48000
            if sr != target_sr:
                data = librosa.resample(data.T, orig_sr=sr, target_sr=target_sr).T
                sr = target_sr

            # Convert to tensor
            data_tensor = torch.tensor(data, dtype=torch.float32)
            if data_tensor.ndim == 1:
                data_tensor = data_tensor.unsqueeze(0)

            # Initialize DeepFilterNet
            df_model, df_state, _ = init_df()
            enhanced_audio_tensor = enhance(df_model, df_state, data_tensor)
            enhanced_audio = enhanced_audio_tensor.cpu().numpy()

            # Resample back to 16kHz for Whisper
            if sr == 48000:
                enhanced_audio = librosa.resample(enhanced_audio.T, orig_sr=48000, target_sr=16000).T
                sr = 16000

            # Save denoised audio
            with tempfile.NamedTemporaryFile(suffix="_denoised.wav", dir=TEMP_DIR, delete=False) as denoised_file:
                denoised_path = Path(denoised_file.name)
                sf.write(str(denoised_path), enhanced_audio, sr)
            
            # Transcribe denoised audio
            denoised_result = model.transcribe(str(denoised_path))
            denoised_text = denoised_result["text"].strip()
        except Exception as denoise_error:
            logger.error(f"Noise reduction failed: {str(denoise_error)}")
            denoised_text = "Noise reduction unavailable"

        return {
            "original_text": original_text,
            "denoised_text": denoised_text
        }

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=f"Transcription error: {str(e)}")
    
    finally:
        # Cleanup temporary files
        for path in [input_path, output_path, denoised_path]:
            if path and path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete {path}: {str(e)}")

@app.get("/api/autocomplete")
async def autocomplete(q: str = Query(..., min_length=2)) -> Dict[str, List[str]]:
    """
    Provide autocomplete suggestions based on search history and semantic similarity
    """
    # Prefix search from Redis
    matches = []
    cursor = 0
    while True:
        cursor, data = redis_conn.zscan('autocomplete_index', cursor, match=f'{q}*')
        matches.extend(data)
        if cursor == 0:
            break

    # Rank by popularity
    phrases = [(p.decode(), int(c)) for p, c in matches]
    phrases.sort(key=lambda x: (-x[1], x[0]))
    popular_suggestions = [p[0] for p in phrases[:10]]

    # Semantic search
    query_embedding = semantic_model.encode(q)
    # Here you would query a vector DB for similar phrases
    semantic_suggestions = [
        f"{q} stylish outfits",
        f"{q} trendy fashion",
        f"{q} seasonal collection"
    ]

    # Combine and deduplicate
    combined_suggestions = list(set(popular_suggestions + semantic_suggestions))[:10]
    return {"suggestions": combined_suggestions}

@app.websocket("/ws/speech-to-search")
async def speech_to_search(websocket: WebSocket):
    """
    Real-time speech-to-text with autocomplete suggestions
    """
    await websocket.accept()
    try:
        buffer = []
        while True:
            audio_chunk = await websocket.receive_bytes()
            buffer.append(np.frombuffer(audio_chunk, dtype=np.float32))
            
            if len(buffer) >= 5:  # Process every 5 chunks
                full_audio = np.concatenate(buffer)
                result = model.transcribe(full_audio)
                await websocket.send_text(result["text"])
                
                # Get autocomplete suggestions
                suggestions = await autocomplete(result["text"])
                await websocket.send_json(suggestions)
                buffer = []
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
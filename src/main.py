"""
FastAPI Speech-to-Text Service with Whisper Integration

Features:
- Audio file transcription endpoint
- WebSocket real-time transcription
- Autocomplete suggestions
- Configurable temp directory
- FFmpeg audio conversion
"""

from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException
from pathlib import Path
import whisper
import tempfile
import os
import logging
import uvicorn
import subprocess
from typing import List, Dict


# Initialize FastAPI app
app = FastAPI(
    title="Speech-to-Text API",
    description="API for converting speech to text using OpenAI Whisper",
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_wav(input_path: Path, output_path: Path) -> None:
    """
    Convert any audio file to 16kHz WAV format using FFmpeg
    
    Args:
        input_path: Path to source audio file
        output_path: Path for converted WAV file
        
    Raises:
        RuntimeError: If conversion fails
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
    Convert uploaded audio file to text
    
    Supported formats: WAV, MP3, OGG, M4A
    Max file size: 10MB
    """
    input_path = output_path = None
    
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

        # Transcribe audio
        result = model.transcribe(str(output_path))
        return {"text": result["text"].strip()}

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=f"Transcription error: {str(e)}")
    
    finally:
        # Cleanup temporary files
        for path in [input_path, output_path]:
            if path and path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete {path}: {str(e)}")

@app.websocket("/ws/speech-to-search")
async def speech_to_search(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming
    
    Accepts raw audio chunks in WAV format
    Returns incremental transcription results
    """
    await websocket.accept()
    temp_file_path = None
    
    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                dir=TEMP_DIR,
                delete=False
            ) as temp_file:
                temp_file.write(audio_chunk)
                temp_file_path = Path(temp_file.name)
                
            result = model.transcribe(str(temp_file_path))
            await websocket.send_text(result["text"].strip())
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception as e:
                logger.warning(f"Cleanup failed: {str(e)}")

@app.get("/api/autocomplete", response_model=Dict[str, List[str]])
async def autocomplete(q: str) -> Dict[str, List[str]]:
    """
    Generate search suggestions based on query
    
    Args:
        q: Search query string
        
    Returns:
        List of autocomplete suggestions
    """
    return {"suggestions": [
        f"{q} red shoes",
        f"{q} blue jeans",
        f"{q} latest trends"
    ]}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000"))
    )
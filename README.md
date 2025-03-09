# Speech-to-Text API with Whisper

This project implements a FastAPI-based speech-to-text service using OpenAI's Whisper model. It supports both file uploads and real-time audio streaming via WebSocket.

## Features

- **File Upload**: Accepts WAV, MP3, OGG, and M4A files.
- **Real-Time Transcription**: WebSocket endpoint for streaming audio.
- **Autocomplete Suggestions**: Simple autocomplete API for search queries.
- **Configurable**: Environment variables for model, temp directory, and file size limits.

## Trade-offs

### Whisper vs. DeepSpeech

- **Whisper**:
  - Pros: State-of-the-art accuracy, supports multiple languages, handles noisy audio well.
  - Cons: Larger model size, requires more computational resources.
- **DeepSpeech**:
  - Pros: Lightweight, faster inference.
  - Cons: Lower accuracy, limited language support.

### Redis vs. Pinecone for Ranking

- **Redis**:
  - Pros: Fast in-memory storage, simple to set up, good for caching.
  - Cons: Limited support for complex ranking algorithms.
- **Pinecone**:
  - Pros: Designed for vector search, supports advanced ranking and similarity search.
  - Cons: Requires external service, more complex setup.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repository-name>.git
   cd <repository-name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
4. Start the server
   ```bash
   uvicorn main:app --reload
   ```

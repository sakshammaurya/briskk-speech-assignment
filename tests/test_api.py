from fastapi.testclient import TestClient

from src import main


client = TestClient(main.app)


def test_autocomplete_returns_demo_suggestions():
    response = client.get("/api/autocomplete", params={"q": "find me"})

    assert response.status_code == 200
    assert response.json()["suggestions"] == [
        "find me red shoes",
        "find me blue jeans",
        "find me latest trends",
    ]


def test_autocomplete_rejects_blank_query():
    response = client.get("/api/autocomplete", params={"q": "   "})

    assert response.status_code == 400


def test_voice_to_text_rejects_unsupported_file_type():
    response = client.post(
        "/api/voice-to-text",
        files={"audio": ("recording.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 415


def test_voice_to_text_rejects_empty_audio():
    response = client.post(
        "/api/voice-to-text",
        files={"audio": ("recording.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400


def test_voice_to_text_returns_transcription(monkeypatch):
    monkeypatch.setattr(main, "transcribe_bytes", lambda audio_data, suffix: "hello world")

    response = client.post(
        "/api/voice-to-text",
        files={"audio": ("recording.wav", b"fake wav payload", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "hello world"}


def test_voice_to_text_rejects_oversized_upload():
    payload = b"x" * (main.MAX_AUDIO_BYTES + 1)
    response = client.post(
        "/api/voice-to-text",
        files={"audio": ("recording.wav", payload, "audio/wav")},
    )

    assert response.status_code == 413


def test_websocket_returns_transcription(monkeypatch):
    monkeypatch.setattr(main, "transcribe_bytes", lambda audio_data, suffix: "websocket transcript")

    with client.websocket_connect("/ws/speech-to-search") as websocket:
        websocket.send_bytes(b"fake wav payload")
        assert websocket.receive_json() == {"text": "websocket transcript"}

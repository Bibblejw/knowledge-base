"""Transcription worker — LemonFox.ai API (cloud, with speaker diarization).

Uses the LemonFox Speech-to-Text API for accurate transcription with
speaker labels. Replaces the local faster-whisper + pyannote pipeline.
"""

import json
import os
import time
from pathlib import Path

import requests

RECORDINGS_DIR = Path("/recordings")
TRANSCRIPTIONS_DIR = RECORDINGS_DIR / ".transcriptions"
LEMONFOX_API = "https://api.lemonfox.ai/v1/audio/transcriptions"


def _status_path(filename: str) -> Path:
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return TRANSCRIPTIONS_DIR / f"{filename}.status.json"


def _result_path(filename: str) -> Path:
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return TRANSCRIPTIONS_DIR / f"{filename}.result.json"


def get_status(filename: str) -> dict:
    """Return current transcription status for a given audio file."""
    status_file = _status_path(filename)
    result_file = _result_path(filename)

    if result_file.exists():
        with open(result_file) as f:
            return json.load(f)

    if status_file.exists():
        with open(status_file) as f:
            return json.load(f)

    return {"status": "not_started"}


def _get_api_key() -> str:
    """Return the LemonFox API key from env or .env file."""
    key = os.environ.get("LEMONFOX_API_KEY", "")
    if key:
        return key

    # Try loading from .env in the app directory
    env_path = Path("/opt/knowledge-base/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("LEMONFOX_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def run_transcription(filename: str, hf_token: str = "") -> dict:
    """Run transcription using the LemonFox API. Blocks until done."""
    audio_path = RECORDINGS_DIR / filename
    if not audio_path.exists():
        return {"status": "error", "error": "File not found"}

    api_key = _get_api_key()
    if not api_key:
        return {
            "status": "error",
            "error": "LemonFox API key not set. Add LEMONFOX_API_KEY to /opt/knowledge-base/.env",
        }

    status = {"status": "processing", "started_at": time.time(), "stage": "uploading"}
    _write_status(filename, status)

    try:
        # Upload file to LemonFox
        with open(audio_path, "rb") as f:
            files = {"file": (filename, f, "audio/m4a")}
            data = {
                "response_format": "verbose_json",
                "speaker_labels": "true",
                "language": "english",
            }

            status["stage"] = "transcribing"
            _write_status(filename, status)

            resp = requests.post(
                LEMONFOX_API,
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data=data,
                timeout=300,
            )

        if resp.status_code != 200:
            error_msg = resp.text[:500]
            result = {"status": "error", "error": f"LemonFox API error ({resp.status_code}): {error_msg}"}
            with open(_result_path(filename), "w") as f:
                json.dump(result, f, indent=2)
            return result

        data = resp.json()

        # Parse segments with speaker labels
        segments = []
        for seg in data.get("segments", []):
            segments.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
                "speaker": seg.get("speaker", None),
            })

        # Build full text with speaker labels
        full_text = "\n".join(
            f"[{s['speaker']}] {s['text']}" if s.get("speaker") else s["text"]
            for s in segments
        )

        result = {
            "status": "completed",
            "segments": segments,
            "full_text": full_text,
            "language": data.get("language", "en"),
            "duration": data.get("duration", 0),
            "completed_at": time.time(),
        }

        with open(_result_path(filename), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        _status_path(filename).unlink(missing_ok=True)
        return result

    except requests.Timeout:
        error_result = {"status": "error", "error": "LemonFox API request timed out"}
        return _save_error(filename, error_result)

    except Exception as e:
        error_result = {"status": "error", "error": str(e)}
        return _save_error(filename, error_result)


def _save_error(filename: str, result: dict) -> dict:
    """Write error result to disk and return it."""
    with open(_result_path(filename), "w") as f:
        json.dump(result, f)
    return result


def _write_status(filename: str, data: dict):
    """Write current processing status."""
    with open(_status_path(filename), "w") as f:
        json.dump(data, f)


def list_transcriptions() -> list[dict]:
    """Return a list of transcribed files with their status."""
    results = []
    if not RECORDINGS_DIR.exists():
        return results

    for f in sorted(RECORDINGS_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in {
            ".mp3", ".wav", ".flac", ".ogg", ".aac",
            ".m4a", ".opus", ".wma", ".aiff",
        }:
            status = get_status(f.name)
            results.append({
                "name": f.name,
                "status": status["status"],
            })

    return results

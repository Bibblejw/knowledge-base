"""Transcription worker — offline speech-to-text with speaker diarization.

Uses faster-whisper for transcription and pyannote.audio for diarization.
Runs asynchronously, writing results as JSON sidecar files alongside the audio.
"""

import json
import os
import time
from pathlib import Path

RECORDINGS_DIR = Path("/recordings")
TRANSCRIPTIONS_DIR = RECORDINGS_DIR / ".transcriptions"


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
            result = json.load(f)
        return {"status": "completed", "segments": result.get("segments", []), "full_text": result.get("full_text", "")}

    if status_file.exists():
        with open(status_file) as f:
            return json.load(f)

    return {"status": "not_started"}


def run_transcription(filename: str, hf_token: str = "") -> dict:
    """Run transcription with diarization. Blocks until done. Called from a thread."""
    audio_path = RECORDINGS_DIR / filename
    if not audio_path.exists():
        return {"status": "error", "error": "File not found"}

    # Mark as processing
    status = {"status": "processing", "started_at": time.time()}
    _write_status(filename, status)

    try:
        status["stage"] = "transcribing"
        _write_status(filename, status)

        segments, info = _transcribe(audio_path)

        # Run diarization if pyannote is available
        diarization_segments = []
        if hf_token:
            try:
                status["stage"] = "diarizing"
                _write_status(filename, status)
                diarization_segments = _diarize(audio_path, hf_token)
            except Exception as e:
                # Diarization is optional — log and continue
                pass

        # Align speaker labels with transcript segments
        aligned = _align(segments, diarization_segments)

        full_text = "\n".join(
            f"[{s['speaker']}] {s['text']}" if s.get("speaker") else s["text"]
            for s in aligned
        )

        result = {
            "status": "completed",
            "segments": aligned,
            "full_text": full_text,
            "language": info.language,
            "duration": info.duration,
            "completed_at": time.time(),
        }

        with open(_result_path(filename), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Clean up status file
        _status_path(filename).unlink(missing_ok=True)

        return result

    except Exception as e:
        error_result = {"status": "error", "error": str(e)}
        with open(_result_path(filename), "w") as f:
            json.dump(error_result, f)
        return error_result


def _transcribe(audio_path: Path):
    """Run faster-whisper transcription."""
    from faster_whisper import WhisperModel

    # Use the "base" model for CPU — good balance of speed vs accuracy
    # Falls back to "tiny" if base is too slow
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(audio_path), beam_size=3)

    result = []
    for seg in segments:
        result.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })

    return result, info


def _diarize(audio_path: Path, hf_token: str):
    """Run pyannote speaker diarization."""
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    diarization = pipeline(str(audio_path))
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    return segments


def _align(transcript_segments, diarization_segments):
    """Assign speaker labels to transcript segments."""
    if not diarization_segments:
        for seg in transcript_segments:
            seg["speaker"] = None
        return transcript_segments

    aligned = []
    for tseg in transcript_segments:
        t_mid = (tseg["start"] + tseg["end"]) / 2
        speaker = None
        for dseg in diarization_segments:
            if dseg["start"] <= t_mid <= dseg["end"]:
                speaker = dseg["speaker"]
                break
        tseg["speaker"] = speaker
        aligned.append(tseg)

    return aligned


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

"""Transcription pipeline — diarize-first with SpeechBrain + Parakeet ASR on Mac.

Pipeline:
  1. Pre-process audio (VAD, denoise, normalize) — on container
  2. Diarize + speaker matching (ECAPA embeddings + sklearn clustering) — on container
  3. Send pre-processed audio to Mac for Parakeet ASR
  4. Align ASR timestamps with speaker labels
  5. Fall back to LemonFox if Mac is unreachable
"""
import json
import os
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, "/opt/knowledge-base")

import requests
import logging
logger = logging.getLogger("transcriber")
from diarization import (
    preprocess_audio,
    diarize_audio,
    align_asr_with_diarization,
    enroll_speaker,
    list_speakers,
    remove_speaker,
    _load_speaker_library,
)

RECORDINGS_DIR = Path("/recordings")
TRANSCRIPTIONS_DIR = RECORDINGS_DIR / ".transcriptions"
MAC_ASR_URL = "http://10.3.0.207:5001/transcribe"
LEMONFOX_API = "https://api.lemonfox.ai/v1/audio/transcriptions"


def _status_path(filename: str) -> Path:
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return TRANSCRIPTIONS_DIR / f"{filename}.status.json"


def _result_path(filename: str) -> Path:
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return TRANSCRIPTIONS_DIR / f"{filename}.result.json"


def get_status(filename: str) -> dict:
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
    key = os.environ.get("LEMONFOX_API_KEY", "")
    if key:
        return key
    env_path = Path("/opt/knowledge-base/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("LEMONFOX_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _is_mac_alive() -> bool:
    try:
        resp = requests.get(f"{MAC_ASR_URL.replace('/transcribe', '/health')}", timeout=3)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def _transcribe_via_mac(audio_path: str) -> dict | None:
    """Send pre-processed audio to Mac for ASR. Returns {text, sentences} or None."""
    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                MAC_ASR_URL,
                files={"audio": ("audio.wav", f, "audio/wav")},
                timeout=600,
            )
        if resp.status_code != 200:
            print(f"Mac ASR error ({resp.status_code}): {resp.text[:200]}")
            return None
        return resp.json()
    except Exception as e:
        print(f"Mac ASR failed: {e}")
        return None


def _transcribe_via_lemonfox(audio_path: str) -> dict:
    """Fallback: send to LemonFox API."""
    api_key = _get_api_key()
    with open(audio_path, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        data = {
            "response_format": "verbose_json",
            "speaker_labels": "true",
            "language": "english",
        }
        resp = requests.post(
            LEMONFOX_API,
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            data=data,
            timeout=300,
        )

    if resp.status_code != 200:
        return {"status": "error", "error": f"LemonFox API error ({resp.status_code}): {resp.text[:500]}"}

    data = resp.json()
    segments = []
    for seg in data.get("segments", []):
        segments.append({
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "text": seg.get("text", "").strip(),
            "speaker": seg.get("speaker", None),
        })

    full_text = "\n".join(
        f"[{s['speaker']}] {s['text']}" if s.get("speaker") else s["text"]
        for s in segments
    )

    return {
        "status": "completed",
        "engine": "lemonfox-ai (fallback)",
        "sentences": segments,
        "full_text": full_text,
        "speaker_turns": [],
        "duration": data.get("duration", 0),
        "completed_at": time.time(),
    }


def run_transcription(filename: str, diarization_threshold: float = 0.65) -> dict:
    """
    Full transcription pipeline:
    Pre-process -> Diarize -> ASR -> Align -> Store
    Falls back to LemonFox if Mac is unreachable.
    """
    audio_path = RECORDINGS_DIR / filename
    if not audio_path.exists():
        result = {"status": "error", "error": "File not found"}
        with open(_result_path(filename), "w") as f:
            json.dump(result, f, indent=2)
        return result

    status = {"status": "processing", "started_at": time.time(), "stage": "starting"}
    _write_status(filename, status)

    try:
        # --- STEP 1: Pre-process ---
        status["stage"] = "preprocessing"
        _write_status(filename, status)
        logger.info("Pre-processing audio...")
        processed_path = preprocess_audio(str(audio_path))
        logger.info("Pre-processing complete")

        # --- STEP 2: Diarize + Speaker Match ---
        status["stage"] = "diarizing"
        _write_status(filename, status)
        logger.info("Running speaker diarization...")
        diarization_segments, speaker_turns = diarize_audio(
            processed_path, threshold=diarization_threshold
        )
        logger.info(f"Diarization complete: {len(diarization_segments)} segments")

        # --- STEP 3: ASR on Mac (or fallback) ---
        status["stage"] = "transcribing"
        _write_status(filename, status)

        library_speakers = list(_load_speaker_library().keys())

        if _is_mac_alive():
            logger.info("Sending to Mac for Parakeet ASR...")
            asr_result = _transcribe_via_mac(processed_path)

            if asr_result and "error" not in asr_result:
                # STEP 4: Align
                status["stage"] = "aligning"
                _write_status(filename, status)

                aligned_sentences = align_asr_with_diarization(asr_result, diarization_segments)

                # Build speaker turns from aligned output
                speaker_turns_aligned = []
                current = None
                for s in aligned_sentences:
                    if current is None or s["speaker"] != current["speaker"]:
                        if current:
                            speaker_turns_aligned.append(current)
                        current = {
                            "speaker": s["speaker"],
                            "start": s["start"],
                            "end": s["end"],
                            "text": s["text"],
                        }
                    else:
                        current["end"] = s["end"]
                        current["text"] += " " + s["text"]
                if current:
                    speaker_turns_aligned.append(current)

                # ── Word-count merge: reassign fragment speakers ──
                # A fragment is a speaker with very few total words
                # (just "Yeah", "Right", "Okay" responses). Reassign
                # to the temporally nearest main speaker.
                from collections import Counter
                turn_word_counts = Counter()
                for t in speaker_turns_aligned:
                    turn_word_counts[t["speaker"]] += len(t["text"].split())
                
                total_words = sum(turn_word_counts.values())
                # A speaker is a fragment if they have very few total words
                # AND a tiny proportion of the conversation — catches cases
                # where the same person was split into a "yeah/okay" cluster
                # by the diarization (e.g. Speaker_09 with 45 words / 4.7%).
                main_speakers = {sp for sp, wc in turn_word_counts.items() 
                                 if not (wc < 50 and total_words > 0 and wc / total_words < 0.08)}
                fragment_speakers = set(turn_word_counts.keys()) - main_speakers
                
                if fragment_speakers and main_speakers:
                    logger.info(f"Word-count merge: {len(fragment_speakers)} fragment speakers → {len(main_speakers)} main speakers")
                    merged = []
                    for t in speaker_turns_aligned:
                        if t["speaker"] in fragment_speakers:
                            mid = (t["start"] + t["end"]) / 2
                            closest = min(main_speakers, key=lambda sp: min(
                                abs((ot["start"] + ot["end"]) / 2 - mid)
                                for ot in speaker_turns_aligned if ot["speaker"] == sp
                            ))
                            t["speaker"] = closest
                        merged.append(t)
                    speaker_turns_aligned = merged
                    logger.info(f"Word-count merge: now {len(set(t['speaker'] for t in speaker_turns_aligned))} speakers")

                full_text = "\n".join(
                    f"[{t['speaker']}] {t['text']}" for t in speaker_turns_aligned
                )

                result = {
                    "status": "completed",
                    "engine": "parakeet-mlx (Mac Mini)",
                    "sentences": aligned_sentences,
                    "speaker_turns": speaker_turns_aligned,
                    "full_text": full_text,
                    "diarization_segments": diarization_segments,
                    "num_speakers": len(set(s["speaker"] for s in aligned_sentences)),
                    "duration_seconds": asr_result.get("duration_seconds", 0),
                    "audio_duration_seconds": asr_result.get("audio_duration_seconds", 0),
                    "speaker_library": library_speakers,
                    "completed_at": time.time(),
                }
            else:
                # ASR failed, fall back
                logger.warning(f"Mac ASR failed, falling back to LemonFox")
                lemo_result = _transcribe_via_lemonfox(processed_path)
                if lemo_result.get("status") == "completed":
                    lemo_result["speaker_turns"] = speaker_turns
                    lemo_result["diarization_segments"] = diarization_segments
                    lemo_result["speaker_library"] = library_speakers
                result = lemo_result
        else:
            logger.warning("Mac unreachable, falling back to LemonFox")
            lemo_result = _transcribe_via_lemonfox(processed_path)
            if lemo_result.get("status") == "completed":
                lemo_result["speaker_turns"] = speaker_turns
                lemo_result["diarization_segments"] = diarization_segments
                lemo_result["speaker_library"] = library_speakers
            result = lemo_result

        # Clean up pre-processed file
        try:
            os.unlink(processed_path)
        except OSError:
            pass

        # Store result
        with open(_result_path(filename), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        _status_path(filename).unlink(missing_ok=True)
        return result

    except Exception as e:
        logger.error(f"Transcription pipeline failed: {e}", exc_info=True)
        error_result = {"status": "error", "error": str(e)}
        with open(_result_path(filename), "w") as f:
            json.dump(error_result, f, indent=2)
        return error_result


def _write_status(filename: str, data: dict):
    with open(_status_path(filename), "w") as f:
        json.dump(data, f)


def list_transcriptions() -> list[dict]:
    results = []
    if not RECORDINGS_DIR.exists():
        return results
    for f in sorted(RECORDINGS_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in {
            ".mp3", ".wav", ".flac", ".ogg", ".aac",
            ".m4a", ".opus", ".wma", ".aiff",
        }:
            status = get_status(f.name)
            results.append({"name": f.name, "status": status["status"]})
    return results

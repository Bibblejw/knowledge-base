"""
Pipeline — standalone composable stages for the transcription pipeline.

Each stage is a self-contained function that can be called independently.
Intermediate outputs are cached as JSON/status files so stages compose.

Use cases:
  pipeline/preprocess/<file>   → just denoise + VAD + normalize
  pipeline/diarize/<file>      → speaker segmentation + matching
  pipeline/transcribe/<file>   → ASR only (no speaker labels)
  pipeline/align/<file>        → map ASR text to diarization segments
  pipeline/postprocess/<file>  → glossary + speaker clues + engagement
  pipeline/run/<file>          → full pipeline (all stages)
  pipeline/run/<file>?stages=preprocess,diarize  → partial run
"""

import json
import os
import sys
import time
import threading
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/opt/knowledge-base")

import logging
import requests

logger = logging.getLogger("pipeline")

from diarization import (
    preprocess_audio,
    diarize_audio,
    align_asr_with_diarization,
    _load_speaker_library,
)
from post_process import apply_glossary
from speaker_clues import annotate_result as apply_speaker_clues
from engagement import annotate_result as apply_engagement

RECORDINGS_DIR = Path("/recordings")
TRANSCRIPTIONS_DIR = RECORDINGS_DIR / ".transcriptions"
MAC_ASR_URL = "http://10.3.0.207:5001/transcribe"
LEMONFOX_API = "https://api.lemonfox.ai/v1/audio/transcriptions"


# ── Internal helpers ───────────────────────────────────────────────────

def _res_path(filename: str) -> Path:
    """Path to the final result JSON."""
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return TRANSCRIPTIONS_DIR / f"{filename}.result.json"


def _stage_path(filename: str, stage: str) -> Path:
    """Path for a cached intermediate stage output."""
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return TRANSCRIPTIONS_DIR / f"{filename}.stage.{stage}.json"


def _write_intermediate(filename: str, stage: str, data: dict):
    """Save an intermediate stage result to disk."""
    path = _stage_path(filename, stage)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _read_intermediate(filename: str, stage: str) -> dict | None:
    """Read a cached intermediate stage result, or None."""
    path = _stage_path(filename, stage)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _clear_intermediate(filename: str, stage: str = None):
    """Remove intermediate files for filename. Stage=None clears all."""
    if stage:
        _stage_path(filename, stage).unlink(missing_ok=True)
    else:
        for p in TRANSCRIPTIONS_DIR.glob(f"{filename}.stage.*"):
            p.unlink(missing_ok=True)


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
        resp = requests.get(
            f"{MAC_ASR_URL.replace('/transcribe', '/health')}", timeout=3
        )
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def _transcribe_via_mac(audio_path: str) -> dict | None:
    """Send pre-processed audio to Mac for Parakeet ASR."""
    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                MAC_ASR_URL,
                files={"audio": ("audio.wav", f, "audio/wav")},
                timeout=600,
            )
        if resp.status_code != 200:
            logger.warning(f"Mac ASR error ({resp.status_code}): {resp.text[:200]}")
            return None
        return resp.json()
    except Exception as e:
        logger.warning(f"Mac ASR failed: {e}")
        return None


def _transcribe_via_lemonfox(audio_path: str) -> dict:
    """Fallback ASR via LemonFox API."""
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
        return {
            "status": "error",
            "error": f"LemonFox API error ({resp.status_code}): {resp.text[:500]}",
        }

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


def _merge_fragment_speakers(speaker_turns_aligned: list) -> list:
    """Reassign fragment speakers (very few words) to nearest main speaker."""
    turn_word_counts = Counter()
    for t in speaker_turns_aligned:
        turn_word_counts[t["speaker"]] += len(t["text"].split())

    total_words = sum(turn_word_counts.values())
    main_speakers = {
        sp
        for sp, wc in turn_word_counts.items()
        if not (wc < 50 and total_words > 0 and wc / total_words < 0.08)
    }
    fragment_speakers = set(turn_word_counts.keys()) - main_speakers

    if fragment_speakers and main_speakers:
        logger.info(
            f"Fragment merge: {len(fragment_speakers)} → {len(main_speakers)}"
        )
        merged = []
        for t in speaker_turns_aligned:
            if t["speaker"] in fragment_speakers:
                mid = (t["start"] + t["end"]) / 2
                closest = min(
                    main_speakers,
                    key=lambda sp: min(
                        abs((ot["start"] + ot["end"]) / 2 - mid)
                        for ot in speaker_turns_aligned
                        if ot["speaker"] == sp
                    ),
                )
                t["speaker"] = closest
            merged.append(t)
        speaker_turns_aligned = merged

    return speaker_turns_aligned


def _build_aligned_turns(aligned_sentences: list) -> list:
    """Build speaker_turns from aligned sentence list."""
    turns = []
    current = None
    for s in aligned_sentences:
        if current is None or s["speaker"] != current["speaker"]:
            if current:
                turns.append(current)
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
        turns.append(current)
    return turns


# ═══════════════════════════════════════════════════════════════════════
# Stage 1 — Pre-process
# ═══════════════════════════════════════════════════════════════════════

def stage_preprocess(filename: str) -> dict:
    """
    VAD + denoise + normalize audio. Returns dict with:
      { "status": "ok", "processed_path": "/tmp/...", "audio_path": "/recordings/..." }
    or error dict.
    """
    audio_path = RECORDINGS_DIR / filename
    if not audio_path.exists():
        return {"status": "error", "error": "file_not_found", "filename": filename}

    logger.info(f"Pre-processing {filename}...")
    try:
        processed = preprocess_audio(str(audio_path))
        result = {
            "status": "ok",
            "stage": "preprocess",
            "filename": filename,
            "audio_path": str(audio_path),
            "processed_path": processed,
        }
        _write_intermediate(filename, "preprocess", result)
        logger.info(f"Pre-processing complete → {processed}")
        return result
    except Exception as e:
        logger.error(f"Pre-processing failed: {e}")
        return {"status": "error", "stage": "preprocess", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# Stage 2 — Diarize
# ═══════════════════════════════════════════════════════════════════════

def stage_diarize(filename: str, *, threshold: float = 0.65) -> dict:
    """
    Speaker segmentation + library matching. Requires stage_preprocess.
    Returns dict with diarization_segments, speaker_turns, etc.
    """
    pre = _read_intermediate(filename, "preprocess") or stage_preprocess(filename)
    if pre.get("status") != "ok":
        return {"status": "error", "stage": "diarize", "error": pre.get("error", "preprocess_failed")}

    processed_path = pre["processed_path"]
    logger.info(f"Diarizing {filename}...")
    try:
        segments, turns = diarize_audio(processed_path, threshold=threshold)
        result = {
            "status": "ok",
            "stage": "diarize",
            "filename": filename,
            "processed_path": processed_path,
            "diarization_segments": segments,
            "speaker_turns": turns,
            "num_speakers_raw": len({s.get("speaker") for s in segments}),
        }
        _write_intermediate(filename, "diarize", result)
        logger.info(f"Diarization complete → {len(segments)} segments, {result['num_speakers_raw']} speakers")
        return result
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        return {"status": "error", "stage": "diarize", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# Stage 3 — ASR (transcribe audio to text)
# ═══════════════════════════════════════════════════════════════════════

def stage_asr(filename: str, *, engine: str = "auto") -> dict:
    """
    Run ASR on pre-processed audio. Engine: "auto" (try Mac, fallback LemonFox),
    "mac", or "lemonfox". Requires stage_preprocess.
    """
    pre = _read_intermediate(filename, "preprocess") or stage_preprocess(filename)
    if pre.get("status") != "ok":
        return {"status": "error", "stage": "asr", "error": pre.get("error", "preprocess_failed")}

    processed_path = pre["processed_path"]
    logger.info(f"ASR for {filename} (engine={engine})...")

    try:
        asr_result = None
        used_engine = None

        if engine in ("auto", "mac"):
            if _is_mac_alive():
                logger.info("Mac alive → Parakeet ASR")
                asr_result = _transcribe_via_mac(processed_path)
                used_engine = "parakeet-mlx"

        if asr_result is None and engine in ("auto", "lemonfox"):
            logger.info("→ LemonFox fallback")
            asr_result = _transcribe_via_lemonfox(processed_path)
            used_engine = "lemonfox-ai"

        if asr_result is None or asr_result.get("status") == "error":
            return {
                "status": "error",
                "stage": "asr",
                "error": asr_result.get("error", "asr_failed") if asr_result else "no_asr_engine",
            }

        full_text = asr_result.get("full_text", "")
        sentences = asr_result.get("sentences", [])
        duration = asr_result.get("duration_seconds", asr_result.get("audio_duration_seconds", 0))

        result = {
            "status": "ok",
            "stage": "asr",
            "filename": filename,
            "processed_path": processed_path,
            "engine": used_engine,
            "sentences": sentences,
            "full_text": full_text,
            "audio_duration_seconds": duration or asr_result.get("audio_duration_seconds", 0),
            "duration_seconds": asr_result.get("duration_seconds", 0),
            "asr_raw": asr_result,
        }
        _write_intermediate(filename, "asr", result)
        logger.info(f"ASR complete → {len(sentences)} sentences, {len(full_text)} chars")
        return result

    except Exception as e:
        logger.error(f"ASR failed: {e}")
        return {"status": "error", "stage": "asr", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# Stage 4 — Align (map ASR text to diarization segments)
# ═══════════════════════════════════════════════════════════════════════

def stage_align(filename: str) -> dict:
    """
    Align ASR output to diarization segments. Requires stage_asr and stage_diarize.
    Produces the full result dict with speaker_turns, sentences, full_text, etc.
    """
    asr_stage = _read_intermediate(filename, "asr")
    if not asr_stage or asr_stage.get("status") != "ok":
        asr_stage = stage_asr(filename)
        if asr_stage.get("status") != "ok":
            return {"status": "error", "stage": "align", "error": asr_stage.get("error", "asr_failed")}

    dia_stage = _read_intermediate(filename, "diarize")
    if not dia_stage or dia_stage.get("status") != "ok":
        dia_stage = stage_diarize(filename)
        if dia_stage.get("status") != "ok":
            return {"status": "error", "stage": "align", "error": dia_stage.get("error", "diarize_failed")}

    logger.info(f"Aligning {filename}...")
    try:
        asr_raw = asr_stage.get("asr_raw", asr_stage)
        diarization_segments = dia_stage["diarization_segments"]

        aligned_sentences = align_asr_with_diarization(asr_raw, diarization_segments)
        speaker_turns_aligned = _build_aligned_turns(aligned_sentences)
        speaker_turns_aligned = _merge_fragment_speakers(speaker_turns_aligned)

        full_text = "\n".join(
            f"[{t['speaker']}] {t['text']}" for t in speaker_turns_aligned
        )

        library_speakers = list(_load_speaker_library().keys())

        result = {
            "status": "ok",
            "stage": "align",
            "filename": filename,
            "engine": asr_stage.get("engine", "unknown"),
            "sentences": aligned_sentences,
            "speaker_turns": speaker_turns_aligned,
            "full_text": full_text,
            "diarization_segments": diarization_segments,
            "num_speakers": len({s["speaker"] for s in aligned_sentences}),
            "audio_duration_seconds": asr_stage.get("audio_duration_seconds", 0),
            "duration_seconds": asr_stage.get("duration_seconds", 0),
            "speaker_library": library_speakers,
            "completed_at": time.time(),
        }
        _write_intermediate(filename, "align", result)
        logger.info(f"Align complete → {len(aligned_sentences)} sentences, {result['num_speakers']} speakers")
        return result

    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        return {"status": "error", "stage": "align", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# Stage 5 — Post-process (glossary → speaker clues → engagement)
# ═══════════════════════════════════════════════════════════════════════

def stage_postprocess(filename: str) -> dict:
    """
    Apply glossary corrections, speaker clue auto-annotation, and
    engagement (audio prosody) analysis. Requires stage_align.
    Saves the final result.json.
    """
    align_stage = _read_intermediate(filename, "align")
    if not align_stage or align_stage.get("status") != "ok":
        align_stage = stage_align(filename)
        if align_stage.get("status") != "ok":
            return {"status": "error", "stage": "postprocess", "error": align_stage.get("error", "align_failed")}

    logger.info(f"Post-processing {filename}...")
    try:
        audio_path = RECORDINGS_DIR / filename
        result = dict(align_stage)
        result["status"] = "completed"

        apply_glossary(result)
        logger.info("Glossary applied")

        apply_speaker_clues(result)
        logger.info("Speaker clues applied")

        apply_engagement(str(audio_path), result)
        logger.info("Engagement analysis applied")

        _clear_intermediate(filename)

        with open(_res_path(filename), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Result saved → {filename}.result.json")

        return result

    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        return {"status": "error", "stage": "postprocess", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# Full pipeline runner
# ═══════════════════════════════════════════════════════════════════════

STAGE_ORDER = ["preprocess", "diarize", "asr", "align", "postprocess"]
STAGE_FUNCS = {
    "preprocess": stage_preprocess,
    "diarize": stage_diarize,
    "asr": stage_asr,
    "align": stage_align,
    "postprocess": stage_postprocess,
}


def run_pipeline(filename: str, stages: list[str] = None) -> dict:
    """
    Run selected stages of the pipeline. stages=None runs all stages.
    Stages are run in dependency order.
    Returns the last stage's result dict.
    """
    if stages is None:
        stages = list(STAGE_ORDER)

    if not stages:
        return {"status": "error", "error": "no_stages_specified", "filename": filename}

    ordered = [s for s in STAGE_ORDER if s in stages]
    for s in stages:
        if s not in STAGE_ORDER:
            return {"status": "error", "error": f"unknown_stage:{s}", "filename": filename}

    logger.info(f"Pipeline run for {filename}: stages={ordered}")

    for stage in ordered:
        fn = STAGE_FUNCS[stage]
        result = fn(filename)
        if result.get("status") in ("error",):
            return result

    last_stage = ordered[-1]
    stage_data = _read_intermediate(filename, last_stage)
    if stage_data:
        return stage_data
    # Fallback: check if final result was written
    res = _res_path(filename)
    if res.exists():
        with open(res) as f:
            return json.load(f)
    return {"status": "error", "error": "pipeline_completed_but_result_not_found"}


def get_pipeline_status(filename: str) -> dict:
    """Check which stages have been completed for a filename."""
    status = {"filename": filename, "stages": {}}
    for stage in STAGE_ORDER:
        cached = _read_intermediate(filename, stage)
        if cached and cached.get("status") == "ok":
            status["stages"][stage] = "completed"
        else:
            status["stages"][stage] = "pending"

    final = _res_path(filename)
    status["result_exists"] = final.exists()
    if final.exists():
        with open(final) as f:
            result = json.load(f)
        status["result_status"] = result.get("status", "unknown")
    else:
        status["result_status"] = None

    return status

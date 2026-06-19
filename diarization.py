"""
Diarization module — SpeechBrain ECAPA embeddings + sklearn clustering.
Runs on the Knowledge Base container (LXC 113).
Handles VAD -> embed -> cluster -> speaker match -> align with ASR timestamps.
"""
import os
import json
import pickle
import time
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import noisereduce as nr
import pyloudnorm as pyln
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger('diarization')

# --- Config ---
SPEAKER_LIBRARY_PATH = "/recordings/.speaker_library.pkl"
MAC_ASR_URL = "http://10.3.0.207:5001/transcribe"

# --- Global caches ---
_embedding_model = None
_silero_vad = None


# ============================================================
# SPEAKER LIBRARY
# ============================================================

def _load_speaker_library():
    if os.path.exists(SPEAKER_LIBRARY_PATH):
        with open(SPEAKER_LIBRARY_PATH, 'rb') as f:
            return pickle.load(f)
    return {}  # {name: numpy_embedding}


def _save_speaker_library(lib):
    os.makedirs(os.path.dirname(SPEAKER_LIBRARY_PATH), exist_ok=True)
    with open(SPEAKER_LIBRARY_PATH, 'wb') as f:
        pickle.dump(lib, f)


def _match_speaker(embedding, library, threshold=0.65):
    """Find best match for embedding in library. Returns (name, score) or (None, best_score)."""
    if not library:
        return None, 0.0
    best_name = None
    best_score = 0.0
    emb = np.array(embedding).reshape(1, -1)
    for name, ref_emb in library.items():
        ref = np.array(ref_emb).reshape(1, -1)
        sim = cosine_similarity(emb, ref)[0][0]
        if sim > best_score:
            best_score = sim
            best_name = name
    if best_score >= threshold:
        return best_name, float(best_score)
    return None, float(best_score)


# ============================================================
# EMBEDDING MODEL
# ============================================================

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading SpeechBrain ECAPA embedding model...")
        from speechbrain.inference.speaker import EncoderClassifier
        _embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/recordings/.model_cache/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
        logger.info("ECAPA model loaded.")
    return _embedding_model


def _get_silero_vad():
    """Get Silero VAD model (single-use, loaded on demand)."""
    import torch
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    return model, utils


def _extract_embedding(audio_path: str) -> np.ndarray:
    """Extract speaker embedding from an audio file. Returns 192-dim vector."""
    model = _get_embedding_model()
    import librosa
    import torch
    signal, fs = librosa.load(audio_path, sr=16000)
    waveform = torch.from_numpy(signal).unsqueeze(0).float()
    embedding = model.encode_batch(waveform)
    return embedding.squeeze().numpy()


# ============================================================
# PRE-PROCESSING
# ============================================================

def preprocess_audio(input_path: str) -> str:
    """
    VAD (Silero) + Denoise + Normalize.
    Uses ffmpeg for fast format conversion, then silero-vad for voice detection.
    Returns path to processed WAV (16kHz, mono).
    """
    import torch

    tmp_wav = input_path + ".preprocessed.wav"
    tmp_raw = input_path + ".raw16k.wav"

    # Step 1: Convert to 16kHz mono WAV with ffmpeg (fast)
    logger.info(f"Converting {input_path} to 16kHz mono...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-ar", "16000", "-ac", "1",
         "-sample_fmt", "s16", tmp_raw,
         "-loglevel", "error"],
        check=True, capture_output=True,
    )

    # Step 2: Load into numpy for processing
    audio_np, sr = sf.read(tmp_raw)
    original_duration = len(audio_np) / sr
    logger.info(f"Loaded {original_duration:.1f}s at {sr}Hz")

    # Step 3: Silero VAD — get speech timestamps
    logger.info("Running Silero VAD...")
    model, (get_speech_timestamps, _, _, _, _) = _get_silero_vad()

    # VAD works on tensor audio
    audio_tensor = torch.from_numpy(audio_np).float()
    speech_ts = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        window_size_samples=512,
    )

    if speech_ts:
        # Concatenate speech segments
        speech_segments = []
        for ts in speech_ts:
            start = ts['start']
            end = ts['end']
            speech_segments.append(audio_np[start:end])
        audio_np = np.concatenate(speech_segments)

    vad_duration = len(audio_np) / sr
    saved = (1 - vad_duration / original_duration) * 100
    logger.info(f"VAD: {original_duration:.1f}s -> {vad_duration:.1f}s ({saved:.0f}% removed, {len(speech_ts)} segments)")
    sf.write(tmp_raw, audio_np, sr)

    # Step 4: Denoise
    logger.info("Denoising...")
    audio_data, sr_denoise = sf.read(tmp_raw)
    reduced = nr.reduce_noise(y=audio_data, sr=sr_denoise, stationary=True, prop_decrease=0.8)
    sf.write(tmp_raw, reduced, sr_denoise)

    # Step 5: Normalize volume
    logger.info("Normalizing volume...")
    audio_data, sr_norm = sf.read(tmp_raw)
    meter = pyln.Meter(sr_norm)
    loudness = meter.integrated_loudness(audio_data)
    normalized = pyln.normalize.loudness(audio_data, loudness, -16.0)
    sf.write(tmp_wav, normalized, sr_norm)

    # Clean up intermediate
    os.unlink(tmp_raw)

    return tmp_wav


# ============================================================
# DIARIZATION
# ============================================================

def diarize_audio(audio_path: str, threshold: float = 0.65) -> dict:
    """
    Full diarization pipeline:
    1. Break audio into short segments (2s windows, 1s stride)
    2. Extract ECAPA embeddings per segment
    3. Cluster segments by speaker
    4. Match clusters against speaker library
    5. Return speaker-labeled segments

    Returns:
        segments: list of {start, end, speaker, speaker_raw} per 2s window
        speaker_turns: merged contiguous same-speaker segments
    """
    import librosa
    import torch

    logger.info("Loading audio for diarization...")
    audio_np, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio_np) / sr

    # Use 2s windows with 1s stride
    window = 2.0
    stride = 1.0
    window_samples = int(window * sr)
    stride_samples = int(stride * sr)

    embeddings = []
    timestamps = []

    model = _get_embedding_model()

    for start_sample in range(0, len(audio_np) - window_samples + 1, stride_samples):
        end_sample = start_sample + window_samples
        segment = audio_np[start_sample:end_sample]

        # Extract embedding
        waveform = torch.from_numpy(segment).unsqueeze(0).float()
        emb = model.encode_batch(waveform)
        embeddings.append(emb.squeeze().numpy())

        start_time = start_sample / sr
        end_time = end_sample / sr
        timestamps.append({"start": round(start_time, 2), "end": round(end_time, 2)})

    logger.info(f"Extracted {len(embeddings)} embeddings from {duration:.0f}s audio")

    if len(embeddings) < 2:
        # Too short for clustering
        return [{"start": 0, "end": round(duration, 2), "speaker": "UNKNOWN", "speaker_raw": "UNKNOWN"}]

    # Cluster embeddings
    embeddings_np = np.array(embeddings)
    n_clusters = min(8, max(1, len(embeddings) // 10))  # heuristic

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average"
    )
    labels = clustering.fit_predict(embeddings_np)

    # Get mean embedding per cluster for speaker matching
    unique_labels = sorted(set(labels))
    cluster_embeddings = {}
    for label in unique_labels:
        mask = labels == label
        cluster_embeddings[label] = embeddings_np[mask].mean(axis=0)

    # Match clusters against library
    library = _load_speaker_library()
    cluster_names = {}
    for label, emb in cluster_embeddings.items():
        name, score = _match_speaker(emb, library, threshold=threshold)
        if name:
            cluster_names[label] = name
            logger.info(f"Cluster {label} matched -> {name} ({score:.3f})")
        else:
            label_name = f"Speaker_{label + 1:02d}"
            cluster_names[label] = label_name
            logger.info(f"Cluster {label} -> {label_name} (no match, score={score:.3f})")

    # Build segments
    segments = []
    for i, label in enumerate(labels):
        segments.append({
            "start": timestamps[i]["start"],
            "end": timestamps[i]["end"],
            "speaker_raw": f"SPEAKER_{label:02d}",
            "speaker": cluster_names[label],
        })

    # Merge contiguous same-speaker into turns
    speaker_turns = []
    current = None
    for seg in segments:
        if current is None or seg["speaker"] != current["speaker"]:
            if current:
                speaker_turns.append(current)
            current = {
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
            }
        else:
            current["end"] = seg["end"]
    if current:
        speaker_turns.append(current)

    num_speakers = len(set(s["speaker"] for s in segments))
    logger.info(f"Diarization done: {len(segments)} segments, {num_speakers} speakers")
    return segments, speaker_turns


# ============================================================
# ALIGNMENT
# ============================================================

def align_asr_with_diarization(asr_result: dict, diarization_segments: list) -> list:
    """
    Align Parakeet ASR sentence-level timestamps with diarization speaker labels.
    Each ASR sentence gets the speaker label from whichever diarization segment
    has the most temporal overlap.

    Returns list of {text, start, end, speaker, tokens}
    """
    aligned = []
    for s in asr_result.get("sentences", []):
        s_start = s["start"]
        s_end = s["end"]

        # Find diarization segment with most overlap
        best_speaker = "UNKNOWN"
        best_overlap = 0

        for ds in diarization_segments:
            overlap_start = max(s_start, ds["start"])
            overlap_end = min(s_end, ds["end"])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = ds["speaker"]

        aligned.append({
            "text": s["text"],
            "start": s_start,
            "end": s_end,
            "speaker": best_speaker,
            "tokens": s.get("tokens", []),
        })

    return aligned


# ============================================================
# SPEAKER ENROLLMENT
# ============================================================

def enroll_speaker(name: str, audio_path: str) -> dict:
    """Enroll a speaker from a voice clip (20-60s)."""
    # Preprocess the clip
    processed_path = preprocess_audio(audio_path)

    # Extract embedding
    embedding = _extract_embedding(processed_path)

    # Save to library
    library = _load_speaker_library()
    library[name] = embedding
    _save_speaker_library(library)

    os.unlink(processed_path)

    return {
        "status": "enrolled",
        "name": name,
        "total_speakers": len(library),
        "speakers": list(library.keys()),
    }


def list_speakers() -> list:
    return list(_load_speaker_library().keys())


def remove_speaker(name: str) -> bool:
    library = _load_speaker_library()
    if name in library:
        del library[name]
        _save_speaker_library(library)
        return True
    return False

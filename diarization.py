"""
Diarization module — SpeechBrain ECAPA embeddings + sklearn clustering.
Runs on the Knowledge Base container (LXC 113).
Handles VAD -> embed -> cluster -> speaker match -> align with ASR timestamps.

Cross-recording speaker matching:
- After clustering each recording, cluster mean embeddings are stored in a
  global discovered-speaker library.
- When a new recording is diarized, unknown clusters are matched against
  this global library first (threshold 0.60), then against user-enrolled
  speakers (threshold 0.65, higher priority).
- This gives consistent speaker labels across recordings without manual enrollment.
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
GLOBAL_SPEAKER_PATH = "/recordings/.global_speaker_library.pkl"
GLOBAL_COUNTER_PATH = "/recordings/.global_speaker_counter.txt"
MAC_ASR_URL = "http://10.3.0.207:5001/transcribe"

# Matching thresholds
ENROLLED_THRESHOLD = 0.65   # user-enrolled: high confidence
GLOBAL_THRESHOLD = 0.60     # cross-recording: slightly lower

# --- Global caches ---
_embedding_model = None
_silero_vad = None


# ============================================================
# SPEAKER LIBRARY — enrolled (user-provided)
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
# GLOBAL SPEAKER LIBRARY — cross-recording discovered speakers
# ============================================================

def _load_global_library():
    """Load cross-recording discovered speaker library.
    
    Format: {global_name: {"embedding": np.ndarray, "recordings": [str, ...], 
                            "first_seen": float, "last_seen": float}}
    """
    if os.path.exists(GLOBAL_SPEAKER_PATH):
        with open(GLOBAL_SPEAKER_PATH, 'rb') as f:
            return pickle.load(f)
    return {}


def _save_global_library(lib):
    os.makedirs(os.path.dirname(GLOBAL_SPEAKER_PATH), exist_ok=True)
    with open(GLOBAL_SPEAKER_PATH, 'wb') as f:
        pickle.dump(lib, f)


def _get_global_counter() -> int:
    """Get the next available global speaker ID."""
    if os.path.exists(GLOBAL_COUNTER_PATH):
        with open(GLOBAL_COUNTER_PATH) as f:
            return int(f.read().strip())
    return 1


def _increment_global_counter() -> int:
    """Increment and return the next global speaker ID."""
    counter = _get_global_counter() + 1
    os.makedirs(os.path.dirname(GLOBAL_COUNTER_PATH), exist_ok=True)
    with open(GLOBAL_COUNTER_PATH, 'w') as f:
        f.write(str(counter))
    return counter - 1  # return the ID that was just consumed


def _reset_global_counter():
    """Reset the global counter (for rebuilding)."""
    os.makedirs(os.path.dirname(GLOBAL_COUNTER_PATH), exist_ok=True)
    with open(GLOBAL_COUNTER_PATH, 'w') as f:
        f.write("1")


def _next_global_name() -> str:
    """Generate the next global speaker name, e.g. Speaker_07."""
    next_id = _get_global_counter()
    _increment_global_counter()
    return f"Speaker_{next_id:02d}"


def _match_global(embedding, global_lib, threshold=GLOBAL_THRESHOLD):
    """Find best match in global library. Returns (name, score) or (None, best_score)."""
    return _match_speaker(embedding, {k: v["embedding"] for k, v in global_lib.items()}, threshold=threshold)


def _update_global_library(cluster_embeddings, cluster_names, recording_name: str, global_lib: dict):
    """Update the global library with new/refined speaker embeddings.
    
    - For existing global speakers: refine the embedding (running average) and add recording.
    - For newly assigned speakers: add a fresh entry.
    """
    now = time.time()
    for label, name in cluster_names.items():
        emb = cluster_embeddings[label]
        if name in global_lib:
            # Refine embedding: running average weighted by recording count
            entry = global_lib[name]
            n = len(entry.get("recordings", []))
            if n > 0:
                # Weighted update: new_avg = (old_avg * n + new_emb) / (n + 1)
                entry["embedding"] = (entry["embedding"] * n + emb) / (n + 1)
            entry["last_seen"] = now
            if recording_name not in entry["recordings"]:
                entry["recordings"].append(recording_name)
        else:
            # New speaker: add to global library
            global_lib[name] = {
                "embedding": emb,
                "recordings": [recording_name],
                "first_seen": now,
                "last_seen": now,
            }
    _save_global_library(global_lib)


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
# DIARIZATION (with cross-recording matching)
# ============================================================

def diarize_audio(audio_path: str, threshold: float = ENROLLED_THRESHOLD) -> dict:
    """
    Full diarization pipeline:
    1. Break audio into short segments (2s windows, 1s stride)
    2. Extract ECAPA embeddings per segment
    3. Cluster segments by speaker (agglomerative clustering)
    4. Match clusters against global + enrolled speaker libraries
    5. Update global library with new/refined embeddings
    6. Return speaker-labeled segments

    Cross-recording matching order:
      a. Enrolled speakers (user-provided, threshold 0.65)
      b. Global discovered speakers (auto-built, threshold 0.60)
      c. If neither matches → assign new global ID

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

    # ── Cross-recording speaker matching ──────────────────────────────
    # Priority: 1) enrolled speakers, 2) global discovered, 3) new ID
    enrolled_lib = _load_speaker_library()
    global_lib = _load_global_library()

    cluster_names = {}
    for label, emb in cluster_embeddings.items():
        # First: try enrolled speakers (highest confidence)
        enrolled_name, enrolled_score = _match_speaker(emb, enrolled_lib, threshold=ENROLLED_THRESHOLD)
        if enrolled_name:
            cluster_names[label] = enrolled_name
            logger.info(f"Cluster {label} → enrolled [{enrolled_name}] ({enrolled_score:.3f})")
            continue

        # Second: try global discovered library (cross-recording match)
        global_name, global_score = _match_global(emb, global_lib, threshold=GLOBAL_THRESHOLD)
        if global_name:
            cluster_names[label] = global_name
            logger.info(f"Cluster {label} → global [{global_name}] ({global_score:.3f})")
            continue

        # Third: no match — assign new global ID
        new_name = _next_global_name()
        cluster_names[label] = new_name
        logger.info(f"Cluster {label} → NEW [{new_name}] (no match, best={global_score:.3f})")

    # ── Update global library with this recording's clusters ──────────
    # Extract recording name from path for tracking
    recording_name = os.path.basename(audio_path)
    if recording_name.endswith(".preprocessed.wav"):
        recording_name = recording_name[:-len(".preprocessed.wav")]
    _update_global_library(cluster_embeddings, cluster_names, recording_name, global_lib)

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
# GLOBAL LIBRARY MANAGEMENT
# ============================================================

def list_global_speakers() -> dict:
    """List all speakers in the global discovered library."""
    lib = _load_global_library()
    return {
        name: {
            "recordings": info["recordings"],
            "first_seen": info.get("first_seen", 0),
            "last_seen": info.get("last_seen", 0),
        }
        for name, info in lib.items()
    }


def rebuild_global_library():
    """Rebuild the global speaker library from all existing transcriptions.
    
    Useful after renaming speakers or when the library gets stale.
    Scans all completed transcription results and re-extracts cluster embeddings.
    """
    logger.info("Rebuilding global speaker library from all transcriptions...")
    
    # Reset counter
    _reset_global_counter()
    
    # Find all completed transcriptions
    trans_dir = Path("/recordings/.transcriptions")
    if not trans_dir.exists():
        logger.info("No transcriptions found, library empty")
        _save_global_library({})
        return {"status": "empty", "message": "No transcriptions found"}
    
    new_lib = {}
    model = _get_embedding_model()
    import torch

    for result_file in sorted(trans_dir.glob("*.result.json")):
        try:
            with open(result_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping {result_file}: {e}")
            continue

        if data.get("status") != "completed":
            continue

        # Extract diarization segments — use speaker_raw to get per-cluster info
        segs = data.get("diarization_segments", [])
        if not segs:
            continue

        # Group segments by speaker_raw to rebuild cluster embeddings
        recording_name = result_file.stem.replace(".result", "")
        raw_clusters = {}
        for s in segs:
            raw = s.get("speaker_raw", "UNKNOWN")
            if raw not in raw_clusters:
                raw_clusters[raw] = {"speaker": s.get("speaker", raw), "count": 0, "segments": []}
            raw_clusters[raw]["count"] += 1
            raw_clusters[raw]["segments"].append({
                "start": s["start"],
                "end": s["end"],
            })

        for raw_key, cluster_info in raw_clusters.items():
            speaker_name = cluster_info["speaker"]
            # If this speaker already exists in new_lib, merge
            if speaker_name in new_lib:
                if recording_name not in new_lib[speaker_name]["recordings"]:
                    new_lib[speaker_name]["recordings"].append(recording_name)
                continue

            # Try to extract an embedding from this speaker's audio
            # We use the first few segments to compute a mean embedding
            preprocessed = str(Path("/recordings") / recording_name) + ".preprocessed.wav"
            if not os.path.exists(preprocessed):
                continue
                
            try:
                audio_np, sr = sf.read(preprocessed)
            except Exception as e:
                logger.warning(f"Cannot read {preprocessed}: {e}")
                continue

            # Sample embeddings from this speaker's segments
            seg_embeddings = []
            for seg in cluster_info["segments"][:10]:  # up to 10 segments per speaker
                start_s = int(seg["start"] * sr)
                end_s = int(seg["end"] * sr)
                if end_s - start_s < sr:  # less than 1s, skip
                    continue
                segment = audio_np[start_s:end_s]
                waveform = torch.from_numpy(segment).unsqueeze(0).float()
                try:
                    emb = model.encode_batch(waveform)
                    seg_embeddings.append(emb.squeeze().numpy())
                except Exception:
                    continue

            if seg_embeddings:
                # Use mean embedding for this speaker
                new_lib[speaker_name] = {
                    "embedding": np.mean(seg_embeddings, axis=0),
                    "recordings": [recording_name],
                    "first_seen": data.get("completed_at", time.time()),
                    "last_seen": data.get("completed_at", time.time()),
                }
                logger.info(f"Rebuilt embedding for {speaker_name} from {recording_name}")

    _save_global_library(new_lib)
    _reset_global_counter()

    # Re-number to match what we just saved
    for i, name in enumerate(sorted(new_lib.keys()), 1):
        pass  # names stay as they were
    # Set counter past the current highest number
    max_num = 0
    for name in new_lib:
        if name.startswith("Speaker_"):
            try:
                n = int(name.split("_")[1])
                max_num = max(max_num, n)
            except (ValueError, IndexError):
                pass
    with open(GLOBAL_COUNTER_PATH, 'w') as f:
        f.write(str(max_num + 1))
    
    transcriptions = list(trans_dir.glob("*.result.json"))
    logger.info(f"Global library rebuilt: {len(new_lib)} speakers from {len(transcriptions)} transcriptions")
    return {"status": "rebuilt", "speaker_count": len(new_lib)}


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

"""
Engagement — Audio-based prosody and engagement analysis.

Extracts per-speaker metrics from the audio waveform using diarization
segments and ASR timestamps. Metrics include:

  - RMS energy (volume / emphasis)
  - Energy variation (dynamic range)
  - Speaking rate (words per second)
  - Pitch estimation (fundamental frequency, Hz)
  - Pitch variation
  - Spectral centroid (voice brightness / tension)

Usage:
  from engagement import extract_engagement
  engagement = extract_engagement(audio_path, result)
"""

import numpy as np
import soundfile as sf
import subprocess
import tempfile
import os
import logging

logger = logging.getLogger("engagement")

# ── Audio reading ───────────────────────────────────────────────────────────

def read_audio(audio_path):
    """Read audio file, returning (float64 mono array, sample_rate).
    Falls back to ffmpeg for M4A and other formats soundfile can't handle."""
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in (".wav", ".flac", ".ogg", ".opus"):
        try:
            audio, sr = sf.read(audio_path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            return audio.astype(np.float64), sr
        except Exception:
            pass

    # ffmpeg fallback
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        cmd = ["ffmpeg", "-y", "-i", audio_path,
               "-acodec", "pcm_s16le", "-ar", "16000",
               "-ac", "1", tmp_path, "-loglevel", "error"]
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        audio, sr = sf.read(tmp_path)
        os.unlink(tmp_path)
        return audio.astype(np.float64), sr
    except Exception as e:
        logger.error(f"Failed to read audio: {e}")
        raise


# ── Audio feature extraction ───────────────────────────────────────────────


def rms_energy(audio):
    """Root-mean-square energy, normalised to [0, 1]."""
    if len(audio) == 0:
        return 0.0
    rms = np.sqrt(np.mean(audio ** 2))
    return float(np.clip(rms * 5, 0.0, 1.0))


def spectral_centroid(audio, sr):
    """Weighted mean of frequencies (brightness). Returns Hz."""
    if len(audio) < 256:
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    total = np.sum(spectrum)
    if total == 0:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


def _frame_pitch(frame, sr):
    """Estimate pitch for a single short frame via autocorrelation (vectorised)."""
    frame = frame - np.mean(frame)
    n = len(frame)
    max_lag = min(int(sr / 50), n - 1)
    min_lag = min(int(sr / 400), max_lag - 1)
    if max_lag <= min_lag:
        return 0.0
    # Vectorised autocorrelation via numpy correlate
    corr = np.correlate(frame, frame, mode='full')[n - 1:] / n
    peak_idx = min_lag + int(np.argmax(corr[min_lag:max_lag]))
    if corr[peak_idx] < 0.3:
        return 0.0
    return float(sr / (peak_idx + 1))


def estimate_pitch(audio, sr):
    """Robust pitch estimation via frame-based autocorrelation. Returns median Hz."""
    frame_len = int(sr * 0.03)
    hop = int(sr * 0.01)
    if len(audio) < frame_len:
        return 0.0
    pitches = []
    for start in range(0, len(audio) - frame_len, hop):
        p = _frame_pitch(audio[start:start + frame_len], sr)
        if 50 < p < 400:
            pitches.append(p)
    return float(np.median(pitches)) if pitches else 0.0


# ── Per-segment analysis ────────────────────────────────────────────────────


def analyze_segment(audio_segment, sr):
    """Compute prosody features for one audio segment."""
    if len(audio_segment) < 256:
        return {"energy": 0.0, "energy_var": 0.0, "pitch": 0.0,
                "pitch_var": 0.0, "spectral_centroid": 0.0, "frame_count": 0}

    energy = rms_energy(audio_segment)
    fl = int(sr * 0.025)
    if len(audio_segment) >= fl * 2:
        frames = [rms_energy(audio_segment[i:i + fl])
                  for i in range(0, len(audio_segment) - fl, fl)]
        energy_var = float(np.std(frames)) if frames else 0.0
    else:
        energy_var = 0.0

    pitch = estimate_pitch(audio_segment, sr)
    pitch_var = 0.0
    if pitch > 0 and len(audio_segment) >= int(sr * 0.1):
        flp = int(sr * 0.03)
        sub = [audio_segment[i:i + flp]
               for i in range(0, len(audio_segment) - flp, int(sr * 0.015))]
        sp = [p for f in sub if 50 < (p := _frame_pitch(f, sr)) < 400]
        pitch_var = float(np.std(sp)) if len(sp) > 2 else 0.0

    sc = spectral_centroid(audio_segment, sr)

    return {
        "energy": round(energy, 4),
        "energy_var": round(energy_var, 4),
        "pitch": round(pitch, 1),
        "pitch_var": round(pitch_var, 1),
        "spectral_centroid": round(sc, 1),
        "frame_count": len(audio_segment) // int(sr * 0.01) if sr > 0 else 0,
    }


# ── Main extraction ─────────────────────────────────────────────────────────


def extract_engagement(audio_path, result):
    """
    Extract per-speaker engagement metrics from audio and transcription result.

    Returns:
        dict with per_speaker, summary, and segments keys.
    """
    segments = result.get("diarization_segments", [])
    if not segments:
        logger.warning("No diarization segments available")
        return {"per_speaker": {}, "summary": {}, "segments": []}

    try:
        audio, sr = read_audio(audio_path)
    except Exception as e:
        logger.error(f"Failed to read audio: {e}")
        return {"per_speaker": {}, "summary": {}, "segments": []}

    total_dur = len(audio) / sr if sr > 0 else 1.0
    analyzed = []
    spk_m = {}

    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", start + 1)
        speaker = seg.get("speaker") or seg.get("speaker_raw", "unknown")
        s = int(max(0, start * sr))
        e = int(min(len(audio), end * sr))
        if e <= s:
            continue

        m = analyze_segment(audio[s:e], sr)
        analyzed.append({"speaker": speaker, "start": start, "end": end,
                         "duration": end - start, **m})

        if speaker not in spk_m:
            spk_m[speaker] = {"segments": 0, "dur": 0.0, "en": [], "ev": [],
                              "pt": [], "pv": [], "sc": [], "wc": 0}
        sm = spk_m[speaker]
        sm["segments"] += 1
        sm["dur"] += end - start
        sm["en"].append(m["energy"])
        sm["ev"].append(m["energy_var"])
        if m["pitch"] > 0:
            sm["pt"].append(m["pitch"])
        if m["pitch_var"] > 0:
            sm["pv"].append(m["pitch_var"])
        sm["sc"].append(m["spectral_centroid"])

    # Word counts from speaker_turns
    for turn in result.get("speaker_turns", []):
        spk = turn.get("speaker", "unknown")
        if spk in spk_m:
            spk_m[spk]["wc"] += len(turn.get("text", "").split())

    per_spk = {}
    for spk, sm in spk_m.items():
        ae = float(np.mean(sm["en"])) if sm["en"] else 0.0
        aev = float(np.mean(sm["ev"])) if sm["ev"] else 0.0
        ap = float(np.mean(sm["pt"])) if sm["pt"] else 0.0
        apv = float(np.mean(sm["pv"])) if sm["pv"] else 0.0
        asc = float(np.mean(sm["sc"])) if sm["sc"] else 0.0
        sp = (sm["dur"] / total_dur * 100) if total_dur > 0 else 0.0
        sr_wps = sm["wc"] / sm["dur"] if sm["dur"] > 0 else 0.0
        # Composite engagement: energy × variety × rate
        eng = np.clip((ae * 2 + np.clip(aev * 5, 0, 1) +
                       np.clip(apv / 20, 0, 1) +
                       np.clip(sr_wps / 4, 0, 1)) / 4, 0, 1)

        per_spk[spk] = {
            "engagement_score": round(float(eng), 3),
            "avg_energy": round(ae, 4),
            "energy_variation": round(aev, 4),
            "avg_pitch_hz": round(ap, 1),
            "pitch_variation_hz": round(apv, 1),
            "spectral_centroid_hz": round(asc, 1),
            "speaking_rate_wps": round(sr_wps, 3),
            "speaking_percentage": round(sp, 1),
            "total_duration_s": round(sm["dur"], 1),
            "word_count": sm["wc"],
            "segment_count": sm["segments"],
        }

    scores = [s["engagement_score"] for s in per_spk.values()]
    summary = {
        "total_duration_s": round(total_dur, 1),
        "num_speakers": len(per_spk),
        "avg_engagement": round(float(np.mean(scores)), 3) if scores else 0.0,
        "most_engaged": max(per_spk, key=lambda k: per_spk[k]["engagement_score"]) if per_spk else None,
        "least_engaged": min(per_spk, key=lambda k: per_spk[k]["engagement_score"]) if per_spk else None,
    }
    return {"per_speaker": per_spk, "summary": summary, "segments": analyzed}


# ── Pipeline integration ────────────────────────────────────────────────────


def annotate_result(audio_path, result):
    """Add engagement data to result dict (in-place)."""
    engagement = extract_engagement(audio_path, result)
    result["engagement"] = engagement
    logger.info(f"Engagement: {len(engagement.get('per_speaker', {}))} speakers")
    return engagement


# ── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys
    logging.basicConfig(level=logging.INFO)

    arg = sys.argv[1] if len(sys.argv) > 1 else "15-00-09.m4a"
    rec = os.path.basename(arg)
    ap = os.path.join("/recordings", rec)
    rp = f"/recordings/.transcriptions/{rec}.result.json"

    if not os.path.exists(rp) or not os.path.exists(ap):
        print(f"Missing: {rp} or {ap}"); sys.exit(1)

    with open(rp) as f:
        result = json.load(f)

    eng = extract_engagement(ap, result)
    s = eng["summary"]
    print(f"\n=== Engagement: {rec} ===\n"
          f"Duration: {s['total_duration_s']:.0f}s  Speakers: {s['num_speakers']}  "
          f"Avg: {s['avg_engagement']:.3f}\n"
          f"Most: {s['most_engaged']}  Least: {s['least_engaged']}\n")
    print(f"{'Speaker':<28} {'Engage':<8} {'Energy':<8} {'E-Var':<8} "
          f"{'Pitch':<7} {'P-Var':<7} {'Rate':<7} {'Spk%':<6}")
    print("-" * 85)
    for spk, m in sorted(eng["per_speaker"].items(),
                         key=lambda x: x[1]["engagement_score"], reverse=True):
        print(f"{spk:<28} {m['engagement_score']:<8.3f} {m['avg_energy']:<8.3f} "
              f"{m['energy_variation']:<8.3f} {m['avg_pitch_hz']:<7.0f} "
              f"{m['pitch_variation_hz']:<7.1f} {m['speaking_rate_wps']:<7.2f} "
              f"{m['speaking_percentage']:<6.1f}")

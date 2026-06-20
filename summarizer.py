"""
Summarizer — LLM-based meeting analysis with hybrid local/remote routing.

Reads a transcription result (with engagement data), sends it to an LLM
and returns structured analysis including:

  - Meeting overview (participants, duration, atmosphere)
  - Topics discussed (per-speaker summaries)
  - Sentiment by topic per speaker
  - Concerns and challenges raised
  - Guidance and recommendations
  - Action items

Routing:
  - Short recordings (<30 min): Ollama on Mac Mini (local, private)
  - Long recordings (>30 min): remote API fallback (configurable)
"""

import json
import logging
import os
import re
import time

import requests

logger = logging.getLogger("summarizer")

# ── Local (Ollama) config ──────────────────────────────────────────────
OLLAMA_URL = "http://10.3.0.207:11434/api/generate"
LOCAL_MODEL = "qwen2.5:7b"

# ── Remote API config (OpenAI-compatible) ──────────────────────────────
REMOTE_API_URL = os.environ.get(
    "SUMMARIZER_API_URL",
    "https://inference-api.nousresearch.com/v1/chat/completions"
)
REMOTE_API_KEY = os.environ.get("SUMMARIZER_API_KEY", "")
REMOTE_MODEL = os.environ.get(
    "SUMMARIZER_REMOTE_MODEL",
    "deepseek/deepseek-v4-flash"
)

# ── Thresholds ─────────────────────────────────────────────────────────
# Recordings longer than this (seconds) use remote API
LONG_RECORDING_THRESHOLD_S = int(os.environ.get("SUMMARIZER_LONG_THRESHOLD", "1800"))  # 30 min
# Max characters to send in one LLM call (local Ollama has limited context)
LOCAL_MAX_CHARS = 32000
REMOTE_MAX_CHARS = 80000


def _check_ollama_available():
    """Quick health check against Ollama on the Mac."""
    try:
        r = requests.get(f"http://10.3.0.207:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _build_prompt(result, max_chars=LOCAL_MAX_CHARS):
    dur = result.get("audio_duration_seconds", 0)
    mins = int(dur // 60)
    secs = int(dur % 60)

    turns = result.get("speaker_turns", [])
    eng = result.get("engagement", {})
    ps = eng.get("per_speaker", {})

    elines = []
    for spk, m in sorted(ps.items()):
        elines.append(
            f"  - {spk}: engagement={m.get('engagement_score','?'):.2f}, "
            f"energy={m.get('avg_energy','?'):.2f}, "
            f"rate={m.get('speaking_rate_wps','?'):.2f} wps, "
            f"speaking_time={m.get('speaking_percentage','?'):.1f}%"
        )

    tlines = []
    chars = 0
    for t in turns:
        line = f"[{t.get('speaker','?')}] {t.get('text','').strip()}"
        chars += len(line)
        if chars > max_chars:
            tlines.append("[... transcript truncated ...]")
            break
        tlines.append(line)

    es = "\n".join(elines) if elines else "(no engagement data)"
    transcript = "\n".join(tlines)

    return f"""You are an AI meeting analyst. Analyse the following meeting transcript and return ONLY valid JSON (no markdown, no explanation).

## Meeting Info
Duration: {mins}m {secs}s

## Speaker Engagement (audio-based)
{es}

## Transcript
{transcript}

## Output Format
Return a JSON object with these fields:

{{
  "overview": {{
    "meeting_type": "string",
    "atmosphere": "string",
    "summary": "string (2-3 sentences)",
    "key_outcomes": ["array of strings"]
  }},
  "participants": [
    {{
      "name": "string",
      "role_in_meeting": "string",
      "contribution_level": "high|medium|low",
      "notable": "string or null"
    }}
  ],
  "topics": [
    {{
      "topic": "string",
      "participants": ["array"],
      "key_points": ["array"],
      "per_speaker_sentiment": [
        {{
          "speaker": "string",
          "sentiment": "positive|neutral|negative|mixed",
          "energy": "high|medium|low",
          "key_statement": "string"
        }}
      ]
    }}
  ],
  "concerns": [
    {{
      "raised_by": "string",
      "about": "string",
      "severity": "high|medium|low",
      "details": "string"
    }}
  ],
  "recommendations": [
    {{
      "type": "guidance|concern|action_item",
      "to": "string",
      "what": "string",
      "reason": "string",
      "priority": "high|medium|low"
    }}
  ],
  "action_items": [
    {{
      "owner": "string",
      "action": "string",
      "deadline": "string or null"
    }}
  ]
}}"""


def _parse_response(text):
    """Extract JSON from LLM response, handling markdown fences and leading/trailing text."""
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0].strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first JSON object
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"error": "parse_failed", "raw": text[:500]}


def _call_ollama(prompt):
    """Call Ollama on the Mac Mini. Returns parsed JSON dict."""
    payload = {
        "model": LOCAL_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 4096},
    }
    logger.info(f"Sending {len(prompt)} chars to Ollama ({LOCAL_MODEL})...")
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        text = resp.json().get("response", "")
        if not text:
            return {"error": "empty_response"}
        return _parse_response(text)
    except requests.exceptions.Timeout:
        return {"error": "timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "connection_failed", "detail": "Mac (10.3.0.207:11434) unreachable"}
    except Exception as e:
        return {"error": str(e)}


def _chunk_transcript(result, chunk_chars=LOCAL_MAX_CHARS):
    """Split a result's speaker_turns into chunks, each producing a partial analysis."""
    turns = result.get("speaker_turns", [])
    if not turns:
        return [result]

    chunks = []
    current = []
    current_chars = 0

    for t in turns:
        line = f"[{t.get('speaker','?')}] {t.get('text','').strip()}"
        line_len = len(line)
        if current_chars + line_len > chunk_chars and current:
            # Save current chunk
            chunk_result = dict(result)
            chunk_result["speaker_turns"] = current
            chunks.append(chunk_result)
            current = []
            current_chars = 0
        current.append(t)
        current_chars += line_len + 2  # +2 for newline overhead

    if current:
        chunk_result = dict(result)
        chunk_result["speaker_turns"] = current
        chunks.append(chunk_result)

    return chunks


def _merge_chunk_summaries(chunks, full_result):
    """Take per-chunk summaries and merge them via a final LLM pass."""
    # Build a condensed summary of chunk outputs
    combined = []
    for i, c in enumerate(chunks):
        topics = c.get("topics", [])
        concerns = c.get("concerns", [])
        recommendations = c.get("recommendations", [])
        overview = c.get("overview", {})
        combined.append(
            f"--- Chunk {i+1} ---\n"
            f"Overview: {overview.get('summary','')}\n"
            f"Topics: {json.dumps([t['topic'] for t in topics[:5]])}\n"
            f"Concerns: {json.dumps([c['about'] for c in concerns[:5]])}\n"
            f"Recommendations: {json.dumps([r['what'] for r in recommendations[:5]])}\n"
        )

    merge_prompt = (
        "You are an AI meeting analyst. Below are per-chunk analyses of a single long meeting. "
        "Merge them into ONE coherent analysis. Return ONLY valid JSON in the standard format:\n\n"
        + "\n".join(combined) +
        "\n\nMerge these into a single JSON object with: overview, participants, topics, "
        "concerns, recommendations, action_items. Deduplicate and combine where possible."
    )

    return _call_ollama(merge_prompt)


def _call_remote_api(prompt):
    """Call a remote OpenAI-compatible API. Returns parsed JSON dict."""
    if not REMOTE_API_KEY:
        return {"error": "no_api_key", "detail": "Set SUMMARIZER_API_KEY env var"}

    headers = {
        "Authorization": f"Bearer {REMOTE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": REMOTE_MODEL,
        "messages": [
            {"role": "system", "content": "You are an AI meeting analyst. Return ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    logger.info(f"Sending {len(prompt)} chars to remote API ({REMOTE_MODEL})...")
    try:
        resp = requests.post(REMOTE_API_URL, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        if not text:
            return {"error": "empty_response"}
        return _parse_response(text)
    except requests.exceptions.Timeout:
        return {"error": "timeout"}
    except requests.exceptions.ConnectionError as e:
        return {"error": "connection_failed", "detail": str(e)}
    except Exception as e:
        return {"error": str(e)}


def summarize(result):
    """
    Run LLM analysis on a transcription result.

    Automatically routes:
      - Short recordings (< threshold) → Ollama (local)
      - Long recordings (>= threshold) → remote API (if key available) or chunked Ollama

    Returns dict with analysis or {'error': ...}.
    """
    dur = result.get("audio_duration_seconds", 0)
    use_remote = dur >= LONG_RECORDING_THRESHOLD_S and bool(REMOTE_API_KEY)

    if use_remote:
        logger.info(f"Recording length {dur}s >= {LONG_RECORDING_THRESHOLD_S}s → using remote API")
        max_chars = REMOTE_MAX_CHARS
        prompt = _build_prompt(result, max_chars=max_chars)
        if len(prompt) > REMOTE_MAX_CHARS * 4:  # Transcript way too long even for remote
            logger.info("Transcript exceeds remote context — chunking")
            chunks = _chunk_transcript(result, REMOTE_MAX_CHARS)
            chunk_results = []
            for i, chunk in enumerate(chunks):
                chunk_prompt = _build_prompt(chunk, max_chars=REMOTE_MAX_CHARS)
                chunk_r = _call_ollama(chunk_prompt)  # Use local for each chunk
                chunk_results.append(chunk_r)

            # Merge chunks via local Ollama
            return _merge_chunk_summaries(chunk_results, result)

        # Single remote call
        result_r = _call_remote_api(prompt)
        if "error" not in result_r:
            return result_r
        # Fall through to local if remote fails
        logger.warning(f"Remote API failed ({result_r.get('error')}), falling back to local")

    # ── Local (Ollama) path ──────────────────────────────────────────
    if not _check_ollama_available():
        return {"error": "ollama_unavailable",
                "detail": "Ollama on Mac (10.3.0.207:11434) is not responding"}

    max_chars = LOCAL_MAX_CHARS
    prompt = _build_prompt(result, max_chars=max_chars)

    if len(prompt) > LOCAL_MAX_CHARS * 1.5:
        # Transcript too long for single local pass — chunk and merge
        logger.info("Transcript too long for single local pass — chunking")
        chunks = _chunk_transcript(result, LOCAL_MAX_CHARS)
        chunk_results = []
        for i, chunk in enumerate(chunks):
            chunk_prompt = _build_prompt(chunk, max_chars=LOCAL_MAX_CHARS)
            chunk_r = _call_ollama(chunk_prompt)
            chunk_results.append(chunk_r)

        return _merge_chunk_summaries(chunk_results, result)

    return _call_ollama(prompt)


def annotate_result(result):
    """Add LLM analysis to result in-place."""
    result["llm_analysis"] = summarize(result)


if __name__ == "__main__":
    import sys, os
    logging.basicConfig(level=logging.INFO)
    rec = sys.argv[1] if len(sys.argv) > 1 else "11-06-17.m4a"
    p = f"/recordings/.transcriptions/{rec}.result.json"
    if not os.path.exists(p):
        print(f"Not found: {p}")
        sys.exit(1)
    with open(p) as f:
        r = json.load(f)
    result = summarize(r)
    print(json.dumps(result, indent=2, ensure_ascii=False))

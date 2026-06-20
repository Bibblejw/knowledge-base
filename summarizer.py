"""
Summarizer — LLM-based meeting analysis using Ollama on the Mac Mini.

Reads a transcription result (with engagement data), sends it to an LLM
via Ollama, and returns structured analysis including:

  - Meeting overview (participants, duration, atmosphere)
  - Topics discussed (per-speaker summaries)
  - Sentiment by topic per speaker
  - Concerns and challenges raised
  - Guidance and recommendations
  - Action items

Usage:
  from summarizer import summarize
  summary = summarize(result_dict)
  print(summary["topics"])
"""

import json
import logging
import re
import requests

logger = logging.getLogger("summarizer")

OLLAMA_URL = "http://10.3.0.207:11434/api/generate"
MODEL = "qwen2.5-coder:7b"
MAX_CHARS = 32000
TIMEOUT = 300


def _build_prompt(result):
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
        if chars > MAX_CHARS:
            tlines.append("[... truncated ...]")
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


def summarize(result):
    """Run LLM analysis on a transcription result. Returns dict."""
    prompt = _build_prompt(result)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 4096},
    }

    logger.info(f"Sending {len(prompt)} chars to Ollama ({MODEL})...")
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        text = resp.json().get("response", "")
        if not text:
            return {"error": "empty_response"}
        parsed = _parse_response(text)
        if "error" in parsed:
            return parsed
        logger.info(f"Analysis: {len(parsed.get('topics',[]))} topics, "
                    f"{len(parsed.get('concerns',[]))} concerns")
        return parsed
    except requests.exceptions.Timeout:
        return {"error": "timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "connection_failed", "detail": "Mac (10.3.0.207:11434) unreachable"}
    except Exception as e:
        return {"error": str(e)}


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
    print(json.dumps(summarize(r), indent=2, ensure_ascii=False))

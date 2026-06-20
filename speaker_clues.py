"""
Speaker Clues — Identify speaker names from conversational cues.

Uses regex patterns to detect:
  - Self-introductions  (``I'm Jozef``, ``my name is Sarah``, ``Jozef here``)
  - Handovers (weak)   (``over to Sarah``, ``back to Chris``)
  - Named references   (``as Russell said``)
  - Greeting cues      (``thanks Jozef``)

Each detection is scored by pattern type and position.
Call ``extract_speaker_clues(result)`` for suggestions.

False positives are filtered by a comprehensive common-word blacklist
and the enrolled speaker library.
"""

import re
import logging

logger = logging.getLogger("speaker-clues")

# ---------------------------------------------------------------------------
# Blacklist — words that are NEVER names (verbs, particles, etc.)
# ---------------------------------------------------------------------------

NON_NAMES = {
    # Determiners / articles
    "a", "an", "the", "this", "that", "these", "those", "all", "some", "any",
    "each", "every", "both", "few", "many", "much", "several", "no", "none",
    "neither", "either", "what", "which", "whose",
    # Pronouns
    "i", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours",
    "theirs", "myself", "yourself", "himself", "herself", "itself", "ourselves",
    "themselves", "someone", "anyone", "everyone", "no one", "nobody",
    "i've", "i'm", "i'll", "i'd", "we've", "we're", "we'll", "we'd",
    # Prepositions
    "in", "on", "at", "to", "for", "with", "by", "from", "of", "about", "into",
    "through", "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "up", "down", "out", "off", "over", "under", "around", "among",
    # Conjunctions (frequently capitalized at turn start)
    "and", "but", "or", "nor", "yet", "for", "so", "because", "although",
    "though", "while", "since", "unless", "until", "if", "when", "where",
    # Common verbs (gerunds and base forms)
    "going", "coming", "looking", "calling", "trying", "getting", "making",
    "taking", "giving", "doing", "having", "saying", "seeing", "using",
    "working", "thinking", "knowing", "being", "putting", "setting",
    "running", "moving", "starting", "turning", "following", "showing",
    "bringing", "keeping", "finding", "holding", "letting", "meeting",
    "playing", "asking", "telling", "reading", "writing", "speaking",
    "sending", "receiving", "handling", "managing", "building",
    "just", "not", "really", "quite", "pretty", "almost", "nearly",
    "so", "very", "too", "also", "just", "still", "even", "always",
    "never", "often", "sometimes", "usually", "already", "yet",
    "actually", "basically", "essentially", "honestly", "literally",
    "definitely", "certainly", "absolutely", "probably", "maybe",
    "perhaps", "hopefully", "unfortunately", "thankfully",
    # Filler / discourse markers (frequently capitalized at turn start)
    "yeah", "yes", "no", "okay", "ok", "right", "well", "so", "now",
    "anyway", "basically", "actually", "obviously", "essentially",
    "absolutely", "exactly", "precisely", "indeed", "sure",
    "great", "good", "fine", "perfect", "awesome", "excellent",
    "wonderful", "fantastic", "cool", "nice",
    "thanks", "thank", "cheers", "welcome", "please", "sorry",
    "excuse", "pardon", "hello", "hi", "hey", "morning", "afternoon",
    "everyone", "everybody", "folks", "guys", "team", "all",
    "however", "therefore", "nevertheless", "meanwhile", "additionally",
    "furthermore", "moreover", "consequently", "accordingly",
    "otherwise", "instead", "regardless", "nonetheless",
    # Time / number words
    "today", "yesterday", "tomorrow", "now", "later", "earlier",
    "next", "last", "previous", "first", "second", "third",
    "one", "two", "three", "four", "five",
    # Misc high-frequency words
    "let", "get", "go", "come", "see", "know", "think", "say", "tell",
    "ask", "make", "take", "give", "put", "set", "bring", "keep",
    "find", "hold", "start", "stop", "begin", "end", "finish",
    "want", "need", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "gonna", "wanna", "gotta",
    "maybe", "perhaps", "hopefully", "ideally", "honestly",
}


def _is_probably_a_name(word):
    """Check if a word looks like a proper name (not a common English word)."""
    if not word:
        return False
    word_stripped = word.strip(".,!?;:\"'()[]{}")
    if not word_stripped:
        return False
    # Must start with uppercase (proper name)
    if not word_stripped[0].isupper():
        return False
    word_lower = word_stripped.lower()
    if word_lower in NON_NAMES:
        return False
    if len(word_lower) < 2:
        return False
    return True


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Self-introduction: "I'm Jozef Woods", "I am Sarah", "my name is Jozef"
SELF_INTRO = re.compile(
    r"(?:i['’]m|i am|my name['’]?s|my name is|the name['’]s|the name is|name['’]s)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)?)",
    re.IGNORECASE
)

# "This is Jozef" — self-ID (works mid-turn too)
THIS_IS_INTRO = re.compile(
    r"[Tt]his\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)?)",
)

# Check-in: "Jozef here", "Jozef Woods speaking", "Sarah reporting in"
CHECK_IN = re.compile(
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)?)\s+"
    r"(?:here|speaking|checking\s+in|reporting\s+in|on\s+the\s+call)",
    re.IGNORECASE
)

# Named reference: "as Jozef said", "like Chris mentioned"
# "Jozef mentioned that", "thanks Jozef"
NAMED_REF = re.compile(
    r"(?:as|like)\s+([A-Z][a-z]+)\s+(?:said|mentioned|noted|pointed|was)"
    r"|^([A-Z][a-z]+)\s+(?:said|mentioned|noted|asked|raised|suggested|noted)\s+"
    r"|^(?:thanks|cheers|thank you)\s*[,;:]?\s*([A-Z][a-z]+)",
    re.IGNORECASE
)

# Greeting: "Morning Jozef", "Good morning Chris", "Hi Sarah"
GREETING = re.compile(
    r"(?:good\s+(?:morning|afternoon|evening)|morning|afternoon|hi|hello|hey)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)?)",
    re.IGNORECASE
)

# Handover: "over to Jozef", "back to Chris", "to you Jozef"
HANDOVER = re.compile(
    r"(?:over\s+to|back\s+to|to\s+you)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)?)",
    re.IGNORECASE
)

# Introduction of others: "this is Jozef" (when it's NOT the speaker — used by context)
INTRO_OTHER = re.compile(
    r"(?:meet|say\s+hello\s+to|introducing|please\s+welcome|let\s+me\s+introduce)\s+"
    r"(?:you\s+to\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)?)",
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Confidence weights
# ---------------------------------------------------------------------------

PATTERN_CONF = {
    "self_intro":  0.90,
    "this_is":     0.80,
    "check_in":    0.80,
    "intro_other": 0.50,
    "handover":    0.45,
    "named_ref":   0.35,
    "greeting":    0.20,
}


def _pos_boost(turn_idx, total_turns):
    """Boost for clues early in the recording."""
    if total_turns <= 1:
        return 1.0
    ratio = turn_idx / total_turns
    if ratio < 0.15:
        return 1.15
    elif ratio < 0.30:
        return 1.08
    elif ratio < 0.70:
        return 1.0
    else:
        return 0.85


def _check_enrolled(name):
    """Check if a name (first word) matches an enrolled speaker."""
    first = name.split()[0].lower()
    try:
        import pickle, os
        lib_path = "/recordings/.global_speaker_library.pkl"
        if not os.path.exists(lib_path):
            return False, None
        with open(lib_path, "rb") as f:
            lib = pickle.load(f)
        for entry in lib.values():
            if isinstance(entry, dict):
                ename = entry.get("name", "")
                if ename.lower().startswith(first):
                    return True, ename
    except Exception:
        pass
    return False, None


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_speaker_clues(result):
    """
    Extract speaker name clues from a transcription result.

    Returns list of:
      {speaker_label, candidate_name, confidence, clue_count, already_mapped, existing_name, turn_index}
    Sorted by confidence descending.
    """
    turns = result.get("speaker_turns") or result.get("diarization_segments", [])
    if not turns:
        return []

    total = len(turns)

    # Collect name votes per speaker: {label: {name: [scores]}}
    votes = {}

    def _vote(spk, name, score, idx):
        if spk not in votes:
            votes[spk] = {}
        name_clean = re.sub(r"\s+", " ", name).strip().title()
        if name_clean not in votes[spk]:
            votes[spk][name_clean] = []
        votes[spk][name_clean].append(score)

    for idx, turn in enumerate(turns):
        text = turn.get("text", "").strip()
        spk = turn.get("speaker") or turn.get("speaker_raw", "")
        if not text:
            continue

        pb = _pos_boost(idx, total)

        # --- 1) Self-intro (reliable) ---
        for m in SELF_INTRO.finditer(text):
            name = m.group(1).strip()
            if _is_probably_a_name(name):
                score = PATTERN_CONF["self_intro"] * pb
                _vote(spk, name, score, idx)

        # --- 2) "This is [Name]" at turn start ---
        for m in THIS_IS_INTRO.finditer(text):
            name = m.group(1).strip()
            if _is_probably_a_name(name):
                score = PATTERN_CONF["this_is"] * pb
                _vote(spk, name, score, idx)

        # --- 3) Check-in patterns ---
        for m in CHECK_IN.finditer(text):
            name = m.group(1).strip()
            if _is_probably_a_name(name):
                score = PATTERN_CONF["check_in"] * pb
                _vote(spk, name, score, idx)

        # --- 4) Named references ---
        for m in NAMED_REF.finditer(text):
            # Multiple capture groups — take the non-None one
            name = next((g for g in m.groups() if g), None)
            if not name or not _is_probably_a_name(name):
                continue
            score = PATTERN_CONF["named_ref"] * pb
            # Apply to the current speaker only if they could be the name-bearer
            enrolled, ename = _check_enrolled(name)
            if enrolled:
                # Known name — assign to matching speaker if we can identify them
                _vote(spk, name, score, idx)

        # --- 5) Handover — name applies to the NEXT speaker ---
        for m in HANDOVER.finditer(text):
            name = m.group(1).strip()
            if not _is_probably_a_name(name):
                continue
            if idx + 1 < total:
                next_spk = turns[idx + 1].get("speaker") or turns[idx + 1].get("speaker_raw", "")
                if next_spk and next_spk != spk:
                    score = PATTERN_CONF["handover"] * pb
                    _vote(next_spk, name, score, idx)

        # --- 6) Introduction of others — applies to NEXT speaker ---
        for m in INTRO_OTHER.finditer(text):
            name = m.group(1).strip()
            if not _is_probably_a_name(name):
                continue
            if idx + 1 < total:
                next_spk = turns[idx + 1].get("speaker") or turns[idx + 1].get("speaker_raw", "")
                if next_spk and next_spk != spk:
                    score = PATTERN_CONF["intro_other"] * pb
                    _vote(next_spk, name, score, idx)

        # --- 7) Greeting — name applies to PREVIOUS or CURRENT speaker ---
        for m in GREETING.finditer(text):
            name = m.group(1).strip()
            if not _is_probably_a_name(name):
                continue
            score = PATTERN_CONF["greeting"] * pb
            # If the speaker greeted themselves (rare), it's weak
            # Usually it's directed at someone else
            if idx > 0:
                prev_spk = turns[idx - 1].get("speaker") or turns[idx - 1].get("speaker_raw", "")
                if prev_spk and prev_spk != spk:
                    _vote(prev_spk, name, score, idx)

    # --- Rank ---
    suggestions = []
    for spk, name_votes in votes.items():
        best_name = None
        best_avg = 0
        for name, scores in name_votes.items():
            avg = sum(scores) / len(scores)
            enrolled, ename = _check_enrolled(name)
            if enrolled:
                avg *= 1.15  # Slight boost for known names
            if avg > best_avg:
                best_avg = avg
                best_name = name

        if best_name and best_avg >= 0.3:
            enrolled, ename = _check_enrolled(best_name)
            suggestions.append({
                "speaker_label": spk,
                "candidate_name": best_name,
                "confidence": round(best_avg, 3),
                "clue_count": len(name_votes.get(best_name, [])),
                "already_mapped": enrolled,
                "existing_name": ename,
            })

    suggestions.sort(key=lambda x: x["confidence"], reverse=True)
    return suggestions


# ---------------------------------------------------------------------------
# Pipeline integration hook
# ---------------------------------------------------------------------------

def annotate_result(result):
    """
    Add speaker clues to result dict (in-place).
    High-confidence matches (>= 0.8) are auto-applied to speaker_turns.
    """
    clues = extract_speaker_clues(result)
    result["speaker_clues"] = clues

    applied = []
    for clue in clues:
        if clue["confidence"] >= 0.8 and not clue["already_mapped"]:
            label = clue["speaker_label"]
            name = clue["candidate_name"]
            for turn in result.get("speaker_turns", []):
                if turn["speaker"] == label:
                    turn["speaker"] = name
            for seg in result.get("diarization_segments", []):
                if seg.get("speaker") == label:
                    seg["speaker"] = name
            applied.append((label, name))
            logger.info(f"Speaker Clues: auto-applied {label} -> {name} (conf={clue['confidence']})")

    if applied:
        logger.info(f"Speaker Clues: {len(applied)} auto-applied")
    return clues


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys
    logging.basicConfig(level=logging.INFO)

    path = sys.argv[1] if len(sys.argv) > 1 else "/recordings/.transcriptions/10-00-29.m4a.result.json"
    with open(path) as f:
        data = json.load(f)

    print(f"=== Speaker Clues Analysis ===")
    print(f"File: {path}\n")

    clues = extract_speaker_clues(data)

    if not clues:
        print("  No clues found.")
    else:
        for c in clues:
            mapped = ""
            if c["already_mapped"]:
                mapped = f"  [ALREADY: {c['existing_name']}]"
            print(f"  {c['speaker_label']:<18} -> {c['candidate_name']:<20}  conf={c['confidence']:.2f}  "
                  f"({c['clue_count']} clues){mapped}")

    print(f"\nTotal: {len(clues)} suggestions")

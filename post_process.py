"""
Post-processing: apply glossary corrections to transcribed text.
Runs after ASR and merge, before saving the result.

Glossary file format (``/recordings/.glossary.txt``):

  # Lines starting with # are comments
  # Empty lines are skipped

  # Acronyms — letter-spaced (handles "B C P", "B.C.P.", "B C Ps" etc.)
  B C P => BCP
  B R P => BRP

  # Word corrections — case-insensitive whole-word replacement
  camban => kanban
  canban => kanban

Usage:
  from post_process import apply_glossary
  apply_glossary(result_dict)  # modifies in-place
"""

import re
import json
import logging

logger = logging.getLogger("post-process")

GLOSSARY_PATH = "/recordings/.glossary.txt"

# Cache compiled patterns for performance
_glossary_cache = None  # (mtime, letter_spaced_rules, word_rules)


def _load_glossary():
    """Load and parse glossary file. Returns (letter_rules, word_rules)."""
    import os
    try:
        mtime = os.path.getmtime(GLOSSARY_PATH)
    except OSError:
        logger.warning(f"Glossary not found at {GLOSSARY_PATH}")
        return [], []

    global _glossary_cache
    if _glossary_cache and _glossary_cache[0] == mtime:
        return _glossary_cache[1], _glossary_cache[2]

    letter_rules = []  # (compiled_regex, replacement)
    word_rules = []    # (pattern_lower, replacement)

    try:
        with open(GLOSSARY_PATH) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=>" not in line:
                    continue

                pattern, replacement = line.split("=>", 1)
                pattern = pattern.strip()
                replacement = replacement.strip()

                if not pattern or not replacement:
                    continue

                # Detect letter-spaced pattern (e.g., "B C P")
                parts = pattern.split()
                if len(parts) >= 2 and all(len(p) == 1 and p.isalpha() for p in parts):
                    # Build regex: word boundary, then optional dots/spaces between letters
                    letter_pattern = r'\b' + r'[.\s]*'.join(re.escape(p) for p in parts) + r'(s?)\b'
                    letter_rules.append((re.compile(letter_pattern, re.IGNORECASE), replacement + r'\1'))
                else:
                    # Word correction — case-insensitive whole-word
                    word_rules.append((pattern.lower(), replacement))
    except Exception as e:
        logger.error(f"Failed to load glossary: {e}")
        return [], []

    _glossary_cache = (mtime, letter_rules, word_rules)
    logger.info(f"Glossary loaded: {len(letter_rules)} letter-spaced rules, {len(word_rules)} word rules")
    return letter_rules, word_rules


def apply_to_text(text: str) -> str:
    """Apply glossary corrections to a text string."""
    if not text:
        return text

    letter_rules, word_rules = _load_glossary()
    result = text

    # Apply letter-spaced acronym rules
    for pattern, replacement in letter_rules:
        count = 0
        result, count = pattern.subn(replacement, result)
        if count > 0:
            logger.debug(f"  Glossary letter-rule: {pattern.pattern} -> {replacement} ({count}x)")

    # Apply word correction rules
    for pattern_lower, replacement in word_rules:
        # Case-insensitive whole-word replacement preserving case
        count = 0
        new_result = []
        for word in result.split():
            stripped = word.strip(".,!?;:\"'()[]{}")
            punct_before = word[:len(word) - len(word.lstrip(".,!?;:\"'()[]{}"))]
            punct_after = word[len(word.rstrip(".,!?;:\"'()[]{}")):]
            if stripped.lower() == pattern_lower:
                # Preserve case style
                if stripped.isupper():
                    corrected = replacement.upper()
                elif stripped[0].isupper():
                    corrected = replacement.capitalize()
                else:
                    corrected = replacement.lower()
                new_result.append(punct_before + corrected + punct_after)
                count += 1
            else:
                new_result.append(word)
        if count > 0:
            result = " ".join(new_result)
            logger.debug(f"  Glossary word-rule: {pattern_lower} -> {replacement} ({count}x)")

    return result


def apply_glossary(result: dict):
    """Modify transcription result in-place, applying glossary corrections."""
    if not result:
        return

    # Correct full_text
    if "full_text" in result:
        result["full_text"] = apply_to_text(result["full_text"])

    # Correct sentence-level text
    for key in ("sentences", "speaker_turns"):
        items = result.get(key, [])
        if isinstance(items, list):
            for item in items:
                if "text" in item and item["text"]:
                    item["text"] = apply_to_text(item["text"])

    # Correct diarization segments
    segments = result.get("diarization_segments", [])
    if isinstance(segments, list) and segments and "text" in segments[0]:
        for seg in segments:
            if "text" in seg and seg["text"]:
                seg["text"] = apply_to_text(seg["text"])

    logger.info("Glossary post-processing applied")


# ── Glossary management ─────────────────────────────────────────────────────

def write_default_glossary():
    """Create the default glossary file if it doesn't exist."""
    content = """# Glossary — Knowledge Base Post-Processing
#
# Format:  pattern => replacement
# Lines starting with # are comments. Empty lines are skipped.
#
# Letter-spaced acronyms (e.g., "B C P") are matched flexibly:
#   "B C P", "B.C.P.", "B C Ps" all match "BCP"
#
# Word corrections are case-insensitive whole-word replacements.
# Capitalization of the replacement follows the original.

# --- Acronyms (letter-spaced) ---
B C P => BCP
B R P => BRP
T R P => TRP
T S R P => TSRP
S E R P => SERP
I T D R => ITDR
I T S R => ITSR

# --- Word corrections ---
camban => kanban
canban => kanban
robusting => robust
"""
    import os
    path = GLOSSARY_PATH
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(content)
        logger.info(f"Created default glossary at {path}")
        return True
    return False


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.DEBUG)
    write_default_glossary()

    test_texts = [
        "The B C P process needs review",
        "B.R.P. and T R P are related",
        "I need to update the B C Ps for Q3",
        "Let's review the camban board",
        "The canban system works well",
        "No changes needed here",
    ]
    for t in test_texts:
        result = apply_to_text(t)
        print(f"  {t:<45} -> {result}")

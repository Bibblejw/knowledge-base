"""
Pipeline transcription grading tool.

Compares Parakeet pipeline output against MacWhisper reference transcripts
to measure word accuracy, speaker overlap (F1), and per-speaker quality.

Usage:
  python3 grade.py                          # grade all recordings with known references
  python3 grade.py 11-02-09.m4a             # grade a specific recording
  python3 grade.py --json                   # machine-readable output

Output: summary table + /recordings/.transcriptions/grade_results.json
"""

import json, re, os, sys
from collections import Counter

try:
    from jiwer import wer as jiwer_wer
    HAVE_JIWER = True
except ImportError:
    HAVE_JIWER = False

# ── Paths ──
TRANS_DIR = "/recordings/.transcriptions"
REFS_DIR = "/recordings/references"

# Known reference mappings: recording -> reference filename
# Edit this to add new references as you produce them
REF_MAP = {
    "10-00-29.m4a": "11-06-26 - Standup.md",
    "11-02-09.m4a": "11-06-26 - Chris.md",
    "15-28-36.m4a": "26-06-11 - Arun Intro.md",
    "10-01-29.m4a": "26-06-12 - Standup.md",
    "12-30-57.m4a": "26-06-12 - ITDR Monthly.md",
    "13-07-36.m4a": "26-06-12 - Russel.md",
    "14-30-30.m4a": "26-06-12 - Sarah.md",
    "11-06-17.m4a": "26-06-18 - Chris.md",
    "11-17-46.m4a": "26-06-18 - Sarah.md",
}


# ── Reference parsing ─────────────────────────────────────────────────────────

def parse_reference(ref_text):
    """Parse MacWhisper markdown into (full_text, speaker_set, [(spk, text), ...])."""
    pairs = []
    current_speaker = "Unknown"
    for line in ref_text.split("\n"):
        line = line.strip()
        m = re.match(r'\*\*(.+?)\*\*', line)
        if m:
            current_speaker = m.group(1).strip()
            continue
        if re.match(r'^\*\d+:\d+', line):
            continue
        if line and not line.startswith("*"):
            pairs.append((current_speaker, line))
    speakers = {s for s, _ in pairs}
    full_text = " ".join(t for _, t in pairs)
    return full_text, speakers, pairs


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_wer(ref, hyp):
    """Word Error Rate."""
    if HAVE_JIWER:
        return jiwer_wer(ref, hyp)
    # Fallback: word-level edit distance
    rw = ref.lower().split()
    hw = hyp.lower().split()
    if not rw:
        return 0.0 if not hw else 1.0
    n, m = len(rw), len(hw)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if rw[i-1] == hw[j-1] else 1
            curr[j] = min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev = curr
    return prev[m] / n


def compute_word_accuracy(ref, hyp):
    """1.0 - WER, clipped to [0, 1]."""
    return max(0.0, min(1.0, 1.0 - compute_wer(ref, hyp)))


def compute_word_overlap(ref_text, hyp_text):
    """Jaccard word overlap as a secondary content metric."""
    rw = set(ref_text.lower().split())
    hw = set(hyp_text.lower().split())
    if not rw or not hw:
        return 0.0
    return len(rw & hw) / len(rw | hw)


# ── Grading ───────────────────────────────────────────────────────────────────

def grade_recording(rec_name):
    """Grade a single recording. Returns dict of metrics or None."""
    result_path = os.path.join(TRANS_DIR, rec_name + ".result.json")
    ref_filename = REF_MAP.get(rec_name)
    if not ref_filename:
        return None  # no reference available

    ref_path = os.path.join(REFS_DIR, ref_filename)

    if not os.path.exists(result_path):
        print(f"  SKIP {rec_name}: no result file")
        return None
    if not os.path.exists(ref_path):
        print(f"  SKIP {rec_name}: no reference {ref_filename}")
        return None

    with open(result_path) as f:
        data = json.load(f)
    with open(ref_path) as f:
        ref_text = f.read()

    # Reference
    ref_full, ref_speakers, ref_pairs = parse_reference(ref_text)

    # Pipeline
    pipe_full = data.get("full_text", "")
    pipe_turns = data.get("speaker_turns", [])
    pipe_speakers = {t["speaker"] for t in pipe_turns}

    # Word accuracy
    word_acc = compute_word_accuracy(ref_full, pipe_full)
    word_overlap = compute_word_overlap(ref_full, pipe_full)
    ref_wc = len(ref_full.split())
    pipe_wc = len(pipe_full.split())

    # Speaker metrics
    common = ref_speakers & pipe_speakers
    precision = len(common) / len(pipe_speakers) if pipe_speakers else 0
    recall = len(common) / len(ref_speakers) if ref_speakers else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-speaker accuracy
    per_speaker = {}
    for spk in sorted(pipe_speakers):
        spk_text = " ".join(t["text"] for t in pipe_turns if t["speaker"] == spk)
        # Find best-matching reference speaker by word overlap
        best_ref = None
        best_overlap = 0
        for rspk in ref_speakers:
            rspk_text = " ".join(t for s, t in ref_pairs if s == rspk)
            overlap = len(set(spk_text.lower().split()) & set(rspk_text.lower().split()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_ref = rspk
        if best_ref:
            rspk_text = " ".join(t for s, t in ref_pairs if s == best_ref)
            per_speaker[spk] = {
                "matched_ref": best_ref,
                "accuracy": round(compute_word_accuracy(rspk_text, spk_text), 4),
                "ref_wc": len(rspk_text.split()),
                "pipe_wc": len(spk_text.split()),
            }

    return {
        "word_accuracy": round(word_acc, 4),
        "word_overlap": round(word_overlap, 4),
        "wer": round(1.0 - word_acc, 4),
        "ref_wc": ref_wc,
        "pipe_wc": pipe_wc,
        "word_count_ratio": round(pipe_wc / ref_wc, 3) if ref_wc else 0,
        "ref_speakers": sorted(ref_speakers),
        "pipe_speakers": sorted(pipe_speakers),
        "common_speakers": sorted(common),
        "speaker_precision": round(precision, 3),
        "speaker_recall": round(recall, 3),
        "speaker_f1": round(f1, 3),
        "per_speaker": per_speaker,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    want_json = "--json" in sys.argv

    # Determine which recordings to grade
    targets = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not targets:
        targets = sorted(REF_MAP.keys())

    results = {}
    for rec in targets:
        r = grade_recording(rec)
        if r:
            results[rec] = r

    if not results:
        print("No results to grade. Check your REF_MAP or recording names.")
        return

    if want_json:
        print(json.dumps(results, indent=2))
        return

    # Pretty-print table
    header = f"{'Recording':<18} {'Acc':<7} {'WER':<7} {'WCR':<7} {'SpkF1':<7} {'RefSpk':<8} {'PipSpk':<8}"
    print(header)
    print("-" * 70)
    total_acc = total_f1 = 0
    for rec, r in sorted(results.items()):
        print(f"{rec:<18} {r['word_accuracy']:<7.3f} {r['wer']:<7.3f} {r['word_count_ratio']:<7.3f} "
              f"{r['speaker_f1']:<7.2f} {len(r['ref_speakers']):<8} {len(r['pipe_speakers']):<8}")
        total_acc += r['word_accuracy']
        total_f1 += r['speaker_f1']
    n = len(results)
    print("-" * 70)
    print(f"{'AVERAGE':<18} {total_acc/n:<7.3f} {'':7} {'':7} {total_f1/n:<7.2f}")

    # Per-speaker breakdown
    print("\n\n── Per-Speaker Accuracy ──")
    for rec, r in sorted(results.items()):
        if r['per_speaker']:
            print(f"\n{rec}:")
            for spk, info in sorted(r['per_speaker'].items()):
                match_str = f"  → ref: {info['matched_ref']}" if info['matched_ref'] != spk else ""
                print(f"  {spk:<30} acc={info['accuracy']:.3f}  wc={info['ref_wc']}/{info['pipe_wc']} {match_str}")

    # Save detailed results
    out_path = os.path.join(TRANS_DIR, "grade_results.json")
    with open(out_path, "w") as f:
        json.dump({"recordings": results, "summary": {
            "avg_word_accuracy": round(total_acc / n, 4),
            "avg_speaker_f1": round(total_f1 / n, 4),
            "count": n,
        }}, f, indent=2)
    print(f"\n\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()

"""
Knowledge Base — Audio Recordings Vault.

A self-hosted web app for uploading, browsing, playing, and transcribing
audio recordings. Built to grow into a broader knowledge management system.
"""

import os
import time
import json
import threading
from pathlib import Path

import requests
from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)

import transcriber
import metadata

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────

RECORDINGS_DIR = Path("/recordings")
ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".aac",
    ".m4a", ".opus", ".wma", ".aiff", ".alac",
}

app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

# Speaker colour palette (cycling)
SPEAKER_COLORS = [
    "#58a6ff",  # blue
    "#3fb950",  # green
    "#d29922",  # yellow/gold
    "#f85149",  # red
    "#bc8cff",  # purple
    "#79c0ff",  # light blue
    "#56d364",  # light green
    "#e3b341",  # amber
    "#ff7b72",  # salmon
    "#d2a8ff",  # lavender
]


# ── Helpers ────────────────────────────────────────────────────────────

def is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def is_audio(filename: str) -> bool:
    return is_allowed(filename)


def human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def speaker_color(name: str) -> str:
    """Deterministic colour for a speaker name."""
    idx = hash(name) % len(SPEAKER_COLORS)
    return SPEAKER_COLORS[idx]


def list_recordings() -> list[dict]:
    """Return sorted list of recording info dicts."""
    recordings = []
    if not RECORDINGS_DIR.exists():
        return recordings

    for f in sorted(RECORDINGS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.is_file() and is_allowed(f.name):
            stat = f.stat()
            trans_status = transcriber.get_status(f.name)
            # Extract speaker list from result if completed
            speakers = []
            if trans_status.get("status") == "completed":
                segs = trans_status.get("sentences", [])
                speakers = sorted(set(s["speaker"] for s in segs))
            meta = metadata.get_metadata(f.name)
            recordings.append({
                "name": f.name,
                "size": human_size(stat.st_size),
                "size_bytes": stat.st_size,
                "modified": time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)
                ),
                "url": url_for("serve_file", filename=f.name),
                "transcription": trans_status,
                "speakers": speakers,
                "labels": meta.get("labels", []),
                "group": meta.get("group") or None,
                "group_info": meta.get("group_info") or None,
            })

    return recordings


def get_speaker_stats() -> dict:
    """Aggregate speaker appearances across all transcriptions."""
    from diarization import list_speakers
    enrolled = {s: {"enrolled": True} for s in list_speakers()}
    discovered = {}

    if not RECORDINGS_DIR.exists():
        return {"enrolled": enrolled, "discovered": discovered}

    for f in RECORDINGS_DIR.iterdir():
        if not f.is_file() or not is_allowed(f.name):
            continue
        status = transcriber.get_status(f.name)
        if status.get("status") != "completed":
            continue
        segs = status.get("sentences", [])
        seen = set()
        for s in segs:
            sp = s.get("speaker", "UNKNOWN")
            if sp not in seen:
                seen.add(sp)
                if sp not in discovered:
                    discovered[sp] = {"enrolled": False, "recordings": [], "sentences": 0}
                discovered[sp]["recordings"].append(f.name)
            discovered[sp]["sentences"] += 1

    return {"enrolled": enrolled, "discovered": discovered}


# ── Template Globals ───────────────────────────────────────────────────

@app.context_processor
def utility_processor():
    return dict(speaker_color=speaker_color)


# ── Main Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    import metadata
    return render_template("index.html",
        recordings=list_recordings(),
        all_labels=metadata.get_all_labels(),
        all_groups=metadata.get_groups())


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file provided", 400
    file = request.files["file"]
    if not file.filename:
        return "No file selected", 400
    if not is_allowed(file.filename):
        return f"File type not allowed. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}", 400
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    file.save(str(RECORDINGS_DIR / file.filename))
    return "", 204


@app.route("/files/<path:filename>")
def serve_file(filename):
    return send_from_directory(str(RECORDINGS_DIR), filename)


@app.route("/delete/<path:filename>", methods=["POST"])
def delete_file(filename):
    filepath = RECORDINGS_DIR / filename
    if filepath.exists() and filepath.is_file():
        filepath.unlink()
        # Clean up transcription files
        for f in (RECORDINGS_DIR / ".transcriptions").glob(f"{filename}.*"):
            f.unlink(missing_ok=True)
        # Clean up metadata
        metadata.delete_metadata(filename)
    return "", 204


# ── Recording Detail ───────────────────────────────────────────────────

@app.route("/recordings/<path:filename>")
def recording_detail(filename):
    """View a recording's full transcript."""
    audio_path = RECORDINGS_DIR / filename
    if not audio_path.exists():
        abort(404)

    stat = audio_path.stat()
    status = transcriber.get_status(filename)

    if status.get("status") != "completed":
        return render_template("recording.html",
            recording={
                "name": filename,
                "size": human_size(stat.st_size),
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                "url": url_for("serve_file", filename=filename),
                "status": status.get("status", "unknown"),
            },
            transcript=None,
            speakers=[],
        )

    segments = status.get("sentences", [])
    speaker_turns = status.get("speaker_turns", [])
    speakers = sorted(set(s["speaker"] for s in segments))
    speaker_info = {s: {"color": speaker_color(s)} for s in speakers}

    return render_template("recording.html",
        recording={
            "name": filename,
            "size": human_size(stat.st_size),
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
            "url": url_for("serve_file", filename=filename),
            "status": "completed",
            "duration": status.get("audio_duration_seconds", 0),
            "engine": status.get("engine", ""),
        },
        transcript=segments,
        speakers=speaker_info,
        speaker_turns=speaker_turns,
    )


@app.route("/recordings/<path:filename>/raw_transcript")
def raw_transcript(filename):
    """Return plain text transcript."""
    status = transcriber.get_status(filename)
    if status.get("status") != "completed":
        return jsonify({"error": "Not transcribed"}), 404
    full_text = status.get("full_text", "")
    return full_text, 200, {"Content-Type": "text/plain; charset=utf-8"}


# ── Transcription Routes ───────────────────────────────────────────────

@app.route("/transcribe/<path:filename>", methods=["POST"])
def transcribe_file(filename):
    """Start transcription for a single file."""
    audio_path = RECORDINGS_DIR / filename
    if not audio_path.exists():
        return jsonify({"error": "File not found"}), 404
    if not is_audio(filename):
        return jsonify({"error": "Not an audio file"}), 400

    status = transcriber.get_status(filename)
    if status["status"] == "completed":
        return jsonify({"status": "completed", "message": "Already transcribed"})
    if status["status"] == "processing":
        return jsonify({"status": "processing", "message": "Already in progress"})

    def _run():
        transcriber.run_transcription(filename)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status": "started"}), 202


@app.route("/transcribe/<path:filename>/status")
def transcribe_status(filename):
    status = transcriber.get_status(filename)
    return jsonify(status)


@app.route("/transcribe/<path:filename>/result")
def transcribe_result(filename):
    status = transcriber.get_status(filename)
    return jsonify(status)


# ── Batch Transcription ────────────────────────────────────────────────

@app.route("/batch_transcribe", methods=["POST"])
def batch_transcribe():
    """Transcribe all untranscribed recordings."""
    batch = []
    if RECORDINGS_DIR.exists():
        for f in sorted(RECORDINGS_DIR.iterdir()):
            if f.is_file() and is_allowed(f.name):
                status = transcriber.get_status(f.name)
                if status["status"] not in ("completed", "processing"):
                    batch.append(f.name)

    if not batch:
        return jsonify({"status": "none", "message": "All recordings already transcribed"})

    def _run_batch():
        for name in batch:
            transcriber.run_transcription(name)

    t = threading.Thread(target=_run_batch, daemon=True)
    t.start()

    return jsonify({"status": "started", "count": len(batch), "files": batch}), 202


# ── Speaker Enrollment & Management ────────────────────────────────────

@app.route("/speakers")
def speaker_page():
    """Speaker management page."""
    from diarization import list_speakers, list_global_speakers
    enrolled = list_speakers()
    global_speakers = list_global_speakers()
    stats = get_speaker_stats()
    return render_template("speakers.html",
        enrolled=enrolled,
        stats=stats,
        global_speakers=global_speakers,
        speaker_color=speaker_color,
    )


@app.route("/api/speakers", methods=["GET"])
def api_speaker_list():
    """List all speakers (enrolled + discovered)."""
    stats = get_speaker_stats()
    return jsonify(stats)


@app.route("/api/speakers/enroll", methods=["POST"])
def api_speaker_enroll():
    """Enroll a speaker from an uploaded voice clip."""
    if "audio" not in request.files or "name" not in request.form:
        return jsonify({"error": "audio file and speaker name required"}), 400

    audio_file = request.files["audio"]
    name = request.form["name"].strip()
    if not name:
        return jsonify({"error": "Speaker name required"}), 400

    suffix = Path(audio_file.filename).suffix if audio_file.filename else ".m4a"
    tmp_path = f"/tmp/enroll_{name}{suffix}"
    audio_file.save(tmp_path)

    try:
        from diarization import enroll_speaker
        result = enroll_speaker(name, tmp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/api/speakers/<name>", methods=["DELETE"])
def api_speaker_remove(name):
    """Remove an enrolled speaker."""
    from diarization import remove_speaker
    if remove_speaker(name):
        return jsonify({"status": "removed"})
    return jsonify({"error": "Speaker not found"}), 404


@app.route("/api/speakers/rename", methods=["POST"])
def api_speaker_rename():
    """Rename a discovered speaker across all transcriptions."""
    data = request.get_json()
    old_name = data.get("old_name", "").strip()
    new_name = data.get("new_name", "").strip()
    if not old_name or not new_name:
        return jsonify({"error": "old_name and new_name required"}), 400
    if not new_name.replace("_", "").isalnum():
        return jsonify({"error": "Name must be alphanumeric"}), 400

    renamed = 0
    if RECORDINGS_DIR.exists():
        for f in RECORDINGS_DIR.iterdir():
            if not f.is_file() or not is_allowed(f.name):
                continue
            result_path = RECORDINGS_DIR / ".transcriptions" / f"{f.name}.result.json"
            if not result_path.exists():
                continue
            try:
                with open(result_path) as rp:
                    data = json.load(rp)
                modified = False
                for seg in data.get("sentences", []):
                    if seg.get("speaker") == old_name:
                        seg["speaker"] = new_name
                        modified = True
                for seg in data.get("diarization_segments", []):
                    if seg.get("speaker") == old_name:
                        seg["speaker"] = new_name
                        modified = True
                for turn in data.get("speaker_turns", []):
                    if turn.get("speaker") == old_name:
                        turn["speaker"] = new_name
                        modified = True
                if modified:
                    with open(result_path, "w") as rp:
                        json.dump(data, rp, indent=2, ensure_ascii=False)
                    renamed += 1
            except (json.JSONDecodeError, OSError):
                continue

    return jsonify({"status": "renamed", "old_name": old_name, "new_name": new_name, "files_updated": renamed})


# ── Global Discovered Speakers ─────────────────────────────────────────

@app.route("/api/global_speakers")
def api_global_speakers():
    """List speakers in the cross-recording global library."""
    from diarization import list_global_speakers
    speakers = list_global_speakers()
    return jsonify({"speakers": speakers, "count": len(speakers)})


@app.route("/api/global_speakers/rebuild", methods=["POST"])
def api_rebuild_global():
    """Rebuild the global speaker library from all existing transcriptions."""
    from diarization import rebuild_global_library
    result = rebuild_global_library()
    return jsonify(result)




@app.route("/summarize/<path:filename>", methods=["POST"])
def run_summarize(filename):
    """Trigger LLM analysis for a completed transcription."""
    from summarizer import summarize
    import json, os
    result_path = f"/recordings/.transcriptions/{filename}.result.json"
    if not os.path.exists(result_path):
        return jsonify({"error": "transcription not found"}), 404
    with open(result_path) as f:
        result = json.load(f)
    if result.get("status") != "completed":
        return jsonify({"error": "transcription not yet completed"}), 400
    def _run():
        try:
            analysis = summarize(result)
            result["llm_analysis"] = analysis
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Summarizer: analysis saved for {filename}")
        except Exception as e:
            logger.error(f"Summarizer failed: {e}")
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status": "started"}), 202


@app.route("/summarize/<path:filename>/result")
def summarize_result(filename):
    """Get LLM analysis result."""
    import json, os
    result_path = f"/recordings/.transcriptions/{filename}.result.json"
    if not os.path.exists(result_path):
        return jsonify({"error": "not found"}), 404
    with open(result_path) as f:
        result = json.load(f)
    analysis = result.get("llm_analysis")
    if analysis is None:
        return jsonify({"status": "not_available"})
    return jsonify(analysis)


# ── Pipeline Routes (composable stages) ────────────────────────────────

ALL_STAGES = {"preprocess", "diarize", "asr", "align", "postprocess"}

@app.route("/pipeline/<path:filename>/status")
def pipeline_status(filename):
    """Check which stages have been completed for a file."""
    from pipeline import get_pipeline_status
    return jsonify(get_pipeline_status(filename))


@app.route("/pipeline/<path:filename>/<stage>", methods=["POST"])
def pipeline_stage(filename, stage):
    """Run a single pipeline stage: preprocess, diarize, asr, align, or postprocess."""
    from pipeline import STAGE_FUNCS

    if stage not in ALL_STAGES:
        return jsonify({"error": f"unknown_stage:{stage}. Valid: {', '.join(sorted(ALL_STAGES))}"}), 400

    audio_path = RECORDINGS_DIR / filename
    if not audio_path.exists():
        return jsonify({"error": "file_not_found"}), 404
    if not is_audio(filename):
        return jsonify({"error": "not_an_audio_file"}), 400

    def _run():
        try:
            fn = STAGE_FUNCS[stage]
            fn(filename)
        except Exception as e:
            logger.error(f"Pipeline stage {stage} failed: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status": "started", "stage": stage, "filename": filename}), 202


@app.route("/pipeline/<path:filename>/run", methods=["POST"])
def pipeline_run(filename):
    """Run full pipeline or selected stages. Query param: ?stages=preprocess,diarize,asr"""
    from pipeline import STAGE_ORDER, run_pipeline

    audio_path = RECORDINGS_DIR / filename
    if not audio_path.exists():
        return jsonify({"error": "file_not_found"}), 404
    if not is_audio(filename):
        return jsonify({"error": "not_an_audio_file"}), 400

    stages_param = request.args.get("stages", "")
    if stages_param:
        stages = [s.strip() for s in stages_param.split(",") if s.strip()]
        invalid = [s for s in stages if s not in ALL_STAGES]
        if invalid:
            return jsonify({"error": f"unknown_stages:{invalid}"}), 400
    else:
        stages = None

    def _run():
        try:
            run_pipeline(filename, stages=stages)
        except Exception as e:
            logger.error(f"Pipeline run failed: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({
        "status": "started",
        "filename": filename,
        "stages": stages or list(STAGE_ORDER),
    }), 202


@app.route("/pipeline/<path:filename>/result")
def pipeline_result(filename):
    """Get the final pipeline result (same as transcribe result)."""
    result_path = RECORDINGS_DIR / ".transcriptions" / f"{filename}.result.json"
    if not result_path.exists():
        return jsonify({"error": "not_found", "message": "Pipeline has not completed yet"}), 404
    with open(result_path) as f:
        return jsonify(json.load(f))


@app.route("/pipeline/<path:filename>/stage_result/<stage>")
def pipeline_stage_result(filename, stage):
    """Get an intermediate stage result."""
    from pipeline import _read_intermediate
    if stage not in ALL_STAGES:
        return jsonify({"error": f"unknown_stage:{stage}"}), 400
    data = _read_intermediate(filename, stage)
    if data is None:
        return jsonify({"error": "stage_not_completed", "stage": stage}), 404
    return jsonify(data)


# ── Entry point ────────────────────────────────────────────────────────

# ── Metadata API (labels & groups) ─────────────────────────────────────

@app.route("/api/recordings/<path:filename>/labels", methods=["GET", "POST"])
def api_recording_labels(filename):
    """Get or set labels on a recording. POST body: {"labels": ["meeting", "project-x"]}"""
    if request.method == "GET":
        return jsonify({
            "filename": filename,
            "labels": metadata.get_labels(filename),
        })
    data = request.get_json() or {}
    labels = data.get("labels", [])
    if not isinstance(labels, list):
        return jsonify({"error": "labels must be a list"}), 400
    result = metadata.set_labels(filename, labels)
    return jsonify(result)


@app.route("/api/recordings/<path:filename>/labels/<label>", methods=["POST", "DELETE"])
def api_recording_label(filename, label):
    """Add or remove a single label."""
    if request.method == "POST":
        result = metadata.add_label(filename, label)
    else:
        result = metadata.remove_label(filename, label)
    return jsonify(result)


@app.route("/api/recordings/<path:filename>/group", methods=["GET", "POST"])
def api_recording_group(filename):
    """Get or set a recording's group. POST body: {"group": "alpha"}"""
    if request.method == "GET":
        return jsonify({
            "filename": filename,
            "group": metadata.get_group(filename),
        })
    data = request.get_json() or {}
    group = data.get("group", None)
    result = metadata.set_group(filename, group)
    return jsonify(result)


@app.route("/api/recordings/<path:filename>/notes", methods=["GET", "POST"])
def api_recording_notes(filename):
    """Get or set notes on a recording. POST body: {"notes": "..."}"""
    if request.method == "GET":
        meta = metadata.get_metadata(filename)
        return jsonify({"filename": filename, "notes": meta.get("notes", "")})
    data = request.get_json() or {}
    result = metadata.set_notes(filename, data.get("notes", ""))
    return jsonify(result)


@app.route("/api/groups", methods=["GET", "POST"])
def api_groups():
    """List all groups or create a new one.
    POST body: {"name": "Alpha", "label": "Project Alpha", "color": "#58a6ff", "description": "..."}"""
    if request.method == "GET":
        groups = metadata.get_groups()
        # Enrich with recording counts
        all_meta = metadata.get_all_metadata()
        counts = {}
        for fname, m in all_meta.items():
            g = m.get("group")
            if g:
                counts[g] = counts.get(g, 0) + 1
        result = {}
        for key, g in groups.items():
            result[key] = dict(g, recording_count=counts.get(key, 0))
        return jsonify(result)

    data = request.get_json() or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    group = metadata.create_group(
        name=name,
        label=data.get("label", name),
        color=data.get("color", "#58a6ff"),
        description=data.get("description", ""),
    )
    return jsonify(group), 201


@app.route("/api/groups/<name>", methods=["DELETE"])
def api_group_delete(name):
    """Delete a group."""
    if metadata.delete_group(name):
        return jsonify({"status": "deleted", "group": name})
    return jsonify({"error": "group_not_found"}), 404


@app.route("/api/recordings")
def api_recordings_list():
    """List recordings with metadata, supports filtering.
    Query params: ?label=meeting&group=alpha&labels=meeting,standup"""
    label = request.args.get("label", "").strip() or None
    group = request.args.get("group", "").strip() or None
    labels_param = request.args.get("labels", "").strip() or None
    labels = [l.strip() for l in labels_param.split(",") if l.strip()] if labels_param else None

    # If filters provided, only return matching filenames
    if label or group or labels:
        fnames = metadata.filter_recordings(label=label, group=group, labels=labels)
        result = []
        for fname in fnames:
            path = RECORDINGS_DIR / fname
            if not path.exists():
                continue
            stat = path.stat()
            trans_status = transcriber.get_status(fname)
            meta = metadata.get_metadata(fname)
            result.append({
                "name": fname,
                "size": human_size(stat.st_size),
                "size_bytes": stat.st_size,
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                "url": url_for("serve_file", filename=fname),
                "transcription": trans_status,
                "labels": meta.get("labels", []),
                "group": meta.get("group") or None,
                "group_info": meta.get("group_info") or None,
            })
        return jsonify(result)

    # No filters — return all with metadata
    all_meta = metadata.get_all_metadata()
    result = []
    if RECORDINGS_DIR.exists():
        for f in sorted(RECORDINGS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.is_file() and is_allowed(f.name):
                stat = f.stat()
                trans_status = transcriber.get_status(f.name)
                speakers = []
                if trans_status.get("status") == "completed":
                    segs = trans_status.get("sentences", [])
                    speakers = sorted(set(s["speaker"] for s in segs))
                m = all_meta.get(f.name, {})
                result.append({
                    "name": f.name,
                    "size": human_size(stat.st_size),
                    "size_bytes": stat.st_size,
                    "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                    "url": url_for("serve_file", filename=f.name),
                    "transcription": trans_status,
                    "speakers": speakers,
                    "labels": m.get("labels", []),
                    "group": m.get("group") or None,
                    "group_info": m.get("group_info") or None,
                })
    return jsonify(result)


@app.route("/api/labels")
def api_all_labels():
    """Get all labels used across recordings."""
    return jsonify({"labels": metadata.get_all_labels()})


@app.route("/api/metadata/rebuild", methods=["POST"])
def api_metadata_rebuild():
    """Prune metadata for recordings that no longer exist on disk."""
    removed = metadata.rebuild_from_disk()
    return jsonify({"status": "ok", "orphaned_entries_removed": removed})



if __name__ == "__main__":
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=8080, debug=True)

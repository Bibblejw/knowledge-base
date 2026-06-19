"""Knowledge Base — Audio Recordings Vault.

A self-hosted web app for uploading, browsing, playing, and transcribing
audio recordings. Built to grow into a broader knowledge management system.
"""

import os
import time
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

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────

RECORDINGS_DIR = Path("/recordings")
ALLOWED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".aac",
    ".m4a",
    ".opus",
    ".wma",
    ".aiff",
    ".alac",
}

HF_TOKEN = os.environ.get("HF_TOKEN", "")
LEMONFOX_API_KEY = os.environ.get("LEMONFOX_API_KEY", "")

app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB


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


def list_recordings() -> list[dict]:
    """Return sorted list of recording info dicts."""
    recordings = []
    if not RECORDINGS_DIR.exists():
        return recordings

    for f in RECORDINGS_DIR.iterdir():
        if f.is_file() and is_allowed(f.name):
            stat = f.stat()
            trans_status = transcriber.get_status(f.name)
            recordings.append(
                {
                    "name": f.name,
                    "size": human_size(stat.st_size),
                    "size_bytes": stat.st_size,
                    "modified": time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)
                    ),
                    "url": url_for("serve_file", filename=f.name),
                    "transcription": trans_status,
                }
            )

    recordings.sort(key=lambda r: r["name"].lower())
    return recordings


# ── Main Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", recordings=list_recordings())


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
    return "", 204


# ── Transcription Routes ───────────────────────────────────────────────

@app.route("/transcribe/<path:filename>", methods=["POST"])
def transcribe_file(filename):
    """Start transcription for a single file."""
    audio_path = RECORDINGS_DIR / filename
    if not audio_path.exists():
        return jsonify({"error": "File not found"}), 404

    if not is_audio(filename):
        return jsonify({"error": "Not an audio file"}), 400

    # Check if already completed
    status = transcriber.get_status(filename)
    if status["status"] == "completed":
        return jsonify({"status": "completed", "message": "Already transcribed"})

    if status["status"] == "processing":
        return jsonify({"status": "processing", "message": "Already in progress"})

    # Launch background transcription
    def _run():
        transcriber.run_transcription(filename)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({"status": "started"}), 202


@app.route("/transcribe/<path:filename>/status")
def transcribe_status(filename):
    """Get transcription status for a file."""
    status = transcriber.get_status(filename)
    return jsonify(status)


@app.route("/transcribe/<path:filename>/result")
def transcribe_result(filename):
    """Get full transcription result."""
    status = transcriber.get_status(filename)
    return jsonify(status)


# ── Speaker enrollment ─────────────────────────────────────────────────

@app.route("/speakers", methods=["GET"])
def speaker_list():
    """List enrolled speakers."""
    from diarization import list_speakers
    speakers = list_speakers()
    return jsonify({"speakers": speakers, "count": len(speakers)})


@app.route("/speakers/enroll", methods=["POST"])
def speaker_enroll():
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


@app.route("/speakers/<name>", methods=["DELETE"])
def speaker_remove(name):
    """Remove an enrolled speaker."""
    from diarization import remove_speaker
    if remove_speaker(name):
        return jsonify({"status": "removed"})
    return jsonify({"error": "Speaker not found"}), 404


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=8080, debug=True)

"""Knowledge Base — Audio Recordings Vault.

A self-hosted web app for uploading, browsing, and playing audio recordings.
Built to grow into a broader knowledge management system.
"""

import os
import time
from pathlib import Path

from flask import (
    Flask,
    abort,
    render_template,
    request,
    send_from_directory,
    url_for,
)

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

app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB


# ── Helpers ────────────────────────────────────────────────────────────

def is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


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
            recordings.append(
                {
                    "name": f.name,
                    "size": human_size(stat.st_size),
                    "size_bytes": stat.st_size,
                    "modified": time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)
                    ),
                    "url": url_for("serve_file", filename=f.name),
                }
            )

    recordings.sort(key=lambda r: r["name"].lower())
    return recordings


# ── Routes ─────────────────────────────────────────────────────────────

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


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=8080, debug=True)

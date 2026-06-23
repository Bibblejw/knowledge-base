"""
Metadata — labels, groups, and annotation for recordings.

Stores per-recording metadata as a single JSON file:
  /recordings/.metadata/metadata.json

Structure:
{
  "groups": {
    "alpha": {"name": "alpha", "label": "Project Alpha", "color": "#58a6ff", "description": ""},
    ...
  },
  "recordings": {
    "09-01-32.m4a": {
      "labels": ["meeting", "standup"],
      "group": "alpha",
      "notes": "",
      "created_at": "2025-06-16T10:00:00"
    },
    ...
  }
}
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

METADATA_DIR = Path("/recordings/.metadata")
METADATA_FILE = METADATA_DIR / "metadata.json"


def _load() -> dict:
    """Load the metadata file, or return empty defaults."""
    if not METADATA_FILE.exists():
        return {"groups": {}, "recordings": {}}
    with open(METADATA_FILE) as f:
        return json.load(f)


def _save(data: dict):
    """Save the metadata file atomically."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = METADATA_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, METADATA_FILE)


def _ensure_recording(data: dict, filename: str) -> dict:
    """Ensure a recording entry exists, returning it."""
    if filename not in data["recordings"]:
        data["recordings"][filename] = {
            "labels": [],
            "group": None,
            "notes": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    return data["recordings"][filename]


# ═══════════════════════════════════════════════════════════════════════
# Labels
# ═══════════════════════════════════════════════════════════════════════

def get_labels(filename: str) -> list[str]:
    """Get labels for a recording."""
    data = _load()
    rec = data["recordings"].get(filename, {})
    return rec.get("labels", [])


def set_labels(filename: str, labels: list[str]) -> dict:
    """Replace all labels for a recording. Returns full metadata."""
    data = _load()
    rec = _ensure_recording(data, filename)
    rec["labels"] = sorted(set(l.strip().lower() for l in labels if l.strip()))
    _save(data)
    return rec


def add_label(filename: str, label: str) -> dict:
    """Add a single label to a recording."""
    data = _load()
    rec = _ensure_recording(data, filename)
    label = label.strip().lower()
    if label and label not in rec["labels"]:
        rec["labels"].append(label)
        rec["labels"] = sorted(rec["labels"])
        _save(data)
    return rec


def remove_label(filename: str, label: str) -> dict:
    """Remove a single label from a recording."""
    data = _load()
    rec = _ensure_recording(data, filename)
    label = label.strip().lower()
    if label in rec["labels"]:
        rec["labels"].remove(label)
        _save(data)
    return rec


def get_all_labels() -> list[str]:
    """Return all unique labels in use across all recordings."""
    data = _load()
    all_labels = set()
    for rec in data["recordings"].values():
        for lbl in rec.get("labels", []):
            all_labels.add(lbl)
    return sorted(all_labels)


# ═══════════════════════════════════════════════════════════════════════
# Groups
# ═══════════════════════════════════════════════════════════════════════

def get_groups() -> dict:
    """Get all groups as {key: {name, label, color, description}}."""
    data = _load()
    return data.get("groups", {})


def create_group(name: str, label: str = None, color: str = "#58a6ff", description: str = "") -> dict:
    """Create or update a group. Returns the group dict."""
    key = name.strip().lower().replace(" ", "-")
    if not key:
        raise ValueError("Group name required")
    data = _load()
    group = {
        "name": key,
        "label": label or name.strip(),
        "color": color,
        "description": description.strip(),
    }
    data.setdefault("groups", {})[key] = group
    _save(data)
    return group


def delete_group(name: str) -> bool:
    """Delete a group. Un-assigns it from all recordings."""
    key = name.strip().lower().replace(" ", "-")
    data = _load()
    groups = data.get("groups", {})
    if key not in groups:
        return False
    del groups[key]
    # Un-assign from recordings
    for rec in data["recordings"].values():
        if rec.get("group") == key:
            rec["group"] = None
    _save(data)
    return True


def set_group(filename: str, group: str) -> dict:
    """Assign a recording to a group. group=None to unset."""
    data = _load()
    rec = _ensure_recording(data, filename)
    if group is not None:
        group = group.strip().lower().replace(" ", "-")
    rec["group"] = group
    _save(data)
    return rec


def get_group(filename: str) -> str | None:
    """Get the group a recording belongs to, or None."""
    data = _load()
    rec = data["recordings"].get(filename, {})
    return rec.get("group")


# ═══════════════════════════════════════════════════════════════════════
# Notes
# ═══════════════════════════════════════════════════════════════════════

def set_notes(filename: str, notes: str) -> dict:
    """Set freeform notes on a recording."""
    data = _load()
    rec = _ensure_recording(data, filename)
    rec["notes"] = notes.strip()
    _save(data)
    return rec


# ═══════════════════════════════════════════════════════════════════════
# Query
# ═══════════════════════════════════════════════════════════════════════

def filter_recordings(
    label: str = None,
    group: str = None,
    labels: list[str] = None,
) -> list[str]:
    """
    Return filenames matching filters.
    - label: recordings that have this label
    - group: recordings in this group
    - labels: recordings that match ALL of these labels (AND)
    """
    data = _load()
    results = []
    for fname, rec in data["recordings"].items():
        if label and label not in rec.get("labels", []):
            continue
        if labels:
            rec_labels = set(rec.get("labels", []))
            if not all(l in rec_labels for l in labels):
                continue
        if group and rec.get("group") != group:
            continue
        results.append(fname)
    return sorted(results)


def get_metadata(filename: str) -> dict:
    """Get full metadata for a recording, with group label resolved."""
    data = _load()
    rec = data["recordings"].get(filename, {
        "labels": [],
        "group": None,
        "notes": "",
        "created_at": None,
    })
    group_key = rec.get("group")
    group_info = None
    if group_key:
        group_info = data.get("groups", {}).get(group_key)
    return {
        "labels": rec.get("labels", []),
        "group": group_key,
        "group_info": group_info,
        "notes": rec.get("notes", ""),
        "created_at": rec.get("created_at"),
    }


def get_all_metadata() -> dict:
    """Get metadata for all recordings (lightweight, for listing)."""
    data = _load()
    result = {}
    for fname, rec in data["recordings"].items():
        group_key = rec.get("group")
        result[fname] = {
            "labels": rec.get("labels", []),
            "group": group_key,
            "group_info": data.get("groups", {}).get(group_key) if group_key else None,
            "notes": rec.get("notes", ""),
        }
    return result


def delete_metadata(filename: str):
    """Remove metadata for a recording (called when file is deleted)."""
    data = _load()
    data["recordings"].pop(filename, None)
    _save(data)


def rebuild_from_disk() -> int:
    """Prune metadata entries for recordings that no longer exist on disk."""
    recordings_dir = Path("/recordings")
    existing = {f.name for f in recordings_dir.iterdir() if f.is_file() and not f.name.startswith(".")}
    data = _load()
    orphaned = [f for f in data["recordings"] if f not in existing]
    for f in orphaned:
        del data["recordings"][f]
    _save(data)
    return len(orphaned)

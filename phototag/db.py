"""SQLite store for everything derived from the album. Lives in data/, never leaves the machine."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

IGNORE = "!ignore"  # person_id for detections that are not real faces
OTHER = "!other"    # person_id for real faces that are not family (classmates, strangers): never matched, photo stays private
LOCAL = "local:"    # person_id prefix for people named only on this machine (living relatives, not in the CC7)

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos(
    id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, rel TEXT NOT NULL, sha1 TEXT,
    dup_of INTEGER, width INTEGER, height INTEGER, year INTEGER, year_src TEXT,
    folder_person TEXT, error TEXT, text_hint TEXT);
CREATE TABLE IF NOT EXISTS faces(
    id INTEGER PRIMARY KEY, photo_id INTEGER NOT NULL REFERENCES photos(id),
    x REAL, y REAL, w REAL, h REAL, score REAL, emb BLOB,
    cluster_id INTEGER, person_id TEXT, label_src TEXT);
CREATE INDEX IF NOT EXISTS faces_photo ON faces(photo_id);
CREATE INDEX IF NOT EXISTS faces_cluster ON faces(cluster_id);
CREATE INDEX IF NOT EXISTS faces_person ON faces(person_id);
CREATE INDEX IF NOT EXISTS photos_sha1 ON photos(sha1);
"""


def connect(data_dir: Path) -> sqlite3.Connection:
    data_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(data_dir / "phototag.db", check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    have = {r["name"] for r in db.execute("PRAGMA table_info(faces)")}
    for col, typ in (("sat", "REAL"), ("labeled_at", "TEXT")):   # added after the first release
        if col not in have:
            db.execute(f"ALTER TABLE faces ADD COLUMN {col} {typ}")
    if "era" not in {r["name"] for r in db.execute("PRAGMA table_info(photos)")}:
        db.execute("ALTER TABLE photos ADD COLUMN era INTEGER")   # decade a vision model guessed, e.g. 1950
    db.commit()
    return db


def emb_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def blob_to_emb(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def load_config(data_dir: Path) -> dict:
    p = data_dir / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_config(data_dir: Path, cfg: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "config.json").write_text(json.dumps(cfg, indent=2))

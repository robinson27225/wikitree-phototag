"""Read handwriting and captions with a local open vision model (Gemma 3 via Ollama), then look for relatives' names in it."""
from __future__ import annotations

import base64
import io
import json
import sqlite3
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

OLLAMA = "http://127.0.0.1:11434"
MODEL = "gemma3:4b"
PROMPT = ("This is a scan from a family photo album. If there is handwriting or printed text on it "
          "(for example a caption on the back of a photograph), transcribe it exactly as written. "
          "If there is no handwriting or caption, answer NONE. Reply with only the transcription.")
Image.MAX_IMAGE_PIXELS = None
PHOTO_PROMPT = ("This is a scanned family photograph. Reply with JSON only, with two keys. "
                '"text": any handwriting or printed caption visible on it, transcribed exactly, or "" if there is none. '
                '"decade": your best estimate of the decade it was taken as a number like 1950, judging from clothing, '
                "hairstyles, photo quality and surroundings, or null if you cannot tell.")
_NOISE = ("converter trial", "processtext.com")


def _encode(path: str) -> str:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def transcribe(path: str, model: str = MODEL) -> str:
    body = {"model": model, "stream": False, "options": {"temperature": 0}, "prompt": PROMPT, "images": [_encode(path)]}
    req = urllib.request.Request(f"{OLLAMA}/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    text = json.load(urllib.request.urlopen(req, timeout=600))["response"].strip()
    if text.upper().startswith("NONE") or any(n in text.lower() for n in _NOISE):
        return ""
    return text


def analyze(path: str, model: str = MODEL) -> tuple[str, int | None]:
    """(caption text, estimated decade) for a photo, in one call."""
    body = {"model": model, "stream": False, "format": "json", "options": {"temperature": 0},
            "prompt": PHOTO_PROMPT, "images": [_encode(path)]}
    req = urllib.request.Request(f"{OLLAMA}/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    raw = json.load(urllib.request.urlopen(req, timeout=600))["response"]
    try:
        data = json.loads(raw)
    except ValueError:
        return "", None
    text = str(data.get("text") or "").strip()
    if text.upper().startswith("NONE") or any(n in text.lower() for n in _NOISE):
        text = ""
    try:
        decade = int(data.get("decade"))
        decade = decade // 10 * 10 if 1840 <= decade <= 2020 else None
    except (TypeError, ValueError):
        decade = None
    return text, decade


def read_text(db: sqlite3.Connection, only_faceless: bool = True, limit: int | None = None, model: str = MODEL) -> int:
    """Faceless scans: caption text only. With only_faceless=False (photos with faces): caption text plus a rough era."""
    where = "p.dup_of IS NULL AND p.error IS NULL AND p.text_hint IS NULL"
    if only_faceless:
        where += " AND NOT EXISTS (SELECT 1 FROM faces f WHERE f.photo_id=p.id)"
    rows = db.execute(f"SELECT p.id, p.path FROM photos p WHERE {where} ORDER BY p.rel" + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    print(f"reading text on {len(rows)} scans with {model} (local, via Ollama)")
    found = 0
    for i, r in enumerate(rows, 1):
        try:
            if only_faceless:
                text, era = transcribe(r["path"], model), None
            else:
                text, era = analyze(r["path"], model)
        except Exception as e:
            raise SystemExit(f"Ollama not reachable or model missing ({e}). Start it with `ollama serve` and `ollama pull {model}`.")
        db.execute("UPDATE photos SET text_hint=?, era=? WHERE id=?", (text, era, r["id"]))
        found += bool(text)
        if i % 20 == 0:
            db.commit()
            print(f"  {i}/{len(rows)}  with text: {found}")
    db.commit()
    return found

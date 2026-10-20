"""Walk the album, drop duplicates, find every face, and store an embedding for each."""
from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import models
from .cc7 import Roster
from .db import emb_to_blob

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}
MAX_SIDE = 1600       # detection runs on a downscaled copy; boxes are stored as fractions
MIN_FACE_PX = 36      # smaller faces give unreliable embeddings
THUMB_PX = 128
_YEAR = re.compile(r"(?<!\d)(18[5-9]\d|19\d\d|20[0-2]\d)(?!\d)")

Image.MAX_IMAGE_PIXELS = None


def mean_saturation(bgr: np.ndarray) -> float:
    """0 = black-and-white, higher = more colourful. A cheap hint that a print is old."""
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255.0)


def backfill_saturation(db: sqlite3.Connection, data_dir: Path) -> None:
    rows = db.execute("SELECT id FROM faces WHERE sat IS NULL").fetchall()
    for r in rows:
        img = cv2.imread(str(data_dir / "thumbs" / f"{r['id']}.jpg"))
        if img is not None:
            db.execute("UPDATE faces SET sat=? WHERE id=?", (mean_saturation(img), r["id"]))
    db.commit()


def iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."):
            yield p


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def year_hint(img: Image.Image, rel: Path) -> tuple[int | None, str | None]:
    """A year for the photo, if the file says so. Scanner timestamps are ignored on purpose."""
    try:
        exif = img.getexif()
        taken = exif.get_ifd(0x8769).get(36867) or ""   # DateTimeOriginal
        m = _YEAR.search(str(taken))
        if m and int(m.group(1)) < 2000:               # a 1950s print "taken" in 2018 is a scan date
            return int(m.group(1)), "exif"
    except Exception:
        pass
    years = set(_YEAR.findall(rel.stem))
    if len(years) == 1:
        return int(years.pop()), "filename"
    return None, None


def load_rgb(path: Path) -> tuple[Image.Image, Image.Image]:
    """(oriented original for metadata, RGB working copy)."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I;16", "I", "F"):
        arr = np.asarray(img, dtype=np.float32)
        arr = (arr - arr.min()) / max(float(np.ptp(arr)), 1.0) * 255
        img = Image.fromarray(arr.astype(np.uint8))
    return img, img.convert("RGB")


def detect_faces(detector, recognizer, rgb: Image.Image):
    """Yield (box_fractions, score, embedding, bgr_crop) for each face found."""
    w0, h0 = rgb.size
    scale = min(1.0, MAX_SIDE / max(w0, h0))
    small = rgb.resize((round(w0 * scale), round(h0 * scale))) if scale < 1 else rgb
    bgr = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    detector.setInputSize((w, h))
    _, found = detector.detect(bgr)
    for row in (found if found is not None else []):
        x, y, fw, fh = row[:4]
        if min(fw, fh) < MIN_FACE_PX:
            continue
        emb = recognizer.feature(recognizer.alignCrop(bgr, row))[0]
        emb = emb / (np.linalg.norm(emb) or 1.0)
        pad = 0.3
        x0, y0 = max(0, int(x - fw * pad)), max(0, int(y - fh * pad))
        x1, y1 = min(w, int(x + fw * (1 + pad))), min(h, int(y + fh * (1 + pad)))
        crop = cv2.resize(bgr[y0:y1, x0:x1], (THUMB_PX, THUMB_PX))
        box = tuple(float(v) for v in (x / w, y / h, fw / w, fh / h))  # numpy scalars would be stored as BLOBs
        yield box, float(row[-1]), emb, crop


def scan(root: Path, db: sqlite3.Connection, roster: Roster, data_dir: Path, models_dir: Path, limit: int | None = None):
    detector, recognizer = models.load(models_dir)
    thumbs = data_dir / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    known = {r["path"] for r in db.execute("SELECT path FROM photos")}
    files = [p for p in iter_images(root) if str(p) not in known]
    if limit:
        files = files[:limit]
    print(f"{len(files)} new images to scan ({len(known)} already done)")
    n_faces = n_dups = 0
    for i, path in enumerate(files, 1):
        rel = path.relative_to(root)
        top = rel.parts[0] if len(rel.parts) > 1 else ""
        person = roster.from_folder_name(top) if top else None
        try:
            digest = sha1_of(path)
            dup = db.execute("SELECT id FROM photos WHERE sha1=? AND dup_of IS NULL", (digest,)).fetchone()
            if dup:
                db.execute("INSERT INTO photos(path,rel,sha1,dup_of,folder_person) VALUES(?,?,?,?,?)",
                           (str(path), str(rel), digest, dup["id"], person.id if person else None))
                n_dups += 1
                continue
            img, rgb = load_rgb(path)
            year, src = year_hint(img, rel)
            cur = db.execute(
                "INSERT INTO photos(path,rel,sha1,width,height,year,year_src,folder_person) VALUES(?,?,?,?,?,?,?,?)",
                (str(path), str(rel), digest, rgb.width, rgb.height, year, src, person.id if person else None))
            photo_id = cur.lastrowid
            for box, score, emb, crop in detect_faces(detector, recognizer, rgb):
                fc = db.execute("INSERT INTO faces(photo_id,x,y,w,h,score,emb,sat) VALUES(?,?,?,?,?,?,?,?)",
                                (photo_id, *box, score, emb_to_blob(emb), mean_saturation(crop)))
                cv2.imwrite(str(thumbs / f"{fc.lastrowid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                n_faces += 1
        except Exception as e:  # a corrupt scan should not stop a 3 GB album
            db.execute("INSERT OR REPLACE INTO photos(path,rel,error) VALUES(?,?,?)", (str(path), str(rel), str(e)[:200]))
        if i % 25 == 0:
            db.commit()
            print(f"  {i}/{len(files)}  faces so far: {n_faces}  duplicates skipped: {n_dups}")
    db.commit()
    print(f"done: {n_faces} faces, {n_dups} duplicate files skipped")

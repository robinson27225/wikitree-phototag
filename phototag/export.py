"""The privacy gate. A photo is exported only if every face in it is named and every named person is deceased."""
from __future__ import annotations

import csv
import re
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from .cc7 import Roster
from .db import IGNORE, LOCAL, OTHER
from .xmp import build_xmp

WEB_FORMATS = {".jpg", ".jpeg", ".png", ".gif"}  # WikiTree accepts these; anything else is converted to JPEG


@dataclass
class Decision:
    photo_id: int
    path: str
    rel: str
    year: int | None
    width: int
    height: int
    people: list[str] = field(default_factory=list)
    regions: list[dict] = field(default_factory=list)
    export: bool = False
    kind: str = "ready"   # ready | unnamed | living | auto
    reason: str = ""


def plan(db: sqlite3.Connection, roster: Roster, include_auto: bool = False) -> list[Decision]:
    decisions = []
    photos = db.execute("SELECT * FROM photos WHERE dup_of IS NULL AND error IS NULL AND width IS NOT NULL ORDER BY rel").fetchall()
    for p in photos:
        faces = db.execute("SELECT * FROM faces WHERE photo_id=?", (p["id"],)).fetchall()
        real = [f for f in faces if f["person_id"] != IGNORE]
        if not real:
            continue  # no people in it (document, landscape, ...)
        d = Decision(p["id"], p["path"], p["rel"], p["year"], p["width"], p["height"])
        d.people = sorted({f["person_id"] for f in real if f["person_id"] and f["person_id"] != OTHER})
        d.regions = [{"person_id": f["person_id"], "x": f["x"], "y": f["y"], "w": f["w"], "h": f["h"]}
                     for f in real if f["person_id"] and f["person_id"] != OTHER]
        unknown = sum(1 for f in real if not f["person_id"])
        others = sum(1 for f in real if f["person_id"] == OTHER)
        living = [pid for pid in d.people if not roster.is_deceased(pid)]
        if unknown:
            d.kind, d.reason = "unnamed", f"{unknown} face(s) not yet named"
        elif others:
            d.kind, d.reason = "other", f"{others} face(s) set aside as not family (could be living)"
        elif living:
            d.kind, d.reason = "living", "possibly living: " + ", ".join(living)
        elif not include_auto and any(f["label_src"] == "auto" for f in real):
            d.kind, d.reason = "auto", "contains names nobody confirmed (auto-accepted)"
        else:
            d.export = True
        decisions.append(d)
    return decisions


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def _wt(person) -> str:
    return f"[[{person.id}|{person.name}]]"


def caption_for(pid: str, d: Decision, roster: Roster) -> str:
    me = roster.get(pid)
    others = [roster.get(o) for o in d.people if o != pid and roster.get(o)]
    text = _wt(me)
    if others:
        text += " with " + ", ".join(_wt(o) for o in others[:-1]) + (" and " if len(others) > 1 else "") + _wt(others[-1])
    if d.year:
        text += f", {d.year}"
    return text + "."


def export(db: sqlite3.Connection, roster: Roster, out: Path, report_dir: Path, include_auto: bool = False) -> dict:
    decisions = plan(db, roster, include_auto)
    out.mkdir(parents=True, exist_ok=True)
    rows, withheld = [], []
    for d in decisions:
        if not d.export:
            withheld.append({"file": d.rel, "reason": d.reason})
            continue
        src = Path(d.path)
        for pid in d.people:
            folder = out / pid
            folder.mkdir(exist_ok=True)
            ext = src.suffix.lower()
            name = f"{pid}_{_safe(src.stem)}{'.jpg' if ext not in WEB_FORMATS else ext}"
            dest = folder / name
            if ext in WEB_FORMATS:
                shutil.copy2(src, dest)
            else:
                img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
                img.save(dest, "JPEG", quality=92)
            regions = [{"name": roster.get(r["person_id"]).name, **{k: r[k] for k in "xywh"}} for r in d.regions]
            dest.with_suffix(".xmp").write_text(
                build_xmp(d.width, d.height, regions, d.people, caption_for(pid, d, roster)), encoding="utf-8")
            rows.append({"profile": pid, "file": f"{pid}/{name}", "caption": caption_for(pid, d, roster),
                         "people": " ".join(d.people), "year": d.year or "", "source": d.rel})
    with open(out / "captions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["profile", "file", "caption", "people", "year", "source"])
        w.writeheader()
        w.writerows(rows)
    with open(report_dir / "withheld.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["file", "reason"])
        w.writeheader()
        w.writerows(withheld)
    return {"exported_photos": sum(d.export for d in decisions), "profile_files": len(rows),
            "profiles": len({r["profile"] for r in rows}), "withheld": len(withheld)}


def write_sidecars(db: sqlite3.Connection, roster: Roster, write: bool = False, force: bool = False,
                   include_auto: bool = False) -> dict:
    """Tag the album itself: `<photo>.jpg.xmp` beside every original that has named faces (digiKam / darktable /
    most photo managers read these). Originals are not touched. Living and locally named people are included,
    because these files stay on your machine; set-aside faces are not. Existing sidecars are never overwritten
    unless `force`. Without `write` nothing is written (a dry run)."""
    stats = {"photos": 0, "written": 0, "skipped_existing": 0}
    for p in db.execute("SELECT * FROM photos WHERE dup_of IS NULL AND error IS NULL AND width IS NOT NULL").fetchall():
        regions, subjects = [], []
        for f in db.execute("SELECT * FROM faces WHERE photo_id=? AND person_id IS NOT NULL", (p["id"],)):
            pid = f["person_id"]
            if pid in (IGNORE, OTHER) or (f["label_src"] == "auto" and not include_auto):
                continue
            if pid.startswith(LOCAL):
                name = pid[len(LOCAL):]
            elif roster.get(pid):
                name = roster.get(pid).name
                subjects.append(pid)
            else:
                continue
            regions.append({"name": name, **{k: f[k] for k in "xywh"}})
        if not regions:
            continue
        stats["photos"] += 1
        target = Path(p["path"] + ".xmp")
        if target.exists() and not force:
            stats["skipped_existing"] += 1
            continue
        if write:
            names = [r["name"] for r in regions]
            target.write_text(build_xmp(p["width"], p["height"], regions, sorted(set(subjects)), ", ".join(dict.fromkeys(names))), encoding="utf-8")
            stats["written"] += 1
    return stats

"""WikiTree PhotoTag: init -> models -> scan -> cluster -> label -> export."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import cluster as clustering
from . import db as dbm
from . import export as exporting
from . import backs, models, scan as scanning, web
from .cc7 import DEFAULT_PRESUMED_DECEASED_BEFORE, Roster

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    ap = argparse.ArgumentParser(prog="phototag", description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data", help="where the database and thumbnails live (stays local)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="remember your CC7 csv and photo folder")
    p.add_argument("--cc7", type=Path, required=True)
    p.add_argument("--photos", type=Path, required=True)
    p.add_argument("--presumed-deceased-before", type=int, default=DEFAULT_PRESUMED_DECEASED_BEFORE,
                   help="people born on or before this year with no death date count as deceased")
    sub.add_parser("models", help="download the two open face models (~38 MB)")
    p = sub.add_parser("scan", help="find faces in every photo"); p.add_argument("--limit", type=int)
    p = sub.add_parser("cluster", help="group similar faces"); p.add_argument("--threshold", type=float, default=clustering.DEFAULT_THRESHOLD)
    p = sub.add_parser("read-text", help="read captions/handwriting with local Gemma 3 (needs `ollama serve`)")
    p.add_argument("--all", action="store_true", help="also read photos that contain faces (slow)")
    p.add_argument("--limit", type=int); p.add_argument("--model", default=backs.MODEL)
    p = sub.add_parser("label", help="open the local naming UI"); p.add_argument("--port", type=int, default=8765)
    p.add_argument("--blur", action="store_true", help="pixelate every face except deceased named people (safe for screenshots)")
    p = sub.add_parser("accept", help="auto-name groups that clearly match people you already named")
    p.add_argument("--min", type=float, default=0.6, dest="min_score")
    p.add_argument("--hints", type=float, metavar="SHARE", help="also accept groups where one folder / full-name file name / caption hint covers this share of faces (e.g. 0.7)")
    p = sub.add_parser("xmp", help="write <photo>.xmp sidecars (names + face regions) next to your originals; dry run unless --write")
    p.add_argument("--write", action="store_true"); p.add_argument("--force", action="store_true", help="overwrite existing sidecars")
    p.add_argument("--include-auto", action="store_true")
    p = sub.add_parser("export", help="write photos of deceased relatives, one folder per WikiTree profile")
    p.add_argument("--out", type=Path, default=ROOT / "out")
    p.add_argument("--include-auto", action="store_true",
                   help="also export photos whose names were auto-accepted (not confirmed by you). Off by default: a wrong guess could put a living person under a deceased profile")
    sub.add_parser("seed", help="name faces in single-face photos whose file name is a full CC7 name")
    a = ap.parse_args(argv)

    cfg = dbm.load_config(a.data)
    if a.cmd == "init":
        # keep our own copy: the app should not break because a Downloads folder got tidied
        a.data.mkdir(parents=True, exist_ok=True)
        local = a.data / "cc7.csv"
        shutil.copyfile(a.cc7.expanduser(), local)
        dbm.save_config(a.data, {"cc7": str(local), "photos": str(a.photos.expanduser()),
                                 "cutoff": a.presumed_deceased_before})
        r = Roster.from_csv(local, a.presumed_deceased_before)
        dead = sum(r.is_deceased(pid) for pid in r.people)
        print(f"{len(r.people)} people in the CC7; {dead} treated as deceased, {len(r.people) - dead} withheld as possibly living")
        return
    if a.cmd == "models":
        return models.download(ROOT / "models")
    if not cfg:
        raise SystemExit("Run `phototag init --cc7 FILE --photos DIR` first.")
    if not Path(cfg["cc7"]).exists():
        raise SystemExit(f"CC7 file not found: {cfg['cc7']}\nRe-export your CC7 from WikiTree, then run:\n"
                         f"  phototag init --cc7 NEW_FILE.csv --photos {cfg['photos']}\n(your labels are safe in {a.data / 'phototag.db'})")
    roster = Roster.from_csv(cfg["cc7"], cfg["cutoff"])
    db = dbm.connect(a.data)
    if a.cmd == "scan":
        scanning.scan(Path(cfg["photos"]), db, roster, a.data, ROOT / "models", a.limit)
    elif a.cmd == "cluster":
        print(f"{clustering.cluster(db, a.threshold)} groups of faces")
    elif a.cmd == "read-text":
        print(f"{backs.read_text(db, not a.all, a.limit, a.model)} scans had readable text")
    elif a.cmd == "label":
        scanning.backfill_saturation(db, a.data)
        web.serve(db, roster, a.data, a.port, a.blur, ROOT / "models")
    elif a.cmd == "seed":
        print(f"seeded {clustering.seed_from_filenames(db, roster)} faces from file names")
    elif a.cmd == "accept":
        n = 0
        rows, total = clustering.cluster_summaries(db, roster, 0, 10_000)
        for c in rows:
            s = c["suggestions"][0] if c["suggestions"] else None
            pick = s["person_id"] if s and s["plausible"] and s["score"] >= a.min_score else None
            strong = [h for h in c["hints"] if h["strong"]]
            if not pick and a.hints and strong and strong[0]["faces"] / c["size"] >= a.hints and c["size"] >= 5:
                pick = strong[0]["person_id"]
            if pick:
                db.execute("UPDATE faces SET person_id=?, label_src='auto', labeled_at=datetime('now') WHERE cluster_id=? AND person_id IS NULL",
                           (pick, c["cluster_id"]))
                n += c["size"]
        db.commit()
        print(f"auto-named {n} faces")
    elif a.cmd == "xmp":
        r = exporting.write_sidecars(db, roster, a.write, a.force, a.include_auto)
        print(r if a.write else f"dry run: {r['photos']} photos have named faces ({r['skipped_existing']} already have a sidecar). Add --write to create them.")
    elif a.cmd == "export":
        summary = exporting.export(db, roster, a.out, a.data, a.include_auto)
        print(summary)
        print(f"upload-ready folders + captions.csv in {a.out}; withheld list in {a.data / 'withheld.csv'}")

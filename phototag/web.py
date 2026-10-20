"""Local labeling UI. Binds to 127.0.0.1 only; nothing is uploaded anywhere."""
from __future__ import annotations

import io
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import cluster as clustering
from . import export as exporting
from .cc7 import Roster
from .db import IGNORE, LOCAL, OTHER

STATIC = Path(__file__).parent / "static"


def pixelate(img: Image.Image, box: tuple[int, int, int, int] | None = None, blocks: int = 8) -> Image.Image:
    """Irreversibly pixelate a region (or the whole image): downscale to `blocks` cells, then upscale."""
    box = box or (0, 0, *img.size)
    x0, y0, x1, y1 = (max(0, box[0]), max(0, box[1]), min(img.width, box[2]), min(img.height, box[3]))
    if x1 <= x0 or y1 <= y0:
        return img
    region = img.crop((x0, y0, x1, y1))
    cells = max(1, min(blocks, region.width, region.height))
    img.paste(region.resize((cells, cells), Image.BILINEAR).resize(region.size, Image.NEAREST), (x0, y0))
    return img


def sensitive_detections(img: Image.Image, models_dir: Path) -> list[tuple[float, float, float, float]]:
    """Every face-like region at a very low confidence, as pixel boxes. Used only to over-blur."""
    det = cv2.FaceDetectorYN.create(str(models_dir / "face_detection_yunet_2023mar.onnx"), "", img.size, 0.3, 0.3, 300)
    _, found = det.detect(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    return [tuple(map(float, r[:4])) for r in (found if found is not None else [])]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter or 1.0)


def active_minutes(db: sqlite3.Connection, idle_gap_min: int = 5) -> float:
    """Time spent naming, counting only gaps shorter than `idle_gap_min` between consecutive labels."""
    ts = [r[0] for r in db.execute("SELECT DISTINCT labeled_at FROM faces WHERE labeled_at IS NOT NULL ORDER BY 1")]
    from datetime import datetime
    t = [datetime.fromisoformat(x) for x in ts]
    return round(sum(min((b - a).total_seconds(), idle_gap_min * 60) for a, b in zip(t, t[1:])) / 60 if len(t) > 1 else 0.0, 1)


def make_handler(db: sqlite3.Connection, roster: Roster, data_dir: Path, blur: bool = False, models_dir: Path | None = None,
                 clear_born_before: int | None = None):
    """blur=True hides every face except those named as deceased people, so the UI is safe to screenshot."""
    def is_public(person_id: str | None) -> bool:
        if not person_id or person_id in (IGNORE, OTHER) or not roster.is_deceased(person_id):
            return False
        if clear_born_before:   # screenshots: only the older generation stays unblurred
            p = roster.get(person_id)
            return bool(p and p.birth_year and p.birth_year < clear_born_before)
        return True

    def person_json(pid):
        if blur and pid != OTHER and not is_public(pid):   # screenshots must not name living relatives either
            return {"id": pid, "label": "Living relative (hidden)", "name": "Living relative", "deceased": False}
        p = roster.get(pid)
        if pid == OTHER:
            return {"id": pid, "label": "Not family / unknown (set aside)", "name": "Not family", "deceased": False}
        if pid.startswith(LOCAL):
            return {"id": pid, "label": f"{pid[len(LOCAL):]} (living / local only)", "name": pid[len(LOCAL):], "deceased": False}
        if not p:
            return {"id": pid, "label": pid, "deceased": False}
        return {"id": p.id, "label": p.label, "name": p.name, "deceased": roster.is_deceased(p.id)}

    lock = threading.Lock()   # one sqlite connection is shared by all request threads

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200):
            self._send(json.dumps(obj).encode(), "application/json", status)

        def do_GET(self):
            with lock:
                self._get()

        def do_POST(self):
            with lock:
                self._post()

        def _get(self):
            from urllib.parse import urlparse, parse_qs
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            parts = [x for x in u.path.split("/") if x]
            try:
                if not parts:
                    return self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if parts[0] == "thumb":
                    fid = int(parts[1].split('.')[0])
                    data = (data_dir / "thumbs" / f"{fid}.jpg").read_bytes()
                    if blur:
                        row = db.execute("SELECT person_id FROM faces WHERE id=?", (fid,)).fetchone()
                        if not (row and is_public(row["person_id"])):
                            buf = io.BytesIO()
                            pixelate(Image.open(io.BytesIO(data)).convert("RGB"), blocks=6).save(buf, "JPEG", quality=80)
                            data = buf.getvalue()
                    return self._send(data, "image/jpeg")
                if parts[0] == "photo":
                    return self._photo(int(parts[1].split(".")[0]), int(q.get("face", 0)))
                if parts[0] == "api":
                    return self._api_get(parts[1], q)
            except (FileNotFoundError, ValueError):
                return self._send(b"not found", "text/plain", 404)
            self._send(b"not found", "text/plain", 404)

        def _photo(self, photo_id: int, face_id: int):
            row = db.execute("SELECT path FROM photos WHERE id=?", (photo_id,)).fetchone()
            if not row:
                raise FileNotFoundError
            img = ImageOps.exif_transpose(Image.open(row["path"])).convert("RGB")
            img.thumbnail((1400, 1400))
            W, H = img.size
            faces = db.execute("SELECT * FROM faces WHERE photo_id=?", (photo_id,)).fetchall()
            if blur:  # hide before drawing anything on top
                m = 0.25
                def hide(x, y, w, h):
                    pixelate(img, (int(x - w * m), int(y - h * m), int(x + w * (1 + m)), int(y + h * (1 + m))))
                kept = []
                for f in faces:
                    box = (f["x"] * W, f["y"] * H, f["w"] * W, f["h"] * H)
                    if is_public(f["person_id"]):
                        kept.append(box)
                    else:
                        hide(*box)
                if models_dir:  # a second, very sensitive pass catches faces the main detector missed
                    for box in sensitive_detections(img.copy(), models_dir):
                        if not any(_iou(box, k) > 0.3 for k in kept):
                            hide(*box)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default(size=max(14, W // 60))
            for f in faces:
                color = "#ff3b30" if f["id"] == face_id else ("#34c759" if f["person_id"] else "#ffcc00")
                draw.rectangle([f["x"] * W, f["y"] * H, (f["x"] + f["w"]) * W, (f["y"] + f["h"]) * H], outline=color, width=3)
                p = roster.get(f["person_id"]) if f["person_id"] else None
                if p and is_public(p.id):
                    draw.text((f["x"] * W, (f["y"] + f["h"]) * H + 2), p.name, fill=color, font=font, stroke_width=2, stroke_fill="black")
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            self._send(buf.getvalue(), "image/jpeg")

        def _api_get(self, name: str, q: dict):
            if name == "state":
                n = lambda sql: db.execute(sql).fetchone()[0]
                return self._json({
                    "photos": n("SELECT COUNT(*) FROM photos WHERE dup_of IS NULL AND error IS NULL"),
                    "duplicates": n("SELECT COUNT(*) FROM photos WHERE dup_of IS NOT NULL"),
                    "faces": n("SELECT COUNT(*) FROM faces"),
                    "named": n("SELECT COUNT(*) FROM faces WHERE person_id IS NOT NULL AND person_id NOT IN ('!ignore', '!other')"),
                    "set_aside": n("SELECT COUNT(*) FROM faces WHERE person_id = '!other'"),
                    "unnamed": n("SELECT COUNT(*) FROM faces WHERE person_id IS NULL"),
                    "active_minutes": active_minutes(db),
                    "people_named": n("SELECT COUNT(DISTINCT person_id) FROM faces WHERE person_id IS NOT NULL AND person_id NOT IN ('!ignore', '!other')")})
            if name == "clusters":
                rows, total = clustering.cluster_summaries(db, roster, int(q.get("offset", 0)), int(q.get("limit", 12)),
                                                           "older" if q.get("order") == "older" else "size")
                for c in rows:
                    for s in c["suggestions"]:
                        s["person"] = person_json(s["person_id"])
                    for h in c["hints"]:
                        h["person"] = person_json(h["person_id"])
                return self._json({"clusters": rows, "total": total})
            if name == "named":
                out = []
                for r in db.execute("SELECT person_id, COUNT(*) n, COUNT(DISTINCT photo_id) photos FROM faces "
                                    "WHERE person_id IS NOT NULL AND person_id != '!ignore' GROUP BY person_id ORDER BY (person_id = '!other'), n DESC"):
                    if blur and not is_public(r["person_id"]) and r["person_id"] != OTHER:
                        continue
                    face_ids = [x[0] for x in db.execute("SELECT id FROM faces WHERE person_id=? ORDER BY score DESC LIMIT 14", (r["person_id"],))]
                    out.append({**person_json(r["person_id"]), "faces": r["n"], "photos": r["photos"], "face_ids": face_ids,
                                "photo_ids": [x[0] for x in db.execute("SELECT photo_id FROM faces WHERE person_id=? ORDER BY score DESC LIMIT 14", (r["person_id"],))]})
                return self._json(out)
            if name == "person-faces":
                rows = db.execute("SELECT id, photo_id FROM faces WHERE person_id=? ORDER BY score DESC", (q.get("person_id", ""),)).fetchall()
                return self._json([{"id": r["id"], "photo_id": r["photo_id"]} for r in rows])
            if name == "people":
                found = [person_json(p.id) for p in roster.search(q.get("q", ""))]
                needle = " ".join(q.get("q", "").lower().split())   # living relatives named only on this machine
                if needle:
                    for r in db.execute("SELECT DISTINCT person_id FROM faces WHERE person_id LIKE 'local:%'"):
                        if needle in r["person_id"][len(LOCAL):].lower():
                            found.append(person_json(r["person_id"]))
                return self._json(found)
            if name == "focus":
                res = clustering.focus(db, roster, q.get("person_id", ""), float(q.get("min", 0.35)),
                                       int(q.get("limit", 150)), int(q.get("offset", 0)))
                res["person"] = person_json(q.get("person_id", ""))
                return self._json(res)
            if name == "export-preview":
                ds = exporting.plan(db, roster)
                withheld = [d for d in ds if not d.export]
                reasons: dict[str, int] = {}
                for d in withheld:
                    reasons[d.kind] = reasons.get(d.kind, 0) + 1
                return self._json({
                    "ready": sum(d.export for d in ds), "withheld": len(withheld), "reasons": reasons,
                    "ready_people": len({p for d in ds if d.export for p in d.people})})
            self._json({"error": "unknown"}, 404)

        def _post(self):
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._json({"error": "json only"}, 415)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            name = self.path.strip("/").split("/")[-1]
            if name == "label":
                pid = body["person_id"]
                if pid.startswith(LOCAL):
                    pid = LOCAL + " ".join(pid[len(LOCAL):].split())[:80]
                    if pid == LOCAL:
                        return self._json({"error": "name required"}, 400)
                elif pid not in (IGNORE, OTHER) and not roster.get(pid):
                    return self._json({"error": "unknown WikiTree id"}, 400)
                src = body.get("source", "user")
                if "face_ids" in body:
                    ids = [int(i) for i in body["face_ids"]]
                    db.executemany("UPDATE faces SET person_id=?, label_src=?, labeled_at=datetime('now') WHERE id=? AND person_id IS NULL", [(pid, src, i) for i in ids])
                else:
                    db.execute("UPDATE faces SET person_id=?, label_src=?, labeled_at=datetime('now') WHERE cluster_id=? AND person_id IS NULL",
                               (pid, src, body["cluster_id"]))
                db.commit()
                return self._json({"ok": True})
            if name == "set-aside-photos":   # everyone still unnamed in these faces' photos is not family
                ids = [int(i) for i in body["face_ids"]]
                marks = ",".join("?" * len(ids))
                photo_ids = [r[0] for r in db.execute(f"SELECT DISTINCT photo_id FROM faces WHERE id IN ({marks})", ids)]
                pm = ",".join("?" * len(photo_ids))
                cur = db.execute(f"UPDATE faces SET person_id=?, label_src='user', labeled_at=datetime('now') WHERE person_id IS NULL AND photo_id IN ({pm})",
                                 [OTHER, *photo_ids])
                db.commit()
                return self._json({"ok": True, "faces": cur.rowcount, "photos": len(photo_ids)})
            if name in ("unlabel", "relabel"):
                src = body["person_id"]
                # whole person, or just the faces ticked in the UI
                where, args = "person_id=?", [src]
                if "face_ids" in body:
                    ids = [int(i) for i in body["face_ids"]]
                    where, args = f"person_id=? AND id IN ({','.join('?' * len(ids))})", [src, *ids]
                if name == "unlabel":   # back to the unnamed pile, still grouped
                    cur = db.execute(f"UPDATE faces SET person_id=NULL, label_src=NULL WHERE {where}", args)
                else:
                    dst = body["to_person_id"]
                    if dst.startswith(LOCAL):
                        dst = LOCAL + " ".join(dst[len(LOCAL):].split())[:80]
                    elif dst not in (IGNORE, OTHER) and not roster.get(dst):
                        return self._json({"error": "unknown WikiTree id"}, 400)
                    cur = db.execute(f"UPDATE faces SET person_id=?, label_src='user', labeled_at=datetime('now') WHERE {where}", [dst, *args])
                db.commit()
                return self._json({"ok": True, "faces": cur.rowcount})
            if name == "eject":
                ids = [int(i) for i in body["face_ids"]]
                new = (db.execute("SELECT COALESCE(MAX(cluster_id),0)+1 FROM faces").fetchone()[0])
                db.executemany("UPDATE faces SET cluster_id=? WHERE id=? AND person_id IS NULL", [(new, i) for i in ids])
                db.commit()
                return self._json({"ok": True, "new_cluster": new})
            self._json({"error": "unknown"}, 404)

    return Handler


def serve(db, roster, data_dir: Path, port: int = 8765, blur: bool = False, models_dir: Path | None = None,
          clear_born_before: int | None = None):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(db, roster, data_dir, blur, models_dir, clear_born_before))
    print(f"Labeling UI: http://127.0.0.1:{port}   (Ctrl-C to stop)" + ("   [privacy blur on]" if blur else ""))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

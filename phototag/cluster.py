"""Group look-alike faces so a person only has to be named once, then suggest who each group is."""
from __future__ import annotations

import sqlite3

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from .cc7 import Roster
from . import dates
from .db import blob_to_emb

DEFAULT_THRESHOLD = 0.36  # mean cosine similarity inside a group; SFace's own same-person cut-off


def _unlabeled(db: sqlite3.Connection):
    return db.execute(
        "SELECT f.id, f.emb FROM faces f JOIN photos p ON p.id=f.photo_id "
        "WHERE f.person_id IS NULL AND p.dup_of IS NULL").fetchall()


def cluster(db: sqlite3.Connection, threshold: float = DEFAULT_THRESHOLD) -> int:
    """Average-linkage clustering of the unnamed faces. Average linkage keeps one blurry
    face from chaining two different people together; over-splitting is harmless because
    naming happens per group and later groups are matched against everyone already named."""
    rows = _unlabeled(db)
    db.execute("UPDATE faces SET cluster_id=NULL WHERE person_id IS NULL")
    if len(rows) < 2:
        db.commit()
        return len(rows)
    ids = [r["id"] for r in rows]
    embs = np.stack([blob_to_emb(r["emb"]) for r in rows]).astype(np.float64)
    labels = fcluster(linkage(embs, method="average", metric="cosine"), t=1 - threshold, criterion="distance")
    sizes = np.bincount(labels)
    rank = {lab: n for n, lab in enumerate(sorted(set(labels), key=lambda l: -sizes[l]), 1)}
    db.executemany("UPDATE faces SET cluster_id=? WHERE id=?", [(rank[l], i) for l, i in zip(labels, ids)])
    db.commit()
    return len(rank)


def exemplars(db: sqlite3.Connection) -> dict[str, np.ndarray]:
    """Embeddings of every face a human has already named, per person."""
    out: dict[str, list] = {}
    for r in db.execute("SELECT person_id, emb FROM faces WHERE person_id IS NOT NULL AND person_id NOT IN ('!ignore', '!other')"):
        out.setdefault(r["person_id"], []).append(blob_to_emb(r["emb"]))
    return {pid: np.stack(v) for pid, v in out.items()}


def suggest(centroid: np.ndarray, ex: dict[str, np.ndarray], roster: Roster, windows: list, k: int = 3):
    """Rank named people by how closely they resemble `centroid`. A person who could not have been alive
    at the time of at least half the group's dated photos sinks."""
    scored = []
    for pid, mat in ex.items():
        top = np.sort(mat @ centroid)[-3:]
        s = float(top.mean())
        dated = [w for w in windows if w]
        ok = not dated or sum(dates.fits(roster, pid, w) for w in dated) >= len(dated) / 2
        scored.append({"person_id": pid, "score": s, "plausible": ok, "rank": s - (0.0 if ok else 1.0)})
    scored.sort(key=lambda d: -d["rank"])
    return scored[:k]


def hints_for(rows, roster: Roster, db: sqlite3.Connection) -> list[dict]:
    """Weak, human-readable clues about who a group is: album folder, names in file names, captions Gemma read."""
    votes: dict[tuple[str, str], set[int]] = {}
    for r in rows:
        if r["folder_person"]:
            votes.setdefault(("folder", r["folder_person"]), set()).add(r["id"])
        stem = r["rel"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        full = {p.id for p in roster.find_in_text(stem)}
        for pid in full:
            votes.setdefault(("file name", pid), set()).add(r["id"])
        for pid in roster.from_filename(r["rel"]) - full:   # first names are weak: 'Sean' may be a living, unlisted Sean
            votes.setdefault(("first name in file name", pid), set()).add(r["id"])
        if r["text_hint"]:
            for p in roster.find_in_text(r["text_hint"]):
                votes.setdefault(("caption", p.id), set()).add(r["id"])
    n = len(rows)
    by_person: dict[str, dict] = {}
    for (src, pid), ids in votes.items():
        if len(ids) >= min(3, n) and len(ids) / n >= 0.3:
            h = by_person.setdefault(pid, {"sources": [], "person_id": pid, "faces": 0})
            h["sources"].append(src)
            h["faces"] = max(h["faces"], len(ids))
    hints = [{**h, "source": " + ".join(h["sources"]), "strong": bool(set(h["sources"]) - {"first name in file name"})}
             for h in by_person.values()]
    hints.sort(key=lambda h: -h["faces"])
    return hints[:4]


def older_first(db: sqlite3.Connection, roster: Roster) -> list[int]:
    """Cluster ids ordered by how likely the group is of people who are no longer alive.

    There is no face-age model here; it combines clues: black-and-white prints, a known early
    year, albums under Ancestors/, and album folders named for someone deceased."""
    rows = db.execute(
        "SELECT f.cluster_id, COUNT(*) n, AVG(f.sat < 0.10) bw, AVG(p.year) year, AVG(p.rel LIKE 'Ancestors/%') anc "
        "FROM faces f JOIN photos p ON p.id=f.photo_id WHERE f.person_id IS NULL AND f.cluster_id IS NOT NULL "
        "AND p.dup_of IS NULL GROUP BY f.cluster_id").fetchall()
    dead_folder: dict[int, float] = {}
    for r in db.execute("SELECT f.cluster_id, p.folder_person, COUNT(*) n FROM faces f JOIN photos p ON p.id=f.photo_id "
                        "WHERE f.person_id IS NULL AND f.cluster_id IS NOT NULL AND p.folder_person IS NOT NULL "
                        "GROUP BY f.cluster_id, p.folder_person"):
        if roster.is_deceased(r["folder_person"]):
            dead_folder[r["cluster_id"]] = dead_folder.get(r["cluster_id"], 0) + r["n"]

    def score(r) -> float:
        y = r["year"]
        s_year = 0.25 if y is None else 1.0 if y <= 1960 else 0.6 if y <= 1980 else 0.3 if y <= 1995 else 0.0
        return (0.35 * s_year + 0.30 * (r["bw"] or 0) + 0.20 * (r["anc"] or 0)
                + 0.15 * min(1.0, dead_folder.get(r["cluster_id"], 0) / r["n"]))
    return [r["cluster_id"] for r in sorted(rows, key=lambda r: (-score(r), -r["n"]))]


def cluster_summaries(db: sqlite3.Connection, roster: Roster, offset: int = 0, limit: int = 20, order: str = "size"):
    ex = exemplars(db)
    windows = dates.photo_windows(db, roster)
    if order == "older":
        ids = older_first(db, roster)[offset:offset + limit]
    else:
        ids = [r[0] for r in db.execute(
            "SELECT cluster_id FROM faces WHERE person_id IS NULL AND cluster_id IS NOT NULL "
            "GROUP BY cluster_id ORDER BY COUNT(*) DESC, cluster_id LIMIT ? OFFSET ?", (limit, offset))]
    total = db.execute("SELECT COUNT(DISTINCT cluster_id) FROM faces WHERE person_id IS NULL AND cluster_id IS NOT NULL").fetchone()[0]
    out = []
    for cid in ids:
        rows = db.execute(
            "SELECT f.id, f.emb, f.score, p.id AS photo_id, p.rel, p.year, p.folder_person, p.text_hint FROM faces f "
            "JOIN photos p ON p.id=f.photo_id WHERE f.cluster_id=? AND f.person_id IS NULL ORDER BY f.score DESC", (cid,)).fetchall()
        cent = np.mean([blob_to_emb(r["emb"]) for r in rows], axis=0)
        cent /= np.linalg.norm(cent) or 1.0
        years = sorted(r["year"] for r in rows if r["year"])
        year = years[len(years) // 2] if years else None
        suggestions = suggest(cent, ex, roster, [windows.get(r["photo_id"]) for r in rows]) if ex else []
        out_hints = hints_for(rows, roster, db)
        out.append({
            "cluster_id": cid, "size": len(rows), "photos": len({r["photo_id"] for r in rows}), "year": year,
            "faces": [{"id": r["id"], "photo_id": r["photo_id"], "rel": r["rel"], "caption": (r["text_hint"] or "")[:160]} for r in rows[:48]],
            "suggestions": suggestions, "hints": out_hints})
    return out, total


def seed_from_filenames(db: sqlite3.Connection, roster: Roster) -> int:
    """'Alice Hester Harrison 001.jpg' with exactly one face in it: that face is Alice. Marked auto, so
    it stays out of the default export until you confirm it."""
    n = 0
    for p in db.execute("SELECT id, rel FROM photos WHERE dup_of IS NULL AND error IS NULL").fetchall():
        people = roster.find_in_text(p["rel"].rsplit("/", 1)[-1].rsplit(".", 1)[0])
        faces = db.execute("SELECT id, person_id FROM faces WHERE photo_id=?", (p["id"],)).fetchall()
        if len(people) == 1 and len(faces) == 1 and faces[0]["person_id"] is None:
            db.execute("UPDATE faces SET person_id=?, label_src='auto', labeled_at=datetime('now') WHERE id=?", (people[0].id, faces[0]["id"]))
            n += 1
    db.commit()
    return n


def focus(db: sqlite3.Connection, roster: Roster, person_id: str, min_score: float = 0.35,
          limit: int = 150, offset: int = 0) -> dict:
    """Every still-unnamed face ranked by how much it resembles the faces already named `person_id`."""
    ex = exemplars(db).get(person_id)
    if ex is None:
        return {"named": 0, "total": 0, "faces": []}
    rows = db.execute(
        "SELECT f.id, f.emb, f.cluster_id, p.id AS photo_id, p.rel, p.year FROM faces f JOIN photos p ON p.id=f.photo_id "
        "WHERE f.person_id IS NULL AND p.dup_of IS NULL").fetchall()
    if not rows:
        return {"named": len(ex), "total": 0, "faces": []}
    windows = dates.photo_windows(db, roster)
    sims = np.stack([blob_to_emb(r["emb"]) for r in rows]) @ ex.T
    k = min(3, sims.shape[1])
    score = np.sort(sims, axis=1)[:, -k:].mean(axis=1)    # same measure as the group suggestions
    order = [i for i in np.argsort(-score) if score[i] >= min_score]
    page = order[offset:offset + limit]
    return {"named": len(ex), "total": len(order), "faces": [
        {"id": rows[i]["id"], "photo_id": rows[i]["photo_id"], "rel": rows[i]["rel"], "year": rows[i]["year"],
         "cluster_id": rows[i]["cluster_id"], "score": float(score[i]),
         "plausible": dates.fits(roster, person_id, windows.get(rows[i]["photo_id"])),
         "window": windows.get(rows[i]["photo_id"])} for i in page]}

"""Narrow who can be in a photo by when it was taken.

Few scans carry a date, so the window for each photo is assembled from three sources, strongest first:
  1. a year in the file name / EXIF                     -> +-1 year
  2. the people already named in the photo              -> after the youngest one was born,
                                                           before the first one died
  3. a rough decade guessed by a local vision model     -> +-30 years (it is often 10-30 years off)
Sources 2 and 3 are intersected when they overlap. Windows only ever *demote* candidates; they never
exclude anyone, because a scanner timestamp or a wrong guess must not hide the right person.
"""
from __future__ import annotations

import re
import sqlite3

from .cc7 import Roster
from .db import IGNORE, OTHER

ERA_SLACK = (30, 40)   # decade-30 .. decade+9+30: a generous window around a coarse guess
FAR_FUTURE = 2100


def photo_windows(db: sqlite3.Connection, roster: Roster) -> dict[int, tuple[int, int, str]]:
    """photo id -> (earliest year, latest year, source). Photos nothing is known about are absent."""
    named: dict[int, list[str]] = {}
    for r in db.execute("SELECT photo_id, person_id FROM faces WHERE person_id IS NOT NULL AND person_id NOT IN (?, ?)", (IGNORE, OTHER)):
        named.setdefault(r["photo_id"], []).append(r["person_id"])
    out: dict[int, tuple[int, int, str]] = {}
    for p in db.execute("SELECT id, year, era FROM photos WHERE dup_of IS NULL").fetchall():
        if p["year"]:
            out[p["id"]] = (p["year"] - 1, p["year"] + 1, "dated")
            continue
        lo, hi, src = None, None, None
        people = [roster.get(pid) for pid in named.get(p["id"], [])]
        births = [x.birth_year for x in people if x and x.birth_year]
        deaths = [x.death_year for x in people if x and x.death_year]
        if births or deaths:
            lo = max(births) if births else 0
            hi = min(deaths) if deaths else FAR_FUTURE
            src = "people"
        if p["era"]:
            elo, ehi = p["era"] - ERA_SLACK[0], p["era"] + ERA_SLACK[1]
            if lo is None:
                lo, hi, src = elo, ehi, "era"
            elif max(lo, elo) <= min(hi, ehi):
                lo, hi, src = max(lo, elo), min(hi, ehi), "people+era"
        if lo is not None and (lo > 0 or hi < FAR_FUTURE):
            out[p["id"]] = (lo, hi, src)
    return out


def fits(roster: Roster, person_id: str, window: tuple[int, int, str] | None) -> bool:
    """Could `person_id` have been alive at some point inside the photo's window?"""
    p = roster.get(person_id)
    if p is None or window is None:
        return True
    lo, hi, _ = window
    if p.birth_year and p.birth_year > hi + 1:
        return False
    if p.death_year and p.death_year < lo - 1:
        return False
    return True


# --- printed date stamps ("MAY 64", "JUN '72", "89 8 13") that labs put on prints ---------------------------------
_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
_MONTH_YEAR = re.compile(rf"(?<![a-z])(?:{_MONTHS})[a-z]*\.?\s*['’]?\s*(\d{{4}}|\d{{2}})(?!\d)", re.I)
_YY_M_D = re.compile(r"^\W*['’]?(\d{2})\s+(\d{1,2})\s+(\d{1,2})\W*$")


def _year(yy: str) -> int | None:
    n = int(yy)
    if len(yy) == 4:
        return n if 1860 <= n <= 2026 else None
    n = 1900 + n if n >= 40 else 2000 + n       # print stamps start in the 1940s
    return n if 1940 <= n <= 2026 else None


def year_from_stamp(text: str) -> int | None:
    """A photo year from a short printed date stamp. Long texts (captions, documents) are ignored: a year in
    prose is as likely to be a birth or a marriage as the date the picture was taken."""
    text = " ".join((text or "").split())
    if not text or len(text) > 24:
        return None
    m = _YY_M_D.match(text)
    if m and 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31:
        return _year(m.group(1))
    years = {_year(y) for y in _MONTH_YEAR.findall(text)} - {None}
    return years.pop() if len(years) == 1 else None


def apply_stamps(db: sqlite3.Connection) -> int:
    """Give undated photos the year from a stamp Gemma read. Safe to run repeatedly."""
    n = 0
    for r in db.execute("SELECT id, text_hint FROM photos WHERE year IS NULL AND text_hint != '' AND text_hint IS NOT NULL").fetchall():
        y = year_from_stamp(r["text_hint"])
        if y:
            db.execute("UPDATE photos SET year=?, year_src='stamp' WHERE id=?", (y, r["id"]))
            n += 1
    db.commit()
    return n

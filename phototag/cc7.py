"""WikiTree CC7 extract: who is in the family tree, and who may be shown publicly.

The CC7 CSV has no "living" flag, so we infer it conservatively: someone is
treated as deceased only if they have a death date, or were born so long ago
that they cannot plausibly be alive. Anyone else is treated as possibly living
and their photos never reach the export folder.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PRESUMED_DECEASED_BEFORE = 1910
_SUFFIXES = {"sr", "jr", "ii", "iii", "iv"}


def norm(s: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def year_of(date: str | None) -> int | None:
    m = re.match(r"\s*(\d{4})", date or "")
    year = int(m.group(1)) if m else 0
    return year or None


@dataclass(frozen=True)
class Person:
    id: str
    given: str
    birth_last: str
    current_last: str
    birth_date: str
    death_date: str
    birth_place: str
    degree: int

    @property
    def birth_year(self) -> int | None:
        return year_of(self.birth_date)

    @property
    def death_year(self) -> int | None:
        return year_of(self.death_date)

    @property
    def name(self) -> str:
        return f"{self.given} {self.birth_last}".strip()

    @property
    def lifespan(self) -> str:
        b = self.birth_year or "?"
        return f"{b}–{self.death_year or ''}".rstrip("–") if self.death_date else f"{b}–"

    @property
    def label(self) -> str:
        return f"{self.name} ({self.lifespan}) {self.id}"

    def name_keys(self) -> set[str]:
        first = (self.given.split() or [""])[0]
        lasts = {self.birth_last, self.current_last} - {""}
        keys = {norm(f"{g} {l}") for g in (self.given, first) for l in lasts}
        return {k for k in keys if " " in k}


class Roster:
    def __init__(self, people: list[Person], presumed_deceased_before: int = DEFAULT_PRESUMED_DECEASED_BEFORE):
        self.people = {p.id: p for p in people}
        self.cutoff = presumed_deceased_before
        self._by_key: dict[str, list[Person]] = {}
        for p in people:
            for k in p.name_keys():
                self._by_key.setdefault(k, []).append(p)

    @classmethod
    def from_csv(cls, path: str | Path, presumed_deceased_before: int = DEFAULT_PRESUMED_DECEASED_BEFORE) -> "Roster":
        people = []
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if not r.get("WikiTree ID"):
                    continue
                people.append(Person(
                    id=r["WikiTree ID"], given=r["Given Name(s)"], birth_last=r["Last name at birth"],
                    current_last=r["Current last name"], birth_date=r["Birth date"], death_date=r["Death date"],
                    birth_place=r["Birth place"], degree=int(r["Degree"] or 0)))
        return cls(people, presumed_deceased_before)

    def get(self, pid: str) -> Person | None:
        return self.people.get(pid)

    def is_deceased(self, pid: str) -> bool:
        """True only when we are confident. Unknown ids count as possibly living."""
        p = self.people.get(pid)
        if p is None:
            return False
        if p.death_date.strip():
            return True
        return p.birth_year is not None and p.birth_year <= self.cutoff

    def plausible(self, pid: str, year: int | None) -> bool:
        """Could this person appear in a photo taken in `year`? Unknown dates are plausible."""
        p = self.people.get(pid)
        if p is None or year is None:
            return True
        if p.birth_year and year < p.birth_year - 1:
            return False
        if p.death_year and year > p.death_year + 1:
            return False
        return True

    def search(self, query: str, limit: int = 12) -> list[Person]:
        tokens = norm(query).split()
        if not tokens:
            return []
        hits = []
        for p in self.people.values():
            hay = norm(f"{p.given} {p.birth_last} {p.current_last} {p.id}")
            if all(t in hay for t in tokens):
                hits.append(p)
        hits.sort(key=lambda p: (p.degree, p.name))
        return hits[:limit]

    def match_name(self, text: str) -> list[Person]:
        return list(self._by_key.get(norm(text), []))

    def find_in_text(self, text: str) -> list[Person]:
        """People whose full name appears in free text (e.g. handwriting on the back of a photo)."""
        hay = f" {norm(text)} "
        found: dict[str, Person] = {}
        for key, ps in self._by_key.items():
            if len(ps) == 1 and f" {key} " in hay:
                found[ps[0].id] = ps[0]
        return list(found.values())

    def from_folder_name(self, folder: str) -> Person | None:
        """'0A2 Carl William Robinson Sr' -> the matching person, using Sr/Jr to break ties."""
        name = re.sub(r"^\s*(?=\S*\d)\S{1,5}\s+", "", folder)  # drop leading album code like '0A2' / '7'
        tokens = norm(name).split()
        suffix = tokens[-1] if tokens and tokens[-1] in _SUFFIXES else None
        if suffix:
            tokens = tokens[:-1]
        matches = self.match_name(" ".join(tokens))
        if len(matches) == 1:
            return matches[0] if suffix in (None, "sr") else None  # a lone match can't be the Jr/III
        dated = sorted((p for p in matches if p.birth_year), key=lambda p: p.birth_year)
        if suffix == "sr" and dated:
            return dated[0]
        if suffix in ("jr", "ii") and len(dated) > 1:
            return dated[-1]
        return None

    def from_filename(self, rel: str, max_degree: int = 3) -> set[str]:
        """Close relatives whose first name appears in a file name like 'CarlSr_Fran_Teresa 003.jpg'.

        Only near relatives are considered, and an ambiguous first name (two equally close Carls)
        is skipped rather than guessed. 'Sr'/'Jr' after a name picks the older/younger one."""
        stem = re.sub(r"[\s_-]*\d+(\s*copy\s*\d*)?$", "", Path(rel).stem, flags=re.I)
        words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", re.sub(r"[_\-]+", " ", stem)).split()
        found: set[str] = set()
        for i, w in enumerate(words):
            t = norm(w)
            if len(t) < 3 or t in _SUFFIXES:
                continue
            suffix = norm(words[i + 1]) if i + 1 < len(words) and norm(words[i + 1]) in _SUFFIXES else None
            cands = [p for p in self.people.values() if p.degree <= max_degree and (norm(p.given).split() or [""])[0].startswith(t)]
            if not cands:
                continue
            best = min(p.degree for p in cands)
            cands = [p for p in cands if p.degree == best]
            if suffix and len(cands) > 1:
                dated = sorted((p for p in cands if p.birth_year), key=lambda p: p.birth_year)
                cands = dated[:1] if suffix == "sr" else dated[-1:]
            if len(cands) == 1:
                found.add(cands[0].id)
        return found

import csv
import pytest

from phototag.cc7 import Roster

HEADER = ["WikiTree ID", "Degree", "Parents", "Siblings", "Spouses", "Children", "Given Name(s)", "Last name at birth",
          "Current last name", "Birth date", "Birth place", "Death date", "Death place", "Age", "Manager", "Created", "Modified"]
PEOPLE = [
    ("Robinson-1", 2, "Carl William", "Robinson", "Robinson", "1901-08-02", "1991-10-01"),   # deceased (death date)
    ("Robinson-2", 1, "Carl William", "Robinson", "Robinson", "1933-11-13", "2003-04-25"),   # deceased, Jr-ish
    ("Budd-1", 2, "Leona Catherine", "Budd", "Robinson", "1904-02-17", "1966-07-03"),
    ("Old-1", 5, "Johann Georg", "Greb", "Greb", "1800", ""),                              # no death date, born long ago
    ("Living-1", 0, "Azure Rae", "Robinson", "Robinson", "1975-07-17", ""),                # possibly living
    ("Peasley-5", 1, "Frances Lee", "Peasley", "Robinson", "1937-07-15", ""),              # no death date, born 1937
]


@pytest.fixture
def roster(tmp_path):
    p = tmp_path / "cc7.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerow([""] * len(HEADER))
        for pid, deg, given, last, cur, b, d in PEOPLE:
            w.writerow([pid, deg, 2, "", "", "", given, last, cur, b, "", d, "", "", "", "", ""])
    return Roster.from_csv(p)

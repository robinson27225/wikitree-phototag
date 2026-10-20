import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from phototag import dates, db as dbm, export
from phototag.xmp import build_xmp


def photo(db, tmp_path, name, year=None, era=None, faces=()):
    path = tmp_path / f"{name}.jpg"
    Image.new("RGB", (100, 50)).save(path)
    pid = db.execute("INSERT INTO photos(path,rel,width,height,year,era) VALUES(?,?,100,50,?,?)", (str(path), path.name, year, era)).lastrowid
    for person, src in faces:
        db.execute("INSERT INTO faces(photo_id,x,y,w,h,score,emb,person_id,label_src) VALUES(?,.1,.1,.2,.2,.9,?,?,?)",
                   (pid, dbm.emb_to_blob(np.zeros(128)), person, src))
    db.commit()
    return pid


def test_window_comes_from_year_then_people_then_era(roster, tmp_path):
    db = dbm.connect(tmp_path)
    dated = photo(db, tmp_path, "dated", year=1950)
    with_people = photo(db, tmp_path, "people", faces=[("Robinson-1", "user"), ("Budd-1", "user")])   # b.1901/1904, d.1991/1966
    era_only = photo(db, tmp_path, "era", era=1940)
    both = photo(db, tmp_path, "both", era=1850, faces=[("Robinson-2", "user")])                       # b.1933 vs a wildly early guess
    nothing = photo(db, tmp_path, "nothing")
    w = dates.photo_windows(db, roster)
    assert w[dated] == (1949, 1951, "dated")
    assert w[with_people] == (1904, 1966, "people")            # after the younger was born, before the first died
    assert w[era_only] == (1910, 1980, "era")
    assert w[both][2] == "people"                              # a wrong era guess never overrides the tree
    assert nothing not in w


def test_fits_demotes_but_unknown_people_and_dates_always_fit(roster):
    assert dates.fits(roster, "Robinson-1", (1904, 1966, "people"))
    assert not dates.fits(roster, "Living-1", (1904, 1966, "people"))        # b.1975, photo window ends 1966
    assert not dates.fits(roster, "Budd-1", (1990, 2000, "dated"))           # d.1966
    assert dates.fits(roster, "Nope-9", (1904, 1966, "people")) and dates.fits(roster, "Budd-1", None)


def test_xmp_is_well_formed_and_carries_names_regions_and_caption():
    xml = build_xmp(200, 100, [{"name": "Carl & Sons <Jr>", "x": .1, "y": .2, "w": .2, "h": .3}], ["Robinson-1"], "Carl, 1955")
    root = ET.fromstring(xml.split("?>", 1)[1].rsplit("<?xpacket", 1)[0])
    text = ET.tostring(root, encoding="unicode")
    assert "Carl &amp; Sons &lt;Jr&gt;" in text and "Robinson-1" in text and "PersonInImage" in text and "Carl, 1955" in text


def test_sidecars_beside_originals_skip_existing_set_aside_and_auto(roster, tmp_path):
    db = dbm.connect(tmp_path)
    a = photo(db, tmp_path, "family", faces=[("Robinson-1", "user"), ("local:Dan Robinson", "user"), (dbm.OTHER, "user")])
    photo(db, tmp_path, "guessed", faces=[("Budd-1", "auto")])
    assert export.write_sidecars(db, roster)["written"] == 0                      # dry run
    assert not (tmp_path / "family.jpg.xmp").exists()
    r = export.write_sidecars(db, roster, write=True)
    assert r == {"photos": 1, "written": 1, "skipped_existing": 0}
    side = (tmp_path / "family.jpg.xmp").read_text()
    assert "Dan Robinson" in side and "Carl William Robinson" in side and "Robinson-1" in side
    assert side.count("mwg-rs:Name=") == 2                                         # set-aside face is not tagged
    (tmp_path / "family.jpg.xmp").write_text("mine")
    assert export.write_sidecars(db, roster, write=True)["skipped_existing"] == 1  # never clobbers
    assert (tmp_path / "family.jpg.xmp").read_text() == "mine"
    assert export.write_sidecars(db, roster, write=True, force=True)["written"] == 1


def test_date_stamps_become_years_but_prose_does_not():
    assert dates.year_from_stamp("MAY 64") == 1964
    assert dates.year_from_stamp("JUN '72") == 1972
    assert dates.year_from_stamp("Dec. 1985") == 1985
    assert dates.year_from_stamp("89 8 13") == 1989
    assert dates.year_from_stamp("'05 11 02") == 2005
    assert dates.year_from_stamp("Jeremiah Alan Sweeny Dec. 21, 1997 7 Lbs. 13 Oz.") is None   # prose: a birth, not a print date
    assert dates.year_from_stamp("for Grandpa Carl William Robinson b. 11-13-1933") is None
    assert dates.year_from_stamp("MAY 64 JUN 71") is None                                      # two different years: ambiguous
    assert dates.year_from_stamp("") is None and dates.year_from_stamp("Costa Atlantica") is None


def test_apply_stamps_only_fills_undated_photos(tmp_path):
    db = dbm.connect(tmp_path)
    a = db.execute("INSERT INTO photos(path,rel,text_hint) VALUES('a','a.jpg','MAY 64')").lastrowid
    b = db.execute("INSERT INTO photos(path,rel,text_hint,year,year_src) VALUES('b','b.jpg','MAY 64',1950,'filename')").lastrowid
    db.commit()
    assert dates.apply_stamps(db) == 1
    assert dates.apply_stamps(db) == 0
    assert db.execute("SELECT year, year_src FROM photos WHERE id=?", (a,)).fetchone()[:] == (1964, "stamp")
    assert db.execute("SELECT year FROM photos WHERE id=?", (b,)).fetchone()[0] == 1950           # a real date is never overwritten

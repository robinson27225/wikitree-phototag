import numpy as np
from PIL import Image

from phototag import db as dbm, export


def add_photo(db, tmp_path, name, faces):
    path = tmp_path / f"{name}.jpg"
    Image.new("RGB", (200, 100), "gray").save(path)
    pid = db.execute("INSERT INTO photos(path,rel,width,height,year) VALUES(?,?,200,100,1955)", (str(path), path.name)).lastrowid
    for person, src in faces:
        db.execute("INSERT INTO faces(photo_id,x,y,w,h,score,emb,person_id,label_src) VALUES(?,?,?,?,?,?,?,?,?)",
                   (pid, .1, .1, .2, .2, .9, dbm.emb_to_blob(np.zeros(128)), person, src))
    db.commit()


def run(db, roster, tmp_path, **kw):
    return export.export(db, roster, tmp_path / "out", tmp_path, **kw), tmp_path / "out"


def test_only_all_deceased_and_fully_named_photos_leave(roster, tmp_path):
    db = dbm.connect(tmp_path / "d")
    add_photo(db, tmp_path, "ok", [("Robinson-1", "user"), ("Budd-1", "user")])
    add_photo(db, tmp_path, "living_in_it", [("Robinson-1", "user"), ("Living-1", "user")])
    add_photo(db, tmp_path, "unnamed_face", [("Robinson-1", "user"), (None, None)])
    add_photo(db, tmp_path, "unknown_id", [("Mystery-9", "user")])
    add_photo(db, tmp_path, "born_1937_no_death", [("Peasley-5", "user")])
    summary, out = run(db, roster, tmp_path)
    assert summary["exported_photos"] == 1 and summary["withheld"] == 4
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["Budd-1", "Robinson-1"]   # copied to each profile
    assert not (out / "Living-1").exists() and not (out / "Peasley-5").exists()
    withheld = (tmp_path / "withheld.csv").read_text()
    assert "possibly living: Living-1" in withheld and "not yet named" in withheld


def test_captions_link_the_other_people(roster, tmp_path):
    db = dbm.connect(tmp_path / "d")
    add_photo(db, tmp_path, "pair", [("Robinson-1", "user"), ("Budd-1", "user")])
    run(db, roster, tmp_path)
    csv_text = (tmp_path / "out" / "captions.csv").read_text()
    assert "[[Robinson-1|Carl William Robinson]] with [[Budd-1|Leona Catherine Budd]], 1955." in csv_text
    assert (tmp_path / "out" / "Budd-1" / "Budd-1_pair.xmp").exists()


def test_ignored_detections_do_not_block_and_auto_labels_need_opting_in(roster, tmp_path):
    db = dbm.connect(tmp_path / "d")
    add_photo(db, tmp_path, "statue", [("Robinson-1", "user"), (dbm.IGNORE, "user")])
    add_photo(db, tmp_path, "auto", [("Budd-1", "auto")])
    summary, _ = run(db, roster, tmp_path)
    assert summary["exported_photos"] == 1                      # auto-accepted guesses stay private by default
    summary, _ = run(db, roster, tmp_path, include_auto=True)
    assert summary["exported_photos"] == 2


def test_locally_named_people_are_never_exported(roster, tmp_path):
    db = dbm.connect(tmp_path / "d")
    add_photo(db, tmp_path, "with_cousin", [("Robinson-1", "user"), ("local:Dan Robinson", "user")])
    summary, _ = run(db, roster, tmp_path)
    assert summary["exported_photos"] == 0


def test_set_aside_faces_keep_the_photo_private(roster, tmp_path):
    db = dbm.connect(tmp_path / "d")
    add_photo(db, tmp_path, "school", [("Robinson-1", "user"), (dbm.OTHER, "user")])
    summary, _ = run(db, roster, tmp_path)
    assert summary["exported_photos"] == 0
    assert "set aside as not family" in (tmp_path / "withheld.csv").read_text()

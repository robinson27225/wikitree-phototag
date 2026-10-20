import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from phototag import db as dbm, web


@pytest.fixture
def server(roster, tmp_path):
    db = dbm.connect(tmp_path)
    pid = db.execute("INSERT INTO photos(path,rel,width,height) VALUES('x','x.jpg',10,10)").lastrowid
    for _ in range(4):
        db.execute("INSERT INTO faces(photo_id,x,y,w,h,score,emb,person_id,label_src) VALUES(?,0,0,.1,.1,.9,?,'Robinson-1','user')",
                   (pid, dbm.emb_to_blob(np.ones(128))))
    db.commit()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.make_handler(db, roster, tmp_path))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield db, f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def test_move_some_faces_to_another_person_and_back(server):
    db, base = server
    ids = [r[0] for r in db.execute("SELECT id FROM faces ORDER BY id LIMIT 2")]
    assert post(base, "/api/relabel", {"person_id": "Robinson-1", "face_ids": ids, "to_person_id": "local:Carl Jr"})["faces"] == 2
    assert db.execute("SELECT COUNT(*) FROM faces WHERE person_id='Robinson-1'").fetchone()[0] == 2
    assert db.execute("SELECT COUNT(*) FROM faces WHERE person_id='local:Carl Jr'").fetchone()[0] == 2
    assert post(base, "/api/unlabel", {"person_id": "local:Carl Jr", "face_ids": ids[:1]})["faces"] == 1
    assert db.execute("SELECT COUNT(*) FROM faces WHERE person_id IS NULL").fetchone()[0] == 1
    faces = json.load(urllib.request.urlopen(base + "/api/person-faces?person_id=Robinson-1"))
    assert len(faces) == 2


def test_only_the_ticked_faces_move_and_unknown_ids_are_refused(server):
    db, base = server
    one = db.execute("SELECT id FROM faces LIMIT 1").fetchone()[0]
    req = urllib.request.Request(base + "/api/relabel", json.dumps({"person_id": "Robinson-1", "face_ids": [one], "to_person_id": "Nope-1"}).encode(),
                                 {"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(req)
    assert db.execute("SELECT COUNT(*) FROM faces WHERE person_id='Robinson-1'").fetchone()[0] == 4


def test_people_search_includes_locally_named_living_relatives(server):
    db, base = server
    db.execute("UPDATE faces SET person_id='local:Naomi Ruth Robinson' WHERE id=1")
    db.commit()
    found = json.load(urllib.request.urlopen(base + "/api/people?q=naomi%20ruth"))
    assert [p["id"] for p in found] == ["local:Naomi Ruth Robinson"]


def test_blur_mode_hides_living_names_and_cards(roster, tmp_path):
    db = dbm.connect(tmp_path)
    pid = db.execute("INSERT INTO photos(path,rel,width,height) VALUES('x','x.jpg',10,10)").lastrowid
    for who in ("Robinson-1", "Living-1", "local:Dan Robinson"):
        db.execute("INSERT INTO faces(photo_id,x,y,w,h,score,emb,person_id,label_src) VALUES(?,0,0,.1,.1,.9,?,?,'user')",
                   (pid, dbm.emb_to_blob(np.ones(128)), who))
    db.commit()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.make_handler(db, roster, tmp_path, blur=True))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_port}"
        named = json.load(urllib.request.urlopen(base + "/api/named"))
        assert [p["id"] for p in named] == ["Robinson-1"]                      # living and local-only cards are not listed
        found = json.load(urllib.request.urlopen(base + "/api/people?q=azure%20rae"))
        assert found and all("Azure" not in p["label"] for p in found)         # search results mask living names
    finally:
        srv.shutdown()

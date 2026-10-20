import numpy as np

from phototag import cluster, db as dbm


def test_two_people_become_two_groups_and_naming_one_suggests_the_other(roster, tmp_path):
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(2, 128))
    db = dbm.connect(tmp_path)
    pid = db.execute("INSERT INTO photos(path,rel,width,height) VALUES('x','x.jpg',10,10)").lastrowid
    for base, n in ((a, 12), (b, 8)):
        for _ in range(n):
            v = base + rng.normal(scale=0.15, size=128)
            db.execute("INSERT INTO faces(photo_id,x,y,w,h,score,emb) VALUES(?,0,0,.1,.1,.9,?)", (pid, dbm.emb_to_blob(v / np.linalg.norm(v))))
    assert cluster.cluster(db) == 2
    groups = [r[0] for r in db.execute("SELECT COUNT(*) FROM faces GROUP BY cluster_id ORDER BY 1 DESC")]
    assert groups == [12, 8]
    db.execute("UPDATE faces SET person_id='Robinson-1', label_src='user' WHERE cluster_id=1")
    db.commit()
    assert cluster.cluster(db) == 1              # only unnamed faces are re-grouped
    ex = cluster.exemplars(db)
    assert set(ex) == {"Robinson-1"}


def test_set_aside_faces_are_not_regrouped_or_used_as_examples(roster, tmp_path):
    db = dbm.connect(tmp_path)
    pid = db.execute("INSERT INTO photos(path,rel,width,height) VALUES('x','x.jpg',10,10)").lastrowid
    v = np.ones(128) / np.sqrt(128)
    for who in (None, None, dbm.OTHER):
        db.execute("INSERT INTO faces(photo_id,x,y,w,h,score,emb,person_id) VALUES(?,0,0,.1,.1,.9,?,?)", (pid, dbm.emb_to_blob(v), who))
    db.commit()
    assert cluster.cluster(db) == 1
    assert cluster.exemplars(db) == {}


def test_older_first_puts_early_black_and_white_groups_ahead(roster, tmp_path):
    db = dbm.connect(tmp_path)
    v = dbm.emb_to_blob(np.ones(128) / np.sqrt(128))
    def add(rel, year, sat, cluster):
        pid = db.execute("INSERT INTO photos(path,rel,width,height,year) VALUES(?,?,10,10,?)", (rel, rel, year)).lastrowid
        db.execute("INSERT INTO faces(photo_id,x,y,w,h,score,emb,sat,cluster_id) VALUES(?,0,0,.1,.1,.9,?,?,?)", (pid, v, sat, cluster))
    for i in range(5):
        add(f"2005/new{i}.jpg", 2005, 0.4, 1)         # big, recent, colourful group
    add("Ancestors/1920.jpg", 1920, 0.02, 2)          # small, old, black-and-white
    db.commit()
    assert cluster.older_first(db, roster) == [2, 1]
    summaries, _ = cluster.cluster_summaries(db, roster, order="older")
    assert [c["cluster_id"] for c in summaries] == [2, 1]


def test_focus_ranks_unnamed_faces_by_resemblance_to_a_named_person(roster, tmp_path):
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(2, 128))
    db = dbm.connect(tmp_path)
    def add(photo, who, base, year=None, person=None):
        pid = db.execute("INSERT INTO photos(path,rel,width,height,year) VALUES(?,?,10,10,?)", (photo, photo, year)).lastrowid
        v = base + rng.normal(scale=0.2, size=128)
        db.execute("INSERT INTO faces(photo_id,x,y,w,h,score,emb,person_id) VALUES(?,0,0,.1,.1,.9,?,?)",
                   (pid, dbm.emb_to_blob(v / np.linalg.norm(v)), person))
    for i in range(4):
        add(f"named{i}.jpg", None, a, person="Robinson-1")      # Robinson-1 (1901-1991): the person we focus on
    for i in range(3):
        add(f"likely{i}.jpg", None, a)
    add("dated_after_death.jpg", None, a, year=2010)
    for i in range(3):
        add(f"other{i}.jpg", None, b)
    db.commit()
    res = cluster.focus(db, roster, "Robinson-1", min_score=0.5)
    rels = [f["rel"] for f in res["faces"]]
    assert res["named"] == 4 and set(rels) == {"likely0.jpg", "likely1.jpg", "likely2.jpg", "dated_after_death.jpg"}
    assert [f["plausible"] for f in res["faces"] if f["rel"] == "dated_after_death.jpg"] == [False]
    assert cluster.focus(db, roster, "Budd-1")["named"] == 0       # nobody named yet: nothing to compare to

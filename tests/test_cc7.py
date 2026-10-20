def test_deceased_rules(roster):
    assert roster.is_deceased("Robinson-1")        # has a death date
    assert roster.is_deceased("Old-1")             # born 1800, no death date: presumed deceased
    assert not roster.is_deceased("Living-1")      # born 1975, no death date
    assert not roster.is_deceased("Peasley-5")     # born 1937, no death date: withheld
    assert not roster.is_deceased("Nobody-99")     # unknown ids are never exported


def test_search_and_names(roster):
    assert [p.id for p in roster.search("leona budd")] == ["Budd-1"]
    assert [p.id for p in roster.find_in_text("Aunt Leona Catherine Budd, 1952")] == ["Budd-1"]
    assert roster.find_in_text("some random words") == []


def test_folder_names_use_sr_jr_to_break_ties(roster):
    assert roster.from_folder_name("0A2 Carl William Robinson Sr").id == "Robinson-1"
    assert roster.from_folder_name("7 Carl William Robinson Jr").id == "Robinson-2"
    assert roster.from_folder_name("9 Azure Rae Robinson").id == "Living-1"
    assert roster.from_folder_name("GoodTimes") is None


def test_date_plausibility_is_soft_about_unknowns(roster):
    assert roster.plausible("Robinson-1", 1950)
    assert not roster.plausible("Robinson-1", 1890)     # before birth
    assert not roster.plausible("Robinson-1", 2010)     # long after death
    assert roster.plausible("Living-1", 2020) and roster.plausible("Living-1", None)


def test_lone_name_match_is_not_assumed_to_be_the_jr(roster):
    from phototag.cc7 import Roster
    only = Roster([roster.get("Robinson-2")])
    assert only.from_folder_name("0A2 Carl William Robinson Sr").id == "Robinson-2"
    assert only.from_folder_name("7 Carl William Robinson Jr") is None


def test_filename_hints(roster):
    assert roster.from_filename("Fran_Azure 068.jpg") == {"Peasley-5", "Living-1"}   # nickname = prefix of the first name
    assert roster.from_filename("FrancesLee_Budd 003 copy 2.jpg") >= {"Peasley-5"}
    assert roster.from_filename("IMG_0042.jpg") == set()
    assert roster.from_filename("Johann 001.jpg") == set()                        # degree 5: too distant to guess
    # the closest relative with that first name wins; equally close ties are skipped unless Sr/Jr decides
    assert roster.from_filename("CarlSr 002.jpg") == {"Robinson-2"}

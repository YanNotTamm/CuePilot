"""Tests for Mashup / Acapella Finder."""

from core.compatibility.mashup import _has_instrumental, _has_vocals, find_mashups


def _track(tid, key="Am", bpm=128.0, vocals=True, instrumental=True, acapella_only=False):
    return {
        "id": tid,
        "key": key,
        "bpm": bpm,
        "hasVocals": vocals,
        "instrumentalOnly": not instrumental,
        "acapellaOnly": acapella_only,
    }


def test_has_vocals_and_instrumental():
    assert _has_vocals(_track("a"))
    assert not _has_vocals(_track("a", vocals=False))
    assert _has_instrumental(_track("a"))
    assert not _has_instrumental(_track("a", instrumental=False))


def test_find_mashups_finds_compatible_pair():
    lib = [
        _track("A", key="Am", bpm=128),   # acapella source
        _track("B", key="C", bpm=128),    # relative key, exact bpm → instrumental
    ]
    results = find_mashups(lib, top_n=5, min_score=0.4)
    assert len(results) >= 1
    best = results[0]
    assert best.acapella_track_id == "A"
    assert best.instrumental_track_id == "B"
    assert best.score > 0.5
    assert "acapella over instrumental" in best.reasons


def test_find_mashups_excludes_incompatible():
    lib = [
        _track("A", key="Am", bpm=128),
        _track("B", key="F#m", bpm=175),
    ]
    results = find_mashups(lib, top_n=5, min_score=0.6)
    assert results == []


def test_find_mashups_respects_instrumental_only():
    lib = [
        _track("A", key="Am", bpm=128),
        _track("B", key="C", bpm=128, instrumental=False),  # acapella only
    ]
    results = find_mashups(lib, top_n=5, min_score=0.4)
    # B has no instrumental → no candidate with B as instrumental base.
    assert not any(r.instrumental_track_id == "B" for r in results)


def test_find_mashups_ignores_self_pairing():
    lib = [_track("A", key="Am", bpm=128)]
    assert find_mashups(lib) == []


def test_find_mashups_to_dict():
    lib = [_track("A", key="Am", bpm=128), _track("B", key="C", bpm=128)]
    results = find_mashups(lib, min_score=0.4)
    assert results
    data = results[0].to_dict()
    assert data["acapellaTrackId"] == "A"
    assert "score" in data and "reasons" in data
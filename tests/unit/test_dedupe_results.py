from src.utils.dedupe_results import dedupe_candidates

def test_dedupe_by_citation_keeps_best_confidence():
    cands = [
        {"element":"A","citation":"skos:abc","confidence":0.9},
        {"element":"A (dup)","citation":"skos:abc","confidence":0.8},
        {"element":"B","citation":"archi:id-1","confidence":0.7},
        {"element":"C","confidence":0.6},  # no citation → keep
    ]
    out = dedupe_candidates(cands)
    # keeps one per citation + the no-citation
    assert len(out) == 3
    seen = {c.get("citation") for c in out if c.get("citation")}
    assert seen == {"skos:abc","archi:id-1"}
    # best of duplicates retained
    top_a = [c for c in out if c.get("citation") == "skos:abc"][0]
    assert top_a["confidence"] == 0.9

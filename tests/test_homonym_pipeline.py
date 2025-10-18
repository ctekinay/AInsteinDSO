# tests/test_homonym_pipeline.py
from src.knowledge.homonym_guard import set_homonym_lexicon, apply_homonym_guard

def test_guard_noop_when_empty():
    set_homonym_lexicon({})
    ranked = apply_homonym_guard(
        query="understand current implementation of telemetry",
        primary_category="architectural_guidance",
        candidates=[
            {"label":"Current (electrical)","source":"kg","confidence":0.9},
            {"label":"Telemetry intake pipeline","source":"archimate","confidence":0.7},
        ],
    )
    assert ranked[0]["label"] == "Current (electrical)"

def test_guard_reweights_on_current_term():
    set_homonym_lexicon({
        "current": {
            "architectural_guidance": {
                "kg_weight_adjustment": -0.6,
                "semantic_weight_adjustment": +0.3,
                "archimate_weight_adjustment": +0.3,
            }
        }
    })
    ranked = apply_homonym_guard(
        query="understand the current implementation of our telemetry data intake",
        primary_category="architectural_guidance",
        candidates=[
            {"label":"Current (electrical)","source":"kg","confidence":0.90},
            {"label":"Telemetry data intake (semantic)","source":"semantic","confidence":0.76},
            {"label":"Telemetry intake pipeline (ArchiMate)","source":"archimate","confidence":0.70},
        ],
    )
    top_sources = [c["source"] for c in ranked[:2]]
    assert "kg" not in top_sources

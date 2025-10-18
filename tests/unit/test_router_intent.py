from src.routing.query_router import QueryRouter
from src.knowledge.homonym_guard import set_homonym_lexicon, apply_homonym_guard

def test_router_basic_intents(tmp_path):
    router = QueryRouter("config/vocabularies.json", kg_loader=None)
    assert router.route("what is reactive power?") in {"structured_model","togaf_method","unstructured_docs"}
    assert router.route("compare active vs reactive power") in {"structured_model","unstructured_docs"}

def test_guard_reweights_current():
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
        query="understand the current ingestion pipeline",
        primary_category="architectural_guidance",
        candidates=[
            {"label":"Current (electrical)","source":"kg","confidence":0.90},
            {"label":"Ingestion pipeline (semantic)","source":"semantic","confidence":0.76},
            {"label":"Telemetry intake (ArchiMate)","source":"archimate","confidence":0.70},
        ],
    )
    top_sources = [c["source"] for c in ranked[:2]]
    assert "kg" not in top_sources

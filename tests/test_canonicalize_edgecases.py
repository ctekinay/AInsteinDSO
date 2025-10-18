# tests/test_canonicalize_edgecases.py
from src.utils.citation_ids import configure_prefixes, canonicalize_id

def setup_function(_):
    configure_prefixes({
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "eurlex": "http://data.europa.eu/eli/terms/",
        "default4": "http://vocab.alliander.com/default4/",
    })

def test_curie_case_normalization():
    assert canonicalize_id("SKOS:Concept") == "skos:Concept"

def test_path_vs_fragment():
    assert canonicalize_id("http://data.europa.eu/eli/terms/631-20") == "eurlex:631-20"
    assert canonicalize_id("http://data.europa.eu/eli/terms#631-20") == "eurlex:631-20"

def test_http_https_and_www():
    a = canonicalize_id("https://www.w3.org/2004/02/skos/core#Concept")
    b = canonicalize_id("http://w3.org/2004/02/skos/core#Concept")
    assert a == b == "skos:Concept"

def test_unknown_prefix_learns_extra():
    assert canonicalize_id("default4:Telemetry") == "default4:Telemetry"
    assert canonicalize_id("http://vocab.alliander.com/default4/Telemetry") == "default4:Telemetry"

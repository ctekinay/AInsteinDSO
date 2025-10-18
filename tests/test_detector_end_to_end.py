# tests/test_detector_end_to_end.py
from src.knowledge.homonym_detector import detect_ambiguous_terms_in_rdf

class _KGLoaderStub:
    def __init__(self, graph):
        self.graph = graph

def test_detector_flags_expected_ambiguities(toy_graph):
    kg = _KGLoaderStub(toy_graph)
    amb = detect_ambiguous_terms_in_rdf(kg)
    for lbl in ("service","model","network","flow","phase"):
        assert lbl in amb, f"Expected ambiguous label '{lbl}'"
        assert len({i["context"] for i in amb[lbl]}) >= 2

def test_detector_not_flag_non_ambiguous(toy_graph):
    kg = _KGLoaderStub(toy_graph)
    amb = detect_ambiguous_terms_in_rdf(kg)
    assert "current" not in amb, "Non-ambiguous 'current' should not be flagged"

def test_detector_handles_missing_definition(toy_graph):
    kg = _KGLoaderStub(toy_graph)
    amb = detect_ambiguous_terms_in_rdf(kg)
    assert "requirement" not in amb, "Single label without definition is not ambiguous by itself"

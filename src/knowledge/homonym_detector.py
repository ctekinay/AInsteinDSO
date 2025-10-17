# src/knowledge/homonym_detector.py
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple
from rdflib import URIRef, Literal, Graph
from rdflib.namespace import RDF, RDFS, SKOS, DCTERMS

# Heuristics to map namespace/prefix → coarse contexts.
# Adjust/extend freely to your corpus.
NAMESPACE_CONTEXT = {
    "iec": "electrical",
    "cim": "electrical",
    "entsoe": "electrical",   # market/grid → treat as electrical-ish for collisions
    "elit": "regulatory",
    "lido": "regulatory",
    "archi": "architectural",
    "default6": "electrical",
    "default11": "electrical",
    "default13": "business_process",
    "default4": "organizational",
}

# Keyword fallback for context classification if prefix isn’t decisive.
KEYWORD_CONTEXT = {
    "electrical": ["voltage", "current", "power", "transformer", "phase", "busbar", "frequency", "reactive", "active", "feeder"],
    "architectural": ["system", "component", "interface", "data flow", "application", "archimate", "topic", "queue", "message", "service", "api", "pipeline", "model"],
    "business_process": ["service provider", "contract", "customer", "market", "workflow", "process"],
    "regulatory": ["regulation", "compliance", "standard", "requirement", "eur-lex", "directive"],
    "organizational": ["department", "unit", "role", "team", "org", "premises"],
}

def _lit_values_lang_pref(g: Graph, s: URIRef, p, preferred: Tuple[str, ...] = ("en", "nl")) -> Iterable[str]:
    """Yield literals for (s, p) preferring specific languages, then no-lang, then the rest."""
    vals = list(g.objects(s, p))
    # exact-language first
    for lang in preferred:
        for o in vals:
            if isinstance(o, Literal) and getattr(o, "language", None) == lang:
                yield str(o)
    # language-agnostic
    for o in vals:
        if isinstance(o, Literal) and getattr(o, "language", None) is None:
            yield str(o)
    # any remaining
    for o in vals:
        if isinstance(o, Literal):
            yield str(o)

def _namespace_maps(g: Graph) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (prefix->base, base->prefix) maps from the graph’s namespace manager (stripped of trailing #/)."""
    prefix_to_base, base_to_prefix = {}, {}
    for pref, ns in g.namespace_manager.namespaces():
        base = str(ns).rstrip("#/")
        prefix_to_base[pref] = base
        base_to_prefix[base] = pref
    return prefix_to_base, base_to_prefix

def _guess_prefix(g: Graph, uri: str, base_to_prefix: Dict[str, str]) -> str:
    """Try to infer the registered prefix from the URI base; fall back to a host-derived token."""
    for base, pref in base_to_prefix.items():
        if uri.startswith(base):
            return pref
    # crude host fallback
    try:
        host = uri.split("://", 1)[1].split("/", 1)[0]
        base = host.split(".")[-2] if "." in host else host
        return base.replace("-", "_").replace(".", "_").lower()
    except Exception:
        return "unknown"

def _guess_context(prefix: str, definition: str) -> str:
    """1) prefix-based context; 2) fallback to keyword-based context; 3) general."""
    if prefix in NAMESPACE_CONTEXT:
        return NAMESPACE_CONTEXT[prefix]
    text = (definition or "").lower()
    for ctx, kws in KEYWORD_CONTEXT.items():
        if sum(1 for kw in kws if kw in text) >= 2:
            return ctx
    return "general"

def detect_ambiguous_terms_in_rdf(kg_loader, lang_pref: Tuple[str, ...] = ("en", "nl")) -> Dict[str, List[dict]]:
    """
    Scan the RDF graph for labels (prefLabel/label) that are used by ≥2 distinct concepts
    mapping to different coarse contexts. Returns:

        {
          "service": [
            {"uri": "...", "prefLabel": "Service", "definition": "...", "prefix": "iec", "context": "electrical"},
            {"uri": "...", "prefLabel": "Service", "definition": "...", "prefix": "archi", "context": "architectural"},
            ...
          ],
          ...
        }
    """
    g: Graph = kg_loader.graph
    _, base_to_prefix = _namespace_maps(g)

    # Prefer SKOS Concepts; fall back to any subject that has prefLabel (handles mixed data)
    subjects = set(g.subjects(RDF.type, SKOS.Concept)) or {s for s, _ in g.subject_objects(SKOS.prefLabel)}

    label_to_concepts: Dict[str, List[dict]] = defaultdict(list)

    for s in subjects:
        # Label (prefLabel first, fallback to rdfs:label)
        pref = next(iter(_lit_values_lang_pref(g, s, SKOS.prefLabel, lang_pref)), None) \
            or next(iter(_lit_values_lang_pref(g, s, RDFS.label, lang_pref)), None)
        if not pref:
            continue

        # Definition fallback chain (skos:definition → dcterms:description → rdfs:comment → "")
        definition = next(iter(_lit_values_lang_pref(g, s, SKOS.definition, lang_pref)), None) \
            or next(iter(_lit_values_lang_pref(g, s, DCTERMS.description, lang_pref)), None) \
            or next(iter(_lit_values_lang_pref(g, s, RDFS.comment, lang_pref)), "")

        uri = str(s)
        prefix = _guess_prefix(g, uri, base_to_prefix)
        context = _guess_context(prefix, definition or "")

        label_to_concepts[pref.strip().lower()].append({
            "uri": uri,
            "prefLabel": pref,
            "definition": definition or "",
            "prefix": prefix,
            "context": context,
        })

    # Keep only labels that map to multiple contexts
    ambiguous: Dict[str, List[dict]] = {}
    for label, items in label_to_concepts.items():
        ctxs = {i["context"] for i in items}
        if len(ctxs) >= 2:
            ambiguous[label] = items

    return ambiguous

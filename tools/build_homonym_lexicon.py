#!/usr/bin/env python3
# tools/build_homonym_lexicon.py
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, SKOS, DCTERMS
from lxml import etree

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("homonym_builder")

NAMESPACE_CONTEXT = {
    "iec": "electrical","cim": "electrical","entsoe": "electrical",
    "elit": "regulatory","lido": "regulatory",
    "archi": "architectural",
    "default6": "electrical","default11":"electrical",
    "default13":"business_process","default4":"organizational",
}
KEYWORD_CONTEXT = {
    "electrical": ["voltage","current","power","transformer","phase","busbar","frequency","reactive","active","feeder"],
    "architectural": ["system","component","interface","data flow","application","archimate","topic","queue","message","service","api","pipeline","model"],
    "business_process": ["service provider","contract","customer","market","workflow","process"],
    "regulatory": ["regulation","compliance","standard","requirement","eur-lex","directive"],
    "organizational": ["department","unit","role","team","org","premises"],
}
MIN_NUDGE, MAX_NUDGE = -0.8, 0.8
def clamp(x: float) -> float: return max(MIN_NUDGE, min(MAX_NUDGE, x))

def _lit_values_lang_pref(g: Graph, s: URIRef, p, preferred: Tuple[str, ...]):
    vals = list(g.objects(s, p))
    for lang in preferred:
        for o in vals:
            if isinstance(o, Literal) and getattr(o, "language", None) == lang:
                yield str(o)
    for o in vals:
        if isinstance(o, Literal) and getattr(o, "language", None) is None:
            yield str(o)
    for o in vals:
        if isinstance(o, Literal):
            yield str(o)

def _namespace_maps(g: Graph):
    p2b, b2p = {}, {}
    for pref, ns in g.namespace_manager.namespaces():
        base = str(ns).rstrip("#/")
        p2b[pref] = base; b2p[base] = pref
    return p2b, b2p

def _guess_prefix(g: Graph, uri: str, b2p: Dict[str,str]) -> str:
    for base, pref in b2p.items():
        if uri.startswith(base): return pref
    try:
        host = uri.split("://", 1)[1].split("/", 1)[0]
        base = host.split(".")[-2] if "." in host else host
        return base.replace("-", "_").replace(".", "_").lower()
    except Exception:
        return "unknown"

def _guess_context(prefix: str, definition: str) -> str:
    if prefix in NAMESPACE_CONTEXT: return NAMESPACE_CONTEXT[prefix]
    text = (definition or "").lower()
    for ctx, kws in KEYWORD_CONTEXT.items():
        if sum(1 for kw in kws if kw in text) >= 2: return ctx
    return "general"

def iter_concepts(g: Graph, lang_pref: Tuple[str, ...]):
    _, b2p = _namespace_maps(g)
    subs = set(g.subjects(RDF.type, SKOS.Concept)) or {s for s, _ in g.subject_objects(SKOS.prefLabel)}
    for s in subs:
        pref = next(iter(_lit_values_lang_pref(g, s, SKOS.prefLabel, lang_pref)), None) \
            or next(iter(_lit_values_lang_pref(g, s, RDFS.label, lang_pref)), None)
        if not pref: continue
        definition = next(iter(_lit_values_lang_pref(g, s, SKOS.definition, lang_pref)), None) \
            or next(iter(_lit_values_lang_pref(g, s, DCTERMS.description, lang_pref)), None) \
            or next(iter(_lit_values_lang_pref(g, s, RDFS.comment, lang_pref)), "")
        uri = str(s)
        prefix = _guess_prefix(g, uri, b2p)
        context = _guess_context(prefix, definition or "")
        yield {"uri": uri, "prefLabel": pref, "definition": definition or "", "prefix": prefix, "context": context}

def detect_ambiguous_terms(g: Graph, lang_pref: Tuple[str, ...]) -> Dict[str, List[dict]]:
    label_to_concepts: Dict[str, List[dict]] = {}
    for c in iter_concepts(g, lang_pref):
        k = c["prefLabel"].strip().lower()
        label_to_concepts.setdefault(k, []).append(c)
    ambiguous = {}
    for label, items in label_to_concepts.items():
        ctxs = {i["context"] for i in items}
        if len(ctxs) >= 2:
            ambiguous[label] = items
    return ambiguous

def load_archimate_terms(paths: List[Path]) -> Set[str]:
    names: Set[str] = set()
    for p in paths:
        try:
            tree = etree.parse(str(p))
            for el in tree.xpath("//*[@name]"):
                nm = (el.attrib.get("name") or "").strip()
                if nm: names.add(nm.lower())
        except Exception as e:
            log.warning("Archi parse failed for %s: %s", p, e)
    return names

def generate_lexicon(ambiguous: Dict[str, List[dict]], archi_terms: Set[str]) -> Dict[str, Dict]:
    lex: Dict[str, Dict] = {}
    for term, items in ambiguous.items():
        ctxs = {i["context"] for i in items}
        has_elec = "electrical" in ctxs
        has_non_elec = any(c in ctxs for c in ("architectural","business_process","organizational","regulatory"))
        in_archi = term in archi_terms
        entry: Dict[str, Dict] = {}
        if has_elec and has_non_elec:
            entry["architectural_guidance"] = {
                "preferred_sense": "non_electrical","avoid_sense": "electrical",
                "kg_weight_adjustment": clamp(-0.6),
                "semantic_weight_adjustment": clamp(0.3),
                "archimate_weight_adjustment": clamp(0.4 if in_archi else 0.3),
            }
            entry["knowledge_retrieval"] = {"preferred_sense": "electrical","kg_weight_adjustment": clamp(0.2)}
            entry["analytical_task"] = {
                "preferred_sense": "multi_source",
                "kg_weight_adjustment": 0.0,
                "semantic_weight_adjustment": clamp(0.1),
                "archimate_weight_adjustment": clamp(0.1 if in_archi else 0.0),
            }
        elif len(ctxs) >= 2:
            entry["analytical_task"] = {"preferred_sense":"multi_source","kg_weight_adjustment":0.0}
        if entry: lex[term] = entry
    return lex

def merge_overrides(lex: Dict, override_path: Path | None) -> Dict:
    if not override_path or not override_path.exists(): return lex
    try:
        overrides = json.loads(override_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed reading overrides %s: %s", override_path, e); return lex
    for term, by_mode in overrides.items():
        t = term.lower(); lex.setdefault(t, {})
        for mode, payload in (by_mode or {}).items():
            dst = lex[t].setdefault(mode, {})
            for k, v in (payload or {}).items():
                if k.endswith("_weight_adjustment") and isinstance(v, (int,float)):
                    v = clamp(float(v))
                dst[k] = v
    return lex

def main():
    ap = argparse.ArgumentParser(description="Generate homonym_lexicon.json + ambiguous_terms.json")
    ap.add_argument("--ttl", required=True)
    ap.add_argument("--archi", nargs="*", default=[])
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--lang", nargs="*", default=["en","nl"])
    ap.add_argument("--amb-out", required=True)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    g = Graph()
    log.info("Loading RDF: %s", args.ttl)
    g.parse(str(args.ttl), format="turtle")

    amb = detect_ambiguous_terms(g, tuple(args.lang))
    Path(args.amb_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.amb_out).write_text(json.dumps(amb, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Wrote %s (%d labels)", args.amb_out, len(amb))

    archi_terms = load_archimate_terms([Path(p) for p in args.archi]) if args.archi else set()
    lex = generate_lexicon(amb, archi_terms)
    if args.overrides:
        lex = merge_overrides(lex, Path(args.overrides))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(lex, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Wrote %s (%d terms)", args.out, len(lex))

if __name__ == "__main__":
    main()

# src/knowledge/homonym_guard.py
from __future__ import annotations
from typing import Dict, List
import logging, re

logger = logging.getLogger(__name__)

# In-memory lexicon (lowercased terms)
_LEX: Dict[str, Dict[str, Dict]] = {}

# Cache compiled regex for terms
_TERM_CACHE: Dict[str, re.Pattern] = {}

# Normalize various source labels into a small set
_SOURCE_ALIASES = {
    "kg": "kg",
    "knowledge_graph": "kg",
    "graph": "kg",

    "semantic": "semantic",
    "embedding": "semantic",
    "vector": "semantic",

    "archimate": "archimate",
    "archi": "archimate",
    "model": "archimate",

    "doc": "doc",
    "document": "doc",
    "pdf": "doc",
    "file": "doc",
}

# Priority used for stable sorting after nudging
_SOURCE_PRIORITY = {
    "archimate": 3,
    "doc": 2,
    "semantic": 2,
    "kg": 1,
    "other": 0,
}

def _normalize_source(s: str | None) -> str:
    if not s:
        return "other"
    key = s.strip().lower()
    return _SOURCE_ALIASES.get(key, key if key in _SOURCE_PRIORITY else "other")

def _term_regex(term: str) -> re.Pattern:
    """
    Word-boundary regex with simple pluralization (s/es).
    Prevents 'model' matching 'remodeled'.
    """
    r = _TERM_CACHE.get(term)
    if not r:
        r = re.compile(rf"\b{re.escape(term)}(es|s)?\b", re.I)
        _TERM_CACHE[term] = r
    return r

def set_homonym_lexicon(lex: Dict[str, Dict[str, Dict]]):
    """
    Load/replace the homonym lexicon.
    Expected shape (all keys case-insensitive):
      {
        "current": {
          "architectural_guidance": {
            "kg_weight_adjustment": -0.6,
            "semantic_weight_adjustment": +0.3,
            "archimate_weight_adjustment": +0.4,
            "preferred_sense": "…",
            "avoid_sense": "…"
          },
          "knowledge_retrieval": { "kg_weight_adjustment": +0.2 },
          "analytical_task": { ... }
        },
        ...
      }
    """
    global _LEX
    _LEX = {str(k).lower(): v or {} for k, v in (lex or {}).items()}

def apply_homonym_guard(
    query: str,
    primary_category: str,             # "architectural_guidance" | "knowledge_retrieval" | "analytical_task"
    candidates: List[Dict],
    intent_confidence: float = 1.0,    # allow caller to pass classifier confidence
) -> List[Dict]:
    """
    Soft re-ranking of candidates. Never filters; only nudges confidences.

    Each candidate should minimally contain:
      - "label" or "element": str
      - "source": one of {"kg","semantic","archimate","doc"} (aliases normalized)
      - "confidence": float
    """
    if not candidates or not _LEX:
        return candidates

    # If the intent router is uncertain, skip nudging and let your dual-track fallback reconcile.
    if intent_confidence < 0.60:
        return _stable_sort(candidates)

    q = (query or "").lower()
    fired_terms = []

    for term, by_mode in _LEX.items():
        if term in q and primary_category in by_mode:
            rule = by_mode[primary_category] or {}
            rx = _term_regex(term)
            adjusted_any = False

            for c in candidates:
                # normalize fields
                src = c["source"] = _normalize_source(c.get("source"))
                text = (c.get("element") or c.get("label") or "").lower()
                try:
                    conf = float(c.get("confidence") or 0.0)
                except Exception:
                    conf = 0.0

                if not text or conf <= 0.0:
                    continue

                if rx.search(text):
                    # apply source-specific nudges
                    if src == "kg":
                        conf *= (1 + float(rule.get("kg_weight_adjustment", 0)))
                    elif src == "semantic":
                        conf *= (1 + float(rule.get("semantic_weight_adjustment", 0)))
                    elif src == "archimate":
                        conf *= (1 + float(rule.get("archimate_weight_adjustment", 0)))

                    # write back (never let it go negative)
                    c["confidence"] = max(conf, 0.0)
                    adjusted_any = True

            if adjusted_any:
                fired_terms.append(term)

    if fired_terms:
        logger.info(
            "Homonym guard applied | terms=%s | mode=%s | conf>=%.2f",
            sorted(set(fired_terms)),
            primary_category,
            intent_confidence,
        )

    return _stable_sort(candidates)

def _stable_sort(cands: List[Dict]) -> List[Dict]:
    """
    Deterministic sorting after adjustments:
      1) confidence (desc)
      2) source priority (desc)
      3) original index (stable)
    """
    # preserve original order index
    indexed = []
    for i, c in enumerate(cands):
        # ensure normalized source for comparison
        c["source"] = _normalize_source(c.get("source"))
        try:
            conf = float(c.get("confidence") or 0.0)
        except Exception:
            conf = 0.0
        indexed.append((i, conf, _SOURCE_PRIORITY.get(c["source"], 0), c))

    indexed.sort(key=lambda t: (t[1], t[2], -t[0]), reverse=True)
    return [c for (_i, _conf, _prio, c) in indexed]

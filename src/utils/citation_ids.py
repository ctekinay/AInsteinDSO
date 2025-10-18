from __future__ import annotations
import re
from typing import Dict, Set, Tuple

__all__ = [
    "configure_prefixes",
    "canonicalize_id",
    "get_known_prefixes",
    "__CITATION_IDS_VERSION__",
]

# Bump this so you can verify you're on the fixed file
__CITATION_IDS_VERSION__ = "2025-10-15-robust-fixed"

# Registries
_P2BASES: Dict[str, Set[str]] = {}   # prefix -> set(normalized base variants)
_BASE2P: Dict[str, str] = {}         # normalized base -> prefix

_CURIE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*:[^\s]+$")

def _strip_scheme_and_www(url: str) -> str:
    s = url.strip()
    s = re.sub(r"^https?://", "", s, flags=re.I)
    s = re.sub(r"^www\.", "", s, flags=re.I)
    return s

def _normalize_base(base: str) -> str:
    # Normalization = strip scheme + www only (keep tail)
    return _strip_scheme_and_www(base)

def _base_variants(base: str) -> Set[str]:
    """
    For a declared base like 'http://.../eli/terms/' we register normalized variants:
      'data.europa.eu/eli/terms/'
      'data.europa.eu/eli/terms#'
      'data.europa.eu/eli/terms'
    Matching then uses 'longest prefix wins'.
    """
    norm = _normalize_base(base)
    root = norm.rstrip("/#")
    return {norm, root + "/", root + "#", root}

def _register_base(prefix: str, base: str) -> None:
    variants = _base_variants(base)
    bucket = _P2BASES.setdefault(prefix, set())
    for v in variants:
        bucket.add(v)
        _BASE2P[v] = prefix

def configure_prefixes(from_graph: Dict[str, str] | None = None,
                       extra_prefixes: Dict[str, str] | None = None) -> None:
    """Reset and load bases from an rdflib graph’s namespaces and/or extras."""
    _P2BASES.clear()
    _BASE2P.clear()
    if from_graph:
        for p, base in from_graph.items():
            if p and base:
                _register_base(p.lower(), str(base))
    if extra_prefixes:
        for p, base in extra_prefixes.items():
            if p and base:
                _register_base(p.lower(), str(base))

def get_known_prefixes() -> Dict[str, str]:
    """Pretty-print view; not used for matching."""
    out: Dict[str, str] = {}
    for p, variants in _P2BASES.items():
        pick = next((v for v in variants if v.endswith("/") or v.endswith("#")), None)
        out[p] = "http://" + (pick or next(iter(variants)))
    return out

def _pick_longest_matching_base(url_norm: str, raw_url: str | None = None) -> tuple[str | None, str | None]:
    """
    Return (prefix, base_variant) for the **longest** registered base that matches.

    We do two passes:
      1) Compare the normalized URL (scheme+www stripped) to registered normalized bases.
      2) If no hit, compare the RAW URL against scheme/www-expanded forms of each base,
         then convert that expanded variant back to its normalized twin for slicing.
    """
    # Pass 1: normalized vs normalized bases
    best_pref = None
    best_base = None
    best_len = -1
    for base, pref in _BASE2P.items():
        if url_norm.startswith(base) and len(base) > best_len:
            best_len = len(base)
            best_base = base
            best_pref = pref
    if best_pref:
        return best_pref, best_base

    # Pass 2: try raw URL against expanded variants
    if raw_url:
        # Generate scheme/www variants for each registered base
        def _expanded_variants(b: str):
            return (
                f"http://{b}", f"https://{b}",
                f"http://www.{b}", f"https://www.{b}",
            )

        best_pref = None
        best_len = -1
        best_variant = None
        for base, pref in _BASE2P.items():
            for variant in _expanded_variants(base):
                if raw_url.startswith(variant) and len(variant) > best_len:
                    best_len = len(variant)
                    best_variant = variant
                    best_pref = pref

        if best_pref and best_variant:
            # Convert matched raw variant back to its normalized base for slicing
            norm_variant = _strip_scheme_and_www(best_variant)
            return best_pref, norm_variant

    return None, None


def canonicalize_id(raw: str) -> str:
    """
    CURIE:
      - lowercases the prefix: 'SKOS:Concept' → 'skos:Concept'
    URL:
      - try both normalized-URL and raw-URL matches (scheme/www variants),
      - return 'prefix:local' on hit, else return original string.
    """
    if not raw:
        return raw
    s = raw.strip()

    # Already CURIE → normalize prefix case only
    if _CURIE_RE.match(s):
        pref, local = s.split(":", 1)
        return f"{pref.lower()}:{local}"

    # URL → try to convert
    if "://" in s:
        url_norm = _strip_scheme_and_www(s)
        pref, base = _pick_longest_matching_base(url_norm, raw_url=s)  # ← pass raw_url
        if pref and base is not None:
            local = url_norm[len(base):].lstrip("/#")
            if local:
                return f"{pref}:{local}"

    # Fallback: unchanged
    return s

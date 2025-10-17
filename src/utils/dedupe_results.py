# src/utils/dedupe_results.py
from __future__ import annotations
from typing import List, Dict, Tuple

def _citation_of(c: Dict) -> str | None:
    # Accept both 'citation' and 'citation_id'
    return c.get("citation") or c.get("citation_id")

def dedupe_candidates(candidates: List[Dict]) -> List[Dict]:
    """
    Keep at most one item per citation (preferring highest confidence),
    and keep all items that have no citation at all.
    Order is stable among kept items (best-of-group keeps first position).
    """
    if not candidates:
        return []

    best_by_cit: Dict[str, Tuple[float, int, Dict]] = {}
    no_citation_bucket: List[Tuple[int, Dict]] = []

    for idx, c in enumerate(candidates):
        cit = _citation_of(c)
        conf = float(c.get("confidence", 0.0))
        if not cit:
            no_citation_bucket.append((idx, c))
            continue
        prev = best_by_cit.get(cit)
        if not prev or conf > prev[0]:
            best_by_cit[cit] = (conf, idx, c)

    # Reconstruct in original order: first the best representatives in the order they appeared,
    # then the no-citation items in their original order.
    kept = sorted(best_by_cit.values(), key=lambda t: t[1])
    out = [c for (_conf, _idx, c) in kept] + [c for (_i, c) in no_citation_bucket]
    return out

"""Split a global batch of B sequences into whole sequences per node."""
from __future__ import annotations

from typing import Dict, List


def split_counts(B: int, shares: Dict[str, float], members: List[str]) -> Dict[str, int]:
    """
    Largest-remainder split of B by shares, over `members` only. Always sums to B.
    Ties go to the member listed first, so every node computes the same answer.
    Missing or all-zero shares mean an even split.
    """
    if not members:
        return {}
    w = {m: max(0.0, float(shares.get(m, 0.0))) for m in members}
    total = sum(w.values())
    if total <= 0:
        w, total = {m: 1.0 for m in members}, float(len(members))
    exact = {m: B * w[m] / total for m in members}
    counts = {m: int(exact[m]) for m in members}
    left = B - sum(counts.values())
    order = sorted(members, key=lambda m: (-(exact[m] - counts[m]), members.index(m)))
    for m in order[:left]:
        counts[m] += 1
    return counts

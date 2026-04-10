"""Phase 2 ranker placeholder."""

from __future__ import annotations


class Ranker:
    """Placeholder for DNN ranker described in Master.md Phase 2."""

    def __init__(self) -> None:
        self.ready = False

    def score(self, _features):
        raise NotImplementedError("Phase 2 ranker is not implemented in MVP.")

"""Deduplication methods: exact hash and MinHash-LSH fuzzy dedup."""

from __future__ import annotations

import hashlib
from typing import Any

from dokime.core.filters import Filter


class ExactDedup(Filter):
    """Remove exact duplicates using SHA-256 hashing."""

    def __init__(self, text_field: str = "text") -> None:
        self.text_field = text_field
        self._seen: set[str] = set()

    def filter(self, sample: dict[str, Any]) -> bool:
        text = sample.get(self.text_field, "")
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in self._seen:
            return False
        self._seen.add(h)
        return True

    def name(self) -> str:
        return "ExactDedup"


class MinHashDedup(Filter):
    """Remove near-duplicates using MinHash-LSH.

    Requires: pip install dokime[dedup]
    """

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        text_field: str = "text",
    ) -> None:
        try:
            from datasketch import MinHashLSH
        except ImportError:
            raise ImportError("Install dedup support: pip install dokime[dedup]") from None

        self.threshold = threshold
        self.num_perm = num_perm
        self.text_field = text_field
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._count = 0

    def _get_minhash(self, text: str) -> Any:
        from datasketch import MinHash

        m = MinHash(num_perm=self.num_perm)
        for word in text.lower().split():
            m.update(word.encode("utf-8"))
        return m

    def filter(self, sample: dict[str, Any]) -> bool:
        text = sample.get(self.text_field, "")
        if not text:
            return False

        mh = self._get_minhash(text)
        result = self._lsh.query(mh)

        if result:
            return False

        key = f"doc_{self._count}"
        self._lsh.insert(key, mh)
        self._count += 1
        return True

    def name(self) -> str:
        return f"MinHashDedup(threshold={self.threshold})"

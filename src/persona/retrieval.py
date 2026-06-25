"""Hybrid retrieval pipeline. Fetch-then-rerank with adjacent-turn stitching."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from persona_core.qdrant_store import QdrantStore, ScoredHit
from persona_core.records import EpisodicRecord, RecordType

from persona.scoring import (
    DEFAULT_CONSOLIDATION_MULTIPLIER,
    ScoreWeights,
    final_score,
)


@dataclass(frozen=True)
class RetrievalConfig:
    candidate_n: int = 20
    top_k: int = 4
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    consolidation_multiplier: float = DEFAULT_CONSOLIDATION_MULTIPLIER
    stitch_radius: int = 1
    types: list[RecordType] | None = None


@dataclass
class RetrievedItem:
    record: EpisodicRecord
    score: float
    similarity: float
    is_stitch: bool = False


def retrieve_relevant(
    store: QdrantStore,
    query_vector: list[float],
    config: RetrievalConfig,
    now: datetime,
) -> list[RetrievedItem]:
    candidates = store.query(
        query_vector=query_vector,
        limit=config.candidate_n,
        types=config.types,
    )
    if not candidates:
        return []

    scored = _rerank(candidates, config, now)
    top = scored[: config.top_k]

    if config.stitch_radius > 0:
        stitched = _gather_stitches(store, top, config.stitch_radius)
        # Merge, deduplicate by id, preserve order: top_k first, then stitches.
        seen = {item.record.id for item in top}
        for s in stitched:
            if s.record.id not in seen:
                top.append(s)
                seen.add(s.record.id)

    return top


def _rerank(
    candidates: list[ScoredHit],
    config: RetrievalConfig,
    now: datetime,
) -> list[RetrievedItem]:
    items: list[RetrievedItem] = []
    for hit in candidates:
        score = final_score(
            record=hit.record,
            similarity=hit.similarity,
            weights=config.weights,
            now=now,
            consolidation_factor=config.consolidation_multiplier,
        )
        items.append(RetrievedItem(record=hit.record, score=score, similarity=hit.similarity))
    items.sort(key=lambda i: i.score, reverse=True)
    return items


def _gather_stitches(
    store: QdrantStore,
    top: list[RetrievedItem],
    radius: int,
) -> list[RetrievedItem]:
    out: list[RetrievedItem] = []
    for item in top:
        if item.record.session_id is None:
            continue
        prior = store.fetch_session_window(
            session_id=item.record.session_id,
            before=item.record.created_at,
            limit=radius,
        )
        for r in prior:
            out.append(RetrievedItem(record=r, score=0.0, similarity=0.0, is_stitch=True))
    return out

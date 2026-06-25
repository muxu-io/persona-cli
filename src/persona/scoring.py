"""Hybrid retrieval scoring. Pure functions; no I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from persona_core.records import EpisodicRecord

# Tunables — defaults from design doc §9.2
DEFAULT_CONSOLIDATION_MULTIPLIER = 0.6
RECENCY_HALF_LIFE_MINUTES = 60 * 24 * 7  # one week


@dataclass(frozen=True)
class ScoreWeights:
    similarity: float = 0.6
    recency: float = 0.2
    salience: float = 0.2


def recency_score(record: EpisodicRecord, now: datetime) -> float:
    """Exponential decay with one-week half-life. Returns value in (0, 1]."""
    age_minutes = (now - record.created_at).total_seconds() / 60.0
    if age_minutes <= 0:
        return 1.0
    return math.pow(0.5, age_minutes / RECENCY_HALF_LIFE_MINUTES)


def consolidation_multiplier(
    record: EpisodicRecord,
    multiplier: float = DEFAULT_CONSOLIDATION_MULTIPLIER,
) -> float:
    return 1.0 if record.consolidated else multiplier


def final_score(
    record: EpisodicRecord,
    similarity: float,
    weights: ScoreWeights,
    now: datetime,
    consolidation_factor: float = DEFAULT_CONSOLIDATION_MULTIPLIER,
) -> float:
    base = (
        weights.similarity * similarity
        + weights.recency * recency_score(record, now)
        + weights.salience * record.current_salience
    )
    return base * consolidation_multiplier(record, consolidation_factor)

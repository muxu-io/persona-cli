from datetime import UTC, datetime, timedelta

import pytest
from persona_core.records import EpisodicRecord, RecordType

from persona.scoring import (
    ScoreWeights,
    consolidation_multiplier,
    final_score,
    recency_score,
)


def _rec(consolidated: bool = False, salience: float = 0.4, age_minutes: float = 0.0):
    rec = EpisodicRecord(
        type=RecordType.TURN_PAIR,
        session_id="s",
        content={"user": "x", "assistant": "y"},
        initial_salience=salience,
        consolidated=consolidated,
    )
    rec.created_at = datetime.now(tz=UTC) - timedelta(minutes=age_minutes)
    return rec


def test_recency_recent_is_higher_than_old():
    new = recency_score(_rec(age_minutes=0), now=datetime.now(tz=UTC))
    old = recency_score(_rec(age_minutes=10080), now=datetime.now(tz=UTC))  # one week
    assert 0.0 < old < new <= 1.0


def test_recency_zero_age_is_one():
    now = datetime.now(tz=UTC)
    assert recency_score(_rec(age_minutes=0), now=now) == 1.0


def test_consolidation_multiplier_dampens_unconsolidated():
    assert consolidation_multiplier(_rec(consolidated=True)) == 1.0
    assert consolidation_multiplier(_rec(consolidated=False)) == 0.6


def test_final_score_combines_components():
    rec = _rec(consolidated=True, salience=0.5, age_minutes=0)
    weights = ScoreWeights(similarity=0.6, recency=0.2, salience=0.2)
    # Pin `now` to created_at so recency is exactly 1.0 (no float drift on age).
    score = final_score(
        record=rec,
        similarity=0.8,
        weights=weights,
        now=rec.created_at,
    )
    # 0.6*0.8 + 0.2*1.0 + 0.2*0.5 = 0.48 + 0.2 + 0.1 = 0.78, multiplier=1.0.
    # Use approx because IEEE 754 makes the literal sum 0.7799999999999999.
    assert score == pytest.approx(0.78, rel=1e-9)


def test_final_score_unconsolidated_is_dampened():
    rec = _rec(consolidated=False, salience=0.5, age_minutes=0)
    weights = ScoreWeights(similarity=0.6, recency=0.2, salience=0.2)
    score = final_score(rec, similarity=0.8, weights=weights, now=rec.created_at)
    # base 0.78 * 0.6 = 0.468
    assert score == pytest.approx(0.468, rel=1e-3)

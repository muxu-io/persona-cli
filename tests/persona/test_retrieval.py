from datetime import UTC, datetime, timedelta

import pytest
from persona_core.qdrant_store import QdrantStore
from persona_core.records import EpisodicRecord, RecordType

from persona.retrieval import RetrievalConfig, retrieve_relevant
from persona.scoring import ScoreWeights


@pytest.fixture
def store():
    s = QdrantStore.in_memory(collection="t", vector_size=4, persona_id="t")
    s.ensure_collection()
    yield s


def _rec(content_user, embedding, *, consolidated=False, salience=0.4, age_min=0):
    rec = EpisodicRecord(
        type=RecordType.TURN_PAIR,
        session_id="s-1",
        content={"user": content_user, "assistant": "ok"},
        initial_salience=salience,
        consolidated=consolidated,
        embedding=embedding,
    )
    rec.created_at = datetime.now(tz=UTC) - timedelta(minutes=age_min)
    return rec


def test_retrieve_returns_top_k_after_rerank(store):
    # alpha is most similar, beta is least; gamma sits in between but is consolidated.
    store.write(_rec("alpha", [1.0, 0.0, 0.0, 0.0], consolidated=False))
    store.write(_rec("beta", [0.0, 1.0, 0.0, 0.0], consolidated=False))
    store.write(_rec("gamma", [0.7, 0.7, 0.0, 0.0], consolidated=True))
    config = RetrievalConfig(candidate_n=10, top_k=2, weights=ScoreWeights())
    results = retrieve_relevant(
        store=store,
        query_vector=[1.0, 0.0, 0.0, 0.0],
        config=config,
        now=datetime.now(tz=UTC),
    )
    # Top-K=2 should include alpha and gamma; stitching may add same-session neighbors.
    non_stitch = [r for r in results if not r.is_stitch]
    assert len(non_stitch) == 2
    contents = [r.record.content["user"] for r in non_stitch]
    assert "alpha" in contents
    assert "gamma" in contents  # boosted by being consolidated even though similarity is lower


def test_retrieve_unconsolidated_dampened_below_consolidated(store):
    # Two records with identical similarity; the consolidated one should win.
    store.write(_rec("recent-raw", [1.0, 0.0, 0.0, 0.0], consolidated=False))
    store.write(_rec("recent-owned", [1.0, 0.0, 0.0, 0.0], consolidated=True))
    config = RetrievalConfig(candidate_n=10, top_k=1, weights=ScoreWeights())
    results = retrieve_relevant(
        store=store,
        query_vector=[1.0, 0.0, 0.0, 0.0],
        config=config,
        now=datetime.now(tz=UTC),
    )
    assert results[0].record.content["user"] == "recent-owned"


def test_stitching_pulls_immediately_preceding_turn(store):
    base = datetime.now(tz=UTC)
    for i, name in enumerate(["t0", "t1", "t2", "match"]):
        rec = _rec(name, [1.0, 0.0, 0.0, 0.0])
        rec.created_at = base + timedelta(minutes=i)
        store.write(rec)
    # Make only "match" exactly similar to query; force a stitch lookup.
    config = RetrievalConfig(candidate_n=4, top_k=1, weights=ScoreWeights(), stitch_radius=1)
    results = retrieve_relevant(
        store=store,
        query_vector=[1.0, 0.0, 0.0, 0.0],
        config=config,
        now=base + timedelta(minutes=10),
    )
    contents = [r.record.content["user"] for r in results]
    # The retrieved item plus its immediately preceding stitched neighbor should appear.
    assert "match" in contents
    assert "t2" in contents  # stitched neighbor

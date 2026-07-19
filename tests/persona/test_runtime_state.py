import pytest
from persona_core.fatigue import FatigueLevel
from persona_core.qdrant_store import QdrantStore
from persona_core.records import EpisodicRecord, RecordType

from persona.runtime_state import PresentationState, derive_presentation_state


def _store_with_unconsolidated(n: int) -> QdrantStore:
    store = QdrantStore.in_memory(collection="persona_memory", vector_size=4, persona_id="alice")
    store.ensure_collection()
    for _ in range(n):
        rec = EpisodicRecord(
            type=RecordType.TURN_PAIR,
            session_id="s",
            content={"user": "x", "assistant": "y"},
            initial_salience=0.4,
            consolidated=False,
        )
        rec.embedding = [0.0, 0.0, 0.0, 0.0]
        store.write(rec)
    return store


def test_derive_returns_rested_when_count_zero():
    ps = derive_presentation_state("alice", _store_with_unconsolidated(0))
    assert ps == PresentationState(
        persona_id="alice", fatigue_level=FatigueLevel.RESTED, scenario_id=None
    )


def test_derive_returns_tired_at_default_threshold():
    ps = derive_presentation_state("alice", _store_with_unconsolidated(35))
    assert ps.fatigue_level == FatigueLevel.TIRED


def test_derive_returns_exhausted_at_high_count():
    ps = derive_presentation_state("alice", _store_with_unconsolidated(100))
    assert ps.fatigue_level == FatigueLevel.EXHAUSTED


def test_derive_falls_back_to_rested_when_store_unreachable():
    class _Broken:
        def count_unconsolidated(self):
            raise RuntimeError("qdrant down")

    ps = derive_presentation_state("alice", _Broken())
    assert ps.fatigue_level == FatigueLevel.RESTED
    assert ps.persona_id == "alice"


def test_presentation_state_is_frozen():
    import dataclasses

    ps = PresentationState(persona_id="x", fatigue_level=FatigueLevel.RESTED, scenario_id=None)
    assert dataclasses.is_dataclass(ps)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ps.persona_id = "y"  # type: ignore[misc]

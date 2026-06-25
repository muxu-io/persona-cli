"""Shared presentation-state derivation. Imported by voice-svc today; future
avatar-svc imports the same module so voice and face stay synchronized to the
same source of truth.

Fatigue is derived from the live unconsolidated-memory count in Qdrant (filtered
by persona_id), not from any persisted runtime field — the count is a hot-path
quantity that only the store knows exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from persona_core.qdrant_store import QdrantStore

from persona.fatigue import FatigueLevel, FatigueThresholds, derive_fatigue_level


@dataclass(frozen=True)
class PresentationState:
    persona_id: str
    fatigue_level: FatigueLevel
    scenario_id: str | None


def derive_presentation_state(
    persona_id: str,
    store: QdrantStore,
    thresholds: FatigueThresholds | None = None,
) -> PresentationState:
    """Derive PresentationState from the persona's live Qdrant memory.

    If the store is unreachable, default to RESTED — voice/avatar consumers must
    produce neutral output rather than crash."""
    try:
        unconsolidated = store.count_unconsolidated()
    except Exception:
        return PresentationState(
            persona_id=persona_id,
            fatigue_level=FatigueLevel.RESTED,
            scenario_id=None,
        )
    fatigue = derive_fatigue_level(unconsolidated, thresholds or FatigueThresholds())
    return PresentationState(
        persona_id=persona_id,
        fatigue_level=fatigue,
        scenario_id=None,  # populated when scenario tracking lands
    )

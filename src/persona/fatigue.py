"""Fatigue derivation and the optional system-prompt addendum."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FatigueLevel(StrEnum):
    RESTED = "rested"
    TIRED = "tired"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class FatigueThresholds:
    rested_to_tired: int = 30
    tired_to_exhausted: int = 80


def derive_fatigue_level(unconsolidated_count: int, thresholds: FatigueThresholds) -> FatigueLevel:
    if unconsolidated_count >= thresholds.tired_to_exhausted:
        return FatigueLevel.EXHAUSTED
    if unconsolidated_count >= thresholds.rested_to_tired:
        return FatigueLevel.TIRED
    return FatigueLevel.RESTED


_TIRED_ADDENDUM = (
    "You're feeling cognitively tired — there is a lot of recent experience "
    "you haven't fully made sense of yet. It's harder than usual to integrate "
    "new things on top of all that."
)

_EXHAUSTED_ADDENDUM = (
    "You're cognitively exhausted. You've been going for a while without rest "
    "and the unprocessed material is piling up. Your responses may come out "
    "uneven; that's fine. Don't pretend to be sharper than you are."
)


def fatigue_addendum(level: FatigueLevel) -> str | None:
    """Returns the addendum string when fatigue ≥ tired, else None.

    Default off in production per design doc §3.8 — call sites should gate
    on a feature flag. This function just supplies the text.
    """
    if level == FatigueLevel.TIRED:
        return _TIRED_ADDENDUM
    if level == FatigueLevel.EXHAUSTED:
        return _EXHAUSTED_ADDENDUM
    return None

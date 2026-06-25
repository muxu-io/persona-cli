from persona.fatigue import (
    FatigueLevel,
    FatigueThresholds,
    derive_fatigue_level,
    fatigue_addendum,
)


def test_rested_below_first_threshold():
    assert derive_fatigue_level(0, FatigueThresholds()) == FatigueLevel.RESTED
    assert derive_fatigue_level(29, FatigueThresholds()) == FatigueLevel.RESTED


def test_tired_at_first_threshold():
    assert derive_fatigue_level(30, FatigueThresholds()) == FatigueLevel.TIRED
    assert derive_fatigue_level(79, FatigueThresholds()) == FatigueLevel.TIRED


def test_exhausted_at_second_threshold():
    assert derive_fatigue_level(80, FatigueThresholds()) == FatigueLevel.EXHAUSTED
    assert derive_fatigue_level(500, FatigueThresholds()) == FatigueLevel.EXHAUSTED


def test_custom_thresholds():
    th = FatigueThresholds(rested_to_tired=10, tired_to_exhausted=20)
    assert derive_fatigue_level(9, th) == FatigueLevel.RESTED
    assert derive_fatigue_level(15, th) == FatigueLevel.TIRED
    assert derive_fatigue_level(20, th) == FatigueLevel.EXHAUSTED


def test_fatigue_addendum_text_only_when_tired_or_above():
    assert fatigue_addendum(FatigueLevel.RESTED) is None
    assert fatigue_addendum(FatigueLevel.TIRED) is not None
    assert fatigue_addendum(FatigueLevel.EXHAUSTED) is not None

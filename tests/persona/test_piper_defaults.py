from persona.piper_defaults import PIPER_DEFAULTS, resolve_piper_voice


def test_feminine_keywords_pick_feminine_default():
    assert resolve_piper_voice("cis woman, femme but unfussy") == PIPER_DEFAULTS["feminine"]
    assert resolve_piper_voice("female, mid-30s") == PIPER_DEFAULTS["feminine"]
    assert resolve_piper_voice("Femme") == PIPER_DEFAULTS["feminine"]


def test_masculine_keywords_pick_masculine_default():
    assert resolve_piper_voice("cis man, masc presenting") == PIPER_DEFAULTS["masculine"]
    assert resolve_piper_voice("male, 40s") == PIPER_DEFAULTS["masculine"]


def test_unrecognized_text_picks_neutral():
    assert resolve_piper_voice("non-binary") == PIPER_DEFAULTS["neutral"]
    assert resolve_piper_voice("") == PIPER_DEFAULTS["neutral"]
    assert resolve_piper_voice(None) == PIPER_DEFAULTS["neutral"]


def test_keys_present():
    assert "feminine" in PIPER_DEFAULTS
    assert "masculine" in PIPER_DEFAULTS
    assert "neutral" in PIPER_DEFAULTS

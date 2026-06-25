"""Piper preset voice resolution by presented_gender.

Authored override always wins via tts_params.piper_voice (see spec § 6.2);
this module is the floor when no override is set."""

from __future__ import annotations

import re

PIPER_DEFAULTS: dict[str, str] = {
    "feminine": "en_GB-alba-medium",
    "masculine": "en_GB-northern_english_male-medium",
    "neutral": "en_US-libritts_r-medium",
}

# word-boundary-anchored, case-insensitive
_FEMININE_RE = re.compile(r"\b(woman|female|femme)\b", re.IGNORECASE)
_MASCULINE_RE = re.compile(r"\b(man|male|masc)\b", re.IGNORECASE)


def resolve_piper_voice(presented_gender: str | None) -> str:
    if not presented_gender:
        return PIPER_DEFAULTS["neutral"]
    text = presented_gender
    if _FEMININE_RE.search(text):
        return PIPER_DEFAULTS["feminine"]
    if _MASCULINE_RE.search(text):
        return PIPER_DEFAULTS["masculine"]
    return PIPER_DEFAULTS["neutral"]

"""Resolve a self-contained voice spec for the voice service."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from persona_core.schema import Persona

from persona.fatigue import FatigueLevel
from persona.piper_defaults import resolve_piper_voice


class _MediaGetter(Protocol):
    def get_media(self, persona_id: str, name: str) -> bytes | None: ...


@dataclass(frozen=True)
class VoiceSpec:
    engine: str  # "xtts-v2" | "piper"
    language: str
    baseline_speed: float
    fatigue_level: str
    piper_voice: str | None
    sample_audio_b64: str | None


def resolve_voice_spec(
    persona: Persona,
    fatigue_level: FatigueLevel,
    media: _MediaGetter,
) -> VoiceSpec:
    voice_atoms = _voice_atoms(persona)
    presented_gender = _presented_gender(persona)

    tts_ref = voice_atoms.get("tts_ref")
    tts_params = voice_atoms.get("tts_params") or {}

    explicit_engine = tts_params.get("engine")
    language = tts_params.get("language", "en")
    baseline_speed = float(tts_params.get("speed", 1.0))
    explicit_piper_voice = tts_params.get("piper_voice")

    # Resolve the sample from store-svc by media key. Path(tts_ref).name keeps both
    # new bare-name refs and legacy full-path refs resolving to the same key.
    sample_bytes: bytes | None = None
    if tts_ref:
        sample_bytes = media.get_media(persona.persona_id, Path(str(tts_ref)).name)

    if explicit_engine == "piper":
        engine = "piper"
        sample_bytes = None
    elif explicit_engine == "xtts-v2" and sample_bytes is not None:
        engine = "xtts-v2"
    elif sample_bytes is not None:
        engine = "xtts-v2"
    else:
        engine = "piper"

    piper_voice = explicit_piper_voice if engine == "piper" else None
    if engine == "piper" and not piper_voice:
        piper_voice = resolve_piper_voice(presented_gender)

    sample_b64: str | None = None
    if engine == "xtts-v2" and sample_bytes is not None:
        sample_b64 = base64.b64encode(sample_bytes).decode("ascii")

    return VoiceSpec(
        engine=engine,
        language=language,
        baseline_speed=baseline_speed,
        fatigue_level=str(fatigue_level),
        piper_voice=piper_voice,
        sample_audio_b64=sample_b64,
    )


def _voice_atoms(persona: Persona) -> dict:
    dim = persona.substrate.get("physical.voice")
    if dim is None:
        return {}
    return dict(dim.structured)


def _presented_gender(persona: Persona) -> str | None:
    dim = persona.substrate.get("physical.age_and_presentation")
    if dim is None:
        return None
    return dim.structured.get("presented_gender")

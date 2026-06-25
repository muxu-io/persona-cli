from __future__ import annotations

import base64

from persona_core.parser import parse_persona_file

from persona.fatigue import FatigueLevel
from persona.voice_spec import resolve_voice_spec

_HEADER = (
    "---\n"
    "persona_id: tester\n"
    "spec_version: 1\n"
    "---\n\n"
    "# Tester\n\n"
    "## Identity\n\n"
    "```yaml\n"
    "name: Tester\n"
    "age: 30\n"
    "```\n\n"
)


def _write_persona(tmp_path, body: str):
    p = tmp_path / "personas" / "tester.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_HEADER + body, encoding="utf-8")
    return parse_persona_file(p)


class _FakeMedia:
    """In-memory media getter keyed by bare filename."""

    def __init__(self, blobs: dict[str, bytes] | None = None) -> None:
        self._blobs = blobs or {}

    def get_media(self, persona_id: str, name: str) -> bytes | None:
        return self._blobs.get(name)


_NO_MEDIA = _FakeMedia()


def test_piper_fallback_when_no_sample(tmp_path):
    body = (
        "## Substrate\n\n"
        "### Physical · Voice\n\n"
        "```yaml\n"
        "presence: always_on\n"
        "tts_params:\n"
        "  language: en\n"
        "```\n\n"
        "A plain speaking voice.\n\n"
        "### Physical · Age and presentation\n\n"
        "```yaml\n"
        "presence: always_on\n"
        "presented_gender: woman, mid-30s\n"
        "```\n\n"
        "Presents as a woman.\n"
    )
    persona = _write_persona(tmp_path, body)
    spec = resolve_voice_spec(persona, FatigueLevel.RESTED, media=_NO_MEDIA)
    assert spec.engine == "piper"
    assert spec.piper_voice == "en_GB-alba-medium"  # feminine default
    assert spec.sample_audio_b64 is None
    assert spec.fatigue_level == "rested"
    assert spec.language == "en"
    assert spec.baseline_speed == 1.0


def test_xtts_when_sample_present(tmp_path):
    media = _FakeMedia({"voice-sample.wav": b"RIFFwavbytes"})
    body = (
        "## Substrate\n\n"
        "### Physical · Voice\n\n"
        "```yaml\n"
        "presence: always_on\n"
        "tts_ref: state/tester/media/voice-sample.wav\n"
        "tts_params:\n"
        "  language: en\n"
        "  speed: 1.1\n"
        "```\n\n"
        "A recorded voice.\n"
    )
    persona = _write_persona(tmp_path, body)
    spec = resolve_voice_spec(persona, FatigueLevel.TIRED, media=media)
    assert spec.engine == "xtts-v2"
    assert spec.baseline_speed == 1.1
    assert spec.fatigue_level == "tired"
    assert spec.piper_voice is None
    assert base64.b64decode(spec.sample_audio_b64) == b"RIFFwavbytes"


def test_xtts_resolves_bare_name_ref(tmp_path):
    """A bare-filename tts_ref resolves to the same media key as a full-path ref."""
    media = _FakeMedia({"voice-sample.wav": b"SAMPLEBYTES"})
    body = (
        "## Substrate\n\n"
        "### Physical · Voice\n\n"
        "```yaml\n"
        "presence: always_on\n"
        "tts_ref: voice-sample.wav\n"
        "tts_params:\n"
        "  engine: xtts-v2\n"
        "```\n\n"
        "A recorded voice.\n"
    )
    persona = _write_persona(tmp_path, body)
    spec = resolve_voice_spec(persona, FatigueLevel.RESTED, media=media)
    assert spec.engine == "xtts-v2"
    assert base64.b64decode(spec.sample_audio_b64) == b"SAMPLEBYTES"


def test_explicit_piper_engine_overrides_sample(tmp_path):
    media = _FakeMedia({"voice-sample.wav": b"RIFF"})
    body = (
        "## Substrate\n\n"
        "### Physical · Voice\n\n"
        "```yaml\n"
        "presence: always_on\n"
        "tts_ref: state/tester/media/voice-sample.wav\n"
        "tts_params:\n"
        "  engine: piper\n"
        "  piper_voice: en_US-libritts_r-medium\n"
        "```\n\n"
        "A recorded voice, but piper is forced.\n"
    )
    persona = _write_persona(tmp_path, body)
    spec = resolve_voice_spec(persona, FatigueLevel.RESTED, media=media)
    assert spec.engine == "piper"
    assert spec.piper_voice == "en_US-libritts_r-medium"
    assert spec.sample_audio_b64 is None

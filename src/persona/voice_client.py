"""Async HTTP client for the voice service. Posts each sentence (with a
resolved voice spec) to /synthesize and pushes streamed PCM into an audio sink.
Per-sentence retry posture: a
single sentence's 503 logs a warning and skips that sentence, but the next
sentence still attempts. Connection refused at start raises VoiceUnavailable
so the caller can downgrade to text-only mode for the rest of the turn."""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from typing import Protocol

import httpx

from persona.voice_spec import VoiceSpec

log = logging.getLogger("persona.voice_client")

# Catch ConnectionRefusedError (raised by some sync test fakes), OSError
# (covers low-level socket errors), and httpx.HTTPError (the httpx exception
# hierarchy — including ConnectError, ReadError, etc.).
_TRANSPORT_ERRORS = (ConnectionRefusedError, OSError, httpx.HTTPError)

_RATE_RE = re.compile(r"rate=(\d+)", re.IGNORECASE)
_DEFAULT_PCM_RATE = 24000

# Strip parenthetical stage directions and bracketed asides from text before
# sending it to TTS. Persona memory keeps the original (the asides are real
# context — "(laughs)" tells us something about her). Only the audio render
# drops them, so the listener doesn't hear "open paren laughs close paren".
# We strip both round and square brackets; nested or unmatched brackets are
# left alone (the regex requires a balanced pair).
_STAGE_DIRECTION_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_WHITESPACE_RE = re.compile(r"\s+")


def _parse_pcm_rate(content_type: str | None) -> int:
    """Extract `rate=N` from a Content-Type like `audio/L16; rate=22050`.
    Falls back to 24000 when the header is missing or unparseable so callers
    always have a usable rate."""
    if not content_type:
        return _DEFAULT_PCM_RATE
    m = _RATE_RE.search(content_type)
    if not m:
        return _DEFAULT_PCM_RATE
    try:
        return int(m.group(1))
    except ValueError:
        return _DEFAULT_PCM_RATE


def _strip_stage_directions(text: str) -> str:
    """Remove `(...)` and `[...]` annotations and collapse the resulting
    whitespace. Returns the empty string if the text was nothing but
    annotations."""
    stripped = _STAGE_DIRECTION_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", stripped).strip()


class VoiceUnavailable(RuntimeError):
    """Raised when the voice service cannot be reached at all. Caller should
    print a one-time warning and continue with text-only output."""


class _AudioSink(Protocol):
    async def open(self, samplerate: int | None = None) -> None: ...
    async def write(self, chunk: bytes) -> None: ...
    async def close(self) -> None: ...


class _HttpClient(Protocol):
    def stream(self, method: str, url: str, json: dict): ...


class VoiceClient:
    def __init__(
        self,
        spec: VoiceSpec,
        base_url: str,
        http_client: _HttpClient,
        sink: _AudioSink,
    ):
        self.spec = spec
        self.base_url = base_url.rstrip("/")
        self._http = http_client
        self._sink = sink

    async def speak_stream(self, sentences: AsyncIterator[str]) -> None:
        opened = False
        unreachable_at_start = True

        try:
            async for sentence in sentences:
                speakable = _strip_stage_directions(sentence)
                if not speakable:
                    continue

                try:
                    async with self._http.stream(
                        "POST",
                        f"{self.base_url}/synthesize",
                        json={
                            "text": speakable,
                            "engine": self.spec.engine,
                            "language": self.spec.language,
                            "baseline_speed": self.spec.baseline_speed,
                            "fatigue_level": self.spec.fatigue_level,
                            "piper_voice": self.spec.piper_voice,
                            "sample_audio_b64": self.spec.sample_audio_b64,
                            "response_format": "pcm",
                        },
                    ) as response:
                        unreachable_at_start = False
                        if response.status_code != 200:
                            log.warning(
                                "voice-svc returned %s for sentence %r; skipping",
                                response.status_code,
                                sentence[:40],
                            )
                            continue
                        if not opened:
                            rate = _parse_pcm_rate(response.headers.get("content-type"))
                            await self._sink.open(samplerate=rate)
                            opened = True
                        async for chunk in response.aiter_bytes():
                            await self._sink.write(chunk)
                except _TRANSPORT_ERRORS as e:
                    if unreachable_at_start:
                        raise VoiceUnavailable(str(e)) from e
                    log.warning("voice-svc unreachable mid-turn: %s", e)
                    continue
                except Exception as e:  # noqa: BLE001
                    # Anything that wasn't a transport error — sink/codec/etc.
                    # Skip this sentence and try the next. Sink errors are
                    # local to one sentence and shouldn't kill the turn.
                    log.warning("voice playback error mid-sentence: %s", e)
                    continue
        finally:
            if opened:
                await self._sink.close()

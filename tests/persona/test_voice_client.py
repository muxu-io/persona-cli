import asyncio

import pytest

from persona.voice_client import VoiceClient, VoiceUnavailable
from persona.voice_spec import VoiceSpec

_SPEC = VoiceSpec(
    engine="piper",
    language="en",
    baseline_speed=1.0,
    fatigue_level="rested",
    piper_voice="en_GB-alba-medium",
    sample_audio_b64=None,
)


class _FakeAudioSink:
    def __init__(self):
        self.chunks: list[bytes] = []
        self.opened = 0
        self.closed = 0
        self.opened_at_rate: int | None = None

    async def open(self, samplerate: int | None = None):
        self.opened += 1
        self.opened_at_rate = samplerate

    async def write(self, chunk: bytes):
        self.chunks.append(chunk)

    async def close(self):
        self.closed += 1


class _FakeStreamCtx:
    def __init__(self, status_code, chunks, headers=None):
        self.status_code = status_code
        self._chunks = chunks
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c


class _FakeHttpClient:
    """Minimal stand-in for httpx.AsyncClient.stream()."""

    def __init__(self, status_code: int = 200, chunks=(b"\x00\x00" * 240,), headers=None):
        self._status_code = status_code
        self._chunks = chunks
        self._headers = headers or {}
        self.calls: list[dict] = []

    def stream(self, method, url, json):
        self.calls.append({"method": method, "url": url, "json": json})
        return _FakeStreamCtx(self._status_code, self._chunks, self._headers)


async def _sentences(items):
    for s in items:
        await asyncio.sleep(0)
        yield s


@pytest.mark.asyncio
async def test_speak_stream_posts_one_request_per_sentence_and_writes_audio():
    sink = _FakeAudioSink()
    http = _FakeHttpClient(status_code=200, chunks=(b"\x00\x00" * 240,))
    client = VoiceClient(spec=_SPEC, base_url="http://x", http_client=http, sink=sink)

    await client.speak_stream(_sentences(["one.", "two.", "three."]))

    assert len(http.calls) == 3
    body = http.calls[0]["json"]
    assert body["text"] == "one."
    assert body["engine"] == "piper"
    assert body["language"] == "en"
    assert body["baseline_speed"] == 1.0
    assert body["fatigue_level"] == "rested"
    assert body["piper_voice"] == "en_GB-alba-medium"
    assert body["sample_audio_b64"] is None
    assert body["response_format"] == "pcm"
    assert "persona_id" not in body
    assert sink.opened == 1
    assert sink.closed == 1
    assert len(sink.chunks) >= 3


@pytest.mark.asyncio
async def test_speak_stream_503_per_sentence_is_logged_then_continues():
    sink = _FakeAudioSink()
    http = _FakeHttpClient(status_code=503, chunks=())
    client = VoiceClient(spec=_SPEC, base_url="http://x", http_client=http, sink=sink)
    await client.speak_stream(_sentences(["one.", "two."]))
    assert len(http.calls) == 2
    assert sink.chunks == []


@pytest.mark.asyncio
async def test_speak_stream_connection_error_at_start_is_voice_unavailable():
    sink = _FakeAudioSink()

    class _BadHttp:
        def stream(self, method, url, json):
            raise ConnectionRefusedError("voice-svc unreachable")

    client = VoiceClient(spec=_SPEC, base_url="http://x", http_client=_BadHttp(), sink=sink)

    with pytest.raises(VoiceUnavailable):
        await client.speak_stream(_sentences(["one.", "two."]))


@pytest.mark.asyncio
async def test_speak_stream_opens_sink_at_rate_from_content_type():
    """The voice client should parse the engine's sample rate from
    Content-Type and open the sink at that rate. Required for Piper (22050)
    where the CLI default of 24000 would chipmunk the playback."""
    sink = _FakeAudioSink()
    http = _FakeHttpClient(
        status_code=200,
        chunks=(b"\x00\x00" * 240,),
        headers={"content-type": "audio/L16; rate=22050"},
    )
    client = VoiceClient(spec=_SPEC, base_url="http://x", http_client=http, sink=sink)

    await client.speak_stream(_sentences(["one."]))

    assert sink.opened_at_rate == 22050


@pytest.mark.asyncio
async def test_speak_stream_strips_stage_directions_before_synthesizing():
    """LLMs sometimes emit roleplay annotations like `(laughs)` or `[shrugs]`.
    The voice path must strip these before posting to TTS — the audience
    shouldn't hear "open paren laughs close paren". Persona memory still gets
    the full original text upstream of voice."""
    sink = _FakeAudioSink()
    http = _FakeHttpClient(status_code=200, chunks=(b"\x00\x00" * 240,))
    client = VoiceClient(spec=_SPEC, base_url="http://x", http_client=http, sink=sink)

    await client.speak_stream(
        _sentences(
            [
                "(laughs) Oh hi there.",
                "[looks up] How are you?",
                "(shrugs)",  # annotation-only sentence — should be skipped entirely
            ]
        )
    )

    posted_texts = [c["json"]["text"] for c in http.calls]
    assert posted_texts == ["Oh hi there.", "How are you?"]


@pytest.mark.asyncio
async def test_speak_stream_skips_sentence_on_sink_error_and_continues():
    """A sink error on one sentence (e.g., codec hiccup) must not kill the
    rest of the turn. We log and move on to the next sentence."""

    class _FlakyOpenSink(_FakeAudioSink):
        async def open(self, samplerate=None):
            raise ValueError("simulated sink failure")

    sink = _FlakyOpenSink()
    http = _FakeHttpClient(status_code=200, chunks=(b"\x00\x00" * 240,))
    client = VoiceClient(spec=_SPEC, base_url="http://x", http_client=http, sink=sink)

    # Should not raise — the loop should log and continue past the sink error.
    await client.speak_stream(_sentences(["one.", "two.", "three."]))
    assert len(http.calls) == 3


@pytest.mark.asyncio
async def test_speak_stream_falls_back_to_default_rate_when_header_missing():
    """If voice-svc forgets to include a rate, the client falls back to 24000
    rather than crashing. Defensive — voice-svc shouldn't omit it but the
    client should not be brittle."""
    sink = _FakeAudioSink()
    http = _FakeHttpClient(
        status_code=200,
        chunks=(b"\x00\x00" * 240,),
        headers={"content-type": "application/octet-stream"},
    )
    client = VoiceClient(spec=_SPEC, base_url="http://x", http_client=http, sink=sink)

    await client.speak_stream(_sentences(["one."]))

    assert sink.opened_at_rate == 24000

import pytest


@pytest.mark.asyncio
async def test_sounddevice_sink_open_write_close_lifecycle(monkeypatch):
    """The sink should open a stream once, write each chunk, and close once.
    No real audio device is touched in tests — sounddevice is monkey-patched."""
    from persona import audio_sink

    events: list[tuple[str, ...]] = []

    class FakeStream:
        def __init__(self, **kwargs):
            events.append(("init", str(kwargs.get("samplerate")), str(kwargs.get("channels"))))

        def start(self):
            events.append(("start",))

        def write(self, data):
            events.append(("write", str(len(data))))

        def stop(self):
            events.append(("stop",))

        def close(self):
            events.append(("close",))

    fake_sd = type("FakeSD", (), {"RawOutputStream": FakeStream})
    monkeypatch.setattr(audio_sink, "sd", fake_sd)

    sink = audio_sink.SounddeviceSink(samplerate=24000, channels=1)
    await sink.open()
    await sink.write(b"\x00\x00" * 240)
    await sink.write(b"\x00\x00" * 240)
    await sink.close()

    kinds = [e[0] for e in events]
    assert kinds == ["init", "start", "write", "write", "stop", "close"]


@pytest.mark.asyncio
async def test_sounddevice_sink_handles_odd_byte_chunks(monkeypatch):
    """httpx may split an HTTP body at arbitrary byte boundaries, so chunks
    arriving at the sink can have odd byte counts. The sink must buffer
    sub-sample bytes across calls; otherwise sounddevice raises
    `len(data) not divisible by samplesize` and the voice task dies."""
    from persona import audio_sink

    written: list[bytes] = []

    class FakeStream:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def write(self, data):
            assert len(data) % 2 == 0, f"sink wrote odd-byte chunk: {len(data)}"
            written.append(bytes(data))

        def stop(self):
            pass

        def close(self):
            pass

    fake_sd = type("FakeSD", (), {"RawOutputStream": FakeStream})
    monkeypatch.setattr(audio_sink, "sd", fake_sd)

    sink = audio_sink.SounddeviceSink(samplerate=22050, channels=1, dtype="int16")
    await sink.open()
    # Three odd-byte chunks: 5 + 3 + 4 = 12 bytes total. Should land as
    # whole-sample writes only (4, 4, 4) or similar even chunks.
    await sink.write(b"\x01\x02\x03\x04\x05")
    await sink.write(b"\x06\x07\x08")
    await sink.write(b"\x09\x0a\x0b\x0c")
    await sink.close()

    total_written = sum(len(w) for w in written)
    # 12 bytes total, all 6 samples should have been written. The exact
    # chunking is sink-internal; only the total + alignment matter.
    assert total_written == 12
    for w in written:
        assert len(w) % 2 == 0


@pytest.mark.asyncio
async def test_sounddevice_sink_open_samplerate_override(monkeypatch):
    """`open(samplerate=N)` overrides the constructor default — used when the
    real engine's rate is learned from the synth response Content-Type."""
    from persona import audio_sink

    captured: dict = {}

    class FakeStream:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            pass

        def write(self, data):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    fake_sd = type("FakeSD", (), {"RawOutputStream": FakeStream})
    monkeypatch.setattr(audio_sink, "sd", fake_sd)

    sink = audio_sink.SounddeviceSink(samplerate=24000, channels=1)
    await sink.open(samplerate=22050)
    await sink.close()

    assert captured["samplerate"] == 22050

"""Sounddevice-backed audio sink for the CLI surface. Wraps sounddevice's
RawOutputStream behind an async-friendly interface so the voice client can
push chunks via await."""

from __future__ import annotations

import asyncio

try:
    import sounddevice as sd  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    sd = None  # type: ignore[assignment]


_DTYPE_BYTES = {"int16": 2, "int32": 4, "float32": 4, "int8": 1}


class SounddeviceSink:
    def __init__(self, samplerate: int = 24000, channels: int = 1, dtype: str = "int16"):
        self._samplerate = samplerate
        self._channels = channels
        self._dtype = dtype
        self._stream = None
        # Carry across writes to handle upstream chunks that don't fall on
        # sample boundaries (e.g. httpx splitting an HTTP response). Without
        # this, sounddevice raises `len(data) not divisible by samplesize`.
        self._carry = b""

    @property
    def _sample_size(self) -> int:
        return _DTYPE_BYTES[self._dtype] * self._channels

    async def open(self, samplerate: int | None = None) -> None:
        """Open the audio stream. `samplerate` overrides the constructor
        default — use it when the engine's actual rate is known (e.g. from
        a Content-Type header on the synth response). Falls back to the
        constructor default when not provided."""
        if sd is None:
            raise RuntimeError("sounddevice not available in this environment")
        rate = samplerate if samplerate else self._samplerate
        self._stream = sd.RawOutputStream(
            samplerate=rate,
            channels=self._channels,
            dtype=self._dtype,
        )
        await asyncio.to_thread(self._stream.start)

    async def write(self, chunk: bytes) -> None:
        if self._stream is None:
            return
        # Align to sample boundaries: stash any trailing odd bytes for the
        # next call. Sounddevice rejects partial samples.
        data = self._carry + chunk
        ss = self._sample_size
        whole = (len(data) // ss) * ss
        if whole == 0:
            self._carry = data
            return
        self._carry = data[whole:]
        await asyncio.to_thread(self._stream.write, data[:whole])

    async def close(self) -> None:
        if self._stream is None:
            return
        # Drop any sub-sample carry — at most a few bytes, inaudible.
        self._carry = b""
        await asyncio.to_thread(self._stream.stop)
        await asyncio.to_thread(self._stream.close)
        self._stream = None

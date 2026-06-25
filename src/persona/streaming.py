"""Async helper: tee a token stream into (sentence_iterator, full_text_future).
Both consume the upstream exactly once."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from persona.sentence_split import StreamingSentenceSplitter


def tee_into_sentences(
    token_stream: AsyncIterator[str],
) -> tuple[AsyncIterator[str], asyncio.Future[str]]:
    """Returns (sentence_iterator, full_text_future). Both consume token_stream
    exactly once. The iterator yields sentences progressively as the splitter
    disambiguates them; the future resolves with the full concatenated text
    when the upstream ends."""
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    full_text_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

    async def producer() -> None:
        splitter = StreamingSentenceSplitter()
        accumulator: list[str] = []
        try:
            async for chunk in token_stream:
                accumulator.append(chunk)
                for sentence in splitter.feed(chunk):
                    await queue.put(sentence)
            for sentence in splitter.flush():
                await queue.put(sentence)
        except Exception as exc:  # noqa: BLE001
            if not full_text_future.done():
                full_text_future.set_exception(exc)
            await queue.put(None)
            return
        if not full_text_future.done():
            full_text_future.set_result("".join(accumulator))
        await queue.put(None)

    asyncio.create_task(producer())

    async def iterator() -> AsyncIterator[str]:
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item

    return iterator(), full_text_future

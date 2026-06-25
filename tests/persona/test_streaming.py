import asyncio

import pytest

from persona.streaming import tee_into_sentences


async def _gen(chunks):
    for c in chunks:
        await asyncio.sleep(0)
        yield c


@pytest.mark.asyncio
async def test_tee_emits_sentences_and_full_text():
    chunks = ["Hello there", ". How are ", "you? Doing fine."]
    sentence_iter, full_text_future = tee_into_sentences(_gen(chunks))

    sentences = [s async for s in sentence_iter]
    assert sentences == ["Hello there.", "How are you?", "Doing fine."]

    full = await full_text_future
    assert full == "Hello there. How are you? Doing fine."


@pytest.mark.asyncio
async def test_tee_empty_stream():
    sentence_iter, full_text_future = tee_into_sentences(_gen([]))
    sentences = [s async for s in sentence_iter]
    assert sentences == []
    assert (await full_text_future) == ""


@pytest.mark.asyncio
async def test_tee_full_text_resolves_even_when_sentences_iter_consumed_first():
    chunks = ["One. Two. Three"]
    sentence_iter, full_text_future = tee_into_sentences(_gen(chunks))
    sentences = [s async for s in sentence_iter]
    assert sentences == ["One.", "Two.", "Three"]
    full = await full_text_future
    assert full == "One. Two. Three"


@pytest.mark.asyncio
async def test_tee_sentences_yielded_progressively_not_all_at_end():
    """Important property: a slow upstream should produce sentences as they
    complete, not buffer everything until the stream ends."""
    received_at: list[float] = []
    loop = asyncio.get_running_loop()

    async def slow_chunks():
        yield "First sentence. "
        await asyncio.sleep(0.05)
        yield "Second sentence. "
        await asyncio.sleep(0.05)
        yield "Third sentence."

    sentence_iter, _ = tee_into_sentences(slow_chunks())
    async for _ in sentence_iter:
        received_at.append(loop.time())

    assert len(received_at) == 3
    assert received_at[1] - received_at[0] > 0.02
    assert received_at[2] - received_at[1] > 0.02

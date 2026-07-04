"""Single-pass generation loop. Calls Ollama, persists turn-pair, slides working memory."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol

from persona_core.embedding import EmbeddingClient
from persona_core.qdrant_store import QdrantStore
from persona_core.records import EpisodicRecord, RecordType
from persona_core.schema import Dimension, Persona

from persona.context import assemble_context
from persona.fatigue import FatigueLevel
from persona.retrieval import RetrievedItem
from persona.streaming import tee_into_sentences

if TYPE_CHECKING:
    from persona.scenario import Scenario

DEFAULT_NUM_PREDICT = 1500
DEFAULT_NUM_CTX = 16384


class _GenTransport(Protocol):
    def generate(
        self,
        model: str,
        prompt: str,
        options: dict | None = None,
        stream: bool = False,
        think: bool = False,
    ) -> dict[str, Any]: ...


class GenerationClient:
    def __init__(
        self,
        model: str,
        transport: _GenTransport | None = None,
        host: str | None = None,
    ):
        self.model = model
        if transport is None:
            import ollama

            self._transport = ollama.Client(host=host) if host else ollama.Client()
        else:
            self._transport = transport

    def generate(
        self,
        prompt: str,
        num_ctx: int = DEFAULT_NUM_CTX,
        num_predict: int = DEFAULT_NUM_PREDICT,
        think: bool = False,
    ) -> str:
        result = self._transport.generate(
            model=self.model,
            prompt=prompt,
            options={"num_ctx": num_ctx, "num_predict": num_predict},
            think=think,
        )
        return result["response"]

    async def generate_stream(
        self,
        prompt: str,
        num_ctx: int = DEFAULT_NUM_CTX,
        num_predict: int = DEFAULT_NUM_PREDICT,
        think: bool = False,
    ) -> AsyncIterator[str]:
        """Stream chunks of the model's response. Wraps the (sync) Ollama
        streaming iterator in an async-friendly shape using a thread.

        Empty chunks are filtered; the iterator naturally terminates when the
        underlying generator is exhausted. Reasoning (when ``think`` is set)
        arrives in a separate ``thinking`` field, which this loop never reads —
        so it is discarded and never streamed to voice."""
        sync_iter = self._transport.generate(
            model=self.model,
            prompt=prompt,
            options={"num_ctx": num_ctx, "num_predict": num_predict},
            stream=True,
            think=think,
        )
        sentinel = object()

        def _next() -> object:
            return next(sync_iter, sentinel)

        while True:
            item = await asyncio.to_thread(_next)
            if item is sentinel:
                return
            # Ollama 0.6+ yields pydantic GenerateResponse objects from
            # streaming; older versions yield plain dicts. Test stubs use
            # dicts. Handle both via attribute-then-dict access.
            text = getattr(item, "response", None)
            if text is None and isinstance(item, dict):
                text = item.get("response", "")
            if text:
                yield text


def run_turn(
    persona: Persona,
    user_message: str,
    store: QdrantStore,
    embedder: EmbeddingClient,
    gen_client: GenerationClient,
    session_id: str,
    working_memory: list[EpisodicRecord],
    working_memory_size: int = 8,
    triggered_dims: list[Dimension] | None = None,
    retrieved: list[RetrievedItem] | None = None,
    fatigue_level: FatigueLevel = FatigueLevel.RESTED,
    addendum_enabled: bool = False,
    scenario: Scenario | None = None,
    scenario_id: str | None = None,
    runtime_state=None,
    think: bool = False,
) -> str:
    """One conversational turn. Returns the assistant response and mutates state."""
    ctx = assemble_context(
        persona=persona,
        user_message=user_message,
        triggered_dims=triggered_dims or [],
        working_memory=working_memory,
        retrieved=retrieved or [],
        fatigue_level=fatigue_level,
        addendum_enabled=addendum_enabled,
        scenario=scenario,
    )

    response = gen_client.generate(ctx.text, think=think)

    record = EpisodicRecord(
        type=RecordType.TURN_PAIR,
        session_id=session_id,
        content={"user": user_message, "assistant": response},
        initial_salience=0.4,  # design doc §9.2 default
        scenario_id=(
            scenario_id if scenario_id is not None else (scenario.scenario_id if scenario else None)
        ),
    )
    record.embedding = embedder.embed(_embed_text(record))
    store.write(record)

    working_memory.append(record)
    while len(working_memory) > working_memory_size:
        working_memory.pop(0)

    # Hot loop is store-svc-free: runtime state is persisted once at session end
    # by the CLI, not per turn. Fatigue for each turn is derived from the live
    # Qdrant count by the caller before the turn.

    return response


async def run_turn_async(
    persona: Persona,
    user_message: str,
    store: QdrantStore,
    embedder: EmbeddingClient,
    gen_client: GenerationClient,
    session_id: str,
    working_memory: list[EpisodicRecord],
    speak,  # async callable: (AsyncIterator[str]) -> None
    working_memory_size: int = 8,
    triggered_dims: list[Dimension] | None = None,
    retrieved: list[RetrievedItem] | None = None,
    fatigue_level: FatigueLevel = FatigueLevel.RESTED,
    addendum_enabled: bool = False,
    scenario: Scenario | None = None,
    scenario_id: str | None = None,
    runtime_state=None,
    think: bool = False,
) -> str:
    """Streaming variant of run_turn: tees Ollama output into sentences (for
    voice) and full text (for qdrant). Voice failures NEVER block the qdrant
    write — `speak` may raise; the qdrant path still runs."""
    ctx = assemble_context(
        persona=persona,
        user_message=user_message,
        triggered_dims=triggered_dims or [],
        working_memory=working_memory,
        retrieved=retrieved or [],
        fatigue_level=fatigue_level,
        addendum_enabled=addendum_enabled,
        scenario=scenario,
    )

    token_stream = gen_client.generate_stream(ctx.text, think=think)
    sentence_iter, full_text_future = tee_into_sentences(token_stream)

    voice_task = asyncio.create_task(speak(sentence_iter))
    text = await full_text_future

    record = EpisodicRecord(
        type=RecordType.TURN_PAIR,
        session_id=session_id,
        content={"user": user_message, "assistant": text},
        initial_salience=0.4,
        scenario_id=(
            scenario_id if scenario_id is not None else (scenario.scenario_id if scenario else None)
        ),
    )
    record.embedding = embedder.embed(_embed_text(record))
    store.write(record)

    working_memory.append(record)
    while len(working_memory) > working_memory_size:
        working_memory.pop(0)

    # Hot loop is store-svc-free: runtime state is persisted at session end, not
    # per turn (see run_turn).

    # Wait for voice playback to complete; swallow voice errors so they
    # never fail the turn.
    try:
        await voice_task
    except Exception as e:  # noqa: BLE001
        import logging

        logging.getLogger("persona.run_turn_async").warning("voice failed: %s", e)

    return text


def _embed_text(record: EpisodicRecord) -> str:
    """Text used for the record's embedding. Concatenates user + assistant for turn-pairs."""
    if record.type == RecordType.TURN_PAIR:
        return f"{record.content['user']}\n{record.content['assistant']}"
    return record.content.get("event", str(record.content))

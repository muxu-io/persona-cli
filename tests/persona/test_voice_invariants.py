"""Spec § 10: voice must not perturb the rest of the runtime. A 30-turn
session with voice on should produce structurally identical qdrant records and
working memory to a 30-turn session with voice off. Runtime state is persisted
only at session end (not per turn), so it is not exercised here."""

import pytest
from persona_core.generation import GenerationClient, run_turn, run_turn_async
from persona_core.qdrant_store import QdrantStore
from persona_core.records import EpisodicRecord
from persona_core.schema import Persona

N_TURNS = 30


class _StubGen:
    def generate(self, model, prompt, options=None, stream=False, think=False):
        text = "Hello back. This is a deterministic reply."
        if stream:
            return iter([{"response": text}, {"response": ""}])
        return {"response": text}


class _StubEmbed:
    def embed(self, text: str):
        return [float(len(text) % 7)] * 8


async def _silent_speak(sentences):
    async for _ in sentences:
        pass


@pytest.mark.asyncio
async def test_voice_on_vs_off_produce_structurally_identical_state():
    persona = Persona(
        persona_id="alice",
        spec_version=1,
        identity={"name": "Alice"},
        substrate={},
        self_concept={},
    )

    # Voice OFF run
    store_off = QdrantStore.in_memory(
        collection="persona_memory", vector_size=8, persona_id="alice"
    )
    store_off.ensure_collection()
    wm_off: list[EpisodicRecord] = []
    gen_off = GenerationClient(model="x", transport=_StubGen())
    for i in range(N_TURNS):
        run_turn(
            persona=persona,
            user_message=f"message {i}",
            store=store_off,
            embedder=_StubEmbed(),  # type: ignore[arg-type]
            gen_client=gen_off,
            session_id="s",
            working_memory=wm_off,
            runtime_state=None,
        )

    # Voice ON run (with silent voice — equivalent to playback success)
    store_on = QdrantStore.in_memory(collection="persona_memory", vector_size=8, persona_id="alice")
    store_on.ensure_collection()
    wm_on: list[EpisodicRecord] = []
    gen_on = GenerationClient(model="x", transport=_StubGen())
    for i in range(N_TURNS):
        await run_turn_async(
            persona=persona,
            user_message=f"message {i}",
            store=store_on,
            embedder=_StubEmbed(),  # type: ignore[arg-type]
            gen_client=gen_on,
            session_id="s",
            working_memory=wm_on,
            speak=_silent_speak,
            runtime_state=None,
        )

    # Counts
    assert store_off.count() == store_on.count() == N_TURNS
    assert store_off.count_unconsolidated() == store_on.count_unconsolidated()
    # Working memory
    assert len(wm_off) == len(wm_on)
    # Same record types and same user/assistant content per index
    for a, b in zip(wm_off, wm_on, strict=True):
        assert a.type == b.type
        assert a.content == b.content

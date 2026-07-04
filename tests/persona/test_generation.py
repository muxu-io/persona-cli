from datetime import date as _date

from persona_core.embedding import EmbeddingClient
from persona_core.qdrant_store import QdrantStore
from persona_core.schema import Dimension, Persona, Presence

from persona.generation import GenerationClient, run_turn
from persona.scenario import Scenario as _Scenario


class _FakeOllamaGen:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, model, prompt, options=None, think=False):
        self.calls.append((model, prompt, options))
        return {"response": self.reply, "done": True}


class _FakeOllamaEmb:
    def embeddings(self, model, prompt):
        return {"embedding": [1.0, 0.0, 0.0, 0.0]}


def _persona():
    return Persona(
        persona_id="ada",
        spec_version=1,
        identity={"name": "Ada"},
        substrate={
            "cog": Dimension(
                name="cog",
                presence=Presence.ALWAYS_ON,
                prose="x",
                structured={},
            )
        },
        self_concept={},
        traumas=[],
    )


def test_run_turn_returns_response_and_writes_record():
    store = QdrantStore.in_memory(collection="t", vector_size=4, persona_id="t")
    store.ensure_collection()
    embedder = EmbeddingClient(model="nomic", transport=_FakeOllamaEmb())
    gen = GenerationClient(model="llama", transport=_FakeOllamaGen(reply="quietly: sure."))
    persona = _persona()
    response = run_turn(
        persona=persona,
        user_message="alright?",
        store=store,
        embedder=embedder,
        gen_client=gen,
        session_id="s-1",
        working_memory=[],
    )
    assert response == "quietly: sure."
    assert store.count() == 1


def test_run_turn_appends_to_working_memory():
    store = QdrantStore.in_memory(collection="t", vector_size=4, persona_id="t")
    store.ensure_collection()
    embedder = EmbeddingClient(model="nomic", transport=_FakeOllamaEmb())
    gen = GenerationClient(model="llama", transport=_FakeOllamaGen(reply="ok."))
    persona = _persona()
    wm = []
    run_turn(persona, "first?", store, embedder, gen, "s-1", wm)
    run_turn(persona, "second?", store, embedder, gen, "s-1", wm)
    assert len(wm) == 2
    assert wm[0].content["user"] == "first?"
    assert wm[1].content["user"] == "second?"
    assert store.count() == 2


def test_run_turn_slides_window_at_max():
    store = QdrantStore.in_memory(collection="t", vector_size=4, persona_id="t")
    store.ensure_collection()
    embedder = EmbeddingClient(model="nomic", transport=_FakeOllamaEmb())
    gen = GenerationClient(model="llama", transport=_FakeOllamaGen(reply="ok."))
    persona = _persona()
    wm = []
    for i in range(10):
        run_turn(persona, f"u-{i}", store, embedder, gen, "s-1", wm, working_memory_size=8)
    assert len(wm) == 8
    assert wm[0].content["user"] == "u-2"  # oldest two slid out
    assert wm[-1].content["user"] == "u-9"


def _scenario_for_test():
    return _Scenario(
        scenario_id="shift-cut-corridor",
        persona_id="ada",
        spec_version=1,
        title="t",
        created=_date(2026, 5, 1),
        scene="A scene.",
        interlocutor=None,
        interlocutor_name=None,
        interlocutor_relation=None,
    )


def test_run_turn_tags_record_with_scenario_id_when_scenario_passed():
    store = QdrantStore.in_memory(collection="t", vector_size=4, persona_id="t")
    store.ensure_collection()
    embedder = EmbeddingClient(model="nomic", transport=_FakeOllamaEmb())
    gen = GenerationClient(model="llama", transport=_FakeOllamaGen(reply="ok."))
    persona = _persona()
    wm = []
    run_turn(
        persona=persona,
        user_message="hi",
        store=store,
        embedder=embedder,
        gen_client=gen,
        session_id="s-1",
        working_memory=wm,
        scenario=_scenario_for_test(),
    )
    assert len(wm) == 1
    assert wm[0].scenario_id == "shift-cut-corridor"


def test_run_turn_no_scenario_id_when_scenario_absent():
    store = QdrantStore.in_memory(collection="t", vector_size=4, persona_id="t")
    store.ensure_collection()
    embedder = EmbeddingClient(model="nomic", transport=_FakeOllamaEmb())
    gen = GenerationClient(model="llama", transport=_FakeOllamaGen(reply="ok."))
    persona = _persona()
    wm = []
    run_turn(
        persona=persona,
        user_message="hi",
        store=store,
        embedder=embedder,
        gen_client=gen,
        session_id="s-1",
        working_memory=wm,
    )
    assert wm[0].scenario_id is None


def test_run_turn_explicit_scenario_id_overrides_object():
    """If a caller passes both, the explicit string wins. Mainly for tests / future flexibility."""
    store = QdrantStore.in_memory(collection="t", vector_size=4, persona_id="t")
    store.ensure_collection()
    embedder = EmbeddingClient(model="nomic", transport=_FakeOllamaEmb())
    gen = GenerationClient(model="llama", transport=_FakeOllamaGen(reply="ok."))
    persona = _persona()
    wm = []
    run_turn(
        persona=persona,
        user_message="hi",
        store=store,
        embedder=embedder,
        gen_client=gen,
        session_id="s-1",
        working_memory=wm,
        scenario=_scenario_for_test(),
        scenario_id="explicit-override",
    )
    assert wm[0].scenario_id == "explicit-override"


def test_run_turn_does_not_persist_runtime_state_per_turn():
    """The hot loop is store-svc-free: run_turn must NOT call runtime_state.save()
    (runtime state is persisted once at session end by the CLI). It still writes
    the turn-pair to Qdrant."""
    from persona_core.schema import Persona

    from persona.generation import GenerationClient, run_turn

    persona = Persona(
        persona_id="alice",
        spec_version=1,
        identity={"name": "Alice"},
        substrate={},
        self_concept={},
    )
    store = QdrantStore.in_memory(collection="persona_memory", vector_size=8, persona_id="alice")
    store.ensure_collection()

    class StubGen:
        def generate(self, model, prompt, options=None, think=False):
            return {"response": "hello back"}

    gen = GenerationClient(model="x", transport=StubGen())

    class StubEmbed:
        def embed(self, text):
            return [0.0] * 8

    class _RecordingRuntimeState:
        saved = False

        def save(self):
            self._saved = True

    rs = _RecordingRuntimeState()

    run_turn(
        persona=persona,
        user_message="hi",
        store=store,
        embedder=StubEmbed(),  # type: ignore[arg-type]
        gen_client=gen,
        session_id="s",
        working_memory=[],
        runtime_state=rs,
    )

    assert getattr(rs, "_saved", False) is False  # never persisted per turn
    assert store.count_unconsolidated() == 1  # turn-pair still written to Qdrant


def test_generate_stream_yields_chunks():
    import asyncio

    from persona.generation import GenerationClient

    class StubStreamingTransport:
        def generate(self, model, prompt, options=None, stream=False, think=False):
            if stream:
                return iter(
                    [
                        {"response": "Hello "},
                        {"response": "there."},
                        {"response": ""},
                    ]
                )
            return {"response": "Hello there."}

    gen = GenerationClient(model="x", transport=StubStreamingTransport())

    async def collect():
        out = []
        async for chunk in gen.generate_stream("prompt"):
            out.append(chunk)
        return out

    chunks = asyncio.run(collect())
    assert chunks == ["Hello ", "there."]


def test_generate_stream_handles_ollama_response_objects():
    """Regression: Ollama 0.6+ yields pydantic GenerateResponse objects from
    streaming, not dicts. The earlier `isinstance(item, dict)` check filtered
    those out, producing an empty stream and a silent chat session."""
    import asyncio

    from persona.generation import GenerationClient

    class _FakeOllamaResponse:
        """Stand-in for ollama.GenerateResponse — has `.response` attribute,
        is NOT a dict."""

        def __init__(self, response: str):
            self.response = response

    class StubStreamingTransport:
        def generate(self, model, prompt, options=None, stream=False, think=False):
            assert stream is True
            return iter(
                [
                    _FakeOllamaResponse("Hi "),
                    _FakeOllamaResponse("there."),
                    _FakeOllamaResponse(""),  # empty chunk filtered
                ]
            )

    gen = GenerationClient(model="x", transport=StubStreamingTransport())

    async def collect():
        return [c async for c in gen.generate_stream("prompt")]

    chunks = asyncio.run(collect())
    assert chunks == ["Hi ", "there."]


def test_generate_forwards_think_flag_to_transport():
    class _RecordingTransport:
        def __init__(self):
            self.think = "unset"

        def generate(self, model, prompt, options=None, think=False):
            self.think = think
            return {"response": "ok"}

    t = _RecordingTransport()
    gen = GenerationClient(model="x", transport=t)

    gen.generate("prompt")
    assert t.think is False  # default: no reasoning

    gen.generate("prompt", think=True)
    assert t.think is True


def test_generate_stream_forwards_think_flag_to_transport():
    import asyncio

    class _RecordingStreamTransport:
        def __init__(self):
            self.think = "unset"

        def generate(self, model, prompt, options=None, stream=False, think=False):
            self.think = think
            return iter([{"response": "hi"}])

    t = _RecordingStreamTransport()
    gen = GenerationClient(model="x", transport=t)

    async def drain(think):
        return [c async for c in gen.generate_stream("prompt", think=think)]

    asyncio.run(drain(False))
    assert t.think is False

    asyncio.run(drain(True))
    assert t.think is True


def test_generate_stream_discards_reasoning_chunks():
    """The feature's core promise: with think on, Ollama returns reasoning in a
    separate `thinking` field with an empty `response`. Those chunks must never
    reach the stream (and thus never be spoken)."""
    import asyncio

    class _ThinkingChunk:
        def __init__(self, response="", thinking=""):
            self.response = response
            self.thinking = thinking

    class _ReasoningTransport:
        def generate(self, model, prompt, options=None, stream=False, think=False):
            return iter(
                [
                    _ThinkingChunk(thinking="Let me reason... "),
                    _ThinkingChunk(thinking="the answer is 4."),
                    _ThinkingChunk(response="It's four."),
                ]
            )

    gen = GenerationClient(model="x", transport=_ReasoningTransport())

    async def drain():
        return [c async for c in gen.generate_stream("prompt", think=True)]

    assert asyncio.run(drain()) == ["It's four."]


def test_run_turn_forwards_think_to_gen_client():
    store = QdrantStore.in_memory(collection="t", vector_size=4, persona_id="t")
    store.ensure_collection()
    embedder = EmbeddingClient(model="nomic", transport=_FakeOllamaEmb())

    class _RecordingTransport:
        think = "unset"

        def generate(self, model, prompt, options=None, think=False):
            _RecordingTransport.think = think
            return {"response": "ok"}

    gen = GenerationClient(model="llama", transport=_RecordingTransport())
    run_turn(_persona(), "hi", store, embedder, gen, "s-1", [], think=True)
    assert _RecordingTransport.think is True


def test_run_turn_async_streams_to_voice_and_writes_qdrant_independently():
    """Voice client errors must not prevent the qdrant write."""
    import asyncio

    from persona_core.qdrant_store import QdrantStore
    from persona_core.schema import Persona

    from persona.generation import GenerationClient, run_turn_async

    persona = Persona(
        persona_id="alice",
        spec_version=1,
        identity={"name": "Alice"},
        substrate={},
        self_concept={},
    )
    store = QdrantStore.in_memory(collection="alice", vector_size=8, persona_id="alice")
    store.ensure_collection()

    class StubStreamingTransport:
        def generate(self, model, prompt, options=None, stream=False, think=False):
            assert stream is True
            return iter(
                [
                    {"response": "Hi. "},
                    {"response": "How are you?"},
                    {"response": ""},
                ]
            )

    gen = GenerationClient(model="x", transport=StubStreamingTransport())

    class StubEmbed:
        def embed(self, t):
            return [0.0] * 8

    async def explode_voice(sentences):
        # Eagerly raise — simulate voice-svc unreachable at turn start.
        raise RuntimeError("voice down")

    asyncio.run(
        run_turn_async(
            persona=persona,
            user_message="hi",
            store=store,
            embedder=StubEmbed(),  # type: ignore[arg-type]
            gen_client=gen,
            session_id="s",
            working_memory=[],
            speak=explode_voice,
            runtime_state=None,
        )
    )

    # Despite voice failing, the turn-pair should be persisted.
    assert store.count() == 1

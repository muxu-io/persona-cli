"""Smoke-test the CLI voice flag wiring without touching real services.

We don't run the full chat loop here; we just assert that the argparse layer
recognizes the new flag and that PERSONA_VOICE env is honored as a fallback."""


def test_voice_flag_default_on():
    from persona.cli import _voice_enabled

    # No env, no flag → on (spec § 5.2 default-on).
    assert _voice_enabled(flag_voice=None, env=None) is True


def test_voice_flag_explicit_off_via_flag():
    from persona.cli import _voice_enabled

    assert _voice_enabled(flag_voice=False, env=None) is False


def test_voice_flag_off_via_env():
    from persona.cli import _voice_enabled

    assert _voice_enabled(flag_voice=None, env="off") is False
    assert _voice_enabled(flag_voice=None, env="OFF") is False


def test_voice_flag_explicit_flag_overrides_env():
    from persona.cli import _voice_enabled

    # flag wins over env
    assert _voice_enabled(flag_voice=True, env="off") is True
    assert _voice_enabled(flag_voice=False, env="on") is False


def test_run_voice_turn_resolves_spec(monkeypatch):
    """_run_voice_turn must resolve a VoiceSpec at the call site and construct
    VoiceClient with spec=..., NOT persona_id=...

    We stub the audio sink, httpx client, VoiceClient and run_turn_async so the
    only real work is resolve_voice_spec running against a parsed persona. The
    minimal fixture has no physical.voice dimension, so it resolves to a piper
    fallback — which is exactly the documented behavior."""
    import asyncio
    from pathlib import Path

    from persona_core.parser import parse_persona_file

    import persona.cli as cli
    from persona.fatigue import FatigueLevel
    from persona.voice_spec import VoiceSpec

    persona = parse_persona_file(Path(__file__).parent.parent / "fixtures" / "minimal_persona.md")

    captured: dict = {}

    class _StubSink:
        def __init__(self, *a, **k):
            pass

    class _StubAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return object()

        async def __aexit__(self, *a):
            return False

    class _StubVoiceClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def speak_stream(self, sentences):  # pragma: no cover - unused
            return None

    async def _stub_run_turn_async(**kwargs):
        # Confirm the spec flowed into a real VoiceClient before generation.
        return "hello"

    # Patch the names resolved inside the function's local imports.
    monkeypatch.setattr("persona.audio_sink.SounddeviceSink", _StubSink)
    monkeypatch.setattr("httpx.AsyncClient", _StubAsyncClient)
    monkeypatch.setattr("persona.voice_client.VoiceClient", _StubVoiceClient)
    monkeypatch.setattr("persona.generation.run_turn_async", _stub_run_turn_async)

    response, still_on = asyncio.run(
        cli._run_voice_turn(
            voice_base_url="http://127.0.0.1:7000",
            persona=persona,
            user_message="hi",
            store=None,
            media=None,
            embedder=None,
            gen_client=None,
            session_id="sid",
            working_memory=[],
            triggered_dims=[],
            retrieved=[],
            fatigue_level=FatigueLevel.RESTED,
            addendum_enabled=False,
            scenario=None,
            runtime_state=None,
        )
    )

    assert response == "hello"
    assert still_on is True
    # Spec-based construction: a VoiceSpec was passed, no persona_id leaked.
    assert "persona_id" not in captured
    assert isinstance(captured.get("spec"), VoiceSpec)
    # No voice dimension on the fixture → piper fallback.
    assert captured["spec"].engine == "piper"
    assert captured["spec"].fatigue_level == str(FatigueLevel.RESTED)


def test_run_voice_turn_forwards_think(monkeypatch):
    """A `/think` turn must pass think=True into run_turn_async on the voice path."""
    import asyncio
    from pathlib import Path

    from persona_core.parser import parse_persona_file

    import persona.cli as cli
    from persona.fatigue import FatigueLevel

    persona = parse_persona_file(Path(__file__).parent.parent / "fixtures" / "minimal_persona.md")

    seen: dict = {}

    class _StubSink:
        def __init__(self, *a, **k):
            pass

    class _StubAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return object()

        async def __aexit__(self, *a):
            return False

    class _StubVoiceClient:
        def __init__(self, **kwargs):
            pass

        async def speak_stream(self, sentences):  # pragma: no cover - unused
            return None

    async def _stub_run_turn_async(**kwargs):
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr("persona.audio_sink.SounddeviceSink", _StubSink)
    monkeypatch.setattr("httpx.AsyncClient", _StubAsyncClient)
    monkeypatch.setattr("persona.voice_client.VoiceClient", _StubVoiceClient)
    monkeypatch.setattr("persona.generation.run_turn_async", _stub_run_turn_async)

    asyncio.run(
        cli._run_voice_turn(
            voice_base_url="http://127.0.0.1:7000",
            persona=persona,
            user_message="hi",
            store=None,
            media=None,
            embedder=None,
            gen_client=None,
            session_id="sid",
            working_memory=[],
            triggered_dims=[],
            retrieved=[],
            fatigue_level=FatigueLevel.RESTED,
            addendum_enabled=False,
            scenario=None,
            runtime_state=None,
            think=True,
        )
    )

    assert seen.get("think") is True

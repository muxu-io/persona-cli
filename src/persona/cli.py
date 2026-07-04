"""Persona CLI. `python -m persona chat <persona-id>` starts a session.

The runtime reads definitions, runtime state, and scenarios from the persona-store
cold path (StoreClient) and talks to Qdrant directly on the hot loop. There are no
local persona/state files and no per-persona snapshots.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import UTC, date, datetime

from persona_core.bootstrap import PersonaNotInStore, load_persona
from persona_core.embedding import EmbeddingClient
from persona_core.qdrant_store import QdrantStore
from persona_core.store_client import StoreClient
from persona_core.triggering import TriggerConfig, select_triggered

from persona.fatigue import FatigueThresholds, derive_fatigue_level
from persona.generation import GenerationClient, run_turn
from persona.retrieval import RetrievalConfig, retrieve_relevant
from persona.scenario import Scenario
from persona.scoring import ScoreWeights


def _store_client() -> StoreClient:
    return StoreClient(os.environ.get("PERSONA_STORE_URL", "http://localhost:7600"))


def parse_think_prefix(line: str) -> tuple[bool, str]:
    """Detect a leading ``/think`` command on a prompt line.

    ``"/think msg"`` -> ``(True, "msg")``; anything else -> ``(False, line)``.
    Only ``/think`` followed by whitespace or end-of-line triggers, so
    ``"/thinking ..."`` is treated as an ordinary message.
    """
    if line == "/think":
        return True, ""
    if line.startswith("/think") and line[len("/think")].isspace():
        return True, line[len("/think") :].strip()
    return False, line


def _voice_enabled(flag_voice: bool | None, env: str | None) -> bool:
    """Resolve the voice toggle. Flag wins over env. Default is on."""
    if flag_voice is not None:
        return flag_voice
    if env is not None and env.strip().lower() in {"off", "false", "0", "no"}:
        return False
    return True


def _scenario_from_store(client: StoreClient, persona_id: str, scenario_id: str) -> Scenario | None:
    """Build a Scenario from the store's scenario body {scene, interlocutor}."""
    data = client.get_scenario(persona_id, scenario_id)
    if data is None:
        return None
    body = data.get("body", {}) or {}
    interlocutor = body.get("interlocutor") or {}
    return Scenario(
        scenario_id=scenario_id,
        persona_id=persona_id,
        spec_version=1,
        title=data.get("title", scenario_id),
        created=date.today(),
        scene=body.get("scene", ""),
        interlocutor=interlocutor.get("prose"),
        interlocutor_name=interlocutor.get("name"),
        interlocutor_relation=interlocutor.get("relation"),
    )


async def _run_voice_turn(
    *,
    voice_base_url: str,
    persona,
    user_message: str,
    store,
    media,
    embedder,
    gen_client,
    session_id: str,
    working_memory: list,
    triggered_dims: list,
    retrieved: list,
    fatigue_level,
    addendum_enabled: bool,
    scenario,
    runtime_state,
    think: bool = False,
) -> tuple[str, bool]:
    """Run one turn with voice on. Returns (response_text, voice_still_available).
    On VoiceUnavailable the caller downgrades to text-only for the rest of the
    session."""
    import httpx

    from persona.audio_sink import SounddeviceSink
    from persona.generation import run_turn_async
    from persona.voice_client import VoiceClient, VoiceUnavailable
    from persona.voice_spec import resolve_voice_spec

    sink = SounddeviceSink(samplerate=24000, channels=1)
    async with httpx.AsyncClient(timeout=30.0) as http:
        spec = resolve_voice_spec(persona, fatigue_level, media=media)
        client = VoiceClient(
            spec=spec,
            base_url=voice_base_url,
            http_client=http,
            sink=sink,
        )
        try:
            response = await run_turn_async(
                persona=persona,
                user_message=user_message,
                store=store,
                embedder=embedder,
                gen_client=gen_client,
                session_id=session_id,
                working_memory=working_memory,
                speak=client.speak_stream,
                triggered_dims=triggered_dims,
                retrieved=retrieved,
                fatigue_level=fatigue_level,
                addendum_enabled=addendum_enabled,
                scenario=scenario,
                runtime_state=runtime_state,
                think=think,
            )
            return response, True
        except VoiceUnavailable as e:
            print(f"[voice-svc unreachable: {e}; continuing in text-only mode]")
            return "", False


def _build_store(persona_id: str, vector_size: int = 768) -> QdrantStore:
    return QdrantStore.http(
        host="localhost",
        port=6333,
        collection="persona_memory",
        vector_size=vector_size,
        persona_id=persona_id,
    )


def cmd_chat(args: argparse.Namespace) -> int:
    client = _store_client()

    scenario = None
    if args.scenario:
        scenario = _scenario_from_store(client, args.persona_id, args.scenario)
        if scenario is None:
            available = [s["scenario_id"] for s in client.list_scenarios(args.persona_id)]
            avail_str = ", ".join(available) if available else "(none)"
            print(
                f'error: unknown scenario "{args.scenario}" for persona '
                f'"{args.persona_id}"; available: {avail_str}',
                file=sys.stderr,
            )
            return 2

    embedder = EmbeddingClient(model="nomic-embed-text")
    store = _build_store(args.persona_id, vector_size=768)
    gen = GenerationClient(
        model=os.environ.get("PERSONA_MODEL", "huihui_ai/qwen3.5-abliterated:9b")
    )

    try:
        loaded = load_persona(args.persona_id, client, embedder)
    except PersonaNotInStore:
        print(
            f'error: persona "{args.persona_id}" not found in store-svc',
            file=sys.stderr,
        )
        return 2

    session_id = str(uuid.uuid4())
    loaded.runtime_state.session_count += 1
    loaded.runtime_state.last_session_at = datetime.now(tz=UTC)

    voice_on = _voice_enabled(args.voice, os.environ.get("PERSONA_VOICE"))

    working_memory: list = []
    print(f"[connected to {loaded.persona.persona_id}; session {session_id[:8]}]")
    if scenario is not None:
        print(f'[scenario: {scenario.scenario_id} — "{scenario.title}"]')
    print("[type 'exit', Ctrl-C, or Ctrl-D to end]")
    try:
        while True:
            try:
                user = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user.lower() in {"exit", "quit"}:
                break
            think, user = parse_think_prefix(user)
            if not user:
                continue

            # Retrieval
            query_vec = embedder.embed(user)
            retrieved = retrieve_relevant(
                store=store,
                query_vector=query_vec,
                config=RetrievalConfig(weights=ScoreWeights()),
                now=datetime.now(tz=UTC),
            )
            # Triggering
            triggered = select_triggered(
                persona=loaded.persona,
                user_message=user,
                embedder=embedder,
                config=TriggerConfig(),
            )
            # Fatigue
            unconsolidated = store.count_unconsolidated()
            fatigue = derive_fatigue_level(unconsolidated, FatigueThresholds())

            if voice_on:
                response, still_on = asyncio.run(
                    _run_voice_turn(
                        voice_base_url=args.voice_base_url,
                        persona=loaded.persona,
                        user_message=user,
                        store=store,
                        media=client,
                        embedder=embedder,
                        gen_client=gen,
                        session_id=session_id,
                        working_memory=working_memory,
                        triggered_dims=triggered,
                        retrieved=retrieved,
                        fatigue_level=fatigue,
                        addendum_enabled=args.addendum,
                        scenario=scenario,
                        runtime_state=loaded.runtime_state,
                        think=think,
                    )
                )
                if not still_on:
                    voice_on = False
                    response = run_turn(
                        persona=loaded.persona,
                        user_message=user,
                        store=store,
                        embedder=embedder,
                        gen_client=gen,
                        session_id=session_id,
                        working_memory=working_memory,
                        triggered_dims=triggered,
                        retrieved=retrieved,
                        fatigue_level=fatigue,
                        addendum_enabled=args.addendum,
                        scenario=scenario,
                        runtime_state=loaded.runtime_state,
                        think=think,
                    )
            else:
                response = run_turn(
                    persona=loaded.persona,
                    user_message=user,
                    store=store,
                    embedder=embedder,
                    gen_client=gen,
                    session_id=session_id,
                    working_memory=working_memory,
                    triggered_dims=triggered,
                    retrieved=retrieved,
                    fatigue_level=fatigue,
                    addendum_enabled=args.addendum,
                    scenario=scenario,
                    runtime_state=loaded.runtime_state,
                    think=think,
                )
            print(response)
            print()
    except KeyboardInterrupt:
        # Ctrl-C mid-turn (e.g. during LLM generation or voice synthesis):
        # exit cleanly, persisting runtime state once at session end.
        print()
    finally:
        # Session-end persist: one PUT to store-svc. The hot loop never wrote runtime.
        loaded.runtime_state.save()

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    client = _store_client()
    store = _build_store(args.persona_id, vector_size=768)

    persona = client.get_persona(args.persona_id)
    if persona is None:
        print(f"no persona {args.persona_id} in store")
        return 0

    from persona_core.state import RuntimeState

    rs = RuntimeState.load(client, args.persona_id)
    print(f"persona_id: {rs.persona_id}")
    print(f"session_count: {rs.session_count}")
    print(f"last_session_at: {rs.last_session_at}")
    print(f"last_sleep_pass_at: {rs.last_sleep_pass_at}")
    try:
        total = store.count()
        unconsolidated = store.count_unconsolidated()
        fatigue = derive_fatigue_level(unconsolidated, FatigueThresholds())
        print(f"records_total: {total}")
        print(f"unconsolidated_count: {unconsolidated}")
        print(f"fatigue_level: {fatigue.value}")
    except Exception as e:
        print(f"qdrant unreachable: {e}")
    return 0


def cmd_scenarios(args: argparse.Namespace) -> int:
    client = _store_client()
    items = client.list_scenarios(args.persona_id)
    if not items:
        print("(no scenarios authored)")
        return 0
    for s in items:
        print(f"{s['scenario_id']} — {s.get('title', '')}")
    return 0


def cmd_inventory(args: argparse.Namespace) -> int:
    import json as _json

    from persona_core.inventory import compute_inventory
    from persona_core.records import RecordType

    client = _store_client()
    persona = client.get_persona(args.persona_id)
    if persona is None:
        print(f"error: no persona {args.persona_id} in store", file=sys.stderr)
        return 2

    qdrant_count: int | None
    if args.no_qdrant:
        qdrant_count = None
    else:
        try:
            store = _build_store(args.persona_id, vector_size=768)
            qdrant_count = len(store.list_by_type(RecordType.SEEDED_NARRATIVE))
        except Exception:
            qdrant_count = None

    report = compute_inventory(
        persona=persona,
        media_exists=lambda name: client.get_media(args.persona_id, name) is not None,
        qdrant_seeded_count=qdrant_count,
    )
    payload = {
        "persona_id": report.persona_id,
        "entries": [
            {
                "slot": e.slot,
                "category": e.category.value,
                "label": e.label,
                "detail": e.detail,
            }
            for e in report.entries
        ],
    }
    print(_json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="persona")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_chat = sub.add_parser("chat", help="Start an interactive session")
    p_chat.add_argument("persona_id")
    p_chat.add_argument(
        "--addendum",
        action="store_true",
        help="Enable fatigue addendum in system prompt",
    )
    p_chat.add_argument(
        "--scenario",
        default=None,
        help="Scenario id stored for the persona in store-svc",
    )
    p_chat.add_argument(
        "--voice",
        dest="voice",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Speak replies aloud via voice-svc. Default: on (override with --no-voice).",
    )
    p_chat.add_argument(
        "--voice-base-url",
        default=os.environ.get("VOICE_SVC_URL", "http://127.0.0.1:7000"),
        help="Voice service base URL (default: http://127.0.0.1:7000)",
    )
    p_chat.set_defaults(func=cmd_chat)

    p_status = sub.add_parser("status", help="Show persona runtime status")
    p_status.add_argument("persona_id")
    p_status.set_defaults(func=cmd_status)

    p_scn = sub.add_parser("scenarios", help="List scenarios authored for a persona")
    p_scn.add_argument("persona_id")
    p_scn.set_defaults(func=cmd_scenarios)

    p_inv = sub.add_parser(
        "inventory", help="Print structured JSON of blanks/partials on a persona"
    )
    p_inv.add_argument("persona_id")
    p_inv.add_argument(
        "--no-qdrant",
        action="store_true",
        help="Skip Qdrant probe (use when Qdrant is offline; trauma-yet flag is suppressed).",
    )
    p_inv.set_defaults(func=cmd_inventory)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

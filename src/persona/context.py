"""Context assembly. Produces the final prompt text for one generation call."""

from __future__ import annotations

from dataclasses import dataclass

from persona_core.records import EpisodicRecord
from persona_core.schema import Dimension, Persona, Presence

from persona.fatigue import FatigueLevel, fatigue_addendum
from persona.retrieval import RetrievedItem
from persona.scenario import Scenario

SYSTEM_PROMPT = (
    "You are a persona, not an assistant. Stay in character. Speak in your "
    "own voice. Don't break character to comment on yourself or the "
    "conversation. Respond as the person described below would respond — "
    "with their cadence, their values, their evasions, their warmth or "
    "coldness. Be specific and embodied. Don't summarize what you remember; "
    "let memory shape the response naturally."
)


@dataclass
class ContextAssembly:
    text: str
    approx_tokens: int


def assemble_context(
    persona: Persona,
    user_message: str,
    triggered_dims: list[Dimension],
    working_memory: list[EpisodicRecord],
    retrieved: list[RetrievedItem],
    fatigue_level: FatigueLevel,
    addendum_enabled: bool,
    scenario: Scenario | None = None,
) -> ContextAssembly:
    parts: list[str] = []

    parts.append("[SYSTEM]")
    parts.append(SYSTEM_PROMPT)

    if addendum_enabled:
        addendum = fatigue_addendum(fatigue_level)
        if addendum:
            parts.append("")
            parts.append(addendum)

    parts.append("")
    parts.append("[IDENTITY]")
    parts.append(_format_identity(persona.identity))

    parts.append("")
    parts.append("[ALWAYS-ON SUBSTRATE — VOICE ANCHORS]")
    for dim in persona.substrate.values():
        if dim.voice_anchor:
            parts.append(f"- {dim.name}: {dim.voice_anchor}")
        elif dim.presence == Presence.ALWAYS_ON:
            parts.append(f"- {dim.name}: {dim.prose}")

    parts.append("")
    parts.append("[ALWAYS-ON SELF-CONCEPT]")
    for dim in persona.self_concept.values():
        if dim.presence == Presence.ALWAYS_ON:
            parts.append(f"### {dim.name}")
            parts.append(dim.prose)

    if scenario is not None:
        parts.append("")
        parts.append("[SCENARIO]")
        parts.append(scenario.scene)
        header = _scenario_interlocutor_header(scenario)
        if header is not None:
            parts.append("")
            parts.append(header)
        if scenario.interlocutor:
            if header is None:
                parts.append("")
            parts.append(scenario.interlocutor)

    if triggered_dims:
        parts.append("")
        parts.append("[ON YOUR MIND RIGHT NOW]")
        for dim in triggered_dims:
            parts.append(f"### {dim.name}")
            parts.append(dim.prose)
            if dim.perception:
                parts.append(f"(How you experience this: {dim.perception})")

    if retrieved:
        parts.append("")
        parts.append("[REMEMBERED]")
        for item in retrieved:
            parts.append(_format_record(item.record))

    if working_memory:
        parts.append("")
        parts.append("[RECENT EXCHANGES]")
        for rec in working_memory:
            parts.append(_format_record(rec))

    parts.append("")
    parts.append("[USER]")
    parts.append(user_message)
    parts.append("")
    parts.append("[YOU]")

    text = "\n".join(parts)
    return ContextAssembly(text=text, approx_tokens=_approx_tokens(text))


def _scenario_interlocutor_header(scenario: Scenario) -> str | None:
    """Render the `Speaking to: …` header line. Returns None if both atoms are absent."""
    name = scenario.interlocutor_name
    relation = scenario.interlocutor_relation
    if name and relation:
        return f"Speaking to: {name} — your {relation}"
    if name:
        return f"Speaking to: {name}"
    if relation:
        return f"Speaking to: your {relation}"
    return None


def _format_identity(identity: dict) -> str:
    lines = []
    for k, v in identity.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _format_record(rec: EpisodicRecord) -> str:
    if "user" in rec.content and "assistant" in rec.content:
        return f"[exchange] user: {rec.content['user']}\n[exchange] you: {rec.content['assistant']}"
    if "event" in rec.content:
        return f"[memory] {rec.content['event']}"
    return f"[memory] {rec.content}"


def _approx_tokens(text: str) -> int:
    """Rough approximation; ~4 chars per token. Used for budget guardrails, not exact count."""
    return max(1, len(text) // 4)

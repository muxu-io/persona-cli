from datetime import date as _date

from persona_core.records import EpisodicRecord, RecordType
from persona_core.schema import Dimension, Persona, Presence

from persona.context import ContextAssembly, assemble_context
from persona.fatigue import FatigueLevel
from persona.retrieval import RetrievedItem
from persona.scenario import Scenario


def _persona() -> Persona:
    return Persona(
        persona_id="ada",
        spec_version=1,
        identity={"name": "Ada", "age": 56, "role": "trauma nurse"},
        substrate={
            "cognitive_profile": Dimension(
                name="cognitive_profile",
                presence=Presence.ALWAYS_ON,
                prose="quick, concrete, hedges in groups",
                structured={},
            ),
            "physical.body": Dimension(
                name="physical.body",
                presence=Presence.TRIGGERED,
                voice_anchor="compact, wiry",
                prose="Compact and wiry, slightly hunched...",
                structured={},
            ),
        },
        self_concept={
            "moral_self_image": Dimension(
                name="moral_self_image",
                presence=Presence.ALWAYS_ON,
                prose="Sees herself as flawed but trying.",
                structured={},
            ),
        },
        traumas=[],
    )


def _turn_pair(user, assistant, sid="s-1"):
    return EpisodicRecord(
        type=RecordType.TURN_PAIR,
        session_id=sid,
        content={"user": user, "assistant": assistant},
        initial_salience=0.4,
    )


def test_assemble_includes_identity_and_always_on():
    persona = _persona()
    out = assemble_context(
        persona=persona,
        user_message="how do you feel?",
        triggered_dims=[],
        working_memory=[],
        retrieved=[],
        fatigue_level=FatigueLevel.RESTED,
        addendum_enabled=False,
    )
    assert "Ada" in out.text
    assert "trauma nurse" in out.text
    assert "quick, concrete" in out.text  # always-on substrate prose? voice anchor included
    assert "compact, wiry" in out.text  # voice anchor of triggered dim still always-on
    assert "flawed but trying" in out.text  # self-concept prose


def test_assemble_pulls_in_triggered_dim_full_prose_only_when_triggered():
    persona = _persona()
    body = persona.substrate["physical.body"]
    out_triggered = assemble_context(
        persona=persona,
        user_message="how do you look?",
        triggered_dims=[body],
        working_memory=[],
        retrieved=[],
        fatigue_level=FatigueLevel.RESTED,
        addendum_enabled=False,
    )
    out_not_triggered = assemble_context(
        persona=persona,
        user_message="how do you feel?",
        triggered_dims=[],
        working_memory=[],
        retrieved=[],
        fatigue_level=FatigueLevel.RESTED,
        addendum_enabled=False,
    )
    assert "Compact and wiry, slightly hunched" in out_triggered.text
    assert "Compact and wiry, slightly hunched" not in out_not_triggered.text


def test_assemble_includes_working_memory_in_order():
    persona = _persona()
    wm = [
        _turn_pair("turn-1-u", "turn-1-a"),
        _turn_pair("turn-2-u", "turn-2-a"),
    ]
    out = assemble_context(
        persona=persona,
        user_message="now?",
        triggered_dims=[],
        working_memory=wm,
        retrieved=[],
        fatigue_level=FatigueLevel.RESTED,
        addendum_enabled=False,
    )
    pos1 = out.text.find("turn-1-u")
    pos2 = out.text.find("turn-2-u")
    assert pos1 != -1 and pos2 != -1
    assert pos1 < pos2


def test_assemble_includes_retrieved_episodes():
    persona = _persona()
    retrieved = [
        RetrievedItem(record=_turn_pair("about-dog", "Bramble"), score=0.7, similarity=0.7),
    ]
    out = assemble_context(
        persona=persona,
        user_message="what was my dog's name?",
        triggered_dims=[],
        working_memory=[],
        retrieved=retrieved,
        fatigue_level=FatigueLevel.RESTED,
        addendum_enabled=False,
    )
    assert "Bramble" in out.text


def test_addendum_when_tired_and_enabled():
    persona = _persona()
    out_on = assemble_context(
        persona=persona,
        user_message="how are you?",
        triggered_dims=[],
        working_memory=[],
        retrieved=[],
        fatigue_level=FatigueLevel.TIRED,
        addendum_enabled=True,
    )
    out_off = assemble_context(
        persona=persona,
        user_message="how are you?",
        triggered_dims=[],
        working_memory=[],
        retrieved=[],
        fatigue_level=FatigueLevel.TIRED,
        addendum_enabled=False,
    )
    assert "cognitively tired" in out_on.text
    assert "cognitively tired" not in out_off.text
    assert isinstance(out_on, ContextAssembly)
    assert isinstance(out_off, ContextAssembly)


def _scenario(
    scenario_id: str = "shift-cut-corridor",
    scene: str = "You're in the corridor, fluorescents overhead.",
    interlocutor: str | None = None,
    interlocutor_name: str | None = None,
    interlocutor_relation: str | None = None,
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        persona_id="ada",
        spec_version=1,
        title="t",
        created=_date(2026, 5, 1),
        scene=scene,
        interlocutor=interlocutor,
        interlocutor_name=interlocutor_name,
        interlocutor_relation=interlocutor_relation,
    )


def test_assemble_context_no_scenario_omits_block():
    persona = _persona()
    out = assemble_context(
        persona=persona,
        user_message="hi",
        triggered_dims=[],
        working_memory=[],
        retrieved=[],
        fatigue_level=FatigueLevel.RESTED,
        addendum_enabled=False,
    )
    assert "[SCENARIO]" not in out.text


def test_assemble_context_with_scenario_inserts_block_in_correct_position():
    persona = _persona()
    out = assemble_context(
        persona=persona,
        user_message="hi",
        triggered_dims=[],
        working_memory=[],
        retrieved=[],
        fatigue_level=FatigueLevel.RESTED,
        addendum_enabled=False,
        scenario=_scenario(scene="The corridor smells of bleach."),
    )
    assert "[SCENARIO]" in out.text
    assert "The corridor smells of bleach." in out.text
    pos_self_concept = out.text.find("[ALWAYS-ON SELF-CONCEPT]")
    pos_scenario = out.text.find("[SCENARIO]")
    pos_user = out.text.find("[USER]")
    assert pos_self_concept != -1 and pos_scenario != -1 and pos_user != -1
    assert pos_self_concept < pos_scenario < pos_user


def test_assemble_context_scenario_interlocutor_full():
    persona = _persona()
    out = assemble_context(
        persona=persona,
        user_message="hi",
        triggered_dims=[],
        working_memory=[],
        retrieved=[],
        fatigue_level=FatigueLevel.RESTED,
        addendum_enabled=False,
        scenario=_scenario(
            interlocutor="A nurse on the same ward.",
            interlocutor_name="Mhairi",
            interlocutor_relation="colleague",
        ),
    )
    assert "Speaking to: Mhairi — your colleague" in out.text
    assert "A nurse on the same ward." in out.text


def test_assemble_context_scenario_interlocutor_name_only():
    persona = _persona()
    out = assemble_context(
        persona=persona,
        user_message="hi",
        triggered_dims=[],
        working_memory=[],
        retrieved=[],
        fatigue_level=FatigueLevel.RESTED,
        addendum_enabled=False,
        scenario=_scenario(
            interlocutor="The interlocutor prose.",
            interlocutor_name="David",
        ),
    )
    assert "Speaking to: David" in out.text
    assert "your" not in out.text.split("Speaking to: David")[1].split("\n")[0]
    assert "The interlocutor prose." in out.text


def test_assemble_context_scenario_interlocutor_relation_only():
    persona = _persona()
    out = assemble_context(
        persona=persona,
        user_message="hi",
        triggered_dims=[],
        working_memory=[],
        retrieved=[],
        fatigue_level=FatigueLevel.RESTED,
        addendum_enabled=False,
        scenario=_scenario(
            interlocutor="Prose body.",
            interlocutor_relation="husband",
        ),
    )
    assert "Speaking to: your husband" in out.text
    assert "Prose body." in out.text


def test_assemble_context_scenario_interlocutor_prose_only():
    persona = _persona()
    out = assemble_context(
        persona=persona,
        user_message="hi",
        triggered_dims=[],
        working_memory=[],
        retrieved=[],
        fatigue_level=FatigueLevel.RESTED,
        addendum_enabled=False,
        scenario=_scenario(interlocutor="Pure prose, no atoms."),
    )
    assert "Speaking to:" not in out.text
    assert "Pure prose, no atoms." in out.text


def test_assemble_context_scenario_no_interlocutor():
    persona = _persona()
    out = assemble_context(
        persona=persona,
        user_message="hi",
        triggered_dims=[],
        working_memory=[],
        retrieved=[],
        fatigue_level=FatigueLevel.RESTED,
        addendum_enabled=False,
        scenario=_scenario(scene="Just a scene."),
    )
    assert "[SCENARIO]" in out.text
    assert "Just a scene." in out.text
    assert "Speaking to:" not in out.text

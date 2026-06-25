from datetime import date
from pathlib import Path

import pytest

from persona.scenario import (
    Scenario,
    ScenarioNotFoundError,
    ScenarioParseError,
    list_scenarios,
    load_scenario,
)


def _write_scenario(base: Path, persona_id: str, scenario_id: str, body: str) -> Path:
    """Write a scenario file at base/<persona_id>/scenarios/<scenario_id>.md and return the path."""
    sdir = base / persona_id / "scenarios"
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / f"{scenario_id}.md"
    path.write_text(body)
    return path


MINIMAL_BODY = """\
---
scenario_id: shift-cut-corridor
persona_id: ada-mcleish
spec_version: 1
created: 2026-05-01
title: "Shift cut, corridor, fluorescent lights"
---

# Shift cut, corridor, fluorescent lights

## Scene

You've just been pulled aside by the charge nurse. The corridor smells of bleach.
"""


def _body_for(scenario_id: str) -> str:
    return MINIMAL_BODY.replace("shift-cut-corridor", scenario_id)


FULL_INTERLOCUTOR_BODY = """\
---
scenario_id: full-case
persona_id: ada-mcleish
spec_version: 1
created: 2026-05-01
title: "Full interlocutor case"
---

## Scene

Some scene paragraph.

## Interlocutor

```yaml
name: "David"
relation: "husband"
```

A nurse on the same ward. She doesn't know what you just heard.
"""


PARTIAL_INTERLOCUTOR_BODY = """\
---
scenario_id: partial-case
persona_id: ada-mcleish
spec_version: 1
created: 2026-05-01
title: "Partial interlocutor"
---

## Scene

Scene paragraph.

## Interlocutor

```yaml
relation: "husband"
```

Prose body.
"""


PROSE_ONLY_INTERLOCUTOR_BODY = """\
---
scenario_id: prose-only
persona_id: ada-mcleish
spec_version: 1
created: 2026-05-01
title: "Prose only interlocutor"
---

## Scene

Scene paragraph.

## Interlocutor

Prose paragraph with no YAML fence at all.
"""


MISSING_SCENE_BODY = """\
---
scenario_id: missing-scene
persona_id: ada-mcleish
spec_version: 1
created: 2026-05-01
title: "Missing scene"
---

## Interlocutor

Some prose.
"""


ID_MISMATCH_BODY = """\
---
scenario_id: wrong-id
persona_id: ada-mcleish
spec_version: 1
created: 2026-05-01
title: "Mismatched id"
---

## Scene

Scene paragraph.
"""


PERSONA_MISMATCH_BODY = """\
---
scenario_id: persona-mismatch
persona_id: someone-else
spec_version: 1
created: 2026-05-01
title: "Wrong persona"
---

## Scene

Scene paragraph.
"""


def test_load_scenario_minimal(tmp_path):
    path = _write_scenario(tmp_path, "ada-mcleish", "shift-cut-corridor", MINIMAL_BODY)
    s = load_scenario(path)
    assert isinstance(s, Scenario)
    assert s.scenario_id == "shift-cut-corridor"
    assert s.persona_id == "ada-mcleish"
    assert s.spec_version == 1
    assert s.title == "Shift cut, corridor, fluorescent lights"
    assert s.created == date(2026, 5, 1)
    assert "charge nurse" in s.scene
    assert "bleach" in s.scene
    assert s.interlocutor is None
    assert s.interlocutor_name is None
    assert s.interlocutor_relation is None


def test_load_scenario_with_interlocutor_full(tmp_path):
    path = _write_scenario(tmp_path, "ada-mcleish", "full-case", FULL_INTERLOCUTOR_BODY)
    s = load_scenario(path)
    assert s.interlocutor_name == "David"
    assert s.interlocutor_relation == "husband"
    assert s.interlocutor is not None
    assert "She doesn't know" in s.interlocutor


def test_load_scenario_with_interlocutor_partial(tmp_path):
    path = _write_scenario(tmp_path, "ada-mcleish", "partial-case", PARTIAL_INTERLOCUTOR_BODY)
    s = load_scenario(path)
    assert s.interlocutor_name is None
    assert s.interlocutor_relation == "husband"
    assert s.interlocutor is not None
    assert "Prose body" in s.interlocutor


def test_load_scenario_with_interlocutor_prose_only(tmp_path):
    path = _write_scenario(tmp_path, "ada-mcleish", "prose-only", PROSE_ONLY_INTERLOCUTOR_BODY)
    s = load_scenario(path)
    assert s.interlocutor_name is None
    assert s.interlocutor_relation is None
    assert s.interlocutor is not None
    assert "no YAML fence" in s.interlocutor


def test_load_scenario_missing_scene_raises(tmp_path):
    path = _write_scenario(tmp_path, "ada-mcleish", "missing-scene", MISSING_SCENE_BODY)
    with pytest.raises(ScenarioParseError, match="Scene"):
        load_scenario(path)


def test_load_scenario_id_mismatch_raises(tmp_path):
    # Filename stem is "id-mismatch" but frontmatter says "wrong-id".
    path = _write_scenario(tmp_path, "ada-mcleish", "id-mismatch", ID_MISMATCH_BODY)
    with pytest.raises(ScenarioParseError, match="scenario_id"):
        load_scenario(path)


def test_load_scenario_persona_id_mismatch_raises(tmp_path):
    # Path says "ada-mcleish" but frontmatter says "someone-else".
    path = _write_scenario(tmp_path, "ada-mcleish", "persona-mismatch", PERSONA_MISMATCH_BODY)
    with pytest.raises(ScenarioParseError, match="persona_id"):
        load_scenario(path)


def test_load_scenario_not_found(tmp_path):
    with pytest.raises(ScenarioNotFoundError):
        load_scenario(tmp_path / "ada-mcleish" / "scenarios" / "nope.md")


def test_list_scenarios_empty_when_dir_absent(tmp_path):
    # No directory created.
    assert list_scenarios(tmp_path / "ada-mcleish" / "scenarios") == []


def test_list_scenarios_returns_sorted_ids(tmp_path):
    _write_scenario(tmp_path, "ada-mcleish", "z-last", _body_for("z-last"))
    _write_scenario(tmp_path, "ada-mcleish", "a-first", _body_for("a-first"))
    _write_scenario(tmp_path, "ada-mcleish", "m-middle", _body_for("m-middle"))
    ids = list_scenarios(tmp_path / "ada-mcleish" / "scenarios")
    assert ids == ["a-first", "m-middle", "z-last"]

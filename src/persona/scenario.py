"""Scenario file loader.

Markdown body with YAML frontmatter and an optional YAML fence inside `## Interlocutor`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import frontmatter
import yaml


class ScenarioNotFoundError(Exception):
    """Raised when a scenario file does not exist at the requested path."""


class ScenarioParseError(Exception):
    """Raised when a scenario file fails validation."""


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    persona_id: str
    spec_version: int
    title: str
    created: date
    scene: str  # body of "## Scene"
    interlocutor: str | None  # prose body of "## Interlocutor", or None
    interlocutor_name: str | None  # YAML atom inside "## Interlocutor", or None
    interlocutor_relation: str | None  # YAML atom inside "## Interlocutor", or None


_YAML_FENCE_RE = re.compile(
    r"^```yaml\s*\n(.*?)\n```\s*$",
    re.DOTALL | re.MULTILINE,
)


def load_scenario(path: Path) -> Scenario:
    """Load a scenario from `<base>/<persona_id>/scenarios/<scenario_id>.md`.

    `persona_id` is derived from `path.parent.parent.name` and cross-checked against frontmatter.
    `scenario_id` is derived from `path.stem` and cross-checked against frontmatter.
    """
    if not path.exists():
        raise ScenarioNotFoundError(str(path))

    expected_persona_id = path.parent.parent.name
    expected_scenario_id = path.stem

    post = frontmatter.load(path)

    fm_scenario_id = post.metadata.get("scenario_id")
    fm_persona_id = post.metadata.get("persona_id")
    if fm_scenario_id != expected_scenario_id:
        raise ScenarioParseError(
            f"frontmatter scenario_id {fm_scenario_id!r} does not match filename stem "
            f"{expected_scenario_id!r}"
        )
    if fm_persona_id != expected_persona_id:
        raise ScenarioParseError(
            f"frontmatter persona_id {fm_persona_id!r} does not match parent directory "
            f"{expected_persona_id!r}"
        )

    spec_version = int(post.metadata.get("spec_version", 1))
    title = str(post.metadata["title"])
    created_raw = post.metadata["created"]
    created = created_raw if isinstance(created_raw, date) else date.fromisoformat(str(created_raw))

    sections = _split_by_h2(post.content)
    if "Scene" not in sections or not sections["Scene"].strip():
        raise ScenarioParseError("missing or empty `## Scene` section")
    scene = sections["Scene"].strip()

    interlocutor_block = sections.get("Interlocutor")
    if interlocutor_block is None:
        interlocutor = None
        interlocutor_name = None
        interlocutor_relation = None
    else:
        atoms, prose = _extract_yaml_block(interlocutor_block)
        interlocutor_name = atoms.get("name")
        interlocutor_relation = atoms.get("relation")
        interlocutor = prose if prose else None

    return Scenario(
        scenario_id=expected_scenario_id,
        persona_id=expected_persona_id,
        spec_version=spec_version,
        title=title,
        created=created,
        scene=scene,
        interlocutor=interlocutor,
        interlocutor_name=interlocutor_name,
        interlocutor_relation=interlocutor_relation,
    )


def list_scenarios(scenarios_dir: Path) -> list[str]:
    """Return sorted scenario_ids (filename stems) for `<scenarios_dir>/*.md`.

    Returns `[]` if dir absent.
    """
    if not scenarios_dir.exists() or not scenarios_dir.is_dir():
        return []
    return sorted(p.stem for p in scenarios_dir.glob("*.md"))


def _split_by_h2(content: str) -> dict[str, str]:
    """Split body by `## Heading` blocks → {heading: body}.

    Unknown / out-of-order headings preserved.
    """
    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in content.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = m.group(1)
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def _extract_yaml_block(body: str) -> tuple[dict[str, Any], str]:
    """Pull the first ```yaml fence; return (atoms, remaining_prose)."""
    m = _YAML_FENCE_RE.search(body)
    if not m:
        return {}, body.strip()
    parsed = yaml.safe_load(m.group(1)) or {}
    remaining = (body[: m.start()] + body[m.end() :]).strip()
    return parsed, remaining

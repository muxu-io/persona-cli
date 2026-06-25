import argparse
import json
from pathlib import Path

import httpx
import respx
from persona_core.parser import parse_persona_file
from persona_core.serialization import persona_to_definition

from persona.cli import cmd_inventory

BASE = "http://store:7600"
FIXTURE = Path(__file__).parent.parent / "fixtures" / "inventory_blanks_persona.md"


@respx.mock
def test_inventory_prints_json_for_blanks_fixture(monkeypatch, capsys):
    monkeypatch.setenv("PERSONA_STORE_URL", BASE)
    persona = parse_persona_file(FIXTURE)
    respx.get(f"{BASE}/personas/blanks-test").mock(
        return_value=httpx.Response(
            200,
            json={
                "persona_id": "blanks-test",
                "spec_version": persona.spec_version,
                "definition": persona_to_definition(persona),
                "tags": [],
            },
        )
    )

    rc = cmd_inventory(argparse.Namespace(persona_id="blanks-test", no_qdrant=True))
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["persona_id"] == "blanks-test"
    slots = {e["slot"] for e in payload["entries"]}
    assert "voice_sample" in slots
    assert "dimension:sexuality" in slots

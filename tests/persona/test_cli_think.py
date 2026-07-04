"""Unit tests for the `/think` per-turn prefix parser (spec
2026-07-04-think-prefix-design)."""

from persona.cli import parse_think_prefix


def test_no_prefix_returns_false_and_line_unchanged():
    assert parse_think_prefix("how do I do this?") == (False, "how do I do this?")


def test_think_prefix_sets_flag_and_strips_token():
    assert parse_think_prefix("/think how do I do this?") == (True, "how do I do this?")


def test_think_prefix_alone_yields_empty_message():
    assert parse_think_prefix("/think") == (True, "")


def test_thinking_is_not_a_think_prefix():
    # Word boundary: only '/think' + whitespace/EOL triggers.
    assert parse_think_prefix("/thinking about it") == (False, "/thinking about it")


def test_extra_whitespace_after_token_is_stripped():
    assert parse_think_prefix("/think    spaced out") == (True, "spaced out")

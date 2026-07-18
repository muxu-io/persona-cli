# persona-cli

Interactive runtime CLI for locally-hosted "persona" LLMs — models that accumulate
episodic memory and evolve a personality shaped by past interaction. Imports only
`persona-core`; loads definitions/state/scenarios from `persona-store` over HTTP,
retrieves memories from Qdrant, generates against ollama, optionally streams voice
to `voice-svc`.

## Install

    git clone https://gitlab.com/muxu-io/persona-cli.git
    cd persona-cli
    poetry install

## Use (against a running moai stack)

Set `PERSONA_MODEL` to the model your stack serves (otherwise it defaults to the
old 9B), then run from the checkout with `poetry run`:

    PERSONA_MODEL=<model> poetry run persona chat ada-mcleish [--scenario <id>] [--no-voice]
    poetry run persona status ada-mcleish
    poetry run persona scenarios ada-mcleish
    poetry run persona inventory ada-mcleish

Inside a `chat` session, prefix a message with `/think` to make the model reason
for that one turn (e.g. `/think what changed between us since last time?`).
Replies are fast by default; the reasoning is discarded, not shown or spoken.

Requires the runtime services on localhost (ollama, qdrant, persona-store; voice-svc
optional). Voice playback needs PortAudio (sounddevice).

License: Apache-2.0.

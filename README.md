# persona-cli

Interactive runtime CLI for locally-hosted "persona" LLMs — models that accumulate
episodic memory and evolve a personality shaped by past interaction. Imports only
`persona-core`; loads definitions/state/scenarios from `persona-store` over HTTP,
retrieves memories from Qdrant, generates against ollama, optionally streams voice
to `voice-svc`.

## Install

    pip install --extra-index-url https://gitlab.com/api/v4/projects/83381755/packages/pypi/simple persona-cli

## Use (against a running moai stack)

    persona chat ada-mcleish [--scenario <id>] [--no-voice]
    persona status ada-mcleish
    persona scenarios ada-mcleish
    persona inventory ada-mcleish

Requires the runtime services on localhost (ollama, qdrant, persona-store; voice-svc
optional). Voice playback needs PortAudio (sounddevice).

License: Apache-2.0.

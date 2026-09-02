# Claude Fable and Mythos 5.1 Online Catalog

## Goal

Add Anthropic's active Claude Fable 5.1 and invite-only Claude Mythos 5.1 models to Geist's online model catalog using their documented Claude API identifiers.

## Scope

- Register `claude-fable-5-1` and `claude-mythos-5-1` under the existing Anthropic provider.
- Use Anthropic's documented 1M-token context window, 128K-token maximum output, vision, tool-use, and streaming capabilities.
- Mark Fable as recommended and Mythos as non-recommended because Mythos access is invite-only through Project Glasswing.
- Add matching metadata to the model-sync configuration so future Anthropic API syncs preserve the intended catalog details.
- Include both models in the frontend static fallback so they remain selectable when the model endpoint is unavailable.
- Add focused backend and frontend tests for model visibility and metadata.

No provider, route, credential, dependency, or database changes are required because both models use Geist's existing Anthropic Messages API integration.

## Verification

1. Run focused backend registry/API tests for the two model IDs and their documented limits.
2. Run the focused frontend provider-selector test for Anthropic fallback models.
3. Run relevant formatting/static checks for changed files.
4. If feasible, start the Docker stack, inspect logs, and curl the models endpoint on `localhost:3000`.

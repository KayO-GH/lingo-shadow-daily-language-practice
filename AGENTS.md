# AGENTS

## Project Intent

Build a standalone Gradio app that turns a learner's real daily-life use cases into a practical starter pack of high-utility sentences and downloadable target-language audio.

## Product Rules

- Start from the learner's own routines, responsibilities, and recurring conversations.
- Prioritize short, reusable sentences that cover common daily verbs.
- Default the target language to French, but keep the app multilingual.
- Always produce downloadable study artifacts, not just on-screen text.

## Technical Rules

- Keep the project standalone inside this folder.
- Prefer simple dependencies and clear fallbacks over brittle integrations.
- Load local secrets from this app's `.env` first, then the external fallback path if present.
- Keep non-network logic testable without calling the live APIs.

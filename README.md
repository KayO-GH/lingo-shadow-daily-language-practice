---
title: LingoShadow - Daily Language Practice
emoji: "🎧"
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
python_version: "3.12"
pinned: false
tags:
  - track:backyard
  - sponsor:openai
  - sponsor:modal
  - achievement:fieldnotes
models:
  - Qwen/Qwen3-8B
  - CohereLabs/tiny-aya-global
  - kyutai/tts-1.6b-en_fr
  - facebook/mms-tts-deu
  - hexgrad/Kokoro-82M
---

# LingoShadow - Daily Language Practice

`LingoShadow - Daily Language Practice` is a standalone Gradio project for building a personalized self-study pack from real daily-life situations. The app now supports multiple target languages, with the TTS backend selected per target language using the earlier language-router plan from this project.

## Links

- Hugging Face Space: https://huggingface.co/spaces/build-small-hackathon/lingo-shadow-daily-language-practice
- Live app: https://build-small-hackathon-lingo-shadow-daily-languag-56b0bec.hf.space/
- GitHub repo: https://github.com/KayO-GH/daily-language-practice
- Demo video: https://www.loom.com/share/1ea6c133476e4a7a8578787043a303dc
- Social post: https://www.linkedin.com/posts/kwadwo-agyapon-ntra_smallmodels-tinymodels-buildsmall-activity-7472256515995922433-QX1u
- Blog post: https://kayo-gh.github.io/building-lingoshadow/

## Try the demo

1. Open the live app.
2. Keep the default target language as `French`, or choose another supported target language.
3. Use the default daily-routine prompt or describe your own commute, errands, work, family, travel, and shopping situations.
4. Click `Build study pack`.
5. Review the generated sentence table, play the first MP3 preview, and download the ZIP bundle.

## Model stack

- Generation: `Qwen/Qwen3-8B` with `8.2B` parameters
- Translation: `CohereLabs/tiny-aya-global` with `3.35B` parameters
- TTS:
  - English, French: `kyutai/tts-1.6b-en_fr` at approximately `1.8B`
  - German: `facebook/mms-tts-deu` at `36M`
  - Spanish, Italian, Portuguese, Japanese: `hexgrad/Kokoro-82M` at `82M`

Each individual model in the default app stack is below the Build Small hackathon `<= 32B` model cap. For transparency, the default French stack is approximately `13.35B` aggregate parameters across generation, translation, and TTS, but the hackathon constraint is applied per model.

## What it does

- asks the learner to describe their general use cases and daily routines
- generates practical target-language sentences that are likely to come up often
- prioritizes useful verbs and recurring situations
- returns a simple 45-minute daily study routine
- highlights the core verbs the learner should review first
- generates downloadable target-language audio tracks through a dedicated Modal-backed TTS service
- groups audio into files of up to `20` sentences per MP3 track plus a downloadable ZIP bundle

## Supported languages

- currently supported target languages: English, French, Spanish, German, Italian, Portuguese, Japanese
- source-language prompts can be generated from English, French, Spanish, German, Portuguese, Italian, or Japanese
- new target languages will be added as quickly as I can integrate reliable TTS models for them
- the app does not auto-detect TTS language; it uses the selected target language
- the default built-in router matrix is:
  - `en`, `fr` -> `kyutai/tts-1.6b-en_fr`
  - `de` -> `facebook/mms-tts-deu`
  - `es`, `it`, `pt`, `ja` -> `hexgrad/Kokoro-82M`
- the app still supports language-specific Modal env overrides when you want to split those backends across separate endpoints
- legacy single-backend env vars still work as a fallback

## Environment variables

This app loads environment variables in this order:

1. `lingo-shadow-daily-language-practice/.env`
2. the local fallback env file, when present in the developer workspace

App-side variables:

- `HF_TOKEN`: required for Qwen sentence generation
- `MODAL_TTS_BASE_URL`: legacy fallback Modal TTS endpoint base URL
- `MODAL_TTS_AUTH_TOKEN`: legacy fallback bearer token sent to the Modal TTS service
- `MODAL_TTS_TIMEOUT_SECONDS`: optional request timeout override for TTS calls
- `MODAL_TTS_BASE_URL_<LANG_CODE>`: preferred language-specific Modal TTS endpoint, for example `MODAL_TTS_BASE_URL_FR` or `MODAL_TTS_BASE_URL_ES`
- `MODAL_TTS_AUTH_TOKEN_<LANG_CODE>`: optional language-specific bearer token
- `MODAL_TTS_MODEL_<LANG_CODE>`: language-specific model label shown in the UI and exported study packs
- `MODAL_TTS_VOICE_<LANG_CODE>`: language-specific voice label shown in the UI and exported study packs
- `MODAL_TTS_VOICE_REPO_<LANG_CODE>`: optional language-specific voice repository label
- `MODAL_TTS_PARAMS_<LANG_CODE>`: optional language-specific TTS parameter count used for stack disclosure

Modal-side variables:

- `MODAL_TTS_AUTH_TOKEN`: optional bearer token expected by the Modal service
- `MODAL_TTS_APP_NAME`: optional Modal app name override when deploying separate services from the same file
- `MODAL_TTS_SECRET_NAME`: optional Modal secret name override if you do not want to reuse `language-learning-bearer`

This repo currently expects that Modal secret to be attached from a secret named `language-learning-bearer`.

## Run locally

```bash
uv venv .venv
source .venv/bin/activate
uv sync
uv run gradio app.py
```

## Deploy the Modal TTS service

The repo includes [`modal_tts_service.py`](modal_tts_service.py), which serves one language-routed TTS API behind:

- `GET /healthz`
- `POST /warmup`
- `POST /synthesize-track`

Deploy it once and let it route by the request `language` code. A typical pattern is:

```bash
uv run --with modal modal deploy modal_tts_service.py
```

After deployment, you can either:

- point every language at the same routed endpoint with `MODAL_TTS_BASE_URL`, or
- split languages across dedicated endpoints with `MODAL_TTS_BASE_URL_FR`, `MODAL_TTS_BASE_URL_DE`, `MODAL_TTS_BASE_URL_ES`, and so on

The deployed function attaches the Modal secret named `language-learning-bearer` by default, which must contain `MODAL_TTS_AUTH_TOKEN`.

The request body for `POST /synthesize-track` is:

```json
{
  "sentences": ["Bonjour.", "Comment allez-vous ?"],
  "language": "fr",
  "slow_audio": false
}
```

The response is raw `audio/mpeg` bytes for the concatenated track.

## Notes

- The generated ZIP includes JSON, CSV, a text summary, a daily routine file, a focus-verbs file, and the MP3 tracks.
- Preview audio uses the first generated MP3 track.
- If `HF_TOKEN` is unavailable, generation fails with a clear setup message.
- If a language-specific Modal TTS endpoint is unavailable or misconfigured, the app fails with a clear Gradio error instead of silently switching languages.
- On macOS, the app can still fall back to a matching local `say` voice when a Modal request fails and a compatible system voice is installed for that target language.

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
---

# LingoShadow - Daily Language Practice

`LingoShadow - Daily Language Practice` is a standalone Gradio project for building a personalized self-study pack from real daily-life situations. The current v1 is intentionally scoped to **French-only audio** so the speech quality can be materially better than the earlier English-voice workaround.

## Hackathon-safe model stack

- Generation: `Qwen/Qwen2.5-7B-Instruct` with `7,615,616,512` parameters
- TTS: `kyutai/tts-1.6b-en_fr` with approximately `1.8B` parameters, served through Modal
- Total model budget: approximately `9,415,616,512` parameters

This keeps the app comfortably inside the Build Small hackathon `<= 32B` total-parameter limit while moving French TTS to a model that is explicitly built for English and French.

## What it does

- asks the learner to describe their general use cases and daily routines
- generates practical French sentences that are likely to come up often
- prioritizes useful verbs and recurring situations
- returns a simple 45-minute daily study routine
- highlights the core verbs the learner should review first
- generates downloadable French audio tracks through a dedicated Modal-backed TTS service
- groups audio into files of up to `20` sentences per WAV track plus a downloadable ZIP bundle

## Current scope

- French is the only supported target language in v1
- source-language prompts can still be generated from English, French, Spanish, German, or Portuguese
- the app does not auto-detect language for TTS
- voice selection is fixed internally to a curated French voice embedding from `kyutai/tts-voices`

## Environment variables

This app loads environment variables in this order:

1. `daily-language-practice/.env`
2. `/Users/Kwadwo/Documents/PROJECTS/NITA-bill-review/.env`

App-side variables:

- `HF_TOKEN`: required for Qwen sentence generation
- `MODAL_TTS_BASE_URL`: required for the Modal TTS endpoint base URL
- `MODAL_TTS_AUTH_TOKEN`: bearer token sent to the Modal TTS service
- `MODAL_TTS_TIMEOUT_SECONDS`: optional request timeout override for TTS calls

Modal-side variables:

- `MODAL_TTS_AUTH_TOKEN`: optional bearer token expected by the Modal service

This repo currently expects that Modal secret to be attached from a secret named `language-learning-bearer`.

## Run locally

```bash
uv venv .venv
source .venv/bin/activate
uv sync
uv run python app.py
```

## Deploy the Modal TTS service

The repo includes [`modal_tts_service.py`](/Users/Kwadwo/Documents/PROJECTS/HF-Build-Small/daily-language-practice/modal_tts_service.py), which serves `kyutai/tts-1.6b-en_fr` behind:

- `GET /healthz`
- `POST /synthesize-track`

Deploy it with:

```bash
uv run --with modal modal deploy modal_tts_service.py
```

The deployed function attaches the Modal secret named `language-learning-bearer`, which must contain `MODAL_TTS_AUTH_TOKEN`.

The request body for `POST /synthesize-track` is:

```json
{
  "sentences": ["Bonjour.", "Comment allez-vous ?"],
  "slow_audio": false
}
```

The response is raw `audio/wav` bytes for the concatenated track.

## Notes

- The generated ZIP includes JSON, CSV, a text summary, a daily routine file, a focus-verbs file, and the WAV tracks.
- Preview audio uses the first generated WAV track.
- If `HF_TOKEN` is unavailable, generation fails with a clear setup message.
- If the Modal TTS endpoint is unavailable or misconfigured, the app fails with a clear Gradio error instead of silently falling back to the old Hugging Face routed Kokoro path.

# Daily Language Practice

`Daily Language Practice` is a standalone Gradio project for building a personalized self-study pack from your real daily-life situations. It combines high-frequency sentences, verb-focused coverage, a simple 45-minute routine, and downloadable target-language audio tracks.

## Hackathon-safe model stack

- Generation: `Qwen/Qwen2.5-7B-Instruct` with `7,615,616,512` parameters
- TTS: `hexgrad/Kokoro-82M` with `82,000,000` parameters
- Total model budget: `7,697,616,512` parameters

This keeps the app comfortably inside the Build Small hackathon `<= 32B` total-parameter limit while preserving good text quality and much better audio quality than the previous `gTTS` prototype path.

## What it does

- asks the learner to describe their general use cases and daily routines
- chooses practical sentences that are likely to come up often
- prioritizes useful verbs and recurring situations
- returns a simple 45-minute daily study routine
- highlights the core verbs the learner should review first
- translates the pack into the target language
- generates audio tracks with up to 20 sentences per MP3 plus a downloadable ZIP bundle

## What changed after reading the transcripts

After transcript retrieval started working, the app was refined around the repeated ideas in the reference videos:

- personalized materials are more useful than generic beginner lists
- the learner should study the words and sentences they are actually going to use
- verbs deserve special emphasis because they drive comprehension
- listening and speaking practice should be built into the routine, not left as an afterthought
- a lightweight daily routine is more actionable than a vague pile of examples

The current build turns those ideas into:

- a verb-prioritized sentence generator
- a transcript-inspired 45-minute study routine
- downloadable audio tracks designed for repeated listening and shadowing
- study-pack files that include the routine, focus verbs, CSV, JSON, and audio

## API and env handling

This app loads environment variables in this order:

1. `daily-language-practice/.env`
2. `/Users/Kwadwo/Documents/PROJECTS/NITA-bill-review/.env`

The current build uses `HF_TOKEN` for both generation and audio. It loads environment variables in this order, then calls Hugging Face Inference directly for the two audited model IDs above.

## Run locally

```bash
uv venv .venv
source .venv/bin/activate
uv sync
uv run python app.py
```

## Notes

- Default target language: `French`
- The generated ZIP includes JSON, CSV, a text summary, a daily routine file, a focus-verbs file, and the MP3 tracks.
- The app groups audio into files of up to `20` sentences each.
- If `HF_TOKEN` is unavailable, the app fails with a clear setup message rather than silently inventing output.

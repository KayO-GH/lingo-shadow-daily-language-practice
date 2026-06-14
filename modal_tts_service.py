"""Modal ASGI app for language-routed TTS deployments."""

from __future__ import annotations

import io
import os
import subprocess
import wave

import modal

APP_NAME = os.getenv("MODAL_TTS_APP_NAME", "daily-language-practice-tts")
HF_CACHE_DIR = "/cache/huggingface"
MODAL_AUTH_SECRET_NAME = os.getenv("MODAL_TTS_SECRET_NAME", "language-learning-bearer")
KYUTAI_TTS_MODEL = "kyutai/tts-1.6b-en_fr"
KYUTAI_TTS_VOICE_REPO = "kyutai/tts-voices"
KYUTAI_ENGLISH_VOICE = "unmute-prod-website/p329_022.wav"
KYUTAI_FRENCH_VOICE = "voice-donations/Hugo_the_frenchie_enhanced.wav"
KOKORO_TTS_MODEL = "hexgrad/Kokoro-82M"
MMS_GERMAN_TTS_MODEL = "facebook/mms-tts-deu"
KYUTAI_SAMPLE_RATE = 24_000
KOKORO_SAMPLE_RATE = 24_000
SENTENCE_PAUSE_SECONDS = 2.0
SLOW_AUDIO_SPEED_MULTIPLIER = 0.9

LANGUAGE_ROUTER: dict[str, dict[str, str]] = {
    "en": {
        "backend": "kyutai",
        "label": "English",
        "model": KYUTAI_TTS_MODEL,
        "voice": KYUTAI_ENGLISH_VOICE,
        "voice_repo": KYUTAI_TTS_VOICE_REPO,
    },
    "fr": {
        "backend": "kyutai",
        "label": "French",
        "model": KYUTAI_TTS_MODEL,
        "voice": KYUTAI_FRENCH_VOICE,
        "voice_repo": KYUTAI_TTS_VOICE_REPO,
    },
    "es": {
        "backend": "kokoro",
        "label": "Spanish",
        "model": KOKORO_TTS_MODEL,
        "voice": "ef_dora",
        "voice_repo": "",
        "lang_code": "e",
    },
    "de": {
        "backend": "mms",
        "label": "German",
        "model": MMS_GERMAN_TTS_MODEL,
        "voice": "checkpoint default",
        "voice_repo": "",
    },
    "it": {
        "backend": "kokoro",
        "label": "Italian",
        "model": KOKORO_TTS_MODEL,
        "voice": "if_sara",
        "voice_repo": "",
        "lang_code": "i",
    },
    "pt": {
        "backend": "kokoro",
        "label": "Portuguese",
        "model": KOKORO_TTS_MODEL,
        "voice": "pf_dora",
        "voice_repo": "",
        "lang_code": "p",
    },
    "ja": {
        "backend": "kokoro",
        "label": "Japanese",
        "model": KOKORO_TTS_MODEL,
        "voice": "jf_alpha",
        "voice_repo": "",
        "lang_code": "j",
    },
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"HF_HOME": HF_CACHE_DIR})
    .apt_install("ffmpeg", "espeak-ng")
    .pip_install(
        "click==8.1.8",
        "fastapi==0.115.13",
        "kokoro==0.9.4",
        "misaki[en,ja]",
        "moshi==0.2.11",
        "numpy==2.2.6",
        "pydantic==2.11.7",
        "sentencepiece==0.2.0",
        "torch==2.7.1",
        "transformers==4.52.4",
    )
    .run_commands("python -m unidic download")
)
cache_volume = modal.Volume.from_name("daily-language-practice-tts-cache", create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def _pcm_to_wav_bytes(pcm, sample_rate: int) -> bytes:
    import numpy as np

    clipped = np.clip(pcm, -1.0, 1.0)
    pcm_i16 = (clipped * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_i16.tobytes())
    return buffer.getvalue()


def _apply_tempo_filter(wav_bytes: bytes, speed_multiplier: float) -> bytes:
    if speed_multiplier <= 0:
        raise ValueError("speed_multiplier must be positive.")
    if speed_multiplier == 1.0:
        return wav_bytes

    result = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-filter:a",
            f"atempo={speed_multiplier}",
            "-f",
            "wav",
            "pipe:1",
        ],
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


def _encode_mp3_bytes(wav_bytes: bytes) -> bytes:
    result = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "128k",
            "-f",
            "mp3",
            "pipe:1",
        ],
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


@app.function(
    gpu="L40S",
    timeout=900,
    scaledown_window=300,
    secrets=[modal.Secret.from_name(MODAL_AUTH_SECRET_NAME)],
    volumes={"/cache": cache_volume},
)
@modal.asgi_app()
def fastapi_app():
    import numpy as np
    import torch
    from fastapi import Body, FastAPI, Header, HTTPException, Response
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import TTSModel
    from transformers import AutoTokenizer, VitsModel

    state: dict[str, object] = {"backends": {}}
    web_app = FastAPI(title="LingoShadow - Daily Language Practice TTS")

    def get_language_config(language_code: str) -> dict[str, str]:
        config = LANGUAGE_ROUTER.get(language_code)
        if config is None:
            supported = ", ".join(sorted(LANGUAGE_ROUTER))
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported `language`. Supported values: {supported}.",
            )
        return config

    def load_kyutai_backend(config: dict[str, str]) -> dict[str, object]:
        checkpoint_info = CheckpointInfo.from_hf_repo(config["model"])
        tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info,
            n_q=32,
            temp=0.6,
            device="cuda",
        )
        voice_path = tts_model.get_voice_path(config["voice"])
        condition_attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)
        return {
            "kind": "kyutai",
            "model": tts_model,
            "condition_attributes": condition_attributes,
            "sample_rate": KYUTAI_SAMPLE_RATE,
        }

    def load_mms_backend(config: dict[str, str]) -> dict[str, object]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(config["model"])
        model = VitsModel.from_pretrained(config["model"])
        model.to(device)
        return {
            "kind": "mms",
            "device": device,
            "model": model,
            "tokenizer": tokenizer,
            "sample_rate": int(model.config.sampling_rate),
        }

    def load_kokoro_backend(config: dict[str, str]) -> dict[str, object]:
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code=config["lang_code"])
        return {
            "kind": "kokoro",
            "pipeline": pipeline,
            "voice": config["voice"],
            "sample_rate": KOKORO_SAMPLE_RATE,
        }

    def ensure_backend(language_code: str) -> dict[str, object]:
        backends = state["backends"]
        assert isinstance(backends, dict)
        cached = backends.get(language_code)
        if isinstance(cached, dict):
            return cached

        config = get_language_config(language_code)
        backend_kind = config["backend"]
        if backend_kind == "kyutai":
            backend = load_kyutai_backend(config)
        elif backend_kind == "mms":
            backend = load_mms_backend(config)
        elif backend_kind == "kokoro":
            backend = load_kokoro_backend(config)
        else:
            raise RuntimeError(f"Unsupported backend kind: {backend_kind}")

        backends[language_code] = backend
        return backend

    def synthesize_with_kyutai(text: str, backend: dict[str, object]) -> np.ndarray:
        tts_model = backend["model"]
        condition_attributes = backend["condition_attributes"]
        assert isinstance(tts_model, TTSModel)

        entries = tts_model.prepare_script([text], padding_between=1)
        result = tts_model.generate(
            [entries],
            [condition_attributes],
            on_frame=lambda _frame: None,
        )

        with tts_model.mimi.streaming(1), torch.no_grad():
            pcm_chunks: list[np.ndarray] = []
            for frame in result.frames[tts_model.delay_steps :]:
                pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcm_chunks.append(np.clip(pcm[0, 0], -1.0, 1.0))

        if not pcm_chunks:
            raise RuntimeError("Kyutai returned no audio frames.")

        return np.concatenate(pcm_chunks, axis=-1)

    def synthesize_with_mms(text: str, backend: dict[str, object]) -> np.ndarray:
        tokenizer = backend["tokenizer"]
        model = backend["model"]
        device = backend["device"]
        assert isinstance(model, VitsModel)
        assert isinstance(device, str)

        inputs = tokenizer(text, return_tensors="pt")
        if device == "cuda":
            inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            waveform = model(**inputs).waveform.squeeze(0).cpu().numpy()
        return np.asarray(waveform, dtype=np.float32)

    def synthesize_with_kokoro(text: str, backend: dict[str, object], slow_audio: bool) -> np.ndarray:
        pipeline = backend["pipeline"]
        voice = backend["voice"]
        assert isinstance(voice, str)

        speed = SLOW_AUDIO_SPEED_MULTIPLIER if slow_audio else 1.0
        pcm_chunks: list[np.ndarray] = []
        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            pcm_chunks.append(np.asarray(audio, dtype=np.float32))

        if not pcm_chunks:
            raise RuntimeError("Kokoro returned no audio frames.")

        return np.concatenate(pcm_chunks, axis=-1)

    def synthesize_sentence(text: str, language_code: str, slow_audio: bool) -> tuple[np.ndarray, int]:
        backend = ensure_backend(language_code)
        sample_rate = backend["sample_rate"]
        assert isinstance(sample_rate, int)

        kind = backend["kind"]
        assert isinstance(kind, str)
        if kind == "kyutai":
            pcm = synthesize_with_kyutai(text, backend)
        elif kind == "mms":
            pcm = synthesize_with_mms(text, backend)
        elif kind == "kokoro":
            pcm = synthesize_with_kokoro(text, backend, slow_audio=slow_audio)
        else:
            raise RuntimeError(f"Unsupported loaded backend kind: {kind}")

        return pcm, sample_rate

    @web_app.on_event("startup")
    async def startup() -> None:
        os.environ.setdefault("HF_HOME", HF_CACHE_DIR)

    @web_app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "supported_languages": {
                code: {
                    "target_language": config["label"],
                    "backend": config["backend"],
                    "model": config["model"],
                    "voice": config["voice"],
                    "voice_repo": config["voice_repo"],
                }
                for code, config in LANGUAGE_ROUTER.items()
            },
        }

    @web_app.post("/synthesize-track")
    async def synthesize_track(
        payload: dict[str, object] = Body(...),
        authorization: str | None = Header(default=None),
    ) -> Response:
        expected_token = os.getenv("MODAL_TTS_AUTH_TOKEN", "").strip()
        if expected_token and authorization != f"Bearer {expected_token}":
            raise HTTPException(status_code=401, detail="Unauthorized.")

        sentences_value = payload.get("sentences")
        if not isinstance(sentences_value, list):
            raise HTTPException(status_code=422, detail="`sentences` must be a JSON array of strings.")

        language_value = payload.get("language")
        if not isinstance(language_value, str) or not language_value.strip():
            raise HTTPException(status_code=422, detail="`language` must be a non-empty string.")
        language_code = language_value.strip().casefold()
        config = get_language_config(language_code)

        slow_audio_value = payload.get("slow_audio", False)
        if not isinstance(slow_audio_value, bool):
            raise HTTPException(status_code=422, detail="`slow_audio` must be a boolean.")

        cleaned_sentences = [str(sentence).strip() for sentence in sentences_value if str(sentence).strip()]
        if not cleaned_sentences:
            raise HTTPException(status_code=400, detail="At least one non-empty sentence is required.")

        try:
            tracks: list[np.ndarray] = []
            track_sample_rate: int | None = None

            for index, sentence in enumerate(cleaned_sentences):
                pcm, sample_rate = synthesize_sentence(
                    sentence,
                    language_code=language_code,
                    slow_audio=slow_audio_value,
                )
                if track_sample_rate is None:
                    track_sample_rate = sample_rate
                elif track_sample_rate != sample_rate:
                    raise RuntimeError(
                        "TTS backend returned inconsistent sample rates across the same track."
                    )

                tracks.append(pcm)
                if index < len(cleaned_sentences) - 1:
                    pause = np.zeros(int(sample_rate * SENTENCE_PAUSE_SECONDS), dtype=np.float32)
                    tracks.append(pause)

            if track_sample_rate is None:
                raise RuntimeError("No audio was produced.")

            pcm = np.concatenate(tracks, axis=-1)
            wav_bytes = _pcm_to_wav_bytes(pcm, track_sample_rate)
            if slow_audio_value and config["backend"] != "kokoro":
                wav_bytes = _apply_tempo_filter(wav_bytes, SLOW_AUDIO_SPEED_MULTIPLIER)
            audio_bytes = _encode_mp3_bytes(wav_bytes)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}") from exc

        return Response(content=audio_bytes, media_type="audio/mpeg")

    return web_app

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
AUDIO_PARALLEL_WORKERS = 3
WORKER_FUNCTION_KWARGS = {
    "gpu": "L40S",
    "timeout": 900,
    "scaledown_window": 300,
    "max_containers": AUDIO_PARALLEL_WORKERS,
    "secrets": [modal.Secret.from_name(MODAL_AUTH_SECRET_NAME)],
    "volumes": {"/cache": cache_volume},
}
WEB_FUNCTION_KWARGS = {
    "timeout": 900,
    "scaledown_window": 300,
    "secrets": [modal.Secret.from_name(MODAL_AUTH_SECRET_NAME)],
    "volumes": {"/cache": cache_volume},
}
_BACKEND_CACHE: dict[str, dict[str, object]] = {}


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


def _wav_bytes_to_pcm(wav_bytes: bytes) -> tuple[object, int]:
    import numpy as np

    buffer = io.BytesIO(wav_bytes)
    with wave.open(buffer, "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise RuntimeError("Expected mono 16-bit WAV audio from worker synthesis.")
        sample_rate = wav_file.getframerate()
        pcm_i16 = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    return pcm_i16.astype(np.float32) / 32767.0, sample_rate


def _get_language_config(language_code: str) -> dict[str, str]:
    config = LANGUAGE_ROUTER.get(language_code)
    if config is None:
        supported = ", ".join(sorted(LANGUAGE_ROUTER))
        raise ValueError(f"Unsupported `language`. Supported values: {supported}.")
    return config


def _load_kyutai_backend(config: dict[str, str]) -> dict[str, object]:
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import TTSModel

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


def _load_mms_backend(config: dict[str, str]) -> dict[str, object]:
    import torch
    from transformers import AutoTokenizer, VitsModel

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


def _load_kokoro_backend(config: dict[str, str]) -> dict[str, object]:
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=config["lang_code"])
    return {
        "kind": "kokoro",
        "pipeline": pipeline,
        "voice": config["voice"],
        "sample_rate": KOKORO_SAMPLE_RATE,
    }


def _ensure_backend(language_code: str) -> dict[str, object]:
    cached = _BACKEND_CACHE.get(language_code)
    if isinstance(cached, dict):
        return cached

    config = _get_language_config(language_code)
    backend_kind = config["backend"]
    if backend_kind == "kyutai":
        backend = _load_kyutai_backend(config)
    elif backend_kind == "mms":
        backend = _load_mms_backend(config)
    elif backend_kind == "kokoro":
        backend = _load_kokoro_backend(config)
    else:
        raise RuntimeError(f"Unsupported backend kind: {backend_kind}")

    _BACKEND_CACHE[language_code] = backend
    return backend


def _synthesize_with_kyutai(text: str, backend: dict[str, object]) -> object:
    import numpy as np
    import torch

    tts_model = backend["model"]
    condition_attributes = backend["condition_attributes"]
    entries = tts_model.prepare_script([text], padding_between=1)
    result = tts_model.generate(
        [entries],
        [condition_attributes],
        on_frame=lambda _frame: None,
    )

    with tts_model.mimi.streaming(1), torch.no_grad():
        pcm_chunks: list[object] = []
        for frame in result.frames[tts_model.delay_steps :]:
            pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
            pcm_chunks.append(np.clip(pcm[0, 0], -1.0, 1.0))

    if not pcm_chunks:
        raise RuntimeError("Kyutai returned no audio frames.")

    return np.concatenate(pcm_chunks, axis=-1)


def _synthesize_with_mms(text: str, backend: dict[str, object]) -> object:
    import numpy as np
    import torch

    tokenizer = backend["tokenizer"]
    model = backend["model"]
    device = backend["device"]
    inputs = tokenizer(text, return_tensors="pt")
    if device == "cuda":
        inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        waveform = model(**inputs).waveform.squeeze(0).cpu().numpy()
    return np.asarray(waveform, dtype=np.float32)


def _synthesize_with_kokoro(text: str, backend: dict[str, object], slow_audio: bool) -> object:
    import numpy as np

    pipeline = backend["pipeline"]
    voice = backend["voice"]
    speed = SLOW_AUDIO_SPEED_MULTIPLIER if slow_audio else 1.0
    pcm_chunks: list[object] = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        pcm_chunks.append(np.asarray(audio, dtype=np.float32))

    if not pcm_chunks:
        raise RuntimeError("Kokoro returned no audio frames.")

    return np.concatenate(pcm_chunks, axis=-1)


def _synthesize_sentence(text: str, language_code: str, slow_audio: bool) -> tuple[object, int]:
    backend = _ensure_backend(language_code)
    sample_rate = backend["sample_rate"]
    kind = backend["kind"]
    if kind == "kyutai":
        pcm = _synthesize_with_kyutai(text, backend)
    elif kind == "mms":
        pcm = _synthesize_with_mms(text, backend)
    elif kind == "kokoro":
        pcm = _synthesize_with_kokoro(text, backend, slow_audio=slow_audio)
    else:
        raise RuntimeError(f"Unsupported loaded backend kind: {kind}")

    return pcm, int(sample_rate)


def _synthesize_group_wav(sentences: list[str], language_code: str, slow_audio: bool) -> tuple[bytes, int]:
    import numpy as np

    cleaned_sentences = [str(sentence).strip() for sentence in sentences if str(sentence).strip()]
    if not cleaned_sentences:
        raise ValueError("At least one non-empty sentence is required for TTS synthesis.")

    tracks: list[object] = []
    track_sample_rate: int | None = None
    for index, sentence in enumerate(cleaned_sentences):
        pcm, sample_rate = _synthesize_sentence(sentence, language_code=language_code, slow_audio=slow_audio)
        if track_sample_rate is None:
            track_sample_rate = sample_rate
        elif track_sample_rate != sample_rate:
            raise RuntimeError("TTS backend returned inconsistent sample rates across the same track.")

        tracks.append(pcm)
        if index < len(cleaned_sentences) - 1:
            pause = np.zeros(int(sample_rate * SENTENCE_PAUSE_SECONDS), dtype=np.float32)
            tracks.append(pause)

    if track_sample_rate is None:
        raise RuntimeError("No audio was produced.")

    return _pcm_to_wav_bytes(np.concatenate(tracks, axis=-1), track_sample_rate), track_sample_rate


def _build_worker_metadata(language_code: str) -> dict[str, object]:
    config = _get_language_config(language_code)
    backend = _ensure_backend(language_code)
    return {
        "language": language_code,
        "target_language": config["label"],
        "backend": config["backend"],
        "model": config["model"],
        "voice": config["voice"],
        "voice_repo": config["voice_repo"],
        "sample_rate": int(backend["sample_rate"]),
    }


def _split_sentences_for_workers(sentences: list[str], worker_count: int = AUDIO_PARALLEL_WORKERS) -> list[list[str]]:
    cleaned_sentences = [sentence for sentence in sentences if str(sentence).strip()]
    if not cleaned_sentences:
        return []

    active_workers = min(max(1, worker_count), len(cleaned_sentences))
    base_size, remainder = divmod(len(cleaned_sentences), active_workers)
    groups: list[list[str]] = []
    start_index = 0
    for worker_index in range(active_workers):
        group_size = base_size + (1 if worker_index < remainder else 0)
        end_index = start_index + group_size
        groups.append(cleaned_sentences[start_index:end_index])
        start_index = end_index
    return groups


def _assemble_parallel_wavs(worker_results: list[dict[str, object]]) -> bytes:
    import numpy as np

    if not worker_results:
        raise RuntimeError("No audio was produced.")

    tracks: list[object] = []
    track_sample_rate: int | None = None
    for index, result in enumerate(worker_results):
        wav_bytes = result["wav_bytes"]
        sample_rate = int(result["sample_rate"])
        if not isinstance(wav_bytes, bytes):
            raise RuntimeError("Worker synthesis returned an invalid audio payload.")

        pcm, decoded_sample_rate = _wav_bytes_to_pcm(wav_bytes)
        if decoded_sample_rate != sample_rate:
            raise RuntimeError("Worker synthesis returned mismatched sample rate metadata.")

        if track_sample_rate is None:
            track_sample_rate = sample_rate
        elif track_sample_rate != sample_rate:
            raise RuntimeError("TTS backend returned inconsistent sample rates across the same track.")

        tracks.append(pcm)
        if index < len(worker_results) - 1:
            pause = np.zeros(int(sample_rate * SENTENCE_PAUSE_SECONDS), dtype=np.float32)
            tracks.append(pause)

    if track_sample_rate is None:
        raise RuntimeError("No audio was produced.")

    return _pcm_to_wav_bytes(np.concatenate(tracks, axis=-1), track_sample_rate)


@app.function(**WORKER_FUNCTION_KWARGS)
def synthesize_sentence_group(
    sentences: list[str],
    language_code: str,
    slow_audio: bool,
    warm_only: bool = False,
) -> dict[str, object]:
    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    metadata = _build_worker_metadata(language_code)
    if warm_only:
        metadata["warmed"] = True
        metadata["wav_bytes"] = b""
        return metadata

    wav_bytes, sample_rate = _synthesize_group_wav(sentences, language_code=language_code, slow_audio=slow_audio)
    metadata["sample_rate"] = sample_rate
    metadata["wav_bytes"] = wav_bytes
    return metadata


@app.function(**WEB_FUNCTION_KWARGS)
@modal.asgi_app()
def fastapi_app():
    from fastapi import Body, FastAPI, Header, HTTPException, Response

    web_app = FastAPI(title="LingoShadow - Daily Language Practice TTS")
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

    @web_app.post("/warmup")
    async def warmup(
        payload: dict[str, object] = Body(...),
        authorization: str | None = Header(default=None),
    ) -> dict[str, object]:
        expected_token = os.getenv("MODAL_TTS_AUTH_TOKEN", "").strip()
        if expected_token and authorization != f"Bearer {expected_token}":
            raise HTTPException(status_code=401, detail="Unauthorized.")

        language_value = payload.get("language")
        if not isinstance(language_value, str) or not language_value.strip():
            raise HTTPException(status_code=422, detail="`language` must be a non-empty string.")
        language_code = language_value.strip().casefold()

        try:
            _get_language_config(language_code)
            warm_results = [
                result
                async for result in synthesize_sentence_group.map.aio(
                    [[] for _ in range(AUDIO_PARALLEL_WORKERS)],
                    [language_code] * AUDIO_PARALLEL_WORKERS,
                    [False] * AUDIO_PARALLEL_WORKERS,
                    kwargs={"warm_only": True},
                )
            ]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"TTS warmup failed: {exc}") from exc

        warmed = dict(warm_results[0]) if warm_results else _build_worker_metadata(language_code)
        warmed.pop("wav_bytes", None)
        warmed["warmed_workers"] = len(warm_results)
        warmed["status"] = "warmed"
        return warmed

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
        try:
            config = _get_language_config(language_code)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        slow_audio_value = payload.get("slow_audio", False)
        if not isinstance(slow_audio_value, bool):
            raise HTTPException(status_code=422, detail="`slow_audio` must be a boolean.")

        cleaned_sentences = [str(sentence).strip() for sentence in sentences_value if str(sentence).strip()]
        if not cleaned_sentences:
            raise HTTPException(status_code=400, detail="At least one non-empty sentence is required.")

        try:
            sentence_groups = _split_sentences_for_workers(cleaned_sentences, AUDIO_PARALLEL_WORKERS)
            worker_results = [
                result
                async for result in synthesize_sentence_group.map.aio(
                    sentence_groups,
                    [language_code] * len(sentence_groups),
                    [slow_audio_value] * len(sentence_groups),
                    kwargs={"warm_only": False},
                )
            ]
            wav_bytes = _assemble_parallel_wavs(worker_results)
            if slow_audio_value and config["backend"] != "kokoro":
                wav_bytes = _apply_tempo_filter(wav_bytes, SLOW_AUDIO_SPEED_MULTIPLIER)
            audio_bytes = _encode_mp3_bytes(wav_bytes)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}") from exc

        return Response(content=audio_bytes, media_type="audio/mpeg")

    return web_app

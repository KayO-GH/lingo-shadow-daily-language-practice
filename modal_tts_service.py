"""Modal ASGI app for French-only Kyutai TTS."""

import io
import os
import wave

import modal

APP_NAME = "daily-language-practice-tts"
HF_CACHE_DIR = "/cache/huggingface"
MODAL_AUTH_SECRET_NAME = "language-learning-bearer"
KYUTAI_TTS_MODEL = "kyutai/tts-1.6b-en_fr"
KYUTAI_TTS_VOICE_REPO = "kyutai/tts-voices"
KYUTAI_TTS_VOICE = "voice-donations/Hugo_the_frenchie_enhanced.wav"
KYUTAI_SAMPLE_RATE = 24_000
SENTENCE_PAUSE_SECONDS = 2.0

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"HF_HOME": HF_CACHE_DIR})
    .pip_install(
        "fastapi==0.115.13",
        "moshi==0.2.11",
        "numpy==2.2.6",
        "pydantic==2.11.7",
        "torch==2.7.1",
    )
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

    state: dict[str, object] = {}
    web_app = FastAPI(title="Daily Language Practice TTS")

    def synthesize_sentence(text: str, slow_audio: bool) -> np.ndarray:
        tts_model: TTSModel = state["tts_model"]  # type: ignore[assignment]
        condition_attributes = state["condition_attributes"]

        # Kyutai does not expose a direct speech-rate knob here, so slower mode
        # uses more generous script padding and longer inter-sentence pauses.
        entries = tts_model.prepare_script([text], padding_between=2 if slow_audio else 1)
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

    @web_app.on_event("startup")
    async def startup() -> None:
        os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
        checkpoint_info = CheckpointInfo.from_hf_repo(KYUTAI_TTS_MODEL)
        tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info,
            n_q=32,
            temp=0.6,
            device="cuda",
        )
        voice_path = tts_model.get_voice_path(KYUTAI_TTS_VOICE)
        condition_attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)

        state["tts_model"] = tts_model
        state["condition_attributes"] = condition_attributes
        state["voice_path"] = voice_path

    @web_app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "model": KYUTAI_TTS_MODEL,
            "voice": KYUTAI_TTS_VOICE,
            "voice_repo": KYUTAI_TTS_VOICE_REPO,
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

        slow_audio_value = payload.get("slow_audio", False)
        if not isinstance(slow_audio_value, bool):
            raise HTTPException(status_code=422, detail="`slow_audio` must be a boolean.")

        cleaned_sentences = [str(sentence).strip() for sentence in sentences_value if str(sentence).strip()]
        if not cleaned_sentences:
            raise HTTPException(status_code=400, detail="At least one non-empty sentence is required.")

        try:
            tracks: list[np.ndarray] = []
            pause = np.zeros(int(KYUTAI_SAMPLE_RATE * SENTENCE_PAUSE_SECONDS), dtype=np.float32)

            for index, sentence in enumerate(cleaned_sentences):
                tracks.append(synthesize_sentence(sentence, slow_audio=slow_audio_value))
                if index < len(cleaned_sentences) - 1:
                    tracks.append(pause)

            pcm = np.concatenate(tracks, axis=-1)
            wav_bytes = _pcm_to_wav_bytes(pcm, KYUTAI_SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}") from exc

        return Response(content=wav_bytes, media_type="audio/wav")

    return web_app

from __future__ import annotations

import importlib
import inspect
import sys
from types import ModuleType


class _FakeImage:
    def __init__(self, python_version: str | None = None) -> None:
        self.python_version = python_version
        self.operations: list[tuple[str, object]] = []

    @classmethod
    def debian_slim(cls, python_version: str) -> "_FakeImage":
        return cls(python_version=python_version)

    def env(self, values: dict[str, str]) -> "_FakeImage":
        self.operations.append(("env", values))
        return self

    def apt_install(self, *packages: str) -> "_FakeImage":
        self.operations.append(("apt_install", packages))
        return self

    def pip_install(self, *packages: str) -> "_FakeImage":
        self.operations.append(("pip_install", packages))
        return self

    def run_commands(self, *commands: str) -> "_FakeImage":
        self.operations.append(("run_commands", commands))
        return self


class _FakeApp:
    def __init__(self, name: str, image: _FakeImage | None = None) -> None:
        self.name = name
        self.image = image

    def function(self, **_kwargs):
        def decorator(fn):
            return fn

        return decorator


class _FakeVolume:
    @staticmethod
    def from_name(name: str, create_if_missing: bool = False) -> dict[str, object]:
        return {"name": name, "create_if_missing": create_if_missing}


class _FakeSecret:
    @staticmethod
    def from_name(name: str) -> dict[str, str]:
        return {"name": name}


def _load_module(monkeypatch) -> ModuleType:
    fake_modal = ModuleType("modal")
    fake_modal.Image = _FakeImage
    fake_modal.App = _FakeApp
    fake_modal.Secret = _FakeSecret
    fake_modal.Volume = _FakeVolume

    def asgi_app():
        def decorator(fn):
            return fn

        return decorator

    fake_modal.asgi_app = asgi_app

    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.delitem(sys.modules, "modal_tts_service", raising=False)
    return importlib.import_module("modal_tts_service")


def test_modal_tts_service_imports_without_kokoro_installed(monkeypatch) -> None:
    module = _load_module(monkeypatch)

    assert "KPipeline" not in module.__dict__
    assert "from kokoro import KPipeline" in inspect.getsource(module._load_kokoro_backend)


def test_modal_tts_service_image_installs_kokoro_runtime_requirements(monkeypatch) -> None:
    module = _load_module(monkeypatch)
    operations = module.image.operations

    pip_installs = [value for op, value in operations if op == "pip_install"]
    assert pip_installs == [
        (
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
    ]

    run_commands = [value for op, value in operations if op == "run_commands"]
    assert run_commands == [("python -m unidic download",)]


def test_modal_tts_service_emits_mp3_audio(monkeypatch) -> None:
    module = _load_module(monkeypatch)

    assert 'media_type="audio/mpeg"' in inspect.getsource(module.fastapi_app)


def test_modal_tts_service_exposes_warmup_endpoint(monkeypatch) -> None:
    module = _load_module(monkeypatch)

    assert '@web_app.post("/warmup")' in inspect.getsource(module.fastapi_app)


def test_modal_tts_service_limits_parallel_workers_to_three(monkeypatch) -> None:
    module = _load_module(monkeypatch)

    assert module.AUDIO_PARALLEL_WORKERS == 3
    assert module.WORKER_FUNCTION_KWARGS["max_containers"] == 3


def test_modal_tts_service_splits_sentences_into_three_contiguous_groups(monkeypatch) -> None:
    module = _load_module(monkeypatch)

    groups = module._split_sentences_for_workers(
        [f"Sentence {index}" for index in range(1, 11)],
        worker_count=3,
    )

    assert groups == [
        ["Sentence 1", "Sentence 2", "Sentence 3", "Sentence 4"],
        ["Sentence 5", "Sentence 6", "Sentence 7"],
        ["Sentence 8", "Sentence 9", "Sentence 10"],
    ]

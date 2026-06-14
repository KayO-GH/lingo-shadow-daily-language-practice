from __future__ import annotations

import app
from study_pack import DEFAULT_SENTENCE_COUNT, MAX_SENTENCE_COUNT, MIN_SENTENCE_COUNT


def test_run_pack_builder_returns_inline_error_state_for_generation_failures(monkeypatch) -> None:
    def fail_generation(**_: object):
        raise RuntimeError(
            "Only generated 6 unique sentences out of the requested 20. "
            "The language model returned incomplete output on some attempts. Try again or reduce the sentence count."
        )

    monkeypatch.setattr(app, "generate_sentence_cards", fail_generation)

    result = app.run_pack_builder(
        "I work from home, buy groceries in person, chat with neighbors, and handle simple errands in French.",
        "French",
        "English",
        20,
    )

    assert "Could not build the study pack" in result[0]
    assert "Try the build again." in result[1]
    assert result[4] == []
    assert result[5] is None
    assert result[6] is None
    assert result[7] == []


def test_run_pack_builder_rejects_out_of_range_sentence_count() -> None:
    result = app.run_pack_builder(
        "I work from home, buy groceries in person, chat with neighbors, and handle simple errands in French.",
        "French",
        "English",
        MAX_SENTENCE_COUNT + 1,
    )

    assert "Could not build the study pack" in result[0]
    assert f"between {MIN_SENTENCE_COUNT} and {MAX_SENTENCE_COUNT}" in result[0]


def test_stream_pack_builder_yields_visible_progress_before_generation(monkeypatch) -> None:
    generation_started = False
    progress_calls: list[tuple[float, str]] = []

    def fake_generation(**_: object):
        nonlocal generation_started
        generation_started = True
        raise RuntimeError("stop after progress")

    def fake_progress(value: float, desc: str) -> None:
        progress_calls.append((value, desc))

    monkeypatch.setattr(app, "generate_sentence_cards", fake_generation)

    stream = app.stream_pack_builder(
        "I work from home, buy groceries in person, chat with neighbors, and handle simple errands in French.",
        "French",
        "English",
        DEFAULT_SENTENCE_COUNT,
        progress=fake_progress,
    )

    first_update = next(stream)
    second_update = next(stream)

    assert generation_started is False
    assert "build-progress-card" in first_update[0]
    assert "Starting build" in first_update[0]
    assert "Generating sentence pack" in second_update[0]
    assert progress_calls == [
        (0.05, "Starting study pack build"),
        (0.35, "Generating and translating sentences"),
    ]


def test_stream_pack_builder_emits_success_notice_once_on_completion(monkeypatch) -> None:
    notices: list[tuple[str, dict[str, object]]] = []

    class FakeStep:
        minutes = 15
        title = "Shadow"
        instructions = "Repeat the target sentences aloud."

    class FakeCard:
        scenario = "Groceries"
        source_sentence = "I need fruit."
        target_sentence = "J'ai besoin de fruits."
        verb_lemma = "avoir besoin"

    class FakePlan:
        cards = [FakeCard()]
        assumptions = ["The learner shops in person."]
        rationale = "Focus on recurring errands first."
        routine_steps = [FakeStep()]
        focus_verbs = ["avoir besoin"]

    class FakeBundle:
        audio_paths = ["track-1.mp3"]
        tts_backend_label = "kyutai/tts-1.6b-en_fr"
        preview_audio_path = "preview.mp3"
        zip_path = "pack.zip"

    monkeypatch.setattr(app, "generate_sentence_cards", lambda **_: FakePlan())
    monkeypatch.setattr(app, "create_study_pack", lambda **_: FakeBundle())
    monkeypatch.setattr(app.gr, "Success", lambda message, **kwargs: notices.append((message, kwargs)))

    updates = list(
        app.stream_pack_builder(
            "I work from home, buy groceries in person, chat with neighbors, and handle simple errands in French.",
            "French",
            "English",
            DEFAULT_SENTENCE_COUNT,
            progress=lambda *_args, **_kwargs: None,
        )
    )

    assert len(updates) == 4
    assert notices == [
        (
            "Study pack ready for French.",
            {
                "duration": app.BUILD_SUCCESS_NOTICE_DURATION_SECONDS,
                "title": "Study pack ready",
            },
        )
    ]
    assert "Built **1** study sentences for **French**" in updates[-1][0]


def test_stream_pack_builder_does_not_emit_success_notice_on_runtime_failure(monkeypatch) -> None:
    notices: list[tuple[str, dict[str, object]]] = []

    def fail_generation(**_: object):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(app, "generate_sentence_cards", fail_generation)
    monkeypatch.setattr(app.gr, "Success", lambda message, **kwargs: notices.append((message, kwargs)))

    updates = list(
        app.stream_pack_builder(
            "I work from home, buy groceries in person, chat with neighbors, and handle simple errands in French.",
            "French",
            "English",
            DEFAULT_SENTENCE_COUNT,
            progress=lambda *_args, **_kwargs: None,
        )
    )

    assert notices == []
    assert "Could not build the study pack" in updates[-1][0]


def test_stream_pack_builder_does_not_emit_success_notice_on_validation_failure(monkeypatch) -> None:
    notices: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(app.gr, "Success", lambda message, **kwargs: notices.append((message, kwargs)))

    updates = list(
        app.stream_pack_builder(
            "I work from home, buy groceries in person, chat with neighbors, and handle simple errands in French.",
            "French",
            "English",
            MAX_SENTENCE_COUNT + 1,
            progress=lambda *_args, **_kwargs: None,
        )
    )

    assert notices == []
    assert "Could not build the study pack" in updates[-1][0]


def test_warmup_selected_language_ignores_failures(monkeypatch) -> None:
    def fail_warmup(_: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(app, "warmup_tts_backend", fail_warmup)

    assert app.warmup_selected_language("French") is None


def test_build_success_state_includes_runtime_models() -> None:
    class FakeStep:
        minutes = 15
        title = "Shadow"
        instructions = "Repeat the target sentences aloud."

    class FakeCard:
        scenario = "Groceries"
        source_sentence = "I need fruit."
        target_sentence = "J'ai besoin de fruits."
        verb_lemma = "avoir besoin"

    class FakePlan:
        cards = [FakeCard()]
        assumptions = ["The learner shops in person."]
        rationale = "Focus on recurring errands first."
        routine_steps = [FakeStep()]
        focus_verbs = ["avoir besoin"]

    class FakeBundle:
        audio_paths = ["track-1.mp3"]
        tts_backend_label = "kyutai/tts-1.6b-en_fr"
        preview_audio_path = "preview.mp3"
        zip_path = "pack.zip"

    result = app.build_success_state(FakePlan(), "French", FakeBundle())

    assert "Qwen/Qwen3-8B" in result[0]
    assert "CohereLabs/tiny-aya-global" in result[0]
    assert "kyutai/tts-1.6b-en_fr" in result[0]


def test_app_config_uses_new_sentence_count_range_and_registers_warmup_events() -> None:
    config = app.demo.get_config_file()
    sentence_slider = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("label") == "Sentence count"
    )

    assert sentence_slider["props"]["minimum"] == MIN_SENTENCE_COUNT
    assert sentence_slider["props"]["maximum"] == MAX_SENTENCE_COUNT
    assert sentence_slider["props"]["value"] == DEFAULT_SENTENCE_COUNT

    api_names = [dependency["api_name"] for dependency in config["dependencies"]]
    warmup_api_names = [name for name in api_names if isinstance(name, str) and name.startswith("warmup_selected_language")]
    assert len(warmup_api_names) == 2
    assert "toggle_motivation_panel" in api_names
    assert "stream_pack_builder" in api_names


def test_app_css_keeps_generated_result_markdown_readable() -> None:
    assert "#results-shell .prose h3" in app.APP_CSS
    assert "#results-shell .prose strong" in app.APP_CSS
    assert "#results-shell .prose li::marker" in app.APP_CSS
    assert "padding: 0.9rem 1rem !important;" in app.APP_CSS
    assert "#results-shell .prose > :first-child" in app.APP_CSS
    assert "color: #0f172a !important;" in app.APP_CSS

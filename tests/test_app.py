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


def test_warmup_selected_language_ignores_failures(monkeypatch) -> None:
    def fail_warmup(_: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(app, "warmup_tts_backend", fail_warmup)

    assert app.warmup_selected_language("French") is None


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
    assert "build_model_stack_md" in api_names

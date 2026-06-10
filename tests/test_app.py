from __future__ import annotations

import app


def test_run_pack_builder_returns_inline_error_state_for_generation_failures(monkeypatch) -> None:
    def fail_generation(**_: object):
        raise RuntimeError(
            "Only generated 6 unique sentences out of the requested 20. "
            "The language model returned incomplete output on some attempts. Try again or reduce the sentence count."
        )

    monkeypatch.setattr(app, "generate_sentence_cards", fail_generation)

    result = app.run_pack_builder(
        "I work from home, buy groceries in person, chat with neighbors, and handle simple errands in French.",
        "English",
        20,
    )

    assert "Could not build the study pack" in result[0]
    assert "Try the build again." in result[1]
    assert result[4] == []
    assert result[5] is None
    assert result[6] is None
    assert result[7] == []

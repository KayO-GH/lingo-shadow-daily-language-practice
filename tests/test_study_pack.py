from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import study_pack

from study_pack import (
    FRENCH_ONLY_ERROR,
    MODAL_TTS_MODEL,
    MODAL_TTS_VOICE,
    SENTENCES_PER_AUDIO_FILE,
    TARGET_LANGUAGE,
    TRANSLATION_MODEL,
    ModalTTSClient,
    SentenceCard,
    StudyRoutineStep,
    build_results_rows,
    create_study_pack,
    extract_json_payload,
    generate_sentence_cards,
    get_model_stack_summary,
    get_native_language_choices,
    get_supported_language_labels,
    normalize_plan,
    translate_sentence_cards,
)


def test_extract_json_payload_reads_fenced_json() -> None:
    raw = """Here you go.

```json
{"sentences": [{"source_sentence": "Hello", "target_sentence": "Bonjour"}]}
```
"""
    payload = extract_json_payload(raw)
    assert payload["sentences"][0]["target_sentence"] == "Bonjour"


def test_supported_languages_are_french_only() -> None:
    assert get_supported_language_labels() == [TARGET_LANGUAGE]
    assert get_native_language_choices() == ["English", "French", "Spanish", "German", "Portuguese"]


def test_normalize_plan_dedupes_by_target_sentence_and_reads_routine() -> None:
    payload = {
        "rationale": "Use daily situations.",
        "assumptions": ["The learner wants portable phrases."],
        "focus_verbs": ["need", "go", "be", "be able to"],
        "study_routine": [
            {"title": "Preview", "minutes": 10, "instructions": "Scan verbs."},
            {"title": "Listen", "minutes": 20, "instructions": "Shadow the audio."},
            {"title": "Speak", "minutes": 15, "instructions": "Recall from prompts."},
        ],
        "sentences": [
            {
                "scenario": "Shop",
                "source_sentence": "I need a bag.",
                "target_sentence": "J'ai besoin d'un sac.",
                "verb_lemma": "need",
                "why_it_is_useful": "Very common request.",
                "pronunciation_hint": "",
            },
            {
                "scenario": "Shop",
                "source_sentence": "I need a bag.",
                "target_sentence": "J'ai besoin d'un sac.",
                "verb_lemma": "to need",
                "why_it_is_useful": "Duplicate example.",
                "pronunciation_hint": "",
            },
            {
                "scenario": "Travel",
                "source_sentence": "Where are we going?",
                "target_sentence": "Ou allons-nous ?",
                "verb_lemma": "go",
                "why_it_is_useful": "Core travel verb.",
                "pronunciation_hint": "",
            },
            {
                "scenario": "Home",
                "source_sentence": "I am ready.",
                "target_sentence": "Je suis pret.",
                "verb_lemma": "be",
                "why_it_is_useful": "Basic state sentence.",
                "pronunciation_hint": "",
            },
            {
                "scenario": "Cafe",
                "source_sentence": "Can I pay now?",
                "target_sentence": "Je peux payer maintenant ?",
                "verb_lemma": "pay",
                "why_it_is_useful": "Payment and permission.",
                "pronunciation_hint": "",
            },
        ],
    }

    plan = normalize_plan(payload, sentence_count=4)
    assert plan.rationale == "Use daily situations."
    assert plan.assumptions == ["The learner wants portable phrases."]
    assert plan.focus_verbs == ["to need", "to go", "to be", "to be able to"]
    assert [step.minutes for step in plan.routine_steps] == [10, 20, 15]
    assert len(plan.cards) == 4
    assert [card.verb_lemma for card in plan.cards] == ["to need", "to go", "to be", "to pay"]


def test_modal_tts_client_posts_sentence_lists_and_slow_audio() -> None:
    captured: dict[str, object] = {}

    def fake_transport(url: str, payload: bytes, headers: dict[str, str], timeout: float) -> bytes:
        captured["url"] = url
        captured["payload"] = json.loads(payload.decode("utf-8"))
        captured["headers"] = headers
        captured["timeout"] = timeout
        return b"RIFFfakewav"

    client = ModalTTSClient(
        base_url="https://tts.example.com",
        auth_token="secret-token",
        timeout_seconds=45.0,
        transport=fake_transport,
    )
    audio = client.synthesize_track(["Bonjour.", "Comment allez-vous ?"], slow_audio=True)

    assert audio == b"RIFFfakewav"
    assert captured["url"] == "https://tts.example.com/synthesize-track"
    assert captured["payload"] == {
        "sentences": ["Bonjour.", "Comment allez-vous ?"],
        "slow_audio": True,
    }
    assert captured["headers"] == {
        "Accept": "audio/wav",
        "Content-Type": "application/json",
        "Authorization": "Bearer secret-token",
    }
    assert captured["timeout"] == 45.0


def test_create_study_pack_writes_bundle_and_zip(tmp_path: Path) -> None:
    cards = [
        SentenceCard(
            scenario="Groceries",
            source_sentence="I need apples.",
            target_sentence="J'ai besoin de pommes.",
            verb_lemma="to need",
            why_it_is_useful="Shopping staple.",
            pronunciation_hint="zhay buh-ZWAN duh pom",
        ),
        SentenceCard(
            scenario="Transit",
            source_sentence="Where does this bus go?",
            target_sentence="Ce bus va ou ?",
            verb_lemma="to go",
            why_it_is_useful="Travel question.",
            pronunciation_hint="suh boos va oo",
        ),
    ]

    calls: list[tuple[list[str], bool]] = []

    def fake_tts_writer(sentences: list[str], destination: Path, slow_audio: bool) -> None:
        calls.append((sentences, slow_audio))
        destination.write_bytes(b"RIFFfakewav")

    bundle = create_study_pack(
        cards=cards,
        target_language=TARGET_LANGUAGE,
        focus_verbs=["to need", "to go"],
        routine_steps=[
            StudyRoutineStep(title="Preview", minutes=10, instructions="Scan verbs."),
            StudyRoutineStep(title="Listen", minutes=20, instructions="Shadow audio."),
            StudyRoutineStep(title="Speak", minutes=15, instructions="Recall from prompts."),
        ],
        output_root=tmp_path,
        tts_writer=fake_tts_writer,
    )

    assert bundle.preview_audio_path.exists()
    assert bundle.preview_audio_path.suffix == ".wav"
    assert bundle.zip_path.exists()
    assert len(bundle.audio_paths) == 1
    assert (bundle.session_dir / "study_pack.csv").exists()
    assert (bundle.session_dir / "study_pack.json").exists()
    assert (bundle.session_dir / "daily_routine.md").exists()
    assert (bundle.session_dir / "focus_verbs.txt").exists()
    assert calls == [(["J'ai besoin de pommes.", "Ce bus va ou ?"], False)]

    payload = json.loads((bundle.session_dir / "study_pack.json").read_text(encoding="utf-8"))
    assert payload[0]["target_sentence"] == "J'ai besoin de pommes."
    summary_text = (bundle.session_dir / "README.txt").read_text(encoding="utf-8")
    assert MODAL_TTS_MODEL in summary_text
    assert MODAL_TTS_VOICE in summary_text


def test_create_study_pack_batches_every_twenty_sentences(tmp_path: Path) -> None:
    cards = [
        SentenceCard(
            scenario=f"Scenario {index}",
            source_sentence=f"Source sentence {index}",
            target_sentence=f"Target sentence {index}",
            verb_lemma=f"verb_{index}",
            why_it_is_useful="Useful daily sentence.",
        )
        for index in range(1, SENTENCES_PER_AUDIO_FILE + 2)
    ]

    calls: list[list[str]] = []

    def fake_tts_writer(sentences: list[str], destination: Path, slow_audio: bool) -> None:
        calls.append(sentences)
        destination.write_bytes(b"RIFFfakewav")

    bundle = create_study_pack(
        cards=cards,
        target_language=TARGET_LANGUAGE,
        output_root=tmp_path,
        tts_writer=fake_tts_writer,
    )

    assert len(bundle.audio_paths) == 2
    assert bundle.audio_paths[0].name == "01_sentences_01_20.wav"
    assert bundle.audio_paths[1].name == "02_sentences_21_21.wav"
    assert len(calls[0]) == SENTENCES_PER_AUDIO_FILE
    assert calls[1] == ["Target sentence 21"]


def test_create_study_pack_rejects_non_french_target_language(tmp_path: Path) -> None:
    cards = [
        SentenceCard(
            scenario="Transit",
            source_sentence="Where is the station?",
            target_sentence="Ou est la gare ?",
            verb_lemma="etre",
            why_it_is_useful="Travel question.",
        )
    ]

    with pytest.raises(ValueError, match=FRENCH_ONLY_ERROR):
        create_study_pack(cards=cards, target_language="Spanish", output_root=tmp_path)


def test_model_stack_summary_mentions_qwen_tiny_aya_and_kyutai() -> None:
    summary = get_model_stack_summary()
    assert "Qwen/Qwen3-8B" in summary
    assert "CohereLabs/tiny-aya-global" in summary
    assert "kyutai/tts-1.6b-en_fr" in summary


def test_build_results_rows_uses_four_columns_with_infinitive_verbs() -> None:
    rows = build_results_rows(
        [
            SentenceCard(
                scenario="Travel",
                source_sentence="I need to go now.",
                target_sentence="Je dois partir maintenant.",
                verb_lemma="to go",
                why_it_is_useful="Common daily verb.",
                pronunciation_hint="",
            )
        ]
    )

    assert rows == [["Travel", "I need to go now.", "Je dois partir maintenant.", "to go"]]


def test_generate_sentence_cards_rejects_non_french_target_language() -> None:
    with pytest.raises(ValueError, match=FRENCH_ONLY_ERROR):
        generate_sentence_cards(
            use_cases="I need Spanish for errands and daily conversations.",
            target_language="Spanish",
            native_language="English",
            sentence_count=20,
            client=SimpleNamespace(),
        )


def test_generate_sentence_cards_retries_until_requested_count(monkeypatch) -> None:
    first_batch = {
        "rationale": "Use daily situations.",
        "assumptions": ["The learner wants practical phrases."],
        "focus_verbs": ["aller", "payer", "demander"],
        "study_routine": [
            {"title": "Preview", "minutes": 10, "instructions": "Scan verbs."},
            {"title": "Listen", "minutes": 20, "instructions": "Shadow audio."},
            {"title": "Speak", "minutes": 15, "instructions": "Recall from prompts."},
        ],
        "sentences": [
            {
                "scenario": f"Scenario {index}",
                "source_sentence": f"Source {index}",
                "target_sentence": "Target 1" if index == 20 else f"Target {index}",
                "verb_lemma": f"verb_{index}",
                "why_it_is_useful": "Useful daily sentence.",
                "pronunciation_hint": "",
            }
            for index in range(1, 21)
        ],
    }
    top_up_batch = {
        "sentences": [
            {
                "scenario": f"Top up {index}",
                "source_sentence": f"Top source {index}",
                "target_sentence": f"Top target {index}",
                "verb_lemma": f"top_verb_{index}",
                "why_it_is_useful": "Top-up sentence.",
                "pronunciation_hint": "",
            }
            for index in range(1, 9)
        ]
    }

    class StubClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat_completion(self, **_: object) -> SimpleNamespace:
            self.calls += 1
            payload = first_batch if self.calls == 1 else top_up_batch
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
            )

    monkeypatch.setattr(study_pack, "HF_GENERATION_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.setattr(study_pack, "translate_sentence_cards", lambda cards, **_: cards)

    plan = generate_sentence_cards(
        use_cases="I need French for errands and daily conversations.",
        target_language=TARGET_LANGUAGE,
        native_language="English",
        sentence_count=20,
        client=StubClient(),
    )

    assert len(plan.cards) == 20
    assert plan.cards[-1].target_sentence == "Top target 1"


def test_generate_sentence_cards_reads_text_from_content_blocks(monkeypatch) -> None:
    payload = {
        "rationale": "Use daily situations.",
        "assumptions": ["The learner wants practical phrases."],
        "focus_verbs": ["aller", "payer", "demander", "prendre"],
        "study_routine": [
            {"title": "Preview", "minutes": 10, "instructions": "Scan verbs."},
            {"title": "Listen", "minutes": 20, "instructions": "Shadow audio."},
            {"title": "Speak", "minutes": 15, "instructions": "Recall from prompts."},
        ],
        "sentences": [
            {
                "scenario": f"Scenario {index}",
                "source_sentence": f"Source {index}",
                "target_sentence": f"Target {index}",
                "verb_lemma": f"verb_{index}",
                "why_it_is_useful": "Useful daily sentence.",
                "pronunciation_hint": "",
            }
            for index in range(1, 9)
        ],
    }

    class StubClient:
        def chat_completion(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=[{"type": "text", "text": json.dumps(payload)}]
                        )
                    )
                ]
            )

    monkeypatch.setattr(study_pack, "translate_sentence_cards", lambda cards, **_: cards)

    plan = generate_sentence_cards(
        use_cases="I need French for errands and daily conversations.",
        target_language=TARGET_LANGUAGE,
        native_language="English",
        sentence_count=8,
        client=StubClient(),
    )

    assert len(plan.cards) == 8
    assert plan.cards[0].target_sentence == "Target 1"


def test_generate_sentence_cards_retries_after_empty_text_output(monkeypatch) -> None:
    payload = {
        "rationale": "Use daily situations.",
        "assumptions": ["The learner wants practical phrases."],
        "focus_verbs": ["aller", "payer", "demander", "prendre"],
        "study_routine": [
            {"title": "Preview", "minutes": 10, "instructions": "Scan verbs."},
            {"title": "Listen", "minutes": 20, "instructions": "Shadow audio."},
            {"title": "Speak", "minutes": 15, "instructions": "Recall from prompts."},
        ],
        "sentences": [
            {
                "scenario": f"Scenario {index}",
                "source_sentence": f"Source {index}",
                "target_sentence": f"Target {index}",
                "verb_lemma": f"verb_{index}",
                "why_it_is_useful": "Useful daily sentence.",
                "pronunciation_hint": "",
            }
            for index in range(1, 9)
        ],
    }

    class StubClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat_completion(self, **_: object) -> SimpleNamespace:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
            )

    client = StubClient()
    monkeypatch.setattr(study_pack, "translate_sentence_cards", lambda cards, **_: cards)
    plan = generate_sentence_cards(
        use_cases="I need French for errands and daily conversations.",
        target_language=TARGET_LANGUAGE,
        native_language="English",
        sentence_count=8,
        client=client,
    )

    assert client.calls == 2
    assert len(plan.cards) == 8


def test_generate_sentence_cards_accepts_small_top_up_batch(monkeypatch) -> None:
    first_batch = {
        "rationale": "Use daily situations.",
        "assumptions": ["The learner wants practical phrases."],
        "focus_verbs": ["work", "buy", "ask"],
        "study_routine": [
            {"title": "Preview", "minutes": 10, "instructions": "Scan verbs."},
            {"title": "Listen", "minutes": 20, "instructions": "Shadow the audio."},
            {"title": "Speak", "minutes": 15, "instructions": "Recall from prompts."},
        ],
        "sentences": [
            {
                "scenario": f"Scenario {index}",
                "source_sentence": f"Source {index}",
                "target_sentence": f"Target {index}",
                "verb_lemma": f"verb_{index}",
                "why_it_is_useful": "Useful daily sentence.",
                "pronunciation_hint": "",
            }
            for index in range(1, 18)
        ],
    }
    top_up_batch = {
        "sentences": [
            {
                "scenario": f"Top up {index}",
                "source_sentence": f"Top source {index}",
                "target_sentence": f"Top target {index}",
                "verb_lemma": f"top_verb_{index}",
                "why_it_is_useful": "Top-up sentence.",
                "pronunciation_hint": "",
            }
            for index in range(1, 4)
        ]
    }

    class StubClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat_completion(self, **_: object) -> SimpleNamespace:
            self.calls += 1
            payload = first_batch if self.calls == 1 else top_up_batch
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
            )

    monkeypatch.setattr(study_pack, "HF_GENERATION_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.setattr(study_pack, "translate_sentence_cards", lambda cards, **_: cards)

    plan = generate_sentence_cards(
        use_cases="I need French for errands and daily conversations.",
        target_language=TARGET_LANGUAGE,
        native_language="English",
        sentence_count=20,
        client=StubClient(),
    )

    assert len(plan.cards) == 20
    assert plan.cards[-1].target_sentence == "Top target 3"


def test_translate_sentence_cards_uses_translation_model_output() -> None:
    cards = [
        SentenceCard(
            scenario="Travel",
            source_sentence="Where is the station?",
            target_sentence="placeholder",
            verb_lemma="to go",
            why_it_is_useful="Useful question.",
        ),
        SentenceCard(
            scenario="Cafe",
            source_sentence="I would like a coffee.",
            target_sentence="placeholder",
            verb_lemma="to order",
            why_it_is_useful="Useful order.",
        ),
    ]

    class StubClient:
        def chat_completion(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs["model"] == TRANSLATION_MODEL
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {"translations": ["Ou est la gare ?", "Je voudrais un cafe."]}
                            )
                        )
                    )
                ]
            )

    translated = translate_sentence_cards(
        cards=cards,
        target_language=TARGET_LANGUAGE,
        native_language="English",
        client=StubClient(),
    )

    assert [card.target_sentence for card in translated] == ["Ou est la gare ?", "Je voudrais un cafe."]
    assert [card.source_sentence for card in translated] == ["Where is the station?", "I would like a coffee."]

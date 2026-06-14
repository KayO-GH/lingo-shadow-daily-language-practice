from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import study_pack

from study_pack import (
    DEFAULT_SENTENCE_COUNT,
    MAX_SENTENCE_COUNT,
    MIN_SENTENCE_COUNT,
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
    get_tts_backend_config,
    normalize_plan,
    translate_sentence_cards,
    validate_sentence_count,
    warmup_tts_backend,
)


def test_extract_json_payload_reads_fenced_json() -> None:
    raw = """Here you go.

```json
{"sentences": [{"source_sentence": "Hello", "target_sentence": "Bonjour"}]}
```
"""
    payload = extract_json_payload(raw)
    assert payload["sentences"][0]["target_sentence"] == "Bonjour"


def test_supported_languages_are_multilingual() -> None:
    assert get_supported_language_labels() == [
        "English",
        "French",
        "Spanish",
        "German",
        "Italian",
        "Portuguese",
        "Japanese",
    ]
    assert get_native_language_choices() == [
        "English",
        "French",
        "Spanish",
        "German",
        "Portuguese",
        "Italian",
        "Japanese",
    ]


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
        return b"ID3fake-mp3"

    client = ModalTTSClient(
        base_url="https://tts.example.com",
        auth_token="secret-token",
        timeout_seconds=45.0,
        transport=fake_transport,
    )
    audio = client.synthesize_track(["Bonjour.", "Comment allez-vous ?"], slow_audio=True)

    assert audio == b"ID3fake-mp3"
    assert captured["url"] == "https://tts.example.com/synthesize-track"
    assert captured["payload"] == {
        "sentences": ["Bonjour.", "Comment allez-vous ?"],
        "language": "fr",
        "slow_audio": True,
    }
    assert captured["headers"] == {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "Authorization": "Bearer secret-token",
    }
    assert captured["timeout"] == 45.0


def test_modal_tts_client_posts_warmup_request() -> None:
    captured: dict[str, object] = {}

    def fake_json_transport(
        url: str,
        payload: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, object]:
        captured["url"] = url
        captured["payload"] = json.loads(payload.decode("utf-8"))
        captured["headers"] = headers
        captured["timeout"] = timeout
        return {"status": "warmed", "language": "fr", "warmed_workers": 3}

    client = ModalTTSClient(
        base_url="https://tts.example.com",
        auth_token="secret-token",
        timeout_seconds=45.0,
        json_transport=fake_json_transport,
    )

    response = client.warmup()

    assert response == {"status": "warmed", "language": "fr", "warmed_workers": 3}
    assert captured["url"] == "https://tts.example.com/warmup"
    assert captured["payload"] == {"language": "fr"}
    assert captured["headers"] == {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer secret-token",
    }
    assert captured["timeout"] == 45.0


def test_warmup_tts_backend_uses_modal_client(monkeypatch) -> None:
    captured: list[str] = []

    class StubClient:
        def warmup(self) -> dict[str, object]:
            captured.append("warmed")
            return {"status": "warmed"}

    monkeypatch.setattr(study_pack, "get_modal_tts_client", lambda target_language: StubClient())

    response = warmup_tts_backend("French")

    assert captured == ["warmed"]
    assert response == {"status": "warmed"}


def test_get_tts_backend_config_uses_language_specific_env(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_TTS_BASE_URL_ES", "https://spanish-tts.example.com")
    monkeypatch.setenv("MODAL_TTS_AUTH_TOKEN_ES", "spanish-token")
    monkeypatch.setenv("MODAL_TTS_MODEL_ES", "facebook/mms-tts-spa")
    monkeypatch.setenv("MODAL_TTS_VOICE_ES", "es-voice")
    monkeypatch.setenv("MODAL_TTS_PARAMS_ES", "123456789")

    backend = get_tts_backend_config("Spanish")

    assert backend.base_url == "https://spanish-tts.example.com"
    assert backend.auth_token == "spanish-token"
    assert backend.model_label == "facebook/mms-tts-spa"
    assert backend.voice_label == "es-voice"
    assert backend.params == 123456789


def test_get_tts_backend_config_uses_historical_defaults() -> None:
    spanish_backend = get_tts_backend_config("Spanish")
    german_backend = get_tts_backend_config("German")
    japanese_backend = get_tts_backend_config("Japanese")

    assert spanish_backend.model_label == "hexgrad/Kokoro-82M"
    assert spanish_backend.voice_label == "ef_dora"
    assert german_backend.model_label == "facebook/mms-tts-deu"
    assert german_backend.voice_label == "checkpoint default"
    assert japanese_backend.model_label == "hexgrad/Kokoro-82M"
    assert japanese_backend.voice_label == "jf_alpha"


def test_validate_sentence_count_accepts_values_within_new_range() -> None:
    assert validate_sentence_count(DEFAULT_SENTENCE_COUNT) == DEFAULT_SENTENCE_COUNT
    assert validate_sentence_count(MIN_SENTENCE_COUNT) == MIN_SENTENCE_COUNT
    assert validate_sentence_count(MAX_SENTENCE_COUNT) == MAX_SENTENCE_COUNT


def test_validate_sentence_count_rejects_values_outside_new_range() -> None:
    with pytest.raises(ValueError, match=f"{MIN_SENTENCE_COUNT} and {MAX_SENTENCE_COUNT}"):
        validate_sentence_count(MIN_SENTENCE_COUNT - 1)

    with pytest.raises(ValueError, match=f"{MIN_SENTENCE_COUNT} and {MAX_SENTENCE_COUNT}"):
        validate_sentence_count(MAX_SENTENCE_COUNT + 1)


def test_default_tts_writer_converts_legacy_wav_responses_to_mp3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    converted_inputs: list[bytes] = []

    def fake_convert(wav_bytes: bytes) -> bytes:
        converted_inputs.append(wav_bytes)
        return b"ID3converted-mp3"

    monkeypatch.setattr(study_pack, "_convert_wav_bytes_to_mp3", fake_convert)

    client = ModalTTSClient(
        base_url="https://tts.example.com",
        auth_token="",
        transport=lambda *_args: b"RIFF\x00\x00\x00\x00WAVElegacy",
    )
    destination = tmp_path / "legacy.mp3"

    backend_label = study_pack.default_tts_writer(
        ["Bonjour."],
        destination,
        slow_audio=False,
        target_language=TARGET_LANGUAGE,
        client=client,
    )

    assert converted_inputs == [b"RIFF\x00\x00\x00\x00WAVElegacy"]
    assert destination.read_bytes() == b"ID3converted-mp3"
    assert backend_label == f"Modal ({client.model_label})"


def test_create_study_pack_writes_bundle_and_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(study_pack, "build_artifact_timestamp", lambda: "20260614_213000")

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

    calls: list[tuple[list[str], bool, str]] = []

    def fake_tts_writer(
        sentences: list[str],
        destination: Path,
        slow_audio: bool,
        target_language: str,
    ) -> None:
        calls.append((sentences, slow_audio, target_language))
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
    assert bundle.preview_audio_path.suffix == ".mp3"
    assert bundle.preview_audio_path.name == "01_sentences_01_02_20260614_213000.mp3"
    assert bundle.zip_path.exists()
    assert bundle.zip_path.name == "daily_language_pack_20260614_213000.zip"
    assert len(bundle.audio_paths) == 1
    assert (bundle.session_dir / "study_pack.csv").exists()
    assert (bundle.session_dir / "study_pack.json").exists()
    assert (bundle.session_dir / "daily_routine.md").exists()
    assert (bundle.session_dir / "focus_verbs.txt").exists()
    assert calls == [(["J'ai besoin de pommes.", "Ce bus va ou ?"], False, TARGET_LANGUAGE)]

    payload = json.loads((bundle.session_dir / "study_pack.json").read_text(encoding="utf-8"))
    assert payload[0]["target_sentence"] == "J'ai besoin de pommes."
    summary_text = (bundle.session_dir / "README.txt").read_text(encoding="utf-8")
    assert MODAL_TTS_MODEL in summary_text
    assert MODAL_TTS_VOICE in summary_text


def test_create_study_pack_batches_every_twenty_sentences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(study_pack, "build_artifact_timestamp", lambda: "20260614_213500")

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

    calls: list[tuple[list[str], str]] = []

    def fake_tts_writer(
        sentences: list[str],
        destination: Path,
        slow_audio: bool,
        target_language: str,
    ) -> None:
        calls.append((sentences, target_language))
        destination.write_bytes(b"RIFFfakewav")

    bundle = create_study_pack(
        cards=cards,
        target_language=TARGET_LANGUAGE,
        output_root=tmp_path,
        tts_writer=fake_tts_writer,
    )

    assert len(bundle.audio_paths) == 2
    assert bundle.audio_paths[0].name == "01_sentences_01_20_20260614_213500.mp3"
    assert bundle.audio_paths[1].name == "02_sentences_21_21_20260614_213500.mp3"
    assert bundle.zip_path.name == "daily_language_pack_20260614_213500.zip"
    assert len(calls[0][0]) == SENTENCES_PER_AUDIO_FILE
    assert calls[0][1] == TARGET_LANGUAGE
    assert calls[1] == (["Target sentence 21"], TARGET_LANGUAGE)


def test_create_study_pack_rejects_unsupported_target_language(tmp_path: Path) -> None:
    cards = [
        SentenceCard(
            scenario="Transit",
            source_sentence="Where is the station?",
            target_sentence="Ou est la gare ?",
            verb_lemma="etre",
            why_it_is_useful="Travel question.",
        )
    ]

    with pytest.raises(ValueError, match="Unsupported target language"):
        create_study_pack(cards=cards, target_language="Dutch", output_root=tmp_path)


def test_model_stack_summary_mentions_qwen_tiny_aya_and_kyutai() -> None:
    summary = get_model_stack_summary()
    assert "Qwen/Qwen3-8B" in summary
    assert "CohereLabs/tiny-aya-global" in summary
    assert "kyutai/tts-1.6b-en_fr" in summary


def test_model_stack_summary_uses_selected_target_language_model(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_TTS_MODEL_ES", "facebook/mms-tts-spa")
    summary = get_model_stack_summary("Spanish")
    assert "facebook/mms-tts-spa" in summary


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


def test_generate_sentence_cards_rejects_unsupported_target_language() -> None:
    with pytest.raises(ValueError, match="Unsupported target language"):
        generate_sentence_cards(
            use_cases="I need Spanish for errands and daily conversations.",
            target_language="Dutch",
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
            for index in range(1, 11)
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


def test_generate_sentence_cards_keeps_small_usable_batches(monkeypatch) -> None:
    def build_batch(prefix: str, count: int) -> dict[str, object]:
        return {
            "rationale": "Use daily situations.",
            "assumptions": ["The learner wants practical phrases."],
            "focus_verbs": ["work", "buy", "ask"],
            "study_routine": [
                {"title": "Preview", "minutes": 10, "instructions": "Scan verbs."},
                {"title": "Listen", "minutes": 20, "instructions": "Shadow audio."},
                {"title": "Speak", "minutes": 15, "instructions": "Recall from prompts."},
            ],
            "sentences": [
                {
                    "scenario": f"{prefix} scenario {index}",
                    "source_sentence": f"{prefix} source {index}",
                    "target_sentence": f"{prefix} target {index}",
                    "verb_lemma": f"{prefix} verb {index}",
                    "why_it_is_useful": "Useful daily sentence.",
                    "pronunciation_hint": "",
                }
                for index in range(1, count + 1)
            ],
        }

    batches = [
        build_batch("first", 3),
        build_batch("second", 4),
        build_batch("third", 3),
    ]

    class StubClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat_completion(self, **_: object) -> SimpleNamespace:
            payload = batches[self.calls]
            self.calls += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
            )

    client = StubClient()
    monkeypatch.setattr(study_pack, "HF_GENERATION_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.setattr(study_pack, "translate_sentence_cards", lambda cards, **_: cards)

    plan = generate_sentence_cards(
        use_cases="I need French for errands and daily conversations.",
        target_language=TARGET_LANGUAGE,
        native_language="English",
        sentence_count=10,
        client=client,
    )

    assert client.calls == 3
    assert len(plan.cards) == 10
    assert plan.cards[0].target_sentence == "first target 1"
    assert plan.cards[-1].target_sentence == "third target 3"


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
            for index in range(1, 11)
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
        sentence_count=10,
        client=StubClient(),
    )

    assert len(plan.cards) == 10
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
            for index in range(1, 11)
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
        sentence_count=10,
        client=client,
    )

    assert client.calls == 2
    assert len(plan.cards) == 10


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

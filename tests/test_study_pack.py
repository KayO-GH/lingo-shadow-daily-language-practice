from __future__ import annotations

import json
from pathlib import Path

from study_pack import (
    SENTENCES_PER_AUDIO_FILE,
    SentenceCard,
    StudyRoutineStep,
    create_study_pack,
    extract_json_payload,
    normalize_plan,
)


def test_extract_json_payload_reads_fenced_json() -> None:
    raw = """Here you go.

```json
{"sentences": [{"source_sentence": "Hello", "target_sentence": "Bonjour"}]}
```
"""
    payload = extract_json_payload(raw)
    assert payload["sentences"][0]["target_sentence"] == "Bonjour"


def test_normalize_plan_dedupes_by_target_sentence_and_reads_routine() -> None:
    payload = {
        "rationale": "Use daily situations.",
        "assumptions": ["The learner wants portable phrases."],
        "focus_verbs": ["avoir besoin", "aller", "etre", "pouvoir"],
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
                "verb_lemma": "avoir besoin",
                "why_it_is_useful": "Very common request.",
                "pronunciation_hint": "",
            },
            {
                "scenario": "Shop",
                "source_sentence": "I need a bag.",
                "target_sentence": "J'ai besoin d'un sac.",
                "verb_lemma": "avoir besoin",
                "why_it_is_useful": "Duplicate example.",
                "pronunciation_hint": "",
            },
            {
                "scenario": "Travel",
                "source_sentence": "Where are we going?",
                "target_sentence": "Ou allons-nous ?",
                "verb_lemma": "aller",
                "why_it_is_useful": "Core travel verb.",
                "pronunciation_hint": "",
            },
            {
                "scenario": "Home",
                "source_sentence": "I am ready.",
                "target_sentence": "Je suis pret.",
                "verb_lemma": "etre",
                "why_it_is_useful": "Basic state sentence.",
                "pronunciation_hint": "",
            },
            {
                "scenario": "Cafe",
                "source_sentence": "Can I pay now?",
                "target_sentence": "Je peux payer maintenant ?",
                "verb_lemma": "pouvoir",
                "why_it_is_useful": "Payment and permission.",
                "pronunciation_hint": "",
            },
        ],
    }

    plan = normalize_plan(payload, sentence_count=4)
    assert plan.rationale == "Use daily situations."
    assert plan.assumptions == ["The learner wants portable phrases."]
    assert plan.focus_verbs == ["avoir besoin", "aller", "etre", "pouvoir"]
    assert [step.minutes for step in plan.routine_steps] == [10, 20, 15]
    assert len(plan.cards) == 4
    assert [card.verb_lemma for card in plan.cards] == ["avoir besoin", "aller", "etre", "pouvoir"]


def test_create_study_pack_writes_bundle_and_zip(tmp_path: Path) -> None:
    cards = [
        SentenceCard(
            scenario="Groceries",
            source_sentence="I need apples.",
            target_sentence="J'ai besoin de pommes.",
            verb_lemma="avoir besoin",
            why_it_is_useful="Shopping staple.",
            pronunciation_hint="zhay buh-ZWAN duh pom",
        ),
        SentenceCard(
            scenario="Transit",
            source_sentence="Where does this bus go?",
            target_sentence="Ce bus va ou ?",
            verb_lemma="aller",
            why_it_is_useful="Travel question.",
            pronunciation_hint="suh boos va oo",
        ),
    ]

    calls: list[tuple[str, str, bool]] = []

    def fake_tts_writer(text: str, lang_code: str, destination: Path, slow_audio: bool) -> None:
        calls.append((text, lang_code, slow_audio))
        destination.write_bytes(f"{lang_code}|{slow_audio}|{text}".encode("utf-8"))

    bundle = create_study_pack(
        cards=cards,
        target_language="French",
        focus_verbs=["avoir besoin", "aller"],
        routine_steps=[
            StudyRoutineStep(title="Preview", minutes=10, instructions="Scan verbs."),
            StudyRoutineStep(title="Listen", minutes=20, instructions="Shadow audio."),
            StudyRoutineStep(title="Speak", minutes=15, instructions="Recall from prompts."),
        ],
        output_root=tmp_path,
        tts_writer=fake_tts_writer,
    )

    assert bundle.preview_audio_path.exists()
    assert bundle.zip_path.exists()
    assert len(bundle.audio_paths) == 1
    assert (bundle.session_dir / "study_pack.csv").exists()
    assert (bundle.session_dir / "study_pack.json").exists()
    assert (bundle.session_dir / "daily_routine.md").exists()
    assert (bundle.session_dir / "focus_verbs.txt").exists()
    assert calls == [("J'ai besoin de pommes.\nCe bus va ou ?", "fr", False)]

    payload = json.loads((bundle.session_dir / "study_pack.json").read_text(encoding="utf-8"))
    assert payload[0]["target_sentence"] == "J'ai besoin de pommes."
    assert "avoir besoin" in (bundle.session_dir / "focus_verbs.txt").read_text(encoding="utf-8")


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

    calls: list[str] = []

    def fake_tts_writer(text: str, lang_code: str, destination: Path, slow_audio: bool) -> None:
        calls.append(text)
        destination.write_bytes(text.encode("utf-8"))

    bundle = create_study_pack(
        cards=cards,
        target_language="French",
        output_root=tmp_path,
        tts_writer=fake_tts_writer,
    )

    assert len(bundle.audio_paths) == 2
    assert bundle.audio_paths[0].name == "01_sentences_01_20.mp3"
    assert bundle.audio_paths[1].name == "02_sentences_21_21.mp3"
    assert calls[0].count("\n") == SENTENCES_PER_AUDIO_FILE - 1
    assert calls[1] == "Target sentence 21"

"""Core generation and packaging logic for the Daily Language Practice app."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FALLBACK_ENV_PATH = Path("/Users/Kwadwo/Documents/PROJECTS/NITA-bill-review/.env")
OUTPUT_ROOT = Path(tempfile.gettempdir()) / "daily_language_practice"
SENTENCES_PER_AUDIO_FILE = 20
TARGET_LANGUAGE = "French"
NATIVE_LANGUAGE_CHOICES = ["English", "French", "Spanish", "German", "Portuguese"]

HF_GENERATION_MODEL = "Qwen/Qwen3-8B"
MODAL_TTS_MODEL = "kyutai/tts-1.6b-en_fr"
MODAL_TTS_VOICE_REPO = "kyutai/tts-voices"
MODAL_TTS_VOICE = "voice-donations/Hugo_the_frenchie_enhanced.wav"
HF_GENERATION_PARAMS = 8_200_000_000
MODAL_TTS_PARAMS = 1_800_000_000
MODAL_TTS_TIMEOUT_SECONDS = 120.0
FRENCH_ONLY_ERROR = "This v1 build currently supports French only."
MACOS_FRENCH_VOICE_CANDIDATES = ("Amélie", "Thomas")
AUDIO_SENTENCE_PAUSE_SECONDS = 2.0
DEFAULT_MACOS_SPEECH_RATE = 180
SLOW_AUDIO_SPEED_MULTIPLIER = 0.9

SUPPORTED_LANGUAGES: dict[str, dict[str, str]] = {
    TARGET_LANGUAGE: {"tts_code": "fr", "language_label": TARGET_LANGUAGE},
}


@dataclass(slots=True)
class SentenceCard:
    scenario: str
    source_sentence: str
    target_sentence: str
    verb_lemma: str
    why_it_is_useful: str
    pronunciation_hint: str = ""


@dataclass(slots=True)
class StudyRoutineStep:
    title: str
    minutes: int
    instructions: str


@dataclass(slots=True)
class GeneratedStudyPlan:
    rationale: str
    assumptions: list[str]
    focus_verbs: list[str]
    routine_steps: list[StudyRoutineStep]
    cards: list[SentenceCard]


@dataclass(slots=True)
class StudyPackBundle:
    session_dir: Path
    zip_path: Path
    audio_paths: list[Path]
    preview_audio_path: Path
    cards: list[SentenceCard]
    tts_backend_label: str


@dataclass(slots=True)
class ModalTTSClient:
    base_url: str
    auth_token: str
    timeout_seconds: float = MODAL_TTS_TIMEOUT_SECONDS
    transport: Callable[[str, bytes, dict[str, str], float], bytes] | None = None

    def synthesize_track(self, sentences: list[str], slow_audio: bool) -> bytes:
        cleaned_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if not cleaned_sentences:
            raise ValueError("At least one non-empty sentence is required for TTS synthesis.")

        if not self.base_url.strip():
            raise RuntimeError("Missing MODAL_TTS_BASE_URL. Add it to .env or to the fallback env file.")

        payload = json.dumps(
            {"sentences": cleaned_sentences, "slow_audio": slow_audio}
        ).encode("utf-8")
        headers = {
            "Accept": "audio/wav",
            "Content-Type": "application/json",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        transport = self.transport or _default_modal_tts_transport
        return transport(
            self.base_url.rstrip("/") + "/synthesize-track",
            payload,
            headers,
            self.timeout_seconds,
        )


def ensure_supported_target_language(language_name: str) -> None:
    if language_name != TARGET_LANGUAGE:
        raise ValueError(FRENCH_ONLY_ERROR)


def get_model_stack_summary() -> str:
    total_params = HF_GENERATION_PARAMS + MODAL_TTS_PARAMS
    return (
        f"{HF_GENERATION_MODEL} ({HF_GENERATION_PARAMS:,} params) + "
        f"{MODAL_TTS_MODEL} (~{MODAL_TTS_PARAMS:,} params) = ~{total_params:,} total params"
    )


def load_environment() -> Path | None:
    """Load env vars from the project and then the shared fallback path."""

    project_env = PROJECT_ROOT / ".env"
    if project_env.exists():
        load_dotenv(project_env, override=False)

    fallback_value = os.getenv("LANGUAGE_PRACTICE_FALLBACK_ENV")
    fallback_path = Path(fallback_value) if fallback_value else DEFAULT_FALLBACK_ENV_PATH
    if fallback_path.exists():
        load_dotenv(fallback_path, override=False)
        return fallback_path

    return None


def get_supported_language_labels() -> list[str]:
    return [TARGET_LANGUAGE]


def get_native_language_choices() -> list[str]:
    return list(NATIVE_LANGUAGE_CHOICES)


def get_tts_code(language_name: str) -> str:
    ensure_supported_target_language(language_name)
    return SUPPORTED_LANGUAGES[language_name]["tts_code"]


def get_modal_tts_client(
    transport: Callable[[str, bytes, dict[str, str], float], bytes] | None = None,
) -> ModalTTSClient:
    load_environment()
    base_url = os.getenv("MODAL_TTS_BASE_URL", "").strip()
    auth_token = os.getenv("MODAL_TTS_AUTH_TOKEN", "").strip()
    timeout_raw = os.getenv("MODAL_TTS_TIMEOUT_SECONDS", "").strip()

    timeout_seconds = MODAL_TTS_TIMEOUT_SECONDS
    if timeout_raw:
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise RuntimeError("MODAL_TTS_TIMEOUT_SECONDS must be a valid number.") from exc

    return ModalTTSClient(
        base_url=base_url,
        auth_token=auth_token,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )


def build_generation_prompt(
    use_cases: str,
    target_language: str,
    native_language: str,
    sentence_count: int,
    used_verbs: list[str] | None = None,
    used_target_sentences: list[str] | None = None,
    batch_index: int = 1,
    total_batches: int = 1,
) -> tuple[str, str]:
    system_prompt = f"""
You are designing a starter language-learning pack for a real person.

Your job:
- infer the situations they are most likely to face from their daily-life description
- produce natural, high-frequency sentences for those situations
- maximize useful verb coverage across the full pack
- prioritize verbs because they carry sentence meaning and speed up comprehension
- build personalized material the learner can study alone at home
- support a daily routine focused on useful input, listening, and speaking practice
- keep the sentences short enough to be easy to repeat aloud
- mix statements, questions, and requests when appropriate
- avoid textbook-only phrasing
- return strict JSON only

Return a JSON object with these keys:
- rationale: short paragraph
- assumptions: array of short strings
- focus_verbs: array of 8 to 15 verb lemmas ordered by importance
- study_routine: array of exactly 3 objects that total 45 minutes
- sentences: array of exactly {sentence_count} objects

Each study_routine object must include:
- title
- minutes
- instructions

Each sentence object must include:
- scenario
- source_sentence
- target_sentence
- verb_lemma
- why_it_is_useful
- pronunciation_hint

Important rules:
- Write source_sentence in {native_language}.
- Write target_sentence in {target_language}.
- pronunciation_hint should be empty unless it helps an {native_language} speaker pronounce the sentence.
- Do not use placeholders such as [name] or [place].
- Prefer portable, reusable lines that a learner can adapt in many settings.
- Build the routine around a solo learner using these materials daily.
""".strip()

    user_prompt = f"""
General use cases from the learner:
{use_cases.strip()}

Build batch {batch_index} of {total_batches} now.
""".strip()

    used_verbs = used_verbs or []
    used_target_sentences = used_target_sentences or []
    if used_verbs:
        user_prompt += "\n\nAlready covered verbs to avoid repeating too much:\n- " + "\n- ".join(used_verbs[:20])
    if used_target_sentences:
        user_prompt += (
            "\n\nAlready covered target sentences to avoid duplicating:\n- "
            + "\n- ".join(used_target_sentences[:12])
        )

    return system_prompt, user_prompt


def extract_json_payload(raw_text: str) -> Any:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", raw_text, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    json_object_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if json_object_match:
        return json.loads(json_object_match.group(1))

    json_array_match = re.search(r"(\[.*\])", raw_text, re.DOTALL)
    if json_array_match:
        return json.loads(json_array_match.group(1))

    raise ValueError("The model response did not contain valid JSON.")


def estimate_generation_max_tokens(sentence_count: int) -> int:
    return min(8192, max(3200, 1800 + (sentence_count * 180)))


def _merge_unique_text(base_items: list[str], additions: list[str], limit: int | None = None) -> list[str]:
    merged = list(base_items)
    seen = {item.casefold() for item in merged}
    for item in additions:
        if item.casefold() in seen:
            continue
        merged.append(item)
        seen.add(item.casefold())
        if limit is not None and len(merged) >= limit:
            break
    return merged


def _request_generation_plan(
    client: InferenceClient,
    use_cases: str,
    target_language: str,
    native_language: str,
    sentence_count: int,
    used_verbs: list[str] | None = None,
    used_target_sentences: list[str] | None = None,
    batch_index: int = 1,
    total_batches: int = 1,
) -> GeneratedStudyPlan:
    system_prompt, user_prompt = build_generation_prompt(
        use_cases=use_cases,
        target_language=target_language,
        native_language=native_language,
        sentence_count=sentence_count,
        used_verbs=used_verbs,
        used_target_sentences=used_target_sentences,
        batch_index=batch_index,
        total_batches=total_batches,
    )

    response = client.chat_completion(
        model=HF_GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=estimate_generation_max_tokens(sentence_count),
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content if response.choices else ""
    if not raw_text:
        raise RuntimeError("The HF generation response did not include text output.")

    payload = extract_json_payload(raw_text)
    return normalize_plan(payload, sentence_count=sentence_count)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _default_modal_tts_transport(url: str, payload: bytes, headers: dict[str, str], timeout: float) -> bytes:
    request = urllib.request.Request(url=url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        message = detail or exc.reason or "unknown error"
        raise RuntimeError(f"Modal TTS request failed with HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Modal TTS request failed: {exc.reason}") from exc

    if "audio/wav" not in content_type and "application/octet-stream" not in content_type:
        raise RuntimeError(f"Modal TTS returned unexpected content type: {content_type or 'missing'}")

    return body


def _resolve_macos_french_voice() -> str | None:
    if sys.platform != "darwin":
        return None

    say_binary = shutil.which("say")
    afconvert_binary = shutil.which("afconvert")
    if not say_binary or not afconvert_binary:
        return None

    try:
        result = subprocess.run(
            [say_binary, "-v", "?"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    installed_voices = {line.split(maxsplit=1)[0] for line in result.stdout.splitlines() if line.strip()}
    for voice in MACOS_FRENCH_VOICE_CANDIDATES:
        if voice in installed_voices:
            return voice
    return None


def _write_macos_fallback_wav(
    sentences: list[str],
    destination: Path,
    slow_audio: bool,
) -> str:
    voice = _resolve_macos_french_voice()
    if not voice:
        raise RuntimeError("No macOS French voice is available for local fallback audio generation.")

    say_binary = shutil.which("say")
    afconvert_binary = shutil.which("afconvert")
    if not say_binary or not afconvert_binary:
        raise RuntimeError("macOS speech tools are unavailable for local fallback audio generation.")

    pause_milliseconds = int(AUDIO_SENTENCE_PAUSE_SECONDS * 1000)
    spoken_text = f" [[slnc {pause_milliseconds}]] ".join(
        sentence.strip() for sentence in sentences if sentence.strip()
    )
    if not spoken_text:
        raise ValueError("At least one non-empty sentence is required for local fallback audio generation.")

    target_rate = round(DEFAULT_MACOS_SPEECH_RATE * SLOW_AUDIO_SPEED_MULTIPLIER) if slow_audio else DEFAULT_MACOS_SPEECH_RATE
    rate = str(target_rate)
    with tempfile.TemporaryDirectory(prefix="daily_language_practice_tts_") as temp_dir:
        temp_aiff = Path(temp_dir) / "track.aiff"
        subprocess.run(
            [say_binary, "-v", voice, "-r", rate, "-o", str(temp_aiff), spoken_text],
            check=True,
        )
        subprocess.run(
            [afconvert_binary, "-f", "WAVE", "-d", "LEI16", str(temp_aiff), str(destination)],
            check=True,
        )

    return f"macOS say ({voice})"


def default_study_routine() -> list[StudyRoutineStep]:
    return [
        StudyRoutineStep(
            title="Verb scan and sentence preview",
            minutes=10,
            instructions="Review the focus verbs first, then read through the new sentences out loud once.",
        ),
        StudyRoutineStep(
            title="Listening and shadowing",
            minutes=20,
            instructions="Play the target-language audio tracks, pause after each sentence, and repeat aloud.",
        ),
        StudyRoutineStep(
            title="Active recall speaking",
            minutes=15,
            instructions="Hide the target sentence, answer from the source prompt, then check and repeat the correct version.",
        ),
    ]


def normalize_plan(payload: Any, sentence_count: int) -> GeneratedStudyPlan:
    if isinstance(payload, list):
        rationale = ""
        assumptions: list[str] = []
        focus_verbs: list[str] = []
        routine_steps: list[StudyRoutineStep] = []
        raw_cards = payload
    elif isinstance(payload, dict):
        rationale = _clean_text(payload.get("rationale"))
        assumptions = [_clean_text(item) for item in payload.get("assumptions", []) if _clean_text(item)]
        focus_verbs = [_clean_text(item) for item in payload.get("focus_verbs", []) if _clean_text(item)]
        routine_steps = []
        for item in payload.get("study_routine", payload.get("routine", [])):
            if not isinstance(item, dict):
                continue
            title = _clean_text(item.get("title"))
            instructions = _clean_text(item.get("instructions"))
            minutes_value = item.get("minutes", 0)
            try:
                minutes = int(minutes_value)
            except (TypeError, ValueError):
                minutes = 0
            if title and instructions and minutes > 0:
                routine_steps.append(
                    StudyRoutineStep(title=title, minutes=minutes, instructions=instructions)
                )
        raw_cards = payload.get("sentences", payload.get("cards", []))
    else:
        raise ValueError("Unexpected model payload shape.")

    if not isinstance(raw_cards, list):
        raise ValueError("The model payload did not include a sentence list.")

    seen_targets: set[str] = set()
    cards: list[SentenceCard] = []
    for raw_card in raw_cards:
        if not isinstance(raw_card, dict):
            continue

        card = SentenceCard(
            scenario=_clean_text(raw_card.get("scenario") or raw_card.get("situation")),
            source_sentence=_clean_text(raw_card.get("source_sentence") or raw_card.get("english_sentence")),
            target_sentence=_clean_text(raw_card.get("target_sentence") or raw_card.get("translation")),
            verb_lemma=_clean_text(raw_card.get("verb_lemma") or raw_card.get("verb")),
            why_it_is_useful=_clean_text(raw_card.get("why_it_is_useful") or raw_card.get("utility")),
            pronunciation_hint=_clean_text(raw_card.get("pronunciation_hint")),
        )

        if not card.source_sentence or not card.target_sentence:
            continue

        dedupe_key = card.target_sentence.casefold()
        if dedupe_key in seen_targets:
            continue

        seen_targets.add(dedupe_key)
        cards.append(card)
        if len(cards) == sentence_count:
            break

    if len(cards) < max(4, min(8, sentence_count // 2)):
        raise ValueError("The model returned too few usable sentences.")

    if not focus_verbs:
        focus_verbs = []
        for card in cards:
            verb = card.verb_lemma
            if verb and verb not in focus_verbs:
                focus_verbs.append(verb)
        focus_verbs = focus_verbs[:12]

    if not routine_steps or sum(step.minutes for step in routine_steps) != 45:
        routine_steps = default_study_routine()

    if not rationale:
        rationale = (
            "The pack focuses on short, reusable sentences that map directly to the learner's"
            " recurring situations and common daily verbs."
        )
    if not assumptions:
        assumptions = ["The learner wants practical spoken sentences before formal grammar study."]

    return GeneratedStudyPlan(
        rationale=rationale,
        assumptions=assumptions,
        focus_verbs=focus_verbs,
        routine_steps=routine_steps,
        cards=cards,
    )


def generate_sentence_cards(
    use_cases: str,
    target_language: str,
    native_language: str,
    sentence_count: int,
    client: InferenceClient | None = None,
) -> GeneratedStudyPlan:
    ensure_supported_target_language(target_language)
    load_environment()
    api_key = os.getenv("HF_TOKEN", "").strip()
    if not api_key and client is None:
        raise RuntimeError("Missing HF_TOKEN. Add it to .env or to the fallback env file.")

    if client is None:
        client = InferenceClient(api_key=api_key)

    batch_size_limit = 20
    total_batches = max(1, (sentence_count + batch_size_limit - 1) // batch_size_limit)
    extra_retry_budget = 3
    planned_batches = total_batches + extra_retry_budget
    collected_cards: list[SentenceCard] = []
    collected_verbs: list[str] = []
    collected_assumptions: list[str] = []
    routine_steps: list[StudyRoutineStep] = []
    rationale = ""

    for batch_index in range(1, planned_batches + 1):
        remaining = sentence_count - len(collected_cards)
        if remaining <= 0:
            break

        requested_count = min(batch_size_limit, max(remaining, 8))
        batch_plan = _request_generation_plan(
            client=client,
            use_cases=use_cases,
            target_language=target_language,
            native_language=native_language,
            sentence_count=requested_count,
            used_verbs=collected_verbs,
            used_target_sentences=[card.target_sentence for card in collected_cards],
            batch_index=batch_index,
            total_batches=planned_batches,
        )

        if not rationale:
            rationale = batch_plan.rationale
        if not routine_steps:
            routine_steps = batch_plan.routine_steps
        collected_assumptions = _merge_unique_text(collected_assumptions, batch_plan.assumptions)
        collected_verbs = _merge_unique_text(collected_verbs, batch_plan.focus_verbs, limit=20)

        seen_targets = {card.target_sentence.casefold() for card in collected_cards}
        for card in batch_plan.cards:
            if card.target_sentence.casefold() in seen_targets:
                continue
            collected_cards.append(card)
            seen_targets.add(card.target_sentence.casefold())
            if len(collected_cards) >= sentence_count:
                break

    if len(collected_cards) < sentence_count:
        raise RuntimeError(
            f"Only generated {len(collected_cards)} unique sentences out of the requested {sentence_count}."
        )

    if not collected_verbs:
        collected_verbs = _merge_unique_text([], [card.verb_lemma for card in collected_cards], limit=12)

    return GeneratedStudyPlan(
        rationale=rationale or (
            "The pack focuses on short, reusable sentences that map directly to the learner's"
            " recurring situations and common daily verbs."
        ),
        assumptions=collected_assumptions or ["The learner wants practical spoken sentences before formal grammar study."],
        focus_verbs=collected_verbs,
        routine_steps=routine_steps or default_study_routine(),
        cards=collected_cards[:sentence_count],
    )


def sanitize_filename(text: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return compact[:36] or "sentence"


def default_tts_writer(
    sentences: list[str],
    destination: Path,
    slow_audio: bool,
    client: ModalTTSClient | None = None,
) -> str:
    tts_client = client or get_modal_tts_client()
    try:
        audio_bytes = tts_client.synthesize_track(sentences, slow_audio=slow_audio)
    except RuntimeError:
        fallback_voice = _resolve_macos_french_voice()
        if not fallback_voice:
            raise
        return _write_macos_fallback_wav(sentences, destination, slow_audio)

    destination.write_bytes(audio_bytes)
    return f"Modal ({MODAL_TTS_MODEL})"


def chunk_cards(cards: list[SentenceCard], chunk_size: int = SENTENCES_PER_AUDIO_FILE) -> list[list[SentenceCard]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [cards[index : index + chunk_size] for index in range(0, len(cards), chunk_size)]


def create_study_pack(
    cards: list[SentenceCard],
    target_language: str,
    focus_verbs: list[str] | None = None,
    routine_steps: list[StudyRoutineStep] | None = None,
    slow_audio: bool = False,
    output_root: Path | None = None,
    tts_writer: Callable[[list[str], Path, bool], str | None] | None = None,
) -> StudyPackBundle:
    if not cards:
        raise ValueError("At least one sentence card is required.")

    ensure_supported_target_language(target_language)
    get_tts_code(target_language)
    writer = tts_writer or default_tts_writer
    base_dir = output_root or OUTPUT_ROOT
    session_dir = base_dir / f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    session_dir.mkdir(parents=True, exist_ok=True)

    audio_paths: list[Path] = []
    tts_backend_label = f"Modal ({MODAL_TTS_MODEL})"
    for batch_index, card_batch in enumerate(chunk_cards(cards), start=1):
        start_number = ((batch_index - 1) * SENTENCES_PER_AUDIO_FILE) + 1
        end_number = start_number + len(card_batch) - 1
        filename = f"{batch_index:02d}_sentences_{start_number:02d}_{end_number:02d}.wav"
        audio_path = session_dir / filename
        track_sentences = [card.target_sentence for card in card_batch]
        backend_label = writer(track_sentences, audio_path, slow_audio)
        if backend_label:
            tts_backend_label = backend_label
        audio_paths.append(audio_path)

    csv_path = session_dir / "study_pack.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer_obj = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "source_sentence",
                "target_sentence",
                "verb_lemma",
                "why_it_is_useful",
                "pronunciation_hint",
            ],
        )
        writer_obj.writeheader()
        for card in cards:
            writer_obj.writerow(asdict(card))

    json_path = session_dir / "study_pack.json"
    json_path.write_text(
        json.dumps([asdict(card) for card in cards], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    focus_verbs = focus_verbs or []
    routine_steps = routine_steps or default_study_routine()

    summary_lines = [
        f"Target language: {target_language}",
        f"Sentence count: {len(cards)}",
        f"Audio track count: {len(audio_paths)}",
        f"Sentences per audio file: up to {SENTENCES_PER_AUDIO_FILE}",
        f"TTS service: {tts_backend_label}",
        f"TTS voice profile: {MODAL_TTS_VOICE}",
        f"Model stack: {get_model_stack_summary()}",
        "",
        "Focus verbs:",
    ]
    summary_lines.extend(f"- {verb}" for verb in focus_verbs)
    summary_lines.extend(
        [
            "",
            "45-minute routine:",
        ]
    )
    summary_lines.extend(
        f"- {step.minutes} min: {step.title} - {step.instructions}" for step in routine_steps
    )
    summary_lines.extend(
        [
            "",
            "Sentences:",
        ]
    )
    summary_lines.extend(
        f"- {card.target_sentence} ({card.source_sentence})" for card in cards
    )
    (session_dir / "README.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    routine_lines = ["# 45-Minute Daily Routine", ""]
    for step in routine_steps:
        routine_lines.extend(
            [
                f"## {step.title} ({step.minutes} min)",
                step.instructions,
                "",
            ]
        )
    (session_dir / "daily_routine.md").write_text("\n".join(routine_lines).strip() + "\n", encoding="utf-8")

    (session_dir / "focus_verbs.txt").write_text(
        "\n".join(focus_verbs) + ("\n" if focus_verbs else ""),
        encoding="utf-8",
    )

    zip_path = session_dir / "daily_language_pack.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(session_dir.iterdir()):
            if item == zip_path:
                continue
            archive.write(item, arcname=item.name)

    return StudyPackBundle(
        session_dir=session_dir,
        zip_path=zip_path,
        audio_paths=audio_paths,
        preview_audio_path=audio_paths[0],
        cards=cards,
        tts_backend_label=tts_backend_label,
    )


def build_results_rows(cards: list[SentenceCard]) -> list[list[str]]:
    return [
        [
            card.scenario,
            card.source_sentence,
            card.target_sentence,
            card.verb_lemma,
        ]
        for card in cards
    ]

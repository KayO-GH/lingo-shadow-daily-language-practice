"""Gradio entrypoint for the LingoShadow - Daily Language Practice app."""

from __future__ import annotations

import logging
import os

import gradio as gr

from study_pack import (
    GeneratedStudyPlan,
    HF_GENERATION_MODEL,
    HF_GENERATION_PARAMS,
    SENTENCES_PER_AUDIO_FILE,
    TARGET_LANGUAGE,
    TRANSLATION_MODEL,
    TRANSLATION_MODEL_PARAMS,
    build_results_rows,
    create_study_pack,
    generate_sentence_cards,
    get_model_stack_summary,
    get_native_language_choices,
    get_supported_language_labels,
    get_tts_backend_config,
    load_environment,
)

APP_TITLE = "LingoShadow - Daily Language Practice"
logger = logging.getLogger(__name__)
APP_DESCRIPTION = """
Turn your real daily routines into a practical language study system.
Describe the situations you actually live in and the app will build high-frequency target-language sentences, focus verbs, a 45-minute routine, and downloadable audio tracks generated through a target-language-specific Modal TTS service.
""".strip()

APP_THEME = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="sky",
    neutral_hue="slate",
).set(
    body_background_fill="linear-gradient(180deg, #fff8ef 0%, #effcfb 48%, #eef4ff 100%)",
    block_background_fill="rgba(255, 255, 255, 0.92)",
    block_border_color="rgba(249, 115, 22, 0.18)",
    button_primary_background_fill="linear-gradient(135deg, #f97316 0%, #f43f5e 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #ea580c 0%, #e11d48 100%)",
    button_primary_text_color="#fffdf9",
    input_background_fill="#ffffff",
    input_border_color="rgba(14, 165, 233, 0.22)",
)

APP_CSS = """
body {
    background:
        radial-gradient(circle at top left, rgba(253, 186, 116, 0.28), transparent 25%),
        radial-gradient(circle at top right, rgba(56, 189, 248, 0.18), transparent 28%),
        linear-gradient(180deg, #fff8ef 0%, #effcfb 46%, #eef4ff 100%);
}
.gradio-container,
.gradio-container-6-17-3 {
    background:
        radial-gradient(circle at top left, rgba(253, 186, 116, 0.28), transparent 25%),
        radial-gradient(circle at top right, rgba(56, 189, 248, 0.18), transparent 28%),
        linear-gradient(180deg, #fff8ef 0%, #effcfb 46%, #eef4ff 100%) !important;
    color: #0f172a !important;
}
#app-shell {
    max-width: 1180px;
    margin: 0 auto;
    padding: 1rem 0 2rem 0;
}
#app-shell > .gradio-row,
#app-shell > .gradio-column {
    gap: 1rem;
}
#flag-ribbon {
    margin-bottom: 0.35rem;
}
.flag-ribbon {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    align-items: center;
    justify-content: center;
}
.flag-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.5rem 0.8rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(249, 115, 22, 0.16);
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    font-size: 0.92rem;
    color: #334155 !important;
    font-weight: 600;
    animation: float-card 6s ease-in-out infinite;
}
.flag-pill:nth-child(2n) {
    animation-delay: 1s;
}
.flag-pill:nth-child(3n) {
    animation-delay: 2s;
}
.flag-emoji {
    font-size: 1.15rem;
}
#hero-grid {
    position: relative;
    overflow: hidden;
    padding: 1.2rem;
    border: 1px solid rgba(249, 115, 22, 0.14);
    border-radius: 28px;
    background:
        radial-gradient(circle at top right, rgba(251, 146, 60, 0.26), transparent 26%),
        radial-gradient(circle at bottom left, rgba(56, 189, 248, 0.18), transparent 24%),
        linear-gradient(135deg, rgba(255, 255, 255, 0.92) 0%, rgba(255, 247, 237, 0.96) 54%, rgba(240, 249, 255, 0.94) 100%);
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
}
#hero-grid::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(255, 255, 255, 0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.08) 1px, transparent 1px);
    background-size: 24px 24px;
    pointer-events: none;
}
.hero-copy,
.hero-card-wrap {
    position: relative;
    z-index: 1;
}
.hero-copy h1 {
    font-size: clamp(2.4rem, 5vw, 3.65rem);
    line-height: 1.02;
    margin-bottom: 0.45rem;
    color: #7c2d12 !important;
    font-weight: 800;
}
.hero-copy p {
    font-size: 1.02rem;
    color: #7c2d12;
}
.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.45rem 0.75rem;
    border-radius: 999px;
    background: rgba(255, 237, 213, 0.94);
    color: #9a3412 !important;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    font-size: 0.78rem;
}
.hero-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin: 0.9rem 0 1rem 0;
}
.hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.65rem 0.85rem;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid rgba(14, 165, 233, 0.14);
    box-shadow: 0 12px 30px rgba(14, 165, 233, 0.08);
    color: #0f172a !important;
    font-weight: 700;
}
.hero-sidecard {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 1.25rem;
    min-height: 100%;
    background: linear-gradient(180deg, rgba(14, 165, 233, 0.94) 0%, rgba(14, 116, 144, 0.98) 100%);
    color: #f8fafc;
    box-shadow: 0 22px 40px rgba(14, 116, 144, 0.24);
}
.hero-sidecard::after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    right: -36px;
    top: -60px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.12);
}
.hero-sidecard h3,
.hero-sidecard p,
.hero-sidecard li,
.hero-sidecard strong {
    position: relative;
    z-index: 1;
}
.passport-stack {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.7rem;
    margin-top: 1rem;
}
.passport-card {
    padding: 0.8rem;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(8px);
}
.passport-card strong {
    display: block;
    font-size: 1.45rem;
    margin-bottom: 0.15rem;
    color: #f8fafc !important;
}
#practice-highlights {
    margin-top: 0.2rem;
}
.highlights-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.9rem;
}
.highlight-card {
    padding: 1rem 1rem 1.1rem 1rem;
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.16);
    box-shadow: 0 16px 34px rgba(15, 23, 42, 0.06);
}
.highlight-card strong {
    display: block;
    margin-bottom: 0.22rem;
    color: #0f172a !important;
    font-size: 1.02rem;
    font-weight: 800;
}
.highlight-card span {
    font-size: 1.35rem;
    display: inline-block;
    margin-bottom: 0.4rem;
}
.highlight-card p,
.panel-heading p,
.control-tip,
.control-tip p,
.examples-note,
#app-shell .prose p,
#app-shell .prose li,
#app-shell .prose span,
#app-shell label,
#app-shell .wrap {
    color: #475569 !important;
}
.highlight-card p,
.control-tip,
.examples-note {
    margin: 0;
}
.hero-copy .prose p,
.hero-copy .prose li,
.hero-copy .prose strong {
    color: #7c2d12 !important;
}
.hero-copy .prose h1,
.hero-copy .prose h2,
.hero-copy .prose h3 {
    color: #7c2d12 !important;
}
.hero-sidecard,
.hero-sidecard p,
.hero-sidecard span,
.hero-sidecard li,
.hero-sidecard strong,
.hero-sidecard h3,
.passport-card span {
    color: #f8fafc !important;
}
.builder-panel textarea,
.control-panel input,
.control-panel textarea,
.control-panel select,
.results-panel,
.status-panel {
    color: #0f172a !important;
}
.builder-panel,
.control-panel,
.results-panel,
.status-panel {
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.84);
    box-shadow: 0 18px 36px rgba(15, 23, 42, 0.05);
    padding: 0.2rem;
}
.panel-heading {
    margin: 0.2rem 0 0.8rem 0;
}
.panel-heading strong {
    font-size: 1.05rem;
    color: #0f172a !important;
    font-weight: 800;
}
.panel-heading p {
    margin: 0.2rem 0 0 0;
    color: #475569;
    font-size: 0.94rem;
}
.control-tip {
    padding: 0.85rem 0.95rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(255, 247, 237, 0.96) 0%, rgba(224, 242, 254, 0.92) 100%);
    border: 1px solid rgba(56, 189, 248, 0.14);
    color: #0f172a;
    margin-bottom: 0.55rem;
}
.control-tip strong {
    display: block;
    color: #9a3412 !important;
    margin-bottom: 0.18rem;
}
.examples-note {
    margin-top: 0.45rem;
    color: #64748b;
    font-size: 0.9rem;
}
#results-shell .gradio-row,
#results-shell .gradio-column {
    gap: 0.9rem;
}
#status-output, #assumptions-output {
    min-height: 3rem;
}
#status-output,
#assumptions-output,
#focus-verbs-output,
#routine-output {
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.86);
}
#results-shell .gradio-dataframe,
#results-shell .gradio-file,
#results-shell .gradio-audio {
    border-radius: 18px;
}
#preview-audio label span svg.feather-music,
#preview-audio .empty .icon svg.feather-music {
    display: none;
}
#preview-audio label span::before,
#preview-audio .empty .icon::before {
    display: inline-block;
    line-height: 1;
    background-color: currentColor;
    content: "";
    mask-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M18.5 6a9 9 0 0 1 0 12"/></svg>');
    mask-position: center;
    mask-repeat: no-repeat;
    mask-size: contain;
    -webkit-mask-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M18.5 6a9 9 0 0 1 0 12"/></svg>');
    -webkit-mask-position: center;
    -webkit-mask-repeat: no-repeat;
    -webkit-mask-size: contain;
}
#preview-audio label span::before {
    width: 0.95rem;
    height: 0.95rem;
}
#preview-audio .empty .icon::before {
    width: 1.45rem;
    height: 1.45rem;
    opacity: 0.72;
}
@keyframes float-card {
    0%,
    100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-4px);
    }
}
@media (max-width: 900px) {
    .highlights-grid,
    .passport-stack {
        grid-template-columns: 1fr;
    }
}
"""

FLAG_BADGES = [
    ("🇫🇷", "French"),
    ("🇪🇸", "Spanish"),
    ("🇩🇪", "German"),
    ("🇮🇹", "Italian"),
    ("🇵🇹", "Portuguese"),
    ("🇯🇵", "Japanese"),
    ("🇬🇧", "English"),
]


def build_flag_ribbon_html() -> str:
    pills = "".join(
        f'<span class="flag-pill"><span class="flag-emoji">{flag}</span> {language}</span>'
        for flag, language in FLAG_BADGES
    )
    return f'<div class="flag-ribbon">{pills}</div>'


def build_hero_chips_html() -> str:
    return """
    <div class="hero-chip-row">
        <div class="hero-chip">🗣️ Real-life phrases</div>
        <div class="hero-chip">🎧 Downloadable audio drills</div>
        <div class="hero-chip">📚 Focus verbs and routines</div>
    </div>
    """.strip()


def build_hero_sidecard_html() -> str:
    return """
    <div class="hero-sidecard">
        <p><strong>Travel-day energy, classroom clarity.</strong></p>
        <h3>Build a mini language camp from one prompt.</h3>
        <p>Pick a target language, describe your routines, and get sentences, verbs, a practice plan, and audio you can loop on the go.</p>
        <div class="passport-stack">
            <div class="passport-card">
                <strong>45 min</strong>
                <span>structured daily practice</span>
            </div>
            <div class="passport-card">
                <strong>ZIP + WAV</strong>
                <span>portable study kit</span>
            </div>
            <div class="passport-card">
                <strong>Daily life</strong>
                <span>groceries, travel, neighbors, family</span>
            </div>
            <div class="passport-card">
                <strong>Bright cues</strong>
                <span>easy scanning and quick reuse</span>
            </div>
        </div>
    </div>
    """.strip()


def build_highlights_html() -> str:
    return f"""
    <div class="highlights-grid">
        <div class="highlight-card">
            <span>🌍</span>
            <strong>{len(FLAG_BADGES)} study destinations</strong>
            <p>Supported target languages stay visible up front so the app immediately reads as language practice.</p>
        </div>
        <div class="highlight-card">
            <span>🧠</span>
            <strong>Routine-first learning</strong>
            <p>The generator turns errands, work, travel, and family talk into reusable phrases instead of abstract textbook examples.</p>
        </div>
        <div class="highlight-card">
            <span>🎶</span>
            <strong>Listen, shadow, repeat</strong>
            <p>Audio previews, bundled tracks, and focus verbs make the pack feel ready for real daily repetition.</p>
        </div>
    </div>
    """.strip()


def build_control_tip_html() -> str:
    return """
    <div class="control-tip">
        <strong>Cheerful builder</strong>
        Add situations you actually live through. The more concrete the prompt, the more useful the sentence pack and audio drills become.
    </div>
    """.strip()


def build_panel_heading_html(title: str, description: str) -> str:
    return f"""
    <div class="panel-heading">
        <strong>{title}</strong>
        <p>{description}</p>
    </div>
    """.strip()

def build_model_stack_md(target_language: str) -> str:
    tts_backend = get_tts_backend_config(target_language)
    total_params = HF_GENERATION_PARAMS + TRANSLATION_MODEL_PARAMS + tts_backend.params
    return (
        f"- Generation: `{HF_GENERATION_MODEL}` ({HF_GENERATION_PARAMS:,} params)\n"
        f"- Translation: `{TRANSLATION_MODEL}` ({TRANSLATION_MODEL_PARAMS:,} params)\n"
        f"- TTS for {target_language}: `{tts_backend.model_label}` (~{tts_backend.params:,} params) via Modal\n"
        f"- TTS voice profile: `{tts_backend.voice_label}`\n"
        f"- Total: **~{total_params:,}** params\n"
        f"- Stack summary: `{get_model_stack_summary(target_language)}`"
    )


def resolve_server_port() -> int | None:
    explicit = os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT")
    return int(explicit) if explicit else None


def build_error_state(message: str):
    return (
        f"### Could not build the study pack\n{message}",
        "### What to try next\n- Try the build again.\n- Reduce the sentence count if the model keeps failing.\n- Add a bit more detail to the use-case description if it is brief.",
        "",
        "",
        [],
        None,
        None,
        [],
    )


def run_pack_builder(
    use_cases: str,
    target_language: str,
    native_language: str,
    sentence_count: int,
):
    load_environment()
    cleaned_use_cases = (use_cases or "").strip()
    if len(cleaned_use_cases) < 20:
        return build_error_state(
            "Describe your daily use cases in a bit more detail so the pack can be personalized."
        )

    try:
        plan: GeneratedStudyPlan = generate_sentence_cards(
            use_cases=cleaned_use_cases,
            target_language=target_language,
            native_language=native_language,
            sentence_count=sentence_count,
        )
        bundle = create_study_pack(
            cards=plan.cards,
            target_language=target_language,
            focus_verbs=plan.focus_verbs,
            routine_steps=plan.routine_steps,
            slow_audio=True,
        )
    except ValueError as exc:
        logger.exception("Validation failure while building the study pack")
        return build_error_state(str(exc))
    except RuntimeError as exc:
        return build_error_state(str(exc))
    except Exception:
        logger.exception("Unexpected failure while building the study pack")
        return build_error_state("An unexpected error stopped the build. Check the terminal logs, then try again.")

    status_md = (
        f"Built **{len(plan.cards)}** study sentences for **{target_language}** and generated "
        f"**{len(bundle.audio_paths)}** downloadable audio track(s) with up to "
        f"**{SENTENCES_PER_AUDIO_FILE}** sentences per file using "
        f"`{HF_GENERATION_MODEL}` for pack generation, `{TRANSLATION_MODEL}` for translation, "
        f"plus **{bundle.tts_backend_label}**."
    )
    assumptions_md = (
        "### Working assumptions\n"
        + "\n".join(f"- {item}" for item in plan.assumptions)
        + f"\n\n### Pack logic\n{plan.rationale}"
    )
    routine_md = "### 45-minute routine\n" + "\n".join(
        f"- **{step.minutes} min:** {step.title} - {step.instructions}" for step in plan.routine_steps
    )
    focus_verbs_md = "### Focus verbs\n" + ", ".join(plan.focus_verbs)

    return (
        status_md,
        assumptions_md,
        routine_md,
        focus_verbs_md,
        build_results_rows(plan.cards),
        str(bundle.preview_audio_path),
        str(bundle.zip_path),
        [str(path) for path in bundle.audio_paths],
    )


def build_app() -> gr.Blocks:
    load_environment()
    default_prompt = (
        "I work from home, buy groceries in person, "
        "chat with neighbors, order food, ask for help when traveling, talk about my schedule, "
        "and handle simple errands with shops, taxis, and family."
    )

    with gr.Blocks(title=APP_TITLE) as demo:
        with gr.Column(elem_id="app-shell"):
            gr.HTML(build_flag_ribbon_html(), elem_id="flag-ribbon")

            with gr.Row(elem_id="hero-grid", equal_height=True):
                with gr.Column(scale=7, elem_classes="hero-copy"):
                    gr.Markdown(
                        """
                        <div class="hero-kicker">Bright language practice studio</div>
                        """.strip()
                    )
                    gr.Markdown(f"# {APP_TITLE}")
                    gr.Markdown(APP_DESCRIPTION)
                    gr.HTML(build_hero_chips_html())
                    with gr.Accordion("Model stack", open=False):
                        model_stack_output = gr.Markdown(build_model_stack_md(TARGET_LANGUAGE))

                with gr.Column(scale=5, elem_classes="hero-card-wrap"):
                    gr.HTML(build_hero_sidecard_html())

            gr.HTML(build_highlights_html(), elem_id="practice-highlights")

            with gr.Row(equal_height=True):
                with gr.Column(scale=5, elem_classes="builder-panel"):
                    gr.HTML(
                        build_panel_heading_html(
                            "Describe your world",
                            "Feed the app the conversations you actually expect, then let it generate a study pack around them.",
                        )
                    )
                    use_cases = gr.Textbox(
                        label="Describe your general use cases",
                        lines=8,
                        value=default_prompt,
                        placeholder="Explain the conversations and situations you expect in daily life.",
                    )
                    gr.Markdown(
                        "Bring in scenes like commuting, shopping, school pickup, appointments, travel, home life, or work."
                    )

                with gr.Column(scale=3, elem_classes="control-panel"):
                    gr.HTML(
                        build_panel_heading_html(
                            "Choose your practice setup",
                            "Mix a target language, source language, and sentence count for a pack sized to your day.",
                        )
                    )
                    gr.HTML(build_control_tip_html())
                    target_language = gr.Dropdown(
                        choices=get_supported_language_labels(),
                        value=TARGET_LANGUAGE,
                        label="Target language",
                        info="Select the language you want to practice hearing and speaking.",
                    )
                    native_language = gr.Dropdown(
                        choices=get_native_language_choices(),
                        value="English",
                        label="Source language",
                        info="Translations and explanations are grounded in this language.",
                    )
                    sentence_count = gr.Slider(
                        minimum=20,
                        maximum=100,
                        value=20,
                        step=1,
                        label="Sentence count",
                        info=f"Audio is grouped into WAV files of up to {SENTENCES_PER_AUDIO_FILE} sentences each.",
                    )
                    build_button = gr.Button("Build study pack", variant="primary")
                    gr.Markdown(
                        '<div class="examples-note">Try a routine-focused prompt first, then widen the sentence count once the tone feels right.</div>'
                    )

            with gr.Column(elem_id="results-shell"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, elem_classes="status-panel"):
                        gr.HTML(
                            build_panel_heading_html(
                                "Build status",
                                "See whether the current pack completed and what assets were generated.",
                            )
                        )
                        status_output = gr.Markdown(label="Status", elem_id="status-output")
                    with gr.Column(scale=1, elem_classes="status-panel"):
                        gr.HTML(
                            build_panel_heading_html(
                                "Focus verbs",
                                "Quick anchors for repetition before you review the full table.",
                            )
                        )
                        focus_verbs_output = gr.Markdown(label="Focus verbs", elem_id="focus-verbs-output")

                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, elem_classes="results-panel"):
                        gr.HTML(
                            build_panel_heading_html(
                                "Practice routine",
                                "A compact 45-minute loop you can run daily with the generated material.",
                            )
                        )
                        routine_output = gr.Markdown(label="Study routine", elem_id="routine-output")
                    with gr.Column(scale=1, elem_classes="results-panel"):
                        gr.HTML(
                            build_panel_heading_html(
                                "Notes and assumptions",
                                "Review what the planner inferred from your prompt before re-running with edits.",
                            )
                        )
                        assumptions_output = gr.Markdown(label="Assumptions", elem_id="assumptions-output")

                with gr.Column(elem_classes="results-panel"):
                    gr.HTML(
                        build_panel_heading_html(
                            "Generated sentence pack",
                            "Scan the scenarios, compare source and target phrasing, and isolate the verbs worth drilling.",
                        )
                    )
                    table_output = gr.Dataframe(
                        headers=[
                            "Scenario",
                            "Source sentence",
                            "Target sentence",
                            "Verb",
                        ],
                        datatype=["str", "str", "str", "str"],
                        row_count=12,
                        column_count=4,
                        interactive=False,
                        wrap=False,
                        label="Generated sentence pack",
                    )

                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, elem_classes="results-panel"):
                        gr.HTML(
                            build_panel_heading_html(
                                "Preview audio",
                                "Listen to the first generated track before downloading the full bundle.",
                            )
                        )
                        preview_audio = gr.Audio(
                            label="Preview the first generated audio track",
                            elem_id="preview-audio",
                        )
                    with gr.Column(scale=1, elem_classes="results-panel"):
                        gr.HTML(
                            build_panel_heading_html(
                                "Downloads",
                                "Grab the ZIP bundle or use the individual WAV files directly.",
                            )
                        )
                        zip_output = gr.File(label="Download the full study pack ZIP")
                        audio_files = gr.File(label="Generated audio tracks", file_count="multiple")

            with gr.Column(elem_classes="builder-panel"):
                gr.HTML(
                    build_panel_heading_html(
                        "Example prompts",
                        "Use one of these cheerful everyday scenarios to seed your first pack quickly.",
                    )
                )
                gr.Examples(
                    examples=[
                        [
                            "I need French for grocery shopping, greeting neighbors, going to the doctor, and asking simple travel questions.",
                            "French",
                            "English",
                            20,
                        ],
                        [
                            "I want French for talking to parents at school pickup, ordering coffee, texting friends, and making weekend plans.",
                            "French",
                            "English",
                            20,
                        ],
                    ],
                    inputs=[use_cases, target_language, native_language, sentence_count],
                )

            target_language.change(
                fn=build_model_stack_md,
                inputs=[target_language],
                outputs=[model_stack_output],
            )

            build_button.click(
                fn=run_pack_builder,
                inputs=[use_cases, target_language, native_language, sentence_count],
                outputs=[
                    status_output,
                    assumptions_output,
                    routine_output,
                    focus_verbs_output,
                    table_output,
                    preview_audio,
                    zip_output,
                    audio_files,
                ],
            )

    return demo


demo = build_app()


def main() -> None:
    demo.launch(theme=APP_THEME, css=APP_CSS, server_port=resolve_server_port())


if __name__ == "__main__":
    main()

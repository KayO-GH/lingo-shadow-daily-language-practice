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
    primary_hue="emerald",
    secondary_hue="amber",
    neutral_hue="stone",
).set(
    body_background_fill="linear-gradient(180deg, #f6efe5 0%, #eef6ef 100%)",
    block_background_fill="rgba(255, 252, 247, 0.92)",
    block_border_color="#dbcbb7",
    button_primary_background_fill="linear-gradient(135deg, #0f766e 0%, #15803d 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #115e59 0%, #166534 100%)",
)

APP_CSS = """
#app-shell {
    max-width: 1180px;
    margin: 0 auto;
}
#hero {
    background: linear-gradient(135deg, rgba(15, 118, 110, 0.10) 0%, rgba(217, 119, 6, 0.10) 100%);
    border: 1px solid rgba(120, 113, 108, 0.16);
    border-radius: 20px;
    padding: 1.2rem 1.2rem 0.4rem 1.2rem;
    margin-bottom: 1rem;
}
#hero h1 {
    font-size: 2rem;
    margin-bottom: 0.3rem;
}
#status-output, #assumptions-output {
    min-height: 3rem;
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
"""

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
            with gr.Column(elem_id="hero"):
                gr.Markdown(f"# {APP_TITLE}")
                gr.Markdown(APP_DESCRIPTION)
                with gr.Accordion("Model stack", open=False):
                    model_stack_output = gr.Markdown(build_model_stack_md(TARGET_LANGUAGE))

            with gr.Row():
                with gr.Column(scale=5):
                    use_cases = gr.Textbox(
                        label="Describe your general use cases",
                        lines=8,
                        value=default_prompt,
                        placeholder="Explain the conversations and situations you expect in daily life.",
                    )
                with gr.Column(scale=2):
                    target_language = gr.Dropdown(
                        choices=get_supported_language_labels(),
                        value=TARGET_LANGUAGE,
                        label="Target language",
                    )
                    native_language = gr.Dropdown(
                        choices=get_native_language_choices(),
                        value="English",
                        label="Source language",
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

            with gr.Row():
                status_output = gr.Markdown(label="Status", elem_id="status-output")
            with gr.Accordion("Notes", open=False):
                assumptions_output = gr.Markdown(label="Assumptions", elem_id="assumptions-output")
            routine_output = gr.Markdown(label="Study routine")
            focus_verbs_output = gr.Markdown(label="Focus verbs")
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

            with gr.Row():
                preview_audio = gr.Audio(
                    label="Preview the first generated audio track",
                    elem_id="preview-audio",
                )
                zip_output = gr.File(label="Download the full study pack ZIP")

            audio_files = gr.File(label="Generated audio tracks", file_count="multiple")

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


if __name__ == "__main__":
    demo.launch(theme=APP_THEME, css=APP_CSS, server_port=resolve_server_port())

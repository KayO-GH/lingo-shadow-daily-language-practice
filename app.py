"""Gradio entrypoint for the Daily Language Practice app."""

from __future__ import annotations

import os

import gradio as gr

from study_pack import (
    FRENCH_ONLY_ERROR,
    GeneratedStudyPlan,
    HF_GENERATION_MODEL,
    HF_GENERATION_PARAMS,
    MODAL_TTS_MODEL,
    MODAL_TTS_PARAMS,
    SENTENCES_PER_AUDIO_FILE,
    TARGET_LANGUAGE,
    build_results_rows,
    create_study_pack,
    generate_sentence_cards,
    get_native_language_choices,
    load_environment,
)

APP_TITLE = "Daily Language Practice"
APP_DESCRIPTION = """
Turn your real daily routines into a practical French study system.
Describe the situations you actually live in and the app will build high-frequency French sentences, focus verbs, a 45-minute routine, and downloadable audio tracks generated through a dedicated Modal-backed French TTS service.
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
"""

MODEL_STACK_MD = (
    "### Model stack\n"
    f"- Generation: `{HF_GENERATION_MODEL}` ({HF_GENERATION_PARAMS:,} params)\n"
    f"- TTS: `{MODAL_TTS_MODEL}` (~{MODAL_TTS_PARAMS:,} params) via Modal\n"
    f"- Total: **~{HF_GENERATION_PARAMS + MODAL_TTS_PARAMS:,}** params\n"
    "- Current scope: **French-only audio in v1**"
)


def resolve_server_port() -> int | None:
    explicit = os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT")
    return int(explicit) if explicit else None


def run_pack_builder(
    use_cases: str,
    native_language: str,
    sentence_count: int,
    slow_audio: bool,
):
    load_environment()
    cleaned_use_cases = (use_cases or "").strip()
    if len(cleaned_use_cases) < 20:
        raise gr.Error("Describe your daily use cases in a bit more detail so the pack can be personalized.")

    try:
        plan: GeneratedStudyPlan = generate_sentence_cards(
            use_cases=cleaned_use_cases,
            target_language=TARGET_LANGUAGE,
            native_language=native_language,
            sentence_count=sentence_count,
        )
        bundle = create_study_pack(
            cards=plan.cards,
            target_language=TARGET_LANGUAGE,
            focus_verbs=plan.focus_verbs,
            routine_steps=plan.routine_steps,
            slow_audio=slow_audio,
        )
    except ValueError as exc:
        if str(exc) == FRENCH_ONLY_ERROR:
            raise gr.Error(str(exc)) from exc
        raise
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc

    status_md = (
        f"Built **{len(plan.cards)}** study sentences for **{TARGET_LANGUAGE}** and generated "
        f"**{len(bundle.audio_paths)}** downloadable audio track(s) with up to "
        f"**{SENTENCES_PER_AUDIO_FILE}** sentences per file using "
        f"`{HF_GENERATION_MODEL}` plus **{bundle.tts_backend_label}**."
    )
    rationale_md = f"### Pack logic\n{plan.rationale}"
    assumptions_md = "### Working assumptions\n" + "\n".join(f"- {item}" for item in plan.assumptions)
    routine_md = "### 45-minute routine\n" + "\n".join(
        f"- **{step.minutes} min:** {step.title} - {step.instructions}" for step in plan.routine_steps
    )
    focus_verbs_md = "### Focus verbs\n" + ", ".join(plan.focus_verbs)

    return (
        status_md,
        rationale_md,
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
        "I want to learn French for daily life. I work from home, buy groceries in person, "
        "chat with neighbors, order food, ask for help when traveling, talk about my schedule, "
        "and handle simple errands with shops, taxis, and family."
    )

    with gr.Blocks(title=APP_TITLE) as demo:
        with gr.Column(elem_id="app-shell"):
            with gr.Column(elem_id="hero"):
                gr.Markdown(f"# {APP_TITLE}")
                gr.Markdown(APP_DESCRIPTION)
                with gr.Accordion("Model stack", open=False):
                    gr.Markdown(MODEL_STACK_MD)

            with gr.Row():
                with gr.Column(scale=5):
                    use_cases = gr.Textbox(
                        label="Describe your general use cases",
                        lines=8,
                        value=default_prompt,
                        placeholder="Explain the French conversations and situations you expect in daily life.",
                    )
                with gr.Column(scale=2):
                    gr.Textbox(
                        value=TARGET_LANGUAGE,
                        label="Target language",
                        interactive=False,
                    )
                    native_language = gr.Dropdown(
                        choices=get_native_language_choices(),
                        value="English",
                        label="Source language",
                    )
                    sentence_count = gr.Slider(
                        minimum=20,
                        maximum=80,
                        value=40,
                        step=1,
                        label="Sentence count",
                        info=f"Audio is grouped into WAV files of up to {SENTENCES_PER_AUDIO_FILE} sentences each.",
                    )
                    slow_audio = gr.Checkbox(
                        value=False,
                        label="Use slower pacing for easier shadowing",
                    )
                    build_button = gr.Button("Build French study pack", variant="primary")

            with gr.Row():
                status_output = gr.Markdown(label="Status", elem_id="status-output")
                assumptions_output = gr.Markdown(label="Assumptions", elem_id="assumptions-output")

            rationale_output = gr.Markdown(label="Rationale")
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
                preview_audio = gr.Audio(label="Preview the first generated audio track")
                zip_output = gr.File(label="Download the full study pack ZIP")

            audio_files = gr.File(label="Generated audio tracks", file_count="multiple")

            gr.Examples(
                examples=[
                    [
                        "I need French for grocery shopping, greeting neighbors, going to the doctor, and asking simple travel questions.",
                        "English",
                        20,
                        False,
                    ],
                    [
                        "I want French for talking to parents at school pickup, ordering coffee, texting friends, and making weekend plans.",
                        "English",
                        40,
                        False,
                    ],
                ],
                inputs=[use_cases, native_language, sentence_count, slow_audio],
            )

            build_button.click(
                fn=run_pack_builder,
                inputs=[use_cases, native_language, sentence_count, slow_audio],
                outputs=[
                    status_output,
                    rationale_output,
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

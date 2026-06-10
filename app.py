"""Gradio entrypoint for the Daily Language Practice app."""

from __future__ import annotations

import os
import socket

import gradio as gr

from study_pack import (
    GeneratedStudyPlan,
    SENTENCES_PER_AUDIO_FILE,
    build_results_rows,
    create_study_pack,
    generate_sentence_cards,
    get_supported_language_labels,
    load_environment,
)

APP_TITLE = "Daily Language Practice"
APP_DESCRIPTION = """
Turn your real daily routines into a practical language study system.
Describe the situations you actually live in, choose a target language, and the app will build high-frequency sentences, focus verbs, a 45-minute routine, and downloadable audio tracks.
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


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
        return True


def resolve_server_port() -> int | None:
    explicit = os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT")
    if explicit:
        return int(explicit)

    for candidate in (7860, 7861, 7862, 8860, 8861, 9000):
        if _port_is_available(candidate):
            return candidate

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("0.0.0.0", 0))
            return int(sock.getsockname()[1])
    except OSError:
        return 7860


def run_pack_builder(
    use_cases: str,
    target_language: str,
    native_language: str,
    sentence_count: int,
    slow_audio: bool,
):
    load_environment()
    cleaned_use_cases = (use_cases or "").strip()
    if len(cleaned_use_cases) < 20:
        raise gr.Error("Describe your daily use cases in a bit more detail so the pack can be personalized.")

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
        slow_audio=slow_audio,
    )

    status_md = (
        f"Built **{len(plan.cards)}** study sentences for **{target_language}** and generated "
        f"**{len(bundle.audio_paths)}** downloadable audio track(s) with up to "
        f"**{SENTENCES_PER_AUDIO_FILE}** sentences per file."
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
                        value="French",
                        label="Target language",
                    )
                    native_language = gr.Dropdown(
                        choices=["English", "French", "Spanish", "German", "Portuguese"],
                        value="English",
                        label="Source language",
                    )
                    sentence_count = gr.Slider(
                        minimum=20,
                        maximum=80,
                        value=40,
                        step=1,
                        label="Sentence count",
                        info=f"Audio is grouped into files of up to {SENTENCES_PER_AUDIO_FILE} sentences each.",
                    )
                    slow_audio = gr.Checkbox(
                        value=False,
                        label="Use slower audio for easier shadowing",
                    )
                    build_button = gr.Button("Build study pack", variant="primary")

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
                    "Why useful",
                    "Pronunciation hint",
                ],
                datatype=["str", "str", "str", "str", "str", "str"],
                row_count=12,
                column_count=6,
                interactive=False,
                wrap=True,
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
                        "French",
                        "English",
                        20,
                        False,
                    ],
                    [
                        "I want Spanish for talking to parents at school pickup, ordering coffee, texting friends, and making weekend plans.",
                        "Spanish",
                        "English",
                        40,
                        False,
                    ],
                ],
                inputs=[use_cases, target_language, native_language, sentence_count, slow_audio],
            )

            build_button.click(
                fn=run_pack_builder,
                inputs=[use_cases, target_language, native_language, sentence_count, slow_audio],
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

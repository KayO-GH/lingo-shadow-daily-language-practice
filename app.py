"""Gradio entrypoint for the LingoShadow - Daily Language Practice app."""

from __future__ import annotations

import logging
import os

import gradio as gr

from study_pack import (
    DEFAULT_SENTENCE_COUNT,
    GeneratedStudyPlan,
    HF_GENERATION_MODEL,
    MAX_SENTENCE_COUNT,
    MIN_SENTENCE_COUNT,
    SENTENCES_PER_AUDIO_FILE,
    TARGET_LANGUAGE,
    TRANSLATION_MODEL,
    build_results_rows,
    create_study_pack,
    generate_sentence_cards,
    get_native_language_choices,
    get_supported_language_labels,
    load_environment,
    validate_sentence_count,
    warmup_tts_backend,
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
.hero-action-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    align-items: center;
    margin: 0.2rem 0 0.8rem 0;
}
#motivation-button button,
#motivation-button .gradio-button {
    border-radius: 999px !important;
    border: 1px solid rgba(124, 45, 18, 0.14) !important;
    background: rgba(255, 255, 255, 0.82) !important;
    color: #7c2d12 !important;
    box-shadow: 0 12px 24px rgba(124, 45, 18, 0.08);
}
#motivation-button button:hover,
#motivation-button .gradio-button:hover {
    background: rgba(255, 247, 237, 0.96) !important;
    border-color: rgba(249, 115, 22, 0.24) !important;
}
#motivation-panel {
    margin: 0 0 1rem 0;
}
.motivation-panel {
    border-radius: 22px;
    border: 1px solid rgba(249, 115, 22, 0.16);
    background: linear-gradient(135deg, rgba(255, 247, 237, 0.98) 0%, rgba(240, 249, 255, 0.96) 100%);
    box-shadow: 0 16px 30px rgba(15, 23, 42, 0.06);
    padding: 1rem 1.05rem;
}
.motivation-panel h3 {
    margin: 0 0 0.35rem 0;
    color: #7c2d12 !important;
    font-size: 1.02rem;
}
.motivation-panel p,
.motivation-panel li {
    color: #7c2d12 !important;
}
.motivation-panel ul {
    margin: 0.6rem 0 0.8rem 1.1rem;
    padding: 0;
}
.motivation-panel a {
    color: #0f766e !important;
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
#builder-row,
#results-shell {
    gap: 1rem;
}
#prompt-workspace,
#setup-stack,
#results-tabs-shell {
    border-radius: 22px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(248, 250, 252, 0.96) 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
    padding: 0.9rem;
}
#prompt-workspace {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(255, 251, 235, 0.94) 100%);
}
#setup-stack {
    background: linear-gradient(180deg, rgba(255, 250, 245, 0.98) 0%, rgba(240, 249, 255, 0.96) 100%);
}
.builder-panel,
.builder-panel > .gr-block,
.builder-panel .gr-block,
.control-panel,
.control-panel > .gr-block,
.control-panel .gr-block {
    background: rgba(255, 255, 255, 0.92) !important;
}
.builder-panel .gr-form,
.builder-panel .gr-group,
.control-panel .gr-form,
.control-panel .gr-group {
    background: rgba(255, 255, 255, 0.9) !important;
    border-color: rgba(148, 163, 184, 0.16) !important;
}
#prompt-workspace .gr-block,
#setup-stack .gr-block,
#results-tabs-shell .gr-block {
    background: transparent !important;
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
.builder-panel .prose p,
.builder-panel .prose li,
.control-panel .prose p,
.control-panel .prose li,
.builder-panel .panel-heading p,
.control-panel .panel-heading p,
.builder-panel .wrap,
.control-panel .wrap,
.builder-panel label,
.control-panel label {
    color: #334155 !important;
}
.builder-panel textarea,
.builder-panel input,
.builder-panel select,
.control-panel textarea,
.control-panel input,
.control-panel select,
.builder-panel .scroll-hide,
.control-panel .scroll-hide,
.builder-panel .wrap-inner,
.control-panel .wrap-inner {
    background: #f8fafc !important;
    color: #0f172a !important;
    border-color: rgba(148, 163, 184, 0.28) !important;
}
.builder-panel textarea::placeholder,
.builder-panel input::placeholder,
.control-panel textarea::placeholder,
.control-panel input::placeholder {
    color: #64748b !important;
}
.builder-panel textarea,
.control-panel textarea {
    line-height: 1.45;
}
#use-cases-input textarea,
#use-cases-input .scroll-hide {
    min-height: 220px !important;
}
#use-cases-input label,
#use-cases-input .wrap {
    color: #1e293b !important;
    font-weight: 700;
}
.builder-panel .gradio-textbox,
.builder-panel .gradio-dropdown,
.builder-panel .gradio-slider,
.control-panel .gradio-textbox,
.control-panel .gradio-dropdown,
.control-panel .gradio-slider {
    background: transparent !important;
}
.builder-panel .gradio-dropdown svg,
.control-panel .gradio-dropdown svg,
.control-panel .gradio-slider svg {
    color: #334155 !important;
    stroke: #334155 !important;
}
.control-panel .gradio-button,
.control-panel button {
    box-shadow: 0 10px 24px rgba(249, 115, 22, 0.2);
}
.builder-panel label span,
.control-panel label span,
.gradio-dropdown label span,
.gradio-dropdown label *,
#audio-shell label,
#audio-shell label span,
#audio-shell label *,
#downloads-shell label,
#downloads-shell label span,
#downloads-shell label *,
#preview-audio label span {
    color: #fffdf9 !important;
}
.examples-note {
    margin-top: 0.45rem;
    color: #64748b;
    font-size: 0.9rem;
}
#results-shell .gradio-row,
#results-shell .gradio-column,
#results-tabs-shell .gradio-row,
#results-tabs-shell .gradio-column {
    gap: 0.85rem;
}
#results-tabs-shell [role="tablist"] {
    gap: 0.55rem;
    margin-bottom: 0.8rem;
}
#results-tabs-shell button[role="tab"] {
    border-radius: 999px !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    background: rgba(255, 255, 255, 0.92) !important;
    color: #334155 !important;
    font-weight: 700 !important;
}
#results-tabs-shell button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #fff7ed 0%, #e0f2fe 100%) !important;
    color: #9a3412 !important;
    border-color: rgba(249, 115, 22, 0.24) !important;
}
#results-shell .tabitem,
#results-tabs-shell .tabitem {
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.76);
}
#results-summary {
    margin-bottom: 0.1rem;
}
#results-summary strong {
    display: block;
    color: #0f172a;
    font-size: 1.08rem;
}
#results-summary p {
    margin: 0.2rem 0 0 0;
    color: #475569;
}
#status-output,
#assumptions-output,
#focus-verbs-output,
#routine-output,
#table-shell,
#audio-shell,
#downloads-shell {
    min-height: 0 !important;
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
    box-sizing: border-box;
    overflow: hidden;
    padding: 0.9rem 1rem !important;
}
#results-shell .prose,
#results-shell .prose p,
#results-shell .prose li,
#results-shell .prose span {
    color: #334155 !important;
}
#results-shell .prose {
    margin: 0 !important;
}
#results-shell .prose h1,
#results-shell .prose h2,
#results-shell .prose h3,
#results-shell .prose h4,
#results-shell .prose strong {
    color: #0f172a !important;
}
#results-shell .prose h1,
#results-shell .prose h2,
#results-shell .prose h3,
#results-shell .prose h4 {
    margin-top: 0 !important;
    margin-bottom: 0.65rem !important;
    line-height: 1.18;
}
#results-shell .prose p,
#results-shell .prose li {
    line-height: 1.5;
}
#results-shell .prose > :first-child {
    margin-top: 0 !important;
}
#results-shell .prose > :last-child {
    margin-bottom: 0 !important;
}
#results-shell .prose li::marker {
    color: #64748b !important;
}
#results-shell .prose code {
    color: #f8fafc !important;
    background: #1e293b !important;
    border: 1px solid rgba(15, 23, 42, 0.16);
    border-radius: 6px;
}
.build-progress-card {
    padding: 0.9rem 1rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(255, 247, 237, 0.96) 0%, rgba(224, 242, 254, 0.9) 100%);
    border: 1px solid rgba(249, 115, 22, 0.2);
}
.build-progress-card strong {
    display: block;
    color: #9a3412 !important;
    font-size: 1rem;
    margin-bottom: 0.25rem;
}
.build-progress-card span {
    color: #334155 !important;
}
.build-progress-track {
    position: relative;
    overflow: hidden;
    height: 0.75rem;
    margin: 0.75rem 0 0.45rem 0;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.18);
}
.build-progress-fill {
    height: 100%;
    min-width: 10%;
    border-radius: inherit;
    background: linear-gradient(90deg, #f97316 0%, #f43f5e 48%, #0ea5e9 100%);
    box-shadow: 0 8px 22px rgba(249, 115, 22, 0.28);
    transition: width 240ms ease;
}
.build-progress-fill::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.58), transparent);
    animation: progress-shimmer 1.2s linear infinite;
}
.build-progress-percent {
    font-size: 0.86rem;
    font-weight: 800;
    color: #0f172a !important;
}
#results-shell .gradio-dataframe,
#results-shell .gradio-file,
#results-shell .gradio-audio {
    border-radius: 18px;
}
#audio-shell label,
#audio-shell label span,
#audio-shell label *,
#downloads-shell label,
#downloads-shell label span,
#downloads-shell label *,
#audio-shell .file-preview-header,
#downloads-shell .file-preview-header {
    color: #fffdf9 !important;
}
#audio-shell,
#downloads-shell,
#table-shell {
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.92);
    padding: 0.2rem;
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
@keyframes progress-shimmer {
    0% {
        transform: translateX(-100%);
    }
    100% {
        transform: translateX(100%);
    }
}
@media (max-width: 900px) {
    .highlights-grid,
    .passport-stack {
        grid-template-columns: 1fr;
    }
    #use-cases-input textarea,
    #use-cases-input .scroll-hide {
        min-height: 180px !important;
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


def build_motivation_html() -> str:
    return """
    <div class="motivation-panel">
        <h3>Language shadowing is how kids learn naturally</h3>
        <p>The idea is simple:</p>
        <ul>
            <li>Practice useful phrases from your real routines instead of generic textbook filler.</li>
            <li>Study with sentences, audio, and repetition so words stay attached to context. (Grammar comes implicitly)</li>
            <li>Keep the loop small enough to repeat daily: build, listen, shadow, and review.</li>
        </ul>
        <p><a href="https://www.youtube.com/watch?v=zas7awYWp2k" target="_blank" rel="noopener noreferrer">Watch this explainer by Mikel, a 12-language polyglot</a></p>
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
                <strong>ZIP + MP3</strong>
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
            <strong>{len(FLAG_BADGES)} languages and counting</strong>
            <p>Adding new languages as fast as the models can make them available.</p>
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


def build_panel_heading_html(title: str, description: str|None = None) -> str:
    description_html = f"<p>{description}</p>" if description else ""
    return f"""
    <div class="panel-heading">
        <strong>{title}</strong>
        {description_html}
    </div>
    """.strip()


def toggle_motivation_panel(is_visible: bool):
    next_visible = not is_visible
    button_label = "Hide project motivation" if next_visible else "Why this project?"
    return gr.update(visible=next_visible), next_visible, gr.update(value=button_label)


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


def build_progress_state(stage: str, detail: str, percent: int):
    bounded_percent = max(0, min(100, percent))
    status_md = f"""
<div class="build-progress-card">
  <strong>{stage}</strong>
  <span>{detail}</span>
  <div class="build-progress-track" aria-label="Build progress">
    <div class="build-progress-fill" style="width: {bounded_percent}%"></div>
  </div>
  <div class="build-progress-percent">{bounded_percent}% complete</div>
</div>
""".strip()

    return (
        status_md,
        "### Working assumptions\nWaiting for the planner to finish.",
        "### 45-minute routine\nThe routine will appear after sentence generation completes.",
        "### Focus verbs\nFocus verbs will appear after generation.",
        [],
        None,
        None,
        [],
    )


def build_success_state(
    plan: GeneratedStudyPlan,
    target_language: str,
    bundle,
):
    status_md = (
        f"Built **{len(plan.cards)}** study sentences for **{target_language}** and generated "
        f"**{len(bundle.audio_paths)}** downloadable audio track(s) with up to "
        f"**{SENTENCES_PER_AUDIO_FILE}** sentences per file using "
        f"`{HF_GENERATION_MODEL}` for pack generation, `{TRANSLATION_MODEL}` for translation, "
        f"with `{bundle.tts_backend_label}` for audio synthesis."
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


def warmup_selected_language(target_language: str) -> None:
    try:
        warmup_tts_backend(target_language)
    except Exception:  # noqa: BLE001
        logger.warning("TTS warmup failed for %s", target_language, exc_info=True)


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
        sentence_count = validate_sentence_count(sentence_count)
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

    return build_success_state(plan, target_language, bundle)


def stream_pack_builder(
    use_cases: str,
    target_language: str,
    native_language: str,
    sentence_count: int,
    progress=gr.Progress(track_tqdm=True),
):
    load_environment()
    cleaned_use_cases = (use_cases or "").strip()
    if len(cleaned_use_cases) < 20:
        yield build_error_state(
            "Describe your daily use cases in a bit more detail so the pack can be personalized."
        )
        return

    try:
        sentence_count = validate_sentence_count(sentence_count)
    except ValueError as exc:
        logger.exception("Validation failure while building the study pack")
        yield build_error_state(str(exc))
        return

    try:
        progress(0.05, desc="Starting study pack build")
        yield build_progress_state(
            "Starting build",
            "Preparing the prompt, model settings, and output workspace.",
            5,
        )

        progress(0.35, desc="Generating and translating sentences")
        yield build_progress_state(
            "Generating sentence pack",
            f"Creating {sentence_count} practical sentences, translations, focus verbs, and routine notes.",
            35,
        )
        plan: GeneratedStudyPlan = generate_sentence_cards(
            use_cases=cleaned_use_cases,
            target_language=target_language,
            native_language=native_language,
            sentence_count=sentence_count,
        )

        progress(0.75, desc="Creating audio and downloads")
        yield build_progress_state(
            "Creating audio tracks",
            f"Synthesizing MP3 audio and packaging downloads for {len(plan.cards)} sentences.",
            75,
        )
        bundle = create_study_pack(
            cards=plan.cards,
            target_language=target_language,
            focus_verbs=plan.focus_verbs,
            routine_steps=plan.routine_steps,
            slow_audio=True,
        )

        progress(1.0, desc="Study pack ready")
        yield build_success_state(plan, target_language, bundle)
    except ValueError as exc:
        logger.exception("Validation failure while building the study pack")
        yield build_error_state(str(exc))
    except RuntimeError as exc:
        yield build_error_state(str(exc))
    except Exception:
        logger.exception("Unexpected failure while building the study pack")
        yield build_error_state("An unexpected error stopped the build. Check the terminal logs, then try again.")


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
                    motivation_open = gr.State(False)
                    with gr.Row(elem_classes="hero-action-row"):
                        motivation_button = gr.Button(
                            "Why this works...",
                            variant="secondary",
                            elem_id="motivation-button",
                        )
                    with gr.Column(visible=False, elem_id="motivation-panel") as motivation_panel:
                        gr.HTML(build_motivation_html())

                with gr.Column(scale=5, elem_classes="hero-card-wrap"):
                    gr.HTML(build_hero_sidecard_html())

            gr.HTML(build_highlights_html(), elem_id="practice-highlights")

            with gr.Row(elem_id="builder-row"):
                with gr.Column(scale=5, elem_classes="builder-panel"):
                    gr.HTML(
                        build_panel_heading_html(
                            "Describe your world",
                            "Feed the app the conversations you actually expect, then let it generate a study pack around them. Add situations you actually live through. The more concrete the prompt, the more useful the sentence pack and audio drills become.",
                        )
                    )
                    with gr.Column(elem_id="prompt-workspace"):
                        use_cases = gr.Textbox(
                            label="Describe your general use cases",
                            lines=6,
                            value=default_prompt,
                            placeholder="Explain the conversations and situations you expect in daily life.",
                            elem_id="use-cases-input",
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
                    with gr.Column(elem_id="setup-stack"):
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
                            minimum=MIN_SENTENCE_COUNT,
                            maximum=MAX_SENTENCE_COUNT,
                            value=DEFAULT_SENTENCE_COUNT,
                            step=1,
                            label="Sentence count",
                            info=f"Audio is grouped into MP3 files of up to {SENTENCES_PER_AUDIO_FILE} sentences each.",
                        )
                        build_button = gr.Button("Build study pack", variant="primary")
                        gr.Markdown(
                            '<div class="examples-note">Try a routine-focused prompt first, then widen the sentence count once the tone feels right.</div>'
                        )

            with gr.Column(elem_id="results-shell"):
                with gr.Column(elem_classes="status-panel"):
                    gr.HTML(
                        build_panel_heading_html(
                            "Build status",
                        )
                    )
                    status_output = gr.Markdown(
                        "Build a pack to see generation status, file counts, and stack details.",
                        label="Status",
                        elem_id="status-output",
                    )
                gr.Markdown(
                    """
                    <div id="results-summary">
                        <strong>Study pack workspace</strong>
                        <p>Generate once, then switch between the study pack and pack details without dragging through empty panels.</p>
                    </div>
                    """.strip()
                )
                with gr.Column(elem_id="results-tabs-shell"):
                    with gr.Tabs():
                        with gr.Tab("Study Pack"):
                            with gr.Column(elem_id="table-shell"):
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
                                    row_count=8,
                                    column_count=4,
                                    interactive=False,
                                    wrap=False,
                                    label="Generated sentence pack",
                                )

                            with gr.Row():
                                with gr.Column(scale=2, elem_id="audio-shell"):
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
                                with gr.Column(scale=1, elem_id="downloads-shell"):
                                    gr.HTML(
                                        build_panel_heading_html(
                                            "Downloads",
                                            "Grab the ZIP bundle or use the individual MP3 files directly.",
                                        )
                                    )
                                    zip_output = gr.File(label="Download the full study pack ZIP")
                                    audio_files = gr.File(label="Generated audio tracks", file_count="multiple")

                        with gr.Tab("Pack Details"):
                            with gr.Row():
                                with gr.Column(scale=1, elem_classes="status-panel"):
                                    gr.HTML(
                                        build_panel_heading_html(
                                            "Focus verbs",
                                            "Quick anchors for repetition before you review the full table.",
                                        )
                                    )
                                    focus_verbs_output = gr.Markdown(
                                        "Focus verbs will appear here after generation.",
                                        label="Focus verbs",
                                        elem_id="focus-verbs-output",
                                    )

                            with gr.Row():
                                with gr.Column(scale=2, elem_classes="results-panel"):
                                    gr.HTML(
                                        build_panel_heading_html(
                                            "Practice routine",
                                            "A compact 45-minute loop you can run daily with the generated material.",
                                        )
                                    )
                                    routine_output = gr.Markdown(
                                        "Your generated routine will land here once the pack is ready.",
                                        label="Study routine",
                                        elem_id="routine-output",
                                    )
                                with gr.Column(scale=1, elem_classes="results-panel"):
                                    gr.HTML(
                                        build_panel_heading_html(
                                            "Notes and assumptions",
                                            "Review what the planner inferred from your prompt before re-running with edits.",
                                        )
                                    )
                                    assumptions_output = gr.Markdown(
                                        "Assumptions and pack logic will appear here.",
                                        label="Assumptions",
                                        elem_id="assumptions-output",
                                    )

            motivation_button.click(
                fn=toggle_motivation_panel,
                inputs=[motivation_open],
                outputs=[motivation_panel, motivation_open, motivation_button],
            )
            target_language.change(
                fn=warmup_selected_language,
                inputs=[target_language],
                outputs=None,
                queue=False,
                show_progress="hidden",
            )

            demo.load(
                fn=warmup_selected_language,
                inputs=[target_language],
                outputs=None,
                queue=False,
                show_progress="hidden",
            )

            build_button.click(
                fn=stream_pack_builder,
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

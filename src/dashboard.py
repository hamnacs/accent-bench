import streamlit as st
import pandas as pd
import plotly.express as px
import whisper
import soundfile as sf
import numpy as np
import tempfile
import os
import time

st.set_page_config(page_title="AccentBench", layout="wide")

st.title("AccentBench")
st.caption("Measuring accent, language-ID, and code-switching robustness in Whisper ASR")

tab1, tab2, tab3 = st.tabs(["Accent Robustness", "Code-Switching (Hindi-English)", "Try It Yourself"])

# =========================================================
# TAB 1 — ACCENT ROBUSTNESS
# =========================================================
with tab1:
    @st.cache_data
    def load_accent_results():
        return pd.read_csv("results/wer_results_full.csv")

    @st.cache_data
    def load_accent_results_small():
        return pd.read_csv("results/wer_results_small.csv")

    @st.cache_data
    def load_language_switch():
        return pd.read_csv("results/language_switch_examples.csv")

    @st.cache_data
    def load_forced_comparison():
        return pd.read_csv("results/before_after_comparison.csv")

    df_base = load_accent_results()
    df_small = load_accent_results_small()
    switch_df = load_language_switch()
    comparison_df = load_forced_comparison()

    st.header("1. Word Error Rate by Accent")

    summary_base = df_base.groupby("accent")["wer"].mean().reset_index().rename(columns={"wer": "WER (base)"})
    summary_small = df_small.groupby("accent")["wer"].mean().reset_index().rename(columns={"wer": "WER (small)"})
    summary = (
        summary_base.merge(summary_small, on="accent")
        .sort_values("WER (base)")
        .reset_index(drop=True)
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Models compared", "Whisper-base vs small")
    col2.metric("Clips per accent", "~100")
    col3.metric("Accent groups", df_base["accent"].nunique())

    # Sort the melted data using the same accent order as the sorted summary table,
    # so the chart order visually matches the table below it
    accent_order = summary["accent"].tolist()
    melted = summary.melt(id_vars="accent", var_name="model", value_name="wer")
    melted["accent"] = pd.Categorical(melted["accent"], categories=accent_order, ordered=True)
    melted = melted.sort_values("accent")

    fig = px.bar(
        melted, x="accent", y="wer", color="model", barmode="group",
        title="Mean WER by Accent: Base vs Small",
        labels={"wer": "Mean WER", "accent": "Accent"},
        category_orders={"accent": accent_order},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(summary.style.format({"WER (base)": "{:.3f}", "WER (small)": "{:.3f}"}))

    st.header("2. Language Misidentification Rate")
    st.write(
        "A more severe failure mode than mistranscription: Whisper sometimes misidentifies "
        "the spoken language entirely, producing non-English output."
    )

    switch_summary = switch_df.groupby("accent").size().reset_index(name="language_switch_count")
    totals = df_base.groupby("accent").size().reset_index(name="total")
    switch_summary = switch_summary.merge(totals, on="accent", how="right").fillna(0)
    switch_summary["language_switch_rate_%"] = (
        switch_summary["language_switch_count"] / switch_summary["total"] * 100
    ).round(1)
    switch_summary = (
        switch_summary.sort_values("language_switch_rate_%", ascending=False)
        .reset_index(drop=True)
    )

    fig2 = px.bar(
        switch_summary, x="accent", y="language_switch_rate_%",
        title="Language Misidentification Rate by Accent (%)",
        color="language_switch_rate_%", color_continuous_scale="Oranges",
        category_orders={"accent": switch_summary["accent"].tolist()},
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(switch_summary)

    with st.expander("See flagged examples (Ground truth vs Prediction)"):
        st.dataframe(switch_df[["accent", "ground_truth", "prediction", "wer"]])

    st.header("3. Forcing English Recovers Accuracy")
    st.write(
        "For clips where Whisper misidentified the language, forcing language detection "
        "to English shows how much error was caused by language-ID failure vs genuine "
        "transcription difficulty."
    )

    before_after_melt = comparison_df.melt(
        id_vars=["accent", "ground_truth"],
        value_vars=["wer_original", "wer_forced_en"],
        var_name="condition", value_name="wer"
    )
    before_after_melt["condition"] = before_after_melt["condition"].map({
        "wer_original": "Original (auto-detected language)",
        "wer_forced_en": "Forced English"
    })

    fig3 = px.box(
        before_after_melt, x="condition", y="wer", points="all",
        title="WER Before vs After Forcing English (language-switch cases only)",
        color="condition"
    )
    st.plotly_chart(fig3, use_container_width=True)

    mean_before = comparison_df["wer_original"].mean()
    mean_after = comparison_df["wer_forced_en"].mean()
    col1, col2 = st.columns(2)
    col1.metric("Mean WER (original)", f"{mean_before:.3f}")
    col2.metric("Mean WER (forced English)", f"{mean_after:.3f}", delta=f"{mean_after - mean_before:.3f}")

    st.dataframe(
        comparison_df[["accent", "ground_truth", "wer_original", "wer_forced_en"]]
        .reset_index(drop=True)
    )

# =========================================================
# TAB 2 — CODE-SWITCHING
# =========================================================
with tab2:
    st.header("Hindi-English Code-Switching")
    st.write(
        "Evaluated on the MUCS 2021 Hindi-English test set, used as a proxy for "
        "Urdu-English speech due to the absence of a public Urdu-English speech corpus."
    )

    @st.cache_data
    def load_codeswitch_base():
        return pd.read_csv("results/codeswitch_results.csv")

    @st.cache_data
    def load_codeswitch_small():
        return pd.read_csv("results/codeswitch_small_results.csv")

    cs_base = load_codeswitch_base()
    cs_small = load_codeswitch_small()

    st.subheader("1. WER: Code-Switching vs Best/Worst Accent Groups")

    accent_best = df_base.groupby("accent")["wer"].mean().min()
    accent_worst = df_base.groupby("accent")["wer"].mean().max()

    comparison = pd.DataFrame({
        "Condition": ["Best accent (German)", "Worst accent (Singaporean)",
                       "Code-switching (base)", "Code-switching (small)"],
        "Mean WER": [accent_best, accent_worst, cs_base["wer"].mean(), cs_small["wer"].mean()]
    })

    fig4 = px.bar(
        comparison, x="Condition", y="Mean WER",
        title="Code-Switching WER Is Far Higher Than Any Single-Accent Group",
        color="Mean WER", color_continuous_scale="Reds",
        category_orders={"Condition": comparison["Condition"].tolist()},
    )
    st.plotly_chart(fig4, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Mean WER (base)", f"{cs_base['wer'].mean():.3f}")
    col2.metric("Mean WER (small)", f"{cs_small['wer'].mean():.3f}",
                delta=f"{cs_small['wer'].mean() - cs_base['wer'].mean():.3f}")

    st.subheader("2. Transliteration Failure Pattern")
    st.write(
        "In most utterances containing embedded English terms, Whisper-small transliterated "
        "those terms into phonetic Devanagari instead of recognizing them as English."
    )

    try:
        translit_df = pd.read_csv("results/transliteration_failures.csv")
        st.metric("Transliteration failure rate", "87.9%", help="Of utterances containing English words in the ground truth")
        with st.expander("See transliteration failure examples"):
            st.dataframe(translit_df.reset_index(drop=True))
    except FileNotFoundError:
        st.info("Run `tally_codeswitch_patterns.py` to generate this data.")

    st.subheader("3. Detected Language Distribution")
    lang_dist = cs_small["detected_language"].value_counts().reset_index()
    lang_dist.columns = ["language", "count"]
    fig5 = px.pie(
        lang_dist, names="language", values="count",
        title="Whisper-small Detected Language (Code-Switched Audio)",
        color="language",
        color_discrete_map={"hi": "#63b3ed", "en": "#2b6cb0", "ur": "#feb2b2", "error": "#e53e3e"},
    )
    st.plotly_chart(fig5, use_container_width=True)

    with st.expander("See raw code-switching transcriptions"):
        st.dataframe(
            cs_small[["utt_id", "ground_truth", "prediction", "detected_language", "wer"]]
            .reset_index(drop=True)
        )

# =========================================================
# TAB 3 — LIVE DEMO
# =========================================================
with tab3:
    st.header("Try It Yourself")
    st.write("Upload a short WAV/MP3 clip and see how Whisper transcribes it.")

    @st.cache_resource
    def load_model():
        return whisper.load_model("base")

    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
    force_english = st.checkbox("Force English language detection", value=False)

    if uploaded is not None:
        model = load_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_array, sr = sf.read(uploaded, dtype="float32")
            sf.write(tmp.name, audio_array, sr)
            tmp_path = tmp.name

        with st.spinner("Transcribing..."):
            kwargs = {"fp16": False}
            if force_english:
                kwargs["language"] = "en"
            result = model.transcribe(tmp_path, **kwargs)

        st.audio(uploaded)
        st.write("**Detected language:**", result.get("language", "unknown"))
        st.write("**Transcription:**")
        st.success(result["text"])

        for attempt in range(5):
            try:
                os.remove(tmp_path)
                break
            except PermissionError:
                time.sleep(0.2)
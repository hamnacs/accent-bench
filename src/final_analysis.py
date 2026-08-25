from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path("results")
FINAL = RESULTS / "final"
PLOTS = RESULTS / "plots"

FINAL.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD FILES
# ============================================================

files = {
    "Urdu Base Auto": (RESULTS / "urdu_base_results.csv", "audio"),
    "Urdu Base Forced": (RESULTS / "urdu_base_forced.csv", "audio"),
    "Urdu Small Auto": (RESULTS / "urdu_small_matched.csv", "audio"),
    "Urdu Small Forced": (RESULTS / "forced_urdu_results.csv", "audio"),

    "Korean Base Auto": (RESULTS / "korean_base_auto.csv", "id"),
    "Korean Base Forced": (RESULTS / "korean_base_forced.csv", "id"),
    "Korean Small Auto": (RESULTS / "korean_small_auto.csv", "id"),
    "Korean Small Forced": (RESULTS / "korean_small_forced.csv", "id"),
}


data = {}

for name, (path, id_column) in files.items():

    print(f"Loading: {path}")

    df = pd.read_csv(path)

    if id_column not in df.columns:
        raise ValueError(
            f"{path} does not contain '{id_column}'. "
            f"Columns found: {df.columns.tolist()}"
        )

    if "wer" not in df.columns:
        raise ValueError(
            f"{path} does not contain 'wer'."
        )

    df = df[[id_column, "wer"]].copy()

    df = df.rename(columns={
        id_column: "clip_id"
    })

    df["clip_id"] = df["clip_id"].astype(str)

    # Keep only successful runs
    df = df.dropna(subset=["wer"])

    data[name] = df


# ============================================================
# FIND COMMON SUCCESSFUL CLIPS
# ============================================================

def common_clips(names):

    common = set(data[names[0]]["clip_id"])

    for name in names[1:]:
        common &= set(data[name]["clip_id"])

    return common


urdu_names = [
    "Urdu Base Auto",
    "Urdu Base Forced",
    "Urdu Small Auto",
    "Urdu Small Forced",
]

korean_names = [
    "Korean Base Auto",
    "Korean Base Forced",
    "Korean Small Auto",
    "Korean Small Forced",
]


urdu_common = common_clips(urdu_names)
korean_common = common_clips(korean_names)


print()
print("=" * 60)
print("COMMON SUCCESSFUL CLIPS")
print("=" * 60)

print(f"Urdu:   {len(urdu_common)}")
print(f"Korean: {len(korean_common)}")


# ============================================================
# CREATE SUMMARY
# ============================================================

summary_rows = []


for language, names, common in [
    ("Urdu", urdu_names, urdu_common),
    ("Korean", korean_names, korean_common),
]:

    for name in names:

        df = data[name]

        values = df[
            df["clip_id"].isin(common)
        ]["wer"]

        if "Base" in name:
            model = "Base"
        else:
            model = "Small"

        if "Forced" in name:
            condition = "Forced"
        else:
            condition = "Automatic"

        summary_rows.append({
            "language": language,
            "model": model,
            "condition": condition,
            "clips": len(values),
            "mean_WER": values.mean(),
            "median_WER": values.median(),
        })


summary = pd.DataFrame(summary_rows)


# ============================================================
# SAVE FINAL COMPARISON
# ============================================================

summary.to_csv(
    FINAL / "final_comparison.csv",
    index=False
)


print()
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# LANGUAGE FORCING EFFECT
# ============================================================

forcing_rows = []


for language in ["Urdu", "Korean"]:

    for model in ["Base", "Small"]:

        auto = summary[
            (summary["language"] == language) &
            (summary["model"] == model) &
            (summary["condition"] == "Automatic")
        ]["mean_WER"].iloc[0]

        forced = summary[
            (summary["language"] == language) &
            (summary["model"] == model) &
            (summary["condition"] == "Forced")
        ]["mean_WER"].iloc[0]

        improvement = ((auto - forced) / auto) * 100

        forcing_rows.append({
            "language": language,
            "model": model,
            "automatic_WER": auto,
            "forced_WER": forced,
            "relative_improvement_percent": improvement,
        })


forcing = pd.DataFrame(forcing_rows)


forcing.to_csv(
    FINAL / "language_forcing_effect.csv",
    index=False
)


print()
print("=" * 60)
print("LANGUAGE FORCING EFFECT")
print("=" * 60)

print(
    forcing.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# GRAPH 1
# ALL CONDITIONS
# ============================================================

labels = [
    f"{row.language} {row.model}\n{row.condition}"
    for row in summary.itertuples()
]

plt.figure(figsize=(12, 6))

plt.bar(
    range(len(summary)),
    summary["mean_WER"]
)

plt.xticks(
    range(len(summary)),
    labels,
    rotation=0
)

plt.ylabel("Mean WER")
plt.title("Whisper WER Across Urdu and Korean")

plt.tight_layout()

plt.savefig(
    PLOTS / "wer_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 2
# AUTOMATIC VS FORCED
# ============================================================

groups = [
    "Urdu Base",
    "Urdu Small",
    "Korean Base",
    "Korean Small",
]

automatic = []
forced = []


for language, model in [
    ("Urdu", "Base"),
    ("Urdu", "Small"),
    ("Korean", "Base"),
    ("Korean", "Small"),
]:

    auto_value = summary[
        (summary["language"] == language) &
        (summary["model"] == model) &
        (summary["condition"] == "Automatic")
    ]["mean_WER"].iloc[0]

    forced_value = summary[
        (summary["language"] == language) &
        (summary["model"] == model) &
        (summary["condition"] == "Forced")
    ]["mean_WER"].iloc[0]

    automatic.append(auto_value)
    forced.append(forced_value)


x = range(len(groups))
width = 0.35


plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    automatic,
    width,
    label="Automatic"
)

plt.bar(
    [i + width / 2 for i in x],
    forced,
    width,
    label="Forced language"
)

plt.xticks(x, groups)

plt.ylabel("Mean WER")
plt.title("Effect of Language Forcing")

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOTS / "language_forcing.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 3
# RELATIVE IMPROVEMENT
# ============================================================

labels = [
    f"{row.language} {row.model}"
    for row in forcing.itertuples()
]

values = forcing[
    "relative_improvement_percent"
]


plt.figure(figsize=(10, 6))

plt.bar(
    labels,
    values
)

plt.ylabel("Relative WER improvement (%)")

plt.title(
    "Relative Improvement from Forced Language"
)

plt.axhline(
    0,
    linewidth=1
)

plt.tight_layout()

plt.savefig(
    PLOTS / "forcing_improvement.png",
    dpi=300
)

plt.close()


# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print()
print("Created:")

print(
    FINAL / "final_comparison.csv"
)

print(
    FINAL / "language_forcing_effect.csv"
)

print(
    PLOTS / "wer_comparison.png"
)

print(
    PLOTS / "language_forcing.png"
)

print(
    PLOTS / "forcing_improvement.png"
)
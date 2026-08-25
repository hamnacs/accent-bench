
"""
AccentBench — HiKE Korean-English benchmark figures

Run from the AccentBench project root:
    python src/make_hike_figures.py

Reads:
    results/hike_analysis/*.csv

Writes:
    results/hike_analysis/figures/*.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

ANALYSIS = ROOT / "results" / "hike_analysis"
FIGURES = ANALYSIS / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)


def save_fig(name: str):
    """Save the current matplotlib figure."""
    path = FIGURES / name

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# 1. Overall WER
# ---------------------------------------------------------------------

overall = pd.read_csv(ANALYSIS / "overall_wer.csv")

order = [
    "base_auto",
    "base_forced_ko",
    "small_auto",
    "small_forced_ko",
]

if "experiment" in overall.columns:
    overall["experiment"] = pd.Categorical(
        overall["experiment"],
        categories=order,
        ordered=True,
    )

    overall = overall.sort_values("experiment")


plt.figure(figsize=(9, 5.5))

bars = plt.bar(
    overall["experiment"].astype(str),
    overall["mean_wer"],
)

plt.ylabel("Mean WER")
plt.xlabel("Experiment")
plt.title("HiKE Korean-English: Overall Word Error Rate")

plt.ylim(
    0,
    max(0.8, overall["mean_wer"].max() * 1.18),
)

for bar, value in zip(bars, overall["mean_wer"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.3f}",
        ha="center",
        va="bottom",
    )

save_fig("01_overall_wer.png")


# ---------------------------------------------------------------------
# 2. WER by code-switching level — best configuration
# ---------------------------------------------------------------------

cs = pd.read_csv(
    ANALYSIS / "best_model_by_cs_level.csv"
)

preferred_order = [
    "word",
    "phrase",
    "sentence",
]

if "cs_level" in cs.columns:
    cs["cs_level"] = pd.Categorical(
        cs["cs_level"],
        categories=preferred_order,
        ordered=True,
    )

    cs = cs.sort_values("cs_level")


plt.figure(figsize=(8, 5.5))

bars = plt.bar(
    cs["cs_level"].astype(str),
    cs["mean_wer"],
)

plt.ylabel("Mean WER")
plt.xlabel("Code-switching level")
plt.title("Best Model WER by Code-Switching Level")

plt.ylim(
    0,
    max(0.6, cs["mean_wer"].max() * 1.2),
)

for bar, value in zip(bars, cs["mean_wer"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.012,
        f"{value:.3f}",
        ha="center",
        va="bottom",
    )

save_fig("02_wer_by_codeswitching_level.png")


# ---------------------------------------------------------------------
# 3. WER by category — best configuration
# ---------------------------------------------------------------------

category = pd.read_csv(
    ANALYSIS / "best_model_by_category.csv"
)

category = category.sort_values(
    "mean_wer",
    ascending=True,
)


plt.figure(figsize=(10, 6.5))

bars = plt.barh(
    category["category"],
    category["mean_wer"],
)

plt.xlabel("Mean WER")
plt.ylabel("Category")
plt.title("Best Model WER by Domain")

plt.xlim(
    0,
    max(0.55, category["mean_wer"].max() * 1.18),
)

for bar, value in zip(
    bars,
    category["mean_wer"],
):
    plt.text(
        value + 0.008,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
    )

save_fig("03_wer_by_category.png")


# ---------------------------------------------------------------------
# 4. English-token preservation
# ---------------------------------------------------------------------

english = pd.read_csv(
    ANALYSIS / "english_token_preservation.csv"
)

if "experiment" in english.columns:
    order_map = {
        name: i
        for i, name in enumerate(order)
    }

    english["_order"] = english["experiment"].map(
        order_map
    )

    english = english.sort_values("_order")


rate_col = "token_preservation_rate"

if rate_col not in english.columns:
    raise KeyError(
        f"Expected '{rate_col}' in "
        "english_token_preservation.csv. "
        f"Found: {english.columns.tolist()}"
    )


plt.figure(figsize=(9, 5.5))

bars = plt.bar(
    english["experiment"].astype(str),
    english[rate_col] * 100,
)

plt.ylabel("English token preservation (%)")
plt.xlabel("Experiment")
plt.title(
    "English Token Preservation in Korean-English Speech"
)

plt.ylim(0, 100)

for bar, value in zip(
    bars,
    english[rate_col] * 100,
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
    )

save_fig("04_english_token_preservation.png")


# ---------------------------------------------------------------------
# 5. Automatic vs forced Korean
# ---------------------------------------------------------------------

comparison = pd.read_csv(
    ANALYSIS / "auto_vs_forced.csv"
)

required_columns = {
    "model",
    "language_mode",
    "clips",
    "mean_wer",
    "median_wer",
}

missing = required_columns - set(
    comparison.columns
)

if missing:
    raise KeyError(
        "auto_vs_forced.csv is missing columns: "
        f"{sorted(missing)}"
    )


models = ["base", "small"]

auto_values = []
forced_values = []

for model in models:

    model_rows = comparison[
        comparison["model"]
        .astype(str)
        .str.lower()
        == model
    ]

    auto_row = model_rows[
        model_rows["language_mode"]
        .astype(str)
        .str.lower()
        == "auto"
    ]

    forced_row = model_rows[
        model_rows["language_mode"]
        .astype(str)
        .str.lower()
        == "forced_ko"
    ]

    if auto_row.empty:
        raise ValueError(
            f"No automatic result found for model: {model}"
        )

    if forced_row.empty:
        raise ValueError(
            f"No forced Korean result found for model: {model}"
        )

    auto_values.append(
        float(auto_row["mean_wer"].iloc[0])
    )

    forced_values.append(
        float(forced_row["mean_wer"].iloc[0])
    )


x = np.arange(len(models))
width = 0.35


fig, ax = plt.subplots(figsize=(8, 5.5))

bars1 = ax.bar(
    x - width / 2,
    auto_values,
    width,
    label="Automatic",
)

bars2 = ax.bar(
    x + width / 2,
    forced_values,
    width,
    label="Forced Korean",
)


ax.set_ylabel("Mean WER")
ax.set_xlabel("Model")
ax.set_title(
    "Automatic vs Forced Korean Language Setting"
)

ax.set_xticks(x)
ax.set_xticklabels(
    ["Whisper Base", "Whisper Small"]
)

ax.set_ylim(
    0,
    max(
        0.8,
        max(forced_values) * 1.18,
    ),
)

ax.legend()


for bars_group in (bars1, bars2):

    for bar in bars_group:

        value = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )


save_fig("05_auto_vs_forced_korean.png")


# ---------------------------------------------------------------------
# Complete
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("HIKE FIGURES COMPLETE")
print("=" * 70)

print(f"Output directory: {FIGURES}")

print("Created:")

for path in sorted(FIGURES.glob("*.png")):
    print(f"  - {path.name}")

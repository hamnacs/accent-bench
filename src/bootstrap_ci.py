from pathlib import Path

import numpy as np
import pandas as pd


N_BOOT = 10000
SEED = 42
RESULTS = Path("results")


def bootstrap_ci(values):
    """Calculate a percentile bootstrap 95% CI for a mean."""
    v = np.asarray(values, dtype=float)

    if len(v) == 0:
        raise ValueError("Cannot bootstrap an empty array")

    if np.isnan(v).any():
        raise ValueError("NaN found in paired differences")

    rng = np.random.RandomState(SEED)

    indices = rng.randint(
        0,
        len(v),
        size=(N_BOOT, len(v))
    )

    bootstrap_means = v[indices].mean(axis=1)

    return np.percentile(
        bootstrap_means,
        [2.5, 97.5]
    )


rows = []


def record(name, direction, expected_n, values):
    v = np.asarray(values, dtype=float)

    if expected_n is not None and len(v) != expected_n:
        raise AssertionError(
            f"{name}: expected n={expected_n}, got n={len(v)}"
        )

    low, high = bootstrap_ci(v)

    rows.append({
        "comparison": name,
        "difference": direction,
        "n": len(v),
        "mean_difference": v.mean(),
        "ci_low": low,
        "ci_high": high,
        "n_boot": N_BOOT,
        "seed": SEED,
        "rng": "np.random.RandomState",
        "ci_method": "percentile",
    })

    print(name)
    print(f"  n = {len(v)}")
    print(f"  mean difference = {v.mean():.4f}")
    print(f"  95% CI = [{low:.4f}, {high:.4f}]")
    print()


# ============================================================
# Track 1: Singaporean English
# ============================================================

before_after = pd.read_csv(
    RESULTS / "before_after_comparison.csv"
)

singaporean = before_after[
    before_after["accent"].str.contains(
        "Singaporean",
        case=False,
        na=False
    )
]

record(
    "Track 1: Singaporean, original minus forced English",
    "original - forced_en",
    14,
    (
        singaporean["wer_original"]
        - singaporean["wer_forced_en"]
    ).values,
)


# ============================================================
# Track 2: Hindi-English Code-Switching
# ============================================================

codeswitch_base = pd.read_csv(
    RESULTS / "codeswitch_results.csv"
)

codeswitch_small = pd.read_csv(
    RESULTS / "codeswitch_small_results.csv"
)

# Keep only rows with valid WER values.
codeswitch_base = codeswitch_base.dropna(
    subset=["wer"]
)

codeswitch_small = codeswitch_small.dropna(
    subset=["wer"]
)

# Match Base and Small by utterance ID.
codeswitch_matched = codeswitch_base[
    ["utt_id", "wer"]
].rename(
    columns={"wer": "base_wer"}
).merge(
    codeswitch_small[
        ["utt_id", "wer"]
    ].rename(
        columns={"wer": "small_wer"}
    ),
    on="utt_id",
    how="inner"
)

print(
    f"Track 2 matched clips: "
    f"{len(codeswitch_matched)}"
)

assert len(codeswitch_base) == 99, (
    f"Expected 99 valid Base clips, "
    f"got {len(codeswitch_base)}"
)

assert len(codeswitch_small) == 98, (
    f"Expected 98 valid Small clips, "
    f"got {len(codeswitch_small)}"
)

assert len(codeswitch_matched) == 97, (
    f"Expected 97 matched Track 2 clips, "
    f"got {len(codeswitch_matched)}"
)

record(
    "Track 2: Hindi-English proxy, Base minus Small",
    "base - small",
    97,
    (
        codeswitch_matched["base_wer"]
        - codeswitch_matched["small_wer"]
    ).values,
)


# ============================================================
# Track 3: HiKE Small
# ============================================================

hike = pd.read_csv(
    RESULTS / "hike_whisper_predictions.csv"
)

hike_auto = hike[
    hike["experiment"] == "small_auto"
][
    ["sample_id", "wer"]
].rename(
    columns={"wer": "auto_wer"}
)

hike_forced = hike[
    hike["experiment"] == "small_forced_ko"
][
    ["sample_id", "wer"]
].rename(
    columns={"wer": "forced_wer"}
)

hike_matched = hike_auto.merge(
    hike_forced,
    on="sample_id"
)

record(
    "Track 3: HiKE Small, forced Korean minus automatic",
    "forced_ko - auto",
    100,
    (
        hike_matched["forced_wer"]
        - hike_matched["auto_wer"]
    ).values,
)


# ============================================================
# Track 4: Urdu
# ============================================================

urdu_files = {
    "base_auto": RESULTS / "urdu_base_results.csv",
    "base_forced": RESULTS / "urdu_base_forced.csv",
    "small_auto": RESULTS / "urdu_small_matched.csv",
    "small_forced": RESULTS / "forced_urdu_results.csv",
}

urdu_wer = {}

for name, path in urdu_files.items():
    data = pd.read_csv(path).dropna(
        subset=["wer"]
    )

    urdu_wer[name] = data.set_index(
        "audio"
    )["wer"]


common_audio = sorted(
    set.intersection(
        *[
            set(series.index)
            for series in urdu_wer.values()
        ]
    )
)

print(
    f"Matched Urdu clips: "
    f"{len(common_audio)}"
)

assert len(common_audio) == 80, (
    f"Expected 80 matched Urdu clips, "
    f"got {len(common_audio)}"
)

urdu_matched = pd.DataFrame({
    name: series.loc[common_audio]
    for name, series in urdu_wer.items()
})


record(
    "Track 4: Urdu Base, automatic minus forced",
    "auto - forced",
    80,
    (
        urdu_matched["base_auto"]
        - urdu_matched["base_forced"]
    ).values,
)


record(
    "Track 4: Urdu Small, automatic minus forced",
    "auto - forced",
    80,
    (
        urdu_matched["small_auto"]
        - urdu_matched["small_forced"]
    ).values,
)


# ============================================================
# Track 4: Korean Small
# ============================================================

korean_auto = pd.read_csv(
    RESULTS / "korean_small_auto.csv"
).set_index("id")

korean_forced = pd.read_csv(
    RESULTS / "korean_small_forced.csv"
).set_index("id")

common_ids = sorted(
    set(korean_auto.index)
    &
    set(korean_forced.index)
)

print(
    f"Matched Korean clips: "
    f"{len(common_ids)}"
)

assert len(common_ids) == 100, (
    f"Expected 100 matched Korean clips, "
    f"got {len(common_ids)}"
)

korean_matched = pd.DataFrame({
    "auto_wer": korean_auto.loc[
        common_ids,
        "wer"
    ],
    "forced_wer": korean_forced.loc[
        common_ids,
        "wer"
    ],
})


record(
    "Track 4: Korean Small, automatic minus forced",
    "auto - forced",
    100,
    (
        korean_matched["auto_wer"]
        - korean_matched["forced_wer"]
    ).values,
)


# ============================================================
# Save results
# ============================================================

output_path = RESULTS / "bootstrap_ci_results.csv"

pd.DataFrame(rows).to_csv(
    output_path,
    index=False
)

print("=" * 70)
print("Bootstrap analysis complete.")
print(f"Saved: {output_path}")
print("=" * 70)
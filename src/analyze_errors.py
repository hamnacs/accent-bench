import pandas as pd
import re

df = pd.read_csv("results/wer_results_full.csv")

def looks_non_english(text):
    if not isinstance(text, str) or text.strip() == "":
        return False
    # Flag predictions containing non-Latin script (Cyrillic, CJK, etc.)
    if re.search(r'[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]', text):
        return True
    # Flag common Malay/Indonesian function words as a heuristic
    malay_markers = [' dalam ', ' yang ', ' untuk ', ' dan ', ' akan ', ' menjadi ', ' beri ', ' auf ']
    padded = f" {text.lower()} "
    return any(marker in padded for marker in malay_markers)

df["language_switch"] = df["prediction"].apply(looks_non_english)

summary = df.groupby("accent").agg(
    n=("wer", "count"),
    mean_wer=("wer", "mean"),
    language_switch_count=("language_switch", "sum"),
)
summary["language_switch_rate"] = (summary["language_switch_count"] / summary["n"] * 100).round(1)

print(summary.sort_values("language_switch_rate", ascending=False))

# Save the flagged examples for your dashboard's "Example Errors" section
flagged = df[df["language_switch"]][["accent", "ground_truth", "prediction", "wer"]]
flagged.to_csv("results/language_switch_examples.csv", index=False)
print(f"\nSaved {len(flagged)} flagged examples to results/language_switch_examples.csv")
import pandas as pd
import re

df = pd.read_csv("results/wer_results_small.csv")

def looks_non_english(text):
    if not isinstance(text, str) or text.strip() == "":
        return False
    if re.search(r'[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]', text):
        return True
    malay_markers = [' dalam ', ' yang ', ' untuk ', ' dan ', ' akan ', ' menjadi ', ' beri ', ' auf ']
    padded = f" {text.lower()} "
    return any(marker in padded for marker in malay_markers)

df["language_switch"] = df["prediction"].apply(looks_non_english)

summary = df.groupby("accent").agg(
    n=("wer", "count"),
    mean_wer=("wer", "mean"),
    switch_count=("language_switch", "sum"),
)
summary["switch_rate_pct"] = (summary["switch_count"] / summary["n"] * 100).round(1)

print(summary.sort_values("switch_rate_pct", ascending=False))
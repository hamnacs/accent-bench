import pandas as pd
from pathlib import Path

INPUT = Path("data/zeroth_korean/data/test-00000-of-00001.parquet")
OUTPUT = Path("results/korean_test_100.csv")

df = pd.read_parquet(INPUT)

print("Total test clips:", len(df))
print("Total speakers:", df["speaker_id"].nunique())

# Take 10 clips from each of the 10 speakers.
# This gives every test speaker equal representation.
selected_parts = []

for speaker_id, group in df.groupby("speaker_id", sort=True):
    n = min(10, len(group))
    selected_parts.append(group.sample(n=n, random_state=42))

selected = pd.concat(selected_parts).sort_index()

out = selected[["id", "speaker_id", "chapter_id", "text"]].copy()
out["audio_index"] = selected.index

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print()
print("Selected:", len(out))
print("Unique speakers:", out["speaker_id"].nunique())
print()
print("Clips per speaker:")
print(out["speaker_id"].value_counts().sort_index())
print()
print("Saved to:", OUTPUT)